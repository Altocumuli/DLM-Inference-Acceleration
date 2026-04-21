"""
CANDLE — Consistency-Anchored Determinism LEap.

设计动机：
- 复用 CLAD v2 的 O1（信息密度加权分支评分）先选出 winner anchor；
- 在 winner branch 上，不再仅做全局单 token O2，而是围绕 anchor 做
  LocalLeap 风格的局部确定性传播；
- 若局部传播未触发，则退回原始 O2 作为 fallback。

与 CLAD v2 的关键区别：
1. anchor 来源不是 raw confidence，而是 O1 winner branch；
2. 传播只发生在 winner anchor 的局部邻域，而不是整个 block；
3. 局部传播带 gate，避免把不可靠 anchor 的错误扩散出去。
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llada_clad_v2_decode import (
    CladConfig as BaseCladConfig,
    CladHistoryBuffer,
    _append_trace,
    _apply_o2_second_token,
    _clad_branch_score_v2_details,
    _llada_forward_logits,
    _neg_entropy_confidence,
    _sample_tokens,
    _user_prompt_input_ids,
)


@dataclass
class DecodeStats:
    """CANDLE 调用期间的阶段命中统计。"""

    phase1_iters: int = 0
    phase1_tokens: int = 0
    phase2_iters: int = 0
    phase2_accepted: int = 0
    local_leap_iters: int = 0
    local_leap_tokens: int = 0  # 仅统计 anchor 之外的额外 token 数
    o2_iters: int = 0  # 仅统计 local leap 未触发时的全局 O2 fallback
    phase3_iters: int = 0
    total_iters: int = 0

    def hit_rates(self) -> dict:
        n = max(self.total_iters, 1)
        return {
            "phase1_hit_rate": self.phase1_iters / n,
            "phase2_trigger_rate": self.phase2_iters / n,
            "phase2_accepted_rate": self.phase2_accepted / n,
            "local_leap_hit_rate": self.local_leap_iters / n,
            "local_leap_token_rate": self.local_leap_tokens / n,
            "o2_hit_rate": self.o2_iters / n,
            "phase3_fallback_rate": self.phase3_iters / n,
        }


@dataclass
class CandleConfig(BaseCladConfig):
    """CANDLE 解码配置：CLAD v2 + anchor-conditioned local leap。"""

    use_local_leap: bool = True
    local_radius: int = 2
    local_relaxed_threshold: float = 0.78
    max_local_accept: int = 2

    # winner branch 至少满足以下 gate 才允许 local leap
    anchor_score_threshold: float = 0.0
    local_consistency_floor: float = 0.55
    local_entropy_floor: float = -0.02


def _local_gate_status(winner_details: dict, config: CandleConfig) -> Tuple[bool, str]:
    if not config.use_local_leap:
        return False, "disabled"
    if winner_details is None:
        return False, "missing_winner_details"
    if winner_details["score"] < config.anchor_score_threshold:
        return False, "score_below_threshold"
    if (
        winner_details["weighted_consistency"] < config.local_consistency_floor
        and winner_details["entropy_reduction"] < config.local_entropy_floor
    ):
        return False, "consistency_and_entropy_below_floor"
    return True, "passed"


def _apply_local_leap_tokens(
    x: torch.Tensor,
    branch_logits_1: torch.Tensor,
    anchor_pos: int,
    block_start: int,
    block_end: int,
    config: CandleConfig,
) -> dict:
    """
    在 winner branch 上围绕 anchor 做局部宽松接受。

    Returns:
        {
            "fired": bool,
            "candidate_positions": [...],
            "candidate_probs": [...],
            "accepted_positions": [...],
            "accepted_tokens": [...],
            "accepted_probs": [...],
            "radius": int,
            "relaxed_threshold": float,
        }
    """
    result = {
        "fired": False,
        "candidate_positions": [],
        "candidate_probs": [],
        "accepted_positions": [],
        "accepted_tokens": [],
        "accepted_probs": [],
        "radius": int(config.local_radius),
        "relaxed_threshold": float(config.local_relaxed_threshold),
    }

    if config.max_local_accept <= 0 or config.local_radius <= 0:
        return result

    branch_block = x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id
    if remaining_mask.sum() == 0:
        return result

    left = max(0, anchor_pos - config.local_radius)
    right = min(block_end - block_start, anchor_pos + config.local_radius + 1)
    local_positions = [pos for pos in range(left, right) if remaining_mask[pos].item()]
    if not local_positions:
        return result

    bl = branch_logits_1[0, block_start:block_end]
    probs = torch.softmax(bl, dim=-1)
    max_p, tok = probs.max(dim=-1)

    ranked = sorted(
        (
            (pos, float(max_p[pos].item()), int(tok[pos].item()))
            for pos in local_positions
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    result["candidate_positions"] = [int(pos) for pos, _, _ in ranked]
    result["candidate_probs"] = [float(prob) for _, prob, _ in ranked]

    accepted = 0
    for pos, prob, token in ranked:
        if accepted >= config.max_local_accept:
            break
        if prob < config.local_relaxed_threshold:
            continue
        x[0, block_start + pos] = token
        result["accepted_positions"].append(int(pos))
        result["accepted_tokens"].append(int(token))
        result["accepted_probs"].append(float(prob))
        accepted += 1

    result["fired"] = accepted > 0
    return result


def _candle_lookahead_fill(
    model,
    x: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_probs: torch.Tensor,
    active_mask: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    block_logits_2d: torch.Tensor,
    config: CandleConfig,
    device: torch.device,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
    block_iter: Optional[int] = None,
) -> bool:
    """CANDLE 的 Phase-2：O1 选 anchor + local leap + O2 fallback。"""
    conf_at_mask = torch.where(
        active_mask, cur_probs, torch.tensor(float("-inf"), device=device)
    )
    num_candidates = min(config.num_lookahead, int(active_mask.sum().item()))
    if num_candidates == 0:
        return False

    _, candidate_indices = torch.topk(conf_at_mask, num_candidates)
    cur_attn_mask = global_attn_mask[:, :, :current_window_end, :current_window_end]
    cur_pos_ids = global_position_ids[:, :current_window_end]

    best_score = -1.0
    best_pos = None
    best_tok = None
    best_logits = None
    winner_details = None
    candidate_traces = []

    for cand_pos in candidate_indices.tolist():
        branch_x = x[:, :current_window_end].clone()
        branch_x[0, block_start + cand_pos] = cur_tokens[cand_pos].item()

        with torch.no_grad():
            branch_logits = _llada_forward_logits(
                model, branch_x, cur_attn_mask, cur_pos_ids, forward_counter
            )

        details = _clad_branch_score_v2_details(
            branch_x,
            branch_logits,
            cur_tokens,
            block_logits_2d,
            block_start,
            block_end,
            config,
        )
        score = details["score"]
        candidate_traces.append(
            {
                "pos": int(cand_pos),
                "token": int(cur_tokens[cand_pos].item()),
                **details,
            }
        )

        if score > best_score:
            best_score = score
            best_pos = cand_pos
            best_tok = cur_tokens[cand_pos].item()
            best_logits = branch_logits
            winner_details = details

    if best_pos is None or best_logits is None or winner_details is None:
        return False

    x[0, block_start + best_pos] = best_tok
    local_gate_passed, local_gate_reason = _local_gate_status(winner_details, config)
    local_info = {
        "fired": False,
        "candidate_positions": [],
        "candidate_probs": [],
        "accepted_positions": [],
        "accepted_tokens": [],
        "accepted_probs": [],
        "radius": int(config.local_radius),
        "relaxed_threshold": float(config.local_relaxed_threshold),
    }

    if local_gate_passed:
        local_info = _apply_local_leap_tokens(
            x,
            best_logits,
            best_pos,
            block_start,
            block_end,
            config,
        )

    o2_info = {
        "fired": False,
        "best_pos": None,
        "best_token": None,
        "best_prob": None,
        "threshold2": float(config.accept_threshold2),
    }
    if not local_info["fired"]:
        o2_info = _apply_o2_second_token(x, best_logits, block_start, block_end, config)

    print(
        f"    [CANDLE] Phase-2: anchor_pos={best_pos} tok={best_tok} "
        f"score={best_score:.3f} local={len(local_info['accepted_positions'])} "
        f"o2={int(o2_info['fired'])}"
    )

    if stats is not None:
        stats.phase2_accepted += 1
        if local_info["fired"]:
            stats.local_leap_iters += 1
            stats.local_leap_tokens += len(local_info["accepted_positions"])
        elif o2_info["fired"]:
            stats.o2_iters += 1

    accepted_positions = [int(best_pos)]
    accepted_tokens = [int(best_tok)]
    accepted_positions.extend(local_info["accepted_positions"])
    accepted_tokens.extend(local_info["accepted_tokens"])
    if o2_info["fired"] and o2_info["best_pos"] is not None:
        accepted_positions.append(int(o2_info["best_pos"]))
        accepted_tokens.append(int(o2_info["best_token"]))

    _append_trace(
        trace_out,
        {
            "block": int(block_start // config.block_length),
            "iter": int(block_iter) if block_iter is not None else None,
            "phase": "phase2",
            "anchor_pos": int(best_pos),
            "anchor_token": int(best_tok),
            "accepted_positions": accepted_positions,
            "accepted_tokens": accepted_tokens,
            "n_tokens": len(accepted_positions),
            "winner_score": float(best_score),
            "winner_details": winner_details,
            "candidate_scores": candidate_traces,
            "local_gate_passed": bool(local_gate_passed),
            "local_gate_reason": local_gate_reason,
            "local_leap": local_info,
            "o2": o2_info,
        },
    )
    return True


def _candle_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CandleConfig,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
) -> torch.Tensor:
    """单个 block 上执行 CANDLE 解码。"""
    device = x.device
    block_len = block_end - block_start
    post_steps = 0
    iter_count = 0

    history_buffer = CladHistoryBuffer(top_v=config.top_v, device=device)
    max_iterations = block_len + config.max_post_steps + 10

    for _ in range(max_iterations):
        cur_x = x[:, :current_window_end].clone()
        active_mask = cur_x[0, block_start:block_end] == config.mask_id
        if not active_mask.any():
            post_steps += 1
            if post_steps > config.max_post_steps:
                break

        old_block = x[0, block_start:block_end].clone()
        cur_attn_mask = global_attn_mask[:, :, :current_window_end, :current_window_end]
        cur_pos_ids = global_position_ids[:, :current_window_end]

        with torch.no_grad():
            logits = _llada_forward_logits(
                model, cur_x, cur_attn_mask, cur_pos_ids, forward_counter
            )

        block_logits_2d = logits[0, block_start:block_end]
        neg_ent = _neg_entropy_confidence(block_logits_2d)
        tokens, token_probs = _sample_tokens(block_logits_2d, config.temperature)

        accepted = False
        if history_buffer.has_history:
            pos, tok = history_buffer.get_consistent_positions(
                neg_ent, tokens, active_mask
            )
            if pos is not None and len(pos) > 0:
                x[0, block_start:block_end][pos] = tok
                print(f"    [CANDLE] Phase-1 consistency: accepted {len(pos)} tokens")
                accepted = True
                if stats is not None:
                    stats.phase1_iters += 1
                    stats.phase1_tokens += len(pos)
                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase1",
                        "accepted_positions": [int(p) for p in pos.tolist()],
                        "accepted_tokens": [int(t) for t in tok.tolist()],
                        "n_tokens": int(len(pos)),
                    },
                )

        if (
            not accepted
            and iter_count >= config.lookahead_warmup
            and config.num_lookahead > 0
        ):
            if stats is not None:
                stats.phase2_iters += 1
            accepted = _candle_lookahead_fill(
                model,
                x,
                tokens,
                token_probs,
                active_mask,
                block_start,
                block_end,
                current_window_end,
                global_attn_mask,
                global_position_ids,
                block_logits_2d,
                config,
                device,
                forward_counter,
                stats=stats,
                trace_out=trace_out,
                block_iter=iter_count,
            )

        if not accepted:
            if stats is not None:
                stats.phase3_iters += 1
            conf_at_mask = torch.where(
                active_mask, token_probs, torch.tensor(float("-inf"), device=device)
            )
            high_conf = (conf_at_mask > config.threshold) & active_mask
            if high_conf.any():
                x[0, block_start:block_end][high_conf] = tokens[high_conf]
                accepted_positions = torch.nonzero(high_conf, as_tuple=False).view(-1)
                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase3",
                        "fallback_mode": "threshold_multi",
                        "accepted_positions": [
                            int(p) for p in accepted_positions.tolist()
                        ],
                        "accepted_tokens": [
                            int(tokens[p].item()) for p in accepted_positions
                        ],
                        "accepted_probs": [
                            float(token_probs[p].item()) for p in accepted_positions
                        ],
                        "n_tokens": int(accepted_positions.numel()),
                    },
                )
            else:
                best_pos = conf_at_mask.argmax()
                x[0, block_start:block_end][best_pos] = tokens[best_pos]
                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase3",
                        "fallback_mode": "argmax_single",
                        "accepted_positions": [int(best_pos.item())],
                        "accepted_tokens": [int(tokens[best_pos].item())],
                        "accepted_probs": [float(token_probs[best_pos].item())],
                        "n_tokens": 1,
                    },
                )

        non_mask_non_prompt = (x[0, block_start:block_end] != config.mask_id) & (
            torch.arange(block_len, device=device)
            >= (prompt_length - block_start if block_start < prompt_length else 0)
        )
        if non_mask_non_prompt.any():
            edit_tokens, edit_probs = _sample_tokens(
                block_logits_2d, config.temperature
            )
            edit_mask = (
                non_mask_non_prompt
                & (edit_probs > config.editing_threshold)
                & (edit_tokens != x[0, block_start:block_end])
            )
            if edit_mask.any():
                x[0, block_start:block_end][edit_mask] = edit_tokens[edit_mask]

        updated_active = x[0, block_start:block_end] == config.mask_id
        history_buffer.update(neg_ent, tokens, updated_active)
        iter_count += 1
        if stats is not None:
            stats.total_iters += 1

        if torch.equal(old_block, x[0, block_start:block_end]):
            break

    return x


def generate_with_candle(
    model,
    tokenizer,
    prompt: str,
    config: Optional[CandleConfig] = None,
    stats_out: Optional[List] = None,
    trace_out: Optional[List[dict]] = None,
) -> Tuple[str, int]:
    """使用 CANDLE 策略对 LLaDA2.1-mini 进行解码。"""
    if config is None:
        config = CandleConfig()

    _stats = DecodeStats() if stats_out is not None else None
    forward_counter: List[int] = [0]

    print(
        f"[CANDLE] top_v={config.top_v} k={config.num_lookahead} "
        f"α={config.consistency_weight} β={config.entropy_weight} "
        f"thr2={config.accept_threshold2} r={config.local_radius} "
        f"thr_local={config.local_relaxed_threshold}"
    )

    try:
        _dev = next(model.parameters()).device
    except StopIteration:
        _dev = torch.device("cpu")

    input_ids = _user_prompt_input_ids(tokenizer, prompt).to(_dev)
    prompt_length = input_ids.shape[1]
    num_blocks = (
        prompt_length + config.gen_length + config.block_length - 1
    ) // config.block_length
    total_length = num_blocks * config.block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=_dev))
    global_attn_mask = (
        block_mask.repeat_interleave(config.block_length, dim=0)
        .repeat_interleave(config.block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)
    global_position_ids = torch.arange(total_length, device=_dev).unsqueeze(0)

    x = torch.full((1, total_length), config.mask_id, dtype=torch.long, device=_dev)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // config.block_length
    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * config.block_length
        block_end = min((block_idx + 1) * config.block_length, total_length)
        print(f"[CANDLE] block {block_idx} ({block_start}:{block_end})")

        if (x[:, block_start:block_end] == config.mask_id).sum() == 0:
            continue

        x = _candle_decode_block(
            model,
            x,
            block_start,
            block_end,
            block_end,
            prompt_length,
            global_attn_mask,
            global_position_ids,
            config,
            forward_counter,
            stats=_stats,
            trace_out=trace_out,
        )

        if config.eos_early_stop and config.eos_id in x[:, prompt_length:]:
            print("[CANDLE] EOS, stop")
            break

    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[CANDLE] done, tokens={generated_tokens.shape[1]}")

    if stats_out is not None and _stats is not None:
        stats_out.append(_stats)
    return result_text.strip(), forward_counter[0]
