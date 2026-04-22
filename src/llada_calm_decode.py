"""
CALM — Consistency-Anchored Local March.

核心思想：
- 在 Phase-1 中先接受跨步一致的 anchor token；
- 然后使用当前步 logits，在 anchor 邻域内按退火阈值额外接受高置信 token；
- 若当前步没有一致性 anchor，则直接进入 fallback，不启用 Phase-2。
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llada_clad_decode import (
    CladHistoryBuffer,
    _append_trace,
    _llada_forward_logits,
    _neg_entropy_confidence,
    _sample_tokens,
    _user_prompt_input_ids,
)


@dataclass
class DecodeStats:
    """CALM 调用期间的阶段命中统计。"""

    phase1_iters: int = 0
    phase1_tokens: int = 0
    calm_neighbor_iters: int = 0
    calm_neighbor_tokens: int = 0
    phase3_iters: int = 0
    total_iters: int = 0

    def hit_rates(self) -> dict:
        n = max(self.total_iters, 1)
        return {
            "phase1_hit_rate": self.phase1_iters / n,
            "calm_neighbor_hit_rate": self.calm_neighbor_iters / n,
            "calm_neighbor_token_rate": self.calm_neighbor_tokens / n,
            "phase3_fallback_rate": self.phase3_iters / n,
        }


@dataclass
class CalmConfig:
    """CALM 解码配置。"""

    top_v: int = 4
    neighbor_radius: int = 1
    max_neighbor_accept_per_anchor: int = 1

    local_threshold_start: float = 0.90
    local_threshold_end: float = 0.72
    local_threshold_gamma: float = 1.0

    # inherited LLaDA2.1 params
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


def _annealed_local_threshold(
    iter_count: int,
    max_iterations: int,
    config: CalmConfig,
) -> float:
    if max_iterations <= 1:
        return float(config.local_threshold_end)
    progress = min(max(iter_count / float(max_iterations - 1), 0.0), 1.0)
    if config.local_threshold_gamma == 1.0:
        coeff = 1.0 - progress
    else:
        coeff = (1.0 - progress) ** float(config.local_threshold_gamma)
    return float(
        config.local_threshold_end
        + (config.local_threshold_start - config.local_threshold_end) * coeff
    )


def _apply_calm_neighbor_accept(
    x: torch.Tensor,
    token_probs: torch.Tensor,
    tokens: torch.Tensor,
    anchor_positions: List[int],
    active_mask: torch.Tensor,
    block_start: int,
    config: CalmConfig,
    local_threshold: float,
) -> dict:
    """围绕一致性 anchor 做局部高置信扩张，不额外 forward。"""
    accepted_positions: List[int] = []
    accepted_tokens: List[int] = []
    accepted_probs: List[float] = []
    neighborhoods: List[dict] = []
    already_taken = set(anchor_positions)
    block_len = active_mask.shape[0]

    for anchor_pos in anchor_positions:
        left = max(0, anchor_pos - config.neighbor_radius)
        right = min(block_len, anchor_pos + config.neighbor_radius + 1)
        candidates = []
        for pos in range(left, right):
            if pos in already_taken:
                continue
            if not active_mask[pos].item():
                continue
            prob = float(token_probs[pos].item())
            if prob < local_threshold:
                continue
            candidates.append((pos, prob, int(tokens[pos].item())))

        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[: config.max_neighbor_accept_per_anchor]

        for pos, prob, tok in selected:
            x[0, block_start + pos] = tok
            accepted_positions.append(int(pos))
            accepted_tokens.append(int(tok))
            accepted_probs.append(float(prob))
            already_taken.add(pos)

        neighborhoods.append(
            {
                "anchor_pos": int(anchor_pos),
                "candidate_positions": [int(pos) for pos, _, _ in candidates],
                "candidate_probs": [float(prob) for _, prob, _ in candidates],
                "accepted_neighbor_positions": [int(pos) for pos, _, _ in selected],
                "accepted_neighbor_tokens": [int(tok) for _, _, tok in selected],
                "accepted_neighbor_probs": [float(prob) for _, prob, _ in selected],
            }
        )

    return {
        "fired": len(accepted_positions) > 0,
        "accepted_positions": accepted_positions,
        "accepted_tokens": accepted_tokens,
        "accepted_probs": accepted_probs,
        "local_threshold": float(local_threshold),
        "anchor_neighborhoods": neighborhoods,
    }


def _calm_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CalmConfig,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
) -> torch.Tensor:
    """在单个 block 上执行 CALM 解码。"""
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
                anchor_positions = [int(p) for p in pos.tolist()]
                x[0, block_start:block_end][pos] = tok
                print(f"    [CALM] Phase-1 consistency: accepted {len(pos)} anchors")
                accepted = True
                if stats is not None:
                    stats.phase1_iters += 1
                    stats.phase1_tokens += len(pos)

                local_threshold = _annealed_local_threshold(
                    iter_count, max_iterations, config
                )
                neighbor_info = _apply_calm_neighbor_accept(
                    x,
                    token_probs,
                    tokens,
                    anchor_positions,
                    active_mask,
                    block_start,
                    config,
                    local_threshold,
                )
                if stats is not None and neighbor_info["fired"]:
                    stats.calm_neighbor_iters += 1
                    stats.calm_neighbor_tokens += len(
                        neighbor_info["accepted_positions"]
                    )

                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase1",
                        "anchor_positions": anchor_positions,
                        "anchor_tokens": [int(t) for t in tok.tolist()],
                        "accepted_positions": anchor_positions
                        + neighbor_info["accepted_positions"],
                        "accepted_tokens": [int(t) for t in tok.tolist()]
                        + neighbor_info["accepted_tokens"],
                        "n_tokens": int(
                            len(pos) + len(neighbor_info["accepted_positions"])
                        ),
                        "local_threshold": float(local_threshold),
                        "anchor_neighborhoods": neighbor_info["anchor_neighborhoods"],
                        "neighbor_accept": neighbor_info,
                    },
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
                        "reason": "no_consistency_anchor",
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
                        "reason": "no_consistency_anchor",
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


def generate_with_calm(
    model,
    tokenizer,
    prompt: str,
    config: Optional[CalmConfig] = None,
    stats_out: Optional[List] = None,
    trace_out: Optional[List[dict]] = None,
) -> Tuple[str, int]:
    """使用 CALM 策略对 LLaDA2.1-mini 进行解码。"""
    if config is None:
        config = CalmConfig()

    _stats = DecodeStats() if stats_out is not None else None
    forward_counter: List[int] = [0]

    print(
        f"[CALM] top_v={config.top_v} radius={config.neighbor_radius} "
        f"max_local={config.max_neighbor_accept_per_anchor} "
        f"tau=({config.local_threshold_start:.2f}->{config.local_threshold_end:.2f}) "
        f"gamma={config.local_threshold_gamma:.2f}"
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
        print(f"[CALM] block {block_idx} ({block_start}:{block_end})")

        if (x[:, block_start:block_end] == config.mask_id).sum() == 0:
            continue

        x = _calm_decode_block(
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
            print("[CALM] EOS, stop")
            break

    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[CALM] done, tokens={generated_tokens.shape[1]}")

    if stats_out is not None and _stats is not None:
        stats_out.append(_stats)
    return result_text.strip(), forward_counter[0]
