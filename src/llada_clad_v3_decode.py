"""
CLAD v3 — Consistency-guided Lookahead Adaptive Decoding (v3)

在 v2（O1 信息密度加权评分 + O2 多 token 自适应接受）基础上增加：

  **O3｜分支批量化 forward**：将 Phase-2 的多个候选分支序列拼成 batch [B, seq_len]，
  一次 `model.model` 前向得到所有分支 logits，逻辑与串行等价，GPU 利用率更高。

  **O4｜级联草稿前瞻（Spiffy-inspired）**：
  - Level-1：在当前 mask 上取置信度最高的 2 个位置 p1、p2，构造 2 条分支，
    与 O3 结合为 **1 次 batched forward**；
  - 用 O1 综合评分选出最优 Level-1 分支；
  - Level-2：在该分支对应的 logits 上，对剩余 mask 取最高置信位置 q，
    若 ≥ accept_threshold2，则 **额外接受** 第二个 token（与 O2 一致，无额外前向）。

若 `use_cascaded_draft=False`，则退化为「对 top-k 各一条分支全部 batch 一次」的纯 O3 行为。
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────


@dataclass
class CladV3Config:
    """CLAD v3：O1/O2 + O3 batched forward + O4 cascaded L1 draft"""

    # 阶段一
    top_v: int = 4

    # 阶段二（非级联模式下最多几条 Level-1 候选分支）
    num_lookahead: int = 2
    consistency_weight: float = 0.5  # α：加权一致性
    entropy_weight: float = 0.2  # β：熵降奖励；future_conf = 1 - α - β
    lookahead_warmup: int = 3

    # O2 / Level-2 二次接受
    accept_threshold2: float = 0.90

    # O3 / O4
    use_batched_phase2: bool = True  # True：Phase-2 多分支一次 forward
    use_cascaded_draft: bool = True  # True：O4，L1 仅用 top-2 位置做 2 路 + L2 二次接受

    # 熵归一化（与 LLaDA2.1-mini 词表规模一致 order）
    vocab_size: int = 156896

    # LLaDA2.1 块解码
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


@dataclass
class DecodeStats:
    """generate_with_clad_v3 调用期间各解码阶段的命中统计。"""

    phase1_iters: int = 0  # 阶段一（一致性快通道）命中迭代次数
    phase1_tokens: int = 0  # 阶段一累计接受 token 数
    phase2_iters: int = 0  # 阶段二触发次数（warmup 后、阶段一未命中时尝试）
    phase2_accepted: int = 0  # 阶段二成功接受次数（返回 True）
    o2_iters: int = 0  # O2 额外接受第二 token 的次数
    phase3_iters: int = 0  # 阶段三 fallback 次数
    total_iters: int = 0  # 总迭代次数

    def hit_rates(self) -> dict:
        n = max(self.total_iters, 1)
        return {
            "phase1_hit_rate": self.phase1_iters / n,
            "phase2_trigger_rate": self.phase2_iters / n,
            "phase2_accepted_rate": self.phase2_accepted / n,
            "o2_hit_rate": self.o2_iters / n,
            "phase3_fallback_rate": self.phase3_iters / n,
        }


def _user_prompt_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    chat_inp = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(chat_inp, torch.Tensor):
        input_ids = chat_inp
    else:
        input_ids = chat_inp["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.long()


def _llada_forward_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    out = base.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    if forward_counter is not None:
        forward_counter[0] += 1
    return base.lm_head(out.last_hidden_state).float()


def _llada_forward_logits_batched(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    批量前向：input_ids [B, L]，attention_mask [B, 1, L, L]（或模型接受的形状），
    position_ids [B, L]。一次计数为 1 次主干前向。
    """
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    out = base.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    if forward_counter is not None:
        forward_counter[0] += 1
    return base.lm_head(out.last_hidden_state).float()


def _sample_tokens(
    logits: torch.Tensor, temperature: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        scaled = F.softmax(logits / temperature, dim=-1)
        tokens = torch.multinomial(
            scaled.view(-1, scaled.size(-1)), num_samples=1
        ).view(scaled.shape[:-1])
        token_probs = torch.gather(scaled, -1, tokens.unsqueeze(-1)).squeeze(-1)
    else:
        token_probs, tokens = probs.max(dim=-1)
    return tokens, token_probs


def _neg_entropy_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


def _per_position_entropy(logits_2d: torch.Tensor) -> torch.Tensor:
    """全词表 Shannon 熵，[L]。"""
    probs = F.softmax(logits_2d, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


class CladHistoryBuffer:
    def __init__(self, top_v: int, device: torch.device):
        self.top_v = top_v
        self.device = device
        self._history: deque = deque(maxlen=2)

    def reset(self):
        self._history.clear()

    @property
    def has_history(self) -> bool:
        return len(self._history) > 0

    def update(
        self, neg_ent: torch.Tensor, tokens: torch.Tensor, active_mask: torch.Tensor
    ):
        conf_at_mask = torch.where(
            active_mask,
            neg_ent,
            torch.tensor(float("-inf"), device=self.device),
        )
        k = min(self.top_v, int(active_mask.sum().item()))
        if k == 0:
            return
        _, top_positions = torch.topk(conf_at_mask, k)
        self._history.append((top_positions, tokens[top_positions]))

    def get_consistent_positions(
        self,
        cur_neg_ent: torch.Tensor,
        cur_tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.has_history:
            return None, None
        conf_at_mask = torch.where(
            active_mask,
            cur_neg_ent,
            torch.tensor(float("-inf"), device=self.device),
        )
        k = min(self.top_v, int(active_mask.sum().item()))
        if k == 0:
            return None, None
        _, cur_top = torch.topk(conf_at_mask, k)
        cur_top_set = set(cur_top.tolist())
        prev_pos, prev_tok = self._history[-1]
        consistent_pos, consistent_tok = [], []
        for pos, ptok in zip(prev_pos.tolist(), prev_tok.tolist()):
            if pos in cur_top_set and active_mask[pos]:
                if cur_tokens[pos].item() == ptok:
                    consistent_pos.append(pos)
                    consistent_tok.append(ptok)
        if not consistent_pos:
            return None, None
        return (
            torch.tensor(consistent_pos, dtype=torch.long, device=self.device),
            torch.tensor(consistent_tok, dtype=torch.long, device=self.device),
        )


def _clad_branch_score_v2(
    branch_x: torch.Tensor,
    branch_logits: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_logits_block: torch.Tensor,
    block_start: int,
    block_end: int,
    config: CladV3Config,
) -> float:
    """
    O1 综合分：α * WeightedConsistency + β * EntropyReduction + (1-α-β) * FutureConf。
    branch_logits: [1, window_end, vocab]
    cur_logits_block: [block_len, vocab] 填分支前的块内 logits。
    """
    block_len = block_end - block_start
    branch_block = branch_x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id
    if remaining_mask.sum() == 0:
        return 1.0

    branch_block_logits = branch_logits[0, block_start:block_end]
    probs = F.softmax(branch_block_logits, dim=-1)
    cur_ent = _per_position_entropy(cur_logits_block)
    br_ent = _per_position_entropy(branch_block_logits)

    branch_new_tokens = probs.argmax(dim=-1)
    matches = branch_new_tokens[remaining_mask] == cur_tokens[remaining_mask]
    H = cur_ent[remaining_mask].clamp(min=1e-8)
    h_sum = H.sum().clamp(min=1e-8)
    weighted_cons = (H * matches.float()).sum() / h_sum
    log_v = math.log(float(config.vocab_size))
    ent_red = (br_ent[remaining_mask] - cur_ent[remaining_mask]) / log_v
    ent_red = ent_red.mean().clamp(-1.0, 1.0)

    max_probs = probs[remaining_mask].max(dim=-1).values
    future_conf = max_probs.mean()

    a, b = config.consistency_weight, config.entropy_weight
    score = a * weighted_cons + b * ent_red + (1.0 - a - b) * future_conf
    return float(score.item())


def _apply_o2_second_token(
    x: torch.Tensor,
    branch_logits_1: torch.Tensor,
    block_start: int,
    block_end: int,
    config: CladV3Config,
) -> bool:
    """
    在已写入 Level-1 一个 token 后的序列 x 上，利用 branch_logits_1（对应该单填状态）
    尝试再接受一个高置信 mask 位置。返回是否写入了第二个 token。
    """
    branch_block = x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id
    if remaining_mask.sum() == 0:
        return False
    bl = branch_logits_1[0, block_start:block_end]
    probs = F.softmax(bl, dim=-1)
    max_p, tok = probs.max(dim=-1)
    max_p = torch.where(remaining_mask, max_p, torch.tensor(-1.0, device=max_p.device))
    q = int(max_p.argmax().item())
    if max_p[q].item() >= config.accept_threshold2:
        x[0, block_start + q] = tok[q].item()
        print(f"    [CLAD v3] O2 second token: pos={q} " f"conf={max_p[q].item():.3f}")
        return True
    return False


def _clad_v3_lookahead_fill(
    model,
    x: torch.Tensor,
    base_logits: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_probs: torch.Tensor,
    active_mask: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    block_logits_2d: torch.Tensor,
    config: CladV3Config,
    device: torch.device,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
) -> bool:
    """
    Phase-2：O3 批量 forward + O4 级联（可选）+ O2 二次接受。
    """
    conf_at_mask = torch.where(
        active_mask, cur_probs, torch.tensor(float("-inf"), device=device)
    )
    n_active = int(active_mask.sum().item())
    if n_active == 0:
        return False

    k = min(config.num_lookahead, n_active)
    _, candidate_indices = torch.topk(conf_at_mask, k)
    candidates = candidate_indices.tolist()

    cur_attn_1 = global_attn_mask[:, :, :current_window_end, :current_window_end]
    cur_pos_1 = global_position_ids[:, :current_window_end]

    def run_sequential_branches() -> bool:
        """回退：逐分支前向（与旧版等价）。"""
        best_score = -1.0
        best_pos = None
        best_tok = None
        best_logits = None
        for cand_pos in candidates:
            branch_x = x[:, :current_window_end].clone()
            branch_x[0, block_start + cand_pos] = cur_tokens[cand_pos].item()
            with torch.no_grad():
                branch_logits = _llada_forward_logits(
                    model, branch_x, cur_attn_1, cur_pos_1, forward_counter
                )
            sc = _clad_branch_score_v2(
                branch_x,
                branch_logits,
                cur_tokens,
                block_logits_2d,
                block_start,
                block_end,
                config,
            )
            if sc > best_score:
                best_score = sc
                best_pos = cand_pos
                best_tok = cur_tokens[cand_pos].item()
                best_logits = branch_logits
        if best_pos is None:
            return False
        x[0, block_start + best_pos] = best_tok
        print(
            f"    [CLAD v3] Phase-2 (serial): pos={best_pos} tok={best_tok} score={best_score:.3f}"
        )
        o2_fired = _apply_o2_second_token(
            x, best_logits, block_start, block_end, config
        )
        if stats is not None:
            stats.phase2_accepted += 1
            if o2_fired:
                stats.o2_iters += 1
        return True

    if not config.use_batched_phase2:
        return run_sequential_branches()

    # —— O4：Level-1 仅取 top-2 做 2 路 batch；否则取 top-k 做 k 路 batch ——
    if config.use_cascaded_draft and len(candidates) >= 2:
        level1_positions = candidates[:2]
    else:
        level1_positions = candidates

    rows: List[torch.Tensor] = []
    for pos in level1_positions:
        bx = x[:, :current_window_end].clone()
        bx[0, block_start + pos] = cur_tokens[pos].item()
        rows.append(bx)
    batch_x = torch.cat(rows, dim=0)
    bsz = batch_x.shape[0]
    attn_b = cur_attn_1.expand(bsz, -1, -1, -1)
    pos_b = cur_pos_1.expand(bsz, -1)

    with torch.no_grad():
        logits_b = _llada_forward_logits_batched(
            model, batch_x, attn_b, pos_b, forward_counter
        )

    best_j = -1
    best_score = -1.0
    for j in range(bsz):
        sc = _clad_branch_score_v2(
            batch_x[j : j + 1],
            logits_b[j : j + 1],
            cur_tokens,
            block_logits_2d,
            block_start,
            block_end,
            config,
        )
        if sc > best_score:
            best_score = sc
            best_j = j

    if best_j < 0:
        return False

    best_pos = level1_positions[best_j]
    best_tok = cur_tokens[best_pos].item()
    x[0, block_start + best_pos] = best_tok
    print(
        f"    [CLAD v3] Phase-2 (batched bsz={bsz}): best_pos={best_pos} "
        f"tok={best_tok} score={best_score:.3f}"
    )

    winner_logits = logits_b[best_j : best_j + 1]
    o2_fired = _apply_o2_second_token(x, winner_logits, block_start, block_end, config)
    if stats is not None:
        stats.phase2_accepted += 1
        if o2_fired:
            stats.o2_iters += 1
    return True


def _clad_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CladV3Config,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
) -> torch.Tensor:
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
                print(f"    [CLAD v3] Phase-1 consistency: accepted {len(pos)} tokens")
                accepted = True
                if stats is not None:
                    stats.phase1_iters += 1
                    stats.phase1_tokens += len(pos)

        if (
            not accepted
            and iter_count >= config.lookahead_warmup
            and config.num_lookahead > 0
        ):
            if stats is not None:
                stats.phase2_iters += 1
            accepted = _clad_v3_lookahead_fill(
                model,
                x,
                logits,
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
            else:
                best_pos = conf_at_mask.argmax()
                x[0, block_start:block_end][best_pos] = tokens[best_pos]

        if stats is not None:
            stats.total_iters += 1

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

        if torch.equal(old_block, x[0, block_start:block_end]):
            break

    return x


def generate_with_clad_v3(
    model,
    tokenizer,
    prompt: str,
    config: Optional[CladV3Config] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """
    stats_out: 若传入非 None 的列表，本次调用结束后会 append 一个 DecodeStats 实例。
    """
    if config is None:
        config = CladV3Config()

    _stats = DecodeStats() if stats_out is not None else None
    forward_counter: List[int] = [0]
    print(
        f"[CLAD v3] top_v={config.top_v} num_lookahead={config.num_lookahead} "
        f"α={config.consistency_weight} β={config.entropy_weight} "
        f"batched={config.use_batched_phase2} cascaded={config.use_cascaded_draft}"
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
        print(f"[CLAD v3] block {block_idx} ({block_start}:{block_end})")
        if (x[:, block_start:block_end] == config.mask_id).sum() == 0:
            continue
        x = _clad_decode_block(
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
        )
        if config.eos_early_stop and config.eos_id in x[:, prompt_length:]:
            print("[CLAD v3] EOS, stop")
            break

    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[CLAD v3] done, tokens={generated_tokens.shape[1]}")
    if stats_out is not None and _stats is not None:
        stats_out.append(_stats)
    return result_text.strip(), forward_counter[0]


# 兼容 run_benchmark 旧式 import 名称
generate_with_clad = generate_with_clad_v3
