"""
CLAD v2 — Consistency-guided Lookahead Adaptive Decoding (v2)

在 v1（llada_clad_decode.py）基础上新增两项优化：

  **O1｜信息密度加权的一致性评分**
    将 v1 的 "consistency_propagation + future_confidence" 二分评分，
    替换为信息论驱动的三分联合评分：

      score = α × weighted_consistency
            + β × entropy_reduction
            + (1-α-β) × future_confidence

    - weighted_consistency：以位置 Shannon 熵为权重的加权一致率。
      高熵（高信息密度）位置预测稳定，说明该 token 真正锚定了上下文。
    - entropy_reduction：填入 token 后，剩余 mask 位置的平均负熵增量，
      归一化到 [-1,1]（除以 log V），量化互信息贡献。
    - future_confidence：继承 v1 的 LoPA-style 项。

    理论依据：*Mask Is What DLLM Needs* 训练侧"高信息密度位置应被重点优化"
    的推理侧镜像——解码时让高信息密度位置先稳定，更能锚定全局上下文。

    默认参数：α=0.5（consistency_weight），β=0.2（entropy_weight）

  **O2｜多 token 自适应接受**
    选出最优分支并写入第 1 个 token 后，利用该分支的 logits（零额外 forward）
    检查剩余 mask 位置中置信度最高的位置：

      if branch_probs[best_remaining_pos] ≥ accept_threshold2:
          额外接受第 2 个 token

    理论收益：Phase-2 每次激活从「接受 1 / k+1 次 forward」
              升级为「最多接受 2 / k+1 次 forward」，
              高置信场景下吞吐理论提升约 2×。

    默认参数：accept_threshold2=0.90；设为 1.0 退化回 v1 行为。

════════════════════════════════════════════════
主入口：generate_with_clad
配置类：CladConfig（含 entropy_weight, accept_threshold2，两个新字段）
统计类：DecodeStats（含 o2_iters，相比 v1 多一项）
════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 统计类
# ─────────────────────────────────────────────────────────────


@dataclass
class DecodeStats:
    """generate_with_clad（v2）调用期间各解码阶段的命中统计。"""

    phase1_iters: int = 0  # 阶段一（一致性快通道）命中迭代次数
    phase1_tokens: int = 0  # 阶段一累计接受 token 数
    phase2_iters: int = 0  # 阶段二触发次数（warmup 后、阶段一未命中时尝试）
    phase2_accepted: int = 0  # 阶段二实际成功接受次数
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


def _append_trace(trace_out: Optional[List[dict]], event: dict):
    """将结构化事件写入 trace_out（若启用）。"""
    if trace_out is not None:
        trace_out.append(event)


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────


@dataclass
class CladConfig:
    """CLAD v2 解码配置参数（O1 信息密度加权评分 + O2 多 token 接受）"""

    # ── 阶段一：一致性快速通道（CCD-inspired）────────────────
    top_v: int = 4

    # ── 阶段二（O1 三分评分）─────────────────────────────────
    num_lookahead: int = 2
    consistency_weight: float = 0.5  # α，加权一致率权重
    entropy_weight: float = 0.2  # β，熵降奖励权重；future_conf 权重 = 1-α-β
    lookahead_warmup: int = 3

    # ── O2：多 token 自适应接受 ───────────────────────────────
    accept_threshold2: float = 0.90  # 第二 token 接受阈值；1.0 = 禁用 O2

    # ── 熵归一化词表规模（与 LLaDA2.1-mini 一致）──────────────
    vocab_size: int = 156896

    # ── 继承 LLaDA2.1 参数 ────────────────────────────────────
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


# ─────────────────────────────────────────────────────────────
# 辅助：输入构造 & 前向
# ─────────────────────────────────────────────────────────────


def _user_prompt_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    """单条 user 消息 → input_ids [1, L]，与 baseline apply_chat_template 一致。"""
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
    """LLaDA2.1-mini 前向计算 logits [1, L, vocab]。"""
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


# ─────────────────────────────────────────────────────────────
# 辅助：token 采样 & 熵计算
# ─────────────────────────────────────────────────────────────


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (token_ids [L], token_probs [L])，temperature=0 为贪婪。"""
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
    """负熵置信度 [L]：越大越确定（用于历史 buffer 的 top-V 选择）。"""
    probs = F.softmax(logits, dim=-1)
    return torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


def _per_position_entropy(logits_2d: torch.Tensor) -> torch.Tensor:
    """全词表 Shannon 熵 H_i [L]，越大越不确定（信息密度越高）。"""
    probs = F.softmax(logits_2d, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


# ─────────────────────────────────────────────────────────────
# 历史 Buffer（与 v1 相同）
# ─────────────────────────────────────────────────────────────


class CladHistoryBuffer:
    """
    维护最近 1 步 top-V 位置的 argmax token，供阶段一一致性快通道使用。
    """

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
        """
        双步一致性检验：当前步 top-V ∩ 上步 top-V，且 argmax token 相同。
        返回 (positions, tokens) 或 (None, None)。
        """
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


# ─────────────────────────────────────────────────────────────
# O1：信息密度加权的分支评分
# ─────────────────────────────────────────────────────────────


def _clad_branch_score_v2(
    branch_x: torch.Tensor,  # [1, window_end]
    branch_logits: torch.Tensor,  # [1, window_end, vocab]
    cur_tokens: torch.Tensor,  # [block_len] 填前 argmax token
    cur_logits_block: torch.Tensor,  # [block_len, vocab] 填前块内 logits
    block_start: int,
    block_end: int,
    config: CladConfig,
) -> float:
    """
    O1 三分联合评分：

      score = α × weighted_consistency
            + β × entropy_reduction
            + (1-α-β) × future_confidence

    weighted_consistency：
      以位置熵 H_i（信息密度）为权重的加权一致率。
      高熵位置若在该分支下预测不变，说明 token 真正锚定了上下文。

    entropy_reduction：
      branch 状态下剩余 mask 位置的 Shannon 熵均值与 cur 的差值，
      归一化到 [-1,1]（除以 log V）。越正说明该 token 为上下文"去掉了歧义"。

    future_confidence：
      剩余 mask 位置的平均最大概率（继承自 v1 / LoPA）。
    """
    branch_block = branch_x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id
    if remaining_mask.sum() == 0:
        return 1.0

    branch_block_logits = branch_logits[0, block_start:block_end]
    probs = F.softmax(branch_block_logits, dim=-1)

    cur_ent = _per_position_entropy(cur_logits_block)  # [block_len]
    br_ent = _per_position_entropy(branch_block_logits)  # [block_len]

    # weighted_consistency
    branch_new_tokens = probs.argmax(dim=-1)
    matches = branch_new_tokens[remaining_mask] == cur_tokens[remaining_mask]
    H = cur_ent[remaining_mask].clamp(min=1e-8)
    h_sum = H.sum().clamp(min=1e-8)
    weighted_cons = (H * matches.float()).sum() / h_sum

    # entropy_reduction（越正越好，归一化）
    log_v = math.log(float(config.vocab_size))
    ent_red = (br_ent[remaining_mask] - cur_ent[remaining_mask]) / log_v
    ent_red = ent_red.mean().clamp(-1.0, 1.0)

    # future_confidence
    max_probs = probs[remaining_mask].max(dim=-1).values
    future_conf = max_probs.mean()

    a, b = config.consistency_weight, config.entropy_weight
    score = a * weighted_cons + b * ent_red + (1.0 - a - b) * future_conf
    return float(score.item())


def _clad_branch_score_v2_details(
    branch_x: torch.Tensor,
    branch_logits: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_logits_block: torch.Tensor,
    block_start: int,
    block_end: int,
    config: CladConfig,
) -> dict:
    """返回 v2 分支评分细项，供 case study 解释分支选择原因。"""
    branch_block = branch_x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id
    if remaining_mask.sum() == 0:
        return {
            "score": 1.0,
            "weighted_consistency": 1.0,
            "entropy_reduction": 0.0,
            "future_confidence": 1.0,
            "remaining_mask_count": 0,
        }

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
    return {
        "score": float(score.item()),
        "weighted_consistency": float(weighted_cons.item()),
        "entropy_reduction": float(ent_red.item()),
        "future_confidence": float(future_conf.item()),
        "remaining_mask_count": int(remaining_mask.sum().item()),
    }


# ─────────────────────────────────────────────────────────────
# O2：利用已有 logits 额外接受第二个 token
# ─────────────────────────────────────────────────────────────


def _apply_o2_second_token(
    x: torch.Tensor,
    branch_logits_1: torch.Tensor,  # [1, window_end, vocab] 已填第 1 个 token 后的 logits
    block_start: int,
    block_end: int,
    config: CladConfig,
) -> dict:
    """
    在已写入 Level-1 一个 token 后的序列 x 上，
    利用 branch_logits_1（填该 token 后的前向结果）
    尝试再接受一个高置信 mask 位置。

    零额外 forward 成本。

    Returns:
        包含 fired / best_pos / best_token / best_prob 等信息的 dict。
    """
    branch_block = x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id
    if remaining_mask.sum() == 0:
        return {
            "fired": False,
            "best_pos": None,
            "best_token": None,
            "best_prob": None,
            "threshold2": float(config.accept_threshold2),
        }

    bl = branch_logits_1[0, block_start:block_end]
    probs = F.softmax(bl, dim=-1)
    max_p, tok = probs.max(dim=-1)
    max_p = torch.where(remaining_mask, max_p, torch.tensor(-1.0, device=max_p.device))
    best_pos = max_p.argmax()
    best_prob = float(max_p[best_pos].item())
    best_token = int(tok[best_pos].item())
    result = {
        "fired": False,
        "best_pos": int(best_pos.item()),
        "best_token": best_token,
        "best_prob": best_prob,
        "threshold2": float(config.accept_threshold2),
    }
    if best_prob >= config.accept_threshold2:
        x[0, block_start + best_pos] = best_token
        print(
            f"    [CLAD v2] O2: extra token pos={best_pos.item()} "
            f"prob={best_prob:.3f}"
        )
        result["fired"] = True
    return result


# ─────────────────────────────────────────────────────────────
# 阶段二：前瞻填充（O1 评分 + O2 二次接受）
# ─────────────────────────────────────────────────────────────


def _clad_lookahead_fill(
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
    block_logits_2d: torch.Tensor,  # [block_len, vocab]：填前块内 logits（供 O1 熵计算）
    config: CladConfig,
    device: torch.device,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
    block_iter: Optional[int] = None,
) -> bool:
    """
    CLAD v2 阶段二：O1 信息密度加权评分前瞻 + O2 二次接受。

    对 top-k 候选位置各做一次串行前向，用 O1 三分联合评分选最优分支，
    写入第 1 个 token 后用 O2 尝试零代价接受第 2 个 token。

    Returns:
        True 表示成功选出分支并填入至少一个 token。
    """
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

    if best_pos is None:
        return False

    x[0, block_start + best_pos] = best_tok
    print(
        f"    [CLAD v2] Phase-2: pos={best_pos} tok={best_tok} score={best_score:.3f}"
    )

    if stats is not None:
        stats.phase2_accepted += 1

    # O2：利用已有 logits 尝试接受第二个 token
    o2_info = _apply_o2_second_token(x, best_logits, block_start, block_end, config)
    if o2_info["fired"] and stats is not None:
        stats.o2_iters += 1
    accepted_positions = [int(best_pos)]
    accepted_tokens = [int(best_tok)]
    if o2_info["fired"] and o2_info["best_pos"] is not None:
        accepted_positions.append(int(o2_info["best_pos"]))
        accepted_tokens.append(int(o2_info["best_token"]))
    _append_trace(
        trace_out,
        {
            "block": int(block_start // config.block_length),
            "iter": int(block_iter) if block_iter is not None else None,
            "phase": "phase2",
            "accepted_positions": accepted_positions,
            "accepted_tokens": accepted_tokens,
            "n_tokens": 1 + (1 if o2_info["fired"] else 0),
            "winner_score": float(best_score),
            "winner_details": winner_details,
            "candidate_scores": candidate_traces,
            "o2": o2_info,
        },
    )

    return True


# ─────────────────────────────────────────────────────────────
# Block 级解码
# ─────────────────────────────────────────────────────────────


def _clad_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CladConfig,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
) -> torch.Tensor:
    """单个 block 上的 CLAD v2 解码（阶段一快通道 → 阶段二 O1+O2 → 阶段三 fallback）。"""
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

        # ── 阶段一：一致性快速通道 ──────────────────────────────────────────
        accepted = False
        if history_buffer.has_history:
            pos, tok = history_buffer.get_consistent_positions(
                neg_ent, tokens, active_mask
            )
            if pos is not None and len(pos) > 0:
                x[0, block_start:block_end][pos] = tok
                print(f"    [CLAD v2] Phase-1 consistency: accepted {len(pos)} tokens")
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

        # ── 阶段二：O1 加权评分前瞻 + O2 二次接受 ─────────────────────────
        if (
            not accepted
            and iter_count >= config.lookahead_warmup
            and config.num_lookahead > 0
        ):
            if stats is not None:
                stats.phase2_iters += 1
            accepted = _clad_lookahead_fill(
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
                trace_out=trace_out,
                block_iter=iter_count,
            )

        # ── 阶段三：退回阈值填充 ────────────────────────────────────────────
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

        # ── 编辑已生成 token ────────────────────────────────────────────────
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

        # ── 更新历史 buffer ─────────────────────────────────────────────────
        updated_active = x[0, block_start:block_end] == config.mask_id
        history_buffer.update(neg_ent, tokens, updated_active)
        iter_count += 1
        if stats is not None:
            stats.total_iters += 1

        if torch.equal(old_block, x[0, block_start:block_end]):
            break

    return x


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────


def generate_with_clad(
    model,
    tokenizer,
    prompt: str,
    config: Optional[CladConfig] = None,
    stats_out: Optional[List] = None,
    trace_out: Optional[List[dict]] = None,
) -> Tuple[str, int]:
    """
    使用 CLAD v2 策略（O1 + O2）对 LLaDA2.1-mini 进行解码。

    stats_out: 若传入非 None 的列表，调用结束后 append 一个 DecodeStats 实例。

    Returns:
        (生成文本, forward_count)
    """
    if config is None:
        config = CladConfig()

    _stats = DecodeStats() if stats_out is not None else None
    forward_counter: List[int] = [0]

    print(
        f"[CLAD v2] top_v={config.top_v} k={config.num_lookahead} "
        f"α={config.consistency_weight} β={config.entropy_weight} "
        f"thr2={config.accept_threshold2}"
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
        print(f"[CLAD v2] block {block_idx} ({block_start}:{block_end})")

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
            trace_out=trace_out,
        )

        if config.eos_early_stop and config.eos_id in x[:, prompt_length:]:
            print("[CLAD v2] EOS, stop")
            break

    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[CLAD v2] done, tokens={generated_tokens.shape[1]}")

    if stats_out is not None and _stats is not None:
        stats_out.append(_stats)
    return result_text.strip(), forward_counter[0]
