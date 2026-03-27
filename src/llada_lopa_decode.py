"""
LoPA (Lookahead Parallel Decoding) - 修复版
符合扩散模型迭代本质的实现
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import copy


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


@dataclass
class LopaConfig:
    """LoPA 解码配置参数"""

    num_lookahead: int = 3
    branch_confidence_mode: str = "avg_prob"
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


class LopaDecodingBranch:
    """代表一个解码分支的状态"""

    def __init__(
        self,
        branch_id: int,
        x_t: torch.Tensor,
        block_states: Dict,
        confidence: float = 1.0,
        prompt_length: int = 0,
        is_anchor: bool = True,
    ):
        self.branch_id = branch_id
        self.x_t = x_t
        self.block_states = block_states
        self.confidence = confidence
        self.step_confidences: List[float] = []
        self.tokens_filled_this_step: int = 0
        self.prompt_length = prompt_length
        self.is_anchor = is_anchor

    def copy(self):
        """创建分支的深拷贝"""
        new_branch = LopaDecodingBranch(
            branch_id=self.branch_id,
            x_t=self.x_t.clone(),
            block_states=copy.deepcopy(self.block_states),
            confidence=self.confidence,
            prompt_length=self.prompt_length,
            is_anchor=self.is_anchor,
        )
        new_branch.step_confidences = self.step_confidences.copy()
        new_branch.tokens_filled_this_step = self.tokens_filled_this_step
        return new_branch


def _llada_forward_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    """LLaDA2.1-mini 前向计算 logits。
    与 LLaDA2.1 原生 forward() 保持一致：lm_head 输出后转 float32。
    forward_counter: 若传入单元素 list，则每次成功前向 +1（用于统计 TPF）。
    """
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    outputs = base.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    if forward_counter is not None:
        forward_counter[0] += 1
    return base.lm_head(outputs.last_hidden_state).float()


def _sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """采样并返回 token 和置信度"""
    if temperature == 0.0:
        probs = F.softmax(logits, dim=-1)
        token = torch.argmax(logits, dim=-1, keepdim=True)
        token_prob = torch.gather(probs, -1, token)
        return token.squeeze(-1), token_prob.squeeze(-1)

    if temperature != 1.0:
        logits = logits / temperature

    # top-k
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # top-p
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
        probs.shape[:-1]
    )
    token_prob = torch.gather(probs, -1, token.unsqueeze(-1)).squeeze(-1)
    return token, token_prob


def generate_with_lopa(
    model,
    tokenizer,
    prompt: str,
    config: Optional[LopaConfig] = None,
) -> Tuple[str, int]:
    """
    使用修复版 LoPA 策略对 LLaDA2.1-mini 进行解码。
    符合扩散模型迭代本质：每个 block 内多次迭代，每次重新计算 logits。

    Returns:
        (生成文本, forward_count)：主干前向次数，用于 TPF = 输出 token 数 / forward_count。
    """
    if config is None:
        config = LopaConfig()

    forward_counter: List[int] = [0]

    print(f"[LoPA] Starting generation with {config.num_lookahead} lookahead branches")

    # 1. 构造输入（与 tokenizer.apply_chat_template 一致）
    input_ids = _user_prompt_input_ids(tokenizer, prompt).to(model.device)

    # 2. 初始化序列模板（与原生 generate 一致）
    prompt_length = input_ids.shape[1]
    num_blocks = (
        prompt_length + config.gen_length + config.block_length - 1
    ) // config.block_length
    total_length = num_blocks * config.block_length

    # 预创建全局 block attention mask（与原生 generate 一致）
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(config.block_length, dim=0)
        .repeat_interleave(config.block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    # 创建初始序列
    x = torch.full(
        (1, total_length), config.mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt_length] = input_ids.clone()

    # 3. 按 block 迭代解码
    prefill_blocks = prompt_length // config.block_length

    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * config.block_length
        block_end = min((block_idx + 1) * config.block_length, total_length)

        print(f"[LoPA] Processing block {block_idx} (pos {block_start}:{block_end})")

        # 检查当前 block 是否还有 mask
        current_block = x[:, block_start:block_end]
        if (current_block == config.mask_id).sum() == 0:
            print(f"    [LoPA] Block already complete, skipping")
            continue

        # 在当前 block 上执行迭代式 LoPA 解码
        x = _lopa_decode_block(
            model,
            x,
            block_start,
            block_end,
            prompt_length,
            block_diffusion_attention_mask,
            position_ids,
            config,
            forward_counter,
        )

        # 检查是否需要提前停止
        if config.eos_early_stop:
            generated_part = x[:, prompt_length:]
            if config.eos_id in generated_part:
                print(f"[LoPA] EOS detected, stopping early")
                break

    # 4. 后处理
    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_positions = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_positions) >= 2 and len(eos_positions[1]) > 0:
        first_eos_pos = eos_positions[1][0].item()
        generated_tokens = generated_part[:, : first_eos_pos + 1]
    else:
        generated_tokens = generated_part

    # 只解码生成部分，不拼 prompt，与 baseline 保持一致
    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print(f"[LoPA] Generation completed, total tokens: {generated_tokens.shape[1]}")
    return result_text.strip(), forward_counter[0]


def _lopa_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    prompt_length: int,
    global_attention_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: LopaConfig,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    在单个 block 上执行迭代式 LoPA 解码。
    符合扩散模型本质：多次迭代，每次重新计算 logits。
    """
    block_length = block_end - block_start
    device = x.device
    post_steps = 0

    # 上限 = block_len（最坏情况每次只填1个）+ max_post_steps（编辑阶段）+ 缓冲
    max_iterations = block_length + config.max_post_steps + 10
    for iteration in range(max_iterations):
        current_window_end = block_end
        cur_x = x[:, :current_window_end].clone()

        # 获取当前 block 状态
        cur_block = cur_x[:, block_start:block_end]
        old_block_tokens = cur_block.clone()
        active_mask = cur_block == config.mask_id

        # 检查是否需要继续
        if not active_mask.any():
            post_steps += 1
            if post_steps > config.max_post_steps:
                break
            # 继续迭代以允许编辑

        # 准备 attention mask 和 position_ids
        cur_attn_mask = global_attention_mask[
            :, :, :current_window_end, :current_window_end
        ]
        cur_position_ids = global_position_ids[:, :current_window_end]

        # 前向计算（每次迭代都重新计算，符合扩散模型本质）
        with torch.no_grad():
            logits = _llada_forward_logits(
                model,
                cur_x,
                cur_attn_mask,
                cur_position_ids,
                forward_counter=forward_counter,
            )

        block_logits = logits[:, block_start:block_end, :]

        # 采样
        x0, x0_p = _sample_with_temperature_topk_topp(
            block_logits, temperature=config.temperature
        )

        # 执行 LoPA：基础填充 + 前瞻分支 + 选择
        updated_x = _execute_lopa_step(
            cur_x,
            x0,
            x0_p,
            active_mask,
            block_start,
            block_end,
            prompt_length,
            config,
            model,
            global_attention_mask,
            global_position_ids,
            forward_counter,
        )

        if updated_x is None:
            # 没有有效更新
            break

        x[:, :current_window_end] = updated_x

        # 检查是否有实际变化
        new_block = x[:, block_start:block_end]
        if torch.equal(old_block_tokens, new_block):
            break

    return x


def _execute_lopa_step(
    cur_x: torch.Tensor,
    x0: torch.Tensor,  # [1, block_length]
    x0_p: torch.Tensor,  # [1, block_length]
    active_mask: torch.Tensor,  # [1, block_length]
    block_start: int,
    block_end: int,
    prompt_length: int,
    config: LopaConfig,
    model,
    global_attention_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    forward_counter: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    """
    执行单次 LoPA 步骤：
    1. 基础填充（B0）
    2. 生成前瞻分支
    3. 批量评估选择最优
    """
    block_length = block_end - block_start
    device = cur_x.device

    # Step 1: 基础填充（B0）
    mask_confidence = torch.where(active_mask, x0_p, -torch.inf)
    high_conf_mask = (mask_confidence[0] > config.threshold) & active_mask[0]

    if not high_conf_mask.any():
        # 没有高置信度位置，尝试填充置信度最高的一个
        if active_mask.sum() == 0:
            return None
        _, idx = torch.topk(mask_confidence[0], k=1)
        high_conf_mask[idx] = True

    # 创建基础分支 B0（修复形状不匹配）
    base_x = cur_x.clone()
    # high_conf_mask 是 [block_length], x0 是 [1, block_length]
    # 需要让 high_conf_mask 与 x0 的维度匹配
    base_x[:, block_start:block_end][0, high_conf_mask] = x0[0, high_conf_mask]

    # 计算已填充数
    num_filled = high_conf_mask.sum().item()

    # Step 2: 检查剩余 mask 并生成前瞻分支
    remaining_mask = base_x[:, block_start:block_end] == config.mask_id
    lookahead_branches = []

    if remaining_mask.sum() > 0 and config.num_lookahead > 0:
        # 计算剩余位置的置信度
        remaining_conf = torch.where(remaining_mask, x0_p, -torch.inf)[0]
        num_candidates = min(config.num_lookahead, remaining_mask.sum().item())

        if num_candidates > 0:
            topk_values, topk_indices = torch.topk(remaining_conf, num_candidates)

            # 为每个候选位置创建前瞻分支
            for i, pos in enumerate(topk_indices):
                lookahead_x = base_x.clone()
                sampled_token = x0[0, pos].item()
                lookahead_x[:, block_start + pos] = sampled_token
                lookahead_branches.append(lookahead_x)

    # 如果没有前瞻分支，直接返回基础分支
    if len(lookahead_branches) == 0:
        return base_x

    # Step 3: 逐个评估所有分支（避免 batch attention mask 形状问题）
    all_branches = [base_x] + lookahead_branches
    current_window_end = block_end

    # 计算分支置信度（逐个前向）
    branch_scores = []
    for branch_x in all_branches:
        branch_input = branch_x[:, :current_window_end]
        cur_attn_mask = global_attention_mask[
            :, :, :current_window_end, :current_window_end
        ]
        cur_position_ids = global_position_ids[:, :current_window_end]

        # 单分支前向
        with torch.no_grad():
            branch_logits = _llada_forward_logits(
                model,
                branch_input,
                cur_attn_mask,
                cur_position_ids,
                forward_counter=forward_counter,
            )

        # Branch Confidence（论文 Eq.2）：对该分支剩余 mask 位置的平均置信度。
        # 衡量"未来还有多少位置可以高置信度被填充"，即 future parallelism potential。
        # 注意：不是对已填充位置计算，而是对剩余 mask 位置计算。
        branch_block = branch_x[:, block_start:block_end]
        remaining_mask_positions = branch_block[0] == config.mask_id

        if remaining_mask_positions.sum() > 0:
            branch_block_logits = branch_logits[:, block_start:block_end, :]
            probs = F.softmax(branch_block_logits[0], dim=-1)
            # 取每个剩余 mask 位置的最大概率作为该位置的置信度
            max_probs_at_mask = probs[remaining_mask_positions].max(dim=-1).values
            score = max_probs_at_mask.mean().item()
        else:
            # 无剩余 mask，该分支已完全填充，给予高分（优先选择已完成的分支）
            score = 1.0

        branch_scores.append(score)

    # 选择最优分支
    best_idx = max(range(len(branch_scores)), key=lambda i: branch_scores[i])
    best_x = all_branches[best_idx]

    # 日志
    branch_labels = ["Base"] + [f"L{i+1}" for i in range(len(lookahead_branches))]
    branch_info = " | ".join(
        [
            f"{'★' if i == best_idx else ' '}{label}: {score:.3f}"
            for i, (label, score) in enumerate(zip(branch_labels, branch_scores))
        ]
    )
    print(f"    [LoPA] Step: filled {num_filled}, branches: {branch_info}")

    return best_x
