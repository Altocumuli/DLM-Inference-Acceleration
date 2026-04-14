# 扩散语言模型毕设：课题理解与设计笔记

## 1. 现有 `generate` 策略解读（LLaDA2.1-mini / modeling_llada2_moe.py）

以下根据 `inclusionAI/LLaDA2.1-mini` 中 `modeling_llada2_moe.py` 的 `generate()` 实现整理，便于后续做「解掩码策略」对比实验时有一份统一的参考。

### 1.1 整体思路（与自回归的区别）

- **自回归**：从左到右一个一个 token 生成，每一步只能看到已生成部分。
- **本实现**：先按「块」铺满整段长度，全部用 `mask_id` 占位；再**按块、迭代地把 mask 解成真实 token**，块内可并行看整块，块与块之间用**块对角因果 mask** 保证只看过去块、不看未来块。

因此：**解掩码策略** = 在「每块、每轮」里：**解多少 mask、按什么规则接受/拒绝 token、是否对已生成 token 做编辑、何时停**。

---

### 1.2 主要参数与含义（与 baseline 脚本对应）

| 参数 | 默认值（代码） | 当前 baseline | 含义 |
|------|----------------|---------------|------|
| `block_length` | 32 | 32 | 每块长度；块内并行解码，块间因果 |
| `gen_length` | 2048 | 512 | 最多生成 token 数（不含 prompt） |
| `eos_early_stop` | False | **True** | 生成部分一旦没有 mask 且出现 EOS 就提前结束 |
| `threshold` | 0.95 | **0.7** | **解 mask**：只有模型对该位置概率 > threshold 才接受为该步「确定」token |
| `editing_threshold` | 0.9 | **0.5** | **编辑**：已非 mask 的位置，若新采样 token 概率 > 该阈值且 token 变了，则用新 token 覆盖 |
| `max_post_steps` | 16 | 16 | 当某块内已无 mask 后，仍允许再做多少轮「只做编辑」的 refinement，超过则进入下一块 |
| `temperature` | 0.0 | 0.0 | 0 表示贪婪（取最大概率）；>0 则按分布采样 |
| `steps` | 32 | 未传（用默认） | 在**当前实现里未被使用**（见下） |
| `num_to_transfer` | 1 | 未传（用默认 1） | 每轮最多「确定」下来的 mask 数量（见下） |

说明：`configuration_llada2_moe.py` 里主要是**模型结构**（层数、注意力头、RoPE、MoE 等），没有上述生成相关参数；生成策略全部由 `generate()` 的参数控制。

---

### 1.3 单块内的迭代逻辑（解掩码 + 编辑）

对**每一个块**（从含 prompt 的块开始，到覆盖 `gen_length` 的最后一块），内部是一个 `while True` 循环，直到满足退出条件：

1. **当前块状态**
   - `cur_x`：当前序列（prompt + 到当前块末尾），其中未确定位置为 `mask_id`。
   - `active_block_mask`：当前块里哪些位置仍是 mask。
   - 若当前块内**已经没有任何 mask**，则 `post_steps += 1`；若 `post_steps > max_post_steps`，则**退出该块**（进入下一块）。

2. **一次前向 + 采样**
   - 对 `cur_x` 做一次 `forward`（带块对角因果 attention mask），得到当前块对应的 logits。
   - 用 `_sample_with_temperature_topk_topp` 得到「当前块每个位置的预测 token」`x0` 和对应概率 `x0_p`（temperature=0 即 argmax）。

3. **解掩码（mask → 确定 token）**
   - 只考虑**当前仍是 mask 的位置**。
   - 若某 mask 位置的概率 `x0_p > threshold`，则视为「高置信」，可被选中为**本步要确定的 token**。
   - 每步最多确定 `num_to_transfer` 个位置（baseline 为 1）：
     - 若高置信 mask 数量 ≥ `num_to_transfer`，则这些高置信位置都参与「确定」；
     - 否则，在所有 mask 里按置信度取 top-`num_to_transfer` 确定。
   - 被选中的位置：用 `x0` 对应位置的 token 写回 `cur_x`，即该位置从 mask 变为真实 token。

4. **编辑（已生成 token 的替换）**
   - 只考虑**当前块内已不是 mask、且不是 prompt** 的位置。
   - 若某位置「新采样的 token」与旧 token 不同，且新 token 概率 `> editing_threshold`，则用新 token 覆盖该位置。

5. **本步是否有更新**
   - 若本步既没有解掉任何 mask，也没有任何编辑发生，则**退出该块**（进入下一块）。
   - 否则继续下一轮迭代（回到步骤 1）。

6. **eos_early_stop（块级）**
   - 每处理完一块，若 `eos_early_stop=True`，则检查：当前已生成部分（prompt 之后）是否还有 mask？
   - 若**没有 mask** 且**存在 eos_id**，则**整个 generate 结束**，不再往后扩块。

所以：**每步解多少 mask** 由 `threshold` + `num_to_transfer` 共同决定；**是否编辑、多严格** 由 `editing_threshold` 决定；**何时停** 由「块内无 mask 且无编辑」或 `max_post_steps`，以及 `eos_early_stop` 控制。

---

### 1.4 返回值

- 代码先取 `generated_answer = x[:, : prompt_length + gen_length]`，再在**生成部分**里找**第一个 eos_id** 的位置；若找到则截断到「该 eos 含在内」，若没找到则取满 `gen_length`。
- 因此返回的是：**input_ids（prompt） + 生成 token（至多到第一个 EOS 或 gen_length）**。

---

### 1.5 chat 模板（benchmark 侧）

benchmark 跑数统一使用 `tokenizer.apply_chat_template([{"role":"user","content":...}], add_generation_prompt=True, ...)`，**不再**提供 `--thinking` 或单独的 thinking 修复模块（历史说明见 `progress.md` 条目 31、39）。

---

### 1.6 未使用的参数与辅助函数

- **`steps`**：在 docstring 里写的是「每块迭代 refinement 的步数」，且有一行 `steps = min(steps, gen_length // minimal_topk)`，但**后续循环并未使用** `steps`；实际是「块内一直迭代直到无 mask 且无编辑，或超过 `max_post_steps`」。若要实现「每块固定步数」的变体，需要改循环逻辑并可能用到下面函数。
- **`_get_num_transfer_tokens(block_length, steps)`**：静态方法，根据 `block_length` 和 `steps` 得到每一步要「转移」的 token 数量序列（均匀分配），**在当前 `generate()` 中未被调用**，可能为其他解码策略或后续扩展预留。

---

### 本节小结（便于和 dInfer/其他策略对比）

- **块长**：`block_length=32`，块内并行、块间因果。
- **解掩码规则**：每步用 `threshold` 筛出高置信 mask，最多确定 `num_to_transfer` 个（baseline=1）。
- **编辑规则**：非 mask、非 prompt 位置，若新 token 概率 > `editing_threshold` 且 token 变化则覆盖。
- **停止**：块内无 mask 且无编辑则换块；若 `eos_early_stop` 且已生成部分无 mask 且含 EOS 则整体结束；最终按第一个 EOS 或 `gen_length` 截断返回。

---

## 1.7 LoPA (Lookahead Parallel Decoding) 解码策略设计

### 1.7.1 LoPA 核心思想与动机

**背景**：原 LLaDA2.1-mini 的解码策略虽然支持块内并行填充，但采用单一的贪婪策略（基于置信度阈值），可能导致局部最优问题。

**LoPA 思想**：
- 在每个 block 解码过程中，不是单一路径，而是维护多个**分支**（Anchor + Lookahead）；
- 通过 **Branch Confidence** 评估选择最优分支继续。

**与原论文的关系**：
- 参考 LoPA 论文（https://arxiv.org/abs/2512.16229）的核心思想；
- 针对 LLaDA2.1-mini 的 block-wise 特性进行适配，实现 "per-block 简化版 LoPA"。

### 1.7.2 技术实现架构

**LoPA 三阶段算法**（在每次迭代中执行）：

**第一阶段：基础分支构建（B0）**
- 使用标准 LLaDA2.1 置信度驱动采样（阈值 threshold）
- 将当前 block 内高置信度 mask 位置填充为预测 token，构建基础分支 B0
- 得到剩余未填充位置集合 MB0

**第二阶段：前瞻分支生成**
- 从 MB0 中按置信度排序，选择最高的 k 个位置 {p1, ..., pk}
- 对每个位置 pj，在 B0 基础上独立地将该位置从 mask 采样为具体 token
- 生成 k 个前瞻分支 {B1, ..., Bk}，每个分支只在一个特定位置进行前瞻填充
- **所有分支使用相同的解码策略**，确保算法一致性

**第三阶段：逐个评估选择**
- 对 B0 和 k 个前瞻分支分别进行独立前向传播（避免 LLaDA2.1-mini 的 batch mask 限制）
- 计算每个分支的 Branch Confidence（论文 Eq.2）
- 选择置信度最高的分支作为本次迭代的输出

**Branch Confidence 计算（论文 Eq.2）**：
$$C(B_j) = \frac{1}{|M_{B_j}|} \sum_{i \in M_{B_j}} \text{Conf}(i)$$
- 对该分支**剩余未填充位置** $M_{B_j}$ 的平均置信度（最大概率）
- 衡量"未来还有多少位置可以高置信度被填充"，即 future parallelism potential
- **注意**：不是对已填充位置计算，而是对剩余 mask 位置计算——这是与朴素实现的关键区别

**与论文的差异及合理性说明**：
- **分支验证方式**：论文设计为单次 batch forward 验证所有分支；实现中改为逐个 forward，原因是 LLaDA2.1-mini 对 batch attention mask 形状有严格要求（`(4,1,1,1)` 格式），无法直接打包多分支。逐个 forward 不影响正确性，仅增加计算量。
- **attention 机制**：论文在 D2F 上将 block causal attention 替换为 full attention 以提升分支间信息流；LLaDA2.1 必须保持 block causal mask，否则模型行为不符合训练时的假设。当前实现保持 block causal mask，是必要的适配。
- **attention mask 格式**：LLaDA2.1 的 `model.model()` 内部由 `_prepare_4d_causal_attention_mask_for_sdpa` 负责将 0/1 格式转为 additive 格式；我们传入 0/1 格式（与原生 `generate()` 一致），不做额外转换。
- **logits 精度**：`_llada_forward_logits` 在 `lm_head` 之后调用 `.float()`，与 LLaDA2.1 原生 `forward()` 保持一致（避免 bfloat16 精度下的数值误差）。

### 1.7.3 关键数据结构

```python
@dataclass
class LopaConfig:
    num_lookahead: int = 3                    # Lookahead 分支数
    branch_confidence_mode: str = "avg_prob"  # 分支置信度计算模式
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    # ... 其他参数与 baseline 一致

class LopaDecodingBranch:
    branch_id: int              # 分支标识
    x_t: torch.Tensor          # 当前序列状态
    confidence: float          # 分支置信度
    is_anchor: bool            # 是否为 Anchor 分支
    step_confidences: List[float]  # 历史置信度
```

### 1.7.4 算法流程

1. **初始化**：构造与原生 generate 一致的 block attention mask 和 position_ids；创建全 mask 初始序列。
2. **Block 迭代**：对每个需要解码的 block，调用 `_lopa_decode_block`：
   - 内部 `for iteration` 循环（符合扩散模型迭代本质）
   - 每次迭代：重新 forward 计算 logits → 执行 LoPA 三阶段 → 更新序列状态
   - 若 block 内容无变化则提前退出
3. **EOS 检测**：每个 block 处理完后检查是否出现 EOS，提前停止。
4. **后处理**：截断到第一个 EOS 位置，解码输出文本。

### 1.7.5 与 Baseline 的对比

| 维度 | Baseline (LLaDA2.1 原生) | LoPA |
|------|-------------------------|------|
| 分支数 | 1（单路径） | 1 + N（Base + Lookahead） |
| 迭代方式 | `while True` 直到收敛 | `for iteration` 多次迭代 |
| 前向计算 | 每 block 多次迭代，每次 1 次前向 | 每 block 多次迭代，每次 (N+1) 次前向 |
| logits 更新 | 每次迭代重新计算 | 每次迭代重新计算 |
| 分支评估 | 单分支 | 逐个分支前向（避免 batch mask 限制） |
| 与扩散模型一致性 | ✅ 完全符合 | ✅ 符合迭代本质 |

### 1.7.6 配置与使用

**默认配置**：
- `num_lookahead=3`：3 个 Lookahead 分支
- `branch_confidence_mode="avg_prob"`：平均概率作为置信度
- 其他参数与 baseline 保持一致

**使用方式**：
```bash
# 使用 LoPA 策略
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode lopa

# 对比 baseline 与 lopa
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode baseline
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode lopa
```

---

## 1.8 CCD (Coherent Contextual Decoding) 解码策略设计

### 1.8.1 CCD 核心思想与动机

**背景**：传统扩散语言模型解码策略仅依赖当前步的单步预测分布，缺乏对历史轨迹的利用，容易陷入局部最优。

**CCD 核心洞察**：
- 扩散模型在解码过程中，对同一个 mask 位置会在多个步骤中反复预测（随着上下文逐渐丰富）
- 跨步预测一致的 token 具有更高可靠性，更值得优先解码
- 通过检测跨步 argmax 一致性，无需置信度阈值即可自适应决定接受数量

**与原论文的关系**：
- 参考论文："Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models"
- 原始实现：https://github.com/tonyckc/CCD-DLM-code（`_sample_inference_lookahead`）
- 针对 LLaDA2.1-mini 的 block-wise 特性进行适配

### 1.8.2 技术实现架构（双步一致性检验）

**实现策略（与 CCD 开源代码对齐）**：

每步维护 top-V 高置信度位置及其 argmax token 的历史记录（最近 1 步），执行**双步一致性检验**：
- 当前步 top-V ∩ 上步 top-V，且 argmax token 相同 → **直接接受**（无需置信度阈值）
- 无一致候选时，退回到置信度阈值驱动的单步填充

**与 LLaDA2.1 block-wise 迭代的结合方式**：
- 全局 block attention mask（0/1 格式）一次性构建
- 历史 buffer 在**每个 block 内独立维护**，跨 block 不共享

**每次迭代的执行流程**：

1. 前向计算得到 block 所有位置的 logits、负熵置信度、argmax token
2. 尝试**一致性快速接受**（seek_mode=True 时启用）：双步检验找出稳定位置，直接接受
3. 无稳定位置时**退回阈值填充**：概率 > threshold 的 mask 位置接受；无则接受最高置信度位置
4. 编辑已生成 token（与原生 generate 一致）
5. 更新历史 buffer（存储更新后 active mask 下的 top-V 位置和 argmax token）

### 1.8.3 关键数据结构

```python
@dataclass
class CcdConfig:
    seek_mode: bool = True   # True=双步一致性加速（CCD-DS），False=纯阈值模式
    history_depth: int = 2   # buffer 深度（只用 depth=2，即比较当前与上一步）
    top_v: int = 4           # 每步保留 top-V 置信度位置

class CcdHistoryBuffer:
    # 每个 block 独立实例化，跨 block 不共享
    _history: deque  # 每个元素：(positions [k], argmax_tokens [k])
```

**与 CCD 原始代码的对应关系**：
- 原始代码的 `history_maximum_token_id_list[-2]`（两步前 argmax）→ 我们的 `_history[-1]`（上步 argmax，效果等价）
- 原始代码的 `verification_buffer`（上步 top-V 位置）→ 我们的 `_history[-1]` 中的 positions
- 双步一致性检验逻辑完全对应
- **attention mask 格式**：传入 0/1 格式（与 LLaDA2.1 原生一致），无需额外转换（原始代码基于 Dream 格式不同）
- **logits shift**：LLaDA2.1 是 masked diffusion 模型，无需 shift（原始代码的 shift 是 Dream 模型特有）
- **logits 精度**：`lm_head` 后 `.float()`，与 LLaDA2.1 原生 `forward()` 一致

### 1.8.4 算法流程

1. 初始化全局 block attention mask；创建全 mask 序列
2. 对每个 block，调用 `_ccd_decode_block`：
   - 为该 block 创建独立 `CcdHistoryBuffer`
   - 循环：forward → 双步一致性检验 → （若无则阈值填充）→ 编辑 → 更新 buffer
   - 内容无变化则提前退出
3. EOS 检测，提前停止
4. 后处理截断，解码输出

### 1.8.5 与其他策略的对比

| 维度 | Baseline | LoPA | CCD |
|------|----------|------|-----|
| 核心机制 | 置信度阈值解码 | 多分支前瞻探索 | 双步 argmax 一致性检验 |
| 步数控制 | 固定步数 | 固定步数 | 自适应（一致位置多则跳步多） |
| 前向计算 | 每步 1 次 | 每步 (k+1) 次 | 每步 1 次 |
| 内存开销 | 基准 | 高（多分支） | 极低（只存 top-V argmax） |
| 计算开销 | 基准 | 约 (k+1) 倍 | 约 1 倍（无额外 forward） |
| 预期优势 | 稳定可靠 | 准确性提升 | 速度快 + 质量稳定 |

### 1.8.6 配置与使用

```bash
# 使用 CCD 策略
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode ccd
```

---

## 1.9 CLAD (Consistency-guided Lookahead Adaptive Decoding)——毕设核心创新

### 1.9.1 设计动机与创新点

**背景**：CCD 的一致性机制是"被动等待"——只能在一致性自然出现时加速；LoPA 的前瞻机制是"主动引导"——但评估分支的标准（未来 mask 置信度）仅关注概率，未考虑跨步稳定性。

**核心问题**：能否将 CCD 的一致性思想从"被动检测"提升为"主动引导"？

**CLAD 的回答**：是的。通过将"哪个分支能让其他位置更快达成一致"作为前瞻评分的核心依据：

> **一致性传播假说**：如果向位置 $p$ 填入 token $t$ 后，其他剩余 mask 位置的新预测与当前步预测高度一致，说明 $t$ 是这个上下文中"最自然"的选择——它与周围 token 的共现关系稳定，其他位置在下一步更容易通过一致性检验而被加速接受。

**与 CCD / LoPA 的区别（三者核心差异表）**：

| 维度 | CCD | LoPA | **CLAD** |
|------|-----|------|----------|
| 一致性利用 | 被动检测，出现即接受 | 不使用 | **主动引导前瞻** |
| 前瞻分支评分 | 不做前瞻 | 未来 mask 置信度 | **一致性传播分数** |
| 不确定时策略 | 退回阈值填充 | 前瞻选最优分支 | **先一致性检验，再前瞻，最后阈值** |
| 计算开销 | 每步 1 次 forward | 每步 (k+1) 次 | **按需**：一致命中则 1 次；否则 (k+1) 次 |

### 1.9.2 算法设计（v1）

**三阶段自适应解码（每次迭代）**：

```
阶段一 ──── 一致性快速通道（CCD-inspired）
  ↓ buffer 有历史时
  当前步 top-V ∩ 上步 top-V，且 argmax 相同 → 直接接受（无额外 forward 开销）
  ↓ 无一致候选
阶段二 ──── 一致性传播前瞻（CLAD 创新，预热 lookahead_warmup 步后启用）
  ↓ 对 top-k 不确定位置各生成一个候选分支（填入该位置的 argmax token）
  ↓ 对每个分支做一次 forward，计算分支评分：
    score(B_j) = α × consistency_propagation(B_j) + (1-α) × future_confidence(B_j)
    consistency_propagation：填入后，其他剩余 mask 位置新预测与当前步预测一致的比例（均匀权重）
    future_confidence：其他剩余 mask 位置的平均最大概率（LoPA-style）
  ↓ 选评分最高的分支，填入对应位置的 token（每次只接受 1 个）
  ↓ 无有效前瞻时
阶段三 ──── 阈值填充（退回 baseline 行为）
  高于 threshold 的 mask 位置接受；无则接受最高置信度位置
```

**一致性传播分数（Consistency Propagation Score）的直觉**：

填对了一个词，周围的词就会稳定；填错了一个词，模型对周围词的预测就会抖动。CLAD 用这种“预测稳定性”来判断哪个分支方向对。

---

## 1.10 CLAD-v2——信息量视角的进一步优化

### 1.10.1 v1 的两个核心局限

1. **一致率计算太粗**：v1 用均匀权重的简单平均，所有剩余 mask 位置对一致性得分的贡献相同。但让一个"高熵（高度不确定）"位置变稳定，携带的信息量远多于让一个已经很确定的位置维持原预测。
2. **每次只接受 1 个 token**：阶段二每次做 k 次 forward，最终只填入 1 个 token，TPF（Tokens Per Forward）极低。最优分支的 logits 直接被丢弃，是一种浪费。

### 1.10.2 O1：信息密度加权的一致性评分

**理论基础**：参考 *Mask Is What DLLM Needs*（信息密度驱动训练范式）的推理侧镜像——训练时高信息密度位置应被重点优化，解码时让高信息密度位置稳定也更有意义。

**改动**：将 `consistency_propagation` 替换为三项联合评分：

$$
\text{score} = \alpha \cdot \text{weighted\_consistency} + \beta \cdot \text{entropy\_reduction} + (1-\alpha-\beta) \cdot \text{future\_confidence}
$$

- **weighted_consistency**：以当前步的"熵 $H_i = -\text{neg\_ent}_i$"为权重的加权一致率。高熵位置（不确定性高 = 信息密度高）在评分中权重更大，迫使模型选择能真正锚定上下文的 token。
- **entropy_reduction**：填入 token 后，剩余 mask 位置的平均负熵增量（越正 = 越确定），量化"这个 token 为上下文提供了多少互信息"。结果归一化到 $[-1, 1]$（除以 $\log V$）。
- **future_confidence**：继承 v1 的 LoPA-style 项。

**默认参数**：`consistency_weight=0.5`（α），`entropy_weight=0.2`（β），future_conf 权重 = 0.3。

### 1.10.3 O2：多 token 自适应接受

**核心思想**：选出最优分支并填入第 1 个 token 后，不丢弃该分支的 logits，而是检查剩余 mask 位置中置信度最高的位置：

```
if branch_probs[best_remaining_pos] ≥ accept_threshold2:
    额外接受第 2 个 token（零额外 forward 成本）
```

- 每次 Phase-2 激活：从"接受 1 token / k+1 次 forward"升级为"最多接受 2 token / k+1 次 forward"，TPF 理论上提升约 2×（在高置信场景下）。
- `accept_threshold2=0.90`（默认）；设为 1.0 退化回 v1 行为。

### 1.10.4 参数对比

```python
# CLAD v1（llada_clad_decode.py）
CladConfig(
    top_v=4, num_lookahead=2, consistency_weight=0.6, lookahead_warmup=3
)

# CLAD v2（llada_clad_v2_decode.py）
CladV2Config(
    top_v=4, num_lookahead=2,
    consistency_weight=0.5,   # α（加权一致率）
    entropy_weight=0.2,       # β（熵下降奖励，新增）
    lookahead_warmup=3,
    accept_threshold2=0.90,   # O2 阈值（新增）
)
```

### 1.10.5 五种策略完整对比

| 维度 | Baseline | LoPA | CCD | CLAD v1 | CLAD v2 | **CLAD v3** |
|------|----------|------|-----|---------|---------|------------|
| 核心机制 | 置信度阈值 | 多分支前瞻 | 双步一致性 | 一致性引导前瞻 | 信息密度加权 + 多 token 接受 | **v2 + Phase-2 batch forward + 级联 top-2 L1** |
| 步数控制 | 固定 | 固定 | 自适应 | 自适应 | 自适应 | **自适应** |
| 前向计算/步 | 1 次 | (k+1) 次 | 1 次 | 1 或 (k+1) 次 | 1 或 (k+1) 次 | **Phase-2 常为 1 次 batched（+ 主步 1 次）** |
| 每次 Phase-2 接受 token 数 | — | 1 | — | 1 | 1～2 | **1～2（同 v2）** |
| 一致性评分权重 | — | — | — | 均匀 | 高熵位置加权 | **同 v2** |
| Phase-2 分支计算 | — | 串行多前向 | — | 串行多前向 | 串行多前向 | **默认批量 1 次前向（O3）** |
| 预期 TPF | 基准 | 中 | 高 | 高 | 更高 | **在 v2 基础上降低 Phase-2 前向次数** |

### 1.10.6 配置与使用

```bash
# CLAD v1
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode clad

# CLAD v2（信息密度加权 + 多 token 接受）
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode clad_v2

# 五种策略横向对比
python dlm/src/run_benchmark_llada2.py --benchmark arc_challenge --decode_mode baseline
python dlm/src/run_benchmark_llada2.py --benchmark arc_challenge --decode_mode lopa
python dlm/src/run_benchmark_llada2.py --benchmark arc_challenge --decode_mode ccd
python dlm/src/run_benchmark_llada2.py --benchmark arc_challenge --decode_mode clad
python dlm/src/run_benchmark_llada2.py --benchmark arc_challenge --decode_mode clad_v2
python dlm/src/run_benchmark_llada2.py --benchmark arc_challenge --decode_mode clad_v3
```

### 1.10.7 备注：一致性通道的阈值门控（中期后候选实验）

当前 CCD / CLAD(v1/v2) 的一致性快速通道采用：

- `当前步 top-V ∩ 上步 top-V`
- 且 `argmax token` 跨步一致

作为直接接受条件；该通道本身**未额外设置绝对最小置信度阈值**（例如 `p >= 0.5`）。

这与 baseline 的阈值填充机制不同：`threshold` 主要用于一致性失败后的 fallback 阶段。

为进一步降低“低绝对置信度但偶然一致”的误接受风险，中期后可增补一个可选门控参数：

- `consistency_min_conf`（默认关闭，保持当前行为）
- 当启用时，一致性候选需同时满足 `cur_prob >= consistency_min_conf` 才接受

建议消融网格：`None / 0.5 / 0.6 / 0.7 / 0.8`，并同时报告 Accuracy、Throughput、TPF、Diffusion Steps，评估“质量-速度”折中。

---

## 1.11 CLAD-v3——O3 分支批量 forward + O4 级联草稿

**实现文件**：`dlm/src/llada_clad_v3_decode.py`  
**运行**：`python dlm/src/run_benchmark_llada2.py --decode_mode clad_v3`

在 **CLAD v2 的 O1（信息密度加权分支评分）与 O2（同一分支 logits 上二次高置信接受）** 保持不变的前提下，v3 针对 Phase-2 的 **GPU 利用率** 与 **草稿深度** 做两步扩展。

### 1.11.1 O3｜Phase-2 分支批量化 forward

**动机**：Spiffy、LoPA 原论文均指出，在块因果结构下，多个仅在某 mask 位置不同的候选序列可视为 batch 维上的独立样本，用一次矩阵乘并行算出各分支 logits，比串行 `k` 次 forward 更利于 GPU 吞吐。

**做法**：将 `k` 条 `branch_x`（均为 `[1, window_end]`）沿 batch 维拼为 `input_ids`，形状 `[B, L]`；将全局块注意力子矩阵 `cur_attn_mask` `[1,1,L,L]` 沿 batch 维 `expand(B,1,L,L)`，`position_ids` 同理扩展为 `[B,L]`，调用与 v2 相同的 `model.model` + `lm_head` 路径 **`_llada_forward_logits_batched`**，**`forward_count` 仅 +1**。

**开关**：`CladV3Config.use_batched_phase2=True`（默认）。若在某环境出现与 batch 相关的兼容性错误，可置 `False` 退化为逐分支前向（与旧版等价，仅慢）。

### 1.11.2 O4｜级联草稿前瞻（Spiffy-inspired，与 O3 合用）

**动机**：在「一次 Phase-2 前向」内不仅比较「单点填入」的优劣，还希望利用 **胜者分支上的分布** 再决定是否多接受一个 token（与 O2 思想一致，但 L1 侧只比较 **top-2 置信位置** 两条分支，结构更清晰）。

**默认行为**（`use_cascaded_draft=True`）：

1. **Level-1**：在当前仍为 mask 的位置中，按当前步最大概率取 **前 2 个位置** `p1, p2`，构造两条仅在该点填入 argmax 的分支；**一次 batched forward（B=2）** 得到 `logits[0]`, `logits[1]`。
2. **评分**：对两行分别用 **O1 综合分** `_clad_branch_score_v2`（与 v2 公式一致：加权一致性 + 熵降 + future_conf）比较，取最优行对应的填入位置与 token，写回序列。
3. **Level-2 / O2**：**不新增 forward**，直接使用 **胜者行** 在块内的 logits，对剩余 mask 位置取最大置信度；若 ≥ `accept_threshold2`（默认 0.90），再写入第二个 token。

**关闭 O4 时**（`use_cascaded_draft=False`）：Level-1 改为取 `num_lookahead` 个 top 位置，构造 `B=k` 条分支，一次 batch forward 后同样用 O1 选优，再接 O2。适用于希望与「flat top-k」更接近的对照实验。

### 1.11.3 与 v2 的差异小结

| 维度 | CLAD v2 | CLAD v3 |
|------|---------|---------|
| O1 / O2 | ✅ 同左 | ✅ 同左（同一套 `_clad_branch_score_v2` + `_apply_o2_second_token`） |
| Phase-2 前向 | 多分支时多为 **串行 k 次** | 默认 **1 次 batched**（O3） |
| L1 候选结构 | top-`k` 各一分支 | 默认 **top-2** 级联叙事（O4），可关 |
| 逻辑等价性 | — | 在「分支独立、评分公式不变」前提下与串行等价；数值应一致或仅浮点差异 |

### 1.11.4 配置示例

```python
# llada_clad_v3_decode.CladV3Config（节选）
CladV3Config(
    top_v=4,
    num_lookahead=2,
    consistency_weight=0.5,
    entropy_weight=0.2,
    lookahead_warmup=3,
    accept_threshold2=0.90,
    use_batched_phase2=True,   # O3
    use_cascaded_draft=True,    # O4（False 则 flat top-k batch）
    vocab_size=156896,
    # 其余 gen_length / block_length / threshold 等与 v1/v2 对齐
)
```

```bash
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode clad_v3 --max_examples 50
```

---

## 2. Benchmarks 与评测总体设计

### 2.1 选择的基准数据集（按任务类型）

**Math / 推理**
- **GSM8K**（`openai/gsm8k`，`main` 配置）：小学到早期代数难度的英文数学题，要求多步推理，答案中带有自然语言解题过程与最终数字结果（形如 `#### 72`）。
- **AIME2025**（`opencompass/AIME2025`）：AIME 竞赛 2025-I / 2025-II 的正式试题，难度远高于 GSM8K，更接近「竞赛数学」水平，适合考察模型在高难度题上的解码稳定性。

**Code / 编程**（已实现支持）
- **MBPP**（`Muennighoff/mbpp`，`sanitized` 配置）：入门到中等难度的 Python 编程小题，每题包含自然语言描述、参考解与若干测试用例，适合评估代码生成的功能正确性（pass@k）。
- **HumanEval**（`openai/openai_humaneval`）：164 个手写 Python 编程题，包含函数签名、docstring、参考解和测试代码，是经典的 code generation benchmark。

**Reasoning / 科学问答**（已实现支持）
- **ARC-Easy**（`allenai/ai2_arc`，`ARC-Easy` 子集，300 条）：小学到初中难度的科学多选题，四选一，适合评估基础科学常识推理。
- **ARC-Challenge**（`allenai/ai2_arc`，`ARC-Challenge` 子集，300 条）：从 ARC-Easy 中筛选出规则方法和词频方法均答错的难题，难度更高，更能区分模型真实推理能力。
- 两个子集均保存在 `dlm/experiments/benchmarks/reasoning/` 目录下（新建子目录）。
- 字段：`id, question, choices（label+text 列表）, answer_key（正确字母）, answer_text（正确选项文本）`。

**Knowledge / 科学问答**（预备）
- **GPQA**（`Idavidrein/gpqa`）：高难度理工科多选题，需要科研级知识与推理。

### 2.2 本地文件组织与统一样本格式

所有原始 benchmark 统一转存为 jsonl，一行一条样本，位于：

**数学基准测试**：
- `dlm/experiments/benchmarks/math/gsm8k_small.jsonl`：字段 `id, split, question, answer`
- `dlm/experiments/benchmarks/math/aime2025_all.jsonl`：字段 `id, subset, question, answer`

**代码基准测试**：
- `dlm/experiments/benchmarks/code/humaneval_all.jsonl`：字段 `id, prompt, reference_code, tests, entry_point`
- `dlm/experiments/benchmarks/code/mbpp_sanitized.jsonl`：字段 `id, prompt, reference_code, tests`

**推理/科学问答基准测试**：
- `dlm/experiments/benchmarks/reasoning/arc_easy.jsonl`：字段 `id, question, choices, answer_key, answer_text`
- `dlm/experiments/benchmarks/reasoning/arc_challenge.jsonl`：字段 `id, question, choices, answer_key, answer_text`

**知识问答基准测试**（预备）：
- 后续 GPQA 等 knowledge 类 benchmark 将采用类似字段：`id, question, options, answer`

### 2.3 评测流水线的总体结构

**运行脚本**：`dlm/src/run_benchmark_llada2.py`
- 加载并修复 LLaDA2.1-mini 模型（含 RoPE `"default"` 注册）
- 从指定的 benchmark jsonl 文件中逐条读取样本
- 用 tokenizer 的 `apply_chat_template` 构造 chat 风格输入
- 使用指定解码策略生成回答
- 将结果写入 `dlm/experiments/runs/` 下带时间戳的 jsonl 文件

**命令行接口**：
- `--benchmark`：选择单个数据集（`gsm8k_small` / `aime2025_all` / `humaneval_all` / `mbpp_sanitized` / `arc_easy` / `arc_challenge`）
- `--benchmarks`：选择多个数据集（**推荐**，模型只加载一次）
- `--decode_mode`：解码策略（`baseline` / `lopa` / `ccd` / `clad` / `clad_v2` / `clad_v3`，默认 `baseline`）
- `--max_examples`：可选，限制最多跑多少条样本

**评测脚本**：`dlm/src/evaluate_benchmark_results.py`
- **仅保留 LLM-as-a-Judge 模式**：所有准确率统计均由 Judge 输出的 CORRECT/INCORRECT 决定
- Judge 使用 **Qwen3.5-9B**，路径：`/etc/moreh/checkpoint/Qwen3.5-9B`，需在 `.venv_eval` 环境下运行
- **Judge Prompt 按题型分支**：
  - **选择题**（ARC 等，record 含 `answer_key` 字段）：展示所有选项 + 正确答案字母，Judge 只需判断模型最终选择的字母是否正确
  - **开放题**（GSM8K、AIME 等）：展示题目 + 参考答案，Judge 判断数值/推理等价性
- **效率指标**：
  - `throughput_tok_per_sec`：总输出 token 数 / 总生成时间（整体吞吐量）
  - `correct_throughput`：正确样本的总输出 token 数 / 所有样本的总生成时间（正确吞吐量，核心横向对比指标）
  - **TPF（Tokens Per Forward）**：需 run 结果中含 `forward_count`（由 `run_benchmark_llada2.py` 写入）。定义 `TPF = output_token_len / forward_count`；报告给出 macro（Σoutput/Σforward）与逐样本均值。旧 jsonl 无 `forward_count` 时无法计算。
- **日志**：Judge 的完整 prompt 和原始回答记录在 `dlm/log/eval/` 下的 `_judge_details.log` 文件中，可用于人工核查误判
- 报告默认输出到 `dlm/experiments/evals/` 目录

```bash
# 运行基准测试（数学 + ARC）
python dlm/src/run_benchmark_llada2.py --benchmarks gsm8k_small arc_easy arc_challenge --decode_mode baseline

# 运行基准测试（高难度数学）
python dlm/src/run_benchmark_llada2.py --benchmark aime2025_all --decode_mode baseline

# 评测结果（需在 .venv_eval 环境下运行）
python dlm/src/evaluate_benchmark_results.py --results_dir dlm/experiments/runs/ --judge_model_path /etc/moreh/checkpoint/Qwen3.5-9B
```

### 2.4 与解码策略实验的衔接方式

**当前 baseline 参数**：`gen_length=2048, block_length=32, threshold=0.7, editing_threshold=0.5, temperature=0.0, max_post_steps=16, eos_early_stop=True`

**后续解码策略变体**：在不改动 benchmark 读写逻辑的前提下，只需根据 `decode_mode` 分支修改生成参数或包装 `generate` 调用逻辑，同一批输入样本可在不同 `decode_mode` 下重复跑一遍，比较：
- Math / knowledge：数值准确率、选项命中率等
- Code：测试通过率（pass@1 / pass@k）

---

## 3. 资源使用与多卡策略（当前约定）

### 3.1 当前默认：单卡推理，便于与同学协作

`run_benchmark_llada2.py` 中的设备选择逻辑：
```python
use_cpu = bool(os.environ.get("DLM_CPU"))

if use_cpu or not torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(..., device_map=None)
    model = model.to(torch.bfloat16).to("cpu")
else:
    model = AutoModelForCausalLM.from_pretrained(
        ..., device_map="auto", torch_dtype=torch.bfloat16,
    )
```

如需绑定到指定卡（例如只用 5 号卡）：
```bash
export HIP_VISIBLE_DEVICES=5
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --max_examples 10
```

### 3.2 tmux 多窗口绑卡并行（推荐多任务场景）

利用 tmux 多窗口实现"每窗口绑定不同 GPU"的并行方式，详见 `dlm/docs/env.md` 第 7.2.1 节。

### 3.3 多进程并行版本（高吞吐量模式）

针对 LLaDA2.1-mini 不支持批量推理的限制，提供 `run_benchmark_llada2_multiprocess.py`：
- 每个 GPU 启动独立 Python 进程，各自加载完整模型，处理分配的样本子集
- 理论吞吐量接近线性提升（4 卡约 4 倍）

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7
python dlm/src/run_benchmark_llada2_multiprocess.py --benchmark gsm8k_small --max_examples 100
```

**选择建议**：
- **单进程**（`run_benchmark_llada2.py`）：调试、小规模实验（< 50 样本）
- **多进程**（`run_benchmark_llada2_multiprocess.py`）：大规模 benchmark（> 50 样本）
