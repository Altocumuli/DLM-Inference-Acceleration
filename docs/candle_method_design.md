# CANDLE 方法设计文档

## 0. 文档目标

本文档用于设计一种建立在 `CLAD v2` 之上的新解码方法，用于探索更具研究潜力的扩散语言模型推理加速方向。该方法的核心目标不是简单堆叠工程 trick，而是尝试形成一条相对完整的方法论叙事：

- `CLAD v2` 负责回答：**哪个位置最值得先填？**
- 新方法负责回答：**一旦找到了可靠的 anchor，它附近还有哪些 token 可以一起安全推进？**

该思路受到 `LocalLeap`（*Accelerating Diffusion LLM Inference via Local Determinism Propagation*）启发，但不会直接复现其“高置信 anchor + 邻域宽松接受”的原始流程，而是将其与 `CLAD v2` 的一致性/信息量评分机制结合，形成一套更适合本工作叙事的新方法。

---

## 1. 命名建议

推荐方法名：

## **CANDLE**

全称：

**Consistency-ANchored Determinism LEap**

命名理由：

1. `Consistency-Anchored` 对应本方法首先使用 `CLAD v2` 的一致性/信息量联合评分来确定 anchor；
2. `Determinism Leap` 对应从 `LocalLeap` 借鉴来的“局部确定性传播”思想；
3. 相比 `v3 / v4`，`CANDLE` 更像一条新的方法线，适合后续论文叙事与投稿包装。

可选备选名：

- `CLAD-L`：CLAD with Local Leap
- `AnchorLeap`：Anchored Local Determinism Propagation
- `ConsiLeap`：Consistency-guided Local Leap

如果后续考虑投稿，`CANDLE` 更适合作为论文中的方法名。

---

## 2. 设计动机

### 2.1 当前 `CLAD v2` 的优势

`CLAD v2` 已经具备两项很有价值的机制：

1. **O1：信息密度加权分支评分**
   - 使用 `weighted_consistency + entropy_reduction + future_confidence` 评估 top-k 候选分支；
   - 能比单纯置信度更稳地选出“更值得信任”的填入位置。

2. **O2：零额外 forward 的第二 token 接受**
   - 在 winner branch 上利用已有 logits 再接受一个高置信 token；
   - 提高了每次 Phase-2 的 token 利用率。

### 2.2 当前 `CLAD v2` 的局限

尽管 `CLAD v2` 已经比 `v1` 更强，但仍存在一个明显瓶颈：

- **winner branch 已经被选出来了，但后续只做一次“全局单 token” 的 O2 接受。**

也就是说，`CLAD v2` 已经知道了：

- 哪个分支在一致性和信息量意义上最合理；

但并没有进一步利用这个“被选中的局部正确方向”，去做更细粒度的局部传播。

### 2.3 `LocalLeap` 的关键启发

`LocalLeap` 的真正核心不在于“阈值调得更巧”，而在于：

1. 先找到一个高置信 anchor；
2. 认为 anchor 周围的 token 往往也处在更容易确定的局部区域；
3. 对这个局部邻域采用更宽松的接受规则；
4. 一次接受多个 token。

其代表性逻辑可以概括为：

- 先找 `confidence >= threshold` 的 anchor；
- 再把 anchor 半径 `r` 内的位置视为“局部可传播区域”；
- 对邻域位置使用 `relaxed_threshold`。

### 2.4 为什么不能直接照搬 `LocalLeap`

若直接把 `LocalLeap` 生搬进当前系统，会有两个问题：

1. **anchor 选择过于依赖 raw confidence**
   - 无法体现本工作中“一致性 + 信息量”的核心贡献；
   - 容易被审稿人理解为“LocalLeap 风格的阈值技巧复用”。

2. **全局邻域扩张太激进**
   - 在 GSM8K / MATH 这类数值约束较强的任务上，容易把局部错误更快扩散；
   - 会削弱 `CLAD v2` 原本通过 O1 获得的 anchor 选择优势。

因此，更合理的路线不是“高置信邻域传播”，而是：

> **一致性锚定后的局部确定性传播**

---

## 3. 核心思想

### 3.1 一句话概括

**先用 `CLAD v2` 的 O1 找到最值得信任的 anchor 分支，再只在该 anchor 的局部邻域内进行 `LocalLeap` 式的多 token 接受。**

### 3.2 角色分工

- `CLAD v2 / O1`：负责选择 anchor
- `CANDLE / Local Leap`：负责 anchor 周围的局部传播

换言之：

- `CLAD v2` 回答：**先填哪里**
- `CANDLE` 回答：**围绕这个位置还能一起填哪里**

这构成了一个更完整的方法闭环。

---

## 4. 方法流程

## 4.1 总体结构

方法仍保留三阶段框架：

1. **Phase-1**：一致性快速通道
2. **Phase-2**：一致性锚定 + 局部确定性传播
3. **Phase-3**：fallback 阈值填充

其中主要改动发生在 `Phase-2`。

## 4.2 Phase-2 新流程

### Step A：候选 anchor 分支生成

与 `CLAD v2` 保持一致：

- 从当前 block 的 masked 位置中选出 top-k 候选位置；
- 对每个候选位置构造一个 branch；
- 对每个 branch forward 一次；
- 用 O1 评分：

$$
\text{score}(B_j) = \alpha \cdot \text{weighted\_consistency}
+ \beta \cdot \text{entropy\_reduction}
+ (1-\alpha-\beta)\cdot \text{future\_confidence}
$$

### Step B：选出 winner anchor

选分数最高的 branch，记其填入位置为：

$$
p^\*
$$

该位置对应的 token 为：

$$
t^\*
$$

先将 `(p^\*, t^\*)` 写入序列。

### Step C：构造局部传播邻域

在 winner branch 的块内，定义 anchor 邻域：

$$
N_r(p^\*) = \{i \mid |i - p^\*| \le r,\ i \in \text{remaining mask}\}
$$

其中 `r` 是局部传播半径。

### Step D：局部宽松接受

只在邻域 `N_r(p^\*)` 中，查看 winner branch logits 对各位置的最大概率：

$$
\max_y p(y \mid B^\*, i)
$$

对于满足以下条件的位置，允许额外接受：

$$
\max_y p(y \mid B^\*, i) \ge \tau_{\text{local}}
$$

其中：

- `tau_local` 为局部宽松阈值；
- 一般设置为低于当前 `accept_threshold2` 的值。

同时为了避免过于激进，第一版建议加入：

- 最多只接受 `m` 个邻域 token；
- 仅接受 top-m 高置信邻域位置。

### Step E：O2 fallback

如果邻域传播没有找到额外可接受 token，则退回当前 `O2`：

- 在整块剩余 masked 位置中找最高置信位置；
- 若概率大于 `accept_threshold2`，则接受 1 个额外 token。

这样可以保证：

- `CANDLE` 至少不弱于 `CLAD v2`；
- 局部传播失败时仍有原始 O2 保底。

---

## 5. 与 `LocalLeap` 的关键区别

## 5.1 anchor 来源不同

`LocalLeap`：

- 由 raw confidence threshold 找 anchor

`CANDLE`：

- 由 O1 的一致性/信息量联合评分选 anchor

这意味着：

- `LocalLeap` 传播的是“高置信局部”
- `CANDLE` 传播的是“被一致性验证过的高价值局部”

## 5.2 传播范围不同

`LocalLeap` 倾向于在当前 mask 区域中直接做 anchor 邻域传播；

`CANDLE` 则只在：

- **winner branch**
- **winner anchor 的局部邻域**

内传播。

这是一个更保守、更适合数学/推理任务的设计。

## 5.3 方法叙事不同

`LocalLeap` 的叙事：

- 局部确定性可以传播

`CANDLE` 的叙事：

- 不是所有局部都值得传播；
- **只有在一致性和信息量意义上可靠的 anchor，才值得触发局部 leap。**

这点对投稿叙事尤其重要。

---

## 6. 第一版实现边界

为了降低实现风险，建议第一版只做最小可验证版本。

### 6.1 第一版包含

1. 保留当前 `CLAD v2` 的 Phase-1 / O1 / O2 主体结构；
2. 在 winner branch 上新增 “local leap accept”；
3. 单 anchor；
4. 固定半径；
5. 固定局部宽松阈值；
6. 最多额外接受 `2~3` 个邻域 token。

### 6.2 第一版不做

1. 不做多 anchor 联合传播；
2. 不做动态半径；
3. 不做动态 relaxed threshold；
4. 不做跨 block 邻域传播；
5. 不做全局 `LocalLeap` 风格的 mask 区域扩张。

这样做的原因是：

- 先验证方向，而不是一开始就做过多机制耦合；
- 更便于和 `CLAD v2` 做清晰 ablation。

---

## 7. 建议新增配置项

建议在新方法的 config 中加入：

```python
use_local_leap: bool = True
local_radius: int = 2
local_relaxed_threshold: float = 0.78
max_local_accept: int = 2
anchor_score_threshold: float = 0.0
local_consistency_floor: float = 0.55
local_entropy_floor: float = -0.02
```

### 参数解释

- `use_local_leap`  
  是否启用 CANDLE 的局部传播机制。

- `local_radius`  
  邻域半径，控制传播范围。

- `local_relaxed_threshold`  
  邻域 token 的接受阈值，应低于 `accept_threshold2`。

- `max_local_accept`  
  一次局部传播最多额外接受多少个 token。

- `anchor_score_threshold`  
  只有 winner branch 的 O1 分数超过该阈值，才允许局部传播。

- `local_consistency_floor`  
  若 winner branch 的 `weighted_consistency` 太低，则不做传播。

- `local_entropy_floor`  
  若 winner branch 的 `entropy_reduction` 太差，则不做传播。

---

## 8. 关键门控策略

为了避免把错误 anchor 的邻域一起扩散，建议在局部传播前加入 gate。

### 建议 gate

只有满足以下至少一条时，才启用 local leap：

1. `winner_score >= anchor_score_threshold`
2. `weighted_consistency >= local_consistency_floor`
3. `entropy_reduction >= local_entropy_floor`

如果 gate 不通过：

- 仅接受 anchor token；
- 然后退回原始 `O2` 或直接结束本次 `Phase-2`。

### 设计直觉

这一步非常关键，因为它决定了：

- 方法是“保守增强版 `CLAD v2`”
- 还是“容易把错误事实扩散得更快的激进版 `LocalLeap`”

我们显然要前者。

---

## 9. 伪代码

```text
Input: current block state x

Phase-1:
    if consistency fast-track hits:
        accept consistent tokens
        return

Phase-2:
    build top-k candidate branches
    forward each branch
    compute O1 score for each branch
    choose winner branch B*
    accept winner anchor token at p*

    if local-leap gate(B*) passes:
        collect remaining masked positions within radius r of p*
        rank them by winner-branch confidence
        accept top-m positions whose confidence >= tau_local

    if no local token accepted:
        run original O2 fallback

Phase-3:
    if Phase-2 failed:
        use threshold fallback
```

---

## 10. 代码实现落点建议

第一版 prototype 最适合直接在 `CLAD v2` 基础上实现，不建议一开始单独开全新文件体系。

### 10.1 推荐落点

主要修改点：

- `dlm/src/llada_clad_v2_decode.py`

重点函数：

1. `_clad_lookahead_fill(...)`
   - 当前 winner branch 选择逻辑保留
   - 在写入 winner token 后加入 `local leap` 逻辑

2. `_apply_o2_second_token(...)`
   - 可保留原函数，作为 fallback
   - 新增一个更通用的 `_apply_local_leap(...)`

### 10.2 推荐新增函数

```python
def _apply_local_leap_tokens(
    x,
    branch_logits_1,
    anchor_pos,
    block_start,
    block_end,
    config,
):
    ...
```

它输出：

- `accepted_positions`
- `accepted_tokens`
- `accepted_probs`
- `fired`
- `used_local_radius`
- `used_local_threshold`

---

## 11. Trace 与可解释性记录

因为这条方法线明显会更依赖“局部传播为何被触发”，所以 trace 必须增强。

建议记录以下字段：

```json
{
  "anchor_pos": ...,
  "anchor_token": ...,
  "anchor_score": ...,
  "local_gate_passed": true,
  "local_radius": 2,
  "local_relaxed_threshold": 0.78,
  "local_candidates": [...],
  "local_accepted_positions": [...],
  "local_accepted_tokens": [...],
  "local_accepted_probs": [...],
  "local_accept_count": 2
}
```

这些字段后续会直接决定：

- case study 能不能解释“为什么这个邻域被一起接受”
- 方法是否能写成一条完整研究故事

---

## 12. 预期收益

### 12.1 可能的正向收益

1. **吞吐提升**
   - 把当前 O2 的“单 token 额外接受”升级为“局部多 token 接受”；
   - 在 winner branch 已较可靠时，可能显著提高 token / forward。

2. **更贴合 ARC / MATH 的局部结构**
   - 选择题和短数学题中，正确局部模式往往具有邻域相关性；
   - 若 anchor 正确，其附近 token 常常更容易一起确定。

3. **方法叙事更完整**
   - `CLAD v2` 已经有“更好的 anchor 选择”；
   - `CANDLE` 让这个 anchor 真正带动局部传播。

### 12.2 可能的风险

1. **GSM8K 类事实漂移被放大**
   - 若 anchor 局部看似高置信、但全局约束未稳住，传播可能扩大错误。

2. **参数更敏感**
   - `radius` 和 `local_relaxed_threshold` 可能显著影响质量-速度折中。

3. **方法边界被质疑**
   - 若实现得太像阈值 trick，novelty 会变弱。

因此第一版必须强调：

- anchor 由 O1 选出；
- local leap 只在 winner branch 上进行；
- 有显式 gate。

---

## 13. 实验与消融建议

### 13.1 第一轮小实验

建议先在：

- `gsm8k_test_only`
- `arc_challenge`

上跑小规模验证。

重点比较：

1. `CLAD v2`
2. `CANDLE (radius=2, max_local_accept=2)`
3. `CANDLE (radius=4, max_local_accept=2)`

### 13.2 关键指标

- Accuracy
- Throughput
- Correct Throughput
- `phase2_accepted_rate`
- `o2_hit_rate`
- 新增：
  - `local_leap_hit_rate`
  - `avg_local_accept_count`

### 13.3 关键消融

1. 去掉 local leap，只保留 O1 + O2
2. 用 raw confidence anchor 代替 O1 winner
3. 固定半径 vs 动态半径
4. 启用 gate vs 不启用 gate

这组消融对于投稿非常重要，因为它能回答：

- “LocalLeap 的收益是否只是阈值技巧？”
- “一致性锚定是否真的比纯置信度锚定更好？”

---

## 14. 与投稿叙事的关系

如果这条方法能跑通，它最强的叙事不是“我又加了一个工程优化”，而是：

> 现有方法要么只看概率传播（LocalLeap），要么只看一致性选点（CLAD v2）；我们的方法把“选点”和“传播”串成了一条完整链路：先用一致性/信息量找到真正值得信任的 anchor，再让局部确定性围绕这个 anchor 传播。

这比单纯继续做 `CLAD v3 / v4` 的工程堆叠更容易包装成投稿故事。

---

## 15. 当前建议

当前最推荐的下一步不是直接做完整论文包装，而是：

1. 先按本文档实现第一版 `CANDLE` prototype；
2. 跑小规模 GSM8K / ARC 验证是否存在稳定速度收益；
3. 再决定是否值得作为一条新的投稿方法线重点推进。

如果第一版实验结果显示：

- ARC / MATH 上吞吐进一步上升；
- 准确率不明显下降；
- case trace 能清晰解释 local leap 的有效性；

那么这条线非常值得继续做下去。
