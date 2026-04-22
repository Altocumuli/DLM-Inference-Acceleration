# CALM 方法设计文档

## 0. 文档目标

本文档用于定义一条与 `CLAD v1 / v2 / CANDLE` 明显不同的新方法线。其核心出发点是：

- 不再继续堆叠更重的分支评分、局部门控和额外超参数；
- 重新回到“**轻量地并行接受多个 token**”这一真正可能显著提升推理速度的方向；
- 将多 token 接受尽量放回到 **Phase-1 一致性主路径** 中；
- 在第一版设计中，**若 Phase-1 未命中，则直接进入 fallback，不启用 Phase-2**。

换言之，`CALM` 的目标不是做一个更复杂的 `CLAD v2++`，而是提出一个：

> **以跨步一致为锚点、以局部高置信传播为扩张、并使用退火阈值控制激进程度的轻量多 token 解码策略。**

---

## 1. 命名建议

推荐方法名：

## **CALM**

全称建议：

**Consistency-Anchored Local March**

也可理解为：

**Consistency-Anchored Local Mask acceptance**

命名理由：

1. `Consistency-Anchored`：强调该方法不是从 raw confidence 出发，而是从跨步一致位置出发；
2. `Local`：强调传播只发生在 anchor 邻域；
3. `March`：强调解码过程中沿着局部稳定区域逐步向前推进。

这个名字的优点是：

- 简短、容易记忆；
- 与方法气质一致：轻量、稳定、逐步推进；
- 比 `v3 / v4` 更适合作为一个可投稿的方法名。

---

## 2. 方法动机

## 2.1 当前路径的问题

从 `CLAD v2 -> CANDLE -> CANDLE-G` 的探索中，我们已经观察到一个越来越明显的问题：

1. **方法越来越重**
   - 更多分支评分
   - 更多局部 gate
   - 更多超参数
   - 更复杂的行为解释

2. **速度收益并不稳定**
   - 为了局部更稳，计算负担变重；
   - 即便局部传播被触发，也不一定能转化为更高的吞吐或正确吞吐。

3. **实现方向逐渐偏离真正高效的加速思路**
   - CCD 和 LocalLeap 的优势都来自：  
     **用一次 forward 支持多个 token 的接受**
   - 而不是来自越来越复杂的多层评分和门控。

这说明，如果继续沿着“在重型 Phase-2 后追加更多局部逻辑”的方向走，可能会出现：

- novelty 似乎更多了，
- 但方法越来越不够轻，
- 也越来越不像真正能把推理加速做强的路线。

## 2.2 一个更自然的问题

当前我们已经知道：

- **跨步一致** 本身就意味着：
  1. 当前预测较稳定；
  2. 这种稳定不是单步偶然，而是跨步重复出现；
  3. 从某种意义上，它天然包含了“高置信 + 一致性”的双重信息。

那么一个更自然的问题是：

> 如果某个位置已经满足跨步一致，它是否可以作为一个比“纯高置信 token”更可信的 anchor？  
> 如果可以，能否围绕这些一致性 anchor，直接在同一步内接受邻域中的其他高置信 token？

这就是 `CALM` 的出发点。

## 2.3 与 `LocalLeap` 的关系

`LocalLeap` 的启发在于：

- 一个高置信 anchor 往往意味着其邻域也更容易确定；
- 因此可以在 anchor 邻域中使用更宽松的阈值，一次接受多个 token。

但 `CALM` 不直接复现 `LocalLeap`，因为：

- `LocalLeap` 的 anchor 是 **高置信 anchor**
- `CALM` 的 anchor 是 **跨步一致 anchor**

因此更准确地说：

> `CALM` 是将 `LocalLeap` 的“局部传播”思想，建立在 `CCD / CLAD v1` 的“一致性锚点”之上。

这使它天然比单纯的置信度传播更稳，也更契合你当前工作的一致性主线。

---

## 3. 核心思想

## 3.1 一句话概括

**当某些 masked 位置已经满足跨步一致时，将它们视为一致性锚点，并在同一步内接受这些锚点邻域中的高置信 token；随着迭代推进，逐步降低邻域接受阈值。**

## 3.2 与现有方法的区别

### Baseline

- 仅根据当前置信度阈值接受 token；
- 不利用跨步信息；
- 不做局部传播。

### CCD

- 只接受满足跨步一致的位置；
- 一致性位置本身可多 token 接受；
- 但不向邻域传播。

### LocalLeap

- 从高置信 anchor 向邻域传播；
- 不依赖跨步一致。

### CALM

- 从 **一致性 anchor** 向邻域传播；
- 邻域接受规则轻量；
- 阈值随推理步逐步退火；
- 第一版中，若 Phase-1 未命中，则直接 fallback，不启用 Phase-2。

这意味着 `CALM` 的逻辑可以概括成：

> **Consistency first, then local confidence propagation.**

---

## 4. 方法流程

## 4.1 总体结构

`CALM` 第一版只保留两个主要阶段：

1. **Phase-1：Consistency-Anchored Local Acceptance**
2. **Phase-3：Fallback Threshold Fill**

也就是说：

- **暂不启用 Phase-2**
- 若一致性通道未命中，则直接进入 fallback

这样做的原因是：

1. 简化方法结构，避免方法叙事再次变重；
2. 先把“轻量一致性 + 邻域扩张”的主假设单独验证清楚；
3. 便于与 `CCD / CLAD v1 / LocalLeap` 的核心机制对照。

## 4.2 Phase-1：一致性锚点识别

与当前 `CLAD v1 / v2` 的一致性快通道基本一致：

- 维护上一步 top-V 高置信位置及其 argmax token；
- 当前步再次计算 top-V；
- 若某位置同时满足：
  1. 属于当前步 top-V；
  2. 属于上一步 top-V；
  3. argmax token 跨步一致；

则该位置被视为 **consistency anchor**。

记 anchor 集合为：

$$
A = \{p \mid p \in \text{top-V}_{t-1} \cap \text{top-V}_t,\ \hat{y}_{t-1}(p)=\hat{y}_t(p)\}
$$

## 4.3 Phase-1 增强：锚点邻域扩张

对于每个一致性锚点 $p \in A$，定义其邻域：

$$
N_r(p) = \{i \mid |i-p|\le r,\ i \in \text{current masked positions}\}
$$

在第一版中：

- 仅使用当前 step 的 logits；
- 不额外 forward；
- 不构造分支；
- 只检查邻域内位置的最大概率。

若邻域位置 $i$ 满足：

$$
\max_y p(y \mid x_t, i) \ge \tau_{\text{local}}(t)
$$

则也直接接受其 argmax token。

也就是说，在一次 forward 内：

- 先接受一致性 anchor 本身；
- 再接受其邻域中的部分高置信 token。

## 4.4 阈值退火

`CALM` 不建议使用固定局部阈值，而是让阈值随着迭代推进逐步下降。

直觉如下：

- 早期步骤：模型还不够确定，应更保守；
- 后期步骤：剩余 token 更少，分布更尖锐，可以更积极接受。

因此定义：

$$
\tau_{\text{local}}(t)
$$

为随推理步 $t$ 退火的函数。

### 线性退火版本

$$
\tau_{\text{local}}(t)=\tau_{\text{start}}-(\tau_{\text{start}}-\tau_{\text{end}})\cdot \frac{t}{T}
$$

其中：

- `tau_start`：初始局部接受阈值
- `tau_end`：末期局部接受阈值
- `T`：该 block 的总迭代预算

### 幂次退火版本

$$
\tau_{\text{local}}(t)=\tau_{\text{end}}+(\tau_{\text{start}}-\tau_{\text{end}})\cdot \left(1-\frac{t}{T}\right)^\gamma
$$

其中：

- `gamma > 1`：更慢下降
- `gamma < 1`：更快下降

第一版建议先用线性退火，便于解释和调参。

## 4.5 Fallback

如果当前步没有任何一致性锚点，或者虽然存在 anchor 但邻域传播没有额外接受任何 token，则：

- 直接进入当前已有的 fallback 阈值填充；
- 不启用任何 Phase-2 分支评分逻辑。

这意味着：

> `CALM` 第一版是一个明确的“Phase-1 增强方法”，而不是多阶段复杂方法。

---

## 5. 方法伪代码

```text
for each decoding iteration:
    compute current logits
    compute current top-V positions and argmax tokens

    anchors = positions that are:
        in previous top-V
        in current top-V
        and have the same argmax token across steps

    if anchors not empty:
        accept all anchors

        for each anchor p:
            define local neighborhood N_r(p)
            for each masked position i in N_r(p):
                if max_prob(i) >= tau_local(t):
                    accept token at i

    else:
        use fallback threshold fill

    update history buffer
```

---

## 6. 关键设计选择

## 6.1 为什么不启用 Phase-2

第一版明确不启用 Phase-2，原因如下：

1. 目标是验证“**一致性 anchor + 邻域扩张**”这一轻量主假设；
2. 若同时保留重型 Phase-2，就很难判断收益究竟来自哪里；
3. 这条线的优势本来就应该是：
   - 少计算
   - 少分支
   - 少超参

如果一上来又把 Phase-2 加回来，就容易重新滑回 `CANDLE / CANDLE-G` 那种越来越重的路径。

## 6.2 为什么 anchor 选一致性而不是高置信

这是 `CALM` 最关键的 novelty 之一。

高置信 anchor 表示：

- 当前步模型相信这个位置

一致性 anchor 则表示：

- 当前步模型相信这个位置；
- 并且上一步模型也相信这个位置；
- 而且两步给出的 argmax token 完全相同。

所以一致性 anchor 天然比单步高置信更可信，尤其适合：

- 数学题
- 推理题
- 任何对局部事实稳定性要求较高的任务

## 6.3 为什么做退火

如果局部接受阈值始终固定，就会遇到两难：

- 阈值高：早期安全，但后期太保守
- 阈值低：后期有效，但早期太激进

退火阈值提供了一个自然的折中：

- 早期先保守
- 后期再逐步放开

这与扩散解码本身“越往后越确定”的直觉是一致的。

---

## 7. 推荐配置项

建议定义如下配置：

```python
@dataclass
class CalmConfig:
    top_v: int = 4
    neighbor_radius: int = 1
    max_neighbor_accept_per_anchor: int = 1

    local_threshold_start: float = 0.90
    local_threshold_end: float = 0.72
    local_threshold_gamma: float = 1.0

    # inherited baseline / llada parameters
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
```

### 参数解释

- `top_v`  
  一致性快通道中保留的 top-V 高置信位置数。

- `neighbor_radius`  
  一致性锚点邻域半径。

- `max_neighbor_accept_per_anchor`  
  每个 anchor 最多额外带动几个邻域 token。

- `local_threshold_start / end`  
  局部接受阈值的起点和终点。

- `local_threshold_gamma`  
  若使用幂次退火时控制曲线形状。

---

## 8. Trace 记录建议

由于 `CALM` 的核心是“一致性 anchor 是否真的带动了局部传播”，因此建议记录：

```json
{
  "phase": "phase1",
  "anchor_positions": [...],
  "anchor_tokens": [...],
  "local_threshold": 0.84,
  "anchor_neighborhoods": [
    {
      "anchor_pos": 12,
      "accepted_neighbor_positions": [13, 14],
      "accepted_neighbor_probs": [0.88, 0.82]
    }
  ]
}
```

如果某次迭代没有一致性命中，则记录：

```json
{
  "phase": "phase3",
  "reason": "no_consistency_anchor"
}
```

这样后面做个案分析时，你就能直接解释：

- 哪些位置是 consistency anchor
- 哪些邻域 token 被一起接受
- 阈值在不同迭代步是如何退火的

---

## 9. 与现有方法的关系

## 9.1 与 CCD 的关系

`CALM` 可以理解为：

> **CCD 的一致性接受 + LocalLeap 的局部传播**

但 anchor 仍来自一致性，而不是置信度。

## 9.2 与 LocalLeap 的关系

`CALM` 吸收的不是 LocalLeap 的全部实现细节，而是其关键思想：

- 局部确定性可以从 anchor 向邻域传播。

区别在于：

- LocalLeap：高置信 anchor
- CALM：一致性 anchor

## 9.3 与 CLAD 的关系

`CALM` 不是 `CLAD v3/v4`，而是一条不同方法线：

- `CLAD`：重心在 lookahead branch selection
- `CALM`：重心在 consistency-anchored local multi-token acceptance

所以它更适合作为**并行路线**，而不是 `CLAD` 的简单版本迭代。

---

## 10. 预期收益

### 10.1 理论收益

1. **更轻量**
   - 无 branch expansion
   - 无 branch scoring
   - 无额外 forward

2. **更接近真正的吞吐提升逻辑**
   - 一次 forward
   - 一次性接受多个 token

3. **更少超参**
   - 核心只剩邻域半径、每锚点接受上限、退火阈值

### 10.2 可能风险

1. 一致性 anchor 太少，导致 local expansion 触发率不高
2. 邻域扩张若太激进，仍可能在数学题里放大错误
3. 若 `top_v` 太小，anchor 不够；太大，则一致性命中可能变噪

---

## 11. 第一轮实验建议

建议第一轮只做小规模 smoke test，不直接全量。

### 建议对照

在：

- `gsm8k_test_only`
- `arc_challenge`

上比较：

1. `clad_v1`
2. `clad_v2`
3. `calm`

这样可以直接回答：

- `CALM` 相比 `CCD/CLAD v1` 的一致性系方法是否更强；
- 相比 `CLAD v2` 这种重型方法，是否能用更轻计算取得更好的吞吐-精度折中。

### 建议初始配置

#### GSM8K

- `top_v = 4`
- `neighbor_radius = 1`
- `max_neighbor_accept_per_anchor = 1`
- `tau_start = 0.90`
- `tau_end = 0.74`

#### ARC

- `top_v = 4`
- `neighbor_radius = 2`
- `max_neighbor_accept_per_anchor = 2`
- `tau_start = 0.90`
- `tau_end = 0.76`

---

## 12. 当前建议

如果你要推进这条新线，我建议下一步顺序是：

1. 先按本文档写一个独立脚本 prototype，例如：
   - `llada_calm_decode.py`
2. 保持方法极简，不接入 Phase-2；
3. 跑小规模 GSM8K / ARC 验证：
   - 一致性 anchor 是否足够多
   - 邻域接受是否真的提高吞吐
   - 退火阈值是否比固定阈值更稳

如果第一轮结果显示：

- accuracy 维持住
- throughput / correct throughput 有明显提升
- trace 中确实出现“一致性 anchor -> 邻域扩张”的清晰模式

那么 `CALM` 将会是一条比当前 `CANDLE / CANDLE-G` 更值得继续推进的方向。
