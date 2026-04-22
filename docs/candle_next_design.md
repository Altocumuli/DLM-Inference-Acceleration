# 下一版 CANDLE 改进方案

## 0. 文档目标

本文档在 `CANDLE` 第一版实验结果基础上，提出下一版方法的改进方向。

当前实验已经表明：

- `CANDLE` 的 local leap 机制不是无效设计；
- 第二轮调参已经体现出明显的任务依赖性；
- 但 `local_leap_hit_rate` 仍偏低，且收益不够稳定；
- 当前瓶颈不再是“是否能传播”，而是“**什么时候值得传播**”。

因此，下一版的改进重点不应放在继续粗暴放宽半径或阈值，而应聚焦于：

> **更聪明的 anchor-conditioned propagation gate**

---

## 1. 当前版本的主要问题

## 1.1 local leap 触发率仍偏低

当前最优配置下，local leap 指标大致为：

- GSM8K：`1.4% / 2.3%`
- ARC：`1.7% / 3.0%`

这说明：

- 机制已经活了；
- 但还没真正进入“可以稳定改写整体行为”的区间。

## 1.2 gate 仍过于粗糙

当前 gate 主要依赖：

- `winner_score`
- `weighted_consistency`
- `entropy_reduction`

而且逻辑比较简单：

- 分数不过阈值就不传播；
- 否则允许整个邻域尝试传播。

这种 gate 有两个问题：

1. **对局部传播风险估计不够细**
   - 只评估 winner branch 本身是否“足够好”
   - 没有评估“邻域 token 本身是否值得一起传播”

2. **无法区分任务结构**
   - GSM8K 和 ARC 的最佳行为明显不同
   - 说明 gate 需要更细粒度地适应局部结构，而不是只看 anchor 整体好坏

## 1.3 当前 local leap 候选选择过于单一

现在 local leap 的接受规则主要是：

- 在 anchor 邻域里按 max probability 排序
- 取 top-m
- 只看是否过 `local_relaxed_threshold`

这存在一个问题：

- 一个 token 局部置信度高，不代表它真的应该和 anchor 一起被接受；
- 特别是在数学题里，局部高置信 token 可能只是“形式上合理”，但并未稳住全局事实。

---

## 2. 下一版的核心思路

下一版建议可以暂时称为：

## **CANDLE-G**

含义：

**CANDLE with Gated Local Propagation**

它的核心不是再发明一套全新的大框架，而是在当前 `CANDLE` 上加强：

- **local gate**
- **local candidate scoring**
- **task-sensitive propagation behavior**

一句话概括：

> 当前版 `CANDLE` 只回答了“winner anchor 是否值得传播”；下一版要进一步回答“anchor 邻域中的哪个 token 值得传播，以及要传播到什么程度”。 

---

## 3. 改进方向一：局部候选再评分

## 3.1 当前问题

现在 local leap 候选只按：

$$
\max_y p(y \mid B^\*, i)
$$

排序。

这太接近 `LocalLeap` 的原始置信度传播逻辑，会削弱本工作的一致性/信息量主线。

## 3.2 改进建议

对邻域内每个候选位置 $i$，引入一个 **local propagation score**：

$$
\text{local\_score}(i) =
\lambda_1 \cdot \text{local\_confidence}(i)
+ \lambda_2 \cdot \text{local\_consistency}(i)
+ \lambda_3 \cdot \text{local\_entropy\_gain}(i)
$$

其中：

### `local_confidence(i)`

就是当前 branch logits 在位置 $i$ 的最大概率。

### `local_consistency(i)`

可定义为：

- 该位置在当前 step 与 winner branch 下一步预测之间是否稳定；
- 或者更简单地，邻域内位置与当前 `cur_tokens` 的一致程度。

### `local_entropy_gain(i)`

利用当前已有 logits，不再做新 forward，仅估计：

- 如果该位置属于高信息位置且 branch 下置信更高，
- 则说明它更值得被一起接受。

在第一版近似实现里，不需要真的为每个 local candidate 再 forward，只需构造一个 proxy score 即可。

## 3.3 预期收益

这样做的好处是：

1. local leap 不再仅仅是“高置信邻域扩张”
2. 传播对象本身也被纳入一致性/信息量叙事
3. 对投稿而言，这比单纯阈值技巧更有方法味道

---

## 4. 改进方向二：分层 gate

## 4.1 当前 gate 只有一层

当前是：

- 先判断 winner branch 是否足够好；
- 若好，就让邻域整体尝试传播。

## 4.2 改成两层 gate

建议改为：

### Gate A：Anchor Gate

判断 winner branch 是否值得传播。

可继续依赖：

- `winner_score`
- `weighted_consistency`
- `entropy_reduction`

### Gate B：Local Gate

对每个邻域候选位置单独判断：

- 是否满足局部置信
- 是否满足局部一致性
- 是否属于高信息或正熵增益位置

只有同时通过两层 gate 的 token 才允许被 local leap 接受。

## 4.3 预期收益

这种设计更适合解释：

- 为什么 ARC 上可以多接受一些 token
- 为什么 GSM8K 上同样的邻域不应该被全放开

---

## 5. 改进方向三：方向性感染，而不是对称半径

## 5.1 当前是对称半径

当前邻域定义：

$$
N_r(p^\*) = [p^\*-r, p^\*+r]
$$

这是一个完全对称的局部窗口。

## 5.2 问题

在语言和推理中，anchor 周围的依赖往往不是完全对称的：

- 有时更偏向 anchor 右侧（后续生成内容）
- 有时左侧更稳定但右侧更危险

## 5.3 改进建议

引入方向性感染：

- `left_radius`
- `right_radius`

甚至可以先做一个很简单的规则：

- 默认只传播 anchor 右侧

原因是：

- 在多数生成场景中，anchor 右侧更像“尚未完全确定的后续延展”
- 左侧位置若仍 masked，往往意味着全局状态还不够稳

## 5.4 简化版建议

第一版改进不必直接暴露左右半径，可以先只做：

- `propagate_right_only = True`

如果它对 GSM8K 有帮助，说明方向性约束值得继续深挖。

---

## 6. 改进方向四：任务自适应 gate

## 6.1 当前实验已经显示任务依赖性

### GSM8K 更喜欢

- 小半径
- 更低 local threshold
- 少接受 token

### ARC 更喜欢

- 一次接受更多 token

## 6.2 问题

如果仍坚持一套统一 gate，很可能会：

- 在 GSM8K 上太激进
- 在 ARC 上又不够充分

## 6.3 改进建议

不一定要显式做 task-specific branching，但可以引入一种“隐式任务自适应”的 gate proxy，例如：

### 方案 A：基于输出风格判断

如果当前 block 内：

- 数字 token 密度更高
- 算式结构更密集

则使用更保守 gate。

### 方案 B：基于 entropy profile 判断

如果邻域内高熵位置很多：

- 说明局部还不够稳定
- 应更保守

如果邻域内熵整体较低：

- 说明这个局部更像一个稳定语义块
- 可更积极传播

## 6.4 建议先不做显式 task switch

下一版仍建议保持统一方法，不显式写：

- GSM8K 用一套
- ARC 用一套

而是把这种任务差异吸收到 gate 本身里，这样更适合论文叙事。

---

## 7. 具体的下一版最小实现建议

为了避免一次改动过大，下一版建议只做这三个最关键改动：

### 改动 1：local candidate score

用 `local_score` 替代单纯 `prob` 排序。

### 改动 2：两层 gate

- `anchor gate`
- `local token gate`

### 改动 3：右侧传播优先

先尝试 `propagate_right_only=True`

---

## 8. 推荐新增配置项

下一版建议新增这些参数：

```python
local_score_weight_conf: float = 0.6
local_score_weight_cons: float = 0.3
local_score_weight_ent: float = 0.1

local_token_consistency_floor: float = 0.5
local_token_entropy_floor: float = 0.0

propagate_right_only: bool = True
```

如果想更细一点，还可以预留：

```python
left_radius: int = 0
right_radius: int = 2
```

---

## 9. 建议记录的新 trace 字段

为支持个案分析，下一版建议额外记录：

```json
{
  "local_gate_passed": true,
  "local_gate_reason": "passed",
  "local_candidates": [
    {
      "pos": 12,
      "prob": 0.91,
      "local_score": 0.74,
      "local_consistency": 0.66,
      "local_entropy_gain": 0.01,
      "accepted": true
    }
  ],
  "propagate_right_only": true
}
```

这样后续你才能真正解释：

- 为什么同样是 local leap，GSM8K 上更保守，ARC 上更激进；
- 为什么某个 token 被传播，而另一个没有。

---

## 10. 下一轮实验建议

如果实现了下一版 `CANDLE-G`，建议不要立刻 full-run，而是先做：

### GSM8K

- `candle_k3_tl070` 作为旧 best baseline
- `candle-g_k3_tl070`

### ARC

- `candle_k3_m3` 作为旧 best baseline
- `candle-g_k3_m3`

重点看：

- accuracy
- correct throughput
- local_leap_hit_rate
- local_leap_token_rate

以及新增的：

- accepted local candidates per trigger
- right-side vs left-side accepted distribution

---

## 11. 当前建议

当前最推荐的路线不是继续做大规模 brute-force 调参，而是：

1. 先把本文件中的三项关键改进落成 prototype；
2. 在 GSM8K/ARC 上分别与当前 best CANDLE config 对照；
3. 如果结果显示：
   - GSM8K 更稳
   - ARC 的 local leap 更有效

那么这条线就更像一条可以往投稿方向推进的方法迭代。
