# 解码策略流程图（Mermaid）

本文用 Mermaid 图概括 **Baseline（LLaDA2.1-mini 原生 `generate`）** 与 **CLAD v1 / v2 / v3** 在「单块内一步迭代」层面的控制流差异。  
渲染：GitHub / VS Code Markdown 预览 / Typora 等支持 Mermaid 的编辑器均可显示。

---

## 1. Baseline（原生 `generate`，块内迭代）

与 `design.md` §1.3 一致：每轮一次 forward，再解掩码与编辑。

```mermaid
flowchart TD
    A[块内 while 循环开始] --> B[构造 cur_x，截断 attention 到 current_window_end]
    B --> C[Forward → logits]
    C --> D[采样：每位置 argmax + 概率]
    D --> E{块内仍有 mask?}
    E -->|否| F[post_steps++；超过 max_post_steps 则退出块]
    E -->|是| G[解掩码：mask 上概率 > threshold 的位置接受]
    G --> H[若仍不足 num_to_transfer：按置信度 top 填 1 个]
    H --> I[编辑：非 mask 且非 prompt，新 token 概率 > editing_threshold 则覆盖]
    I --> J{本步有解掩码或编辑?}
    J -->|否| K[退出当前块]
    J -->|是| A
    F --> L[下一块 / 或 eos_early_stop 结束]
    K --> L
```

**外层**：按块顺序遍历；块间块对角因果 mask。每块结束后可检查 EOS 提前结束整段生成。

---

## 2. CLAD v1（`llada_clad_decode.py`）

在 Baseline 的「单轮」中插入：**阶段一** 双步一致性快通道 → **阶段二** 一致性传播前瞻 → **阶段三** 阈值填充；编辑与 buffer 更新与原生一致。

```mermaid
flowchart TD
    S[块内迭代一步：开始] --> F1[Forward → logits, tokens, probs]
    F1 --> P1{阶段一：双步 top-V 一致性?}
    P1 -->|是，有稳定位置| ACC1[批量写入一致位置]
    ACC1 --> EDIT[编辑非 mask 位置]
    P1 -->|否| W{迭代次数 ≥ lookahead_warmup 且 num_lookahead>0?}
    W -->|否| P3[阶段三：threshold 解掩码 / 单点最优]
    W -->|是| P2[阶段二：对 top-k 候选位置各建分支]
    P2 --> L2[每分支 Forward，算 v1 综合分 score]
    L2 --> PICK[取最高分分支，写入 1 个 token]
    PICK --> EDIT
    P3 --> EDIT
    EDIT --> BUF[更新 history buffer]
    BUF --> S
```

**说明**：`consistency_prop` 为「填入后剩余 mask 上新 argmax 与填前一致」的比例（v1 为**均匀**权重）。

---

## 3. CLAD v2（`llada_clad_v2_decode.py` 设计 / 与 v3 共用 O1·O2 公式）

在 v1 骨架上，**阶段二** 分支评分改为 **O1** 三项加权；阶段二结束后增加 **O2**（复用最优分支 logits，可能再接受 1 个 token）。

```mermaid
flowchart TD
    S[块内迭代一步] --> F1[Forward]
    F1 --> P1{阶段一：双步一致性?}
    P1 -->|是| ACC1[写入一致位置] --> EDIT
    P1 -->|否| W{预热完成且可前瞻?}
    W -->|否| P3[阶段三：阈值填充]
    W -->|是| P2[阶段二：top-k 分支各 Forward]
    P2 --> O1[O1 评分：加权一致 / 熵降 / future_conf 三项]
    O1 --> BEST[选最优分支，写入第 1 个 token]
    BEST --> O2{剩余 mask 上最大置信度 ≥ accept_threshold2?}
    O2 -->|是| T2[再写入第 2 个 token，无额外 Forward]
    O2 -->|否| EDIT
    T2 --> EDIT
    P3 --> EDIT
    EDIT --> BUF[更新 buffer]
```

**说明**：若仓库中 v2 实现与 v3 共用同一套 `_clad_branch_score_v2` / O2，则上图与 v3 的「单分支串行 Phase-2」等价；v3 主要把 Phase-2 的多分支 **Forward 合并为 batch（O3）**。

---

## 4. CLAD v3（`llada_clad_v3_decode.py`）

继承 v2 的 **O1 / O2**；**阶段二** 默认用 **O3 批量 forward**，并用 **O4** 约束 Level-1 仅比较 top-2 位置（或关闭 O4 时比较 top-k）。

```mermaid
flowchart TD
    S[块内迭代一步] --> F1[Forward 主步]
    F1 --> P1{阶段一：双步一致性?}
    P1 -->|是| ACC1[写入] --> EDIT
    P1 -->|否| W{预热完成且可前瞻?}
    W -->|否| P3[阶段三：阈值填充] --> EDIT
    W -->|是| C{use_batched_phase2?}
    C -->|否| SEQ[逐分支 Forward，O1 选优 + O2]
    C -->|是| O4{use_cascaded_draft?}
    O4 -->|是| L1[O4：取置信 top-2 位置，拼 batch B=2]
    O4 -->|否| L1b[Flat：取 top-k，拼 batch B=k]
    L1 --> BF[O3：一次 batched Forward → logits 0..B-1]
    L1b --> BF
    BF --> SC[对每行算 O1 分，选最优行]
    SC --> W1[写入 Level-1 最优 token]
    W1 --> O2{O2：胜者行 logits 上二次接受?}
    O2 -->|是| T2[可选再写 1 token]
    O2 -->|否| EDIT
    T2 --> EDIT
    SEQ --> EDIT
    EDIT --> BUF
```

---

## 5. 对照：四种策略「单步迭代」前向次数（示意）

| 策略 | 典型每迭代主 forward | Phase-2 额外 forward（示意） |
|------|----------------------|------------------------------|
| Baseline | 1 | 0 |
| CLAD v1 | 1 | 0～k（串行 k 次分支） |
| CLAD v2 | 1 | 同 v1（实现为串行时） |
| CLAD v3 | 1 | **0～1 次 batched**（替代 k 次串行） |

实际次数取决于是否命中阶段一、mask 数量及 `use_batched_phase2` 等；上表仅表达相对关系。

---

## 6. 源码索引

| 策略 | 主要文件 |
|------|----------|
| Baseline | `modeling_llada2_moe.py` → `generate()` |
| CLAD v1 | `dlm/src/llada_clad_decode.py` |
| CLAD v2 | `dlm/src/llada_clad_v2_decode.py` |
| CLAD v3 | `dlm/src/llada_clad_v3_decode.py` |

更细的设计说明见 `dlm/docs/design.md`（§1.3、§1.9–§1.11）。
