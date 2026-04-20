# Chart Data Reference

本文件整理 `dlm/docs/make_charts.py` 当前用于生成图表的实验数据，便于在撰写论文、整理图注和检查正文数值时快速查阅。

数据来源：
- 主结果：`dlm/experiments/evals/` 中的 LLM-as-a-Judge 评测报告
- GSM8K 口径：`gsm8k_test_only`（官方 test 随机抽取 300 条）
- ARC：每个 benchmark 300 条
- MATH：`math500` 全量

当前三种策略顺序统一为：

```text
Baseline / CLAD v1 / CLAD v2
```

## 1. Main Results

### 1.1 Accuracy

```text
GSM8K(test-only): 82.7 / 79.3 / 77.3
ARC-Easy:         92.3 / 88.7 / 90.7
ARC-Challenge:    77.3 / 78.7 / 78.0
MATH500:          50.6 / 49.8 / 49.6
```

### 1.2 Throughput

```text
GSM8K(test-only): 5.39 / 6.65 / 6.42
ARC-Easy:         4.52 / 4.45 / 4.62
ARC-Challenge:    4.23 / 4.24 / 4.43
MATH500:          5.01 / 6.08 / 6.32
```

### 1.3 Correct Throughput

```text
GSM8K(test-only): 2.96 / 4.09 / 3.50
ARC-Easy:         3.65 / 3.57 / 3.88
ARC-Challenge:    2.49 / 3.00 / 3.14
MATH500:          1.19 / 1.36 / 1.34
```

### 1.4 Avg Generation Time

```text
GSM8K(test-only): 71.919 / 50.888 / 53.802
ARC-Easy:         81.7   / 83.5   / 78.4
ARC-Challenge:    101.4  / 99.3   / 95.9
MATH500:          180.6  / 155.7  / 147.3
```

### 1.5 Figures Using Main Results

- `fig1_accuracy_grouped.png`
- `fig2_throughput_grouped.png`
- `fig3_correct_throughput.png`
- `fig4_gen_time.png`
- `fig5_radar_gsm8k.png`
- `fig6_scatter_acc_thr.png`
- `fig7_gsm8k_arc_easy_three_strategies.png`
- `fig7_arc_challenge_math_three_strategies.png`
- `fig8_heatmap.png`
- `fig11_radar_four_bench_avg.png`
- `fig12_scatter_acc_thr_four_bench_avg.png`

## 2. MATH500 Level-5

样本数：

```text
n = 134
```

### 2.1 Accuracy

```text
24.63 / 24.63 / 24.63
```

### 2.2 Throughput

```text
4.538 / 5.592 / 5.875
```

### 2.3 Correct Throughput

```text
0.493 / 0.706 / 0.699
```

### 2.4 Avg Generation Time

```text
285.20 / 241.23 / 221.03
```

### 2.5 TPF (for plotting only)

```text
1.798 / 2.009 / 2.046
```

### 2.6 Figures Using Level-5 Data

- `fig9_math500_level5_metrics.png`
- `fig10_math500_level5_acc_cthr.png`

## 3. GSM8K Test-only Supplementary Experiments

这些数据主要用于图 13、图 14、图 15，对应“超参数敏感性”“k 敏感性”和“阶段命中率统计”。

## 3.1 CLAD v2 Sensitivity: accept_threshold2

```text
x:    0.80 / 0.85 / 0.90 / 0.95 / 1.01
acc:  78.0 / 77.7 / 77.3 / 77.0 / 78.7
thr:  6.52 / 6.48 / 6.42 / 6.54 / 6.64
cthr: 3.68 / 3.65 / 3.50 / 3.62 / 4.06
o2:   1.7  / 1.6  / 1.5  / 1.3  / 0.0
```

解释：
- `1.01` 可视为“关闭 O2”
- `o2` 为 O2 extra-accept rate（百分比）

## 3.2 CLAD v2 Sensitivity: alpha

```text
x:    0.30 / 0.50 / 0.70
acc:  78.0 / 77.3 / 77.3
thr:  6.48 / 6.42 / 6.34
cthr: 3.58 / 3.50 / 3.44
o2:   1.5  / 1.5  / 1.5
```

## 3.3 CLAD v2 Sensitivity: beta

```text
x:    0.00 / 0.20 / 0.40
acc:  77.3 / 77.3 / 77.7
thr:  6.72 / 6.42 / 6.32
cthr: 3.63 / 3.50 / 3.61
o2:   1.5  / 1.5  / 1.4
```

### 3.4 Figure Using These Data

- `fig13_gsm8k_test_only_v2_sensitivity.png`

## 4. Lookahead Branch Count k Sensitivity

### 4.1 CLAD v1

```text
k:    0 / 1 / 2 / 3
acc:  78.0 / 78.0 / 79.3 / 79.0
thr:  7.39 / 6.67 / 6.65 / 6.15
cthr: 4.22 / 3.76 / 4.09 / 3.71
```

说明：
- `k=0` 时 Phase-2 不启用，因此这里是“Phase-1 + Phase-3 only”的退化对照。

### 4.2 CLAD v2

```text
k:    0 / 1 / 2 / 3
acc:  78.0 / 77.3 / 77.3 / 77.0
thr:  7.39 / 6.67 / 6.42 / 6.18
cthr: 4.22 / 3.61 / 3.50 / 3.42
```

### 4.3 Figure Using These Data

- `fig14_gsm8k_test_only_k_sensitivity.png`

## 5. Phase-hit Breakdown and Structural Ablations

这里的口径对应图 15。

### 5.1 CLAD v1 (default)

```text
phase1: 68.9
phase2: 3.9
phase3: 27.2
o2:     0.0
acc:    79.3
thr:    6.65
cthr:   4.09
```

### 5.2 CLAD v2 (default)

```text
phase1: 68.7
phase2: 4.1
phase3: 27.2
o2:     1.5
acc:    77.3
thr:    6.42
cthr:   3.50
```

### 5.3 CLAD v2 with beta=0 and threshold2=1.01

```text
phase1: 68.9
phase2: 3.9
phase3: 27.2
o2:     0.0
acc:    79.3
thr:    6.66
cthr:   4.06
```

### 5.4 CLAD v2 with k=0

```text
phase1: 68.1
phase2: 0.0
phase3: 31.9
o2:     0.0
acc:    78.0
thr:    7.39
cthr:   4.22
```

### 5.5 Figure Using These Data

- `fig15_gsm8k_test_only_phase_breakdown.png`

## 6. ARC-Challenge Supplementary Experiments

### 6.1 CLAD v2 (default)

```text
phase1: 74.6
phase2: 8.0
phase3: 17.4
o2:     1.6
acc:    77.0
thr:    4.30
cthr:   3.02
```

### 6.2 CLAD v2 with threshold2=1.01

```text
phase1: 74.6
phase2: 8.1
phase3: 17.3
o2:     0.0
acc:    77.3
thr:    4.24
cthr:   2.91
```

### 6.3 CLAD v2 with beta=0

```text
phase1: 74.6
phase2: 8.0
phase3: 17.4
o2:     1.7
acc:    77.7
thr:    4.26
cthr:   3.00
```

### 6.4 CLAD v2 with beta=0 and threshold2=1.01

```text
phase1: 74.7
phase2: 8.0
phase3: 17.3
o2:     0.0
acc:    76.3
thr:    4.26
cthr:   2.88
```

### 6.5 CLAD v2 with k=0

```text
phase1: 74.0
phase2: 0.0
phase3: 26.0
o2:     0.0
acc:    76.3
thr:    4.83
cthr:   3.30
```

### 6.6 Figure Using These Data

- `fig16_arc_challenge_phase_breakdown.png`
