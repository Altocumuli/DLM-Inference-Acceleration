## 扩散语言模型实验环境说明（LLaDA2.1-mini 为主）

本文件记录在这台 AMD 服务器上，为扩散语言模型（特别是 `inclusionAI/LLaDA2.1-mini`）搭建实验环境的**完整过程和依赖版本**。后续如果环境坏了或要在别的机器上复现，可以直接参考这里一步步重来。

> 当前检测结果（2026-03-13）：
> - 系统自带：Python 3.10.12（`/usr/bin/python3`）
> - 已安装：ROCm 工具（`rocminfo` 可用，说明 AMD GPU 相关驱动/库已配置好）
> - 未安装：`conda` / `mamba`（命令不存在）
> - 已安装：`uv`（例如 `/home/qinghua/.local/bin/uv`）

---

### 1. 环境搭建整体思路（使用 uv）

- **目标**：为 LLaDA2.1-mini 和相关工具准备一个**干净、专用**且便于复现的 Python 虚拟环境，用于：
  - 加载和推理扩散语言模型（dLLM），如 `LLaDA2.1-mini`（参考模型卡说明：`[inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini)`）。
  - 后续可能的解码机制研究（使用 dInfer 框架进行多种解码算法实验：`[inclusionAI/dInfer](https://github.com/inclusionAI/dInfer)`）。
  - 如需要微调，则使用 dFactory 进行 dLLM 微调实验：`[inclusionAI/dFactory](https://github.com/inclusionAI/dFactory)`。
- **方案选择**：
  - 课题组师兄推荐使用 **uv**，本项目统一使用 `uv` 创建和管理虚拟环境（在项目根目录创建 `.venv/`），所有依赖安装在其中。
  - 与传统的 `venv + pip` 相比，`uv` 依赖解析和下载更快，命令统一，便于复现。

### 2. 使用 uv 创建本地虚拟环境（.venv）

```bash
cd /home/qinghua/yangjh
uv venv
source .venv/bin/activate
```

激活成功后命令行前会出现 `(.venv)` 或类似前缀。退出虚拟环境：`deactivate`。

---

### 3. 安装基础依赖

在虚拟环境激活状态下执行：

```bash
uv pip install --upgrade pip
uv pip install "torch==2.4.0" "transformers==4.57.1" "accelerate" "datasets" "huggingface_hub" "sentencepiece"
```

**重要版本说明**：
- **transformers==4.57.1**：社区推荐用于 LLaDA2.1-mini 推理的版本，其他版本可能导致生成异常。
- **torch==2.4.0**：需要安装 ROCm 版（见第 7 节），NVIDIA CUDA 版在 AMD GPU 上不生效。

---

### 4. LLaDA2.1-mini 最小验证脚本

- 脚本位置：`dlm/src/test_llada2_1_mini.py`
- 功能：设置 HF 缓存路径；注册 RoPE `"default"` 以兼容当前 transformers；加载 `inclusionAI/LLaDA2.1-mini` 并做一次简单生成。
- 运行：`python dlm/src/test_llada2_1_mini.py`

**与官方用法的对齐**：
- `device_map="auto"`，`model.to(torch.bfloat16)`
- `apply_chat_template` 后统一取出 `input_ids` 并保证 `(1, seq_len)`、LongTensor、放到正确 device
- `generate(..., eos_early_stop=True, gen_length=512, block_length=32, threshold=0.5, editing_threshold=0, temperature=0.0, max_post_steps=16)`

---

### 4.5 Transformers 与 SGLang 调用方式的区别

| 维度 | 当前方式（Hugging Face Transformers） | 官方 SGLang 推荐 |
|------|--------------------------------------|-----------------|
| **用途** | 本地脚本、单次推理、调试、毕设实验 | 部署服务、高并发、批量请求、生产推理 |
| **加载方式** | `AutoModelForCausalLM.from_pretrained(...)` | `sglang.launch_server` 拉起独立服务 |
| **推理接口** | 直接调用 `model.generate(inputs=input_ids, ...)` | 客户端发请求到服务端 |
| **性能** | 单进程、单 batch，适合单条或小批量 | 静态/动态 batching、KV-cache 复用，吞吐更高 |

**结论**：当前用 Transformers 是为了**先把流程跑通、做解码机制实验和写毕设**；若后续要做高并发或线上服务，再按官方文档部署 SGLang。

---

### 5. dInfer 与 dFactory 在毕设中的定位

- **dInfer**：[inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) —— 扩散语言模型高效推理与多种解码算法，适合做「解码机制」对比实验。
- **dFactory**：[inclusionAI/dFactory](https://github.com/inclusionAI/dFactory) —— dLLM 微调工具，偏训练侧；若毕设侧重解码而非训练，可暂不安装。

---

### 6. Hugging Face 模型缓存目录

- **缓存根目录**：`/etc/moreh/checkpoint/huggingface_hub`
- 在脚本开头（在 `import transformers` 之前）通过环境变量指定：
  ```python
  import os
  os.environ["HUGGINGFACE_HUB_CACHE"] = "/etc/moreh/checkpoint/huggingface_hub"
  ```
  或在运行前：`export HUGGINGFACE_HUB_CACHE=/etc/moreh/checkpoint/huggingface_hub`

---

### 7. 使用 AMD GPU（ROCm 版 PyTorch）

当前脚本已改为**优先使用 GPU**（`device = "cuda" if torch.cuda.is_available() else "cpu"`）。在 AMD 机器上，PyTorch 通过 **ROCm** 仍使用 `"cuda"` 作为设备名，因此代码无需改。

**重要**：`uv pip install torch` 默认装的是 **NVIDIA CUDA 版**，在 AMD GPU 上不会生效。要跑在 AMD 上，需要改装 **ROCm 版 PyTorch**：

1. **查看本机 ROCm 版本**：
   ```bash
   rocminfo | head -5
   ```
2. **重装 torch（ROCm）**（本机已确认 ROCm 6.14，用 rocm6.0 索引即可）：
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```
3. **验证**：
   ```python
   import torch
   print(torch.cuda.is_available())   # 应为 True
   print(torch.cuda.device_count())   # 应 >= 1
   ```

---

#### 7.1 多人共用 GPU 时的使用建议

**查询当前 GPU 使用情况**（建议每次用 CUDA 跑推理前先看一眼）：
```bash
rocm-smi
# 或持续刷新观察:
watch -n 1 rocm-smi
```

主要关注：
- `VRAM%`：60% 以上通常说明已载入较大模型或 batch
- `GPU%`：接近 100% 表示高负载，不适合再新开大模型作业

**当前脚本的用卡方式**：
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

如需只用某一张卡（稳定、简单）：
```bash
export HIP_VISIBLE_DEVICES=5   # 例如只用物理 5 号卡
unset DLM_DATA_PARALLEL

# 单个 benchmark
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --max_examples 10

# 多个 benchmark（推荐：避免重复加载模型）
python dlm/src/run_benchmark_llada2.py --benchmarks gsm8k_small aime2025_all --max_examples 10
```

---

#### 7.2.1 tmux 多窗口绑卡并行（推荐多任务场景）

利用 tmux 多窗口实现"每窗口绑定不同 GPU"的并行方式：

**操作步骤**：
1. 使用 `rocm-smi` 确认有空闲 GPU
2. 开启 tmux 并创建多个窗口/面板
3. 每个窗口设置不同的 `HIP_VISIBLE_DEVICES`
4. 同时运行多个不同的 benchmark/decode_mode

**示例：4 张卡并行运行不同任务**：
```bash
# 窗口 1：GPU4 跑 baseline
tmux new-session -d -s "gpu-parallel"
export HIP_VISIBLE_DEVICES=4
cd /home/qinghua/yangjh && source .venv/bin/activate
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode baseline --max_examples 100

# 窗口 2：GPU5 跑 LoPA
tmux new-window -t "gpu-parallel"
export HIP_VISIBLE_DEVICES=5
cd /home/qinghua/yangjh && source .venv/bin/activate
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode lopa --max_examples 100

# 窗口 3：GPU6 跑 CCD
tmux new-window -t "gpu-parallel"
export HIP_VISIBLE_DEVICES=6
cd /home/qinghua/yangjh && source .venv/bin/activate
python dlm/src/run_benchmark_llada2.py --benchmark aime2025_all --decode_mode ccd

# 窗口 4：GPU7 跑代码基准测试
tmux new-window -t "gpu-parallel"
export HIP_VISIBLE_DEVICES=7
cd /home/qinghua/yangjh && source .venv/bin/activate
python dlm/src/run_benchmark_llada2.py --benchmark humaneval_all --decode_mode baseline --max_examples 50
```

**监控技巧**：
- 开一个额外窗口运行 `watch -n 1 rocm-smi` 实时监控各卡使用状态
- 使用 `tmux list-sessions` 和 `tmux attach -t gpu-parallel` 管理会话
- 各窗口的日志文件会自动分别保存在 `dlm/log/` 下

---

#### 7.2.2 多进程并行版本（高吞吐量模式）

针对 LLaDA2.1-mini 不支持批量推理的限制，另外提供了基于多进程的并行方案 `run_benchmark_llada2_multiprocess.py`：
- **工作原理**：每个 GPU 启动独立的 Python 进程，各进程加载独立的模型副本，处理分配给它的样本子集。
- **适用场景**：大规模 benchmark 实验，需要高吞吐量时使用（例如处理数百个样本）。
- **资源需求**：显存占用 = 单卡模型大小 × GPU 数量，请确保每张卡都有充足显存。

**使用示例（4 卡并行）**：
```bash
export HIP_VISIBLE_DEVICES=4,5,6,7
python dlm/src/run_benchmark_llada2_multiprocess.py --benchmark gsm8k_small --max_examples 100
```

**多进程 vs 单进程的选择建议**：
- **单进程**（`run_benchmark_llada2.py`）：调试、小规模实验（< 50 样本）、显存不足时使用
- **多进程**（`run_benchmark_llada2_multiprocess.py`）：大规模 benchmark（> 50 样本）、追求最大吞吐量时使用

---

#### 7.3 基准测试数据集

**当前支持的基准测试类型**：

**数学基准测试**（已验证可用）：
- **GSM8K**：`dlm/experiments/benchmarks/math/gsm8k_small.jsonl`
  - 字段：`id`, `split`, `question`, `answer`
  - 任务：小学数学应用题求解
- **AIME 2025**：`dlm/experiments/benchmarks/math/aime2025_all.jsonl`
  - 字段：`id`, `subset`, `question`, `answer`
  - 任务：高难度数学竞赛题求解

**代码基准测试**（已实现支持）：
- **HumanEval**：`dlm/experiments/benchmarks/code/humaneval_all.jsonl`
  - 字段：`id`, `prompt`, `reference_code`, `tests`, `entry_point`
  - 任务：函数级代码补全（给定函数签名和文档，补全实现）
  - 评测：基于测试用例执行的 Pass@K 指标
- **MBPP**：`dlm/experiments/benchmarks/code/mbpp_sanitized.jsonl`
  - 字段：`id`, `prompt`, `reference_code`, `tests`
  - 任务：基于自然语言描述生成完整 Python 函数
  - 评测：基于测试用例执行的 Pass@K 指标

**使用示例**：
```bash
# 数学基准测试
python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --max_examples 20
python dlm/src/run_benchmark_llada2.py --benchmark aime2025_all --decode_mode lopa --max_examples 10

# 代码基准测试
python dlm/src/run_benchmark_llada2.py --benchmark humaneval_all --decode_mode baseline --max_examples 50
python dlm/src/run_benchmark_llada2.py --benchmark mbpp_sanitized --decode_mode ccd --max_examples 25

# 混合基准测试
python dlm/src/run_benchmark_llada2.py --benchmarks gsm8k_small humaneval_all mbpp_sanitized --decode_mode baseline
```

---

#### 7.4 结果评测与 LLM-as-a-Judge（当前唯一评测方式）

- **评测脚本**：`dlm/src/evaluate_benchmark_results.py`
  - **当前版本仅保留 LLM-as-a-Judge**：所有准确率统计均由 Judge 输出的 CORRECT/INCORRECT 决定，不再做规则数值抽取。
  - **注意**：当前 Judge 主要针对数学基准测试优化，代码基准测试的评测功能待扩展。
- **Judge 模型**：本机已下载 **Qwen3.5-9B**，路径为 `/etc/moreh/checkpoint/Qwen3.5-9B`（Hugging Face: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)）。运行 Judge 评测时会自动加载该模型，需占用一张 GPU 的显存。

**⚠️ 重要：评测必须使用 `.venv_eval` 环境，不能用 `.venv`**

原因：Qwen3.5-9B 需要 `transformers>=4.57.2`，而推理用的 `.venv` 固定在 `transformers==4.57.1`（LLaDA2.1-mini 要求），两者不兼容。

| 任务 | 虚拟环境 | transformers 版本 |
|------|---------|-----------------|
| LLaDA2.1-mini 推理 | `.venv` | 4.57.1（固定） |
| Qwen3.5-9B Judge 评测 | `.venv_eval` | ≥4.57.2 |

**创建 `.venv_eval`（首次使用时执行一次）**：
```bash
cd /home/qinghua/yangjh
uv venv .venv_eval
source .venv_eval/bin/activate
uv pip install "transformers>=4.57.2" accelerate
```

**运行评测**：
```bash
# 必须先激活 .venv_eval
source .venv_eval/bin/activate

# 评测单个结果文件
python dlm/src/evaluate_benchmark_results.py \
  --results_file dlm/experiments/runs/<文件名>.jsonl \
  --judge_model_path /etc/moreh/checkpoint/Qwen3.5-9B

# 评测整个目录下所有结果文件
python dlm/src/evaluate_benchmark_results.py \
  --results_dir dlm/experiments/runs/ \
  --judge_model_path /etc/moreh/checkpoint/Qwen3.5-9B
```

---

### 8. 后续计划（待完成）

- [进行中] 使用 AMD GPU：安装 ROCm 版 PyTorch 后跑通 `test_llada2_1_mini.py`。
- [待完成] 安装并测试 dInfer，尝试运行其 benchmarks 下的简单示例，理解解码组件接口。
- [待完成] 根据毕设需求决定是否安装 dFactory，并挑选小规模数据集做微调 demo。
