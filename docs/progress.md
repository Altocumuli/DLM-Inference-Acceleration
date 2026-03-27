## 扩散语言模型毕设进度与操作日志

> 建议你每次在服务器上做完「有意义的一步」操作（如：检查环境、创建虚拟环境、安装库、跑实验、写完一段重要代码），就来这里补一句简要记录，方便后期写论文和复现。

---

### 2026-03-13

1. **首次登录与环境检查**
   - 使用命令 `uname -a` 查看内核和架构信息（确认为 x86_64 架构的 Linux 服务器）。
   - 使用命令 `cat /etc/os-release` 查看操作系统版本（确认为 Ubuntu 22.04.4 LTS）。
   - 使用命令 `lscpu` 查看 CPU 详细信息（确认为 2 颗 AMD EPYC 7413，共 96 线程）。
   - 使用命令 `free -h` 查看内存容量和使用情况（总内存约 503 GiB，可用约 487 GiB）。
   - 使用命令 `df -h` 查看磁盘挂载与空间情况（系统盘 `/` 约 438G，数据盘 `/etc/moreh/checkpoint` 约 7T 空间）。
   - 使用命令 `lspci` 查看 PCI 设备，确认存在多块 AMD Instinct MI250X/MI250 GPU 加速卡。

2. **毕设工作目录初始化**
   - 在主目录下创建毕设项目目录 `dlm/`，用于存放扩散语言模型相关代码与实验。
   - 在 `dlm/` 下创建子目录：
     - `docs/`：用于保存文档与学习/实验笔记。
     - `src/`：用于保存源码与脚本。
     - `experiments/`：用于保存实验配置与结果。
   - 在主目录创建总说明文档 `README.md`，记录机器环境概况与项目结构规划。
   - 创建本文件 `dlm/docs/progress.md`，作为后续所有操作的时间线日志。

3. **环境规划与 LLaDA2.1-mini 实验准备**
   - 检查已有软件环境：
     - 使用 `which python3`、`python3 --version` 确认系统 Python 版本。
     - 使用 `which conda` / `which mamba` 确认当前机器上未安装 conda/mamba。
     - 使用 `which uv` 确认已安装 uv（师兄推荐用于管理虚拟环境）。
     - 使用 `rocminfo` 确认 ROCm 工具可用，说明 AMD GPU 软件栈已配置。
   - 为扩散语言模型（尤其是 `inclusionAI/LLaDA2.1-mini`）规划专用虚拟环境：
     - 采用 **uv** 创建和管理虚拟环境（`uv venv`，环境目录 `.venv`）。
     - 使用 `uv pip install` 安装 torch、transformers、accelerate、datasets、huggingface_hub、sentencepiece 等依赖。
     - 在 `dlm/docs/env.md` 中记录环境搭建思路、命令及 LLaDA2.1-mini 验证脚本说明。
   - 梳理 dInfer 与 dFactory 在毕设中的角色：dInfer 用于解码机制实验，dFactory 用于微调（可选）。

4. **改用 uv 管理环境并首次跑通 LLaDA2.1-mini**
   - 使用 `uv venv` 在项目根目录创建虚拟环境 `.venv`（Python 3.12.12）。
   - 使用 `uv pip install` 安装 torch 2.4.0、transformers 5.3.0 等依赖（共 69 个包）。
   - 首次运行 `python dlm/src/test_llada2_1_mini.py`：从 Hugging Face 拉取 LLaDA2.1-mini 配置与模型权重（约 32.5GB），下载完成后进行推理。
   - 将 Hugging Face 模型缓存目录设置为 `/etc/moreh/checkpoint/huggingface_hub`，避免占满 home。

---

### 2026-03-14

5. **首次在 AMD GPU 上跑通 LLaDA2.1-mini**
   - 安装 ROCm 6 版 PyTorch（`uv pip install torch --index-url .../rocm6.0`），脚本输出 `[device] 使用: cuda`，Generate 约 1 分半完成。
   - 首次输出为一长串 `0`，未得到算式答案；改为只解码「prompt 之后」的片段并打印，便于后续对比与调参。

6. **修复运行错误并恢复文档**
   - 运行测试脚本时出现 `KeyError: 'default'`（LLaDA2 使用 `rope_type="default"`，而 transformers 的 `ROPE_INIT_FUNCTIONS` 无该键）。在 `test_llada2_1_mini.py` 中在加载模型前注册 `"default"` RoPE 实现，问题解决。
   - 因故丢失的 `README.md`、`dlm/docs/progress.md`、`dlm/docs/env.md` 已按对话记录恢复。

7. **再次修复 RoPE 并增加运行日志**
   - 出现 `NotImplementedError: Cannot copy out of meta tensor`：模型加载时 device 可能为 meta，在 `_compute_default_rope_parameters` 中若 device 为 None 或 meta 则改为在 CPU 上创建 RoPE 张量，问题解决。
   - 新增 `dlm/log/` 目录，脚本运行时会自动将 stdout/stderr 同时写入 `dlm/log/YYYYMMDD_HHMMSS_脚本名.log`，tmux 后台跑时便于事后查看日志。

---

### 2026-03-15

8. **CPU 运行结果与 Transformers 版本建议**
   - 在 CPU 下运行测试（`DLM_CPU=1`，日志 `20260314_224207_test_llada2_1_mini.log`）：Generate 跑了约 599 步，但返回序列中 token 种类仅 2 种（id 30 出现 4 次、eos_id 1 次），解码结果为「????」，与 CUDA 下“仅 EOS”或“乱码”类似，均属生成/解码异常。
   - 社区有人推荐：**“We recommend using transformers==4.57.1 for inference.”** 当前环境为 transformers 5.3.0，可能与 4.x 在 attention/RoPE 或 generate 接口上存在差异。已在 `dlm/docs/env.md` 中将安装说明改为固定 **transformers==4.57.1**；建议执行 `uv pip install "transformers==4.57.1"` 降级后，分别在 CPU 与 CUDA 下重跑测试验证效果。

9. **降级 transformers 至 4.57.1 后 CUDA 推理正常**
   - 执行 `uv pip install "transformers==4.57.1"` 后，在 CUDA 下重跑测试（日志 `20260314_225026_test_llada2_1_mini.log`）：`[device] 使用: cuda`，Generate 约 78 步完成，返回序列 token 种类数 113，解码得到可读英文回答（关于 AI 未来的观点），推理行为正常。说明 **transformers 版本对 LLaDA2.1-mini 推理影响很大**，后续环境需固定使用 **transformers==4.57.1**。

---

### 2026-03-16

10. **下载并整理首批基准数据集**
    - 在 `dlm/src/` 新增脚本 `download_benchmarks.py`，使用 Hugging Face `datasets` 从 Hub 下载并落地以下数据集到 `dlm/experiments/benchmarks/`：
      - 数学类（math）：
        - `openai/gsm8k` 的 `main` 配置，截取部分 train/test 样本，保存为 `math/gsm8k_small.jsonl`（字段统一为 `id, split, question, answer`）。
        - `opencompass/AIME2025` 的 AIME2025-I / II 两个子集 test split，合并保存为 `math/aime2025_all.jsonl`（字段统一为 `id, subset, question, answer`）。
      - 代码类（code）：
        - `Muennighoff/mbpp` 的 `sanitized` 配置。
        - `openai/openai_humaneval`。
    - 统一采用 jsonl 一行一条样本的格式，为后续统一评测 pipeline 做数据准备。

11. **创建初版评测脚本 `run_benchmark_llada2.py`**
    - 在 `dlm/src/` 新增 `run_benchmark_llada2.py`，封装 LLaDA2.1-mini 的加载与 RoPE `"default"` 修复逻辑，与 `test_llada2_1_mini.py` 保持一致。
    - 当前支持基准：
      - `gsm8k_small`：从 `experiments/benchmarks/math/gsm8k_small.jsonl` 逐条读取样本，使用 tokenizer 的 `apply_chat_template` 构造 chat 输入，并用 baseline 解码配置（`gen_length=512, block_length=32, threshold=0.7, editing_threshold=0.5, temperature=0.0, max_post_steps=16, eos_early_stop=True`）生成答案。
      - `aime2025_all`：从 `experiments/benchmarks/math/aime2025_all.jsonl` 逐条读取样本，使用相同的 chat 模板与生成配置。
    - 命令行参数：
      - `--benchmark`：当前支持 `gsm8k_small` / `aime2025_all`。
      - `--decode_mode`：解码策略名称，占位为 `"baseline"`，便于后续扩展其它策略。
      - `--max_examples`：可选，上限样本数（便于先跑小规模 sanity check）。
    - 每条样本的生成结果以 jsonl 形式写入 `dlm/experiments/runs/YYYYMMDD_HHMMSS_llada2_<benchmark>_decode=<mode>.jsonl`，记录字段包括 `id, benchmark, decode_mode, question, reference_answer, model_answer`，用于后续统计准确率与对比不同解码策略。

12. **多卡尝试与阶段性结论（ROCm, 8 卡共享环境）**
    - 目标：在课题组共享的 8 张 AMD GPU（ROCm）上，为 LLaDA2.1-mini 实现「在后 4 张卡（4,5,6,7）上的多卡推理」，既减少单卡 OOM 风险，又避免和同学抢前 4 张卡。
    - 尝试过程：
      - 在 `run_benchmark_llada2.py` 中，将 `HIP_VISIBLE_DEVICES` 写死为 `"4,5,6,7"`，并且**确保在 `import torch` 之前设置**，使 PyTorch 只“看见”这 4 张卡。
      - GPU 分支使用：
        - `device_map="auto"` + `max_memory={i: "20GiB" for i in range(torch.cuda.device_count())}`；
        - 期望由 transformers/accelerate 将模型按模块切到多张逻辑卡上。
      - 通过 `rocm-smi --showuse` 观察到：
        - 显存（VRAM%）在目标 4 张卡之间比较均匀分布；
        - 但实际 GPU 利用率（GPU%）仍然集中在某一张卡上，其他卡 GPU% 很低（接近 0%），说明推理计算负载并没有做真正的并行均摊。
      - 日志中打印的 `hf_device_map` 结果也表明：在当前 `transformers==4.57.1 + ROCm + trust_remote_code=True` 组合下，自动切分多卡的行为并不稳定，很多情况下仍然退化为实质上的「单卡计算 + 多卡占显存」。
    - 阶段性结论：
      - 就目前环境与版本而言，这种「看起来多卡、实际上 GPU 算力集中在单卡」的状态，对真实吞吐和延迟的提升非常有限，不能算是严格意义上的「高效多卡并行」。
      - 出于毕设时间与精力的考虑，后续实验会**以「逻辑单卡（可控绑定到某一物理卡）」为主**，并在论文中将本次多卡尝试及其局限性作为「工程约束与未来工作」进行说明。

13. **引入「数据并行」选项（DataParallel，多卡各跑不同样本）**
    - 在 `run_benchmark_llada2.py` 中新增环境变量开关 `DLM_DATA_PARALLEL`：
      - 当未设置或为 `0` 时，行为与之前一致：CPU 单卡 / GPU 上使用 `device_map="auto"` 做自动放置/切分。
      - 当设置为 `1` 且当前可见 GPU 数大于 1 时：
        - 使用 `torch.nn.DataParallel` 将 LLaDA2.1-mini 复制到每一张可见 GPU 上（例如通过 `HIP_VISIBLE_DEVICES=4,5,6,7` 选定后 4 张卡）。
        - 同一 batch 中的不同样本会被 DataParallel 自动切分到不同 GPU 上并行前向计算，实现“多张 GPU 同时跑不同样本”的数据并行模式。
    - 注意点：
      - 数据并行会在每张参与的 GPU 上保留一份完整模型权重，对显存要求更高；需要在 `rocm-smi` 中确认每张卡有足够空闲显存后再启用。
      - 该模式适合在「显存足够、希望在多个 GPU 上同时跑更多样本」的场景下使用；对于当前毕设的 baseline 规模，可以作为加速选项，而不是硬性依赖。
    - 修复点：
      - 修复了 `torch.nn.DataParallel` 包装模型后 `generate` 方法访问问题：`DataParallel` 会将原始模型放在 `.module` 属性中，需要使用 `model.module.generate()` 而不是 `model.generate()` 来调用 LLaDA2.1-mini 的自定义生成方法。
    - 技术约束发现：
      - 通过测试发现，**LLaDA2.1-mini 的自定义 `generate` 方法不支持批量推理**（batch_size > 1）。
      - 具体表现：当尝试传入 `batch_size=4` 的输入张量时，在 `modeling_llada2_moe.py` 的第1329行会出现维度不匹配错误：`RuntimeError: The expanded size of the tensor (1) must match the existing size (4) at non-singleton dimension 0`。
      - 这意味着即使使用 `DataParallel`，由于每次只能处理单个样本（batch_size=1），**无法实现真正的数据并行加速**；DataParallel 退化为单卡运行。
      - 结论：在当前 LLaDA2.1-mini 版本下，**多卡数据并行在技术上不可行**，但可以通过多进程并行实现加速。

14. **实现多进程并行推理（进程级数据并行）**
    - 基于前述发现，实现了 `run_benchmark_llada2_multiprocess.py`：
      - **核心思路**：每个 GPU 启动独立 Python 进程，处理分配给它的样本子集，避开了模型层面的 batch_size=1 限制。
      - **技术架构**：
        - 主进程负责样本分割、进程管理和结果汇总；
        - Worker 进程各自在指定 GPU 上加载一份完整模型，处理分配的样本（每次仍为 batch_size=1）；
        - 使用 `multiprocessing.Pool` 进行进程管理，通过 `HIP_VISIBLE_DEVICES` 控制每个 Worker 进程的 GPU 可见性。
      - **预期效果**：在 4 张 GPU 上可实现接近 4 倍的吞吐量提升，适合大规模 benchmark 实验。
      - **使用示例**：
        ```bash
        export HIP_VISIBLE_DEVICES=4,5,6,7
        python dlm/src/run_benchmark_llada2_multiprocess.py --benchmark gsm8k_small --max_examples 32
        ```
    - **与单进程版本的区别**：
      - 单进程版本（`run_benchmark_llada2.py`）：稳定、资源占用低，适合小规模实验和调试；
      - 多进程版本（`run_benchmark_llada2_multiprocess.py`）：吞吐量高、显存占用大，适合大规模 benchmark 实验。

15. **单进程脚本新增多 benchmark 支持**
    - 扩展了 `run_benchmark_llada2.py` 的功能，现在支持一次运行多个 benchmark：
      - **新增 `--benchmarks` 参数**：可指定多个 benchmark，例如 `--benchmarks gsm8k_small aime2025_all`。
      - **模型复用**：模型只加载一次，避免重复加载的时间开销（每次约节省 6-8 秒）。
      - **独立结果文件**：每个 benchmark 仍生成独立的 `.jsonl` 结果文件，便于后续分析。
      - **统一日志**：所有 benchmark 的处理过程记录在同一个日志文件中。
    - **使用示例**：
      ```bash
      # 传统方式：单个 benchmark
      python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --max_examples 50
      
      # 新方式：多个 benchmark
      python dlm/src/run_benchmark_llada2.py --benchmarks gsm8k_small aime2025_all --max_examples 50
      ```
    - **实测效果**：成功处理 GSM8K（300样本）+ AIME2025（30样本），总计330个样本，功能稳定。

---

### 2026-03-17

17. **引入并最终采用 LLM-as-a-Judge 评测方式**
    - 在本机下载并配置 **Qwen3.5-9B** 作为评测 Judge 模型：
      - 本地路径：`/etc/moreh/checkpoint/Qwen3.5-9B`
      - Hugging Face：<https://huggingface.co/Qwen/Qwen3.5-9B>
    - 在 `evaluate_benchmark_results.py` 中最初「新增」了 **LLM Judge** 模式，随后将其升级为**唯一评测方式**：
      - 当前脚本总是使用 Judge 模型对每条样本进行正确性判断，不再做规则数值抽取；
      - **`--judge_model_path`**：指定 Judge 模型路径，默认 `DEFAULT_JUDGE_MODEL_PATH`（即上述 Qwen3.5-9B 路径）。
      - Judge 输入：题目 + 参考答案 + 模型回答；输出要求为 **CORRECT** 或 **INCORRECT**，脚本据此统计准确率。
    - **使用示例（当前唯一推荐方式）**：
      ```bash
      # 使用 LLM Judge 评测整个目录下的所有结果文件
      python dlm/src/evaluate_benchmark_results.py --results_dir dlm/experiments/runs/ --judge_model_path /etc/moreh/checkpoint/Qwen3.5-9B
      ```
    - **意义**：Judge 能更好处理格式多样、等价表述等情形，相比规则抽取更贴近人类评判；本毕设最终采用 **LLM-as-a-Judge + 时间/长度统计** 作为标准评测方式。

18. **实现 LoPA (Lookahead Parallel Decoding) 解码策略（初版 v1）**
    - 创建 `dlm/src/llada_lopa_decode.py`，实现 LoPA 三阶段算法：基础分支构建 → 前瞻分支生成 → 并行评估选择。
    - 集成到 `run_benchmark_llada2.py`，支持 `--decode_mode lopa`。

19. **实现 CCD (Coherent and Consistent Decoding) 解码策略**
    - 创建 `dlm/src/llada_ccd_decode.py`，实现预测-验证-跳步三阶段循环。
    - 集成到 `run_benchmark_llada2.py`，支持 `--decode_mode ccd`。
    - 新增效率指标统计（`gen_time_sec`、`input_token_len`、`output_token_len`）。

20. **扩展代码基准测试支持（HumanEval、MBPP）**
    - `run_benchmark_llada2.py` 新增 `humaneval_all` / `mbpp_sanitized` 支持；
    - 构造英文 prompt（与原始数据集语言一致）；输出记录增加 `prompt`、`formatted_question`、`reference_code`、`tests`、`entry_point` 字段。

---

### 2026-03-18

21. **修复 LoPA / CCD 前向取 logits 的错误**
    - **现象**：LoPA Stage 3 批量前向报 `'MoeModelOutputWithPast' object has no attribute 'logits'`，生成区大量仍为 mask。
    - **原因**：`model.model(...)` 返回 `MoeModelOutputWithPast`，仅含 `last_hidden_state`；logits 需经 `model.lm_head(last_hidden_state)` 得到。
    - **修复**：`llada_lopa_decode.py` 和 `llada_ccd_decode.py` 均新增 `_llada_forward_logits()` 辅助函数。

22. **修复 LoPA tuple index out of range 错误，完善 tmux 绑卡并行支持**
    - **现象**：`tensor.nonzero(as_tuple=True)[1]` 在 tuple 长度不足时越界。
    - **修复**：两处 tuple 索引均加长度检查；`run_benchmark_llada2.py` 改用 `os.environ.setdefault()` 支持外部 `HIP_VISIBLE_DEVICES` 覆盖；`env.md` 新增 tmux 多窗口绑卡并行章节。

23. **彻底重写 LoPA 为 v2 版，解决与扩散模型机制的根本矛盾**
    - **核心发现**：v1 在每个 block 内只做一次前向，违背了扩散模型**迭代去噪**本质（原生 generate 每个 block 内 `while True` 循环，每次迭代重新计算 logits）。
    - **v2 架构**：每个 block 内多次迭代，每次迭代重新 forward；在每次迭代中执行 LoPA 三阶段；使用与原生 generate 一致的 block attention mask。
    - **文件变更**：新增 `llada_lopa_decode_v2.py`（完全重写）；`run_benchmark_llada2.py` 优先导入 v2。

24. **修复 LoPA v2 形状索引错误**
    - **现象**：`IndexError: The shape of the mask [32] at index 0 does not match the shape of the indexed tensor [1, 32]`
    - **修复**：明确指定第 0 维：`base_x[:, block_start:block_end][0, high_conf_mask] = x0[0, high_conf_mask]`

25. **修复 LoPA v2 batch attention mask 形状错误**
    - **现象**：`ValueError: LLaDA2.0 only support block attention mask with shape: (4, 1, 1, 1), the input attention with shape attention_mask.size()=torch.Size([4, 1, 64, 64])`
    - **原因**：LLaDA2.1-mini forward 对 batch attention mask 形状要求严格，`.repeat()` 方式无法满足。
    - **修复**：将批量前向改为**逐个分支前向**，避免复杂的 batch mask 构造问题。

26. **LoPA v2 验证通过，用 v2 正式替换 v1**
    - 日志验证：5 条 GSM8K 样本均正常生成，EOS 正常检测，答案语义正确（72、10、5、42、624）。
    - 操作：删除旧版 `llada_lopa_decode.py`，将 `llada_lopa_decode_v2.py` 重命名为 `llada_lopa_decode.py`；`run_benchmark_llada2.py` 导入简化为直接 `from llada_lopa_decode import ...`。

27. **修复 CCD 实现的根本性错误（attention mask 构造问题）**
    - **现象**：每个 block 都报 `IndexError: The shape of the mask [1, 2112] at index 1 does not match the shape of the indexed tensor [1, 64, 157184]`，全部 fallback，生成内容只有 prompt 模板无实质内容。
    - **根本原因**：旧版 CCD 在 `_perform_ccd_block_step` 内每次 block 都重新构建 attention mask，且构建方式错误——用 `total_length` 大小的 mask 配合只截到 `block_end` 的 input_ids，导致形状不匹配。
    - **深层问题**：旧版 CCD 的整体结构也与 LLaDA2.1 的 block-wise 迭代机制不符，逻辑混乱。
    - **修复方案**：完全重写 `llada_ccd_decode.py`，对齐 LoPA v2 的正确结构：
      - 在 `generate_with_ccd` 入口处**一次性构建全局 block attention mask**，每次 forward 截取 `[:, :, :current_window_end, :current_window_end]` 子矩阵
      - 每个 block 内保持 `while True` 迭代循环（与原生 generate 一致）
      - 每次迭代：forward → 采样 → CCD 验证/接受 → 编辑 → 准备下一步 draft
      - CCD 核心逻辑：`_ccd_verify_and_accept`（验证 draft 一致性，接受多个 token）+ `_ccd_prepare_draft`（为下一步准备候选）

### 2026-03-19

28. **首次完整对比 Baseline vs CCD（GSM8K-small，300 条）**
    - Baseline：准确率 29.3%，平均生成时间 55.4 sec，吞吐量 6.45 tokens/sec
    - CCD：准确率 33.0%，平均生成时间 23.2 sec，吞吐量 11.66 tokens/sec
    - CCD 相比 Baseline：速度提升 **2.4×**，准确率提升 **+3.7pp**，正确吞吐量提升约 **2.7×**。
    - 结论：CCD 在 LLaDA2.1-mini 上的复现成功，同时实现加速与涨点。

29. **优化评测脚本效率指标，新建 evals 目录**
    - 新建 `dlm/experiments/evals/` 目录，将所有评测报告统一存放于此（与 `runs/` 同级）。
    - 将已有的 3 个评测结果文件从项目根目录迁移至 `evals/`。
    - 在 `evaluate_benchmark_results.py` 中新增以下效率指标：
      - `throughput_tok_per_sec`：整体吞吐量（tokens/sec）= **总输出 token 数 / 总生成时间**（正确定义）。
      - `correct_throughput`：正确吞吐量（correct_tokens/sec）= **正确样本的总输出 token 数 / 所有样本的总生成时间**（正确定义），综合衡量"又快又准"的能力，是跨方法横向对比的核心指标。
    - 保留 `avg_gen_time` 和 `avg_output_len` 作参考。
    - 报告默认输出路径改为 `dlm/experiments/evals/`，不再散落在项目根目录。

30. **修复吞吐量计算 bug**
    - **问题**：原实现用 per-sample 的 `len/time` 取均值，baseline 中短样本（时间短、吞吐量高）权重过大，导致 baseline 吞吐量虚高（12.17 vs 正确值 6.45），掩盖了与 CCD 的真实差距。
    - **修复**：改为 `总输出 token 数 / 总生成时间`，与工业界标准吞吐量定义一致。
    - **修正后数据**（GSM8K-small，300 条）：
      - Baseline：6.45 tokens/sec，正确吞吐量 1.89 correct_tokens/sec
      - CCD：11.66 tokens/sec，正确吞吐量 3.85 correct_tokens/sec
      - CCD 正确吞吐量是 Baseline 的 **2.04×**

31. **~~新增 thinking 模式支持~~（已移除，2026-03）**
    - 不再维护 LLaDA benchmark 侧的 `--thinking` / `llada2_chat_prompt`；统一 `apply_chat_template`。

---

### 2026-03-20

32. **修复 LLM Judge 因 max_new_tokens 不足导致的大量误判**
    - **现象**：Judge（Qwen3.5-9B）对明显正确的回答也判为 INCORRECT；查看 `dlm/log/eval/` 下的 judge_details 日志，发现 Judge Raw Response 在输出思维链中途被截断，未能输出最终的 CORRECT/INCORRECT 结论。
    - **根本原因**：`_judge_single_sample` 中 `max_new_tokens` 默认值为 32，而 Qwen3.5-9B 默认开启 thinking，先输出若干 thinking 步骤，32 个 token 远不够，导致截断后默认判 INCORRECT。
    - **修复**：
      1. 将 `max_new_tokens` 从 32 改为 **2048**，确保 Judge 有足够空间输出完整回答。
      2. 在 `tokenizer.apply_chat_template` 中显式传入 `enable_thinking=False`，抑制 thinking 输出，节省时间。
    - **影响**：此前所有评测结果（准确率 29.3%、33.0% 等）均在 Judge 大量误判的情况下得出，**数值不可信，需重新评测**。

33. **新增 ARC（AI2 Reasoning Challenge）基准数据集**
    - 在 `download_benchmarks.py` 中新增 `export_arc(n=300)` 函数：
      - 数据源：`allenai/ai2_arc`（HuggingFace Hub）
      - 分别下载 `ARC-Easy` 和 `ARC-Challenge` 两个子集，各取 300 条（优先 test，不足则补 validation/train）。
      - 字段：`id, question, choices（label+text 列表）, answer_key（正确字母）, answer_text（正确选项文本）`。
      - 保存路径：`dlm/experiments/benchmarks/reasoning/arc_easy.jsonl` 和 `arc_challenge.jsonl`（新建 `reasoning/` 子目录）。
    - 在 `run_benchmark_llada2.py` 中新增 ARC 支持：
      - 新增 `iter_arc_easy_examples` / `iter_arc_challenge_examples` 数据加载函数。
      - 新增 `_format_arc_prompt` 函数：要求模型给出完整分析过程并在最后指明答案字母（A/B/C/D），保留推理过程供错误分析，同时 LLM Judge 有能力从完整回答中判断对错。
      - `--benchmark` / `--benchmarks` 参数新增 `arc_easy` / `arc_challenge` 选项。
      - 输出 record 新增字段：`question, choices, answer_key, reference_answer`。

34. **优化 ARC 评测的 Judge Prompt（选择题专用分支）**
    - **背景**：ARC 是选择题，原 Judge prompt 是为数学开放题设计的，对选择题不够精准。
    - **修改**：在 `evaluate_benchmark_results.py` 的 `_build_judge_prompt` 中新增选择题分支：
      - 当 record 含 `answer_key` 字段时（ARC 等选择题），使用专用 prompt：展示所有选项、明确告知正确答案字母、提示 Judge 关注模型最终选择的字母是否正确。
      - 当 record 不含 `answer_key` 时（GSM8K、AIME 等开放题），保持原有数学推理判断 prompt 不变。
    - `_judge_single_sample` 同步读取 `answer_key` 和 `choices` 字段并传入 prompt 构建函数。

35. **LoPA / CCD 实现与论文对比 check，修正关键错误**

    **LoPA（`llada_lopa_decode.py`）**：
    - **问题**：Branch Confidence 计算定义错误。原实现对**已填充位置**的平均概率打分，而论文 Eq.2 定义为对**剩余 mask 位置**的平均置信度（`max_prob`），衡量"未来还有多少位置可以高置信度被填充"（future parallelism potential）。
    - **修复**：`_execute_lopa_step` 中改为对 `remaining_mask_positions` 的 `max_prob` 均值打分；若分支已无剩余 mask，给予满分 1.0（优先选择已完成的分支）。
    - **其他差异说明**（合理适配，无需修改）：
      - 分支验证方式：论文设计为单次 batch forward；实现改为逐个 forward，原因是 LLaDA2.1-mini 对 batch attention mask 形状有严格要求，无法直接打包多分支。
      - attention 机制：论文在 D2F 上将 block causal 替换为 full attention；LLaDA2.1 必须保持 block causal mask，是必要的适配。

    **CCD（`llada_ccd_decode.py`）**：
    - **问题**：历史 buffer 深度不足。原实现只保留上一步的 (位置, token_id) 对，等价于 $d=1$ 且只保留 argmax，丢失了论文核心的多步历史分布均值机制（论文 LLaDA 系列用 $d=2$，Dream 系列用 $d=3$）。
    - **修复**：重写为 `CcdHistoryBuffer` 类，维护最近 $d$ 步中 top-V 位置的**完整预测分布**（softmax 概率向量），能计算真正的近似边际分布（论文 Eq.6）。
    - **token 选择逻辑**：
      - 历史 buffer 与当前步 top-V 取交集 $I_t^c$（跨步一致的高置信度位置）
      - 对 $I_t^c$ 用多步分布均值的 argmax 选 token
      - 检查 token 稳定性（argmax 在所有历史步中一致）→ 一次性接受所有稳定 token（CCD-DS 自适应加速）
    - **其他差异说明**（合理适配，无需修改）：
      - token 稳定性代理：论文用"边际熵 < ε"；实现用"argmax 跨步一致"作为代理，无需手动调 ε（论文 4.2 节 CCD-DS 实现细节中也提到此做法）。
      - 历史 buffer 跨 block 不共享：因为 block causal mask 使得不同 block 的上下文相互隔离，跨 block 共享历史无意义。

36. **深度 check LoPA / CCD 实现与 LLaDA2.1 源码的一致性，修复若干细节问题**

    本次结合 LoPA 原始代码（`scale_dream_d2f.py`）、CCD 原始代码（`model.py` / `_sample_inference_lookahead`）、LLaDA2.1 `generate()` 源码（`modeling_llada2_moe.py`）三方对比，发现并修复以下问题：

    **修复 1：CCD attention mask 格式错误（`llada_ccd_decode.py`）**
    - **问题**：`generate_with_ccd` 中对 `global_attn_mask` 做了额外的 additive 转换（`(1.0 - mask) * -inf`），导致 mask 被双重处理：先在我们的代码里转成 additive 格式，再进入 `model.model()` 时被 `_prepare_4d_causal_attention_mask_for_sdpa` 再次处理，造成 attention 错误。
    - **修复**：删除该转换，保持与 LLaDA2.1 原生 `generate()` 完全一致的 0/1 格式（1=可见，0=遮蔽）。
    - **LoPA 无此问题**：LoPA 实现从一开始就正确使用 0/1 格式。

    **修复 2：`_llada_forward_logits` 缺少 float32 转换（LoPA + CCD）**
    - **问题**：`base.lm_head()` 输出的是 bfloat16，而 LLaDA2.1 原生 `forward()` 在 `lm_head` 之后做了 `logits = logits.float()`（float32）。我们的实现缺少这一步，导致后续 softmax 等操作在 bfloat16 精度下进行，可能引入数值误差。
    - **修复**：在 `_llada_forward_logits` 末尾加 `.float()`，与原生 `forward()` 保持一致。

    **修复 3：迭代上限不够安全（LoPA + CCD）**
    - **问题**：LoPA 的 `_lopa_decode_block` 用固定 `max_post_steps + 20` 作为上限，CCD 的 `_ccd_decode_block` 用 `max_post_steps + 40`，在极端情况下（每次只填 1 个 token）可能不够。
    - **修复**：改为 `block_len + max_post_steps + 10`，确保即使每次只填 1 个 token 也能完成整个 block。

    **确认无需修改的项目**：
    - **logits shift**：CCD/Dream 原始代码有 `logits = torch.cat([logits[:,:1], logits[:,:-1]], dim=1)`，但这是 Dream 模型特有的 next-token 对齐操作。LLaDA2.1 是 masked diffusion 模型，logits 直接对应当前位置，原生 `generate()` 中无 shift，我们的实现正确。
    - **LoPA attention mask**：已经是 0/1 格式，正确。
    - **`base.model()` + `base.lm_head()` 调用方式**：与 `self.forward()` 等价，正确。

    **CCD 实现与原始代码的设计差异（合理适配）**：
    - 原始 CCD 代码的 verification buffer 存储的是 token 位置（index），通过双步 argmax 一致性检验来加速；我们的实现存储完整概率分布并计算多步均值，更接近论文的理论描述（Eq.6 边际分布近似）。两者都是对"跨步一致性"的合理实现。
    - 原始代码 `verification_buffer_size` 动态增长（从 1 开始，每步 +1）；我们用固定 `top_v=4`。这影响加速效果但不影响正确性。

37. **CCD 回退双步一致性检验；实现毕设原创策略 CLAD**

    **背景**：发现 CCD 的多步分布均值实现与开源代码实际采用的双步 argmax 一致性检验存在差异。鉴于：
    - 开源代码的双步检验更简洁可靠，已被实际验证
    - 需要为毕设提供一个有实质创新的原创策略

    决定拆分为两件事：CCD 回归开源实现；在此基础上另立 CLAD 作为原创创新点。

    **CCD 变更（`llada_ccd_decode.py`）**：
    - `CcdHistoryBuffer` 改为存储 `(positions [k], argmax_tokens [k])`，移除完整概率分布存储
    - 移除 `get_consistent_candidates()`（多步均值）和 `_check_token_stability()`
    - 新增 `get_consistent_positions()`：实现双步 argmax 一致性检验，对应开源代码逻辑
    - `_ccd_select_tokens()` 简化：一致性命中直接接受，否则退回阈值填充
    - 保留所有之前的正确修复：0/1 attention mask 格式、`.float()` 精度、安全迭代上限

    **CLAD 新增（`llada_clad_decode.py`）——毕设核心创新**：
    - **核心思想**：将 CCD 的"被动一致性检测"提升为"主动一致性引导"
    - **三阶段解码**：
      1. 阶段一（一致性快速通道）：与 CCD 相同，双步 argmax 一致 → 直接接受，0 额外 forward 开销
      2. 阶段二（一致性传播前瞻）：**CLAD 原创**，当阶段一无候选时，对 top-k 位置做前瞻，
         用"填入后其他位置有多少能保持预测一致"（consistency_propagation）评估分支，
         选择最能稳定周围位置的分支填入（score = α×consistency_prop + (1-α)×future_conf）
      3. 阶段三（退回阈值填充）：无有效前瞻时的保底行为
    - **创新本质**：两篇论文均未使用"一致性传播"作为前瞻评分标准；LoPA 用未来置信度，CCD 不做前瞻，CLAD 将两者融合并提升
    - 接入 `run_benchmark_llada2.py`（`--decode_mode clad`）
    - 文档更新：`design.md` 新增 §1.9 CLAD 完整设计描述；§1.8 CCD 部分修订为双步实现

---

### 2026-03-21

38. **TPF（Tokens Per Forward）指标：run 记录 forward_count，eval 汇总**

    - **问题**：仅含 `output_token_len` / `gen_time_sec` 的旧 jsonl **无法**推算真实前向次数，故无法从旧结果计算 TPF。
    - **实现**：`run_benchmark_llada2.py` 每条样本写入 `forward_count`（baseline 对 `model.model` 注册 forward hook；LoPA/CCD/CLAD 在 `_llada_forward_logits` 中计数）。
    - **定义**：`TPF = output_token_len / forward_count`；评测报告给出 **macro**（Σoutput/Σforward，论文常用整体口径）与 **mean per sample**（逐样本比值再平均）。
    - **evaluate_benchmark_results.py**：当 jsonl 无 `forward_count` 时提示需用新脚本重跑 benchmark。

---

### 2026-03-22

39. **~~修复 LLaDA2.1-mini thinking 未生效~~**
    - 历史：曾用 `llada2_chat_prompt.py` 绕过模板；现 benchmark 不再测 thinking 路径，已删除该模块与 CLI。

40. **CLAD-v2：信息密度加权评分（O1）+ 多 token 自适应接受（O2）**
    - **O1 动机**：v1 的一致率用均匀权重，忽略了不同位置的信息密度差异。灵感来源：*Mask Is What DLLM Needs* 的推理侧镜像——让高熵（高信息密度）位置稳定比让低熵位置稳定更有意义。
    - **O1 实现**：`_clad_branch_score` 改为三项联合评分：① 以当前步熵为权重的加权一致率、② 填入 token 后剩余 mask 位置的熵下降奖励（归一化到 [-1,1]）、③ LoPA-style 未来置信度。
    - **O2 动机**：v1 每次阶段二做 k 次 forward 只接受 1 个 token，TPF 极低；最优分支 logits 直接丢弃是浪费。
    - **O2 实现**：`_clad_lookahead_fill` 在填入最优 token 后，复用分支 logits 检查剩余 mask 位置的最高置信度；若 ≥ `accept_threshold2`（默认 0.90）则零额外 forward 再接受 1 个 token，理论 TPF 提升约 2×。
    - **新文件**：`dlm/src/llada_clad_v2_decode.py`；接入方式：`--decode_mode clad_v2`。
    - **文档**：`design.md` §1.10。

---

### 2026-03-24

42. **记录中期后续实验想法：跨步一致性最小置信度门控（暂不改代码）**
    - **现状确认**：
      - `CCD-DLM-code` 与当前 `llada_ccd_decode.py` / `llada_clad_v2_decode.py` 的一致性快速通道，均采用
        `top-V 交集 + argmax 一致` 作为接受条件；
      - 一致性通道本身**没有额外绝对置信度阈值**（如 `p>=0.5`）；
      - 阈值参数 `threshold=0.7` 主要用于 fallback 的阈值填充阶段，而非一致性接受阶段。
    - **计划（中期后再做）**：
      - 增加可选参数 `consistency_min_conf`（默认关闭，保持现有行为）；
      - 在一致性命中后增加门控：仅当 `cur_prob >= consistency_min_conf` 时才接受；
      - 设计消融实验：`None / 0.5 / 0.6 / 0.7 / 0.8`，比较 Accuracy、Throughput、TPF、Diffusion Steps。
    - **当前决策**：为了中期报告稳定，不改代码与结果口径，仅在文档中留档该实验方向。

---

### 2026-03-25

### 2026-03-26（文件恢复）

- `dlm/docs/` 下 `thesis_draft.md`、`make_charts.py` 及若干 `*.md`，以及 `dlm/src/` 下全部 `*.py` 曾一度缺失（`__pycache__` 仍在）。
- 已从 **Cursor/VS Code 远端本地历史** 路径 `~/.cursor-server/data/User/History/` 中按各文件 **entries.json 里时间戳最新的快照** 拷回工程目录；`README.md` 同步恢复。
- **说明**：`llada2_chat_prompt.py` 按此前文档已弃用，无 History 条目；当前 `run_benchmark_llada2.py` 等已统一 `apply_chat_template`，不依赖该文件。
- **建议**：尽快 `git init` + 提交，或定期备份 `dlm/`，避免再次丢失。
