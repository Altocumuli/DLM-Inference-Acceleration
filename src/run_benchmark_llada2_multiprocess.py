"""
LLaDA2.1-mini 多进程并行推理版本。

每个 GPU 启动独立进程，处理分配给它的样本子集，最后汇总结果。
由于 LLaDA2.1-mini 的 generate 方法不支持 batch_size > 1，我们采用"进程并行"而非"数据并行"。

用法示例：
    export HIP_VISIBLE_DEVICES=4,5,6,7
    python dlm/src/run_benchmark_llada2_multiprocess.py --benchmark gsm8k_small --max_examples 32
"""

import os
import sys
import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import signal

# 必须在 import torch 之前设置基础环境
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/etc/moreh/checkpoint/huggingface_hub")

# 获取用户指定的 GPU 列表
visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", "4,5,6,7")
gpu_list = [int(x.strip()) for x in visible_devices.split(",")]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_rope_utils


# ---------------------------------------------------------------------------
# 1. 复用核心组件（RoPE修复、模型加载、数据读取）
# ---------------------------------------------------------------------------


def _compute_default_rope_parameters(
    config, device=None, seq_len=None, layer_type=None
):
    """标准 RoPE（无 scaling），与 transformers 其它 ROPE 初始化函数签名一致。"""
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    partial = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim)
    )
    return inv_freq, 1.0


if "default" not in getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", {}):
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
        _compute_default_rope_parameters
    )


MODEL_ID = "/etc/moreh/checkpoint/huggingface_hub/models--inclusionAI--LLaDA2.1-mini/snapshots/bbb5715c881500b34234071e68dbf38c3d657c4e"


def load_model_on_single_gpu(gpu_id: int):
    """在指定的单个 GPU 上加载 LLaDA2.1-mini。"""
    # 设置当前进程只能看到这一张卡
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Worker GPU {gpu_id}] Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map=None,
    )
    model = model.to(torch.bfloat16).to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"[Worker GPU {gpu_id}] Model loaded successfully")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. 单样本推理函数（与原版基本一致）
# ---------------------------------------------------------------------------


def run_generate_single_sample(
    model, tokenizer, prompt: str, decode_mode: str = "baseline"
) -> str:
    """单样本推理，与 run_benchmark_llada2.py 中的 run_generate_single 逻辑一致。"""
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
    input_ids = input_ids.long().to("cuda")

    with torch.no_grad():
        generated_tokens = model.generate(
            inputs=input_ids,
            eos_early_stop=True,
            gen_length=2048,
            block_length=32,
            threshold=0.7,
            editing_threshold=0.5,
            temperature=0.0,
            max_post_steps=16,
        )

    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. 数据加载（复用原版逻辑）
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT / "experiments" / "benchmarks"


def read_jsonl(path: Path):
    """读取 jsonl 文件。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_benchmark_samples(benchmark_name: str, max_examples: int = None) -> List[Dict]:
    """加载指定 benchmark 的样本，返回列表格式便于分割。"""
    if benchmark_name == "gsm8k_small":
        file_path = BENCH_ROOT / "math" / "gsm8k_small.jsonl"
    elif benchmark_name == "math500":
        file_path = BENCH_ROOT / "math" / "math500.jsonl"
    elif benchmark_name == "aime2025_all":
        file_path = BENCH_ROOT / "math" / "aime2025_all.jsonl"
    else:
        raise ValueError(f"未知 benchmark: {benchmark_name}")

    samples = list(read_jsonl(file_path))
    if max_examples is not None:
        samples = samples[:max_examples]

    return samples


# ---------------------------------------------------------------------------
# 4. Worker 进程函数
# ---------------------------------------------------------------------------


def worker_process(
    gpu_id: int, samples: List[Dict], decode_mode: str, worker_idx: int
) -> List[Dict]:
    """
    Worker 进程：在指定 GPU 上处理分配的样本列表。

    Args:
        gpu_id: 物理 GPU ID
        samples: 分配给该 worker 的样本列表
        decode_mode: 解码模式
        worker_idx: worker 编号（用于日志）

    Returns:
        处理完成的结果列表
    """
    try:
        # 忽略信号，避免主进程终止时影响子进程
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        print(
            f"[Worker {worker_idx} GPU {gpu_id}] Starting, {len(samples)} samples assigned"
        )

        # 在当前 GPU 上加载模型
        model, tokenizer = load_model_on_single_gpu(gpu_id)

        results = []
        for i, sample in enumerate(samples):
            try:
                question = sample["question"]
                sample_id = sample.get("id", f"unknown_{i}")
                ref_answer = sample.get("answer")

                print(
                    f"[Worker {worker_idx} GPU {gpu_id}] Processing sample {i+1}/{len(samples)} (id={sample_id})"
                )

                # 生成答案
                generated_text = run_generate_single_sample(
                    model, tokenizer, question, decode_mode
                )

                # 构造结果记录
                result = {
                    "id": sample_id,
                    "benchmark": "multiprocess",  # 标记这是多进程版本的结果
                    "decode_mode": decode_mode,
                    "question": question,
                    "reference_answer": ref_answer,
                    "model_answer": generated_text,
                    "worker_gpu": gpu_id,  # 记录处理的 GPU
                }
                results.append(result)

                print(
                    f"[Worker {worker_idx} GPU {gpu_id}] Completed sample {i+1}, preview: {generated_text[:80]}..."
                )

            except Exception as e:
                print(
                    f"[Worker {worker_idx} GPU {gpu_id}] Error processing sample {i}: {e}"
                )
                # 即使某个样本失败，也继续处理其他样本
                results.append(
                    {
                        "id": sample.get("id", f"error_{i}"),
                        "benchmark": "multiprocess",
                        "decode_mode": decode_mode,
                        "question": sample.get("question", ""),
                        "reference_answer": sample.get("answer", ""),
                        "model_answer": f"ERROR: {str(e)}",
                        "worker_gpu": gpu_id,
                    }
                )

        print(f"[Worker {worker_idx} GPU {gpu_id}] Finished, {len(results)} results")
        return results

    except Exception as e:
        print(f"[Worker {worker_idx} GPU {gpu_id}] Fatal error: {e}")
        return []


# ---------------------------------------------------------------------------
# 5. 主控逻辑
# ---------------------------------------------------------------------------


def split_samples_to_workers(samples: List[Dict], num_workers: int) -> List[List[Dict]]:
    """将样本列表均匀分配给 workers。"""
    chunks = [[] for _ in range(num_workers)]
    for i, sample in enumerate(samples):
        chunks[i % num_workers].append(sample)
    return chunks


def save_results(results: List[Dict], output_file: Path):
    """保存结果到 jsonl 文件。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLaDA2.1-mini 多进程并行 benchmark 推理"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["gsm8k_small", "math500", "aime2025_all"],
        required=True,
        help="要测试的 benchmark 数据集",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="baseline",
        help="解码策略名称",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="最多处理多少条样本（默认全部）",
    )
    args = parser.parse_args()

    print(f"[Main] Starting multiprocess benchmark on GPUs: {gpu_list}")
    print(f"[Main] Benchmark: {args.benchmark}, Max examples: {args.max_examples}")

    # 加载样本数据
    try:
        samples = load_benchmark_samples(args.benchmark, args.max_examples)
        print(f"[Main] Loaded {len(samples)} samples")
    except Exception as e:
        print(f"[Main] Error loading samples: {e}")
        return

    if len(samples) == 0:
        print("[Main] No samples to process")
        return

    # 分配样本到各个 worker
    num_workers = len(gpu_list)
    sample_chunks = split_samples_to_workers(samples, num_workers)

    for i, chunk in enumerate(sample_chunks):
        print(f"[Main] Worker {i} (GPU {gpu_list[i]}): {len(chunk)} samples")

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        ROOT
        / "experiments"
        / "runs"
        / f"{timestamp}_llada2_multiprocess_{args.benchmark}_decode={args.decode_mode}.jsonl"
    )
    print(f"[Main] Results will be saved to: {output_file}")

    # 启动多进程
    start_time = datetime.now()
    try:
        with mp.Pool(num_workers) as pool:
            # 为每个 worker 准备参数
            worker_args = [
                (gpu_list[i], sample_chunks[i], args.decode_mode, i)
                for i in range(num_workers)
            ]

            print(f"[Main] Starting {num_workers} worker processes...")

            # 并行执行
            worker_results = pool.starmap(worker_process, worker_args)

        # 汇总结果
        all_results = []
        for results in worker_results:
            all_results.extend(results)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"[Main] All workers completed in {elapsed:.2f} seconds")
        print(f"[Main] Total results: {len(all_results)}")

        # 保存结果
        save_results(all_results, output_file)
        print(f"[Main] Results saved to: {output_file}")

        # 统计信息
        successful_results = [
            r for r in all_results if not r["model_answer"].startswith("ERROR:")
        ]
        print(
            f"[Main] Successful: {len(successful_results)}, Failed: {len(all_results) - len(successful_results)}"
        )

    except KeyboardInterrupt:
        print("[Main] Interrupted by user")
    except Exception as e:
        print(f"[Main] Error during multiprocess execution: {e}")


if __name__ == "__main__":
    # 设置多进程启动方式（重要：在某些平台上避免问题）
    mp.set_start_method("spawn", force=True)
    main()
