import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# 必须在 import torch 之前设置 GPU 可见性
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/etc/moreh/checkpoint/huggingface_hub")
# 只在没有设置时才默认使用7号卡
os.environ.setdefault("HIP_VISIBLE_DEVICES", "7")
# 启用数据并行
os.environ["DLM_DATA_PARALLEL"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_rope_utils

# LoPA 解码模块
from llada_lopa_decode import generate_with_lopa, LopaConfig

# CCD 解码模块
from llada_ccd_decode import generate_with_ccd, CcdConfig

# CLAD 解码模块（毕设原创策略）
from llada_clad_decode import generate_with_clad, CladConfig

# CLAD-v2 解码模块（信息密度加权评分 + 多 token 自适应接受）
from llada_clad_v2_decode import generate_with_clad as generate_with_clad_v2
from llada_clad_v2_decode import CladConfig as CladV2Config

# CLAD-v3（O1/O2 + O3 批量 Phase-2 forward + O4 级联草稿）
from llada_clad_v3_decode import generate_with_clad_v3
from llada_clad_v3_decode import CladV3Config


"""
用 LLaDA2.1-mini 在本地 benchmarks 上跑不同的解码策略，并记录解码效率指标。

支持的解码策略：
- baseline: LLaDA2.1-mini 原生的解码方法
- lopa: 基于 LoPA (Lookahead Parallel Decoding) 的多分支解码策略
- ccd: 基于 CCD (Coherent and Consistent Decoding) 的双步一致性加速策略
- clad: CLAD v1（Consistency-guided Lookahead Adaptive Decoding），毕设原创策略
- clad_v2: CLAD v2，在 v1 基础上引入信息密度加权一致性评分（O1）和多 token 自适应接受（O2）
- clad_v3: CLAD v3，在 v2 基础上增加 Phase-2 分支批量 forward（O3）与级联草稿前瞻（O4）

当前支持的 benchmark 文件（均为 jsonl，一行一个样本）：
- math:
  - experiments/benchmarks/math/gsm8k_small.jsonl   （字段：id, split, question, answer）
  - experiments/benchmarks/math/math500.jsonl         （Hendrycks MATH test 500 题；字段：id, question, answer, level, subject, …）
  - experiments/benchmarks/math/aime2025_all.jsonl  （字段：id, subset, question, answer）
- code:
  - experiments/benchmarks/code/humaneval_all.jsonl （字段：id, prompt, reference_code, tests, entry_point）
  - experiments/benchmarks/code/mbpp_sanitized.jsonl（字段：id, prompt, reference_code, tests）

每条样本的输出记录中，额外包含以下效率相关字段：
- input_token_len   : prompt 在 tokenizer 下的 token 数
- output_token_len  : 生成文本在 tokenizer 下的 token 数
- gen_time_sec      : 本条样本从调用解码到返回的耗时（秒）
- forward_count     : 解码过程中主干 Transformer（model.model）的前向次数；TPF = output_token_len / forward_count

用法示例：
    # 数学基准测试 - 使用 baseline 解码策略
    python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --max_examples 20
    
    # 代码基准测试 - 使用 LoPA 解码策略
    python dlm/src/run_benchmark_llada2.py --benchmark humaneval_all --decode_mode lopa --max_examples 10
    
    # 使用 CCD 解码策略
    python dlm/src/run_benchmark_llada2.py --benchmark mbpp_sanitized --decode_mode ccd --max_examples 10
    
    # 使用 CLAD v1 解码策略（毕设原创）
    python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode clad --max_examples 10
    
    # 使用 CLAD v2 解码策略（信息密度加权 + 多 token 接受）
    python dlm/src/run_benchmark_llada2.py --benchmark gsm8k_small --decode_mode clad_v2 --max_examples 10
    
    # MATH500 全量（需先用 download_benchmarks.py 导出 math500.jsonl）
    python dlm/src/run_benchmark_llada2.py --benchmark math500 --decode_mode clad_v2
    
    # 一次运行多个 benchmark（包括数学和代码）
    python dlm/src/run_benchmark_llada2.py --benchmarks gsm8k_small math500 humaneval_all mbpp_sanitized --decode_mode baseline
"""


# ---------------------------------------------------------------------------
# 1. 日志与环境初始化（与 test_llada2_1_mini.py 风格一致）
# ---------------------------------------------------------------------------

_script_name = os.path.splitext(os.path.basename(__file__))[0]
_log_dir = (Path(__file__).resolve().parents[1] / "log" / "run").as_posix()
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_script_name}.log",
)
_log_file = open(_log_path, "w", encoding="utf-8")


class _Tee:
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f"[log] 输出已同时写入: {_log_path}")


# ---------------------------------------------------------------------------
# 2. 环境与 RoPE 修复（与 test_llada2_1_mini.py 保持一致）
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


# ---------------------------------------------------------------------------
# 2. 模型加载（只加载一次，后续循环复用）
# ---------------------------------------------------------------------------

MODEL_ID = "/etc/moreh/checkpoint/huggingface_hub/models--inclusionAI--LLaDA2.1-mini/snapshots/bbb5715c881500b34234071e68dbf38c3d657c4e"


def load_llada_model_and_tokenizer():
    """
    加载 LLaDA2.1-mini。
    - 若设置 DLM_CPU=1，则在 CPU 上单卡推理；
    - 否则：
      - 若设置 DLM_DATA_PARALLEL=1 且当前可见 GPU 数 > 1，则使用 DataParallel 做「数据并行」（注意：经测试发现 LLaDA2.1-mini 不支持批量推理，此选项实际无效）；
      - 否则在当前可见 GPU 上单卡推理（通过 HIP_VISIBLE_DEVICES 控制，此脚本当前固定为 7 号卡）。
    """
    use_cpu = bool(os.environ.get("DLM_CPU"))
    use_dp = bool(os.environ.get("DLM_DATA_PARALLEL"))

    if use_cpu or not torch.cuda.is_available():
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map=None,
        )
        model = model.to(torch.bfloat16).to(device)
    else:
        num_devices = torch.cuda.device_count()
        if use_dp and num_devices > 1:
            # 数据并行：在每张可见 GPU 上放一份完整模型副本，由 DataParallel 自动把 batch 里的不同样本分配到不同 GPU。
            base_device = torch.device("cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map=None,
            )
            model = model.to(torch.bfloat16).to(base_device)
            model = torch.nn.DataParallel(model)
            print(
                f"[devices] DataParallel on {list(range(num_devices))} (logical device ids)"
            )
        else:
            # 默认：仍然交给 device_map="auto" 在可见卡之间自动放置/切分
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )
            # 打印实际 device 映射，方便你检查到底用了哪些卡
            try:
                device_map = getattr(model, "hf_device_map", None)
                print(f"[devices] hf_device_map={device_map}")
            except Exception:
                pass

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. 通用工具：读 jsonl、写 jsonl
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # 指向 dlm/
BENCH_ROOT = ROOT / "experiments" / "benchmarks"
RUN_ROOT = ROOT / "experiments" / "runs"


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_completed_sample_ids(path: Path) -> set:
    """读取已有结果文件中的 sample id，用于续跑时跳过已完成样本。"""
    if not path.is_file():
        return set()
    done_ids = set()
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            done_ids.add(sample_id)
    return done_ids


# ---------------------------------------------------------------------------
# 4. 构造 chat 输入并调用 generate（baseline 策略）
# ---------------------------------------------------------------------------


def run_generate_single(
    model,
    tokenizer,
    prompt: str,
    decode_mode: str = "baseline",
    gen_length: int = 2048,
    clad_overrides: Optional[Dict] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """
    使用指定的解码策略进行生成。

    clad_overrides: dict，可覆盖 CLAD v1/v2 的超参数，如：
        {"num_lookahead": 3, "consistency_weight": 0.4,
         "entropy_weight": 0.3, "accept_threshold2": 0.85}
    stats_out: 若传入非 None 的列表，CLAD v1/v2 运行后会 append DecodeStats 实例，
        可从中读取各阶段命中率。

    Returns:
        (生成的文本, forward_count)
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    ov = clad_overrides or {}

    if decode_mode == "baseline":
        return _run_baseline_generate(
            actual_model, tokenizer, prompt, gen_length=gen_length
        )
    if decode_mode == "lopa":
        return _run_lopa_generate(
            actual_model, tokenizer, prompt, gen_length=gen_length
        )
    if decode_mode == "ccd":
        return _run_ccd_generate(actual_model, tokenizer, prompt, gen_length=gen_length)
    if decode_mode == "clad":
        return _run_clad_generate(
            actual_model,
            tokenizer,
            prompt,
            gen_length=gen_length,
            overrides=ov,
            stats_out=stats_out,
        )
    if decode_mode == "clad_v2":
        return _run_clad_v2_generate(
            actual_model,
            tokenizer,
            prompt,
            gen_length=gen_length,
            overrides=ov,
            stats_out=stats_out,
        )
    if decode_mode == "clad_v3":
        return _run_clad_v3_generate(
            actual_model, tokenizer, prompt, gen_length=gen_length
        )
    else:
        raise ValueError(
            f"Unsupported decode_mode: {decode_mode}. "
            f"Supported modes: ['baseline', 'lopa', 'ccd', 'clad', 'clad_v2', 'clad_v3']"
        )


def _run_baseline_generate(
    model, tokenizer, prompt: str, gen_length: int = 2048
) -> Tuple[str, int]:
    """Baseline LLaDA2.1-mini 解码策略。forward 次数通过对主干 ``model.model`` 注册 hook 统计。"""
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
    input_ids = input_ids.long()

    # 将输入移动到模型设备
    main_device = None
    try:
        main_device = next(model.parameters()).device
    except StopIteration:
        main_device = None
    if main_device is not None and main_device.type != "cpu":
        input_ids = input_ids.to(main_device)

    inner = getattr(model, "model", None)
    forward_count = [0]
    hook_handle = None
    if inner is not None:

        def _count_forward(_module, _inp, _out):
            forward_count[0] += 1

        hook_handle = inner.register_forward_hook(_count_forward)

    try:
        with torch.no_grad():
            generated_tokens = model.generate(
                inputs=input_ids,
                eos_early_stop=True,
                gen_length=gen_length,
                block_length=32,
                threshold=0.7,
                editing_threshold=0.5,
                temperature=0.0,
                max_post_steps=16,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return text.strip(), forward_count[0]


def _run_lopa_generate(
    model, tokenizer, prompt: str, gen_length: int = 2048
) -> Tuple[str, int]:
    """LoPA (Lookahead Parallel Decoding) 解码策略"""
    lopa_config = LopaConfig(
        num_lookahead=3,
        branch_confidence_mode="avg_prob",
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )

    text, n_fw = generate_with_lopa(model, tokenizer, prompt, lopa_config)
    return text, n_fw


def _run_ccd_generate(
    model, tokenizer, prompt: str, gen_length: int = 2048
) -> Tuple[str, int]:
    """CCD (Coherent and Consistent Decoding) 双步一致性加速解码策略"""
    ccd_config = CcdConfig(
        seek_mode=True,
        history_depth=2,
        top_v=4,
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    text, n_fw = generate_with_ccd(model, tokenizer, prompt, ccd_config)
    return text, n_fw


def _run_clad_generate(
    model,
    tokenizer,
    prompt: str,
    gen_length: int = 2048,
    overrides: Optional[Dict] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """CLAD v1（Consistency-guided Lookahead Adaptive Decoding）"""
    ov = overrides or {}
    clad_config = CladConfig(
        top_v=ov.get("top_v", 4),
        num_lookahead=ov.get("num_lookahead", 2),
        consistency_weight=ov.get("consistency_weight", 0.6),
        lookahead_warmup=ov.get("lookahead_warmup", 3),
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    text, n_fw = generate_with_clad(
        model, tokenizer, prompt, clad_config, stats_out=stats_out
    )
    return text, n_fw


def _run_clad_v2_generate(
    model,
    tokenizer,
    prompt: str,
    gen_length: int = 2048,
    overrides: Optional[Dict] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """CLAD v2：O1 信息密度加权一致性评分 + O2 多 token 自适应接受"""
    ov = overrides or {}
    clad_v2_config = CladV2Config(
        top_v=ov.get("top_v", 4),
        num_lookahead=ov.get("num_lookahead", 2),
        consistency_weight=ov.get("consistency_weight", 0.5),
        entropy_weight=ov.get("entropy_weight", 0.2),
        lookahead_warmup=ov.get("lookahead_warmup", 3),
        accept_threshold2=ov.get("accept_threshold2", 0.90),
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    text, n_fw = generate_with_clad_v2(
        model, tokenizer, prompt, clad_v2_config, stats_out=stats_out
    )
    return text, n_fw


def _run_clad_v3_generate(
    model, tokenizer, prompt: str, gen_length: int = 2048
) -> Tuple[str, int]:
    """CLAD v3：O1/O2 + O3 批量 Phase-2 forward + O4 级联草稿（top-2 L1 + L2 二次接受）。"""
    cfg = CladV3Config(
        top_v=4,
        num_lookahead=2,
        consistency_weight=0.5,
        entropy_weight=0.2,
        lookahead_warmup=3,
        accept_threshold2=0.90,
        use_batched_phase2=True,
        use_cascaded_draft=True,
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    return generate_with_clad_v3(model, tokenizer, prompt, cfg)


# ---------------------------------------------------------------------------
# 5. 针对不同 benchmark 的适配
# ---------------------------------------------------------------------------


def iter_gsm8k_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "math" / "gsm8k_small.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_gsm8k_test_only_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "math" / "gsm8k_testOnly.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}。请先运行: python dlm/src/download_benchmarks.py --dataset gsm8k_test_only"
        )
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_aime2025_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "math" / "aime2025_all.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_math500_examples(max_examples: int | None = None):
    """Hendrycks MATH 测试集 500 题（download_benchmarks.py --dataset math500）。"""
    path = BENCH_ROOT / "math" / "math500.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}。请先运行: python dlm/src/download_benchmarks.py --dataset math500"
        )
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_humaneval_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "code" / "humaneval_all.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_mbpp_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "code" / "mbpp_sanitized.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_arc_easy_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "reasoning" / "arc_easy_300.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_arc_challenge_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "reasoning" / "arc_challenge_300.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def _format_arc_prompt(ex: dict) -> str:
    """
    将 ARC 样本格式化为模型输入 prompt。
    要求模型给出完整的分析过程，并在最后明确指出答案字母，
    便于后续 LLM Judge 判断正确性，同时保留推理过程供错误分析。
    """
    choices_text = "\n".join(f"  {c['label']}. {c['text']}" for c in ex["choices"])
    return (
        "Answer the following multiple-choice science question. "
        "Think through the problem step by step, then clearly state your final answer "
        "as one of the options (A, B, C, or D).\n\n"
        f"Question: {ex['question']}\n\n"
        f"Choices:\n{choices_text}"
    )


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="使用 LLaDA2.1-mini 在本地 benchmarks 上跑解码（baseline）"
    )
    _all_benchmarks = [
        "gsm8k_small",
        "gsm8k_test_only",
        "math500",
        "aime2025_all",
        "humaneval_all",
        "mbpp_sanitized",
        "arc_easy",
        "arc_challenge",
    ]
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        choices=_all_benchmarks,
        help="要跑的基准数据集标识",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=_all_benchmarks,
        help="要跑的多个基准数据集（用空格分隔）",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="baseline",
        choices=["baseline", "lopa", "ccd", "clad", "clad_v2", "clad_v3"],
        help="解码策略：baseline / lopa / ccd / clad(v1) / clad_v2(O1+O2) / clad_v3(O1+O2+O3+O4)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="最多跑多少条样本（默认全部）",
    )
    # ── CLAD 超参数覆盖（消融实验用）────────────────────────────────────────
    parser.add_argument(
        "--clad_k",
        type=int,
        default=None,
        metavar="K",
        help="覆盖 CLAD v1/v2 的 num_lookahead（前瞻分支数，默认 v1=2 / v2=2）",
    )
    parser.add_argument(
        "--clad_alpha",
        type=float,
        default=None,
        metavar="α",
        help="覆盖 CLAD v1/v2 的 consistency_weight α（默认 v1=0.6 / v2=0.5）",
    )
    parser.add_argument(
        "--clad_beta",
        type=float,
        default=None,
        metavar="β",
        help="覆盖 CLAD v2 的 entropy_weight β（默认 0.2；v1 无此参数）",
    )
    parser.add_argument(
        "--clad_threshold2",
        type=float,
        default=None,
        metavar="thr2",
        help="覆盖 CLAD v2 的 accept_threshold2 O2 阈值（默认 0.90；1.01 可禁用 O2）",
    )
    parser.add_argument(
        "--resume_file",
        type=str,
        default=None,
        help="续跑已有 jsonl 结果文件：读取已完成样本 id，跳过后继续 append 到该文件",
    )
    args = parser.parse_args()

    # 构建超参数覆盖字典（仅含显式传入的参数）
    clad_overrides: Dict = {}
    if args.clad_k is not None:
        clad_overrides["num_lookahead"] = args.clad_k
    if args.clad_alpha is not None:
        clad_overrides["consistency_weight"] = args.clad_alpha
    if args.clad_beta is not None:
        clad_overrides["entropy_weight"] = args.clad_beta
    if args.clad_threshold2 is not None:
        clad_overrides["accept_threshold2"] = args.clad_threshold2

    # 参数验证：必须指定 --benchmark 或 --benchmarks 中的一个
    if not args.benchmark and not args.benchmarks:
        parser.error("必须指定 --benchmark 或 --benchmarks 中的一个")
    if args.benchmark and args.benchmarks:
        parser.error("不能同时指定 --benchmark 和 --benchmarks")
    if args.resume_file and args.benchmarks:
        parser.error("--resume_file 目前仅支持与单个 --benchmark 一起使用")

    # 确定要处理的 benchmark 列表
    if args.benchmark:
        benchmarks_to_run = [args.benchmark]
    else:
        benchmarks_to_run = args.benchmarks

    print(f"[main] 将依次处理以下 benchmarks: {benchmarks_to_run}")

    # 一次性加载模型，多个 benchmark 共用
    model, tokenizer = load_llada_model_and_tokenizer()

    total_processed = 0

    # 循环处理每个 benchmark
    for benchmark_name in benchmarks_to_run:
        print(f"\n{'='*60}")
        print(f"开始处理 benchmark: {benchmark_name}")
        print(f"{'='*60}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 若有超参数覆盖，在文件名中追加 tag，便于区分消融实验结果
        _ov_tag = ""
        if clad_overrides and args.decode_mode in ("clad", "clad_v2"):
            parts = []
            if "num_lookahead" in clad_overrides:
                parts.append(f"k{clad_overrides['num_lookahead']}")
            if "consistency_weight" in clad_overrides:
                parts.append(
                    f"a{clad_overrides['consistency_weight']:.2f}".replace(".", "")
                )
            if "entropy_weight" in clad_overrides:
                parts.append(
                    f"b{clad_overrides['entropy_weight']:.2f}".replace(".", "")
                )
            if "accept_threshold2" in clad_overrides:
                parts.append(
                    f"t2{clad_overrides['accept_threshold2']:.2f}".replace(".", "")
                )
            if parts:
                _ov_tag = "_" + "_".join(parts)
        if args.resume_file:
            out_path = Path(args.resume_file)
        else:
            out_path = (
                RUN_ROOT
                / f"{ts}_llada2_{benchmark_name}_decode={args.decode_mode}{_ov_tag}.jsonl"
            )
        print(f"[run] 输出结果将写入: {out_path}")
        if clad_overrides:
            print(f"[run] CLAD 超参数覆盖: {clad_overrides}")

        completed_ids = set()
        if args.resume_file:
            if not out_path.is_file():
                parser.error(f"--resume_file 指向的文件不存在: {out_path}")
            existing_rows = list(read_jsonl(out_path))
            if existing_rows:
                first = existing_rows[0]
                file_benchmark = first.get("benchmark")
                file_decode_mode = first.get("decode_mode")
                if file_benchmark != benchmark_name:
                    parser.error(
                        f"--resume_file benchmark 不匹配：文件中是 {file_benchmark}，当前是 {benchmark_name}"
                    )
                if file_decode_mode != args.decode_mode:
                    parser.error(
                        f"--resume_file decode_mode 不匹配：文件中是 {file_decode_mode}，当前是 {args.decode_mode}"
                    )
                file_overrides = first.get("clad_overrides")
                if (file_overrides or None) != (clad_overrides or None):
                    parser.error(
                        "--resume_file 的 clad_overrides 与当前命令不一致，"
                        f"文件中是 {file_overrides}，当前是 {clad_overrides or None}"
                    )
            completed_ids = load_completed_sample_ids(out_path)
            print(
                f"[resume] 已完成样本数: {len(completed_ids)}，将跳过这些 id 并继续追加写入"
            )

        # 根据 benchmark 名称选择对应的迭代器
        if benchmark_name == "gsm8k_small":
            iterator = iter_gsm8k_examples(args.max_examples)
        elif benchmark_name == "gsm8k_test_only":
            iterator = iter_gsm8k_test_only_examples(args.max_examples)
        elif benchmark_name == "aime2025_all":
            iterator = iter_aime2025_examples(args.max_examples)
        elif benchmark_name == "math500":
            iterator = iter_math500_examples(args.max_examples)
        elif benchmark_name == "humaneval_all":
            iterator = iter_humaneval_examples(args.max_examples)
        elif benchmark_name == "mbpp_sanitized":
            iterator = iter_mbpp_examples(args.max_examples)
        elif benchmark_name == "arc_easy":
            iterator = iter_arc_easy_examples(args.max_examples)
        elif benchmark_name == "arc_challenge":
            iterator = iter_arc_challenge_examples(args.max_examples)
        else:
            print(f"[错误] 未知 benchmark: {benchmark_name}，跳过")
            continue

        # 处理当前 benchmark 的所有样本
        processed = 0
        skipped_completed = 0
        for ex in iterator:
            sample_id = ex.get("id")
            if sample_id in completed_ids:
                skipped_completed += 1
                continue

            # 根据 benchmark 类型获取问题/提示和参考答案
            if benchmark_name in [
                "gsm8k_small",
                "gsm8k_test_only",
                "aime2025_all",
                "math500",
            ]:
                # 数学基准测试：使用 "question" 字段
                q = ex["question"]
                ref_ans = ex.get("answer")
            elif benchmark_name in ["humaneval_all", "mbpp_sanitized"]:
                # 代码基准测试：使用 "prompt" 字段，构造英文代码生成任务
                if benchmark_name == "humaneval_all":
                    q = (
                        "Complete the following Python function. "
                        "Return only valid Python code for the function implementation, without any explanations.\n\n"
                        f"{ex['prompt']}"
                    )
                else:
                    q = (
                        "Write a Python function that solves the following task.\n"
                        f"Task description: {ex['prompt']}\n\n"
                        "Return only the complete Python function implementation, without any explanations."
                    )
                ref_ans = ex.get("reference_code")
            elif benchmark_name in ["arc_easy", "arc_challenge"]:
                # ARC 选择题：格式化题目 + 选项，参考答案为正确选项的文本
                q = _format_arc_prompt(ex)
                ref_ans = ex.get("answer_text")
            else:
                print(f"[错误] 未知 benchmark 类型: {benchmark_name}，跳过样本")
                continue

            print(f"\n[{benchmark_name}|id={sample_id}] 开始生成...")
            # 与真实送入模型的 chat 模板一致（用于统计 prompt token 数）
            chat_inp = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if isinstance(chat_inp, torch.Tensor):
                _inp = chat_inp
            else:
                _inp = chat_inp["input_ids"]
            if not isinstance(_inp, torch.Tensor):
                _inp = torch.tensor(_inp, dtype=torch.long)
            if _inp.dim() == 1:
                _inp = _inp.unsqueeze(0)
            input_token_len = int(_inp.shape[-1])

            # 扩散模型 forward 序列长度 ≈ input + gen_length；2048 为默认生成空间。
            gen_length = 2048

            t_start = time.time()
            _sample_stats: List = []
            gen_text, forward_count = run_generate_single(
                model,
                tokenizer,
                prompt=q,
                decode_mode=args.decode_mode,
                gen_length=gen_length,
                clad_overrides=clad_overrides if clad_overrides else None,
                stats_out=(
                    _sample_stats if args.decode_mode in ("clad", "clad_v2") else None
                ),
            )
            gen_time_sec = float(time.time() - t_start)

            # 统计输出 token 长度（只对生成的文本再次分词，评估解码效率）
            try:
                output_ids_for_len = tokenizer(
                    gen_text, add_special_tokens=True, return_tensors="pt"
                )["input_ids"]
                output_token_len = int(output_ids_for_len.shape[-1])
            except Exception:
                # 兜底：如果分词失败，长度记为 -1
                output_token_len = -1

            print(
                f"[生成完成] time={gen_time_sec:.3f}s, "
                f"in_len={input_token_len}, out_len={output_token_len}, "
                f"forwards={forward_count} | "
                f"{gen_text[:120].replace(chr(10), ' ')}{'...' if len(gen_text) > 120 else ''}"
            )

            # 构造输出记录，根据基准测试类型使用不同的字段名
            record = {
                "id": sample_id,
                "benchmark": benchmark_name,
                "decode_mode": args.decode_mode,
                "model_answer": gen_text,
                # 供评测脚本使用的效率指标
                "input_token_len": input_token_len,
                "output_token_len": output_token_len,
                "gen_time_sec": gen_time_sec,
                "gen_length": gen_length,
                "forward_count": forward_count,
            }
            # 若有超参数覆盖，记录实际使用的参数（消融实验追溯用）
            if clad_overrides:
                record["clad_overrides"] = clad_overrides
            # 写入 CLAD 解码阶段命中率（由 DecodeStats 提供）
            if _sample_stats:
                record.update(_sample_stats[0].hit_rates())

            # 根据基准测试类型添加特定字段
            if benchmark_name in [
                "gsm8k_small",
                "gsm8k_test_only",
                "aime2025_all",
                "math500",
            ]:
                # 数学基准测试：保留原有字段名
                record["question"] = ex["question"]
                record["reference_answer"] = ref_ans
                if benchmark_name == "math500":
                    # 供 level-5 子集等分层分析
                    if ex.get("level") is not None:
                        record["level"] = ex["level"]
                    if ex.get("subject") is not None:
                        record["subject"] = ex["subject"]
            elif benchmark_name in ["humaneval_all", "mbpp_sanitized"]:
                # 代码基准测试：使用更具描述性的字段名
                record["prompt"] = ex["prompt"]  # 原始任务描述
                record["formatted_question"] = q  # 我们构造的完整提示
                record["reference_code"] = ref_ans  # 参考代码实现
                # 保存测试用例，供后续代码执行评估使用
                if benchmark_name == "humaneval_all":
                    record["tests"] = ex["tests"]
                    record["entry_point"] = ex["entry_point"]
                else:  # mbpp_sanitized
                    record["tests"] = ex["tests"]
            elif benchmark_name in ["arc_easy", "arc_challenge"]:
                # ARC 选择题：保存题目、选项、正确答案 key 和文本
                record["question"] = ex["question"]
                record["choices"] = ex["choices"]
                record["answer_key"] = ex.get("answer_key")
                record["reference_answer"] = ref_ans  # 正确选项的文本，供 judge 评测
            append_jsonl(out_path, [record])
            processed += 1

        print(
            f"[{benchmark_name}] 完成，新增样本数: {processed}，"
            f"跳过已完成样本数: {skipped_completed}，结果保存在: {out_path}"
        )
        total_processed += processed

    print(f"\n[全部完成] 总共处理样本数: {total_processed}")
    print("所有结果文件已保存在 dlm/experiments/runs/ 目录下")


if __name__ == "__main__":
    main()
