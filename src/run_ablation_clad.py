"""
CLAD 超参数消融实验脚本
=======================
一次加载模型，按消融分组顺序扫描配置点，统计准确率、吞吐与阶段命中率。

消融分组（--group）：
  threshold2   CLAD v2  accept_threshold2 ∈ {0.70, 0.80, 0.85, 0.90*, 0.95, 1.01}
  k            CLAD v1 & v2  num_lookahead ∈ {0, 1, 2*, 3, 4}
  alpha        CLAD v2  consistency_weight ∈ {0.0, 0.2, 0.4, 0.5*, 0.6, 0.8}
  beta         CLAD v2  entropy_weight ∈ {0.0, 0.1, 0.2*, 0.3, 0.4}
  all          以上全部

（* 标注为当前论文基准值）

用法示例：
    cd /home/qinghua/yangjh
    python dlm/src/run_ablation_clad.py \\
        --group threshold2 --benchmark gsm8k --num_samples 50
    python dlm/src/run_ablation_clad.py \\
        --group all --benchmark gsm8k --num_samples 50 \\
        --output_dir dlm/experiments/ablation

输出：每个配置点一行 JSON，汇总文件写入 output_dir/ablation_<group>_<benchmark>_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ── GPU 配置（与主脚本保持一致）──────────────────────────────────────────────
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/etc/moreh/checkpoint/huggingface_hub")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "7")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_clad_decode import (
    CladConfig as CladV1Config,
    DecodeStats as DecodeStatsV1,
    generate_with_clad as generate_with_clad_v1,
)
from llada_clad_v2_decode import (
    CladConfig as CladV2Config,
    DecodeStats as DecodeStatsV2,
    generate_with_clad as generate_with_clad_v2,
)

# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "GSAI-ML/LLaDA2.1-mini-Instruct"
ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT / "experiments" / "benchmarks"

# ─────────────────────────────────────────────────────────────────────────────
# 消融分组定义
# ─────────────────────────────────────────────────────────────────────────────

# CLAD v2 基准配置（O1 + O2，使用原生 llada_clad_v2_decode.CladConfig）
_V2_BASE = dict(
    top_v=4,
    num_lookahead=2,
    consistency_weight=0.5,  # α
    entropy_weight=0.2,  # β（O1）
    lookahead_warmup=3,
    accept_threshold2=0.90,  # O2
    block_length=32,
    threshold=0.7,
    editing_threshold=0.5,
    temperature=0.0,
    max_post_steps=16,
    eos_early_stop=True,
)

# CLAD v1 基准配置（使用原生 llada_clad_decode.CladConfig）
_V1_BASE = dict(
    top_v=4,
    num_lookahead=2,
    consistency_weight=0.6,
    lookahead_warmup=3,
    block_length=32,
    threshold=0.7,
    editing_threshold=0.5,
    temperature=0.0,
    max_post_steps=16,
    eos_early_stop=True,
)


def _ablation_groups() -> Dict[str, List[dict]]:
    """返回每个分组的 config 点列表，每条 dict 含 label + override 字段。"""
    groups: Dict[str, List[dict]] = {}

    # ── threshold2（O2 激进程度）───────────────────────────────────────────
    groups["threshold2"] = []
    for thr2 in [0.70, 0.80, 0.85, 0.90, 0.95, 1.01]:
        cfg = dict(_V2_BASE, accept_threshold2=thr2)
        label = f"v2_thr2={thr2:.2f}"
        groups["threshold2"].append({"label": label, "mode": "clad_v2", "config": cfg})

    # ── k（前瞻分支数）────────────────────────────────────────────────────
    groups["k"] = []
    for k_val in [0, 1, 2, 3, 4]:
        # v1
        cfg_v1 = dict(_V1_BASE, num_lookahead=k_val)
        groups["k"].append(
            {"label": f"v1_k={k_val}", "mode": "clad_v1", "config": cfg_v1}
        )
        # v2
        cfg_v2 = dict(_V2_BASE, num_lookahead=k_val)
        groups["k"].append(
            {"label": f"v2_k={k_val}", "mode": "clad_v2", "config": cfg_v2}
        )

    # ── alpha（一致性权重）────────────────────────────────────────────────
    groups["alpha"] = []
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]:
        cfg = dict(_V2_BASE, consistency_weight=alpha)
        label = f"v2_alpha={alpha:.1f}"
        groups["alpha"].append({"label": label, "mode": "clad_v2", "config": cfg})

    # ── beta（熵降权重）───────────────────────────────────────────────────
    groups["beta"] = []
    for beta in [0.0, 0.1, 0.2, 0.3, 0.4]:
        cfg = dict(_V2_BASE, entropy_weight=beta)
        label = f"v2_beta={beta:.1f}"
        groups["beta"].append({"label": label, "mode": "clad_v2", "config": cfg})

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_path: Optional[str] = None):
    mid = model_path or MODEL_ID
    if not torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True, device_map=None
        )
        model = model.to(torch.bfloat16).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 读取
# ─────────────────────────────────────────────────────────────────────────────

_BENCH_FILES = {
    "gsm8k": BENCH_ROOT / "math" / "gsm8k_small.jsonl",
    "arc_challenge": BENCH_ROOT / "arc" / "arc_challenge_all.jsonl",
    "math500": BENCH_ROOT / "math" / "math500.jsonl",
}

_QUESTION_FIELD = {
    "gsm8k": "question",
    "arc_challenge": "question",
    "math500": "question",
}

_ANSWER_FIELD = {
    "gsm8k": "answer",
    "arc_challenge": "answer",
    "math500": "answer",
}


def iter_benchmark(benchmark: str, max_n: int) -> Iterable[dict]:
    bench_path = _BENCH_FILES.get(benchmark)
    if bench_path is None or not bench_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_path}")
    count = 0
    with bench_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if count >= max_n:
                break


# ─────────────────────────────────────────────────────────────────────────────
# 答案提取与比较
# ─────────────────────────────────────────────────────────────────────────────


def extract_answer(text: str) -> Optional[str]:
    """快速启发式答案提取（不依赖 LLM judge）。
    适合 GSM8K / MATH 风格答案，仅用于消融实验内部对比。"""
    # \boxed{...}（MATH500 格式）
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    # "The answer is X" / "答案是 X"
    m = re.search(r"(?:the answer is|答案[是为：:]\s*)([^\n.。]+)", text, re.I)
    if m:
        return m.group(1).strip()
    # "#### X" （GSM8K 官方格式）
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # 最后一个数字
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


def _normalize_num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def answers_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    pred = pred.strip().rstrip(".")
    gold = gold.strip().rstrip(".")
    if pred == gold:
        return True
    p, g = _normalize_num(pred), _normalize_num(gold)
    if p is not None and g is not None:
        return abs(p - g) < 1e-4
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 单个配置点的推理循环
# ─────────────────────────────────────────────────────────────────────────────


def run_config_point(
    model,
    tokenizer,
    mode: str,
    config_kwargs: dict,
    samples: List[dict],
    benchmark: str,
    gen_length: int = 2048,
) -> dict:
    """对给定 config 运行所有 samples，返回汇总指标字典。"""
    q_field = _QUESTION_FIELD[benchmark]
    a_field = _ANSWER_FIELD[benchmark]

    n_correct = 0
    total_time = 0.0
    total_tokens = 0
    correct_tokens = 0

    all_stats: List = []

    for idx, sample in enumerate(samples):
        question = sample[q_field]
        gold = str(sample[a_field]).strip()

        t0 = time.time()
        try:
            stats_holder: List = []
            if mode == "clad_v1":
                cfg = CladV1Config(
                    gen_length=gen_length,
                    **{
                        k: v
                        for k, v in config_kwargs.items()
                        if k in CladV1Config.__dataclass_fields__
                    },
                )
                text, _ = generate_with_clad_v1(
                    model, tokenizer, question, cfg, stats_out=stats_holder
                )
            else:  # clad_v2
                cfg = CladV2Config(
                    gen_length=gen_length,
                    **{
                        k: v
                        for k, v in config_kwargs.items()
                        if k in CladV2Config.__dataclass_fields__
                    },
                )
                text, _ = generate_with_clad_v2(
                    model, tokenizer, question, cfg, stats_out=stats_holder
                )
        except Exception as e:
            print(f"  [WARN] sample {idx} failed: {e}")
            text = ""
            stats_holder = []

        elapsed = time.time() - t0
        total_time += elapsed

        # tokenize 计算 token 数
        with torch.no_grad():
            out_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        n_tok = out_ids.shape[1]
        total_tokens += n_tok

        pred = extract_answer(text)
        correct = answers_match(pred, gold)
        if correct:
            n_correct += 1
            correct_tokens += n_tok

        if stats_holder:
            all_stats.append(stats_holder[0])

        print(
            f"  [{idx+1}/{len(samples)}] correct={correct} "
            f"pred={repr(pred)[:30]} gold={repr(gold)[:20]} "
            f"t={elapsed:.1f}s"
        )

    n = len(samples)
    thr = total_tokens / total_time if total_time > 0 else 0.0
    cthr = correct_tokens / total_time if total_time > 0 else 0.0

    # 聚合阶段命中率
    hit = {}
    if all_stats:
        keys = list(all_stats[0].hit_rates().keys())
        for k in keys:
            hit[k] = sum(s.hit_rates()[k] for s in all_stats) / len(all_stats)

    return {
        "n_correct": n_correct,
        "n_total": n,
        "accuracy": round(100.0 * n_correct / n, 2) if n else 0.0,
        "total_time_sec": round(total_time, 2),
        "avg_gen_time_sec": round(total_time / n, 3) if n else 0.0,
        "throughput_tok_per_sec": round(thr, 3),
        "correct_throughput": round(cthr, 3),
        **{k: round(v, 4) for k, v in hit.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CLAD 超参数消融实验")
    parser.add_argument(
        "--group",
        choices=["threshold2", "k", "alpha", "beta", "all"],
        default="threshold2",
        help="消融分组",
    )
    parser.add_argument(
        "--benchmark",
        choices=["gsm8k", "arc_challenge", "math500"],
        default="gsm8k",
        help="评测 benchmark",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="每个配置点使用的样本数"
    )
    parser.add_argument(
        "--gen_length", type=int, default=512, help="生成区最大 token 数"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="模型路径（默认从 HF Hub 加载）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "experiments" / "ablation"),
        help="输出目录",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"[ablation] group={args.group} benchmark={args.benchmark} n={args.num_samples}"
    )
    print("[ablation] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    print(f"[ablation] Loading {args.num_samples} samples from {args.benchmark}...")
    samples = list(iter_benchmark(args.benchmark, args.num_samples))
    print(f"[ablation] Loaded {len(samples)} samples")

    all_groups = _ablation_groups()
    groups_to_run = list(all_groups.keys()) if args.group == "all" else [args.group]

    for group_name in groups_to_run:
        config_points = all_groups[group_name]
        group_results = []

        print(f"\n{'='*60}")
        print(f"[ablation] Group: {group_name} ({len(config_points)} config points)")
        print(f"{'='*60}")

        for cp in config_points:
            label = cp["label"]
            mode = cp["mode"]
            cfg_kwargs = cp["config"]
            print(f"\n[ablation] >> {label}  (mode={mode})")

            metrics = run_config_point(
                model,
                tokenizer,
                mode=mode,
                config_kwargs=cfg_kwargs,
                samples=samples,
                benchmark=args.benchmark,
                gen_length=args.gen_length,
            )

            row = {
                "label": label,
                "mode": mode,
                "config": cfg_kwargs,
                **metrics,
            }
            group_results.append(row)

            # 实时打印汇总
            print(
                f"  >> acc={metrics['accuracy']}%  "
                f"thr={metrics['throughput_tok_per_sec']:.2f}  "
                f"cthr={metrics['correct_throughput']:.2f}  "
                f"phase1={metrics.get('phase1_hit_rate', 'N/A'):.3f}  "
                f"phase2={metrics.get('phase2_accepted_rate', 'N/A'):.3f}"
            )

        # 写入文件
        out_file = out_dir / f"ablation_{group_name}_{args.benchmark}_{ts}.json"
        payload = {
            "group": group_name,
            "benchmark": args.benchmark,
            "num_samples": len(samples),
            "gen_length": args.gen_length,
            "timestamp": ts,
            "results": group_results,
        }
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[ablation] Saved: {out_file}")

        # 打印 ASCII 汇总表
        _print_summary_table(group_name, group_results)

    print("\n[ablation] Done.")


def _print_summary_table(group_name: str, results: List[dict]):
    """在终端打印简洁的对比表格。"""
    header = f"{'Label':<22} {'Mode':<9} {'Acc%':>6} {'Thr':>7} {'cThr':>7} {'P1%':>6} {'P2%':>6} {'P3%':>6}"
    sep = "-" * len(header)
    print(f"\n── {group_name} ──────────────────────────────────")
    print(header)
    print(sep)
    for r in results:
        p1 = r.get("phase1_hit_rate", float("nan"))
        p2 = r.get("phase2_accepted_rate", float("nan"))
        p3 = r.get("phase3_fallback_rate", float("nan"))
        print(
            f"{r['label']:<22} {r['mode']:<9} "
            f"{r['accuracy']:>6.1f} "
            f"{r['throughput_tok_per_sec']:>7.2f} "
            f"{r['correct_throughput']:>7.2f} "
            f"{p1*100:>5.1f}% "
            f"{p2*100:>5.1f}% "
            f"{p3*100:>5.1f}%"
        )
    print(sep)


if __name__ == "__main__":
    main()
