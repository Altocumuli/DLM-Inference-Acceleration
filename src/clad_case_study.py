"""
CLAD 个案分析脚本
================
对指定样本分别用 Baseline / CLAD v1 / CLAD v2 三种策略生成，
捕获逐步 print 日志并解析出阶段接受事件，输出 Markdown 对比表格。

用法：
    cd /home/qinghua/yangjh
    # 按下标分析 GSM8K 第 3 条和第 10 条
    python dlm/src/clad_case_study.py --benchmark gsm8k --indices 3 10

    # 按真实 sample_id 直接复盘 shortlist 样本
    python dlm/src/clad_case_study.py --benchmark gsm8k_test_only --sample_id gsm8k_test_42

    # 分析 MATH500 level-5 题目（支持 index 或 sample_id）
    python dlm/src/clad_case_study.py --benchmark math500 --sample_id math500_17 --level5_only

    # 指定输出目录
    python dlm/src/clad_case_study.py --benchmark gsm8k_test_only --sample_id gsm8k_test_42 --out_dir dlm/experiments/case_study

输出：
    <out_dir>/case_<benchmark>_<idx>_<ts>.json    -- 完整逐步日志 + 最终文本
    <out_dir>/case_<benchmark>_<idx>_<ts>.md      -- Markdown 对比表格（可直接粘贴到论文附录）
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/etc/moreh/checkpoint/huggingface_hub")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "7")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llada_clad_decode import (
    CladConfig as CladV1Config,
    generate_with_clad as generate_with_clad_v1,
)
from llada_clad_v2_decode import (
    CladConfig as CladV2Config,
    generate_with_clad as generate_with_clad_v2,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT / "experiments" / "benchmarks"

# ─────────────────────────────────────────────────────────────────────────────
# Baseline 生成（直接调用 model.generate）
# ─────────────────────────────────────────────────────────────────────────────

from run_benchmark_llada2 import MODEL_ID, _run_baseline_generate  # type: ignore[import]


# ─────────────────────────────────────────────────────────────────────────────
# 日志解析
# ─────────────────────────────────────────────────────────────────────────────

# 样本 print 行示例:
#   [CLAD] Phase-1 consistency: accepted 3 tokens
#   [CLAD] Phase-2 lookahead: filled pos=5 tok=1234 score=0.712
#   [CLAD v3] Phase-1 consistency: accepted 2 tokens
#   [CLAD v3] Phase-2 (batched bsz=2): best_pos=12 tok=5678 score=0.831
#   [CLAD v3] Phase-2 (serial): pos=3 tok=999 score=0.650

_RE_P1 = re.compile(r"\[CLAD(?: v[23])?\] Phase-1 consistency: accepted (\d+) tokens")
_RE_P2_V1 = re.compile(
    r"\[CLAD\] Phase-2 lookahead: filled pos=(\d+) tok=(\d+) score=([0-9.]+)"
)
_RE_P2_V2 = re.compile(r"\[CLAD v2\] Phase-2: pos=(\d+) tok=(\d+) score=([0-9.]+)")
_RE_P2_V3_BATCH = re.compile(
    r"\[CLAD v3\] Phase-2 \(batched bsz=\d+\): best_pos=(\d+) tok=(\d+) score=([0-9.]+)"
)
_RE_P2_V3_SERIAL = re.compile(
    r"\[CLAD v3\] Phase-2 \(serial\): pos=(\d+) tok=(\d+) score=([0-9.]+)"
)
_RE_BLOCK = re.compile(r"\[CLAD(?: v[123])?\] (?:Processing block|block) (\d+)")


def parse_trace(log: str, strategy: str) -> List[dict]:
    """解析 stdout 日志，返回逐步接受事件列表。"""
    events: List[dict] = []
    cur_block = -1

    for line in log.splitlines():
        m = _RE_BLOCK.search(line)
        if m:
            cur_block = int(m.group(1))
            continue

        m = _RE_P1.search(line)
        if m:
            events.append(
                {
                    "block": cur_block,
                    "phase": "Phase-1",
                    "n_tokens": int(m.group(1)),
                    "pos": None,
                    "score": None,
                }
            )
            continue

        m = (
            _RE_P2_V1.search(line)
            or _RE_P2_V2.search(line)
            or _RE_P2_V3_BATCH.search(line)
            or _RE_P2_V3_SERIAL.search(line)
        )
        if m:
            events.append(
                {
                    "block": cur_block,
                    "phase": "Phase-2",
                    "n_tokens": 1,
                    "pos": int(m.group(1)),
                    "score": float(m.group(3)),
                }
            )
            continue

    return events


def normalize_structured_trace(trace: List[dict]) -> List[dict]:
    """将 decoder 的结构化 trace 规整为表格友好的事件格式。"""
    phase_name = {
        "phase1": "Phase-1",
        "phase2": "Phase-2",
        "phase3": "Phase-3",
    }
    events: List[dict] = []
    for item in trace:
        positions = item.get("accepted_positions") or []
        events.append(
            {
                "block": item.get("block", -1),
                "phase": phase_name.get(item.get("phase"), str(item.get("phase"))),
                "n_tokens": item.get("n_tokens", len(positions) or 0),
                "pos": positions[0] if positions else None,
                "score": item.get("winner_score"),
                "details": item,
            }
        )
    return events


# ─────────────────────────────────────────────────────────────────────────────
# 单个策略运行（捕获 stdout）
# ─────────────────────────────────────────────────────────────────────────────


def run_strategy(
    strategy: str,
    model,
    tokenizer,
    question: str,
    gen_length: int = 512,
) -> Tuple[str, float, List[dict], List[dict]]:
    """运行一个策略，返回 (生成文本, 耗时, 表格事件, 原始结构化 trace)。"""
    buf = io.StringIO()
    structured_trace: List[dict] = []
    t0 = time.time()
    with redirect_stdout(buf):
        if strategy == "baseline":
            text, _ = _run_baseline_generate(
                model, tokenizer, question, gen_length=gen_length
            )
        elif strategy == "clad_v1":
            cfg = CladV1Config(
                top_v=4,
                num_lookahead=2,
                consistency_weight=0.6,
                lookahead_warmup=3,
                gen_length=gen_length,
                block_length=32,
                threshold=0.7,
                editing_threshold=0.5,
                temperature=0.0,
                max_post_steps=16,
                eos_early_stop=True,
            )
            text, _ = generate_with_clad_v1(
                model, tokenizer, question, cfg, trace_out=structured_trace
            )
        elif strategy == "clad_v2":
            cfg = CladV2Config(
                top_v=4,
                num_lookahead=2,
                consistency_weight=0.5,
                entropy_weight=0.2,
                lookahead_warmup=3,
                accept_threshold2=0.90,
                gen_length=gen_length,
                block_length=32,
                threshold=0.7,
                editing_threshold=0.5,
                temperature=0.0,
                max_post_steps=16,
                eos_early_stop=True,
            )
            text, _ = generate_with_clad_v2(
                model, tokenizer, question, cfg, trace_out=structured_trace
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    elapsed = time.time() - t0
    log = buf.getvalue()
    events = (
        normalize_structured_trace(structured_trace)
        if structured_trace
        else parse_trace(log, strategy)
    )
    return text, elapsed, events, structured_trace


# ─────────────────────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────────────────────


def _summarize_events(events: List[dict]) -> dict:
    p1 = sum(1 for e in events if e["phase"] == "Phase-1")
    p1_tok = sum(e["n_tokens"] for e in events if e["phase"] == "Phase-1")
    p2 = sum(1 for e in events if e["phase"] == "Phase-2")
    total = len(events)
    scores = [
        e["score"] for e in events if e["phase"] == "Phase-2" and e["score"] is not None
    ]
    return {
        "total_events": total,
        "phase1_hits": p1,
        "phase1_tokens": p1_tok,
        "phase2_hits": p2,
        "avg_p2_score": round(sum(scores) / len(scores), 4) if scores else None,
    }


def make_markdown(
    sample: dict,
    benchmark: str,
    results: Dict[str, dict],
) -> str:
    question = sample.get("question", sample.get("prompt", ""))
    gold = sample.get("reference_answer", sample.get("answer", ""))

    lines = []
    lines.append("## 个案分析")
    lines.append("")
    lines.append(f"**Benchmark**: {benchmark}")
    lines.append("")
    lines.append("**题目**：")
    lines.append("")
    lines.append(f"> {question[:300]}{'...' if len(question) > 300 else ''}")
    lines.append("")
    lines.append(f"**参考答案**：`{gold}`")
    lines.append("")

    # 策略对比表
    lines.append("### 策略对比")
    lines.append("")
    lines.append(
        "| 策略 | 耗时 (s) | Phase-1 次 | Phase-1 tok | Phase-2 次 | 平均 P2 分数 | 生成摘要 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for strat in ["baseline", "clad_v1", "clad_v2"]:
        r = results.get(strat, {})
        summ = r.get("event_summary", {})
        text_preview = r.get("text", "")[:80].replace("\n", " ") + "..."
        lines.append(
            f"| {strat} "
            f"| {r.get('elapsed', 0):.1f} "
            f"| {summ.get('phase1_hits', '-')} "
            f"| {summ.get('phase1_tokens', '-')} "
            f"| {summ.get('phase2_hits', '-')} "
            f"| {summ.get('avg_p2_score', '-')} "
            f"| {text_preview} |"
        )
    lines.append("")

    # 每个策略的完整输出
    for strat in ["baseline", "clad_v1", "clad_v2"]:
        r = results.get(strat, {})
        lines.append(f"### {strat} — 完整输出")
        lines.append("")
        lines.append("```")
        lines.append(r.get("text", "(empty)"))
        lines.append("```")
        lines.append("")

    # 关键事件序列
    lines.append("### 关键事件序列（前 20 步）")
    lines.append("")
    for strat in ["clad_v1", "clad_v2"]:
        r = results.get(strat, {})
        events = r.get("events", [])[:20]
        lines.append(f"**{strat}**：")
        lines.append("")
        if events:
            lines.append("| Step | Block | Phase | # tok | Pos | Score |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for i, e in enumerate(events):
                lines.append(
                    f"| {i+1} | {e['block']} | {e['phase']} "
                    f"| {e['n_tokens']} | {e.get('pos', '-')} "
                    f"| {e.get('score', '-')} |"
                )
        else:
            lines.append("（无阶段接受事件记录）")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────


def _sample_label(sample: dict, fallback_idx: int) -> str:
    sample_id = sample.get("id")
    return str(sample_id) if sample_id is not None else str(fallback_idx)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def load_samples(
    benchmark: str,
    indices: Optional[List[int]] = None,
    sample_ids: Optional[List[str]] = None,
    level5_only: bool = False,
) -> List[Tuple[int, dict]]:
    bench_files = {
        "gsm8k": BENCH_ROOT / "math" / "gsm8k_small.jsonl",
        "gsm8k_test_only": BENCH_ROOT / "math" / "gsm8k_testOnly.jsonl",
        "arc_challenge": BENCH_ROOT / "arc" / "arc_challenge_all.jsonl",
        "math500": BENCH_ROOT / "math" / "math500.jsonl",
    }
    path = bench_files.get(benchmark)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")

    all_samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if level5_only and obj.get("level") != "Level 5":
                continue
            all_samples.append(obj)

    chosen: List[Tuple[int, dict]] = []
    used = set()

    if sample_ids:
        id_to_sample = {}
        for idx, sample in enumerate(all_samples):
            sid = sample.get("id")
            if sid is not None:
                id_to_sample[str(sid)] = (idx, sample)

        for sid in sample_ids:
            hit = id_to_sample.get(str(sid))
            if hit is None:
                print(f"[WARN] sample_id {sid!r} not found in {path}")
                continue
            idx, sample = hit
            if idx in used:
                continue
            chosen.append((idx, sample))
            used.add(idx)

    for idx in indices or []:
        if idx < len(all_samples):
            if idx in used:
                continue
            chosen.append((idx, all_samples[idx]))
            used.add(idx)
        else:
            print(f"[WARN] index {idx} out of range ({len(all_samples)} samples)")
    return chosen


def main():
    parser = argparse.ArgumentParser(description="CLAD 个案分析")
    parser.add_argument(
        "--benchmark",
        choices=["gsm8k", "gsm8k_test_only", "arc_challenge", "math500"],
        default="gsm8k",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="要分析的样本下标（0-based）",
    )
    parser.add_argument(
        "--sample_id",
        nargs="*",
        default=None,
        help="按 benchmark jsonl 中的真实 id 直接选样，可一次传多个",
    )
    parser.add_argument(
        "--level5_only",
        action="store_true",
        help="仅从 math500 的 Level 5 题目中取样",
    )
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "experiments" / "case_study"),
    )
    args = parser.parse_args()
    if not args.indices and not args.sample_id:
        args.indices = [0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[case_study] Loading model...")
    mid = args.model_path or MODEL_ID
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

    samples = load_samples(
        args.benchmark,
        indices=args.indices,
        sample_ids=args.sample_id,
        level5_only=args.level5_only,
    )
    print(f"[case_study] {len(samples)} samples to analyze")

    strategies = ["baseline", "clad_v1", "clad_v2"]

    for sample_idx, sample in samples:
        question = sample.get("question", sample.get("prompt", ""))
        gold = sample.get("reference_answer", sample.get("answer", ""))
        sample_label = _sample_label(sample, sample_idx)
        print(f"\n{'='*60}")
        print(f"[case_study] Sample #{sample_idx} id={sample_label} gold={gold!r}")
        print(f"Question: {question[:150]}...")
        print("=" * 60)

        results: Dict[str, dict] = {}
        for strat in strategies:
            print(f"\n  >> Running {strat}...")
            text, elapsed, events, structured_trace = run_strategy(
                strat, model, tokenizer, question, gen_length=args.gen_length
            )
            summ = _summarize_events(events)
            print(
                f"     elapsed={elapsed:.1f}s  "
                f"p1_hits={summ['phase1_hits']}  "
                f"p2_hits={summ['phase2_hits']}"
            )
            print(f"     output[:120]: {text[:120]}")
            results[strat] = {
                "text": text,
                "elapsed": round(elapsed, 3),
                "event_summary": summ,
                "events": events,
                "structured_trace": structured_trace,
            }

        # 写 JSON
        stem = _safe_name(sample_label)
        json_path = out_dir / f"case_{args.benchmark}_{stem}_{ts}.json"
        payload = {
            "benchmark": args.benchmark,
            "sample_idx": sample_idx,
            "sample_id": sample.get("id"),
            "question": question,
            "gold_answer": gold,
            "results": results,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 写 Markdown
        md_path = out_dir / f"case_{args.benchmark}_{stem}_{ts}.md"
        md = make_markdown(sample, args.benchmark, results)
        md_path.write_text(md, encoding="utf-8")

        print(f"\n  Saved: {json_path}")
        print(f"  Saved: {md_path}")

    print("\n[case_study] Done.")


if __name__ == "__main__":
    main()
