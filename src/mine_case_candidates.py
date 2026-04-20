"""
CLAD case candidate miner.

Given baseline / CLAD v1 / CLAD v2 run files for the same benchmark,
this script reuses the Judge pipeline to score per-sample correctness and
then mines shortlist candidates for the thesis case studies.

Typical usage:
    python dlm/src/mine_case_candidates.py \
        --benchmark gsm8k_test_only \
        --baseline_file dlm/experiments/runs/...baseline.jsonl \
        --v1_file dlm/experiments/runs/...clad.jsonl \
        --v2_file dlm/experiments/runs/...clad_v2.jsonl \
        --judge_model_path /etc/moreh/checkpoint/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from evaluate_benchmark_results import (
    DEFAULT_JUDGE_MODEL_PATH,
    _judge_single_sample,
    _load_judge_model,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "experiments" / "case_candidates"


def _load_jsonl(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"{path} line {line_no}: missing sample id")
            records[sample_id] = row
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _shared_ids(*record_sets: Dict[str, dict]) -> List[str]:
    shared = set(record_sets[0].keys())
    for rs in record_sets[1:]:
        shared &= set(rs.keys())
    return sorted(shared)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _throughput(sample: dict) -> float:
    total_time = max(_safe_float(sample.get("gen_time_sec")), 1e-8)
    return _safe_float(sample.get("output_token_len")) / total_time


def _phase_rate(sample: dict, key: str) -> float:
    return _safe_float(sample.get(key), 0.0)


def _maybe_existing_correct(sample: dict) -> Optional[bool]:
    for key in ("judge_correct", "correct"):
        if key in sample:
            return bool(sample[key])
    return None


def _judge_correctness(
    sample: dict,
    judge_model,
    judge_tokenizer,
    max_new_tokens: int,
) -> bool:
    cached = _maybe_existing_correct(sample)
    if cached is not None:
        return cached
    judged = _judge_single_sample(
        judge_model,
        judge_tokenizer,
        sample,
        max_new_tokens=max_new_tokens,
    )
    return bool(judged.get("correct", False))


def _strategy_summary(name: str, sample: dict, correct: bool) -> dict:
    has_phase_stats = any(
        key in sample
        for key in (
            "phase1_hit_rate",
            "phase2_trigger_rate",
            "phase2_accepted_rate",
            "o2_hit_rate",
            "phase3_fallback_rate",
        )
    )
    return {
        "strategy": name,
        "decode_mode": sample.get("decode_mode"),
        "correct": bool(correct),
        "gen_time_sec": _safe_float(sample.get("gen_time_sec")),
        "output_token_len": int(_safe_float(sample.get("output_token_len"))),
        "throughput_tok_per_sec": _throughput(sample),
        "phase1_hit_rate": _phase_rate(sample, "phase1_hit_rate"),
        "phase2_trigger_rate": _phase_rate(sample, "phase2_trigger_rate"),
        "phase2_accepted_rate": _phase_rate(sample, "phase2_accepted_rate"),
        "o2_hit_rate": _phase_rate(sample, "o2_hit_rate"),
        "phase3_fallback_rate": _phase_rate(sample, "phase3_fallback_rate"),
        "has_phase_stats": has_phase_stats,
        "model_answer_preview": (sample.get("model_answer") or "")[:240],
    }


def _rel_speedup(reference_time: float, candidate_time: float) -> float:
    if reference_time <= 0:
        return 0.0
    return (reference_time - candidate_time) / reference_time


def _question_preview(sample: dict, limit: int = 220) -> str:
    text = sample.get("question") or sample.get("prompt") or ""
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _base_meta(sample: dict, benchmark: str) -> dict:
    meta = {
        "id": sample.get("id"),
        "benchmark": benchmark,
        "question_preview": _question_preview(sample),
        "reference_answer": sample.get("reference_answer"),
    }
    for key in ("level", "subject", "answer_key", "choices"):
        if key in sample:
            meta[key] = sample.get(key)
    return meta


def _build_case1(sample: dict, baseline: dict, v1: dict, v2: dict) -> Optional[dict]:
    if not (baseline["correct"] and v1["correct"]):
        return None
    v2_degrades = (not v2["correct"]) or (
        v2["gen_time_sec"] > v1["gen_time_sec"] * 1.05
    )
    if not v2_degrades:
        return None

    score = 0.0
    score += 3.0 if not v2["correct"] else 0.0
    score += 1.5 * max(0.0, _rel_speedup(baseline["gen_time_sec"], v1["gen_time_sec"]))
    score += 1.5 * max(0.0, _rel_speedup(v2["gen_time_sec"], v1["gen_time_sec"]))
    score += 1.0 * v1["phase1_hit_rate"]
    score -= 0.5 * v2["phase2_accepted_rate"]
    score -= 0.5 * v2["o2_hit_rate"]

    return {
        "case_type": "case1_gsm8k_v1_advantage",
        "rank_score": round(score, 6),
        "selection_reason": (
            "v1 stays correct while baseline is slower and v2 either slows down "
            "or becomes incorrect; favors strong Phase-1 contribution."
        ),
    }


def _build_case2(sample: dict, baseline: dict, v1: dict, v2: dict) -> Optional[dict]:
    if not v2["correct"]:
        return None
    has_gain = (
        (not baseline["correct"])
        or (not v1["correct"])
        or (
            v2["gen_time_sec"]
            < min(baseline["gen_time_sec"], v1["gen_time_sec"]) * 0.95
        )
    )
    if not has_gain:
        return None
    has_phase_stats = bool(v2.get("has_phase_stats", False))
    if has_phase_stats and v2["phase2_accepted_rate"] <= 0 and v2["o2_hit_rate"] <= 0:
        return None

    score = 0.0
    score += 2.0 if not baseline["correct"] else 0.0
    score += 2.0 if not v1["correct"] else 0.0
    score += 1.5 * max(
        0.0,
        _rel_speedup(
            min(baseline["gen_time_sec"], v1["gen_time_sec"]), v2["gen_time_sec"]
        ),
    )
    if has_phase_stats:
        score += 1.2 * v2["phase2_accepted_rate"]
        score += 1.0 * v2["o2_hit_rate"]

    return {
        "case_type": "case2_arc_v2_advantage",
        "rank_score": round(score, 6),
        "selection_reason": (
            "v2 delivers accuracy and/or latency gains with visible Phase-2/O2 activity."
            if has_phase_stats
            else "v2 delivers accuracy and/or latency gains, but this run lacks phase-rate "
            "fields; shortlist first, then verify Phase-2/O2 by case replay."
        ),
        "phase_signal_mode": (
            "strict" if has_phase_stats else "relaxed_missing_phase_stats"
        ),
    }


def _build_case3(sample: dict, baseline: dict, v1: dict, v2: dict) -> Optional[dict]:
    if not v1["correct"]:
        return None
    v2_degrades = (not v2["correct"]) or (
        v2["gen_time_sec"] > v1["gen_time_sec"] * 1.10
    )
    if not v2_degrades:
        return None
    if v2["phase2_accepted_rate"] <= 0 and v2["o2_hit_rate"] <= 0:
        return None

    score = 0.0
    score += 3.0 if not v2["correct"] else 0.0
    score += 1.5 * max(0.0, _rel_speedup(v1["gen_time_sec"], baseline["gen_time_sec"]))
    score += 1.5 * max(
        0.0, (v2["gen_time_sec"] - v1["gen_time_sec"]) / max(v2["gen_time_sec"], 1e-8)
    )
    score += 1.0 * v2["phase2_accepted_rate"]
    score += 1.0 * v2["o2_hit_rate"]

    return {
        "case_type": "case3_gsm8k_v2_failure",
        "rank_score": round(score, 6),
        "selection_reason": (
            "v2 participates actively in Phase-2/O2 yet loses to v1 on correctness "
            "or latency, making the failure mechanism analyzable."
        ),
    }


def _build_case4(sample: dict, baseline: dict, v1: dict, v2: dict) -> Optional[dict]:
    if sample.get("level") not in (5, "5", "Level 5", "level5"):
        return None
    all_same_correct = (
        baseline["correct"] == v1["correct"] == v2["correct"] and baseline["correct"]
    )
    if not all_same_correct:
        return None
    fastest_name = min(
        [("baseline", baseline), ("clad_v1", v1), ("clad_v2", v2)],
        key=lambda item: item[1]["gen_time_sec"],
    )[0]
    if fastest_name == "baseline":
        return None

    fastest = {"baseline": baseline, "clad_v1": v1, "clad_v2": v2}[fastest_name]
    score = 2.0 * max(
        0.0,
        _rel_speedup(baseline["gen_time_sec"], fastest["gen_time_sec"]),
    )
    score += 0.6 * fastest["phase1_hit_rate"]
    score += 0.6 * fastest["phase2_accepted_rate"]
    score += 0.4 * fastest["o2_hit_rate"]

    return {
        "case_type": "case4_math_level5_same_accuracy_faster",
        "rank_score": round(score, 6),
        "selection_reason": (
            f"{fastest_name} keeps accuracy unchanged on a hard Level-5 sample but is faster."
        ),
    }


def _mine_case_records(
    benchmark: str, merged_samples: Iterable[dict]
) -> Dict[str, List[dict]]:
    candidates: Dict[str, List[dict]] = {}
    for sample in merged_samples:
        baseline = sample["strategies"]["baseline"]
        v1 = sample["strategies"]["clad_v1"]
        v2 = sample["strategies"]["clad_v2"]

        builders = []
        if benchmark in {"gsm8k_small", "gsm8k_test_only"}:
            builders = [_build_case1, _build_case3]
        elif benchmark == "arc_challenge":
            builders = [_build_case2]
        elif benchmark == "math500":
            builders = [_build_case4]

        for builder in builders:
            case_info = builder(sample["meta"], baseline, v1, v2)
            if case_info is None:
                continue
            record = {
                **case_info,
                "meta": sample["meta"],
                "strategies": sample["strategies"],
            }
            candidates.setdefault(case_info["case_type"], []).append(record)

    for case_type, rows in candidates.items():
        rows.sort(key=lambda row: row["rank_score"], reverse=True)
    return candidates


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _markdown_for_case(case_type: str, rows: List[dict], top_k: int) -> List[str]:
    lines = [f"## {case_type}", ""]
    if not rows:
        lines.extend(["No candidates found.", ""])
        return lines

    lines.append(
        "| rank | sample_id | score | correctness (B/V1/V2) | time_sec (B/V1/V2) | phase signals | question |"
    )
    lines.append("| --- | --- | ---: | --- | --- | --- | --- |")

    for idx, row in enumerate(rows[:top_k], start=1):
        meta = row["meta"]
        b = row["strategies"]["baseline"]
        v1 = row["strategies"]["clad_v1"]
        v2 = row["strategies"]["clad_v2"]
        if v2.get("has_phase_stats", False):
            phase_text = (
                f"v1 P1={_format_pct(v1['phase1_hit_rate'])}; "
                f"v2 P2={_format_pct(v2['phase2_accepted_rate'])}, "
                f"O2={_format_pct(v2['o2_hit_rate'])}"
            )
        else:
            phase_text = "phase stats unavailable in source run"
        lines.append(
            f"| {idx} | {meta['id']} | {row['rank_score']:.3f} | "
            f"{int(b['correct'])}/{int(v1['correct'])}/{int(v2['correct'])} | "
            f"{b['gen_time_sec']:.2f}/{v1['gen_time_sec']:.2f}/{v2['gen_time_sec']:.2f} | "
            f"{phase_text} | {meta['question_preview']} |"
        )
    lines.append("")
    return lines


def _write_outputs(
    output_dir: Path,
    benchmark: str,
    merged_samples: List[dict],
    candidates: Dict[str, List[dict]],
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{ts}_{benchmark}_candidate_shortlist.json"
    md_path = output_dir / f"{ts}_{benchmark}_candidate_shortlist.md"

    payload = {
        "benchmark": benchmark,
        "generated_at": ts,
        "sample_count": len(merged_samples),
        "candidates": candidates,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_lines = [
        f"# Candidate Shortlist: {benchmark}",
        "",
        f"- sample_count: {len(merged_samples)}",
        "",
    ]
    for case_type, rows in candidates.items():
        md_lines.extend(_markdown_for_case(case_type, rows, top_k))

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[done] JSON shortlist written to: {json_path}")
    print(f"[done] Markdown shortlist written to: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mine thesis case-study candidates from baseline / CLAD v1 / CLAD v2 runs."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["gsm8k_small", "gsm8k_test_only", "arc_challenge", "math500"],
    )
    parser.add_argument("--baseline_file", required=True)
    parser.add_argument("--v1_file", required=True)
    parser.add_argument("--v2_file", required=True)
    parser.add_argument("--judge_model_path", default=DEFAULT_JUDGE_MODEL_PATH)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on shared samples, for quick smoke tests.",
    )
    args = parser.parse_args()

    baseline_rows = _load_jsonl(Path(args.baseline_file))
    v1_rows = _load_jsonl(Path(args.v1_file))
    v2_rows = _load_jsonl(Path(args.v2_file))

    sample_ids = _shared_ids(baseline_rows, v1_rows, v2_rows)
    if not sample_ids:
        raise ValueError("No shared sample ids across the three run files.")
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    print(f"[info] shared samples: {len(sample_ids)}")
    judge_model, judge_tokenizer = _load_judge_model(args.judge_model_path)

    merged_samples: List[dict] = []
    for idx, sample_id in enumerate(sample_ids, start=1):
        if idx % 20 == 0 or idx == 1:
            print(f"[judge] processing {idx}/{len(sample_ids)}: {sample_id}")

        b_raw = baseline_rows[sample_id]
        v1_raw = v1_rows[sample_id]
        v2_raw = v2_rows[sample_id]

        b_correct = _judge_correctness(
            b_raw, judge_model, judge_tokenizer, args.max_new_tokens
        )
        v1_correct = _judge_correctness(
            v1_raw, judge_model, judge_tokenizer, args.max_new_tokens
        )
        v2_correct = _judge_correctness(
            v2_raw, judge_model, judge_tokenizer, args.max_new_tokens
        )

        merged_samples.append(
            {
                "meta": _base_meta(b_raw, args.benchmark),
                "strategies": {
                    "baseline": _strategy_summary("baseline", b_raw, b_correct),
                    "clad_v1": _strategy_summary("clad_v1", v1_raw, v1_correct),
                    "clad_v2": _strategy_summary("clad_v2", v2_raw, v2_correct),
                },
            }
        )

    candidates = _mine_case_records(args.benchmark, merged_samples)
    _write_outputs(
        Path(args.out_dir),
        args.benchmark,
        merged_samples,
        candidates,
        args.top_k,
    )


if __name__ == "__main__":
    main()
