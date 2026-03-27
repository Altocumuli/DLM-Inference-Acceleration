"""
比较 CCD vs CLAD-v2 在 MATH500 level-5 子集上的表现。

输入：
- 两个 run jsonl（含 gen_time_sec / output_token_len / forward_count / diffusion_steps）
- 两个 correct_id_level jsonl（由 judge_details 解析而来）

输出：
- 控制台摘要
- 一个文本报告
- 一个逐题对照 CSV（便于画图/做误差分析）
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class Metrics:
    total_level5: int
    correct_level5: int
    accuracy: float
    avg_gen_time: float
    avg_output_len: float
    throughput: float
    correct_throughput: float
    avg_forward: float
    total_forward: int
    macro_tpf: float
    mean_tpf: float
    avg_diffusion_steps: float
    total_diffusion_steps: int


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_run_by_id(path: Path) -> Dict[str, Dict]:
    by_id: Dict[str, Dict] = {}
    for row in read_jsonl(path):
        sid = str(row.get("id"))
        by_id[sid] = row
    return by_id


def load_correct_ids_level5(path: Path) -> Set[str]:
    ids: Set[str] = set()
    for row in read_jsonl(path):
        if int(row.get("level", -1)) == 5:
            ids.add(str(row["id"]))
    return ids


def compute_metrics_level5(
    run_by_id: Dict[str, Dict], correct_ids_level5: Set[str]
) -> Metrics:
    level5_rows = [r for r in run_by_id.values() if int(r.get("level", -1)) == 5]
    if not level5_rows:
        raise RuntimeError("run 文件中未找到 level=5 样本，请确认输入文件。")

    total = len(level5_rows)
    correct = 0
    sum_t = 0.0
    sum_out = 0
    sum_correct_out = 0
    sum_fw = 0
    sum_ds = 0
    sum_per_tpf = 0.0
    tpf_count = 0

    for r in level5_rows:
        sid = str(r["id"])
        t = float(r.get("gen_time_sec", 0.0))
        out_len = int(r.get("output_token_len", 0))
        fw = int(r.get("forward_count", 0))
        ds = int(r.get("diffusion_steps", fw))

        is_correct = sid in correct_ids_level5
        if is_correct:
            correct += 1
            sum_correct_out += out_len

        sum_t += t
        sum_out += out_len
        sum_fw += fw
        sum_ds += ds

        if fw > 0:
            sum_per_tpf += out_len / fw
            tpf_count += 1

    accuracy = correct / total
    avg_gen_time = sum_t / total
    avg_output_len = sum_out / total
    throughput = (sum_out / sum_t) if sum_t > 0 else 0.0
    correct_throughput = (sum_correct_out / sum_t) if sum_t > 0 else 0.0
    avg_forward = sum_fw / total
    macro_tpf = (sum_out / sum_fw) if sum_fw > 0 else 0.0
    mean_tpf = (sum_per_tpf / tpf_count) if tpf_count > 0 else 0.0
    avg_ds = sum_ds / total

    return Metrics(
        total_level5=total,
        correct_level5=correct,
        accuracy=accuracy,
        avg_gen_time=avg_gen_time,
        avg_output_len=avg_output_len,
        throughput=throughput,
        correct_throughput=correct_throughput,
        avg_forward=avg_forward,
        total_forward=sum_fw,
        macro_tpf=macro_tpf,
        mean_tpf=mean_tpf,
        avg_diffusion_steps=avg_ds,
        total_diffusion_steps=sum_ds,
    )


def compute_efficiency_on_id_set(run_by_id: Dict[str, Dict], ids: Set[str]) -> Metrics:
    """
    在指定 id 集合上计算效率指标，并把这些 id 视为“共同答对”集合。

    这样可以回答：当两种方法都答对同一批 level-5 题时，谁更高效（吞吐/TPF/扩散步数）。
    """
    valid_ids = [sid for sid in ids if sid in run_by_id]
    if not valid_ids:
        raise RuntimeError("指定 id 集合在 run 文件中为空，请检查输入是否匹配。")

    total = len(valid_ids)
    correct = total

    sum_t = 0.0
    sum_out = 0
    sum_fw = 0
    sum_ds = 0
    sum_per_tpf = 0.0
    tpf_count = 0

    for sid in valid_ids:
        r = run_by_id[sid]
        t = float(r.get("gen_time_sec", 0.0))
        out_len = int(r.get("output_token_len", 0))
        fw = int(r.get("forward_count", 0))
        ds = int(r.get("diffusion_steps", fw))

        sum_t += t
        sum_out += out_len
        sum_fw += fw
        sum_ds += ds

        if fw > 0:
            sum_per_tpf += out_len / fw
            tpf_count += 1

    accuracy = 1.0
    avg_gen_time = sum_t / total
    avg_output_len = sum_out / total
    throughput = (sum_out / sum_t) if sum_t > 0 else 0.0
    # 在“共同答对”子集里，correct_output_len == output_len
    correct_throughput = throughput
    avg_forward = sum_fw / total
    macro_tpf = (sum_out / sum_fw) if sum_fw > 0 else 0.0
    mean_tpf = (sum_per_tpf / tpf_count) if tpf_count > 0 else 0.0
    avg_ds = sum_ds / total

    return Metrics(
        total_level5=total,
        correct_level5=correct,
        accuracy=accuracy,
        avg_gen_time=avg_gen_time,
        avg_output_len=avg_output_len,
        throughput=throughput,
        correct_throughput=correct_throughput,
        avg_forward=avg_forward,
        total_forward=sum_fw,
        macro_tpf=macro_tpf,
        mean_tpf=mean_tpf,
        avg_diffusion_steps=avg_ds,
        total_diffusion_steps=sum_ds,
    )


def build_level5_pair_rows(
    ccd_run: Dict[str, Dict],
    clad_run: Dict[str, Dict],
    ccd_correct_l5: Set[str],
    clad_correct_l5: Set[str],
) -> List[Dict]:
    # 只保留两边都存在且level=5的样本，便于逐题对比
    ids = sorted(set(ccd_run.keys()) & set(clad_run.keys()))
    both_correct_l5 = ccd_correct_l5 & clad_correct_l5
    rows: List[Dict] = []
    for sid in ids:
        r1 = ccd_run[sid]
        r2 = clad_run[sid]
        if int(r1.get("level", -1)) != 5 or int(r2.get("level", -1)) != 5:
            continue
        rows.append(
            {
                "id": sid,
                "subject": r1.get("subject", ""),
                "level": 5,
                "ccd_correct": int(sid in ccd_correct_l5),
                "clad_v2_correct": int(sid in clad_correct_l5),
                "both_correct": int(sid in both_correct_l5),
                "ccd_gen_time_sec": float(r1.get("gen_time_sec", 0.0)),
                "clad_v2_gen_time_sec": float(r2.get("gen_time_sec", 0.0)),
                "ccd_output_len": int(r1.get("output_token_len", 0)),
                "clad_v2_output_len": int(r2.get("output_token_len", 0)),
                "ccd_forward_count": int(r1.get("forward_count", 0)),
                "clad_v2_forward_count": int(r2.get("forward_count", 0)),
                "ccd_diffusion_steps": int(
                    r1.get("diffusion_steps", r1.get("forward_count", 0))
                ),
                "clad_v2_diffusion_steps": int(
                    r2.get("diffusion_steps", r2.get("forward_count", 0))
                ),
            }
        )
    return rows


def save_pair_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("id\n")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CCD vs CLAD-v2 on MATH500 level-5."
    )
    parser.add_argument(
        "--ccd_run",
        type=Path,
        default=Path(
            "/home/qinghua/yangjh/dlm/experiments/runs/20260324_082131_llada2_math500_decode=ccd.jsonl"
        ),
    )
    parser.add_argument(
        "--clad_v2_run",
        type=Path,
        default=Path(
            "/home/qinghua/yangjh/dlm/experiments/runs/20260324_082155_llada2_math500_decode=clad_v2.jsonl"
        ),
    )
    parser.add_argument(
        "--ccd_correct",
        type=Path,
        default=Path(
            "/home/qinghua/yangjh/dlm/experiments/evals/20260325_054754_evaluate_benchmark_results_correct_id_level.jsonl"
        ),
    )
    parser.add_argument(
        "--clad_v2_correct",
        type=Path,
        default=Path(
            "/home/qinghua/yangjh/dlm/experiments/evals/20260325_054845_evaluate_benchmark_results_correct_id_level.jsonl"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/qinghua/yangjh/dlm/experiments/evals"),
    )
    args = parser.parse_args()

    ccd_run = load_run_by_id(args.ccd_run)
    clad_run = load_run_by_id(args.clad_v2_run)
    ccd_correct_l5 = load_correct_ids_level5(args.ccd_correct)
    clad_correct_l5 = load_correct_ids_level5(args.clad_v2_correct)

    ccd_m = compute_metrics_level5(ccd_run, ccd_correct_l5)
    clad_m = compute_metrics_level5(clad_run, clad_correct_l5)

    # 共同答对（level=5）子集：用于对比“在同一批难题上谁更高效”
    common_ids_l5_pair = (
        (ccd_correct_l5 & clad_correct_l5) & set(ccd_run.keys()) & set(clad_run.keys())
    )

    pair_rows = build_level5_pair_rows(
        ccd_run, clad_run, ccd_correct_l5, clad_correct_l5
    )
    pair_csv = args.out_dir / "math500_level5_ccd_vs_clad_v2_per_sample.csv"
    save_pair_csv(pair_csv, pair_rows)

    report_path = args.out_dir / "math500_level5_ccd_vs_clad_v2_report.txt"
    lines = []
    lines.append("=" * 84)
    lines.append("MATH500 Level-5 Comparison: CCD vs CLAD-v2")
    lines.append("=" * 84)
    lines.append(f"CCD run:      {args.ccd_run}")
    lines.append(f"CLAD-v2 run:  {args.clad_v2_run}")
    lines.append(f"CCD correct:  {args.ccd_correct}")
    lines.append(f"CLAD correct: {args.clad_v2_correct}")
    lines.append("")
    lines.append(f"Level-5 samples (CCD):     {ccd_m.total_level5}")
    lines.append(f"Level-5 samples (CLAD-v2): {clad_m.total_level5}")
    lines.append("")
    lines.append("[Accuracy]")
    lines.append(
        f"  CCD      : {ccd_m.correct_level5}/{ccd_m.total_level5} = {ccd_m.accuracy:.2%}"
    )
    lines.append(
        f"  CLAD-v2  : {clad_m.correct_level5}/{clad_m.total_level5} = {clad_m.accuracy:.2%} "
        f"({pct((clad_m.accuracy - ccd_m.accuracy))} vs CCD)"
    )
    lines.append("")
    lines.append("[Efficiency]")
    lines.append(
        f"  Avg Generation Time (s): CCD={ccd_m.avg_gen_time:.3f}, CLAD-v2={clad_m.avg_gen_time:.3f} "
        f"({pct((ccd_m.avg_gen_time - clad_m.avg_gen_time)/ccd_m.avg_gen_time)} faster if positive)"
    )
    lines.append(
        f"  Throughput (tok/s)     : CCD={ccd_m.throughput:.3f}, CLAD-v2={clad_m.throughput:.3f} "
        f"({pct((clad_m.throughput - ccd_m.throughput)/ccd_m.throughput)} vs CCD)"
    )
    lines.append(
        f"  Correct Throughput     : CCD={ccd_m.correct_throughput:.3f}, CLAD-v2={clad_m.correct_throughput:.3f} "
        f"({pct((clad_m.correct_throughput - ccd_m.correct_throughput)/ccd_m.correct_throughput)} vs CCD)"
    )
    lines.append(
        f"  Avg Output Len         : CCD={ccd_m.avg_output_len:.2f}, CLAD-v2={clad_m.avg_output_len:.2f}"
    )
    lines.append("")
    lines.append("[Forward / Diffusion]")
    lines.append(
        f"  Avg Forward Count      : CCD={ccd_m.avg_forward:.2f}, CLAD-v2={clad_m.avg_forward:.2f}"
    )
    lines.append(
        f"  Total Forward Count    : CCD={ccd_m.total_forward}, CLAD-v2={clad_m.total_forward}"
    )
    lines.append(
        f"  TPF (macro)            : CCD={ccd_m.macro_tpf:.4f}, CLAD-v2={clad_m.macro_tpf:.4f}"
    )
    lines.append(
        f"  TPF (mean/sample)      : CCD={ccd_m.mean_tpf:.4f}, CLAD-v2={clad_m.mean_tpf:.4f}"
    )
    lines.append(
        f"  Avg Diffusion Steps    : CCD={ccd_m.avg_diffusion_steps:.2f}, CLAD-v2={clad_m.avg_diffusion_steps:.2f}"
    )
    lines.append(
        f"  Total Diffusion Steps  : CCD={ccd_m.total_diffusion_steps}, CLAD-v2={clad_m.total_diffusion_steps}"
    )
    lines.append("")

    # 共同答对子集效率分析（只比较效率，不再比较准确率：该子集默认全对）
    lines.append("=" * 84)
    lines.append("[Both Correct] Level-5 Common Subset Efficiency")
    lines.append(
        f"Common correct ids (level-5, CCD & CLAD-v2): {len(common_ids_l5_pair)}"
    )

    if len(common_ids_l5_pair) > 0:
        common_ccd_m = compute_efficiency_on_id_set(ccd_run, common_ids_l5_pair)
        common_clad_m = compute_efficiency_on_id_set(clad_run, common_ids_l5_pair)

        lines.append("")
        lines.append("[Common subset - Efficiency]")
        lines.append(
            f"  Avg Generation Time (s): CCD={common_ccd_m.avg_gen_time:.3f}, "
            f"CLAD-v2={common_clad_m.avg_gen_time:.3f} "
            f"({pct((common_ccd_m.avg_gen_time - common_clad_m.avg_gen_time) / common_ccd_m.avg_gen_time)} vs CCD)"
        )
        lines.append(
            f"  Throughput (tok/s)     : CCD={common_ccd_m.throughput:.3f}, "
            f"CLAD-v2={common_clad_m.throughput:.3f} "
            f"({pct((common_clad_m.throughput - common_ccd_m.throughput) / common_ccd_m.throughput)} vs CCD)"
        )
        lines.append(
            f"  Correct Throughput     : CCD={common_ccd_m.correct_throughput:.3f}, "
            f"CLAD-v2={common_clad_m.correct_throughput:.3f}"
        )
        lines.append(
            f"  TPF (macro)            : CCD={common_ccd_m.macro_tpf:.4f}, "
            f"CLAD-v2={common_clad_m.macro_tpf:.4f}"
        )
        lines.append(
            f"  Avg Diffusion Steps    : CCD={common_ccd_m.avg_diffusion_steps:.2f}, "
            f"CLAD-v2={common_clad_m.avg_diffusion_steps:.2f}"
        )

        common_rows = [r for r in pair_rows if int(r.get("both_correct", 0)) == 1]
        common_csv = (
            args.out_dir / "math500_level5_both_correct_ccd_vs_clad_v2_per_sample.csv"
        )
        save_pair_csv(common_csv, common_rows)
        lines.append(f"Common subset CSV saved to: {common_csv}")

    lines.append(f"Per-sample comparison CSV saved to: {pair_csv}")
    lines.append("=" * 84)

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
