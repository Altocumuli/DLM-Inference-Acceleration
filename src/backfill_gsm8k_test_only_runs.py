"""
为缺失 question / reference_answer 的 gsm8k_test_only 运行结果回填字段。

用法示例：
    python dlm/src/backfill_gsm8k_test_only_runs.py \
        --results_file dlm/experiments/runs/20260416_083724_llada2_gsm8k_test_only_decode=clad_v2.jsonl

    python dlm/src/backfill_gsm8k_test_only_runs.py \
        --results_dir dlm/experiments/runs

逻辑：
- 从 experiments/benchmarks/math/gsm8k_testOnly.jsonl 读取官方题目与标准答案
- 按 id 精确匹配 runs 文件中的样本
- 若记录缺失 question 或 reference_answer，则补上并原地覆写
- 已经存在的字段不改动
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "experiments" / "benchmarks" / "math" / "gsm8k_testOnly.jsonl"


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gsm8k_lookup(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 benchmark 文件: {path}. "
            "请先运行: python dlm/src/download_benchmarks.py --dataset gsm8k_test_only"
        )
    lookup: Dict[str, Dict[str, str]] = {}
    for row in read_jsonl(path):
        sample_id = row["id"]
        lookup[sample_id] = {
            "question": row["question"],
            "reference_answer": row["answer"],
        }
    return lookup


def backfill_file(path: Path, lookup: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    rows = read_jsonl(path)
    updated = 0
    skipped = 0
    missing_ids = 0

    for row in rows:
        if row.get("benchmark") != "gsm8k_test_only":
            skipped += 1
            continue
        sample_id = row.get("id")
        if sample_id not in lookup:
            missing_ids += 1
            continue

        changed = False
        if "question" not in row or not row.get("question"):
            row["question"] = lookup[sample_id]["question"]
            changed = True
        if "reference_answer" not in row or not row.get("reference_answer"):
            row["reference_answer"] = lookup[sample_id]["reference_answer"]
            changed = True
        if changed:
            updated += 1

    if updated > 0:
        write_jsonl(rows, path)

    return {
        "updated": updated,
        "skipped": skipped,
        "missing_ids": missing_ids,
        "total": len(rows),
    }


def collect_target_files(
    results_file: str | None, results_dir: str | None
) -> List[Path]:
    if results_file:
        return [Path(results_file).resolve()]
    if not results_dir:
        raise ValueError("必须指定 --results_file 或 --results_dir")

    root = Path(results_dir).resolve()
    return sorted(root.glob("*gsm8k_test_only*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="回填 gsm8k_test_only runs 文件中的 question/reference_answer"
    )
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    targets = collect_target_files(args.results_file, args.results_dir)
    if not targets:
        print("未找到目标 jsonl 文件。")
        return

    lookup = load_gsm8k_lookup(BENCH_PATH)

    print(f"[backfill] benchmark source: {BENCH_PATH}")
    print(f"[backfill] targets: {len(targets)}")
    for path in targets:
        stats = backfill_file(path, lookup)
        print(
            f"[backfill] {path.name}: "
            f"updated={stats['updated']} total={stats['total']} "
            f"skipped={stats['skipped']} missing_ids={stats['missing_ids']}"
        )


if __name__ == "__main__":
    main()
