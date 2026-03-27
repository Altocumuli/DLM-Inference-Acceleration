"""
从 evaluate_benchmark_results 的 judge_details 日志中提取：
  - CORRECT 的样本 ID
  - 并从 MATH500 基准数据（math500.jsonl）读取对应 level

输出：
  - correct_id_level.jsonl
  - correct_id_level.csv

用途：
  - 复用你在中期答辩/论文分析中做 level-wise 误差分析的流程
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_id_to_level(math500_path: Path) -> Dict[str, int]:
    id_to_level: Dict[str, int] = {}
    for row in read_jsonl(math500_path):
        sid = str(row.get("id"))
        lvl = row.get("level")
        if sid is None:
            continue
        if isinstance(lvl, int):
            id_to_level[sid] = lvl
        else:
            # level 缺失或类型异常时，仍允许后续统计但会标记为 None
            try:
                id_to_level[sid] = int(lvl)
            except Exception:
                pass
    return id_to_level


def extract_correct_ids(judge_details_log: Path) -> List[str]:
    """
    judge_details.log 格式（由 evaluate_benchmark_results.py 输出）：
      [Sample ID] xxx
      [Judgment]  CORRECT / INCORRECT

    我们按块读取：遇到 Sample ID 开始记录，下一次判断行决定该 sample 是否正确。
    """
    # 注意：这里是“日志文本中的原样字符”，不要额外转义方括号
    sample_re = re.compile(r"^\[Sample ID\]\s*(.+?)\s*$")
    judgment_re = re.compile(r"^\[Judgment\]\s*(CORRECT|INCORRECT)\s*$")

    correct_ids: List[str] = []
    current_id: Optional[str] = None
    current_judgment: Optional[str] = None

    with judge_details_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.rstrip("\\n")

            m1 = sample_re.match(raw)
            if m1:
                # 之前 sample 如果是 CORRECT，则把它落入列表
                if current_id is not None and current_judgment == "CORRECT":
                    correct_ids.append(current_id)
                current_id = m1.group(1).strip()
                current_judgment = None
                continue

            m2 = judgment_re.match(raw)
            if m2 and current_id is not None:
                current_judgment = m2.group(1)

    # 文件末尾再补一次
    if current_id is not None and current_judgment == "CORRECT":
        correct_ids.append(current_id)

    return correct_ids


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("id,level\\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CORRECT IDs and map to MATH500 level."
    )
    parser.add_argument(
        "--judge_details_log", type=Path, required=True, help="..._judge_details.log"
    )
    parser.add_argument(
        "--math500_jsonl",
        type=Path,
        required=False,
        default=Path(
            "/home/qinghua/yangjh/dlm/experiments/benchmarks/math/math500.jsonl"
        ),
        help="本地 math500.jsonl（含 level 字段）",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=False,
        default=Path("/home/qinghua/yangjh/dlm/experiments/evals"),
        help="输出目录",
    )
    args = parser.parse_args()

    correct_ids = extract_correct_ids(args.judge_details_log)
    id_to_level = load_id_to_level(args.math500_jsonl)

    rows: List[Dict] = []
    missing_level = 0
    level_count: Dict[int, int] = {}

    for sid in correct_ids:
        lvl = id_to_level.get(sid)
        if lvl is None:
            missing_level += 1
        else:
            level_count[lvl] = level_count.get(lvl, 0) + 1
        rows.append({"id": sid, "level": lvl})

    stem = args.judge_details_log.name.replace("_judge_details.log", "")
    out_jsonl = args.out_dir / f"{stem}_correct_id_level.jsonl"
    out_csv = args.out_dir / f"{stem}_correct_id_level.csv"

    save_jsonl(out_jsonl, rows)
    save_csv(out_csv, rows)

    print(f"[done] log: {args.judge_details_log}")
    print(f"[done] correct count: {len(rows)}")
    print(f"[done] missing level: {missing_level}")
    if level_count:
        for k in sorted(level_count):
            print(f"  level {k}: {level_count[k]}")
    print(f"[done] out jsonl: {out_jsonl}")
    print(f"[done] out csv:  {out_csv}")


if __name__ == "__main__":
    main()
