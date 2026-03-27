# dlm/src/download_benchmarks.py
# 用 datasets 从 Hugging Face 下载数据集，并导出为本地 jsonl

from pathlib import Path
import json
import argparse
from datasets import load_dataset  # 需要先: uv pip install datasets

ROOT = Path(__file__).resolve().parents[1]  # 指向 dlm/
OUT_DIR = ROOT / "experiments" / "benchmarks"


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def export_gsm8k_small(n_train: int = 200, n_test: int = 100):
    ds = load_dataset("openai/gsm8k", "main")
    recs = []

    # 只取一部分 train + 一部分 test，字段统一为 question/answer
    for split_name, n_limit in [("train", n_train), ("test", n_test)]:
        split = ds[split_name]
        # 注意：datasets 切片 ds[split][:n] 返回的是「列字典」，
        # 想拿到一条一条的样本，需要用索引：split[i]
        for i in range(min(n_limit, len(split))):
            row = split[i]
            recs.append(
                {
                    "id": f"gsm8k_{split_name}_{i}",
                    "split": split_name,
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )

    out_path = OUT_DIR / "math" / "gsm8k_small.jsonl"
    save_jsonl(recs, out_path)
    print(f"[gsm8k] saved {len(recs)} examples to {out_path}")


def export_aime2025_all():
    # 两个子集: AIME2025-I / AIME2025-II，只有 test split
    recs = []
    for subset in ["AIME2025-I", "AIME2025-II"]:
        ds = load_dataset("opencompass/AIME2025", subset)
        for i, row in enumerate(ds["test"]):
            recs.append(
                {
                    "id": f"{subset}_test_{i}",
                    "subset": subset,
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )
    out_path = OUT_DIR / "math" / "aime2025_all.jsonl"
    save_jsonl(recs, out_path)
    print(f"[AIME2025] saved {len(recs)} examples to {out_path}")


def export_mbpp_sanitized():
    ds = load_dataset("Muennighoff/mbpp", "sanitized")
    recs = []
    for row in ds["test"]:
        recs.append(
            {
                "id": f"mbpp_{row['task_id']}",
                "prompt": row["prompt"],
                "reference_code": row["code"],
                "tests": row["test_list"],
            }
        )
    out_path = OUT_DIR / "code" / "mbpp_sanitized.jsonl"
    save_jsonl(recs, out_path)
    print(f"[MBPP] saved {len(recs)} examples to {out_path}")


def export_humaneval_all():
    ds = load_dataset("openai/openai_humaneval")
    recs = []
    for row in ds["test"]:
        recs.append(
            {
                "id": row["task_id"],
                "prompt": row["prompt"],
                "reference_code": row["canonical_solution"],
                "tests": row["test"],
                "entry_point": row["entry_point"],
            }
        )
    out_path = OUT_DIR / "code" / "humaneval_all.jsonl"
    save_jsonl(recs, out_path)
    print(f"[HumanEval] saved {len(recs)} examples to {out_path}")


def export_arc(n: int = 300):
    """
    下载 ARC (AI2 Reasoning Challenge) 数据集，分为 Easy 和 Challenge 两个难度。
    各取 n 条，优先从 test split 取，不足时从 validation 补充，再不足时从 train 补充。

    字段说明：
    - id           : 原始题目 id
    - subset       : "ARC-Easy" 或 "ARC-Challenge"
    - question     : 题目文本
    - choices      : 选项列表，每项为 {"label": "A", "text": "..."}
    - answer_key   : 正确答案的 label（如 "A"、"B"、"C"、"D"）
    - answer_text  : 正确答案的文本（方便 judge 评测时直接对比）
    """
    for subset, filename in [
        ("ARC-Easy", "arc_easy_300.jsonl"),
        ("ARC-Challenge", "arc_challenge_300.jsonl"),
    ]:
        ds = load_dataset("allenai/ai2_arc", subset)

        # 按 test → validation → train 顺序拼凑 n 条
        recs = []
        for split_name in ["test", "validation", "train"]:
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                if len(recs) >= n:
                    break
                # choices 字段结构：{"text": [...], "label": [...]}
                choices = [
                    {"label": lbl, "text": txt}
                    for lbl, txt in zip(row["choices"]["label"], row["choices"]["text"])
                ]
                # 找到正确答案的文本
                answer_text = next(
                    (c["text"] for c in choices if c["label"] == row["answerKey"]),
                    row["answerKey"],
                )
                recs.append(
                    {
                        "id": row["id"],
                        "subset": subset,
                        "question": row["question"],
                        "choices": choices,
                        "answer_key": row["answerKey"],
                        "answer_text": answer_text,
                    }
                )
            if len(recs) >= n:
                break

        out_path = OUT_DIR / "reasoning" / filename
        save_jsonl(recs, out_path)
        print(f"[ARC] {subset}: saved {len(recs)} examples to {out_path}")


def export_math500_test():
    """
    下载 Hendrycks MATH benchmark 的 test split（500条）到本地。

    数据源：
    - https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark/viewer/default/test

    输出字段（统一为问答评测友好格式）：
    - id         : 数据唯一ID（优先 unique_id，否则 fallback 为 math500_test_i）
    - question   : 题目文本（problem）
    - answer     : 标准最终答案（answer）
    - solution   : 参考解析（solution，供误差分析）
    - subject    : 学科
    - level      : 难度等级（1~5）
    """
    ds = load_dataset("nlile/hendrycks-MATH-benchmark")
    split = ds["test"]  # 500 rows
    recs = []
    for i, row in enumerate(split):
        recs.append(
            {
                "id": row.get("unique_id") or f"math500_test_{i}",
                "question": row["problem"],
                "answer": row["answer"],
                "solution": row.get("solution", ""),
                "subject": row.get("subject"),
                "level": row.get("level"),
            }
        )

    out_path = OUT_DIR / "math" / "math500.jsonl"
    save_jsonl(recs, out_path)
    print(f"[MATH500] saved {len(recs)} examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="下载并导出 benchmark 数据集到本地 jsonl"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=[
            "gsm8k_small",
            "aime2025_all",
            "mbpp_sanitized",
            "humaneval_all",
            "arc",
            "math500",
            "all",
        ],
        help="要导出的数据集；默认 math500",
    )
    parser.add_argument(
        "--arc_n",
        type=int,
        default=300,
        help="导出 ARC 时每个子集的样本数（默认 300）",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("gsm8k_small", "all"):
        export_gsm8k_small()
    if args.dataset in ("aime2025_all", "all"):
        export_aime2025_all()
    if args.dataset in ("mbpp_sanitized", "all"):
        export_mbpp_sanitized()
    if args.dataset in ("humaneval_all", "all"):
        export_humaneval_all()
    if args.dataset in ("arc", "all"):
        export_arc(n=args.arc_n)
    if args.dataset in ("math500", "all"):
        export_math500_test()

    print("Done.")
