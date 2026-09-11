"""
tasks/mmlu/gen_mmlu.py

Tải MMLU test set từ HuggingFace và lưu thành JSON.

Usage:
    python tasks/mmlu/gen_mmlu.py
    python tasks/mmlu/gen_mmlu.py --subjects all
    python tasks/mmlu/gen_mmlu.py --subjects "high_school_math,philosophy,anatomy"
    python tasks/mmlu/gen_mmlu.py --n 500
"""

from datasets import load_dataset
import json
import os
import random
import argparse

# 57 subjects trong MMLU
ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

CHOICES = ["A", "B", "C", "D"]


def build_question_text(row: dict, subject: str) -> str:
    """Ghép question + choices thành 1 string."""
    q = row["question"].strip()
    choices = row["choices"]
    opts = "\n".join(f"{CHOICES[i]}) {choices[i]}" for i in range(len(choices)))
    return f"[{subject}]\n{q}\n{opts}"


def quotas(sizes: dict, n: int) -> dict:
    """Chia n cho các subject, đều nhau nhất có thể.

    Subject nào không đủ câu thì nhận hết phần mình có, phần thừa chia lại cho
    các subject còn chỗ => luôn đạt đủ n nếu tổng pool >= n, và mọi subject đều
    có mặt.
    """
    q = {s: 0 for s in sizes}
    left = min(n, sum(sizes.values()))
    while left:
        open_ = [s for s in sizes if q[s] < sizes[s]]
        share = max(1, left // len(open_))
        for s in open_:
            if not left:
                break
            add = min(share, sizes[s] - q[s], left)
            q[s] += add
            left -= add
    return q


def build_mmlu(
    output_path: str = "data/mmlu/test.json",
    subjects: list = None,
    n: int = None,
    seed: int = 42,
):
    if subjects is None:
        subjects = ALL_SUBJECTS

    # Nạp hết subject trước, rồi mới chia suất. Bản cũ break ngay khi đủ n nên
    # vét cạn subject đầu bảng chữ cái và bỏ trắng phần còn lại.
    pools = {}
    for subject in subjects:
        print(f"  Loading {subject}...")
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"  ⚠ Skipped {subject}: {e}")
            continue

        pools[subject] = [
            {
                "question": build_question_text(row, subject),
                "answer": CHOICES[int(row["answer"])],
                "subject": subject,
            }
            for row in ds
        ]

    sizes = {s: len(v) for s, v in pools.items()}
    quota = quotas(sizes, n) if n else sizes

    # bốc ngẫu nhiên trong từng subject, không lấy k câu đầu (tránh lệch thứ tự gốc)
    rng = random.Random(seed)
    samples = []
    for subject, rows in pools.items():
        idxs = sorted(rng.sample(range(len(rows)), quota[subject]))
        samples += [rows[i] for i in idxs]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(samples)} MMLU samples to {output_path}")
    # Subject distribution
    from collections import Counter
    dist = Counter(s["subject"] for s in samples)
    print(f"Subjects covered: {len(dist)}/{len(subjects)} "
          f"({min(dist.values())}-{max(dist.values())} câu mỗi subject)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects", default="all",
        help="Comma-separated subject names, or 'all' for all 57 subjects"
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Tổng số câu, chia đều cho các subject")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed chọn câu trong từng subject (giữ cố định)")
    parser.add_argument("--output", default="data/mmlu/test.json")
    args = parser.parse_args()

    if args.subjects == "all":
        subjects = ALL_SUBJECTS
    else:
        subjects = [s.strip() for s in args.subjects.split(",")]

    build_mmlu(output_path=args.output, subjects=subjects, n=args.n, seed=args.seed)