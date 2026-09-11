"""Evaluate learned DUS weights on held-out splits.

Loads the weights + standardization stats produced by ``train_dus_weights.py``,
computes the DUS score for each debate round on the requested split(s), and
reports how well DUS separates wrong debates from correct ones.

DUS is evaluated in the *standardized* feature space (the space the weights were
learned in), i.e. DUS = sum_i w_i * (x_i - mean_i) / std_i, using the mean/std
stored in the weights JSON.  This keeps the four signals comparable; applying the
weights to raw features would let answer_entropy dominate purely because of its
larger numeric range.

Metrics: ROC-AUC (rank-based, no sklearn dependency), plus per-feature AUC and a
small threshold table (accuracy / precision / recall for flagging "uncertain").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_dus_weights import FEATURES, load_rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DUS weights on held-out splits.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/split"))
    parser.add_argument("--weights", type=Path, default=Path("results/dus_weights.json"))
    parser.add_argument("--target", choices=["correctness", "consensus"], default="correctness")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to evaluate (default: val test).",
    )
    parser.add_argument("--output", type=Path, default=Path("results/dus_eval.json"))
    return parser.parse_args()


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the Mann-Whitney rank statistic; handles ties.  labels: 1=positive."""
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so tied scores don't bias the statistic.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_table(dus: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for q in (0.5, 0.6, 0.7, 0.8):
        thr = float(np.quantile(dus, q))
        flag = dus >= thr
        tp = int((flag & (labels == 1)).sum())
        fp = int((flag & (labels == 0)).sum())
        fn = int((~flag & (labels == 1)).sum())
        tn = int((~flag & (labels == 0)).sum())
        acc = (tp + tn) / len(labels)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        rows.append(
            {"quantile": q, "threshold": thr, "accuracy": acc, "precision": prec, "recall": rec}
        )
    return rows


def main() -> None:
    args = parse_args()
    spec = json.loads(args.weights.read_text(encoding="utf-8"))
    weights = np.array([spec["dus_weights"][name] for name in FEATURES], dtype=float)
    mean = np.array([spec["feature_standardization"][name]["mean"] for name in FEATURES])
    std = np.array([spec["feature_standardization"][name]["std"] for name in FEATURES])
    std = np.where(std == 0, 1.0, std)   # tranh loi read-only (pandas CoW)

    report = {"weights_file": str(args.weights), "target": args.target, "splits": {}}
    for split in args.splits:
        data = load_rounds(args.input_dir, args.target, split)
        raw = data[FEATURES].to_numpy(dtype=float)
        z = (raw - mean) / std
        dus = z @ weights
        labels = data["uncertain"].to_numpy(dtype=float)

        overall = roc_auc(dus, labels)
        per_feature = {name: roc_auc(z[:, i], labels) for i, name in enumerate(FEATURES)}
        table = threshold_table(dus, labels)

        report["splits"][split] = {
            "n_rows": int(len(data)),
            "uncertain_rate": float(labels.mean()),
            "dus_auc": overall,
            "per_feature_auc": per_feature,
            "threshold_table": table,
        }

        print(f"\n=== split={split}  (n={len(data)}, uncertain={labels.mean():.3f}) ===")
        print(f"  DUS ROC-AUC:            {overall:.4f}")
        print("  per-feature AUC:")
        for name, value in per_feature.items():
            print(f"    {name:27s} {value:.4f}")
        print("  threshold table (flag uncertain when DUS >= thr):")
        print(f"    {'quantile':>8s} {'thr':>8s} {'acc':>6s} {'prec':>6s} {'recall':>6s}")
        for row in table:
            print(
                f"    {row['quantile']:8.2f} {row['threshold']:8.3f} "
                f"{row['accuracy']:6.3f} {row['precision']:6.3f} {row['recall']:6.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
