"""Estimate DUS weights from debate logs (debate_full_*.jsonl qua logio).

The classifier is trained to predict an ``uncertain`` label from the four DUS
signals.  Two targets are supported:

* ``correctness`` (default): ``uncertain = the debate's final answer is wrong``.
  This is the meaningful uncertainty target -- a confident consensus can still
  be wrong, so no single feature trivially determines the label.
* ``consensus``: ``uncertain = not consensus``.  Kept for reference, but note
  that ``consensus`` is definitionally ``answer_entropy == 0`` in this data, so
  answer_entropy leaks the label and dominates the fit.

Split membership comes from results/split/<task>/split_sample_ids.json (viet boi
split.py). Feature/nhan doc TRUC TIEP tu log THO qua logio -- khong con doc
debate_rounds CSV, va disagreement_persistence de nguyen thang goc (z-score
trong ham fit tu chuan hoa, nen thang do khong anh huong).

Inputs are z-scored before fitting so the learned coefficients can be compared.
Non-positive coefficients are clipped to zero (a DUS component must *increase*
uncertainty), and the remaining positive coefficients are normalized to sum to
one, which yields the weights for the bounded DUS formula.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import logio

FEATURES = logio.BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit DUS weights with logistic regression.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/split"),
        help="Directory containing <task>/split_sample_ids.json (split mapping).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dus_weights.json"),
        help="Path of the learned weights JSON.",
    )
    parser.add_argument(
        "--target",
        choices=["correctness", "consensus"],
        default="correctness",
        help="Label to fit. 'correctness' (default) avoids the answer_entropy leak.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which split to fit on (train/val/test). Use 'train' to avoid leakage.",
    )
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def load_rounds(input_dir: Path, target: str, split: str):
    """Load rounds of the requested split from raw logs + split mapping (jsonl-only)."""
    splits = logio.load_splits(str(input_dir))
    if not splits:
        raise FileNotFoundError(
            f"No */split_sample_ids.json under {Path(input_dir).resolve()} (run split.py first)."
        )
    data = logio.load_rounds()
    in_split = np.array(
        [splits.get((str(t), int(s))) == split for t, s in zip(data.task, data.sample_id)]
    )
    data = data[in_split].copy()
    data = data.dropna(subset=FEATURES).copy()

    if target == "correctness":
        data = data.dropna(subset=["final_correct"]).copy()
        data["uncertain"] = (~data["final_correct"].astype(bool)).astype(float)
    else:  # consensus
        data["uncertain"] = (~data["consensus_now"].astype(bool)).astype(float)

    if data["uncertain"].nunique() != 2:
        raise ValueError("Need both positive and negative rows to fit logistic regression.")
    return data


def fit_logistic_regression(
    x: np.ndarray, y: np.ndarray, l2: float, max_iter: int, tolerance: float
) -> tuple[np.ndarray, float, int]:
    """L2-regularized logistic regression via Newton/IRLS; intercept is unpenalized."""
    beta = np.zeros(x.shape[1] + 1, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])

    def loss_for(candidate: np.ndarray) -> float:
        logits = design @ candidate
        return float(
            np.mean(np.logaddexp(0, logits) - y * logits)
            + 0.5 * l2 * np.dot(candidate[1:], candidate[1:])
        )

    for iteration in range(1, max_iter + 1):
        logits = np.clip(design @ beta, -35, 35)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[1:] += l2 * beta[1:]
        curvature = probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / len(y)
        hessian[1:, 1:] += l2 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step) < tolerance:
            return beta[1:], beta[0], iteration

        # Backtracking makes the fit stable even for heavily imbalanced data.
        current_loss = loss_for(beta)
        step_size = 1.0
        while loss_for(beta - step_size * step) > current_loss and step_size > 1e-8:
            step_size *= 0.5
        beta -= step_size * step
    return beta[1:], beta[0], max_iter


def main() -> None:
    args = parse_args()
    data = load_rounds(args.input_dir, args.target, args.split)
    raw_x = data[FEATURES].to_numpy(dtype=float)
    mean = raw_x.mean(axis=0)
    scale = raw_x.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)   # tranh loi read-only (pandas CoW)
    x = (raw_x - mean) / scale
    y = data["uncertain"].to_numpy(dtype=float)

    coefficients, intercept, iterations = fit_logistic_regression(
        x, y, args.l2, args.max_iter, args.tolerance
    )
    # A DUS component must increase uncertainty.  Clip non-positive effects to
    # zero, then normalize so w_i >= 0 and sum(w_i) = 1 as in the proposal.
    positive = np.clip(coefficients, 0.0, None)
    total = positive.sum()
    if total > 0:
        weights = positive / total
    else:  # degenerate: no feature increases uncertainty
        weights = np.full_like(positive, 1.0 / len(positive))

    target_desc = (
        "uncertain = (final_correct == False)"
        if args.target == "correctness"
        else "uncertain = (consensus == False)"
    )
    result = {
        "target": target_desc,
        "target_mode": args.target,
        "split": args.split,
        "formula": "DUS = w1*answer_entropy + w2*confidence_variance + w3*disagreement_persistence + w4*answer_flip_rate",
        "n_rows": int(len(data)),
        "uncertain_rate": float(y.mean()),
        "feature_standardization": {name: {"mean": float(m), "std": float(s)} for name, m, s in zip(FEATURES, mean, scale)},
        "logistic_coefficients_on_standardized_features": {name: float(c) for name, c in zip(FEATURES, coefficients)},
        "intercept": float(intercept),
        "dus_weights": {name: float(w) for name, w in zip(FEATURES, weights)},
        "iterations": iterations,
        "regularization": {"l2": args.l2},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Loaded {len(data)} rounds (target={args.target}, split={args.split})")
    print("DUS weights (sum = 1):")
    for name, weight in result["dus_weights"].items():
        print(f"  {name:27s} {weight:.12f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
