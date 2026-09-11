"""Tinh DUS cho tung round debate va ghi results/dus_per_round.csv.

DUS = sum_i w_i * (x_i - mean_i) / std_i, voi w/mean/std tu results/dus_weights.json.
Doc log THO qua logio (debate_full_*.jsonl). File nay chi con nhiem vu SINH
dus_per_round.csv (verify_threshold.py doc no). Phan tich/so sanh AUC: xem p0, p2.

Chay:
    python analyze_dus_extremes.py
    python analyze_dus_extremes.py --logs-dir results/logs --weights results/dus_weights.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import logio

FEATURES = logio.BASE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinh dus_per_round.csv (DUS cho tung round).")
    p.add_argument("--logs-dir", type=Path, default=Path("results/logs"),
                    help="Thu muc log; doc DE QUY debate_full_*.jsonl (moi seed)")
    p.add_argument("--weights", type=Path, default=Path("results/dus_weights.json"),
                    help="File trong so DUS da hoc (results/dus_weights.json)")
    p.add_argument("--output", type=Path, default=Path("results/dus_per_round.csv"))
    return p.parse_args()


def compute_dus(logs_dir: Path, weights_path: Path):
    spec = json.loads(weights_path.read_text(encoding="utf-8"))
    w = np.array([spec["dus_weights"][f] for f in FEATURES], dtype=float)
    mean = np.array([spec["feature_standardization"][f]["mean"] for f in FEATURES], dtype=float)
    std = np.array([spec["feature_standardization"][f]["std"] for f in FEATURES], dtype=float)
    std = np.where(std == 0, 1.0, std)   # tranh loi read-only (pandas CoW)

    df = logio.load_rounds(str(Path(logs_dir) / "**" / "debate_full_*.jsonl"))
    df = df.dropna(subset=FEATURES + ["final_correct"]).copy()
    df["final_correct"] = df["final_correct"].astype(bool)
    df["dus"] = ((df[FEATURES].to_numpy(dtype=float) - mean) / std) @ w
    df = df.rename(columns={"consensus_now": "consensus"})
    return df[["seed", "task", "sample_id", "round_id", "final_correct", "dus",
               *FEATURES, "consensus"]]


def main() -> None:
    args = parse_args()
    data = compute_dus(args.logs_dir, args.weights).sort_values("dus", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"{len(data)} round -> {args.output}  "
          f"(dung={int(data.final_correct.sum())}, sai={int((~data.final_correct).sum())})")


if __name__ == "__main__":
    main()
