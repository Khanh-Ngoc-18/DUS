"""split.py - Chia sample_id thanh train/val/test theo tung benchmark (jsonl-only).

Vi 3 file log cu (jsonl + 2 csv) khong deu dong, KHONG the split theo dong. Ta chia
theo sample_id (cau hoi): moi dong thuoc cung 1 sample_id luon nam cung 1 split =>
tranh ro ri du lieu giua cac tap.

Ban jsonl-only nay CHI ghi mapping results/split/<benchmark>/split_sample_ids.json.
Khong con copy/filter debate_rounds/debate_agents CSV: cac script (p0, train_dus_weights)
doc mapping nay + log THO qua logio.load_rounds(). => mot nguon chan ly duy nhat, het
desync, va khong con chuan hoa disagreement_persistence /6 (thang do khong con y nghia
vi feature duoc z-score o buoc fit).

Cach dung:
    python split.py
    python split.py --root results/logs --benchmarks gsm8k mmlu strategyqa \
        --train 0.7 --val 0.2 --test 0.1 --seed 42

Ket qua: results/split/<benchmark>/split_sample_ids.json
"""

import argparse
import glob
import json
import random
from pathlib import Path


def sample_ids_for(root: Path, bench: str) -> list:
    """Lay danh sach sample_id duy nhat cua benchmark tu MOI file jsonl (moi seed)."""
    seen, ids = set(), []
    pattern = str(Path(root) / bench / "**" / "debate_full_*.jsonl")
    for f in sorted(glob.glob(pattern, recursive=True)):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sid = json.loads(line)["sample_id"]
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
    return ids


def split_ids(ids, train_ratio, val_ratio, test_ratio, seed) -> dict:
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"train+val+test phai bang 1.0, hien = {total}"
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    return {"train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Chia sample_id train/val/test theo benchmark, ghi mapping JSON (jsonl-only)."
    )
    ap.add_argument("--root", type=Path, default=Path("results/logs"),
                    help="Thu muc log (chua <benchmark>/seed*/debate_full_*.jsonl)")
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "mmlu", "strategyqa"])
    ap.add_argument("--out", type=Path, default=None, help="Mac dinh: results/split")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42, help="Seed cho reproducible split")
    args = ap.parse_args()

    out_dir = args.out if args.out else Path("results/split")
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in args.benchmarks:
        ids = sample_ids_for(args.root, bench)
        if not ids:
            print(f"[WARN] Khong tim thay debate_full_*.jsonl cho '{bench}', bo qua.")
            continue
        splits = split_ids(ids, args.train, args.val, args.test, args.seed)
        bench_out = out_dir / bench
        bench_out.mkdir(parents=True, exist_ok=True)
        mapping_path = bench_out / "split_sample_ids.json"
        mapping_path.write_text(
            json.dumps({k: sorted(v) for k, v in splits.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"{bench}: {len(ids)} sample -> train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}  ({mapping_path})")

    print(f"\nHoan tat. Mapping o: {out_dir}")


if __name__ == "__main__":
    main()
