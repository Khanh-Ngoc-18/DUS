"""P0 - Dua feature tu CRITIC vao va do lai AUC theo protocol dung.

Protocol (khac find_threshold.py):
  - Chi xet round KHONG PHAI round cuoi (round consensus da bi orchestrator cat).
  - Nhan = "dap an se duoc chot tai round r co SAI khong", tai lap dung
    _select_final_answer(): majority 3 agent, hoa thi lay confidence cao nhat.
    Scoring theo dung benchmark (gsm8k: so hoc, tolerance 1e-3).
  - Fit tren split train, danh gia tren val + test.
  - Bootstrap CI resample theo sample_id.

Nguon du lieu: debate_full_*.jsonl qua logio (khong con debate_agents / dus_per_round CSV).
Seed-aware: gom moi seed; loai round cuoi theo tung (seed, task, sample_id).

Output: results/p0_critic_eval.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# score/select_final re-export tu logio de tuong thich nguoc (script khac co the dung p0.score).
from logio import BASE, CRITIC, load_rounds, load_splits, score, select_final  # noqa: F401

RNG = np.random.default_rng(42)


def _dus_from_weights(df: pd.DataFrame, path: str = "results/dus_weights.json") -> np.ndarray:
    """DUS baseline = z(BASE) @ weights, lay tu dus_weights.json (neu co); khong co -> NaN."""
    p = Path(path)
    if not p.exists():
        return np.full(len(df), np.nan)
    spec = json.loads(p.read_text(encoding="utf-8"))
    w = np.array([spec["dus_weights"][f] for f in BASE], float)
    mu = np.array([spec["feature_standardization"][f]["mean"] for f in BASE], float)
    sd = np.array([spec["feature_standardization"][f]["std"] for f in BASE], float)
    sd = np.where(sd == 0, 1.0, sd)
    return ((df[BASE].to_numpy(float) - mu) / sd) @ w


def build() -> pd.DataFrame:
    df = load_rounds()
    df["wrong_if_stop"] = ~df.ok_if_stop
    splits = load_splits()
    df["split"] = [splits.get((t, int(s)), "?") for t, s in zip(df.task, df.sample_id)]
    df["dus"] = _dus_from_weights(df)
    # loai round CUOI cua moi rollout (seed, task, sample_id) - noi duy nhat co the dung som
    df["maxr"] = df.groupby(["seed", "task", "sample_id"]).round_id.transform("max")
    return df[df.round_id < df.maxr].dropna(subset=BASE + CRITIC).copy()


def auc(s, y) -> float:
    s, y = np.asarray(s, float), np.asarray(y, bool)
    r = pd.Series(s).rank().to_numpy()
    a, b = y.sum(), (~y).sum()
    return float("nan") if a == 0 or b == 0 else float((r[y].sum() - a * (a + 1) / 2) / (a * b))


def fit_logit(X, y, iters=800, lr=0.3, l2=1e-3):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(X) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return w


def boot_auc(s, y, groups, n=2000):
    """Bootstrap CI, resample theo sample_id (khong resample theo dong)."""
    s, y, groups = np.asarray(s, float), np.asarray(y, bool), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        v = auc(s[sel], y[sel])
        if not np.isnan(v):
            out.append(v)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_paired(scores: dict, y, groups, pairs, n=2000):
    """CI cua HIEU AUC giua hai model, uoc luong tren CUNG bo resample.

    CI rieng cua tung model (boot_auc o tren) khong tra loi duoc "hai model co
    khac nhau khong": hai khoang chong nhau van co the di kem mot hieu so khac 0
    ro rang, vi hai model duoc cham tren cung nhung cau hoi nen sai so cua chung
    tuong quan manh. Phai bootstrap chinh HIEU SO: moi vong chon mot bo sample_id,
    cham CA HAI model tren dung bo do, roi lay hieu.
    """
    y, groups = np.asarray(y, bool), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    draws = {k: [] for k in scores}
    for _ in range(n):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        ys = y[sel]
        if ys.all() or not ys.any():
            continue
        for k, s in scores.items():
            draws[k].append(auc(np.asarray(s, float)[sel], ys))

    out = {}
    for a, b in pairs:
        d = np.asarray(draws[a]) - np.asarray(draws[b])
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        out[f"{a} - {b}"] = dict(
            delta=float(auc(scores[a], y) - auc(scores[b], y)),
            ci=[lo, hi],
            p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            excludes_zero=bool(lo > 0 or hi < 0),
        )
    return out


def main() -> None:
    df = build()
    df["gid"] = df.task + "_" + df.sample_id.astype(str)
    y = df.wrong_if_stop.to_numpy(bool)
    print(f"round khong-cuoi: {len(df)}  |  sample: {df.gid.nunique()}"
          f"  |  P(sai neu chot) = {y.mean():.3f}\n")

    print("=== AUC tung feature (toan bo round khong-cuoi) ===")
    single = {}
    for f in BASE + CRITIC + ["dus"]:
        single[f] = auc(df[f], y)
        tag = "  <- critic" if f in CRITIC else ""
        print(f"  {f:26s} {single[f]:.3f}{tag}")

    tr, ev = df[df.split == "train"], df[df.split.isin(["val", "test"])]
    print(f"\ntrain={len(tr)} rounds / {tr.gid.nunique()} samples"
          f"   |   val+test={len(ev)} rounds / {ev.gid.nunique()} samples")

    print("\n=== Logistic fit tren TRAIN, danh gia tren VAL+TEST ===")
    res, model_scores = {}, {}
    for name, feats in [("base (DUS 4 feature)", BASE),
                        ("critic only", CRITIC),
                        ("base + critic", BASE + CRITIC)]:
        mu, sd = tr[feats].mean().to_numpy(), tr[feats].std().to_numpy()
        sd = np.where(sd == 0, 1.0, sd)
        w = fit_logit((tr[feats].to_numpy() - mu) / sd, tr.wrong_if_stop.to_numpy(float))
        sc = np.c_[np.ones(len(ev)), (ev[feats].to_numpy() - mu) / sd] @ w
        model_scores[name] = sc
        a = auc(sc, ev.wrong_if_stop.to_numpy(bool))
        lo, hi = boot_auc(sc, ev.wrong_if_stop.to_numpy(bool), ev.gid.to_numpy())
        res[name] = dict(auc=a, ci=[lo, hi], n_feat=len(feats))
        print(f"  {name:24s} AUC = {a:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

    a0 = auc(ev.dus, ev.wrong_if_stop.to_numpy(bool))
    lo, hi = boot_auc(ev.dus, ev.wrong_if_stop.to_numpy(bool), ev.gid.to_numpy())
    res["DUS hien tai (co san)"] = dict(auc=a0, ci=[lo, hi], n_feat=4)
    print(f"  {'DUS hien tai (co san)':24s} AUC = {a0:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

    # CI cua HIEU SO - dat SAU cac boot_auc o tren de khong doi RNG cua chung,
    # nen CI rieng tung model giu nguyen gia tri da bao cao.
    PAIRS = [("critic only", "base (DUS 4 feature)"),
             ("base + critic", "base (DUS 4 feature)"),
             ("base + critic", "critic only")]
    paired = boot_paired(model_scores, ev.wrong_if_stop.to_numpy(bool),
                         ev.gid.to_numpy(), PAIRS)
    print("\n=== Hieu AUC ghep cap (cung bo resample, clustered by question) ===")
    for k, v in paired.items():
        mark = "loai tru 0" if v["excludes_zero"] else "KHONG loai tru 0"
        print(f"  {k:42s} Delta = {v['delta']:+.4f}"
              f"   95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
              f"   p = {v['p_two_sided']:.3f}   -> {mark}")

    print("\n=== Theo benchmark (base + critic, fit tren train) ===")
    feats = BASE + CRITIC
    mu, sd = tr[feats].mean().to_numpy(), tr[feats].std().to_numpy()
    sd = np.where(sd == 0, 1.0, sd)
    w = fit_logit((tr[feats].to_numpy() - mu) / sd, tr.wrong_if_stop.to_numpy(float))
    per_task = {}
    for t, g in ev.groupby("task"):
        sc = np.c_[np.ones(len(g)), (g[feats].to_numpy() - mu) / sd] @ w
        v = auc(sc, g.wrong_if_stop.to_numpy(bool))
        per_task[t] = v
        print(f"  {t:12s} n={len(g):4d}  AUC={v:.3f}")

    Path("results/p0_critic_eval.json").write_text(json.dumps(
        dict(n_rounds=len(df), n_samples=int(df.gid.nunique()),
             base_rate=float(y.mean()), single_feature_auc=single,
             models=res, paired=paired, per_task=per_task), indent=2), encoding="utf-8")
    print("\n-> results/p0_critic_eval.json")


if __name__ == "__main__":
    main()
