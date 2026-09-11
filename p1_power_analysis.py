"""P1 - Phan tich do manh thong ke (power).

Cau hoi: voi AUC quan sat duoc (~0.6), can bao nhieu SAMPLE de CI 95%
(bootstrap theo cluster sample_id) loai tru duoc 0.5?

Uoc luong AUC bang 5-fold CV theo sample => diem out-of-fold cho ca 200 sample,
sau do resample cum de mo phong cac co mau lon hon.

Output: results/p1_power.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("p0", "p0_critic_features.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)

RNG = np.random.default_rng(0)


def fast_auc(s: np.ndarray, y: np.ndarray) -> float:
    """AUC vector hoa hoan toan; tie dung rank trung binh."""
    n = len(s)
    o = np.argsort(s, kind="mergesort")
    _, inv, cnt = np.unique(s[o], return_inverse=True, return_counts=True)
    start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    avg = start + (cnt - 1) / 2.0 + 1.0     # rank trung binh 1-based cho moi nhom tie
    r = np.empty(n, dtype=float)
    r[o] = avg[inv]
    yb = y.astype(bool)
    a, b = yb.sum(), n - yb.sum()
    return np.nan if a == 0 or b == 0 else (r[yb].sum() - a * (a + 1) / 2) / (a * b)


def main() -> None:
    df = p0.build().reset_index(drop=True)
    df["gid"] = df.task + "_" + df.sample_id.astype(str)
    feats = p0.BASE + p0.CRITIC

    # ---- 5-fold CV theo sample -> diem out-of-fold khong thien lech ----
    gids = np.array(df.gid.unique().copy(), copy=True)
    RNG.shuffle(gids)
    oof = np.zeros(len(df))
    for fold in np.array_split(gids, 5):
        te = df.gid.isin(fold).to_numpy()
        tr = ~te
        mu = df.loc[tr, feats].mean().to_numpy()
        sd = df.loc[tr, feats].std().to_numpy()
        sd = np.where(sd == 0, 1.0, sd)
        w = p0.fit_logit((df.loc[tr, feats].to_numpy() - mu) / sd,
                         df.loc[tr, "wrong_if_stop"].to_numpy(float))
        oof[te] = np.c_[np.ones(te.sum()), (df.loc[te, feats].to_numpy() - mu) / sd] @ w

    y = df.wrong_if_stop.to_numpy(bool)
    A = fast_auc(oof, y)

    groups = df.gid.to_numpy()
    uniq = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}

    def boot_ci(score, lab, segs, reps=400):
        n = len(segs)
        off = np.cumsum([0] + [len(x) for x in segs])
        out = []
        for _ in range(reps):
            p = RNG.integers(0, n, n)
            ii = np.concatenate([np.arange(off[j], off[j + 1]) for j in p])
            v = fast_auc(score[ii], lab[ii])
            if not np.isnan(v):
                out.append(v)
        return np.percentile(out, [2.5, 97.5])

    segs0 = [pos[g] for g in uniq]
    sel0 = np.concatenate(segs0)
    lo, hi = boot_ci(oof[sel0], y[sel0], segs0)
    print(f"AUC 5-fold CV: {A:.3f}   95% CI [{lo:.3f}, {hi:.3f}]"
          f"   ({len(df)} round / {len(uniq)} sample)")
    print(f"  -> CI {'LOAI TRU' if lo > 0.5 else 'CHUA loai tru'} 0.5\n")

    print(f"Power de CI95 loai tru 0.5 (gia dinh AUC that = {A:.3f}):")
    print(f"{'N sample':>10} {'power':>8} {'CI rong TB':>12}")
    res = {}
    for N in [65, 100, 200, 300, 500, 800, 1200]:
        hits, wds, T = 0, [], 120
        for _ in range(T):
            pick = RNG.choice(uniq, N, replace=True)
            segs = [pos[g] for g in pick]
            sel = np.concatenate(segs)
            s2, y2 = oof[sel], y[sel]
            if y2.sum() == 0 or y2.all():
                continue
            l, h = boot_ci(s2, y2, segs, reps=200)
            wds.append(h - l)
            hits += int(l > 0.5)
        res[N] = dict(power=hits / T, ci_width=float(np.mean(wds)))
        print(f"{N:>10} {hits / T:>7.0%} {np.mean(wds):>12.3f}")

    Path("results/p1_power.json").write_text(json.dumps(
        dict(auc_cv=float(A), ci=[float(lo), float(hi)],
             n_rounds=len(df), n_samples=int(len(uniq)), power=res),
        indent=2), encoding="utf-8")
    print("\n-> results/p1_power.json")


if __name__ == "__main__":
    main()
