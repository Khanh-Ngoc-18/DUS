"""P2 - Duong cost-accuracy (BAN MULTI-SEED + PER-BENCHMARK).

Cau hoi: chinh sach dung som bang diem bat dinh co danh bai baseline khong,
va ket qua do co ON DINH qua nhieu seed khong (khong phai an may 1 lan chay)?

Khac ban cu:
  - Doc TRUC TIEP tu results/logs/**/debate_full_*.jsonl (khong can dus_per_round.csv).
    Tu nhan dien nhieu seed qua field "seed" trong moi ban ghi (mac dinh 0 neu thieu).
  - Moi seed = 1 lan lap doc lap: fit 5-fold CV rieng => 1 bo metric/seed.
  - Bao cao TUNG BENCHMARK rieng, nguong dung theo tung benchmark (muc 5 review),
    KHONG gop 3 benchmark lai.
  - Tong hop qua cac seed: mean +/- SD cho moi (benchmark, chinh sach) => y nghia thong ke.
  - So sanh co cap unc vs consensus tren tung seed roi mean +/- SD qua seed.

Chinh sach (tat ca suy ra offline tu log chay --no_early_stop, KHONG chay lai LLM):
  always      - debate den het (he thong goc)
  consensus   - dung khi 3 agent thong nhat (= repo hien tai)
  fixed k     - luon dung sau dung k round
  unc < T     - dung khi diem bat dinh thap (T = quantile theo tung benchmark)
  oracle      - can tren: dung o round dau tien ma dap an dung
  self-cons@3 - = fixed_k1: round 0 chua co thong tin cheo (orchestrator chi truyen
                critic_messages khi round_id > 0), nen majority-vote 3 agent tai round 0
                CHINH LA self-consistency 3 mau doc lap khong debate = baseline (iii).

Bang so sanh baseline bat buoc cua review muc 10C duoc IN o cuoi (print_baseline_table):
unc vs (i) consensus, (ii) fixed k re nhat cung ngan sach, (iii) self-consistency@3.

Uoc luong token: total_tokens / so_round (log chi co token tong moi cau). Dung xap xi
uniform token/round nhu ban cu.

Output: results/p2_cost_accuracy.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import logio

_s = importlib.util.spec_from_file_location("p0", "p0_critic_features.py")
p0 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(p0)
_s = importlib.util.spec_from_file_location("p1", "p1_power_analysis.py")
p1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(p1)
auc = p1.fast_auc
RNG = np.random.default_rng(0)

FEATS = p0.BASE + p0.CRITIC
QS = [0.1, 0.2, 0.3, 0.4, 0.5]          # nguong = quantile diem bat dinh (theo tung benchmark)
FIXED_K = [1, 2, 3, 4, 5]
PAIRED_QS = [0.1, 0.2, 0.3, 0.4, 0.5]             # so sanh co cap unc vs consensus


def load_rounds() -> pd.DataFrame:
    """Doc round tu jsonl qua nguon chan ly chung (logio)."""
    return logio.load_rounds().dropna(subset=FEATS).reset_index(drop=True)


def cv_unc(ds: pd.DataFrame) -> np.ndarray:
    """Diem bat dinh out-of-fold: 5-fold CV theo sample (gop 3 benchmark trong CUNG 1 seed).
    Fit tren train => cham cho val => khong ro ri giua cac round cua cung 1 cau."""
    gids = np.array(ds.gid.unique(), copy=True)
    RNG.shuffle(gids)
    oof = np.zeros(len(ds))
    for fold in np.array_split(gids, 5):
        te = ds.gid.isin(fold).to_numpy()
        tr = ~te
        mu = ds.loc[tr, FEATS].mean().to_numpy()
        sd = ds.loc[tr, FEATS].std().to_numpy()
        sd = np.where(sd == 0, 1.0, sd)
        w = p0.fit_logit((ds.loc[tr, FEATS].to_numpy() - mu) / sd,
                         (~ds.loc[tr, "ok_if_stop"]).to_numpy(float))
        oof[te] = np.c_[np.ones(te.sum()), (ds.loc[te, FEATS].to_numpy() - mu) / sd] @ w
    return oof


def stop_index(g: pd.DataFrame, kind: str, T: float = None, k: int = None) -> int:
    """Chi so dong trong nhom g (1 cau hoi) ma chinh sach quyet dinh dung."""
    if kind == "always":
        return len(g) - 1
    if kind == "fixed":
        return min(k, len(g)) - 1
    if kind == "consensus":
        w = np.where(g.consensus_now.to_numpy())[0]
        return int(w[0]) if len(w) else len(g) - 1
    if kind == "unc":
        w = np.where(g.unc.to_numpy() < T)[0]
        return int(w[0]) if len(w) else len(g) - 1
    if kind == "oracle":
        w = np.where(g.ok_if_stop.to_numpy())[0]
        return int(w[0]) if len(w) else len(g) - 1
    raise ValueError(kind)


def policy(groups: list, kind: str, **kw) -> tuple[float, float]:
    """(accuracy, tong token) cua 1 chinh sach tren list cac nhom cau hoi."""
    acc, tok = [], []
    for g in groups:
        i = stop_index(g, kind, **kw)
        acc.append(bool(g.ok_if_stop.iloc[i]))
        tok.append(g.tok_per_round.iloc[0] * (i + 1))
    return float(np.mean(acc)), float(np.sum(tok))


def agg(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return {"mean": None, "sd": None, "per_seed": []}
    return {"mean": float(np.mean(xs)),
            "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
            "per_seed": [float(x) for x in xs]}


def paired_delta(a: list[float], b: list[float]) -> tuple[float, float]:
    """Chenh lech TUNG SEED roi mean/SD (so cap: cung seed = cung tap cau)."""
    d = [x - y for x, y in zip(a, b)]
    return float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0


def print_baseline_table(tasks, acc, tok, seeds) -> None:
    """Muc 10C review: dat unc canh 3 baseline bat buoc, o CUNG ngan sach token.

    (i)   consensus              - baseline hien tai cua repo
    (ii)  fixed k                - chay cung k round
    (iii) self-consistency@3     - = fixed_k1. Round 0 la round DUY NHAT khong co thong
          tin cheo (orchestrator.py:115 chi truyen critic_messages khi round_id > 0),
          nen majority-vote 3 agent tai round 0 chinh la self-consistency 3 mau doc lap
          (2 solver + critic tra loi doc lap), KHONG debate. Suy ra offline, khong chay lai LLM.
    """
    fixed = [f"fixed_k{k}" for k in FIXED_K]
    unc_names = [f"unc<q{int(q * 100):02d}" for q in QS]

    for t in tasks:
        if "always" not in acc[t]:
            continue
        def sd(name):
            return float(np.std(acc[t][name], ddof=1)) if len(acc[t][name]) > 1 else 0.0

        print(f"\n--- {t} ---")
        print(f"  {'baseline':<15}{'token%':>8}{'acc ± SD':>16}")
        tags = {"fixed_k1": "(iii)", "fixed_k2": "(ii)"}
        for name, tagb in ([("always", "goc"), ("consensus", "(i)")]
                           + [(f, tags.get(f, "     ")) for f in fixed]):
            if name not in acc[t]:
                continue
            print(f"  {tagb + ' ' + name:<15}{np.mean(tok[t][name]) * 100:>7.1f}%"
                  f"{np.mean(acc[t][name]):>8.3f} ± {sd(name):<5.3f}")

        print(f"  {'chinh sach unc':<15}{'token%':>8}{'acc ± SD':>16}   "
              f"{'dAcc vs (i)':>14}{'dAcc vs (iii)':>15}   {'dAcc vs (ii) re nhat':<24}")
        for name in unc_names:
            if name not in acc[t]:
                continue
            ta, aa = float(np.mean(tok[t][name])), float(np.mean(acc[t][name]))
            cells = []
            for base in ["consensus", "fixed_k1"]:
                m, s = paired_delta(acc[t][name], acc[t][base])
                cells.append(f"{m:+.3f}±{s:.3f}")
            # (ii) fixed k TOT NHAT trong so nhung k khong dat hon unc => so sanh cung ngan sach
            cand = [f for f in fixed if f in acc[t] and np.mean(tok[t][f]) <= ta + 1e-9]
            if cand:
                bk = max(cand, key=lambda f: np.mean(acc[t][f]))
                m, s = paired_delta(acc[t][name], acc[t][bk])
                cells.append(f"{bk} {m:+.3f}±{s:.3f}")
            else:
                cells.append("(khong co fixed k re hon)")
            print(f"  {name:<15}{ta * 100:>7.1f}%{aa:>8.3f} ± {sd(name):<5.3f}   "
                  f"{cells[0]:>14}{cells[1]:>15}   {cells[2]:<24}")



def main() -> None:
    df = load_rounds()
    seeds = sorted(int(s) for s in df.seed.unique())
    tasks = sorted(df.task.unique())
    print(f"{len(df)} round | {df.groupby(['seed','task']).ngroups} (seed,benchmark) | "
          f"seeds={seeds}\n")

    # ---- gom ket qua tung seed roi tong hop ----
    # acc[task][policy] = [gia tri moi seed];  tok[task][policy] = [ty le token/always moi seed]
    acc = {t: {} for t in tasks}
    tok = {t: {} for t in tasks}
    auc_nc = {t: [] for t in tasks}                       # AUC round khong-cuoi moi seed
    paired = {t: {f"unc<q{int(q*100):02d}": {"dacc": [], "dtok": []} for q in PAIRED_QS}
              for t in tasks}

    def add(store, t, name, v):
        store[t].setdefault(name, []).append(v)

    parts = []                                            # gom (seed, unc) de lam CI gop qua seed
    for seed in seeds:
        ds = df[df.seed == seed].copy().reset_index(drop=True)
        ds["gid"] = ds.task + "_" + ds.sample_id.astype(str)   # cluster theo CAU (khong theo seed)
        ds["unc"] = cv_unc(ds)
        parts.append(ds[["seed", "task", "sample_id", "round_id", "gid", "unc", "ok_if_stop"]])

        for t in tasks:
            dt = ds[ds.task == t]
            if dt.empty:
                continue
            groups = [g for _, g in dt.groupby("sample_id", sort=False)]

            base_acc, base_tok = policy(groups, "always")
            add(acc, t, "always", base_acc)
            add(tok, t, "always", 1.0)

            a, tk = policy(groups, "consensus")
            add(acc, t, "consensus", a); add(tok, t, "consensus", tk / base_tok)

            for k in FIXED_K:
                a, tk = policy(groups, "fixed", k=k)
                add(acc, t, f"fixed_k{k}", a); add(tok, t, f"fixed_k{k}", tk / base_tok)

            for q in QS:
                T = float(np.quantile(dt.unc, q))          # nguong RIENG cho benchmark nay
                a, tk = policy(groups, "unc", T=T)
                add(acc, t, f"unc<q{int(q*100):02d}", a)
                add(tok, t, f"unc<q{int(q*100):02d}", tk / base_tok)

            a, tk = policy(groups, "oracle")
            add(acc, t, "oracle", a); add(tok, t, "oracle", tk / base_tok)

            # AUC tren round KHONG-cuoi (loai round cuoi moi cau)
            maxr = dt.groupby("sample_id").round_id.transform("max")
            nc = dt[dt.round_id < maxr]
            if nc.sample_id.nunique() and nc.ok_if_stop.nunique() > 1:
                auc_nc[t].append(auc(nc.unc.to_numpy(), (~nc.ok_if_stop).to_numpy()))

            # so sanh co cap unc vs consensus (cung tap cau) tren seed nay
            for q in PAIRED_QS:
                T = float(np.quantile(dt.unc, q))
                iu = np.array([stop_index(g, "unc", T=T) for g in groups])
                ic = np.array([stop_index(g, "consensus") for g in groups])
                au = np.mean([bool(g.ok_if_stop.iloc[i]) for g, i in zip(groups, iu)])
                ac = np.mean([bool(g.ok_if_stop.iloc[i]) for g, i in zip(groups, ic)])
                tu = np.sum([g.tok_per_round.iloc[0] * (i + 1) for g, i in zip(groups, iu)])
                tc = np.sum([g.tok_per_round.iloc[0] * (i + 1) for g, i in zip(groups, ic)])
                key = f"unc<q{int(q*100):02d}"
                paired[t][key]["dacc"].append(au - ac)
                paired[t][key]["dtok"].append((tu - tc) / tc)

    # ---- in bang tung benchmark ----
    out = {"n_seeds": len(seeds), "seeds": [int(s) for s in seeds], "per_benchmark": {}}
    for t in tasks:
        print(f"=== BENCHMARK {t}  (K={len(seeds)} seed) ===")
        print(f"{'chinh sach':<14} {'accuracy':>16} {'token% vs always':>20} {'dAcc':>9}")
        base_mean = np.mean(acc[t]["always"])
        pol_out = {}
        order = (["always", "consensus"] + [f"fixed_k{k}" for k in FIXED_K]
                 + [f"unc<q{int(q*100):02d}" for q in QS] + ["oracle"])
        for name in order:
            if name not in acc[t]:
                continue
            aa, tt = agg(acc[t][name]), agg(tok[t][name])
            d = aa["mean"] - base_mean
            print(f"{name:<14} {aa['mean']:>7.3f} ± {aa['sd']:<5.3f} "
                  f"{tt['mean']*100:>10.1f}% ± {tt['sd']*100:<4.1f}% {d:>+9.3f}")
            pol_out[name] = {"acc": aa, "tok": tt, "dacc_vs_always": d}
        au = agg(auc_nc[t])
        if au["mean"] is not None:
            print(f"{'AUC(kh-cuoi)':<14} {au['mean']:>7.3f} ± {au['sd']:<5.3f}"
                  f"   (>0.5 = co tin hieu)")

        print("  paired unc vs consensus (mean±SD qua seed):")
        pr_out = {}
        for key, dd in paired[t].items():
            da, dt_ = agg(dd["dacc"]), agg(dd["dtok"])
            if da["mean"] is None:
                continue
            print(f"    {key}:  dAcc={da['mean']:+.3f} ± {da['sd']:.3f}"
                  f"   dToken={dt_['mean']*100:+.1f}% ± {dt_['sd']*100:.1f}%")
            pr_out[key] = {"dacc": da, "dtok": dt_}
        print()
        out["per_benchmark"][t] = {"policies": pol_out, "auc_noncuoi": au,
                                   "paired_vs_consensus": pr_out}

    # ---- CI GOP QUA SEED cho AUC tung benchmark (round khong-cuoi, cluster theo CAU) ----
    # Cau bi lap lai qua nhieu seed => don vi doc lap la CAU HOI, khong phai rollout.
    # boot_auc resample cum theo gid (cau) => CI khong thoi phong khi seed dung lai cau.
    scored = pd.concat(parts, ignore_index=True)
    print("=== AUC gop qua seed (round khong-cuoi, bootstrap cluster theo cau) ===")
    pooled = {}
    for t in tasks:
        st = scored[scored.task == t]
        maxr = st.groupby(["seed", "sample_id"]).round_id.transform("max")
        nc = st[st.round_id < maxr]
        if nc.empty or nc.ok_if_stop.nunique() < 2:
            continue
        s = nc.unc.to_numpy()
        y = (~nc.ok_if_stop).to_numpy()
        a = p0.auc(s, y)
        lo, hi = p0.boot_auc(s, y, nc.gid.to_numpy())
        n_q = int(nc.gid.nunique())                       # so cau DOC LAP (bi chan boi kich thuoc pool)
        roll = float(nc.groupby("gid").seed.nunique().mean())   # so rollout trung binh / cau
        flag = "  <- CI loai tru 0.5" if lo > 0.5 else "  (CHUA loai tru 0.5)"
        print(f"  {t:12s} AUC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]  "
              f"cau doc lap={n_q}  rollout/cau~{roll:.1f}{flag}")
        pooled[t] = dict(auc=float(a), ci=[float(lo), float(hi)],
                         n_distinct_q=n_q, avg_rollout=roll)
    out["pooled_auc_across_seeds"] = pooled
    print()

    print_baseline_table(tasks, acc, tok, seeds)

    Path("results/p2_cost_accuracy.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-> results/p2_cost_accuracy.json")


if __name__ == "__main__":
    main()
