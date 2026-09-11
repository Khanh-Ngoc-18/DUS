"""P3 - Chon nguong dung T tren du lieu DA GIU RIENG (tra loi review muc 17).

Van de cua p2_cost_accuracy.py: nguong T = np.quantile(dt.unc, q) duoc tinh tren
CHINH tap dang bao ket qua. Diem so unc da la out-of-fold (khong ro ri), nhung
NGUONG thi van "nhin thay" phan bo cua tap danh gia => ket qua co the dep hon thuc te.

Ban nay tach hoan toan ba vai tro:
    train  -> fit diem bat dinh (11 feature, chuan hoa theo train)
    val    -> chon nguong T_q = quantile(q) tren round KHONG-CUOI cua val
    test   -> DONG BANG T roi moi do accuracy / token / stop timing

Chay hai giao thuc de doi chieu:

  [A] NESTED CV  (ket qua CHINH)
      5 outer fold chia theo CAU HOI tren toan bo debate cua moi seed. Voi moi fold:
      phan con lai chia tiep inner-train (75%) / inner-val (25%) theo cau hoi;
      fit score tren inner-train, chon T tren inner-val, cham diem fold dang giu.
      => moi debate duoc danh gia dung 1 lan, out-of-fold, voi T no chua tung thay.
      Giu nguyen 4,500 debate nen khong mat power thong ke.

  [B] HOLDOUT CO DINH 70/20/10 (kiem chung doc lap)
      Dung dung mapping results/split/<bench>/split_sample_ids.json da co san.
      Sach ve mat khai niem nhung test chi ~150 debate/benchmark nen sai so rong.

  [C] DOI CHUNG "T IN-SAMPLE" (do do lon cua chinh loi bi phe binh)
      Dung Y HET diem so cua [A] tren dung nhung debate do, chi doi mot thu:
      T lay quantile tren CHINH fold dang cham thay vi tren inner-val.
      => hieu [C] - [A] la phan lac quan sinh ra RIENG boi viec chon nguong
      tren tap danh gia, khong lan voi bat ky khac biet nao khac.

Neu [A] va [B] cung dau va cung do lon thi ket luan khong phu thuoc cach chia.

Chinh sach: giong p2 (always / consensus / fixed_k / unc<T / oracle). Token uoc
luong theo total_tokens / so_round nhu p2.

Output: results/p3_holdout_policy.json
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

FEATS = p0.BASE + p0.CRITIC
QS = [0.1, 0.2, 0.3, 0.4, 0.5]
FIXED_K = [1, 2, 3, 4, 5]
N_OUTER = 5
INNER_VAL_FRAC = 0.25
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- du lieu
def load() -> pd.DataFrame:
    df = logio.load_rounds().dropna(subset=FEATS).reset_index(drop=True)
    df["gid"] = df.task + "_" + df.sample_id.astype(str)      # cum = CAU HOI
    sp = logio.load_splits()
    df["split"] = [sp.get((t, int(s)), "?") for t, s in zip(df.task, df.sample_id)]
    # round cuoi cua moi debate: noi khong the tiet kiem gi neu dung
    df["maxr"] = df.groupby(["seed", "task", "sample_id"]).round_id.transform("max")
    df["nonfinal"] = df.round_id < df.maxr
    return df


def fit_score(tr: pd.DataFrame, ap: pd.DataFrame) -> np.ndarray:
    """Fit logistic tren round KHONG-CUOI cua tr, tra ve diem cho MOI round cua ap."""
    t = tr[tr.nonfinal]
    mu = t[FEATS].mean().to_numpy()
    sd = t[FEATS].std().to_numpy()
    sd = np.where(sd == 0, 1.0, sd)
    w = p0.fit_logit((t[FEATS].to_numpy() - mu) / sd, (~t.ok_if_stop).to_numpy(float))
    return np.c_[np.ones(len(ap)), (ap[FEATS].to_numpy() - mu) / sd] @ w


def pick_thresholds(val: pd.DataFrame) -> dict[tuple[str, float], float]:
    """T_q = quantile(q) tren round KHONG-CUOI cua val - noi mot rule co the ra tay.

    Nguong RIENG cho tung benchmark (dung nhu p2): thang do bat dinh khac han giua
    gsm8k / mmlu / strategyqa nen mot nguong chung se lech han o hai dau.
    """
    nf = val[val.nonfinal]
    return {(t, q): float(np.quantile(g.unc.to_numpy(), q))
            for t, g in nf.groupby("task") for q in QS}


# ---------------------------------------------------------------- chinh sach
def stop_index(g: pd.DataFrame, kind: str, T: float = None, k: int = None) -> int:
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


def evaluate(ev: pd.DataFrame, T: dict[tuple[str, float], float]) -> pd.DataFrame:
    """1 dong / debate: dung o round nao, dung khong, ton bao nhieu token - moi chinh sach."""
    out = []
    for (seed, task, sid), g in ev.groupby(["seed", "task", "sample_id"], sort=False):
        g = g.sort_values("round_id")
        tpr = float(g.tok_per_round.iloc[0])
        rec = dict(seed=seed, task=task, sample_id=sid, gid=f"{task}_{sid}", tok_per_round=tpr,
                   n_rounds=len(g))
        pol = [("always", dict(kind="always")), ("consensus", dict(kind="consensus")),
               ("oracle", dict(kind="oracle"))]
        pol += [(f"fixed_k{k}", dict(kind="fixed", k=k)) for k in FIXED_K]
        pol += [(f"unc<q{int(q*100):02d}", dict(kind="unc", T=T[(task, q)])) for q in QS]
        for name, kw in pol:
            i = stop_index(g, **kw)
            rec[f"i_{name}"] = i
            rec[f"acc_{name}"] = bool(g.ok_if_stop.iloc[i])
            rec[f"tok_{name}"] = tpr * (i + 1)
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- giao thuc
def protocol_nested(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[A] Nested CV: moi debate cham diem out-of-fold, T tu inner-val.

    Tra ve them [C]: cung diem so, cung debate, chi khac T lay tu CHINH fold do
    (in-sample) => hieu hai ban do dung phan lac quan do chon nguong gay ra.
    """
    held, insample = [], []
    for seed, ds in df.groupby("seed", sort=True):
        gids = np.array(sorted(ds.gid.unique()))
        RNG.shuffle(gids)
        for fold in np.array_split(gids, N_OUTER):
            te = ds.gid.isin(fold)
            inner = ds[~te]
            ig = np.array(sorted(inner.gid.unique()))
            RNG.shuffle(ig)
            n_val = max(1, int(round(len(ig) * INNER_VAL_FRAC)))
            iv = set(ig[:n_val])
            itr = inner[~inner.gid.isin(iv)]
            ival = inner[inner.gid.isin(iv)].copy()
            ev = ds[te].copy()
            ival["unc"] = fit_score(itr, ival)
            ev["unc"] = fit_score(itr, ev)
            held.append(evaluate(ev, pick_thresholds(ival)))
            insample.append(evaluate(ev, pick_thresholds(ev)))     # <- chi khac dong nay
    return pd.concat(held, ignore_index=True), pd.concat(insample, ignore_index=True)


def protocol_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """[B] Holdout 70/20/10 co dinh: train fit score, val chon T, test bao ket qua."""
    parts = []
    for seed, ds in df.groupby("seed", sort=True):
        tr = ds[ds.split == "train"]
        val = ds[ds.split == "val"].copy()
        te = ds[ds.split == "test"].copy()
        if tr.empty or val.empty or te.empty:
            continue
        val["unc"] = fit_score(tr, val)
        te["unc"] = fit_score(tr, te)
        parts.append(evaluate(te, pick_thresholds(val)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- tong hop
POLICIES = ["always", "consensus"] + [f"fixed_k{k}" for k in FIXED_K] \
           + [f"unc<q{int(q*100):02d}" for q in QS] + ["oracle"]


def agg(xs) -> dict:
    xs = [float(x) for x in xs if x is not None and not np.isnan(x)]
    if not xs:
        return {"mean": None, "sd": None, "per_seed": []}
    return {"mean": float(np.mean(xs)),
            "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
            "per_seed": xs}


def boot_ci(vals: np.ndarray, gids: np.ndarray, n=2000) -> list[float]:
    """CI bootstrap resample theo CAU HOI (round/debate cung cau khong doc lap)."""
    uniq = np.unique(gids)
    idx = {g: np.where(gids == g)[0] for g in uniq}
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        out.append(float(np.mean(vals[sel])))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def summarise(res: pd.DataFrame, label: str) -> dict:
    """Bang accuracy/token theo benchmark + so sanh co cap voi consensus + stop timing."""
    out = {}
    print(f"\n{'=' * 78}\n[{label}]  n_debate = {len(res)}\n{'=' * 78}")
    for task, dt in res.groupby("task"):
        seeds = sorted(dt.seed.unique())
        print(f"\n--- {task}  ({len(dt)} debate / {dt.gid.nunique()} cau / {len(seeds)} seed) ---")
        print(f"{'chinh sach':<14}{'accuracy':>16}{'token% vs always':>20}")
        pol_out = {}
        base_tok = {s: dt[dt.seed == s].tok_always.sum() for s in seeds}
        for name in POLICIES:
            a = agg([dt[dt.seed == s][f"acc_{name}"].mean() for s in seeds])
            t = agg([dt[dt.seed == s][f"tok_{name}"].sum() / base_tok[s] for s in seeds])
            print(f"{name:<14}{a['mean']:>8.3f} ± {a['sd']:<5.3f}{t['mean']*100:>13.1f}% ± {t['sd']*100:<4.1f}%")
            pol_out[name] = {"acc": a, "tok": t}

        # so sanh co cap voi consensus: chenh lech TUNG SEED roi mean/SD qua seed,
        # kem CI bootstrap gop qua seed (cluster theo cau) cho dAcc.
        print("  paired vs consensus:")
        pr = {}
        for q in QS:
            name = f"unc<q{int(q*100):02d}"
            da, dtk = [], []
            for s in seeds:
                d = dt[dt.seed == s]
                da.append(d[f"acc_{name}"].mean() - d.acc_consensus.mean())
                tc = d.tok_consensus.sum()
                dtk.append((d[f"tok_{name}"].sum() - tc) / tc)
            diff = (dt[f"acc_{name}"].astype(float) - dt.acc_consensus.astype(float)).to_numpy()
            ci = boot_ci(diff, dt.gid.to_numpy())
            A, T_ = agg(da), agg(dtk)
            print(f"    {name}: dAcc={A['mean']:+.3f} ± {A['sd']:.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"   dToken={T_['mean']*100:+.1f}% ± {T_['sd']*100:.1f}%")
            pr[name] = {"dacc": A, "dtok": T_, "dacc_ci_clustered": ci}

        # stop timing: unc dung SOM HON / BANG / MUON HON consensus
        print("  stop timing vs consensus (earlier / same / later):")
        tm = {}
        for q in QS:
            name = f"unc<q{int(q*100):02d}"
            d = dt[f"i_{name}"].to_numpy() - dt.i_consensus.to_numpy()
            e, sm, l = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
            n = len(d)
            print(f"    {name}: {e:5d} ({e/n*100:4.1f}%) / {sm:5d} ({sm/n*100:4.1f}%) / {l:5d} ({l/n*100:4.1f}%)")
            tm[name] = {"earlier": e, "same": sm, "later": l, "n": n,
                        "earlier_pct": e / n * 100, "same_pct": sm / n * 100, "later_pct": l / n * 100}

        out[task] = {"n_debate": len(dt), "n_question": int(dt.gid.nunique()),
                     "policies": pol_out, "paired_vs_consensus": pr, "stop_timing": tm}
    return out



# ================================================================ [D] baseline khong-debate
# Gop tu p6_baseline_showdown.py. Cac baseline nay KHONG doc transcript va duoc chay
# rieng (results/logs_mv, results/logs_sc), nen chi phi token cua chung la DO TRUC
# TIEP chu khong phai xap xi total/6 nhu cac chinh sach suy ra tu log debate.
N_BOOT = 10000
BOOT_RNG = np.random.default_rng(12345)


def long_table(nested: pd.DataFrame) -> pd.DataFrame:
    """Bang dai 1 dong / (debate, chinh sach) tu ket qua nested CV cua [A]."""
    pols = [c[len("acc_"):] for c in nested.columns if c.startswith("acc_")]
    return pd.concat([pd.DataFrame({
        "seed": nested.seed, "task": nested.task, "sample_id": nested.sample_id,
        "gid": nested.gid, "policy": p,
        "correct": nested["acc_" + p].astype(bool),
        "tok": nested["tok_" + p].astype(float)}) for p in pols], ignore_index=True)


def add_no_debate(long: pd.DataFrame) -> pd.DataFrame:
    """Gan ensemble vote (logs_mv) va self-consistency (logs_sc) neu da chay."""
    import glob as _glob
    extra = []
    for pattern, fixed in (("results/logs_mv/**/majority_vote_*.jsonl", "ensemble_vote"),
                           ("results/logs_sc/**/self_consistency_*.jsonl", None)):
        for f in _glob.glob(pattern, recursive=True):
            name = fixed
            if name is None:
                parts = Path(f).as_posix().split("/")
                agent = next((x for x in parts if x.startswith("agent_")), "agent_?")
                name = "self_consistency_" + agent[-1]
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                extra.append(dict(seed=int(r["seed"]), task=r["task"],
                                  sample_id=r["sample_id"],
                                  gid=r["task"] + "_" + str(r["sample_id"]), policy=name,
                                  correct=bool(r["final_correct"]),
                                  tok=float(r["total_tokens"])))
    if not extra:
        return long
    ex = pd.DataFrame(extra).drop_duplicates(
        subset=["seed", "task", "sample_id", "policy"], keep="first")
    return pd.concat([long, ex], ignore_index=True)


def cluster_bootstrap(long: pd.DataFrame, task: str, pa: str, pb: str) -> dict:
    """Delta = pa - pb tren mot benchmark, CI bootstrap PHAN CUM THEO CAU HOI.

    Phan cum la bat buoc chu khong phai tuy chon: cac seed rut chong lan nhau tu
    cung mot pool cau hoi (1500 luot rut -> 978 / 1123 / 647 cau duy nhat), nen
    coi moi seed la mot lan lap doc lap se danh gia THAP do bat dinh that.
    """
    d = long[long.task == task]
    a = d[d.policy == pa].set_index(["seed", "sample_id"])
    b = d[d.policy == pb].set_index(["seed", "sample_id"])
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return {}
    a, b = a.loc[idx], b.loc[idx]

    gid = a.gid.to_numpy()
    dacc = a.correct.to_numpy(float) - b.correct.to_numpy(float)
    dtok = a.tok.to_numpy(float) - b.tok.to_numpy(float)
    base = long[(long.task == task) & (long.policy == "always")].tok.mean()

    uniq = np.unique(gid)
    pos = {}
    for j, g in enumerate(gid):
        pos.setdefault(g, []).append(j)
    pos = {g: np.array(v) for g, v in pos.items()}

    bacc = np.empty(N_BOOT)
    btok = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sel = np.concatenate([pos[g] for g in BOOT_RNG.choice(uniq, len(uniq), replace=True)])
        bacc[i] = dacc[sel].mean()
        btok[i] = dtok[sel].mean() / base

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # min(...,1.0): phan phoi doi xung quanh 0 co the cho 2*min(...) > 1
    p = min(2 * min((bacc <= 0).mean(), (bacc >= 0).mean()), 1.0)
    return {"n_pairs": int(len(idx)), "n_clusters": int(len(uniq)),
            "dacc": float(dacc.mean()), "dacc_ci": ci(bacc),
            "dacc_p": float(max(p, 1.0 / N_BOOT)),
            "dtok_rel": float(dtok.mean() / base), "dtok_ci": ci(btok)}


def baseline_showdown(nested: pd.DataFrame) -> dict:
    """[D] Moi chinh sach tren cung mot mat phang + kiem dinh so voi DUS-11."""
    long = add_no_debate(long_table(nested))
    tasks = sorted(long.task.unique())
    fixed = ["fixed_k" + str(k) for k in FIXED_K]
    unc = ["unc<q%02d" % int(q * 100) for q in QS]
    nodebate = sorted({p for p in long.policy.unique()
                       if p.startswith(("ensemble", "self_cons"))})

    print("\n" + "=" * 78)
    print("D. MOI CHINH SACH TREN CUNG MOT MAT PHANG")
    print("=" * 78)
    summary = {}
    for t in tasks:
        d = long[long.task == t]
        base = d[d.policy == "always"].tok.mean()
        print("\n--- %s ---" % t)
        rec = {}
        for p in ["always", "consensus"] + fixed + unc + nodebate + ["oracle"]:
            g = d[d.policy == p]
            if g.empty:
                continue
            rec[p] = {"n": int(len(g)), "acc": float(g.correct.mean()),
                      "tok_rel": float(g.tok.mean() / base)}
            print("  %-22s%6d%9.4f%7.1f%%" % (p, len(g), g.correct.mean(),
                                              g.tok.mean() / base * 100))
        summary[t] = rec

    n_comp = len(tasks) * (len(fixed) + len(nodebate))
    print("\n" + "=" * 78)
    print("KIEM DINH SO VOI DUS-11 (unc<q50), bootstrap phan cum theo cau hoi")
    print("=" * 78)
    print("  %d phep so sanh trong ho khong-doc-transcript -> Bonferroni p < %.5f\n"
          % (n_comp, 0.05 / n_comp))
    print("  %-12s%-22s%9s%22s%9s%9s" % ("benchmark", "baseline", "dacc", "CI 95%", "p", "dtok"))
    vs_dus, vs_always = {}, {}
    for t in tasks:
        for p in sorted(long[long.task == t].policy.unique()):
            if p != "unc<q50":
                r = cluster_bootstrap(long, t, p, "unc<q50")
                if r:
                    vs_dus[t + "|" + p] = r
                    if p in fixed + nodebate:
                        star = (" **" if r["dacc_p"] < 0.05 / n_comp
                                else (" *" if r["dacc_p"] < 0.05 else ""))
                        ci = "[%+.4f, %+.4f]" % (r["dacc_ci"][0], r["dacc_ci"][1])
                        print("  %-12s%-22s%+9.4f%22s%9.4f%+8.1f%%%s"
                              % (t, p, r["dacc"], ci, r["dacc_p"],
                                 r["dtok_rel"] * 100, star))
            if p != "always":
                r = cluster_bootstrap(long, t, p, "always")
                if r:
                    vs_always[t + "|" + p] = r
    print("\n  ** vuot nguong Bonferroni  |  * chi co y nghia danh nghia")

    # fixed_k2 so voi CA HAI muc nguong cua DUS-11 (TABLE XI cua paper): q=0.5 da
    # co trong vs_dus11, con q=0.4 phai tinh rieng.
    fk2 = {}
    for t in tasks:
        for q in ("unc<q40", "unc<q50"):
            r = cluster_bootstrap(long, t, "fixed_k2", q)
            if r:
                fk2[t + "|fixed_k2-" + q] = r

    return {"n_boot": N_BOOT, "n_comparisons": n_comp, "summary": summary,
            "fixed_k2_vs_dus11": fk2, "vs_dus11": vs_dus, "vs_always": vs_always}


def main() -> None:
    df = load()
    print(f"{len(df)} round | {df.groupby(['seed','task','sample_id']).ngroups} debate | "
          f"seeds={sorted(df.seed.unique())}")
    print(f"round khong-cuoi: {int(df.nonfinal.sum())}")

    nested, insample = protocol_nested(df)
    holdout = protocol_holdout(df)

    A = summarise(nested, "A. NESTED CV - T tu inner-val, cham fold giu rieng")
    B = summarise(holdout, "B. HOLDOUT 70/20/10 - T tu val, cham tren test")
    C = summarise(insample, "C. DOI CHUNG - cung diem so nhu [A], T lay in-sample")

    # do lon phan lac quan do chon nguong: [C] - [A] tren cung debate, cung diem so
    print(f"\n{'=' * 78}\nPHAN LAC QUAN DO CHON NGUONG IN-SAMPLE  ([C] - [A])\n{'=' * 78}")
    gap = {}
    for task in sorted(A):
        gap[task] = {}
        for name in A[task]["paired_vs_consensus"]:
            a = A[task]["paired_vs_consensus"][name]
            c = C[task]["paired_vs_consensus"][name]
            dt = (c["dtok"]["mean"] - a["dtok"]["mean"]) * 100
            da = c["dacc"]["mean"] - a["dacc"]["mean"]
            gap[task][name] = {"dtok_pp": dt, "dacc": da,
                               "held_out_dtok": a["dtok"]["mean"] * 100,
                               "in_sample_dtok": c["dtok"]["mean"] * 100}
            print(f"  {task:12s} {name}: dToken giu-rieng={a['dtok']['mean']*100:+.1f}%  "
                  f"in-sample={c['dtok']['mean']*100:+.1f}%  => lac quan {-dt:+.1f} diem %")

    showdown = baseline_showdown(nested)

    out = {
        "note": "T chon tren du lieu giu rieng (nested inner-val hoac val cua holdout), "
                "khong bao gio tren tap dang bao ket qua. [C] la doi chung in-sample.",
        "n_outer_folds": N_OUTER, "inner_val_frac": INNER_VAL_FRAC,
        "nested_cv": A, "fixed_holdout": B, "in_sample_control": C,
        "threshold_selection_optimism": gap,
        "baseline_showdown": showdown,
    }
    Path("results/p3_holdout_policy.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Bang per-debate cua [A] nested CV - ket qua CHINH, nguong lay tu inner-val.
    # Xuat de kiem chung doc lap; moi phan tich trong repo doc tu JSON o tren.
    nested.to_csv("results/p3_per_debate.csv", index=False)
    print("  -> results/p3_per_debate.csv "
          f"({len(nested):,} debate x {sum(c.startswith('acc_') for c in nested.columns)} chinh sach)")
    print("\n-> results/p3_holdout_policy.json")


if __name__ == "__main__":
    main()
