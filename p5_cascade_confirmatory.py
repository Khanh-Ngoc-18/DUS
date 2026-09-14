"""P5 - Xac nhan cascade critic-SC: on dinh, cong dung that, frontier, rescue/hurt, va ablation.

Bon manh rieng, GOP mot cho de dung chung du lieu da nap (p3.load/protocol_nested):

  [1] ON DINH CRITIC-SC qua N (so phieu SC, k=1..3) va SEED (5 lan chay doc lap).
  [2] CASCADE CO CONG DUNG THAT (nhu nguyen ban).
  [3] FRONTIER CASCADE vs DUONG CONG SC-SOLO THEO k.
  [5] fixed_k4 / fixed_k5 SO VOI fixed_k2, GHEP CAP.
  [6] ABLATION STAGE 1 VS STAGE 2: Phan ra luong dong gop tiet kiem token (truoc consensus) 
      den tu viec chan ngay tai Round 0 (Stage 1) so voi dung o cac round giua (Stage 2).
  [7] CASCADE COMPONENT ABLATION: Phan tach policy thanh 3 bien the (Full, Stage1-only, Stage2-only) 
      qua 5-fold nested CV (RNG=0).

Output: results/p5_cascade_confirmatory.json (+ in tat ca ra stdout)
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import logio

_s = importlib.util.spec_from_file_location("p3", "p3_holdout_policy.py")
p3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(p3)

TASKS = ["gsm8k", "mmlu", "strategyqa"]
QS = p3.QS
RNG = np.random.default_rng(2026)


# ============================================================ [1] on dinh critic-SC
def _sc_records(pattern: str) -> dict[tuple[int, str, int], list[dict]]:
    """(seed, task, sample_id) -> danh sach phieu (vote) THEO DUNG THU TU vote_id."""
    out: dict[tuple[int, str, int], list[dict]] = {}
    for f in glob.glob(pattern, recursive=True):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (int(r.get("seed", 0)), r["task"], int(r["sample_id"]))
            votes = sorted(r.get("votes", []), key=lambda v: v["vote_id"])
            out[key] = votes
    return out


def _vote_majority(votes: list[dict]) -> tuple[str, float]:
    cands = [(str(v["normalized_answer"]).strip(), float(v["confidence"])) for v in votes]
    return logio.select_final(cands), float(np.mean([c for _, c in cands]))


def critic_sc_stability() -> tuple[dict, pd.DataFrame]:
    recs = _sc_records("results/logs_sc/agent_c/**/self_consistency_*.jsonl")

    by_n = {t: {"n": 0, "flip_1_2": 0, "flip_2_3": 0, "flip_1_3": 0,
                "conf_gap_1_3": []} for t in TASKS}
    for (seed, task, sid), votes in recs.items():
        if len(votes) < 3 or task not in by_n:
            continue
        a1, c1 = _vote_majority(votes[:1])
        a2, c2 = _vote_majority(votes[:2])
        a3, c3 = _vote_majority(votes[:3])
        d = by_n[task]
        d["n"] += 1
        d["flip_1_2"] += int(a1 != a2)
        d["flip_2_3"] += int(a2 != a3)
        d["flip_1_3"] += int(a1 != a3)
        d["conf_gap_1_3"].append(c3 - c1)
    n_stab = {}
    for t, d in by_n.items():
        n = max(d["n"], 1)
        n_stab[t] = {"n_debates": d["n"],
                     "flip_rate_k1_to_k2": d["flip_1_2"] / n,
                     "flip_rate_k2_to_k3": d["flip_2_3"] / n,
                     "flip_rate_k1_to_k3": d["flip_1_3"] / n,
                     "mean_conf_gap_k1_to_k3": float(np.mean(d["conf_gap_1_3"])) if d["conf_gap_1_3"] else None}

    df = p3.load()
    nested, _ = p3.protocol_nested(df)
    seed_stab = {}
    for t in TASKS:
        d = nested[nested.task == t]
        per_seed = []
        for seed, g in d.groupby("seed"):
            gated0 = (g["i_unc<q50"] == 0).mean()
            per_seed.append(float(gated0))
        per_seed = np.array(per_seed)
        seed_stab[t] = {"per_seed_round0_gate_rate": per_seed.tolist(),
                         "mean": float(per_seed.mean()),
                         "sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                         "cv_pct": float(per_seed.std(ddof=1) / per_seed.mean() * 100)
                                   if len(per_seed) > 1 and per_seed.mean() > 0 else None}

    verdict = "STABLE"
    for t in TASKS:
        if n_stab[t]["flip_rate_k1_to_k3"] > 0.15:
            verdict = "UNSTABLE (flip rate qua cao qua N)"
        if seed_stab[t]["cv_pct"] is not None and seed_stab[t]["cv_pct"] > 25:
            verdict = "UNSTABLE (CV qua cao qua seed)"

    return {"stability_over_n_votes": n_stab,
            "stability_over_seed": seed_stab,
            "verdict": verdict,
            "note": "flip_rate = ty le debate ma dap an da so CRITIC doi khi them phieu; "
                    "cv_pct = SD/mean cua ty le dung-o-round-0 qua 5 seed."}, nested


# ============================================================ [3] frontier cascade vs SC-solo(k)
def sc_solo_curve() -> dict:
    agents = {"a": "agent_a", "b": "agent_b", "c": "agent_c"}
    base_tok = {t: None for t in TASKS}
    df = p3.load()
    for t in TASKS:
        base_tok[t] = df[(df.task == t) & (df.round_id == 0)].tok_per_round.to_numpy()
        base_tok[t] = float(np.mean(base_tok[t]) * 6)

    curve = {t: {} for t in TASKS}
    for ax, folder in agents.items():
        recs = _sc_records(f"results/logs_sc/{folder}/**/self_consistency_*.jsonl")
        by_task_k = {t: {1: [], 2: [], 3: []} for t in TASKS}
        tok_task_k = {t: {1: [], 2: [], 3: []} for t in TASKS}
        gts = {}
        for f in glob.glob(f"results/logs_sc/{folder}/**/self_consistency_*.jsonl", recursive=True):
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                gts[(int(r.get("seed", 0)), r["task"], int(r["sample_id"]))] = str(r["ground_truth"]).strip()
        for (seed, task, sid), votes in recs.items():
            if task not in TASKS or len(votes) < 3:
                continue
            gt = gts.get((seed, task, sid))
            if gt is None:
                continue
            tok3 = None
            for v in votes:
                tok3 = (tok3 or 0) + float(v.get("tokens", 0))
            for k in (1, 2, 3):
                ans, _ = _vote_majority(votes[:k])
                by_task_k[task][k].append(logio.score(task, ans, gt))
                tok_task_k[task][k].append((tok3 or 0) * (k / 3.0))
        for t in TASKS:
            for k in (1, 2, 3):
                if not by_task_k[t][k]:
                    continue
                acc = float(np.mean(by_task_k[t][k]))
                tok_pct = float(np.mean(tok_task_k[t][k]) / base_tok[t] * 100)
                curve[t].setdefault(k, {})[ax] = {"acc": acc, "tok_pct": tok_pct,
                                                   "n": len(by_task_k[t][k])}

    envelope = {t: {} for t in TASKS}
    for t in TASKS:
        for k in (1, 2, 3):
            if k not in curve[t] or not curve[t][k]:
                continue
            best_ax = max(curve[t][k], key=lambda ax: curve[t][k][ax]["acc"])
            envelope[t][k] = {"best_model": best_ax, **curve[t][k][best_ax]}
    return {"per_model": curve, "envelope_best_per_k": envelope}


def cascade_vs_sc_frontier(nested: pd.DataFrame) -> dict:
    import contextlib
    import io
    sc = sc_solo_curve()
    with contextlib.redirect_stdout(io.StringIO()):
        A = p3.summarise(nested, "cascade frontier (noi bo)")
    out = {}
    for t in TASKS:
        casc_pts = []
        for q in QS:
            name = f"unc<q{int(q*100):02d}"
            pol = A[t]["policies"][name]
            casc_pts.append({"q": q, "policy": name,
                              "acc": pol["acc"]["mean"], "tok_pct": pol["tok"]["mean"] * 100})
        sc_pts = [{"k": k, **v} for k, v in sc["envelope_best_per_k"].get(t, {}).items()]

        dominated = []
        for sp in sc_pts:
            best_casc_at_or_below = [cp for cp in casc_pts if cp["tok_pct"] <= sp["tok_pct"] + 1e-6]
            if not best_casc_at_or_below:
                dominated.append({"sc_point": sp, "dominates": None,
                                   "reason": "cascade khong co diem nao re bang SC-solo o k nay"})
                continue
            best = max(best_casc_at_or_below, key=lambda cp: cp["acc"])
            dominated.append({"sc_point": sp, "dominates": bool(best["acc"] >= sp["acc"]),
                               "cascade_ref": best})
        out[t] = {"cascade_frontier": casc_pts, "sc_solo_envelope": sc_pts,
                  "pointwise_pareto_vs_sc_solo": dominated}
    return out


# ============================================================ [5] fixed_k4/k5 vs fixed_k2
def fixed_k_paired(nested: pd.DataFrame) -> dict:
    long = p3.add_no_debate(p3.long_table(nested))
    out = {}
    n_comp = len(TASKS) * 2
    for t in TASKS:
        for hi in ("fixed_k3", "fixed_k4", "fixed_k5"):
            r = p3.cluster_bootstrap(long, t, hi, "fixed_k2")
            if r:
                r["bonferroni_alpha"] = 0.05 / n_comp
                r["significant_bonferroni"] = bool(r["dacc_p"] < 0.05 / n_comp)
                r["significant_nominal"] = bool(r["dacc_p"] < 0.05)
                out[f"{t}|{hi}_vs_fixed_k2"] = r
    return {"n_comparisons": n_comp, "results": out}


# ============================================================ [6] ablation: stage 1 vs stage 2
def cascade_ablation(nested: pd.DataFrame, df_raw: pd.DataFrame) -> dict:
    cons_col = next((c for c in ["consensus", "is_consensus", "consensus_reached"] if c in df_raw.columns), None)
    if cons_col:
        cons_mask = df_raw[cons_col] == True
    else:
        cons_mask = df_raw["round_id"] == df_raw.groupby(["seed", "task", "sample_id"])["round_id"].transform("max")

    cons_rounds = (
        df_raw[cons_mask]
        .groupby(["seed", "task", "sample_id"])["round_id"]
        .min()
        .reset_index()
        .rename(columns={"round_id": "i_c"})
    )
    d_merged = nested.merge(cons_rounds, on=["seed", "task", "sample_id"], how="left")
    d_merged["i_c"] = d_merged["i_c"].fillna(5).astype(int)

    out = {}
    for t in TASKS:
        d = d_merged[d_merged.task == t]
        stage1 = (d["i_unc<q50"] == 0).mean()
        earlier_total = (d["i_unc<q50"] < d["i_c"]).mean()
        earlier_s1 = ((d["i_unc<q50"] == 0) & (d["i_unc<q50"] < d["i_c"])).mean()
        earlier_s2 = ((d["i_unc<q50"] > 0) & (d["i_unc<q50"] < d["i_c"])).mean()

        out[t] = {
            "stage1_stop_rate": float(stage1),
            "earlier_than_consensus": float(earlier_total),
            "earlier_due_to_stage1": float(earlier_s1),
            "earlier_due_to_stage2": float(earlier_s2)
        }
    return out


# ============================================================ [7] cascade component ablation (Full/Stage1/Stage2)
def cascade_component_ablation(df_raw: pd.DataFrame) -> dict:
    """
    Chay lai 5-fold nested CV (RNG=0, cung logic shuffle/split) de tach policy unc<qXX 
    thanh 3 bien the nguyen ban.
    """
    rng = np.random.default_rng(0)
    
    # df_raw khong luu cot `unc`: p3.load() luu raw features, sau do p3.fit_score()
    # moi fold moi fit score. Dung cach nay de tranh leakage va dung cung protocol.
    cons_col = next((c for c in ["consensus", "consensus_now", "is_consensus", "consensus_reached"] if c in df_raw.columns), None)
    if cons_col:
        cons_mask = df_raw[cons_col] == True
    else:
        cons_mask = df_raw["round_id"] == df_raw.groupby(["seed", "task", "sample_id"])["round_id"].transform("max")
        
    cons = df_raw[cons_mask].groupby(["seed", "task", "sample_id"])["round_id"].min().reset_index().rename(columns={"round_id": "i_c"})
    df = df_raw.merge(cons, on=["seed", "task", "sample_id"], how="left")
    df["i_c"] = df["i_c"].fillna(5).astype(int)
    
    records = []
    for task in TASKS:
        for seed in sorted(df["seed"].unique()):
            df_ts = df[(df["task"] == task) & (df["seed"] == seed)]
            if df_ts.empty: continue
            
            sids = np.sort(df_ts["sample_id"].unique())
            rng.shuffle(sids)
            folds = np.array_split(sids, 5)
            
            for i in range(5):
                test_sids = folds[i]
                train_sids = np.concatenate([folds[j] for j in range(5) if j != i])
                
                train_df = df_ts[df_ts["sample_id"].isin(train_sids)]
                test_df = df_ts[df_ts["sample_id"].isin(test_sids)]
                
                if train_df[train_df["nonfinal"]].empty:
                    continue
                train_scores = p3.fit_score(train_df, train_df)
                test_scores = p3.fit_score(train_df, test_df)
                train_sc = train_df.copy()
                test_sc = test_df.copy()
                train_sc["_unc"] = train_scores
                test_sc["_unc"] = test_scores
                unc_train = train_sc.loc[train_sc["round_id"] == 0, "_unc"].dropna().to_numpy()
                ths = np.quantile(unc_train, QS) if len(unc_train) > 0 else [0.0] * len(QS)
                
                for sid in test_sids:
                    sample_df = test_sc[test_sc["sample_id"] == sid].sort_values("round_id")
                    if sample_df.empty: continue
                    i_c = int(sample_df["i_c"].iloc[0])
                    
                    row = {"task": task, "seed": seed, "sample_id": sid, "i_c": i_c}
                    for q, th in zip(QS, ths):
                        name = f"q{int(q*100):02d}"
                        i_full, i_s1, i_s2 = i_c, i_c, i_c
                        
                        for _, r_row in sample_df.iterrows():
                            r = int(r_row["round_id"])
                            u = r_row["_unc"]
                            if pd.isna(u): continue
                            
                            # full: Dung som nhat tu round 0 tro di
                            if i_full == i_c and u < th: 
                                i_full = r
                            # stage1_only: Chi gate duy nhat tai round 0
                            if r == 0 and u < th: 
                                i_s1 = 0
                            # stage2_only: Bo qua gate round 0, ap nguong tu round 1
                            if i_s2 == i_c and r >= 1 and u < th: 
                                i_s2 = r
                                
                        row[f"full_unc<{name}"] = i_full
                        row[f"stage1_only_unc<{name}"] = i_s1
                        row[f"stage2_only_unc<{name}"] = i_s2
                        
                    records.append(row)
                    
    res_df = pd.DataFrame(records)
    
    # Pre-compute metrics (Acc/Tok) for O(1) mapping
    lookup = {}
    for _, r_row in df_raw.iterrows():
        lookup[(r_row["seed"], r_row["task"], r_row["sample_id"], r_row["round_id"])] = {
            "acc": float(r_row["ok_if_stop"]), "tok": float(r_row["tok_per_round"])
        }
        
    out = {}
    base_tok_cache = {}
    for t in TASKS:
        bt_arr = df_raw[(df_raw.task == t) & (df_raw.round_id == 0)].tok_per_round.to_numpy()
        base_tok_cache[t] = float(np.mean(bt_arr) * 6) if len(bt_arr) > 0 else 1.0

    for task in TASKS:
        out[task] = {}
        task_res = res_df[res_df["task"] == task]
        if task_res.empty: continue
        
        for q in QS:
            name = f"unc<q{int(q*100):02d}"
            metrics = {"full": {"acc": [], "tok": []}, 
                       "stage1_only": {"acc": [], "tok": []}, 
                       "stage2_only": {"acc": [], "tok": []}}
                       
            for _, row in task_res.iterrows():
                seed, sid = row["seed"], row["sample_id"]
                for var in ["full", "stage1_only", "stage2_only"]:
                    stop_r = row[f"{var}_{name}"]
                    
                    acc = lookup.get((seed, task, sid, stop_r), {}).get("acc", 0.0)
                    tok = sum(lookup.get((seed, task, sid, r), {}).get("tok", 0.0) for r in range(stop_r + 1))
                        
                    metrics[var]["acc"].append(acc)
                    metrics[var]["tok"].append(tok)
                    
            out[task][name] = {
                "full": {
                    "acc": float(np.mean(metrics["full"]["acc"])), 
                    "tok_pct": float(np.mean(metrics["full"]["tok"]) / base_tok_cache[task] * 100)
                },
                "stage1_only": {
                    "acc": float(np.mean(metrics["stage1_only"]["acc"])), 
                    "tok_pct": float(np.mean(metrics["stage1_only"]["tok"]) / base_tok_cache[task] * 100)
                },
                "stage2_only": {
                    "acc": float(np.mean(metrics["stage2_only"]["acc"])), 
                    "tok_pct": float(np.mean(metrics["stage2_only"]["tok"]) / base_tok_cache[task] * 100)
                },
            }
            
    return out


def main() -> None:
    print("[1] ON DINH CRITIC-SC QUA N (k phieu) VA SEED")
    print("=" * 78)
    stab, nested = critic_sc_stability()
    for t in TASKS:
        n = stab["stability_over_n_votes"][t]
        s = stab["stability_over_seed"][t]
        cv = f"{s['cv_pct']:.1f}%" if s["cv_pct"] is not None else "n/a"
        print(f"  {t:<12} lat k1->k3: {n['flip_rate_k1_to_k3']*100:5.1f}%   "
              f"gate@round0 qua seed: {s['mean']*100:5.1f}% ± {s['sd']*100:4.1f}pp (CV={cv})")
    print(f"  => {stab['verdict']}")

    print("\n[3] FRONTIER: CASCADE (cong dung r0sc that, unc<q10..q50) vs SC-SOLO(k=1,2,3)")
    print("=" * 78)
    frontier = cascade_vs_sc_frontier(nested)
    for t in TASKS:
        print(f"\n--- {t} ---")
        print("  cascade:", [(round(p["tok_pct"], 1), round(p["acc"], 3)) for p in frontier[t]["cascade_frontier"]])
        print("  SC-solo:", [(p["k"], p["best_model"], round(p["tok_pct"], 1), round(p["acc"], 3))
                             for p in frontier[t]["sc_solo_envelope"]])
        n_dom = sum(1 for d in frontier[t]["pointwise_pareto_vs_sc_solo"] if d["dominates"])
        n_tot = len(frontier[t]["pointwise_pareto_vs_sc_solo"])
        print(f"  cascade >= SC-solo tai chi phi <= : {n_dom}/{n_tot} diem SC-solo")


    print("\n[5] fixed_k4 / fixed_k5 SO VOI fixed_k2 (ghep cap, bootstrap phan cum theo cau hoi)")
    print("=" * 78)
    fk = fixed_k_paired(nested)
    print(f"  Bonferroni alpha = {0.05/fk['n_comparisons']:.5f} tren {fk['n_comparisons']} phep so sanh")
    for k, r in fk["results"].items():
        star = " **" if r["significant_bonferroni"] else (" *" if r["significant_nominal"] else "")
        print(f"  {k:<28} dAcc={r['dacc']:+.4f} 95%CI[{r['dacc_ci'][0]:+.4f},{r['dacc_ci'][1]:+.4f}] "
              f"p={r['dacc_p']:.4f} dTok={r['dtok_rel']*100:+.1f}%{star}")

    print("\n[6] ABLATION: STAGE 1 (ROUND 0 GATE) vs STAGE 2 (DEBATE STOPPING)")
    print("=" * 78)
    df_raw = p3.load()
    ablation = cascade_ablation(nested, df_raw=df_raw)
    for t in TASKS:
        a = ablation[t]
        print(f"  {t:<12} Dung R0 (Stage 1): {a['stage1_stop_rate']*100:5.1f}% | "
              f"Dung truoc consensus: {a['earlier_than_consensus']*100:5.1f}% "
              f"(Stage 1 gop: {a['earlier_due_to_stage1']*100:4.1f}%, Stage 2 gop: {a['earlier_due_to_stage2']*100:4.1f}%)")

    print("\n[7] CASCADE COMPONENT ABLATION (Full vs Stage1-only vs Stage2-only)")
    print("=" * 78)
    comp_ablation = cascade_component_ablation(df_raw)
    for t in TASKS:
        print(f"\n--- {t} ---")
        for q in [0.25, 0.50, 0.75]:
            name = f"unc<q{int(q*100):02d}"
            if name in comp_ablation[t]:
                r = comp_ablation[t][name]
                print(f"  {name:<9} | "
                      f"Full: {r['full']['acc']:.3f} ({r['full']['tok_pct']:5.1f}%) | "
                      f"Stage1-only: {r['stage1_only']['acc']:.3f} ({r['stage1_only']['tok_pct']:5.1f}%) | "
                      f"Stage2-only: {r['stage2_only']['acc']:.3f} ({r['stage2_only']['tok_pct']:5.1f}%)")

    out = {
        "critic_sc_stability": stab, 
        "cascade_vs_sc_frontier": frontier, 
        "fixed_k4_k5_vs_fixed_k2": fk, 
        "cascade_ablation": ablation,
        "cascade_component_ablation": comp_ablation
    }
    Path("results/p5_cascade_confirmatory.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n-> results/p5_cascade_confirmatory.json")


if __name__ == "__main__":
    main()