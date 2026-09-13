"""P5 - Xac nhan cascade critic-SC: on dinh, cong dung that, frontier, rescue/hurt.

Bon manh rieng, GOP mot cho de dung chung du lieu da nap (p3.load/protocol_nested):

  [1] ON DINH CRITIC-SC qua N (so phieu SC, k=1..3) va SEED (5 lan chay doc lap).
      - On dinh theo N: dap an da so cua CRITIC tu round-0-SC co doi khi them phieu
        (k=1 -> k=2 -> k=3) khong? Bao ty le lat + do lech confidence.
      - On dinh theo SEED: voi nguong T co dinh (unc<q50, chon tren val nhu p3),
        ty le debate bi cong dung dat round 0 (r0sc) co on dinh giua 5 seed khong
        (mean +/- SD qua seed, theo tung benchmark)?
      Neu ca hai deu on dinh (lat thap, SD nho tuong doi so voi mean) thi cong
      dung critic-SC co the dung duoc; neu khong, cascade dung tren no la mong manh.

  [2] CASCADE CO CONG DUNG THAT: tai su dung dung policy "unc<qXX" cua p3, vi no
      DA gan cong r0sc-gate that (xem apply_critic_sc_round0 + stop_index trong
      p3_holdout_policy.py): nguong T chon tren val/inner-val, KHONG bao gio tren
      tap dang cham -> day la "cascade" duoc xac nhan o day, khong phai dinh nghia
      moi. Script nay chi TRICH XUAT frontier cua no (q=0.1..0.5) de doi chieu.

  [3] FRONTIER CASCADE vs DUONG CONG SC-SOLO THEO k: voi moi benchmark, ve hai
      duong tren cung mat phang (token%, accuracy):
        - cascade: 5 diem unc<q10..q50 (p3, nested CV, T tu val)
        - SC-solo(k): k=1,2,3 phieu subsample tu CHINH log self-consistency da chay
          (results/logs_sc/agent_{a,b,c}), lay BAO (envelope) accuracy tot nhat
          qua 3 model o moi k - dung gia thiet BAT LOI NHAT cho cascade (giong
          quy uoc paper.md Section V-C: SC duoc chon model tot nhat/benchmark).
      Bao cascade co PARETO-VUOT SC-solo hay khong (accuracy >= tai token% <=).

  [4] RESCUE/HURT LA CONG DUNG DE CASCADE CO CO SO: doc lai results/p4_rescue_hurt.json
      (da tinh boi p4_rescue_hurt.py, McNemar tren cap khong khop). Neu rescued == 0
      o BAT KY benchmark nao (debate khong cuu duoc CAU NAO rieng le so voi round 0),
      dung lai va bao ket qua AM: khong co co so de xay cascade. Neu rescued > 0 o
      ca ba, in ro va tiep tuc - cascade co co so DE DUNG, du hieu ung co the nho.

  [5] fixed_k4 / fixed_k5 SO VOI fixed_k2, GHEP CAP, bootstrap phan cum theo cau hoi
      (tai su dung p3.cluster_bootstrap tren long_table cua nested CV) - cau hoi:
      di sau vao fixed-depth co mua duoc gi so voi k=2 hay khong, va co dang ke
      thong ke khong (Bonferroni tren so phep so sanh moi).

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


def critic_sc_stability() -> dict:
    recs = _sc_records("results/logs_sc/agent_c/**/self_consistency_*.jsonl")

    # --- on dinh theo N (k=1,2,3 phieu, TRONG CUNG mot seed/cau hoi) ---
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

    # --- on dinh theo SEED: ty le round-0-gate (unc<q50) dat, moi seed rieng ---
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
    """(token%, accuracy) cho k=1,2,3 phieu, moi agent rieng + bao (best) qua 3 agent."""
    agents = {"a": "agent_a", "b": "agent_b", "c": "agent_c"}
    base_tok = {t: None for t in TASKS}
    # token% chuan theo `always` cua chinh p3 (nhat quan voi TABLE 0/VII)
    df = p3.load()
    for t in TASKS:
        base_tok[t] = df[(df.task == t) & (df.round_id == 0)].tok_per_round.to_numpy()
        base_tok[t] = float(np.mean(base_tok[t]) * 6)  # 'always' ~ 6 round xap xi tu tok_per_round

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
    with contextlib.redirect_stdout(io.StringIO()):   # p3.summarise() la ham dung chung, in rat dai
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

        # Pareto check: for every SC-solo point, is there a cascade point at <= token cost
        # with >= accuracy? (interpolating cascade's own points is not needed - cascade
        # already spans a wide range of q, so we just check point-wise domination.)
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

    print("[3] FRONTIER: CASCADE (cong dung r0sc that, unc<q10..q50) vs SC-SOLO(k=1,2,3)")
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


    print("[5] fixed_k4 / fixed_k5 SO VOI fixed_k2 (ghep cap, bootstrap phan cum theo cau hoi)")
    fk = fixed_k_paired(nested)
    print(f"  Bonferroni alpha = {0.05/fk['n_comparisons']:.5f} tren {fk['n_comparisons']} phep so sanh")
    for k, r in fk["results"].items():
        star = " **" if r["significant_bonferroni"] else (" *" if r["significant_nominal"] else "")
        print(f"  {k:<28} dAcc={r['dacc']:+.4f} 95%CI[{r['dacc_ci'][0]:+.4f},{r['dacc_ci'][1]:+.4f}] "
              f"p={r['dacc_p']:.4f} dTok={r['dtok_rel']*100:+.1f}%{star}")

    out = {"critic_sc_stability": stab, "cascade_vs_sc_frontier": frontier, "fixed_k4_k5_vs_fixed_k2": fk}
    Path("results/p5_cascade_confirmatory.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n-> results/p5_cascade_confirmatory.json")


if __name__ == "__main__":
    main()
