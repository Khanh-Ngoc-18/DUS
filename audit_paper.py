"""audit_paper.py - Chung minh moi so lieu trong file paper, sinh PROVENANCE.md.

Script nay CHI DOC. No khong sua, khong tinh lai, khong thay the bat ky phep tinh
nao trong logio / p0 / p1 / p2 / p3 / p4 / verify_threshold. Nhiem vu duy nhat: voi
moi con so xuat hien trong file paper, chi ra no den tu dau va kiem tra co khop khong.

Bon phan:
  [A] Registry  - moi claim van xuoi trong paper -> nguon (file ket qua + duong dan
                  key, hoac cong thuc suy ra tu cac key da co). Doi chieu tung cai.
  [B2] O bang    - parse bang markdown trong file paper, so TUNG O voi gia tri nguon.
  [C] Tinh lai tu LOG THO - cac dai luong khong file ket qua nao chua (blind spot,
                  ty le vong khong doi ket cuc, rescue/hurt, dang thuc entropy =
                  consensus, kich thuoc split, accuracy tren test set MMLU).
  [D] Quet mo coi - liet ke moi so trong paper KHONG duoc [A]/[B2]/[C] phu.

Output:
  PROVENANCE.md            - bang dan nguon doc duoc, nop kem paper
  results/paper_audit.json - ban may doc

Cach dung:  python audit_paper.py [duong/dan/paper.md]
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import logio

# console Windows co the la cp1258: ep utf-8 de in duoc dau trong claim
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAPER = Path(sys.argv[1] if len(sys.argv) > 1 else "paper.md")
R = Path("results")
OUT_MD = Path("PROVENANCE.md")

_cache: dict[str, dict] = {}
rows: list[dict] = []
live: list[dict] = []
covered: set[str] = set()

LOGS = "results/logs/**/debate_full_*.jsonl (qua logio.load_rounds)"


# ---------------------------------------------------------------- helper
def load(name: str) -> dict:
    if name not in _cache:
        _cache[name] = json.loads((R / name).read_text(encoding="utf-8"))
    return _cache[name]


def key(name: str, path: str):
    """Lay gia tri theo duong dan key, vd 'sequential.real_rounds_saved'."""
    cur = load(name)
    for part in path.split("."):
        cur = cur[part]
    return cur


def _norm(s: str) -> str:
    """Chuan hoa mot chuoi so trong paper ve dang so sanh duoc."""
    return (str(s).replace("**", "").replace(",", "").replace("\\", "")
            .replace("−", "-").replace("–", "-").replace("%", "")
            .replace("+", "").replace("$", "")
            .replace("[", "").replace("]", "").strip())


def _dec(claim: str) -> int:
    return len(claim.split(".")[1]) if "." in claim else 0


def matches(claim: str, value) -> bool:
    """Khop neu lam tron gia tri nguon ve dung so chu so cua claim.

    Chap nhan ca dang thap phan (0.339) lan dang phan tram (33.9) cho cung mot
    nguon, vi paper viet ca hai kieu tuy cho.
    """
    c = _norm(claim)
    try:
        target, v = float(c), float(value)
    except (TypeError, ValueError):
        return False
    d = _dec(c)
    return round(v, d) == round(target, d) or round(v * 100, d) == round(target, d)


def check(claim: str, desc: str, src: str, value, section: str = "") -> None:
    rows.append(dict(section=section, claim=claim, description=desc,
                     source=src, computed=value, ok=bool(matches(claim, value))))
    covered.add(_norm(claim))
    covered.add(_norm(claim).lstrip("-"))


def cellcheck(desc: str, cell: str, value, tol: float, section: str) -> None:
    """So mot o bang voi gia tri nguon, sai so tuyet doi <= tol."""
    c = _norm(cell)
    try:
        ok = abs(float(c) - float(value)) <= tol
    except (TypeError, ValueError):
        ok = False
    rows.append(dict(section=section, claim=c, description=desc,
                     source="(o bang)", computed=value, ok=bool(ok)))
    covered.add(c)
    covered.add(c.lstrip("-"))


BODY = PAPER.read_text(encoding="utf-8")
# bo phan base64 nhung anh o cuoi file - khong phai so lieu
BODY = BODY.split("[image1]: <data:image")[0]


def md_table(marker: str) -> list[list[str]]:
    """Lay cac hang cua bang markdown dung sau `marker`, bo hang gach ngang."""
    i = BODY.index(marker)
    out = []
    for line in BODY[i:].splitlines():
        s = line.strip()
        if not s.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set(":- "):
            continue
        out.append(cells)
    return out


# ================================================================ file ket qua
VT, W, P0, P1, P2, P3, P4 = ("verify_threshold.json", "dus_weights.json",
                             "p0_critic_eval.json", "p1_power.json",
                             "p2_cost_accuracy.json", "p3_holdout_policy.json",
                             "p4_rescue_hurt.json")
P5 = "p5_cascade_confirmatory.json"
A = load(P3)["nested_cv"]
H = load(P3)["fixed_holdout"]
NICE = {"GSM8K": "gsm8k", "MMLU": "mmlu", "StrategyQA": "strategyqa"}

# ================================================================ [A] registry
# ---- IV-B quy mo chay ----
check("4500", "so debate", f"{VT} :: sequential.n_rollouts",
      key(VT, "sequential.n_rollouts"), "IV-B")
check("27000", "tong so round", f"{VT} :: naive.n_rounds", key(VT, "naive.n_rounds"), "IV-B")
check("22500", "so round khong-cuoi", f"{VT} :: nonfinal.n", key(VT, "nonfinal.n"), "IV-B")
check("32.9", "ty le sai tren round khong-cuoi", f"{VT} :: nonfinal.error_rate",
      key(VT, "nonfinal.error_rate"), "IV-B")
check("0.326", "base rate cua nhan da sua", f"{P0} :: base_rate", key(P0, "base_rate"), "IV-B")

# ---- IV-C split va power ----
check("18978", "so round trong split train", f"{W} :: n_rows", key(W, "n_rows"), "IV-C")
check("2748", "so cau hoi doc lap", f"{P1} :: n_samples", key(P1, "n_samples"), "IV-C")
for q in ("65", "100", "200"):
    check(f"{key(P1, f'power.{q}.power'):.3f}", f"power tai {q} cau",
          f"{P1} :: power.{q}.power", key(P1, f"power.{q}.power"), "IV-C")

# ---- V-B kiem toan giao thuc truc tiep ----
for claim, desc, path in [
    ("33.9", "ty le round bi gan 'stop' (giao thuc truc tiep)", "naive.stop_rate"),
    ("9149", "so round duoi nguong", "circularity.n_below_threshold"),
    ("-0.604", "nguong T* chon boi giao thuc truc tiep", "naive.threshold"),
    ("4.85", "error rate theo ROUND", "sequential.miss_rate_per_row"),
    ("5.09", "error rate theo DEBATE", "sequential.miss_rate_per_sample"),
    ("1.05", "he so thoi phong per-round -> per-debate", "sequential.inflation"),
    ("1517", "so debate co flag fire", "sequential.n_flag_fired"),
    ("14987", "so round he chuan (consensus) thuc su chay", "sequential.rounds_standard_system"),
    ("0", "so debate dung SOM HON he chuan", "sequential.n_stopped_earlier"),
    ("0.0", "ty le round tiet kiem THUC", "sequential.real_saving_rate"),
    ("11754", "so round co answer_entropy = 0 (= consensus)", "circularity.n_entropy_zero"),
    ("2,037", "round entropy=0 la round CUOI cua debate", "circularity.n_entropy_zero_is_last"),
    ("872", "so round nam dung tai T (mass point)", "mass_point.n_rows_equal_T"),
    ("37.1", "stop rate neu doi dau < thanh <=", "mass_point.stop_rate_le"),
    ("5.63", "error rate neu doi dau < thanh <=", "mass_point.miss_rate_le"),
    ("0.717", "AUC tren MOI round (thoi phong)", "nonfinal.auc_all_rounds"),
    ("0.714", "AUC tren round khong-cuoi", "nonfinal.auc_nonfinal"),
    ("0.003", "do thoi phong AUC", "nonfinal.auc_inflation"),
]:
    check(claim, desc, f"{VT} :: {path}", key(VT, path), "V-B")
check("0.323", "uncertain rate tren train", f"{W} :: uncertain_rate",
      key(W, "uncertain_rate"), "V-B")

# ---- V-C phan biet ----
for claim, name in [("0.705", "base (DUS 4 feature)"), ("0.729", "critic only"),
                    ("0.736", "base + critic")]:
    check(claim, f"ROC-AUC model '{name}'", f"{P0} :: models['{name}'].auc",
          key(P0, f"models.{name}.auc"), "V-C")
check("0.738", "ROC-AUC DUS-11 5-fold CV theo cau", f"{P1} :: auc_cv", key(P1, "auc_cv"), "V-C")
for t in ("gsm8k", "mmlu", "strategyqa"):
    check(f"{key(P0, f'per_task.{t}'):.3f}", f"AUC per-task {t}",
          f"{P0} :: per_task.{t}", key(P0, f"per_task.{t}"), "V-C")

# hieu AUC ghep cap - CI cua HIEU SO, khong phai CI rieng tung model
PAIR_LBL = {"critic only - base (DUS 4 feature)": "CRITIC-7 - DUS-4",
            "base + critic - base (DUS 4 feature)": "DUS-11 - DUS-4",
            "base + critic - critic only": "DUS-11 - CRITIC-7"}
for src, lbl in PAIR_LBL.items():
    v = key(P0, f"paired.{src}")
    check(f"{v['delta']:.3f}", f"hieu AUC ghep cap, {lbl}",
          f"{P0} :: paired['{src}'].delta", v["delta"], "V-C")
    check(f"{v['ci'][0]:.3f}", f"CI duoi cua hieu, {lbl}",
          f"{P0} :: paired['{src}'].ci[0]", v["ci"][0], "V-C")
    check(f"{v['ci'][1]:.3f}", f"CI tren cua hieu, {lbl}",
          f"{P0} :: paired['{src}'].ci[1]", v["ci"][1], "V-C")
    check(f"{v['p_two_sided']:.3f}", f"p hai phia, {lbl}",
          f"{P0} :: paired['{src}'].p_two_sided", v["p_two_sided"], "V-C")

# ---- V-D thoi diem dung ----
check("57.9", "% debate mmlu q50 dung SOM HON consensus",
      f"{P3} :: nested_cv.mmlu.stop_timing['unc<q50'].earlier_pct",
      A["mmlu"]["stop_timing"]["unc<q50"]["earlier_pct"], "V-D")

# ---- V-E danh doi cost-accuracy ----
for b, q, dt, da in [("mmlu", "unc<q50", "-41.1", "+0.001"),
                     ("gsm8k", "unc<q50", "-6.8", "+0.001"), ("strategyqa", "unc<q50", "-1.9", "+0.003")]:
    p = A[b]["paired_vs_consensus"][q]
    # dang co dau (bang) va dang tri tuyet doi (van xuoi) deu xuat hien trong paper
    check(dt, f"dToken {b} {q} vs consensus (nested CV)",
          f"{P3} :: nested_cv.{b}.paired_vs_consensus['{q}'].dtok.mean", p["dtok"]["mean"] * 100, "V-E")
    covered.add(_norm(dt).lstrip("-"))
    check(da, f"dAccuracy {b} {q} vs consensus (nested CV)",
          f"{P3} :: nested_cv.{b}.paired_vs_consensus['{q}'].dacc.mean", p["dacc"]["mean"], "V-E")
    covered.add(_norm(da).lstrip("-"))
check("0.51", "TABLE IX net tong theo %", f"{P4} :: total.net_pct",
      key(P4, "total.net_pct"), "V-F")
check("116", "so cap khong khop cua gsm8k (rescued + corrupted)",
      f"{P4} :: gsm8k.n_discordant", key(P4, "gsm8k.n_discordant"), "V-F")
check("2.2", "SD cua dToken mmlu q50 qua 5 seed",
      f"{P3} :: nested_cv.mmlu.paired_vs_consensus['unc<q50'].dtok.sd",
      A["mmlu"]["paired_vs_consensus"]["unc<q50"]["dtok"]["sd"] * 100, "V-E")
hp = H["mmlu"]["paired_vs_consensus"]["unc<q50"]
check("-41.7", "dToken mmlu q50 tren holdout 70/20/10",
      f"{P3} :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dtok.mean",
      hp["dtok"]["mean"] * 100, "V-E")
covered.add("41.7")
check("12.7", "SD cua dToken mmlu q50 tren holdout",
      f"{P3} :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dtok.sd",
      hp["dtok"]["sd"] * 100, "V-E")
check("-0.016", "dAccuracy mmlu q50 tren holdout 70/20/10",
      f"{P3} :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dacc.mean",
      hp["dacc"]["mean"], "V-E")
check("-0.075", "CI cua dAcc mmlu q50 tren holdout, can duoi",
      f"{P3} :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dacc_ci_clustered[0]",
      hp["dacc_ci_clustered"][0], "V-E")
check("+0.031", "CI cua dAcc mmlu q50 tren holdout, can tren",
      f"{P3} :: fixed_holdout.mmlu.paired_vs_consensus['unc<q50'].dacc_ci_clustered[1]",
      hp["dacc_ci_clustered"][1], "V-E")
check("151", "so debate test cua mmlu", f"{P3} :: fixed_holdout.mmlu.n_debate",
      H["mmlu"]["n_debate"], "V-E")
check("0.587", "accuracy cua `always` tren tap test MMLU",
      f"{P3} :: fixed_holdout.mmlu.policies.always.acc.mean",
      H["mmlu"]["policies"]["always"]["acc"]["mean"], "V-E")
opt = load(P3)["threshold_selection_optimism"]
check("3.7", "do lac quan LON NHAT cua dToken khi chon T in-sample",
      f"{P3} :: max |threshold_selection_optimism.*.*.dtok_pp|",
      max(abs(v["dtok_pp"]) for b in opt.values() for v in b.values()), "V-E")
check("0.006", "do lac quan LON NHAT cua dAccuracy khi chon T in-sample",
      f"{P3} :: max |threshold_selection_optimism.*.*.dacc|",
      max(abs(v["dacc"]) for b in opt.values() for v in b.values()), "V-E")

# ---- V-F rescue / hurt ----
for k in ("gsm8k", "mmlu", "strategyqa", "total"):
    v = key(P4, f"{k}.mcnemar_p")
    check(f"{v:.3f}", f"McNemar p, net cua {k} khac 0", f"{P4} :: {k}.mcnemar_p", v, "V-F")

# ---- VI-A dai luong chi phoi ----
for t in ("gsm8k", "mmlu", "strategyqa"):
    check(f"{key(P2, f'pooled_auc_across_seeds.{t}.auc'):.3f}", f"AUC gop qua seed, {t}",
          f"{P2} :: pooled_auc_across_seeds.{t}.auc",
          key(P2, f"pooled_auc_across_seeds.{t}.auc"), "VI-A")
    check(f"{A[t]['policies']['consensus']['tok']['mean'] * 100:.1f}", f"ty le token cua consensus, {t}",
          f"{P3} :: nested_cv.{t}.policies.consensus.tok.mean",
          A[t]["policies"]["consensus"]["tok"]["mean"] * 100, "VI-A")

# ================================================================ [B2] o bang
# TABLE II - trong so DUS-4
for r in md_table("**TABLE II —")[1:]:
    f = r[0].replace("\\_", "_").replace("*", "").strip()
    cellcheck(f"TABLE II he so tho {f}", r[1],
              key(W, f"logistic_coefficients_on_standardized_features.{f}"), 0.0005, "TABLE II")
    cellcheck(f"TABLE II trong so cuoi {f}", r[2].split("(")[0],
              key(W, f"dus_weights.{f}"), 0.0005, "TABLE II")

# TABLE IV - AUC tung feature
for r in md_table("**TABLE IV —")[1:]:
    f = r[0].replace("\\_", "_").replace("*", "").strip()
    v = key(P0, f"single_feature_auc.{f}")
    cellcheck(f"TABLE IV AUC {f}", r[2], v, 0.0005, "TABLE IV")
    cellcheck(f"TABLE IV |AUC-0.5| {f}", r[3], abs(v - 0.5), 0.0005, "TABLE IV")

# TABLE V - AUC + CI rieng tung model
V_SRC = {"DUS-4 — consensus dynamics": ("base (DUS 4 feature)", P0),
         "CRITIC-7 — critic signals alone": ("critic only", P0),
         "DUS-11 — both families": ("base + critic", P0)}
for r in md_table("**TABLE V —")[1:]:
    lbl = r[0].replace("*", "").strip()
    lo, hi = r[3].strip("[]").split(",")
    if lbl in V_SRC:
        name, _ = V_SRC[lbl]
        m = key(P0, f"models.{name}")
        cellcheck(f"TABLE V AUC {lbl}", r[2], m["auc"], 0.0005, "TABLE V")
        cellcheck(f"TABLE V CI duoi {lbl}", lo, m["ci"][0], 0.0005, "TABLE V")
        cellcheck(f"TABLE V CI tren {lbl}", hi, m["ci"][1], 0.0005, "TABLE V")
    else:                                        # dong 5-fold CV lay tu p1
        cellcheck("TABLE V AUC 5-fold CV", r[2], key(P1, "auc_cv"), 0.0005, "TABLE V")
        cellcheck("TABLE V CI duoi 5-fold CV", lo, key(P1, "ci")[0], 0.0005, "TABLE V")
        cellcheck("TABLE V CI tren 5-fold CV", hi, key(P1, "ci")[1], 0.0005, "TABLE V")

# TABLE Va - hieu AUC ghep cap
INV = {v: k for k, v in PAIR_LBL.items()}
for r in md_table("**TABLE Va —")[1:]:
    lbl = _norm(r[0]).replace("−", "-")
    v = key(P0, f"paired.{INV[lbl]}")
    cellcheck(f"TABLE Va delta {lbl}", r[1], v["delta"], 0.0005, "TABLE Va")
    lo, hi = r[2].strip("[]").split(",")
    cellcheck(f"TABLE Va CI duoi {lbl}", lo, v["ci"][0], 0.0005, "TABLE Va")
    cellcheck(f"TABLE Va CI tren {lbl}", hi, v["ci"][1], 0.0005, "TABLE Va")

# TABLE VI - thoi diem dung so voi consensus
for r in md_table("**TABLE VI —")[1:]:
    b, q = NICE[r[0].replace("*", "").strip()], _norm(r[1])
    t = A[b]["stop_timing"][f"unc<q{int(float(q) * 100)}"]
    for c, fld in [(r[2], "earlier"), (r[3], "same"), (r[4], "later")]:
        cnt, pct = c.replace("**", "").split("·")
        cellcheck(f"TABLE VI {b} q{q} {fld} (dem)", cnt, t[fld], 0.5, "TABLE VI")
        cellcheck(f"TABLE VI {b} q{q} {fld} (%)", pct, t[fld + "_pct"], 0.05, "TABLE VI")

# TABLE VII - accuracy / token cua moi chinh sach
POL = {"always (6 rounds)": "always", "consensus": "consensus",
       "round-0 ensemble vote (fixed\\_k1)": "fixed_k1", "fixed\\_k2": "fixed_k2",
       "fixed\\_k3": "fixed_k3", "unc<q30": "unc<q30", "unc<q40": "unc<q40",
       "unc<q50": "unc<q50", "oracle (unachievable)": "oracle"}
for r in md_table("**TABLE VII —")[1:]:
    p = POL[r[0].replace("**", "").strip()]
    for c, b in zip(r[1:4], ("gsm8k", "mmlu", "strategyqa")):
        acc, tok = c.replace("**", "").split("/")
        cellcheck(f"TABLE VII accuracy {b} {p}", acc,
                  A[b]["policies"][p]["acc"]["mean"], 0.0005, "TABLE VII")
        cellcheck(f"TABLE VII token% {b} {p}", tok,
                  A[b]["policies"][p]["tok"]["mean"] * 100, 0.05, "TABLE VII")

# TABLE VIII - so sanh ghep cap voi consensus
for r in md_table("**TABLE VIII —")[1:]:
    b, q = NICE[r[0].replace("*", "").strip()], _norm(r[1])
    P = A[b]["paired_vs_consensus"][f"unc<q{int(float(q) * 100)}"]
    m, sd = r[2].replace("**", "").split("±")
    cellcheck(f"TABLE VIII {b} q{q} dAcc mean", m, P["dacc"]["mean"], 0.0005, "TABLE VIII")
    cellcheck(f"TABLE VIII {b} q{q} dAcc sd", sd, P["dacc"]["sd"], 0.0005, "TABLE VIII")
    lo, hi = r[3].strip("[]").split(",")
    cellcheck(f"TABLE VIII {b} q{q} dAcc CI lo", lo, P["dacc_ci_clustered"][0], 0.0005, "TABLE VIII")
    cellcheck(f"TABLE VIII {b} q{q} dAcc CI hi", hi, P["dacc_ci_clustered"][1], 0.0005, "TABLE VIII")
    m, sd = r[4].replace("**", "").split("±")
    cellcheck(f"TABLE VIII {b} q{q} dToken mean", m, P["dtok"]["mean"] * 100, 0.05, "TABLE VIII")
    cellcheck(f"TABLE VIII {b} q{q} dToken sd", sd, P["dtok"]["sd"] * 100, 0.05, "TABLE VIII")

# TABLE IX - rescue / hurt
NAME9 = {"GSM8K": "gsm8k", "MMLU": "mmlu", "StrategyQA": "strategyqa", "Total": "total"}
for r in md_table("**TABLE IX —")[1:]:
    k = NAME9[r[0].replace("*", "").strip()]
    v = key(P4, k)
    cellcheck(f"TABLE IX {k} n", r[1], v["n"], 0.5, "TABLE IX")
    for c, fld in [(r[2], "rescued"), (r[3], "corrupted")]:
        cnt, pct = c.replace("**", "").split("·")
        cellcheck(f"TABLE IX {k} {fld} n", cnt, v[fld], 0.5, "TABLE IX")
        cellcheck(f"TABLE IX {k} {fld} %", pct, v[fld + "_pct"], 0.05, "TABLE IX")
    cellcheck(f"TABLE IX {k} net", r[4].split("(")[0], v["net"], 0.5, "TABLE IX")
    cellcheck(f"TABLE IX {k} McNemar p", r[5], v["mcnemar_p"], 0.0005, "TABLE IX")

# TABLE X - dai luong ung vien chi phoi
for r in md_table("**TABLE X —")[1:]:
    b = NICE[r[0].replace("*", "").strip()]
    cellcheck(f"TABLE X accuracy (always) {b}", r[2],
              A[b]["policies"]["always"]["acc"]["mean"], 0.0005, "TABLE X")
    cellcheck(f"TABLE X pooled AUC {b}", r[4],
              key(P2, f"pooled_auc_across_seeds.{b}.auc"), 0.0005, "TABLE X")
    cellcheck(f"TABLE X consensus token% {b}", r[5],
              A[b]["policies"]["consensus"]["tok"]["mean"] * 100, 0.05, "TABLE X")
    cellcheck(f"TABLE X dToken achieved {b}", r[6],
              A[b]["paired_vs_consensus"]["unc<q50"]["dtok"]["mean"] * 100, 0.05, "TABLE X")

# TABLE 0 - moi chinh sach tren cung mot mat phang (nguon: p6_baseline_showdown.json)
P6 = P3          # da gop vao p3; moi key nam duoi "baseline_showdown"
POL0 = {"always (6 rounds)": "always", "consensus (incumbent)": "consensus",
        "majority voting, 3 models, symmetric": "ensemble_vote",
        "fixed\\_k1 *(round-0 ensemble vote)*": "fixed_k1",
        "fixed\\_k2": "fixed_k2", "fixed\\_k3": "fixed_k3",
        "fixed\\_k4": "fixed_k4", "fixed\\_k5": "fixed_k5",
        "DUS-11, *q* = 0.4": "unc<q40", "DUS-11, *q* = 0.5": "unc<q50",
        "oracle (unachievable)": "oracle",
        "self-consistency@3, Qwen2.5-3B": "self_consistency_a",
        "self-consistency@3, Llama3.2-3B": "self_consistency_b",
        "self-consistency@3, Gemma3-4B": "self_consistency_c"}
_seen0 = set()
for r in md_table("**TABLE 0 —")[1:]:
    label = r[0].strip()
    # hang tieu de nhom (in dam, khong co so) va hang chua chay (o = "—") thi bo qua
    p = POL0.get(label)
    if p is None:
        if label.startswith("**"):
            continue
        raise SystemExit(f"TABLE 0: nhan hang khong nhan ra -> {label!r}")
    _seen0.add(p)
    for c, b in zip(r[2:5], ("gsm8k", "mmlu", "strategyqa")):
        if "/" not in c:
            continue
        acc, tok = c.split("/")
        src = key(P6, f"baseline_showdown.summary.{b}.{p}")
        cellcheck(f"TABLE 0 accuracy {b} {p}", acc, src["acc"], 0.0005, "TABLE 0")
        cellcheck(f"TABLE 0 token% {b} {p}", tok, src["tok_rel"] * 100, 0.05, "TABLE 0")
# TABLE 0 phai liet ke DU fixed_k1..k5 - chinh khuyen nghi cua bai (VI-B) doi vay
_missing = [f"fixed_k{k}" for k in (1, 2, 3, 4, 5) if f"fixed_k{k}" not in _seen0]
if _missing:
    raise SystemExit(f"TABLE 0 thieu {_missing}: VI-B yeu cau bao cao MOI k trong dai")

# TABLE XI - fixed_k2 tru DUS-11, bootstrap phan cum theo cau hoi
NICE_XI = {"GSM8K": "gsm8k", "MMLU": "mmlu", "StrategyQA": "strategyqa"}
for r in md_table("**TABLE XI —")[1:]:
    b = NICE_XI[r[0].strip()]
    q = "unc<q40" if "0.4" in r[1] else "unc<q50"
    s = key(P6, f"baseline_showdown.fixed_k2_vs_dus11.{b}|fixed_k2-{q}")
    lo, hi = [x.strip(" []\\") for x in r[3].split(",")]
    cellcheck(f"TABLE XI dacc {b} {q}", r[2], s["dacc"], 0.0005, "TABLE XI")
    cellcheck(f"TABLE XI CI-lo {b} {q}", lo, s["dacc_ci"][0], 0.0005, "TABLE XI")
    cellcheck(f"TABLE XI CI-hi {b} {q}", hi, s["dacc_ci"][1], 0.0005, "TABLE XI")
    cellcheck(f"TABLE XI p {b} {q}", r[4], s["dacc_p"], 0.0015, "TABLE XI")
    cellcheck(f"TABLE XI dtok {b} {q}", r[5], s["dtok_rel"] * 100, 0.05, "TABLE XI")

# TABLE XII - baseline khong-debate tru DUS-11
BASE12 = {"majority voting": "ensemble_vote",
          "SC@3, Qwen2.5-3B": "self_consistency_a",
          "SC@3, Llama3.2-3B": "self_consistency_b",
          "SC@3, Gemma3-4B": "self_consistency_c"}
for r in md_table("**TABLE XII —")[1:]:
    b = NICE_XI[r[0].strip()]
    p = BASE12[r[1].replace("**", "").strip()]
    s = key(P6, f"baseline_showdown.vs_dus11.{b}|{p}")
    lo, hi = [x.strip(" []\\") for x in r[3].split(",")]
    cellcheck(f"TABLE XII dacc {b} {p}", r[2], s["dacc"], 0.0005, "TABLE XII")
    cellcheck(f"TABLE XII CI-lo {b} {p}", lo, s["dacc_ci"][0], 0.0005, "TABLE XII")
    cellcheck(f"TABLE XII CI-hi {b} {p}", hi, s["dacc_ci"][1], 0.0005, "TABLE XII")
    cellcheck(f"TABLE XII p {b} {p}", r[4], s["dacc_p"], 0.0015, "TABLE XII")
    cellcheck(f"TABLE XII dtok {b} {p}", r[5], s["dtok_rel"] * 100, 0.05, "TABLE XII")

# ---- VI-A van xuoi quanh TABLE XI/XII: nguong Bonferroni va cac diem noi bat ----
check("0.0033", "nguong Bonferroni, 15 phep so sanh k/benchmark (VI-A)",
      "0.05 / 15", 0.05 / 15, "VI-A")
check("0.00185", "nguong Bonferroni, 27 phep so sanh (TABLE XII / Limitations)",
      "0.05 / 27", 0.05 / 27, "VI-A")
check("0.0186", "p nominal cua Qwen2.5-3B tren MMLU (khong qua duoc hieu chinh)",
      f"{P6} :: baseline_showdown.vs_dus11['mmlu|self_consistency_a'].dacc_p",
      key(P6, "baseline_showdown.vs_dus11.mmlu|self_consistency_a.dacc_p"), "VI-A")
_XI_DTOK = [abs(key(P6, f"baseline_showdown.fixed_k2_vs_dus11.{b}|fixed_k2-{q}")["dtok_rel"]) * 100
            for b in ("gsm8k", "mmlu", "strategyqa") for q in ("unc<q40", "unc<q50")]
check("8.0", "Δ Token NHO NHAT trong 6 phep so sanh fixed_k2 vs DUS-11 (TABLE XI)",
      f"{P6} :: min |baseline_showdown.fixed_k2_vs_dus11.*.dtok_rel|", min(_XI_DTOK), "VI-A")
check("29.7", "Δ Token LON NHAT trong 6 phep so sanh fixed_k2 vs DUS-11 (TABLE XI)",
      f"{P6} :: max |baseline_showdown.fixed_k2_vs_dus11.*.dtok_rel|", max(_XI_DTOK), "VI-A")
_XII_DACC = [abs(key(P6, f"baseline_showdown.vs_dus11.{b}|{p}")["dacc"]) * 100
             for b in ("gsm8k", "mmlu", "strategyqa")
             for p in ("ensemble_vote", "self_consistency_a", "self_consistency_b", "self_consistency_c")]
check("25.2", "chenh lech LON NHAT giua 12 baseline khong-debate va DUS-11, con so vuot qua hieu chinh (TABLE XII)",
      f"{P6} :: max |baseline_showdown.vs_dus11.*.dacc| qua 12 baseline khong-debate",
      max(_XII_DACC), "VI-A")

# TABLE 0-A - do on dinh cua cong Critic-SC (stage 1) qua seed va qua so phieu
CSC = load(P5)["critic_sc_stability"]
for r in md_table("**TABLE 0-A —")[1:]:
    b = NICE[r[0].strip()]
    flip = CSC["stability_over_n_votes"][b]["flip_rate_k1_to_k3"] * 100
    cellcheck(f"TABLE 0-A flip rate {b}", r[1], flip, 0.1, "TABLE 0-A")
    mean_s, sd_s = r[2].split("±")
    seed_stats = CSC["stability_over_seed"][b]
    cellcheck(f"TABLE 0-A gate rate mean {b}", mean_s, seed_stats["mean"] * 100, 0.2, "TABLE 0-A")
    cellcheck(f"TABLE 0-A gate rate SD {b}", sd_s.replace("pp", ""),
              seed_stats["sd"] * 100, 0.2, "TABLE 0-A")
    cellcheck(f"TABLE 0-A CV {b}", r[3], seed_stats["cv_pct"], 0.2, "TABLE 0-A")

# TABLE VII-A - fixed_k3/k4/k5 ghep cap voi fixed_k2 (Bonferroni tren 9 phep so sanh)
FKV = load(P5)["fixed_k4_k5_vs_fixed_k2"]["results"]
check(f"{0.05 / 9:.4f}", "nguong Bonferroni, 9 phep so sanh", f"{P5} :: 0.05 / 9",
      0.05 / 9, "TABLE VII-A")
for r in md_table("**TABLE VII-A —")[1:]:
    b = NICE[r[0].strip()]
    for i, k in enumerate((3, 4, 5)):
        cell, pcell = r[1 + i * 2], r[2 + i * 2]
        dacc_s, ci_s = cell.split(" [")
        lo, hi = ci_s.rstrip("]").split(",")
        s = FKV[f"{b}|fixed_k{k}_vs_fixed_k2"]
        cellcheck(f"TABLE VII-A {b} fixed_k{k}-fixed_k2 dacc", dacc_s, s["dacc"],
                  0.0005, "TABLE VII-A")
        cellcheck(f"TABLE VII-A {b} fixed_k{k}-fixed_k2 CI lo", lo, s["dacc_ci"][0],
                  0.0005, "TABLE VII-A")
        cellcheck(f"TABLE VII-A {b} fixed_k{k}-fixed_k2 CI hi", hi, s["dacc_ci"][1],
                  0.0005, "TABLE VII-A")
        p_s = pcell.replace("\\*", "").replace("*", "").strip()
        cellcheck(f"TABLE VII-A {b} fixed_k{k}-fixed_k2 p", p_s, s["dacc_p"],
                  0.0015, "TABLE VII-A")

# TABLE VIII-A - phan tich cascade thanh Stage1-only / Stage2-only / Full
CCA = load(P5)["cascade_component_ablation"]
CAB = load(P5)["cascade_ablation"]
for r in md_table("**TABLE VIII-A —")[1:]:
    b = NICE[r[0].strip()]
    label = r[1].replace("*", "").strip()
    variant = ("stage1_only" if label.startswith("Stage1") else
               "stage2_only" if label.startswith("Stage2") else "full")
    v = CCA[b]["unc<q50"][variant]
    cellcheck(f"TABLE VIII-A {b} {variant} acc", r[2], v["acc"], 0.0005, "TABLE VIII-A")
    cellcheck(f"TABLE VIII-A {b} {variant} token%", r[3].replace("*", ""),
              v["tok_pct"], 0.05, "TABLE VIII-A")
    share_cell = r[4].replace("*", "").strip()
    ab = CAB[b]
    if variant == "full":
        pct_str = share_cell.split("(")[1].split("%")[0]
        cellcheck(f"TABLE VIII-A {b} full earlier-share", pct_str,
                  ab["earlier_than_consensus"] * 100, 0.2, "TABLE VIII-A")
    elif variant == "stage1_only":
        cellcheck(f"TABLE VIII-A {b} stage1 share", share_cell,
                  ab["earlier_due_to_stage1"] / ab["earlier_than_consensus"] * 100,
                  0.2, "TABLE VIII-A")
    else:
        cellcheck(f"TABLE VIII-A {b} stage2 share", share_cell,
                  ab["earlier_due_to_stage2"] / ab["earlier_than_consensus"] * 100,
                  0.2, "TABLE VIII-A")

# ================================================================ [C] tinh lai tu log tho
print("Doc log tho ...")
DF = logio.load_rounds()
DF["maxr"] = DF.groupby(["seed", "task", "sample_id"]).round_id.transform("max")
NF = DF[DF.round_id < DF.maxr]
print("Tinh lai tu log tho ...")


def lc(claim: str, desc: str, how: str, value, section: str) -> None:
    live.append(dict(claim=claim, desc=desc, how=how, computed=value))
    check(claim, desc, f"LIVE: {how}  [{LOGS}]", value, section)


# dang thuc answer_entropy = 0 <=> consensus, tren MOI round
lc("27000", "so round dang thuc entropy=0 <=> consensus dung",
   "dem round co (answer_entropy == 0) == consensus_now",
   int(((DF.answer_entropy == 0) == DF.consensus_now).sum()), "V-B")

# chong lan cau hoi giua cac seed - co so cua bootstrap phan cum o TABLE XI
import itertools as _it
_Q = DF.drop_duplicates(["seed", "task", "sample_id"])
for _b, _claim in (("gsm8k", "978"), ("mmlu", "1123"), ("strategyqa", "647")):
    _d = _Q[_Q.task == _b]
    lc(_claim, f"so cau hoi DUY NHAT tren {_b} (1500 luot rut)",
       f"nunique(sample_id) tren {_b}", int(_d.sample_id.nunique()), "VI-A")
_ov = []
for _b in ("gsm8k", "mmlu", "strategyqa"):
    _s = {k: set(g.sample_id) for k, g in _Q[_Q.task == _b].groupby("seed")}
    _ov += [len(a & b) / 300 * 100 for a, b in _it.combinations(_s.values(), 2)]
lc("46", "chong lan cau hoi lon nhat giua hai seed bat ky (%)",
   "max |giao| / 300 tren moi cap seed, moi benchmark", round(max(_ov)), "VI-A")
lc("12", "chong lan cau hoi nho nhat giua hai seed bat ky (%)",
   "min |giao| / 300 tren moi cap seed, moi benchmark", round(min(_ov)), "VI-A")

# do giong prompt: vi sao fixed_k1 != majority voting chay rieng (IV-D)
import difflib as _dl
from tasks import get_benchmark as _gb
_QPLACE = "WHAT_IS_THE_QUESTION"
_sim, _len = {}, {}
for _b in ("gsm8k", "mmlu", "strategyqa"):
    _bm = _gb(_b)
    _a = _bm.build_solver_prompt(_QPLACE, 0, None)
    _c = _bm.build_independent_prompt(_QPLACE)
    _sim[_b] = _dl.SequenceMatcher(None, _a, _c).ratio()
    _len[_b] = (len(_a), len(_c))
for _b, _claim in (("gsm8k", "0.992"), ("mmlu", "0.994"), ("strategyqa", "0.027")):
    lc(_claim, f"do giong prompt solver(r0) vs independent tren {_b}",
       f"SequenceMatcher.ratio(), cau hoi thay bang {_QPLACE!r}", _sim[_b], "IV-D")
lc("2.7", "do giong prompt tren strategyqa, dang phan tram",
   "sim(strategyqa) * 100", _sim["strategyqa"], "IV-D")
lc("8390", "do dai prompt solver(r0) tren strategyqa",
   f"len(build_solver_prompt), cau hoi thay bang {_QPLACE!r}", _len["strategyqa"][0], "IV-D")
lc("1007", "do dai prompt independent tren strategyqa",
   f"len(build_independent_prompt), cau hoi thay bang {_QPLACE!r}", _len["strategyqa"][1], "IV-D")

# khoang cach lon nhat giua hai cach do CUNG mot majority vote (TABLE 0)
_S6 = load(P6)["baseline_showdown"]["summary"]
_gap = max(abs(_S6[_b]["fixed_k1"]["acc"] - _S6[_b]["ensemble_vote"]["acc"])
           for _b in ("gsm8k", "mmlu", "strategyqa")) * 100
lc("6.7", "chenh lech lon nhat giua fixed_k1 va majority voting chay rieng (diem)",
   "max |acc(fixed_k1) - acc(ensemble_vote)| * 100 qua 3 benchmark", _gap, "V")

# dap an khong parse duoc - chi MMLU bi, vi _valid_action khong kiem dinh dang 4 lua chon
_VALID = {"mmlu": {"A", "B", "C", "D"}, "strategyqa": {"YES", "NO"}}


def _parse_ok(task: str, v: str) -> bool:
    v = str(v).strip().upper()
    if task in _VALID:
        return v in _VALID[task]
    try:
        float(v)
        return True
    except ValueError:
        return False


_tot = collections.Counter()
_bad = collections.Counter()
for _f in glob.glob("results/logs/**/debate_full_*.jsonl", recursive=True):
    for _line in open(_f, encoding="utf-8"):
        _line = _line.strip()
        if not _line:
            continue
        _r = json.loads(_line)
        for _rd in _r["rounds"]:
            for _role, _a in _rd["agents"].items():
                _tot[(_r["task"], _role)] += 1
                if not _parse_ok(_r["task"], _a["normalized_answer"]):
                    _bad[(_r["task"], _role)] += 1
_mB = sum(v for k, v in _bad.items() if k[0] == "mmlu")
_mT = sum(v for k, v in _tot.items() if k[0] == "mmlu")
lc("1701", "so dap an MMLU nam ngoai A-D", "dem tren log debate", _mB, "VI-C")
lc("6.30", "ty le dap an MMLU khong parse duoc", f"{_mB} / {_mT}", _mB / _mT * 100, "VI-C")
for _role, _claim in (("solver_b", "13.4"), ("solver_a", "2.5"), ("critic", "3.0")):
    lc(_claim, f"ty le khong parse duoc tren MMLU, {_role}",
       f"{_bad[('mmlu', _role)]} / {_tot[('mmlu', _role)]}",
       _bad[("mmlu", _role)] / _tot[("mmlu", _role)] * 100, "VI-C")
lc("0.00", "ty le khong parse duoc tren GSM8K va StrategyQA",
   "dem tren log debate, ca hai benchmark",
   sum(v for k, v in _bad.items() if k[0] != "mmlu") * 100.0, "VI-C")

# worst-case regret: moi chinh sach thua bao nhieu so voi chinh sach TOT NHAT
# tren tung benchmark (bo oracle - khong dat duoc)
_BEST = {_b: max(v["acc"] for p, v in _S6[_b].items() if p != "oracle")
         for _b in ("gsm8k", "mmlu", "strategyqa")}


def _regret(policy: str) -> float:
    return max((_BEST[_b] - _S6[_b][policy]["acc"]) * 100
               for _b in ("gsm8k", "mmlu", "strategyqa"))


for _p, _claim in (("unc<q50", "3.5"), ("fixed_k2", "4.2"), ("self_consistency_c", "6.7"),
                   ("ensemble_vote", "8.3"), ("self_consistency_a", "15.9")):
    lc(_claim, f"worst-case regret cua {_p} (diem accuracy)",
       "max qua benchmark cua (acc tot nhat - acc chinh sach) * 100", _regret(_p), "VI-A")

lc("29", "do trai rong lon nhat giua ba model self-consistency tren mot benchmark",
   "max - min acc cua SC a/b/c tren MMLU, * 100",
   (max(_S6["mmlu"][f"self_consistency_{x}"]["acc"] for x in "abc")
    - min(_S6["mmlu"][f"self_consistency_{x}"]["acc"] for x in "abc")) * 100, "V")

# vong khong-cuoi khong doi ket cuc
last_ok = DF[DF.round_id == DF.maxr].set_index(["seed", "task", "sample_id"]).ok_if_stop
# NF co nhieu dong / debate, last_ok co mot -> chieu last_ok theo index cua NF
same = pd.Series(
    NF.ok_if_stop.to_numpy()
    == last_ok.reindex(pd.MultiIndex.from_frame(NF[["seed", "task", "sample_id"]])).to_numpy())
lc("20385", "so vong khong-cuoi chot cung ket cuc voi vong cuoi",
   "voi moi round r < round cuoi: so ok_if_stop[r] voi ok_if_stop[cuoi]",
   int(same.sum()), "V-A")
lc("90.6", "ty le vong khong-cuoi khong doi ket cuc",
   f"{int(same.sum()):,} / {len(NF):,}", same.sum() / len(NF), "V-A")

# blind spot
CN = DF[DF.consensus_now]
lc("11754", "so round co consensus", "dem round consensus_now == True", len(CN), "V-A")
lc("2020", "so round consensus nhung chot sai",
   "trong round consensus, dem ok_if_stop == False", int((~CN.ok_if_stop).sum()), "V-A")
lc("17.2", "blind-spot rate chung", f"{int((~CN.ok_if_stop).sum()):,} / {len(CN):,}",
   (~CN.ok_if_stop).mean(), "V-A")
for nice, t in [("2.9", "gsm8k"), ("28.0", "mmlu"), ("24.4", "strategyqa")]:
    g = CN[CN.task == t]
    lc(nice, f"blind-spot rate {t}",
       f"trong round consensus cua {t}, ty le ok_if_stop == False",
       (~g.ok_if_stop).mean(), "TABLE I")

# kich thuoc split theo cau hoi
SP = logio.load_splits()
for t in ("gsm8k", "mmlu", "strategyqa"):
    for s in ("train", "val", "test"):
        n = sum(1 for (tt, _), ss in SP.items() if tt == t and ss == s)
        lc(str(n), f"so cau hoi split {s} cua {t}",
           f"dem sample_id trong results/split/{t}/split_sample_ids.json", n, "IV-C")

# accuracy cua `always` tren MMLU: toan bo va rieng tap test
MM = DF[(DF.task == "mmlu") & (DF.round_id == DF.maxr)]
lc("0.547", "accuracy cua `always` tren toan bo MMLU",
   "trong moi debate mmlu: trung binh theo seed cua ty le dung o round cuoi",
   MM.groupby("seed").ok_if_stop.mean().mean(), "V-E")
MT = MM[[SP.get(("mmlu", int(s))) == "test" for s in MM.sample_id]]
lc("0.587", "accuracy cua `always` tren tap test MMLU",
   "trong debate mmlu thuoc split test: trung binh theo seed cua ty le dung o round cuoi",
   MT.groupby("seed").ok_if_stop.mean().mean(), "V-E")

# ================================================================ [D] quet mo coi
# bo muc REFERENCES (nam xuat ban) va cac trich dan [n] truoc khi quet
SCAN = BODY.split("## **REFERENCES**")[0]
SCAN = SCAN.replace(chr(92) + "[", "[").replace(chr(92) + "]", "]")   # bo escape markdown
SCAN = re.sub(r"\[\s*\d+\s*\]", " ", SCAN)                           # bo trich dan [n]
# lookbehind: khong bat "4" hay "-4" trong "DUS-4"; bat buoc co chu so sau dau cham
NUM = re.compile(r"(?<![\w-])-?\d[\d,]*(?:\.\d+)?")
IGNORE = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "20", "70",
    "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "1.0", "0.918", "1.585", "0.222",
    "100", "100.0", "16.7", "33.3", "50.0", "83.3", "66.7", "300", "1024", "768",
    "1500", "4500", "2000", "25", "75", "0.500", "1.58", "9", "16", "10", "30",
    "40", "60", "1e-3", "0.05", "24", "0.35", "-1", "150", "50", "95",
    "6.25", "28",              # xac suat trung ngau nhien tren 4 lua chon (giai tich, VI-A)
    "0.1", "0.6",              # con so cua [7] trich trong Related Work, khong phai so cua ta
    "01",                      # artifact: regex gop "[0,1]" (khoang gia tri confidence) thanh "0,1"
    "345",                     # artifact: regex gop "*k*=3,4,5" (TABLE VII-A tieu de) thanh "3,4,5"
    "10000",                   # "10,000 resamples" - tham so bootstrap, khong phai ket qua
    "27",                      # so phep so sanh (TABLE XII / Limitations) - tham so thiet ke
    "15",                      # nguong CV pre-registered "15%/25% bar" (V-A) - tham so thiet ke
    "200",                     # "1.000 from 200 [cau hoi]" - tham so power analysis, gia tri power da kiem o noi khac
    "7000",                    # "~7,000" - so uoc luong, tac gia tu ghi ro la xap xi
    "14",                      # "7-14 points" (V, doc mo ta) - khoang uoc luong, khong phai so don le
    "0.76",                    # "p=0.76" - lam tron 2 chu so cua 0.764 (da kiem chinh xac o TABLE IX)
}
orphans = sorted({_norm(m) for m in NUM.findall(SCAN)} - covered - IGNORE)

# ================================================================ xuat
n_ok = sum(r["ok"] for r in rows)
print("\n" + "=" * 78)
print(f"[A]+[B2]+[C]  {n_ok} / {len(rows)} claim KHOP")
bad = [r for r in rows if not r["ok"]]
if bad:
    print(f"\n{len(bad)} claim KHONG khop:")
    for r in bad:
        print(f"  {r['section']:<12s} claim={r['claim']:<14s} computed={r['computed']}"
              f"  ({r['description']})")
else:
    print("Khong co claim nao lech.")
print(f"\n[D]  {len(orphans)} so trong paper chua duoc phu:"
      f" {orphans if orphans else '(khong co)'}")

md = [f"# Provenance — {PAPER.name}", "",
      f"Moi so trong bai doi chieu voi file ket qua hoac tinh lai tu log tho. "
      f"**{n_ok} / {len(rows)} khop.**", "",
      "Sinh boi `python audit_paper.py`. Script chi doc, khong tinh lai bat ky "
      "phep tinh nao cua pipeline.", "",
      "| Section | Claim | Y nghia | Nguon | Gia tri nguon | Khop |",
      "| :---- | ----: | :---- | :---- | ----: | :----: |"]
for r in rows:
    c = r["computed"]
    cs = f"{c:.6g}" if isinstance(c, (int, float)) and not isinstance(c, bool) else str(c)
    md.append(f"| {r['section']} | `{r['claim']}` | {r['description']} | "
              f"`{r['source']}` | {cs} | {'OK' if r['ok'] else '**LECH**'} |")
md += ["", "## Numbers in the paper not covered above", "",
       ("(khong co)" if not orphans else ", ".join(f"`{o}`" for o in orphans))]
OUT_MD.write_text("\n".join(md), encoding="utf-8")

(R / "paper_audit.json").write_text(json.dumps(
    dict(n_claims=len(rows), n_ok=n_ok, claims=rows, live=live, orphans=orphans),
    indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n-> {OUT_MD}")
print(f"-> {R / 'paper_audit.json'}")
