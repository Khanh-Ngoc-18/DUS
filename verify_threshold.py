"""Kiem chung quy tac "dung khi DUS < T": no thuc su dung duoc gi?

Doi chieu GIAO THUC NAIVE (quet nguong tren moi round, nhan = final_correct,
ty le tinh theo dong) voi GIAO THUC DUNG (chi round khong-cuoi, nhan tai chinh
round do, ty le tinh theo sample). Ket qua o muc [3]/[4] la co so cho phan
"cau hoi 1" cua bai: con so naive bao ra la dem lai, khong phai tiet kiem.

Doc results/dus_per_round.csv (27k round, co cot seed). Nguong T KHONG hardcode
nua ma suy ra tu chinh du lieu theo dung cach lam pho bien:
    T* = argmax stop_rate  s.t.  miss_rate(theo DONG) <= EPS

Output: results/verify_threshold.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 0.05
KEY = ["seed", "task", "sample_id"]      # 1 rollout = 1 (seed, task, cau hoi)

df = pd.read_csv("results/dus_per_round.csv").sort_values(KEY + ["round_id"])
df["final_correct"] = df["final_correct"].astype(bool)
df["maxr"] = df.groupby(KEY)["round_id"].transform("max")
df["is_last"] = df["round_id"] == df["maxr"]


def auc(score, label) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    r = pd.Series(score).rank().to_numpy()
    a, b = label.sum(), (~label).sum()
    return float("nan") if a == 0 or b == 0 else float((r[label].sum() - a * (a + 1) / 2) / (a * b))


out: dict = {}
dus, wrong, n = df.dus.to_numpy(), (~df.final_correct).to_numpy(), len(df)

print("== [1] Chon nguong theo GIAO THUC NAIVE ==")
print(f"  {n} round / {df.groupby(KEY).ngroups} rollout / {df.seed.nunique()} seed")
best = None
for T in np.unique(dus):
    stop = dus < T
    if not stop.any():
        continue
    miss_row = float((wrong & stop).sum()) / n
    if miss_row <= EPS and (best is None or stop.mean() > best["stop_rate"]):
        best = dict(T=float(T), stop_rate=float(stop.mean()), miss_row=miss_row)
T = best["T"]
stop = dus < T
print(f"  T* = {T:.6f}   (max stop_rate s.t. miss_rate theo DONG <= {EPS:.0%})")
print(f"  stop_rate BAO CAO = {best['stop_rate']:.4f}   <- thuong duoc goi la 'compute tiet kiem'")
print(f"  miss_rate theo DONG = {best['miss_row']:.4f}")
out["naive"] = dict(threshold=T, **{k: v for k, v in best.items() if k != "T"},
                    n_rounds=int(n), n_rollouts=int(df.groupby(KEY).ngroups))

print("\n== [2] Do nhay quanh nguong (T co phai mass point khong?) ==")
for mask, name in [(dus < T, "dus <  T"), (dus <= T, "dus <= T")]:
    print(f"  {name}: stop_rate={mask.mean():.4f}  miss_rate={(wrong & mask).sum() / n:.4f}")
n_eq = int((dus == T).sum())
print(f"  So dong co dus == T chinh xac: {n_eq}")
out["mass_point"] = dict(n_rows_equal_T=n_eq,
                         stop_rate_lt=float((dus < T).mean()),
                         stop_rate_le=float((dus <= T).mean()),
                         miss_rate_lt=float((wrong & (dus < T)).sum() / n),
                         miss_rate_le=float((wrong & (dus <= T)).sum() / n))

print("\n== [3] Quy tac nay thuc chat la gi? ==")
lo = df[df.dus < T]
ent_eq_cons = bool((df.answer_entropy == 0).equals(df.consensus.astype(bool)))
n_ent0 = int((df.answer_entropy == 0).sum())
n_ent0_last = int(df[df.answer_entropy == 0].is_last.sum())
print(f"  answer_entropy cua cac dong duoi nguong: {np.sort(lo.answer_entropy.unique())}")
print(f"  tat ca dong duoi nguong deu consensus: {bool(lo.consensus.all())}")
print(f"  entropy==0  <=>  consensus: {ent_eq_cons}  ({n_ent0}/{len(df)} dong entropy==0)")
print(f"  Trong he CHUAN (dung tai consensus dau tien), moi round consensus deu la round dung.")
out["circularity"] = dict(all_below_threshold_are_consensus=bool(lo.consensus.all()),
                          entropy_zero_iff_consensus=ent_eq_cons,
                          n_entropy_zero=n_ent0, n_entropy_zero_is_last=n_ent0_last,
                          n_below_threshold=int(len(lo)))

print("\n== [4] Mo phong dung som THUC SU tren he CHUAN (dung tai consensus dau tien) ==")
saved = fired = earlier = stopped_wrong = total = rounds_std = 0
for _, s in df.groupby(KEY, sort=False):
    total += 1
    cons = np.where(s.consensus.to_numpy().astype(bool))[0]
    c = int(cons[0]) if len(cons) else len(s) - 1      # round he CHUAN dung lai
    obs = s.iloc[: c + 1]                               # round he CHUAN thuc su chay
    rounds_std += len(obs)
    hit = np.where(obs.dus.to_numpy() < T)[0]
    if len(hit):
        fired += 1
        d = c - int(hit[0])                             # round elide duoc so voi he CHUAN
        saved += d
        earlier += int(d > 0)
        stopped_wrong += int(not s.final_correct.iloc[0])
miss_sample = stopped_wrong / total
print(f"  rollout={total}   round he CHUAN chay={rounds_std}")
print(f"  rollout co flag fire        = {fired} ({fired / total:.1%})")
print(f"  trong do dung SOM HON he CHUAN = {earlier} ({earlier / total:.1%})"
      f"   <- moi lan fire deu roi DUNG vao round he CHUAN da dung")
print(f"  compute tiet kiem THUC = {saved}/{rounds_std} round = {saved / rounds_std:.1%}")
print(f"  => BAO CAO {best['stop_rate']:.1%}, THUC TE {saved / rounds_std:.1%}")
print(f"  miss_rate theo SAMPLE = {miss_sample:.4f}  (theo DONG = {best['miss_row']:.4f}"
      f", thoi phong {miss_sample / best['miss_row'] if best['miss_row'] else float('nan'):.1f}x)")
out["sequential"] = dict(n_rollouts=total, rounds_standard_system=rounds_std,
                         n_flag_fired=fired, n_stopped_earlier=earlier,
                         real_rounds_saved=saved,
                         real_saving_rate=saved / rounds_std,
                         reported_stop_rate=best["stop_rate"],
                         miss_rate_per_sample=miss_sample,
                         miss_rate_per_row=best["miss_row"],
                         inflation=miss_sample / best["miss_row"] if best["miss_row"] else None)

print("\n== [5] Chi xet round KHONG PHAI round cuoi (noi duy nhat co the tiet kiem) ==")
nt = df[~df.is_last]
print(f"  n={len(nt)}  error_rate={(~nt.final_correct).mean():.4f}")
a_all = auc(df.dus, ~df.final_correct)
a_nt = auc(nt.dus, ~nt.final_correct)
print(f"  AUC(DUS) moi round      = {a_all:.4f}")
print(f"  AUC(DUS) round khong-cuoi = {a_nt:.4f}   (thoi phong {a_all - a_nt:+.4f})")
feat_auc = {}
for f in ["answer_entropy", "confidence_variance", "disagreement_persistence", "answer_flip_rate"]:
    feat_auc[f] = auc(nt[f], ~nt.final_correct)
    print(f"    {f:26s} AUC={feat_auc[f]:.4f}")
print(f"  min(DUS) tren nhom nay = {nt.dus.min():.6f}")
print(f"  So round se dung som   = {(nt.dus < T).sum()}")
per_task = {}
for t, g in nt.groupby("task"):
    ga = df[df.task == t]
    per_task[t] = dict(auc_all_rounds=auc(ga.dus, ~ga.final_correct),
                       auc_nonfinal=auc(g.dus, ~g.final_correct))
    print(f"    {t:12s} moi round={per_task[t]['auc_all_rounds']:.4f}  "
          f"khong-cuoi={per_task[t]['auc_nonfinal']:.4f}")
out["nonfinal"] = dict(n=int(len(nt)), error_rate=float((~nt.final_correct).mean()),
                       auc_all_rounds=a_all, auc_nonfinal=a_nt,
                       auc_inflation=a_all - a_nt, per_feature_auc=feat_auc,
                       min_dus=float(nt.dus.min()),
                       n_would_stop=int((nt.dus < T).sum()), per_task=per_task)

Path("results/verify_threshold.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\n-> results/verify_threshold.json")
