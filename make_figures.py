"""make_figures.py - Sinh hinh so lieu cua paper tu results/ (tra loi review muc 22, 23).

  Fig. 2  fig2_circularity    - TABLE III thanh bieu do: 33.9% bao cao vs 0.0% thuc te (muc 22)
  Fig. 3  fig3_cost_accuracy  - cot: chi phi token giam, accuracy khong doi (muc 23)

Nguyen tac: HINH LA BIEU DO, khong phai hop chu. Moi giai thich de o caption trong
paper, trong hinh chi giu nhan truc, nhan gia tri va chu thich ngan.

Moi so doc TRUC TIEP tu file ket qua, khong go tay:
  Fig. 2 <- results/verify_threshold.json
  Fig. 3 <- results/p3_holdout_policy.json  (nested CV, nguong T chon tren du lieu giu rieng)

Xuat ca PNG (300 dpi, de nhung Word) va SVG (vector, de nhung LaTeX) vao figures/.

Cach dung:  python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle)
import numpy as np
import pandas as pd

OUT = Path("figures")
R = Path("results")

# an toan voi nguoi mu mau + van phan biet duoc khi in den trang
C_CONS, C_NONCONS = "#4C72B0", "#C44E52"
C_ADAPT, C_FIXED, C_INCUMBENT = "#4C72B0", "#B0B0B0", "#C44E52"
BENCH = ["gsm8k", "mmlu", "strategyqa"]
NICE = {"gsm8k": "(a) GSM8K", "mmlu": "(b) MMLU", "strategyqa": "(c) StrategyQA"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
})


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  -> {p}")
    plt.close(fig)


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


# ================================================================ Fig. 2 (muc 22)
def fig_circularity() -> None:
    """Vi sao con so bao cao khong phai tiet kiem: nguong nam gon trong vung consensus."""
    vt = load("verify_threshold.json")
    T = vt["naive"]["threshold"]
    stop_rate = vt["naive"]["stop_rate"] * 100
    n_below = vt["circularity"]["n_below_threshold"]
    rounds_std = vt["sequential"]["rounds_standard_system"]
    saved = vt["sequential"]["real_rounds_saved"]

    d = pd.read_csv(R / "dus_per_round.csv")
    cons = d.consensus.astype(bool)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(d.dus.min(), d.dus.max(), 60)
    ax.hist(d.dus[cons], bins=bins, color=C_CONS, alpha=0.85, label="consensus round", zorder=3)
    ax.hist(d.dus[~cons], bins=bins, color=C_NONCONS, alpha=0.85, label="no consensus", zorder=3)
    ax.axvline(T, color="#222", lw=1.8, ls="--", zorder=5)
    ax.axvspan(bins[0], T, color="#222", alpha=0.07, zorder=1)

    ymax = ax.get_ylim()[1] * 1.14
    ax.set_ylim(0, ymax)
    ax.text(T + 0.06, ymax * 0.90, f"threshold $T$ = {T:.2f}", ha="left", va="center",
            fontsize=8, weight="bold")
    ax.annotate(f"stop region: {n_below:,} rounds ({stop_rate:.1f}%),\nevery one a consensus round",
                xy=(bins[0] + 0.06, ymax * 0.60), xytext=(T + 0.06, ymax * 0.72),
                ha="left", va="center", fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0))
    ax.set_xlabel("uncertainty score  $u_r$")
    ax.set_ylabel("rounds")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.6, zorder=0)
    ax.set_title(f"{saved} of {rounds_std:,} rounds elided",
                 loc="left", fontsize=9, color="#333")
    save(fig, "fig2_circularity")


# ================================================================ Fig. 3
# Bai toan cua paper la CHI PHI, accuracy la RANG BUOC (muc IV-E). Nen hang tren
# la chi phi token - dai luong duoc toi uu - va hang duoi chi de kiem tra rang
# buoc co bi vi pham khong.
#
# Moc so sanh la `always` (chay du 6 vong), giong p2: cau hoi trien khai la "bo
# bot vong thi mat gi", nen he day du phai nam o goc toa do.
#
# Truc hang duoi neo theo HEADROOM CUA ORACLE, khong phai theo bien do cua chinh
# cac cot. Chon truc sao cho chenh lech "trong nho" thi cung thao tung nhu cat
# truc cho no "trong to"; headroom la moc co that - no la phan accuracy thuc su
# dang bo lo - nen no la thuoc do dung de doc cac chenh lech kia.
# Ba baseline khong-debate la BA CO CHE KHAC NHAU, khong phai mot ho - nen ba mau:
#   round-0 vote  : 3 model khac nhau, 1 mau moi model, LAY TU vong 0 cua debate
#                   (bat doi xung: 2 solver @0.3/1024 + critic @0.2/768)
#   majority vote : 3 model khac nhau, 1 mau moi model, chay RIENG va doi xung
#   SC@3          : MOT model, 3 mau tu chinh no
# Hai cai dau khac nhau o CACH DO chu khong o thuat toan (Section IV-D); cai thu
# ba khac han vi no doi da dang model lay da dang mau.
FIG3_FAM = {
    "incumbent": ("#C44E52", "consensus"),
    "fixed": ("#B0B0B0", "fixed depth ($k$=2)"),
    "round0": ("#DD8452", "ensemble vote"),
    "majvote": ("#937860", "majority voting"),
    "selfcons": ("#8172B3", "self-consistency@3 (1 model)"),
    "adaptive": ("#4C72B0", "adaptive (DUS-11)"),
}
FIG3_REF = "always"
SC_NICE = {
    "self_consistency_a": "Qwen",
    "self_consistency_b": "Llama",
    "self_consistency_c": "Gemma",
}


def _best(S: dict, b: str, cands: list) -> str:
    """Chinh sach chinh xac nhat trong mot ho, tren benchmark b."""
    return max(cands, key=lambda p: S[b][p]["acc"])


def fig_cost_accuracy() -> None:
    """Chi phi token (hang tren) va Delta accuracy so voi `always` (hang duoi)."""
    R6 = load("p3_holdout_policy.json")["baseline_showdown"]
    S, A = R6["summary"], R6["vs_always"]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 5.2),
        sharey="row",
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    # 1. Tinh do chech lech accuracy cuc dai (dacc) thuc te de thu nho gioi han truc y
    all_daccs = []
    for b in BENCH:
        sc = _best(S, b, [f"self_consistency_{x}" for x in "abc"])
        rows = [
            ("consensus", "consensus", "incumbent"),
            ("fixed_k2", "fixed $k$=2", "fixed"),
            ("fixed_k1", "round-0 vote", "round0"),
            ("ensemble_vote", "maj vote", "majvote"),
            (sc, "SC " + SC_NICE[sc], "selfcons"),
            ("unc<q50", "DUS 0.5", "adaptive"),
        ]
        for r in rows:
            all_daccs.append(abs(A[f"{b}|{r[0]}"]["dacc"] * 100))

    # Tinh gioi han y cho hang accuracy (nho hon rat nhieu so voi khi dung Oracle headroom)
    max_abs_dacc = max(all_daccs) if all_daccs else 2.0
    lim = max(max_abs_dacc * 1.25, 1.0)  # Tao khoang margin nho xung quanh du lieu

    for col, b in enumerate(BENCH):
        fk = "fixed_k2"
        sc = _best(S, b, [f"self_consistency_{x}" for x in "abc"])
        rows = [
            (FIG3_REF, "always", "adaptive"),
            ("consensus", "consensus", "incumbent"),
            (fk, "fixed $k$=2", "fixed"),
            ("fixed_k1", "round-0 vote", "round0"),
            ("ensemble_vote", "maj vote", "majvote"),
            (sc, "SC " + SC_NICE[sc], "selfcons"),
            ("unc<q50", "DUS 0.5", "adaptive"),
        ]
        names = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        colors = [
            FIG3_FAM[r[2]][0] if r[0] != FIG3_REF else "#333333" for r in rows
        ]
        x = np.arange(len(rows))
        top, bot = axes[0, col], axes[1, col]

        # ---- Hang tren: Chi phi token ----
        tok = np.array([S[b][n]["tok_rel"] * 100 for n in names])
        top.bar(x, tok, width=0.66, color=colors, zorder=3)
        for xi, v in zip(x, tok):
            top.text(
                xi,
                v + 2.5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#444",
                zorder=5,
            )
        top.set_ylim(0, 116)
        top.set_title(NICE[b], loc="left", fontsize=9)
        top.grid(axis="y", alpha=0.22, lw=0.6, zorder=0)
        top.set_xticks(x)
        top.set_xticklabels([])

        # ---- Hang duoi: Accuracy vs Always (Da xoa Oracle va thu nho scale) ----
        bot.axhline(0, color="#333", lw=1.0, zorder=4)
        for xi, n, c in zip(x, names, colors):
            if n == FIG3_REF:
                bot.plot(
                    [xi], [0], marker="D", ms=4.5, color="#333", zorder=6
                )
                continue
            bot.bar(
                xi,
                A[f"{b}|{n}"]["dacc"] * 100,
                width=0.66,
                color=c,
                zorder=3,
            )

        bot.set_ylim(-lim, lim)

        # Tao cac vạch chia (ticks) nho hon va chi tiet hon cho truc y
        step = 0.5 if lim <= 3 else (1.0 if lim <= 6 else 2.0)
        yticks = np.arange(
            -np.floor(lim / step) * step, np.floor(lim / step) * step + step, step
        )
        bot.set_yticks(yticks)

        bot.grid(axis="y", alpha=0.22, lw=0.6, zorder=0)
        bot.set_xticks(x)
        bot.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)

    axes[0, 0].set_ylabel("token cost (% of always)", fontsize=8)
    axes[1, 0].set_ylabel("accuracy vs always (accuracy points)", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=lbl)
        for c, lbl in FIG3_FAM.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    save(fig, "fig3_cost_accuracy")


if __name__ == "__main__":
    print("Fig. 2 circularity"); fig_circularity()
    print("Fig. 3 cost reduction"); fig_cost_accuracy()
    print("\nXong. 2 hinh o figures/ (PNG 300dpi + SVG).")
