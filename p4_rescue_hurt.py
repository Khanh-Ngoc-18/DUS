"""P4 - Debate co dang dong tien khong? (nguon so lieu cua TABLE IX)

So dap an CHOT o round 0 voi dap an CHOT o round cuoi cua cung debate:

  rescued   - round 0 sai  ->  round cuoi dung   (debate cuu duoc)
  corrupted - round 0 dung ->  round cuoi sai    (debate lam hong)
  net       - rescued - corrupted

Round 0 la round DUY NHAT chua co thong tin cheo (orchestrator chi truyen
critic_messages khi round_id > 0), nen no chinh la moc "khong debate". Hieu so
giua hai moc do la toan bo gia tri ma sau round debate mang lai.

Dap an chot moi round lay qua logio.select_final (tai lap _select_final_answer:
majority 3 agent, hoa thi lay confidence cao nhat), cham theo dung benchmark.

Kem McNemar CHINH XAC tren cac cap khong khop (rescued + corrupted). Null la
"debate khong doi accuracy", tuc rescued va corrupted rut ra tu cung mot phan
phoi. Khong co p thi net am tren gsm8k chi la uoc luong diem: -8 tren 116 cap
khong khop khong phan biet duoc voi nhieu, va doc dau cua no la doc qua du lieu.

Output: results/p4_rescue_hurt.json
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pandas as pd

import logio

KEY = ["seed", "task", "sample_id"]


def mcnemar_p(rescued: int, corrupted: int) -> float:
    """McNemar chinh xac, hai phia. Nhi thuc(n, 1/2) doi xung nen p = 2*P(X <= m)."""
    n = rescued + corrupted
    if n == 0:
        return 1.0
    m = min(rescued, corrupted)
    tail = sum(comb(n, i) for i in range(m + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main() -> None:
    df = logio.load_rounds()
    df["maxr"] = df.groupby(KEY).round_id.transform("max")

    first = df[df.round_id == 0].set_index(KEY).ok_if_stop
    last = df[df.round_id == df.maxr].set_index(KEY).ok_if_stop
    rh = pd.DataFrame({"first": first, "last": last}).dropna().reset_index()
    rh["first"] = rh["first"].astype(bool)
    rh["last"] = rh["last"].astype(bool)

    out, tot_r, tot_h = {}, 0, 0
    print(f"{'benchmark':<12}{'n':>7}{'rescued':>18}{'corrupted':>18}{'net':>8}{'p':>9}")
    print("-" * 72)
    for task, g in rh.groupby("task"):
        resc = int((~g["first"] & g["last"]).sum())
        hurt = int((g["first"] & ~g["last"]).sum())
        tot_r += resc
        tot_h += hurt
        p = mcnemar_p(resc, hurt)
        out[task] = dict(n=len(g), rescued=resc, corrupted=hurt, net=resc - hurt,
                         rescued_pct=resc / len(g) * 100, corrupted_pct=hurt / len(g) * 100,
                         net_pct=(resc - hurt) / len(g) * 100,
                         n_discordant=resc + hurt, mcnemar_p=p)
        print(f"{task:<12}{len(g):>7,}{resc:>10,} ({resc/len(g)*100:4.1f}%)"
              f"{hurt:>10,} ({hurt/len(g)*100:4.1f}%){resc-hurt:>+8}{p:>9.3f}")

    n = len(rh)
    p_tot = mcnemar_p(tot_r, tot_h)
    out["total"] = dict(n=n, rescued=tot_r, corrupted=tot_h, net=tot_r - tot_h,
                        rescued_pct=tot_r / n * 100, corrupted_pct=tot_h / n * 100,
                        net_pct=(tot_r - tot_h) / n * 100,
                        n_discordant=tot_r + tot_h, mcnemar_p=p_tot)
    print("-" * 72)
    print(f"{'TOTAL':<12}{n:>7,}{tot_r:>10,} ({tot_r/n*100:4.1f}%)"
          f"{tot_h:>10,} ({tot_h/n*100:4.1f}%){tot_r-tot_h:>+8}{p_tot:>9.3f}"
          f"   ({(tot_r-tot_h)/n*100:+.2f}%)")

    Path("results/p4_rescue_hurt.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n-> results/p4_rescue_hurt.json")


if __name__ == "__main__":
    main()
