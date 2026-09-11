"""logio.py - Nguon chan ly DUY NHAT cho phan tich: doc debate_full_*.jsonl.

jsonl la superset cua debate_rounds/debate_agents CSV (con giau hon: reasoning,
verdict, total_tokens, question, ground_truth). Moi script phan tich import ham o
day thay vi doc CSV => het desync giua 3 file, het dus_per_round.csv stale.

Tu dong seed-aware: glob DE QUY, doc field "seed" trong moi ban ghi (mac dinh 0).

API:
  BASE, CRITIC          - ten 4 + 7 feature
  score(task, pred, gt) - cham dap an dung benchmark (gsm8k: so hoc, tolerance 1e-3)
  select_final(cands)   - tai lap Orchestrator._select_final_answer
  load_rounds(pattern)  - DataFrame 1 dong / (seed, task, sample_id, round_id)
  load_splits(root)     - dict {(task, sample_id): "train"|"val"|"test"} tu split mapping
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["answer_entropy", "confidence_variance", "disagreement_persistence", "answer_flip_rate"]
CRITIC = [
    "critic_conf", "n_disagree", "verdict_conf_mean", "verdict_conf_min",
    "critic_vs_majority", "critic_alone", "conf_gap_critic_solvers",
]

DEFAULT_PATTERN = "results/logs/**/debate_full_*.jsonl"


def score(task: str, pred: str, gt: str) -> bool:
    if task == "gsm8k":
        try:
            return abs(float(str(pred).strip()) - float(str(gt).strip())) < 1e-3
        except (ValueError, TypeError):
            pass
    return str(pred).strip().lower() == str(gt).strip().lower()


def select_final(cands: list[tuple[str, float]]) -> str:
    """Tai lap Orchestrator._select_final_answer: majority, hoa thi lay confidence cao nhat."""
    votes, best = {}, {}
    for a, c in cands:
        votes[a] = votes.get(a, 0) + 1
        best[a] = max(best.get(a, 0.0), c)
    mx = max(votes.values())
    tied = [a for a, v in votes.items() if v == mx]
    return tied[0] if len(tied) == 1 else max(tied, key=lambda a: best[a])


def load_rounds(pattern: str | None = None) -> pd.DataFrame:
    """Doc moi round tu jsonl. 1 dong / (seed, task, sample_id, round_id).

    Cot: seed, task, sample_id, round_id, rounds_used, final_correct, ok_if_stop,
    consensus_now, tok_per_round, total_tokens, ground_truth, BASE(4), CRITIC(7).
    """
    files = glob.glob(pattern or DEFAULT_PATTERN, recursive=True)
    if not files:
        raise SystemExit(f"Khong tim thay {pattern or DEFAULT_PATTERN}. Hay chay MAD.py truoc.")
    rows = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task, sid = r["task"], r["sample_id"]
            seed = int(r.get("seed", 0))
            gt = str(r["ground_truth"]).strip()
            rounds = r["rounds"]
            nr = max(len(rounds), 1)
            total_tokens = r.get("total_tokens", np.nan)
            fc = r.get("final_correct")
            for rd in rounds:
                ag = rd["agents"]
                a, b, c = ag["solver_a"], ag["solver_b"], ag["critic"]
                na = str(a["normalized_answer"]).strip()
                nb = str(b["normalized_answer"]).strip()
                nc = str(c["normalized_answer"]).strip()
                ca, cb, cc = float(a["confidence"]), float(b["confidence"]), float(c["confidence"])
                answers = [na, nb, nc]
                chosen = select_final([(na, ca), (nb, cb), (nc, cc)])
                vconf = [c.get("verdict_vs_solver_a_confidence"),
                         c.get("verdict_vs_solver_b_confidence")]
                vconf = [x for x in vconf if x is not None]
                m = rd["metrics"]
                rows.append(dict(
                    seed=seed, task=task, sample_id=sid, round_id=rd["round_id"],
                    rounds_used=r.get("rounds_used", nr),
                    final_correct=bool(fc) if fc is not None else None,
                    ok_if_stop=score(task, chosen, gt),
                    consensus_now=len(set(answers)) == 1,
                    tok_per_round=total_tokens / nr,
                    total_tokens=total_tokens,
                    ground_truth=gt,
                    answer_entropy=m["answer_entropy"],
                    confidence_variance=m["confidence_variance"],
                    disagreement_persistence=m["disagreement_persistence"],
                    answer_flip_rate=m["answer_flip_rate"],
                    critic_conf=cc,
                    n_disagree=sum(1 for x in (na, nb) if x != nc),
                    critic_vs_majority=float(answers.count(nc) == 1),
                    critic_alone=float(na == nb and nc != na),
                    conf_gap_critic_solvers=cc - np.mean([ca, cb]),
                    verdict_conf_mean=float(np.mean(vconf)) if vconf else np.nan,
                    verdict_conf_min=float(np.min(vconf)) if vconf else np.nan,
                ))
    return pd.DataFrame(rows).sort_values(
        ["seed", "task", "sample_id", "round_id"]).reset_index(drop=True)


def load_splits(root: str = "results/split") -> dict[tuple[str, int], str]:
    """Doc mapping (task, sample_id) -> split tu results/split/<task>/split_sample_ids.json."""
    splits: dict[tuple[str, int], str] = {}
    for f in glob.glob(str(Path(root) / "*" / "split_sample_ids.json")):
        task = Path(f).parent.name
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        for sp in ("train", "val", "test"):
            v = j.get(sp, [])
            ids = v if isinstance(v, list) else v.get("sample_ids", [])
            for sid in ids:
                splits[(task, int(sid))] = sp
    return splits
