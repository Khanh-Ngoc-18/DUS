"""
run_multi_agent_debate.py

Chạy thí nghiệm debate THẬT: 2 solver (Qwen + Llama, local qua Ollama) +
1 critic (Gemma3, local qua Ollama), trên N câu hỏi của
1 hoặc nhiều benchmark (gsm8k, strategyqa, mmlu), tối đa n vòng mỗi câu.

Output cho mỗi benchmark, trong results/logs/<task>/seed<seed>/:
  debate_full_<timestamp>.jsonl
      1 dòng JSON / câu hỏi — TOÀN BỘ debate: mọi vòng, mọi agent (answer,
      normalized_answer, confidence, reasoning, verdict của critic), 4 raw
      feature + consensus mỗi vòng, final_answer, final_correct, total_tokens,
      ground_truth, seed.
      Đây là nguồn chân lý DUY NHẤT — mọi script phân tích đọc file này qua
      logio.load_rounds() (không còn debate_rounds/debate_agents CSV).

Usage:
    python run_multi_agent_debate.py --task gsm8k --n 100 --max_rounds 6
    python run_multi_agent_debate.py --task all --n 100 --max_rounds 6
    python run_multi_agent_debate.py --task strategyqa --n 100 --resume
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from src.orchestrator import MADOrchestrator

logging.basicConfig(
    level=logging.WARNING,  # đặt INFO nếu muốn xem chi tiết từng agent/round
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_multi_agent_debate")

SOLVER_A_CONFIG = "config/model_config_solver_qwen.yaml"   # Solver A = Qwen
SOLVER_B_CONFIG = "config/model_config_solver_llama.yaml"  # Solver B = Llama
CRITIC_CONFIG  = "config/model_config_critic_gemma.yaml"   # Critic = Gemma3
CRITIC_CONFIGS = [CRITIC_CONFIG]                            

ALL_TASKS = ["gsm8k", "strategyqa", "mmlu"]


def load_samples(task: str, n: int, seed: int) -> list:
    data_path = Path(f"data/{task}/test.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if n < len(data):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(data)), n))
    else:
        if n > len(data):
            logger.warning(
                f"[{task}] request n={n} but only {len(data)} samples in data/{task}/test.json "
                f"→ running all {len(data)} samples."
            )
        idxs = list(range(len(data)))
    return [(i, data[i]) for i in idxs]


def load_completed_ids(jsonl_path: Path) -> set:
    """Đọc các file jsonl cùng task đã có (mọi timestamp) để hỗ trợ --resume."""
    done = set()
    if not jsonl_path.parent.exists():
        return done
    for f in jsonl_path.parent.glob("debate_full_*.jsonl"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    done.add(rec["sample_id"])
        except Exception as e:
            logger.warning(f"Cannot read {f}: {e}")
    return done


def run_task(task: str, n: int, max_rounds: int, seed: int, resume: bool, early_stop: bool):
    # base_agent._call_ollama doc MAD_SEED de seed sampling => moi seed = 1 rollout doc lap, tai lap duoc.
    os.environ["MAD_SEED"] = str(seed)
    # Tach log theo seed: khong de 2 seed ghi de nhau, va tranh dung sample_id
    # (sample_id = index cau hoi) va cham nhau khi phan tich. --resume cung tu dong
    # gioi han trong dung seed nay (load_completed_ids doc theo jsonl_path.parent).
    out_dir = Path(f"results/logs/{task}/seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"debate_full_{timestamp}.jsonl"

    completed = load_completed_ids(jsonl_path) if resume else set()
    if completed:
        logger.warning(f"[{task}] --resume: skipping {len(completed)} sample_id that have already been run")

    samples = load_samples(task, n, seed)
    samples = [(i, s) for i, s in samples if i not in completed]

    orchestrator = MADOrchestrator(
        task=task,
        solver_a_config_path=SOLVER_A_CONFIG,
        solver_b_config_path=SOLVER_B_CONFIG,
        critic_config_paths=CRITIC_CONFIGS,
        max_rounds=max_rounds,
        early_stop_on_consensus=early_stop,
    )

    n_correct, n_done = 0, 0

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i, (sample_id, sample) in enumerate(samples, 1):
            question = sample["question"]
            gt = sample.get("answer")

            t0 = time.time()
            try:
                record = orchestrator.run(sample_id=sample_id, question=question, ground_truth=gt)
            except Exception as e:
                logger.error(f"[{task}] Sample {sample_id} lỗi: {e} → bỏ qua")
                continue
            dt = time.time() - t0

            record["seed"] = seed
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()

            n_done += 1
            if record["final_correct"]:
                n_correct += 1

            acc = n_correct / n_done * 100
            print(
                f"[{task}] {i}/{len(samples)} (id={sample_id}) "
                f"rounds={record['rounds_used']} correct={record['final_correct']} "
                f"acc_so_far={acc:.1f}% ({dt:.1f}s)"
            )

    print(f"\n[{task}] DONE — {n_done} câu, accuracy={n_correct}/{n_done} "
          f"({(n_correct/n_done*100 if n_done else 0):.1f}%)")
    print(f"  full log : {jsonl_path}")


def main():
    p = argparse.ArgumentParser(description="2-solver + 1-critic debate runner")
    p.add_argument("--task", choices=ALL_TASKS + ["all"], default="all")
    p.add_argument("--n", type=int, default=100, help="Number of questions per benchmark")
    p.add_argument("--max_rounds", type=int, default=6, help="Maximum debate rounds per question")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling n questions")
    p.add_argument("--resume", action="store_true", help="Skip sample_id that have already been run (read from existing jsonl)")
    p.add_argument("--no_early_stop", action="store_true",
                    help="Always run for max_rounds, don't stop early when 3 agents agree")
    p.add_argument("--verbose", action="store_true", help="Log detailed information for each agent/round")

    # --- no-debate baselines
    mv = p.add_argument_group("no-debate baselines")
    mv.add_argument("--majority_voting", action="store_true",
                    help="Run a no-debate baseline instead: models answer independently, "
                         "then a majority vote decides. Logs to results/logs_mv/ (or "
                         "results/logs_sc/ with --sc_model), never mixed into the debate logs")
    mv.add_argument("--mv_votes", type=int, default=1, metavar="K",
                    help="Independent samples per model (default 1). Without --sc_model this "
                         "gives 3K votes from 3 models; with --sc_model it gives K votes from one")
    mv.add_argument("--sc_model", choices=["a", "b", "c"], default=None,
                    help="Switch to TRUE self-consistency: use only this one voter and take "
                         "--mv_votes samples from it (needs --mv_votes >= 2). a=solver_a, "
                         "b=solver_b, c=critic-slot model (rotates by sample_id as in debate)")
    # --- end no-debate baselines

    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    for task in tasks:
        if args.majority_voting:
            from mv_baseline import run_majority_vote_task
            run_majority_vote_task(
                task=task, n=args.n, seed=args.seed, resume=args.resume,
                n_votes=args.mv_votes, load_samples=load_samples,
                configs=(SOLVER_A_CONFIG, SOLVER_B_CONFIG, CRITIC_CONFIGS),
                sc_model=args.sc_model,
            )
            continue
        run_task(
            task=task,
            n=args.n,
            max_rounds=args.max_rounds,
            seed=args.seed,
            resume=args.resume,
            early_stop=not args.no_early_stop,
        )


if __name__ == "__main__":
    sys.exit(main())
