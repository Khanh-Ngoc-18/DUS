

import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .agents.solver import SolverAgent
from .agents.critic import CriticAgent
from .communication.message import StructuredMessage
from tasks.base import BenchmarkConfig

logger = logging.getLogger(__name__)


def _shannon_entropy(labels: List[str]) -> float:
    if not labels:
        return 0.0
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    n = len(labels)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


def _population_variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)

@dataclass
class DebateAgentSpec:
    """One agent's identity for logging purposes (which model actually ran)."""
    name: str
    config_path: str
    model_name: str


class MADOrchestrator:
    """
    2 solvers (independent models) + 1 critic, debating for up to
    `max_rounds` rounds on a single benchmark task.
    """

    def __init__(
        self,
        task: str,
        solver_a_config_path: str,
        solver_b_config_path: str,
        critic_config_paths: List[str],
        max_rounds: int = 6,
        early_stop_on_consensus: bool = True,
    ):
        from tasks import get_benchmark

        self.benchmark: BenchmarkConfig = get_benchmark(task)
        self.max_rounds = max_rounds
        self.early_stop_on_consensus = early_stop_on_consensus

        self.solver_a_config_path = solver_a_config_path
        self.solver_b_config_path = solver_b_config_path
        # Alternate critic model across samples (round-robin, deterministic).
        self.critic_config_paths = critic_config_paths

        self.solver_a_model = self._peek_model_name(solver_a_config_path)
        self.solver_b_model = self._peek_model_name(solver_b_config_path)
        self.critic_models = [self._peek_model_name(p) for p in critic_config_paths]

    @staticmethod
    def _peek_model_name(config_path: str) -> str:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model", {}).get("name", "unknown")

    def _pick_critic_config(self, sample_id: int) -> str:
        idx = sample_id % len(self.critic_config_paths)
        return self.critic_config_paths[idx]


    def run(self, sample_id: int, question: str, ground_truth: Optional[str] = None) -> dict:
        # Fresh agent instances every sample: agents hold per-question state
        # (SolverAgent._r0_action, CriticAgent._independent_answer) that must
        # not leak across questions.
        solver_a = SolverAgent(config_path=self.solver_a_config_path)
        solver_b = SolverAgent(config_path=self.solver_b_config_path)

        critic_config_path = self._pick_critic_config(sample_id)
        critic = CriticAgent(config_path=critic_config_path)
        critic_model = self._peek_model_name(critic_config_path)

        rounds_log: List[dict] = []
        prev_answers: Optional[Dict[str, str]] = None  # {"solver_a":..,"solver_b":..,"critic":..}
        disagreement_running_count = 0
        total_tokens = 0

        critic_msgs_for_a: List[StructuredMessage] = []
        critic_msgs_for_b: List[StructuredMessage] = []

        final_round_msgs = None

        for round_id in range(self.max_rounds):
            msg_a = solver_a.respond(
                question=question,
                round_id=round_id,
                critic_messages=critic_msgs_for_a if round_id > 0 else None,
                benchmark=self.benchmark,
            )
            total_tokens += msg_a.token_count

            msg_b = solver_b.respond(
                question=question,
                round_id=round_id,
                critic_messages=critic_msgs_for_b if round_id > 0 else None,
                benchmark=self.benchmark,
            )
            total_tokens += msg_b.token_count

            critic_vs_a = critic.respond(
                question=question, solver_message=msg_a, round_id=round_id, benchmark=self.benchmark
            )
            total_tokens += critic_vs_a.token_count
            critic_vs_b = critic.respond(
                question=question, solver_message=msg_b, round_id=round_id, benchmark=self.benchmark
            )
            total_tokens += critic_vs_b.token_count

            critic_msgs_for_a.append(critic_vs_a)
            critic_msgs_for_b.append(critic_vs_b)

            critic_ind: StructuredMessage = critic_vs_a.independent_answer  # locked, same for both calls

            norm_a = self.benchmark.normalize_answer(msg_a.action.strip())
            norm_b = self.benchmark.normalize_answer(msg_b.action.strip())
            norm_c = self.benchmark.normalize_answer(critic_ind.action.strip())

            answers_now = {"solver_a": norm_a, "solver_b": norm_b, "critic": norm_c}
            confidences_now = {
                "solver_a": msg_a.confidence,
                "solver_b": msg_b.confidence,
                "critic": critic_ind.confidence,
            }

            consensus = len(set(answers_now.values())) == 1

            answer_entropy = _shannon_entropy(list(answers_now.values()))
            confidence_variance = _population_variance(list(confidences_now.values()))

            if not consensus:
                disagreement_running_count += 1
            disagreement_persistence = disagreement_running_count

            if prev_answers is None:
                flips = 0
            else:
                flips = sum(
                    1 for k in answers_now if answers_now[k] != prev_answers[k]
                )
            answer_flip_rate = flips / len(answers_now)

            round_record = {
                "round_id": round_id,
                "agents": {
                    "solver_a": {
                        "model": self.solver_a_model,
                        "answer": msg_a.action,
                        "normalized_answer": norm_a,
                        "confidence": msg_a.confidence,
                        "reasoning": list(msg_a.reasoning),
                        "reasoning_text": msg_a.get_reasoning_text(),
                    },
                    "solver_b": {
                        "model": self.solver_b_model,
                        "answer": msg_b.action,
                        "normalized_answer": norm_b,
                        "confidence": msg_b.confidence,
                        "reasoning": list(msg_b.reasoning),
                        "reasoning_text": msg_b.get_reasoning_text(),
                    },
                    "critic": {
                        "model": critic_model,
                        "answer": critic_ind.action,
                        "normalized_answer": norm_c,
                        "confidence": critic_ind.confidence,
                        "reasoning": list(critic_ind.reasoning),
                        "reasoning_text": critic_ind.get_reasoning_text(),
                        "verdict_vs_solver_a": critic_vs_a.action,
                        "verdict_vs_solver_a_confidence": critic_vs_a.confidence,
                        "verdict_vs_solver_b": critic_vs_b.action,
                        "verdict_vs_solver_b_confidence": critic_vs_b.confidence,
                    },
                },
                "consensus": consensus,
                "metrics": {
                    "answer_entropy": answer_entropy,
                    "confidence_variance": confidence_variance,
                    "disagreement_persistence": disagreement_persistence,
                    "answer_flip_rate": answer_flip_rate,
                },
            }
            rounds_log.append(round_record)

            prev_answers = answers_now
            final_round_msgs = (msg_a, msg_b, critic_ind)

            if consensus and self.early_stop_on_consensus:
                logger.info(f"[Sample {sample_id}] consensus reached at round {round_id} → stop")
                break

        final_answer = self._select_final_answer(final_round_msgs)

        correct = None
        if ground_truth is not None:
            correct = self.benchmark.score(final_answer, ground_truth)

        return {
            "sample_id": sample_id,
            "task": self.benchmark.task_name,
            "question": question,
            "ground_truth": ground_truth,
            "critic_model": critic_model,
            "solver_a_model": self.solver_a_model,
            "solver_b_model": self.solver_b_model,
            "rounds": rounds_log,
            "rounds_used": len(rounds_log),
            "final_answer": final_answer,
            "final_correct": correct,
            "total_tokens": total_tokens,
        }

    def _select_final_answer(self, final_round_msgs) -> str:
        msg_a, msg_b, critic_ind = final_round_msgs
        candidates = [
            (self.benchmark.normalize_answer(msg_a.action.strip()), msg_a.confidence),
            (self.benchmark.normalize_answer(msg_b.action.strip()), msg_b.confidence),
            (self.benchmark.normalize_answer(critic_ind.action.strip()), critic_ind.confidence),
        ]
        votes: Dict[str, int] = {}
        best_conf: Dict[str, float] = {}
        for ans, conf in candidates:
            votes[ans] = votes.get(ans, 0) + 1
            best_conf[ans] = max(best_conf.get(ans, 0.0), conf)

        max_votes = max(votes.values())
        tied = [a for a, v in votes.items() if v == max_votes]
        if len(tied) == 1:
            return tied[0]
        return max(tied, key=lambda a: best_conf[a])
