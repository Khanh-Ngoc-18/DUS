from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.communication.message import StructuredMessage


class BenchmarkConfig(ABC):

    @property
    @abstractmethod
    def task_name(self) -> str: ...

    @property
    def answer_format(self) -> str:
        return "unknown"

    @property
    def solver_role(self) -> str:
        return "solver"

    @property
    def critic_role(self) -> str:
        return "critic"

    @property
    def judge_role(self) -> str:
        return "judge"


    @abstractmethod
    def build_solver_prompt(
        self,
        question: str,
        round_id: int,
        critic_messages: Optional[List["StructuredMessage"]] = None,
    ) -> str: ...

    @abstractmethod
    def build_independent_prompt(self, question: str) -> str: ...

    @abstractmethod
    def build_critic_prompt(
        self,
        question: str,
        solver_message: "StructuredMessage",
        critic_independent: "StructuredMessage",
    ) -> str: ...

    @abstractmethod
    def build_judge_prompt(
        self,
        question: str,
        solver_message: "StructuredMessage",
        critic_message: "StructuredMessage",
        verification_result: Optional[dict] = None,
        solver_initial_message: Optional["StructuredMessage"] = None,
    ) -> str: ...

    @abstractmethod
    def score(self, pred: str, gt: str) -> bool: ...

    def normalize_answer(self, raw: str) -> str:
        return raw.strip().lower()

    def allow_flip(self, critic_msg: "StructuredMessage") -> bool:
        text = (critic_msg.get_reasoning_text() or "").lower()
        allow_keywords = [
            "incorrect", "not true", "wrong", "evidence",
            "calculation", "miscalculated", "error", "factual",
            "actually", "in fact", "contradicts", "misidentif",
            "impossible", "cannot", "does not", "is not",
        ]
        if any(kw in text for kw in allow_keywords):
            return True
        if getattr(critic_msg, "confidence", 0) > 0.85 and critic_msg.action == "disagree":
            return True
        return False

    def should_early_stop(
        self,
        solver_msg: "StructuredMessage",
        critic_msg: "StructuredMessage",
        logical: dict,
        round_id: int = 0,
    ) -> bool:
        return (
            critic_msg.action == "agree"
            and logical.get("valid", False)
            and solver_msg.confidence > 0.98
        )

    def select_final_answer(
        self,
        question: str,
        solver_messages: List["StructuredMessage"],
        critic_messages: List["StructuredMessage"],
        verif_final: dict,
        selector,
        critic_ind_r0: Optional["StructuredMessage"] = None,
    ) -> "StructuredMessage":
        last_solver = solver_messages[-1]
        last_critic = critic_messages[-1]

        if getattr(last_solver, "unfounded_flip", False):
            r0 = solver_messages[0]
            r1 = solver_messages[-2] if len(solver_messages) >= 2 else solver_messages[0]
            # Pick whichever has fewer issues (caller provides verif via selector context)
            # Default: prefer r0 (original answer more reliable)
            return r0

        if verif_final["valid"] and last_critic.action == "agree":
            return last_solver

        return selector.select(
            question=question,
            solver_initial=solver_messages[0],
            solver_final=last_solver,
            critic_independent=critic_ind_r0,
        )

    def verifier_settings(self) -> dict:
        return {
            "check_arithmetic": True,
            "check_percentage": True,
        }