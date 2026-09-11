"""
agents/solver.py

Changes vs original:
- Calibrated confidence prompt (model must justify high confidence)
- Self-consistency sampling at round 0 for GSM8K
- Flip guard: require critic.action == "disagree" before allowing flip
- Uncertainty-aware logging
"""

import logging
from collections import Counter
from typing import List, Optional

from .base_agent import BaseAgent
from ..communication.message import StructuredMessage
from ..communication.protocol import (
    parse_agent_response,
    is_uncertain,
    has_specific_fact,
    looks_like_non_answer,
    build_compact_prompt,
)

logger = logging.getLogger(__name__)


# ── Factual-flip indicators ───────────────────────────────────────────────────

_FACTUAL_FLIP_INDICATORS = (
    "wrong", "incorrect", "false", "actually", "in fact",
    "error", "mistaken", "not true", "evidence", "source",
    "documented", "proven", "established",
)


# ── GSM8K reasoning rules (injected into prompt) ─────────────────────────────

_GSM_REASONING_RULES = """
IMPORTANT REASONING RULES:

1. Restart / Interruption (CRITICAL):
- If a process is restarted → ALL previous progress is LOST.
- DO NOT continue from remaining part or reuse previous partial work.
- Example: If 40% done and restart happens → that 40% is LOST → redo 100%.

2. Lost Time / Wasted Work (CRITICAL):
- Total time = time before interruption + restart time + time to redo full task.
- DO NOT ignore the wasted time.

3. Percentage (CRITICAL):
- "Increase by X%" → new value = original × (1 + X/100)
- +150% → ×2.5 | +50% → ×1.5
- DO NOT confuse "increase by X%" with "X% of original".

4. Base of percentage:
- Always identify WHAT the percentage applies to.
- Apply percentage to the CURRENT VALUE unless specified otherwise.

5. Profit:
- profit = final value − total cost
- Always use TOTAL investment when calculating profit.
"""


# ── Confidence calibration suffix ─────────────────────────────────────────────

_CALIBRATION_SUFFIX = """
Rate your confidence HONESTLY using these guidelines:
- 0.95+ : you can cite a specific fact (name, number, date, formula)
- 0.80  : you are reasoning from solid general knowledge
- 0.65  : you are inferring, not directly recalling
- 0.40  : you are guessing

If you cannot recall a specific supporting fact → confidence MUST be below 0.70.
Do NOT fake certainty. A calibrated answer beats an overconfident wrong answer.
"""




def _flip_has_factual_basis(critic_messages):
    if not critic_messages:
        return True

    for m in critic_messages:
        if m.action == "disagree" and m.confidence >= 0.7:
            return True

    combined = " ".join(
        (m.content or "") + " " + m.get_reasoning_text()
        for m in critic_messages
    ).lower()

    return any(kw in combined for kw in _FACTUAL_FLIP_INDICATORS)


def _majority_vote(
    results: List[StructuredMessage],
) -> StructuredMessage:
    """Return the message with the majority action and highest confidence."""
    action_counts = Counter(m.action for m in results)
    majority_action, vote_count = action_counts.most_common(1)[0]

    candidates = [m for m in results if m.action == majority_action]
    best = max(candidates, key=lambda m: m.confidence)

    n = len(results)
    if vote_count == n:
        best.confidence = min(best.confidence * 1.05, 0.99)
    elif vote_count < (n // 2 + 1):
        best.confidence = min(best.confidence, 0.60)

    best.metadata = getattr(best, "metadata", {}) or {}
    best.metadata["vote_count"] = vote_count
    best.metadata["total_samples"] = n
    return best


class SolverAgent(BaseAgent):
    def __init__(self, config_path: str = "config/model_config.yaml"):
        super().__init__(role="solver", config_path=config_path)
        self._r0_action: Optional[str] = None
        self._r0_msg: Optional[StructuredMessage] = None
        self._last_reasoning: Optional[str] = None

    def _build_prompt(self, question, round_id, critic_messages, benchmark) -> str:
        prompt = benchmark.build_solver_prompt(question, round_id, critic_messages)
        if benchmark.task_name == "gsm8k":
            prompt += _GSM_REASONING_RULES
        prompt += _CALIBRATION_SUFFIX
        return prompt

    def _single_call(self, prompt: str, parse_role: str, round_id: int):
        raw, tokens = self.call_llm(prompt)
        msg = parse_agent_response(raw, role=parse_role, round_id=round_id)
        return msg, tokens

    def _sample_with_consistency(
        self,
        prompt: str,
        parse_role: str,
        round_id: int,
        n: int = 3,
    ) -> tuple:
        """
        Sample n times, return (majority_msg, total_tokens).
        Used for GSM8K round 0 where arithmetic self-consistency helps most.
        """
        results = []
        total_tokens = 0
        for _ in range(n):
            msg, tokens = self._single_call(prompt, parse_role, round_id)
            results.append(msg)
            total_tokens += tokens
        best = _majority_vote(results)
        best.token_count = total_tokens
        return best, total_tokens

    def respond(
        self,
        question: str,
        round_id: int = 0,
        critic_messages: Optional[List[StructuredMessage]] = None,
        benchmark=None,
    ) -> StructuredMessage:

        prompt = self._build_prompt(question, round_id, critic_messages, benchmark)
        parse_role = benchmark.solver_role
        total_tokens = 0

        use_consistency = (
            round_id == 0
            and benchmark.task_name == "gsm8k"
        )

        if use_consistency:
            msg, total_tokens = self._sample_with_consistency(
                prompt, parse_role, round_id, n=3
            )
            logger.info(
                f"[Solver R0/GSM] self-consistency "
                f"vote={msg.metadata.get('vote_count')}/{msg.metadata.get('total_samples')} "
                f"action={msg.action}"
            )
        else:
            msg, total_tokens = self._single_call(prompt, parse_role, round_id)

        curr_reasoning = msg.get_reasoning_text().lower()
        from difflib import SequenceMatcher

        def is_similar(a, b, threshold=0.9):
            return SequenceMatcher(None, a, b).ratio() > threshold

        if round_id > 0 and self._last_reasoning is not None:
            if is_similar(curr_reasoning, self._last_reasoning):
                logger.warning("[Solver] Repeating same reasoning → force regenerate")
                retry_prompt = (
                    prompt
                    + f"\n\nYour previous answer was: {msg.action}\n"
                    f"The critic DISAGREED with your answer.\n"
                    f"You MUST analyze the critique carefully:\n"
                    f"- If the critic is correct → identify the wrong step and fix it\n"
                    f"- If the critic is WRONG → explain why and defend your answer\n"
                    f"- If unsure → recompute from scratch\n"
                    f"You MUST explicitly respond to the critic.\n"
                    # 🔥 THÊM DÒNG NÀY
                    f"You MUST use a COMPLETELY DIFFERENT reasoning path than before.\n"
                )
                raw, retry_tokens = self.call_llm(retry_prompt)
                total_tokens += retry_tokens
                msg = parse_agent_response(raw, role=parse_role, round_id=round_id)

        na_retries = 0
        while looks_like_non_answer(msg) and na_retries < 2:
            na_retries += 1
            logger.warning("[Solver] Non-answer/refusal → compact retry %d", na_retries)
            raw, rt = self.call_llm(
                build_compact_prompt(question, benchmark.answer_format)
            )
            total_tokens += rt
            msg = parse_agent_response(raw, role=parse_role, round_id=round_id)


        if "no reasoning" in msg.get_reasoning_text().lower():
            logger.warning("[Solver] Empty reasoning → regenerating")
            retry_prompt = (
                prompt
                + "\n\nYou MUST provide step-by-step reasoning. "
                "Do NOT say 'No reasoning provided'. "
                "Each step must contain real factual information."
            )
            raw, retry_tokens = self.call_llm(retry_prompt)
            total_tokens += retry_tokens
            msg = parse_agent_response(raw, role=parse_role, round_id=round_id)

            if "no reasoning" in msg.get_reasoning_text().lower():
                logger.error("[Solver] Still empty → injecting fallback reasoning")
                msg.reasoning = [
                    "Model failed to generate step-by-step reasoning",
                    "Fallback: answer based on prior factual knowledge only",
                    "Confidence is reduced due to missing explicit reasoning",
                ]
                msg.confidence = 0.30

        if msg.confidence > 0.90 and not has_specific_fact(msg):
            weak_signals = ["obviously", "clearly", "it is known", "common sense", "generally"]
            if any(w in msg.get_reasoning_text().lower() for w in weak_signals):
                logger.warning("[Solver] Overconfident + no specific fact + weak language → reducing confidence")
                msg.confidence = 0.65
            else:
                # High confidence but no specific fact → cap at 0.80
                msg.confidence = min(msg.confidence, 0.80)
                logger.info(f"[Solver] High confidence without specific fact → capped at {msg.confidence:.2f}")

        if round_id == 0:
            self._r0_action = msg.action
            self._r0_msg = msg
        elif self._r0_action is not None and msg.action != self._r0_action:
            has_basis = _flip_has_factual_basis(critic_messages or [])

            if has_basis:
                if is_uncertain(msg):
                    logger.warning("[Solver] Flip but uncertain → revert to R0")
                    msg.action = self._r0_action
                    msg.confidence = min(msg.confidence, 0.60)
                else:
                    logger.info(
                        f"[Solver R{round_id}] Flipped {self._r0_action} → {msg.action} (factual basis found)"
                    )
            else:
                logger.warning(
                    f"[Solver R{round_id}] Flipped {self._r0_action} → {msg.action} "
                    f"WITHOUT factual basis — possible pressure flip."
                )
                msg.action = self._r0_action
                msg.unfounded_flip = True
                msg.metadata = getattr(msg, "metadata", {}) or {}
                msg.metadata["unfounded_flip"] = True
                msg.r0_action = self._r0_action
                msg.confidence = min(msg.confidence, 0.60)
                logger.warning(f"[Solver R{round_id}] Reverted to R0 answer: {self._r0_action}")

        logger.info(f"[Solver R{round_id}] {msg.action} (conf={msg.confidence:.2f})")
        logger.info(f"[Solver reasoning] {msg.get_reasoning_text()}")
        self._last_reasoning = msg.get_reasoning_text().lower()
        msg.token_count = total_tokens
        return msg