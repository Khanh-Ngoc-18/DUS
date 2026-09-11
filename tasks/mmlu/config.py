"""
tasks/mmlu/config.py
MMLU benchmark — multiple-choice questions across 57 subjects.

Format: 4-choice MCQ (A/B/C/D), answer is a single letter.
"""

from typing import List, Optional
from ..base import BenchmarkConfig
from ..registry import register


_MMLU_SUBJECTS_NOTE = (
    "MMLU covers 57 subjects including: mathematics, physics, chemistry, biology, "
    "history, law, medicine, economics, computer science, philosophy, psychology, and more."
)

_MMLU_SOLVER_EXAMPLES = """Examples:

Q: What is the powerhouse of the cell?
A) Nucleus  B) Mitochondria  C) Ribosome  D) Golgi apparatus
{"intent":"biology","reasoning":["Mitochondria produce ATP via oxidative phosphorylation","They are responsible for cellular energy production"],"action":"B","confidence":0.99,"content":"Mitochondria — the powerhouse of the cell"}

Q: Which of the following best describes Keynesian economics?
A) Markets self-correct without intervention  B) Government spending can stimulate demand  C) Money supply controls inflation only  D) Free trade maximizes welfare
{"intent":"economics","reasoning":["Keynes argued aggregate demand drives output","Government fiscal policy can fill output gaps during recession"],"action":"B","confidence":0.97,"content":"Keynesian economics emphasizes government spending to stimulate demand"}

Q: In Python, what does 'list comprehension' create?
A) A new class  B) A generator  C) A new list  D) A dictionary
{"intent":"computer_science","reasoning":["List comprehensions evaluate to a list object","Syntax: [expr for item in iterable]"],"action":"C","confidence":0.99,"content":"List comprehension creates a new list"}
"""

_MMLU_CRITIC_EXAMPLES = """You are checking factual accuracy of a multiple-choice answer.

DISAGREE if:
- The chosen answer letter is factually wrong based on your independent solution
- The reasoning contains a clear factual error or misconception
- A better answer choice exists that the solver missed

AGREE only if:
- The same letter matches your answer
- AND the reasoning is factually sound

Do NOT disagree over:
- Stylistic differences in explanation
- Extra detail that doesn't change the answer
"""


@register("mmlu")
class MMLUConfig(BenchmarkConfig):

    @property
    def task_name(self) -> str:
        return "mmlu"

    @property
    def answer_format(self) -> str:
        return "letter"  # A/B/C/D

    @property
    def solver_role(self) -> str:
        return "solver_mmlu"

    # ── Prompts ───────────────────────────────────────────────────────────────

    def build_solver_prompt(
        self,
        question: str,
        round_id: int,
        critic_messages=None,
    ) -> str:
        critique_section = ""
        if critic_messages:
            last = critic_messages[-1]
            critique_section = (
                f"\n\nCritic found: {last.content}\n"
                f"Only change your answer if there is a clear factual error identified."
            )
        return (
            f"Answer this multiple-choice question. Think step by step.\n\n"
            f"{_MMLU_SUBJECTS_NOTE}\n\n"
            f"{_MMLU_SOLVER_EXAMPLES}\n"
            f"Q: {question}\n\n"
            f"Rules:\n"
            f"- 'action' MUST be exactly one letter: A, B, C, or D\n"
            f"- Provide reasoning steps that justify your chosen answer\n"
            f"- Set confidence 0.99 only if fully certain; lower if uncertain\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"<subject>","reasoning":["reason 1","reason 2"],"action":"<A|B|C|D>","confidence":0.95,"content":"one sentence"}}\n'
            f'The "action" field MUST be exactly one of: A, B, C, D.'
            f"{critique_section}"
        )

    def build_independent_prompt(self, question: str) -> str:
        return (
            f"Answer this multiple-choice question. Think step by step.\n\n"
            f"{_MMLU_SUBJECTS_NOTE}\n\n"
            f"{_MMLU_SOLVER_EXAMPLES}\n"
            f"Q: {question}\n\n"
            f"Rules:\n"
            f"- 'action' MUST be exactly one letter: A, B, C, or D\n"
            f"- Provide reasoning steps that justify your chosen answer\n"
            f"- Set confidence 0.99 only if fully certain\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"<subject>","reasoning":["reason 1","reason 2"],"action":"<A|B|C|D>","confidence":0.95,"content":"one sentence"}}\n'
            f'The "action" field MUST be exactly one of: A, B, C, D.'
        )

    def build_critic_prompt(self, question: str, solver_message, critic_independent) -> str:
        answer_match = critic_independent.action.strip().upper() == solver_message.action.strip().upper()
        match_note = (
            "NOTE: Both chose the same letter. Verify reasoning is factually correct."
            if answer_match
            else f"NOTE: Answers differ — yours={critic_independent.action}, solver={solver_message.action}. Determine which is correct."
        )
        return (
            f"You are checking a multiple-choice answer for factual accuracy.\n\n"
            f"{_MMLU_CRITIC_EXAMPLES}\n"
            f"Question: {question}\n\n"
            f"YOUR answer: {critic_independent.action}\n"
            f"Your reasoning: {critic_independent.get_reasoning_text()}\n\n"
            f"SOLVER's answer: {solver_message.action}\n"
            f"Solver reasoning: {solver_message.get_reasoning_text()}\n\n"
            f"{match_note}\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"fact check","reasoning":["what I verified","verdict"],"action":"agree or disagree","confidence":0.0-1.0,"content":"one sentence — if disagree, state correct answer and why"}}\n'
            f'The "action" field MUST be exactly "agree" or "disagree".'
        )

    def build_judge_prompt(
        self,
        question: str,
        solver_message,
        critic_message,
        verification_result=None,
        solver_initial_message=None,
    ) -> str:
        flip_note = ""
        if solver_initial_message and solver_initial_message.action.strip().upper() != solver_message.action.strip().upper():
            flip_note = (
                f"WARNING: Solver originally answered {solver_initial_message.action}, "
                f"then changed to {solver_message.action}.\n"
                f"Verify BOTH independently before deciding.\n\n"
            )
        return (
            f"You are the final judge for a multiple-choice question.\n\n"
            f"CRITICAL: Answer the question yourself first. "
            f"If the debate is wrong, your independent answer takes priority.\n\n"
            f"PROCESS:\n"
            f"1. Answer the question yourself — write your reasoning\n"
            f"2. Compare with the debate\n"
            f"3. If debate matches → confirm it\n"
            f"4. If debate is wrong → trust your own reasoning\n\n"
            f"{_MMLU_SUBJECTS_NOTE}\n\n"
            f"Question: {question}\n\n"
            f"{flip_note}"
            f"Debate reference:\n"
            f"  Solver: {solver_message.action} — {solver_message.get_reasoning_text()}\n"
            f"  Critic: {critic_message.content}\n\n"
            f"Answer yourself first, then output ONLY a JSON object:\n"
            f'{{"intent":"final judgment","reasoning":["my reasoning step 1","my reasoning step 2","cross-check: matches/differs"],"action":"<A|B|C|D>","confidence":0.0-1.0,"content":"one sentence"}}\n'
            f'"action" MUST be exactly one of: A, B, C, D.'
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, pred: str, gt: str) -> bool:
        return pred.strip().upper() == gt.strip().upper()

    def normalize_answer(self, raw: str) -> str:
        import re
        # Extract first A/B/C/D found
        m = re.search(r"\b([A-D])\b", raw.strip().upper())
        return m.group(1) if m else raw.strip().upper()

    # ── Behavior overrides ────────────────────────────────────────────────────

    def allow_flip(self, critic_msg) -> bool:
        """MMLU: only flip on clear factual counter-evidence."""
        text = (critic_msg.get_reasoning_text() or "").lower()
        factual_keywords = [
            "incorrect", "wrong", "actually", "in fact", "factually",
            "correct answer is", "should be", "is not", "does not",
            "contradicts", "misidentif", "evidence",
        ]
        return any(kw in text for kw in factual_keywords)

    def should_early_stop(self, solver_msg, critic_msg, logical, round_id) -> bool:
        return (
            critic_msg.action == "agree"
            and logical.get("valid", False)
            and solver_msg.confidence >= 0.99
        )

    def verifier_settings(self) -> dict:
        return {
            "check_arithmetic": False,
            "check_percentage": False,
        }