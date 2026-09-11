"""
tasks/gsm8k/config.py
GSM8K benchmark — arithmetic word problems.

Changes vs original:
- Added negative number and fraction examples to solver/independent prompts
- Judge prompt: explicit instruction to override debate if both are wrong
- allow_flip: unchanged (arithmetic-error only)
- should_early_stop: unchanged
"""

from typing import List, Optional
from ..base import BenchmarkConfig
from ..registry import register


_GSM_SOLVER_EXAMPLES = """Examples:

Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest at $2/egg. How much per day?
{"intent":"arithmetic","reasoning":["16-3-4=9 eggs remaining","9×2=18 dollars"],"action":"18","confidence":0.99,"content":"$18 per day"}

Q: A store has 10 apples. They sell 3, get 5 more, then sell 2. How many remain?
{"intent":"arithmetic","reasoning":["10-3=7","7+5=12","12-2=10"],"action":"10","confidence":0.99,"content":"10 apples remain"}

Q: John runs 3 miles at 6 mph, then 2 miles at 4 mph. How many minutes does he run total?
{"intent":"rate","reasoning":["3 miles at 6 mph = 3/6 = 0.5 hours = 30 minutes","2 miles at 4 mph = 2/4 = 0.5 hours = 30 minutes","30+30=60 minutes total"],"action":"60","confidence":0.99,"content":"60 minutes total"}

Q: A shirt costs $40. It is discounted 25%, then taxed 10%. What is the final price?
{"intent":"percentage","reasoning":["25% discount: 40×0.75=30","10% tax: 30×1.10=33"],"action":"33","confidence":0.99,"content":"$33 final price"}

Q: There are 5 boxes. Each box has 3 bags. Each bag has 4 apples. How many apples total?
{"intent":"multiplication","reasoning":["5×3=15 bags","15×4=60 apples"],"action":"60","confidence":0.99,"content":"60 apples total"}

Q: Tom is 3 times older than his son. In 10 years Tom will be twice as old. How old is Tom now?
{"intent":"algebra","reasoning":["Let son=x, Tom=3x","In 10 years: 3x+10=2(x+10)","3x+10=2x+20 → x=10","Tom=3×10=30"],"action":"30","confidence":0.99,"content":"Tom is 30 years old"}

Q: A recipe needs 2.5 cups of flour for 12 cookies. How many cups for 30 cookies?
{"intent":"ratio","reasoning":["ratio: 2.5/12 cups per cookie","30 cookies: 30×(2.5/12)=30×0.2083=6.25"],"action":"6.25","confidence":0.99,"content":"6.25 cups of flour"}

Q: A shop loses $15 per day for 4 days, then gains $8 per day for 3 days. What is the net result?
{"intent":"arithmetic","reasoning":["loss: -15×4=-60","gain: 8×3=24","net: -60+24=-36"],"action":"-36","confidence":0.99,"content":"Net loss of $36"}

Q: If 3/4 of a tank holds 90 liters, how many liters is the full tank?
{"intent":"ratio","reasoning":["3/4 → 90L","1/4 → 90÷3=30L","full tank: 4×30=120L"],"action":"120","confidence":0.99,"content":"120 liters"}

Q: A worker earns $200/day. After a 150% raise, what is the new daily wage?
{"intent":"percentage","reasoning":["150% raise means ×(1+1.5)=×2.5","200×2.5=500"],"action":"500","confidence":0.99,"content":"$500 per day"}
"""

_GSM_CRITIC_EXAMPLES = """You are checking arithmetic step by step.

DISAGREE if:
- Any calculation step has a numeric error (e.g. 3×4=11 is wrong)
- A required step is missing (e.g. forgot to apply tax, forgot to subtract)
- Wrong formula used (e.g. used addition instead of multiplication for area)
- Sign error (e.g. loss treated as gain)

AGREE only if:
- Final answer matches yours
- AND all calculation steps you can verify are correct

Do NOT disagree over:
- Different but equivalent orderings of steps
- Different phrasing of the same arithmetic
- Rounding differences less than 0.01

When you DISAGREE, point to the exact wrong step with the correct value.
"""


@register("gsm8k")
class GSM8KConfig(BenchmarkConfig):

    @property
    def task_name(self) -> str:
        return "gsm8k"

    @property
    def answer_format(self) -> str:
        return "number"

    @property
    def solver_role(self) -> str:
        return "solver_gsm"

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
                f"Only change your answer if there is a clear arithmetic error identified."
            )
        return (
            f"THE PROBLEM TO SOLVE IS:\n{question}\n\n"
            f"Solve the problem above step by step. Show every calculation explicitly.\n"
            f"The examples below only show the required format — do NOT solve the examples.\n\n"
            f"{_GSM_SOLVER_EXAMPLES}\n"
            f"Now solve ONLY this problem: {question}\n\n"
            f"Rules:\n"
            f"- Write EVERY arithmetic step, even simple ones (e.g. '12×3=36')\n"
            f"- Do NOT skip steps\n"
            f"- Watch for negative numbers — losses and deficits produce negative results\n"
            f"- 'action' must be the final number only, no units, no text (e.g. '42' not '$42')\n"
            f"- Set confidence 0.99 only if fully certain of every step; lower if any step is approximate\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"arithmetic","reasoning":["step 1: X=...","step 2: Y=..."],"action":"<final number only>","confidence":0.95,"content":"one sentence"}}\n'
            f'The "action" field MUST be the final numeric answer only (e.g. "42" or "3.5" or "-36").'
            f"{critique_section}"
        )

    def build_independent_prompt(self, question: str) -> str:
        return (
            f"THE PROBLEM TO SOLVE IS:\n{question}\n\n"
            f"Solve the problem above step by step. Show every calculation explicitly.\n"
            f"The examples below only show the required format — do NOT solve the examples.\n\n"
            f"{_GSM_SOLVER_EXAMPLES}\n"
            f"Now solve ONLY this problem: {question}\n\n"
            f"Rules:\n"
            f"- Write EVERY arithmetic step, even simple ones (e.g. '12×3=36')\n"
            f"- Do NOT skip steps\n"
            f"- Watch for negative numbers — losses and deficits produce negative results\n"
            f"- 'action' must be the final number only, no units (e.g. '42' not '$42')\n"
            f"- Set confidence 0.99 only if fully certain of every step\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"arithmetic","reasoning":["step 1: X=...","step 2: Y=..."],"action":"<final number only>","confidence":0.95,"content":"one sentence"}}\n'
            f'The "action" field MUST be the final numeric answer only (e.g. "42" or "-36").'
        )

    def build_critic_prompt(self, question: str, solver_message, critic_independent) -> str:
        answer_match = critic_independent.action.strip() == solver_message.action.strip()
        match_note = (
            "NOTE: Both answers are the same number. Verify every step is arithmetically correct."
            if answer_match
            else f"NOTE: Answers differ — yours={critic_independent.action}, solver={solver_message.action}. Find the wrong step."
        )
        return (
            f"You are checking math step by step.\n\n"
            f"{_GSM_CRITIC_EXAMPLES}\n"
            f"Question: {question}\n\n"
            f"YOUR solution: {critic_independent.action}\n"
            f"Your steps: {critic_independent.get_reasoning_text()}\n\n"
            f"SOLVER's solution: {solver_message.action}\n"
            f"Solver steps: {solver_message.get_reasoning_text()}\n\n"
            f"{match_note}\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"math check","reasoning":["step I checked","what I found","verdict"],"action":"agree or disagree","confidence":0.0-1.0,"content":"one sentence — if disagree state the wrong step"}}\n'
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
        if solver_initial_message and solver_initial_message.action != solver_message.action:
            flip_note = (
                f"WARNING: Solver originally answered {solver_initial_message.action}, "
                f"then changed to {solver_message.action}.\n"
                f"Solver R0 steps: {solver_initial_message.get_reasoning_text()}\n"
                f"Verify BOTH independently — do not trust the flip without checking.\n\n"
            )
        return (
            f"You are the final judge for a math problem.\n\n"
            f"CRITICAL: Solve the problem yourself first. "
            f"If the debate gives a wrong answer, your independent calculation takes priority. "
            f"Debate consensus does NOT override arithmetic truth.\n\n"
            f"PROCESS:\n"
            f"1. Solve the problem yourself from scratch — write every step\n"
            f"2. Compare your answer with the debate\n"
            f"3. If debate matches your answer → confirm it\n"
            f"4. If debate is wrong → trust your own arithmetic\n\n"
            f"{_GSM_SOLVER_EXAMPLES}\n"
            f"Question: {question}\n\n"
            f"{flip_note}"
            f"Debate reference:\n"
            f"  Solver: {solver_message.action} — {solver_message.get_reasoning_text()}\n"
            f"  Critic: {critic_message.content}\n\n"
            f"Solve it yourself first, then output ONLY a JSON object:\n"
            f'{{"intent":"final judgment","reasoning":["my step 1: X=...","my step 2: Y=...","cross-check: matches/differs because..."],"action":"<final number>","confidence":0.0-1.0,"content":"one sentence"}}\n'
            f'"action" MUST be the final numeric answer only. No units, no text.'
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, pred: str, gt: str) -> bool:
        try:
            return abs(float(pred.strip()) - float(gt.strip())) < 1e-3
        except (ValueError, TypeError):
            return pred.strip() == gt.strip()

    def normalize_answer(self, raw: str) -> str:
        import re
        m = re.search(r"-?\d+\.?\d*", raw)
        return m.group(0) if m else raw.strip()

    # ── Behavior overrides ────────────────────────────────────────────────────

    def allow_flip(self, critic_msg) -> bool:
        """GSM8K: only flip on specific arithmetic error."""
        text = (critic_msg.get_reasoning_text() or "").lower()
        arithmetic_keywords = [
            "calculation", "miscalculated", "arithmetic", "wrong step",
            "incorrect", "error", "should be", "not equal", "wrong value",
        ]
        return any(kw in text for kw in arithmetic_keywords)

    def should_early_stop(self, solver_msg, critic_msg, logical, round_id) -> bool:
        """GSM8K: early stop when agree + very high confidence."""
        return (
            critic_msg.action == "agree"
            and logical.get("valid", False)
            and solver_msg.confidence >= 0.99
        )

    def verifier_settings(self) -> dict:
        return {
            "check_arithmetic": True,
            "check_percentage": True,
        }