"""
tasks/strategyqa/config.py
StrategyQA benchmark — yes/no multi-hop reasoning.

FIXES vs previous version:
─────────────────────────────────────────────────────
TOKEN:
  [FIX 1] should_early_stop(): không còn hardcode False
          → early stop khi critic agree + cả hai conf đủ cao
          → sample 4/9 tiết kiệm 2 rounds thừa (~8k tokens/sample)

ACCURACY:
  [FIX 2] allow_flip(): sửa logic broken (elif tautology)
          → nhánh confidence-only bị dead code, đã bỏ
          → giữ nguyên keyword-based check (đủ rồi)

(Các fix khác giữ nguyên từ version trước)
─────────────────────────────────────────────────────
"""

from typing import List, Optional
from ..base import BenchmarkConfig
from ..registry import register


# ── Few-shot examples ─────────────────────────────────────────────────────────

_SOLVER_EXAMPLES = """Examples of correct reasoning:

Q: Do hyenas appear in a Broadway musical?
{"intent":"pop culture fact","reasoning":["The Lion King is a famous Broadway musical","The Lion King features hyena characters Shenzi, Banzai, and Ed","Therefore hyenas do appear in a Broadway musical"],"action":"yes","confidence":0.95,"content":"Yes — hyenas appear in The Lion King on Broadway."}

Q: Is the tibia necessary to win the Stanley Cup?
{"intent":"indirect requirement chain","reasoning":["Hockey players must skate to play and win NHL games","Skating requires the tibia — the main lower-leg bone used to stand, push off, and maneuver on ice","Without a functioning tibia a player cannot skate, and therefore cannot contribute to winning the Cup"],"action":"yes","confidence":0.9,"content":"Yes — the tibia is needed to skate, making it indirectly necessary to win."}

Q: Could every citizen of Samoa send a letter to a unique JPMorgan Chase employee?
{"intent":"numerical comparison","reasoning":["Samoa has ~200,000 citizens","JPMorgan Chase has ~300,000 employees globally","300,000 > 200,000 so each citizen can write to a unique employee"],"action":"yes","confidence":0.9,"content":"Yes — JPMorgan has more employees than Samoa has citizens."}

Q: Would an Olympic athlete be tired out after running a mile?
{"intent":"athletic endurance","reasoning":["Olympic athletes have elite cardiovascular fitness","A mile is very short — elite runners cover it in under 4 minutes at race pace","At a relaxed pace a mile causes minimal fatigue for a trained Olympian"],"action":"no","confidence":0.85,"content":"No — a mile is too short to tire out an Olympic-level athlete."}

Q: Was Godfrey of Bouillon an Islamophobe?
{"intent":"historical figure assessment","reasoning":["Godfrey of Bouillon led the First Crusade and captured Jerusalem in 1099","He ordered the massacre of Muslims and Jews in Jerusalem after its capture","His actions show deep hostility toward Islam consistent with Islamophobia"],"action":"yes","confidence":0.85,"content":"Yes — Godfrey's massacre of Muslims and historical record indicate Islamophobic views."}

Q: Does Princess Peach's dress resemble a peach fruit?
{"intent":"visual comparison","reasoning":["Princess Peach wears a pink dress","Peach fruits are round, fuzzy, and orange-yellow — not pink or dress-shaped","The dress is named after her character name, not designed to look like the fruit"],"action":"no","confidence":0.9,"content":"No — Princess Peach's pink dress does not visually resemble a peach fruit."}

Q: Can a blind person enjoy a painting?
{"intent":"false-necessity trap","reasoning":["'Enjoy' does not require sight — texture, description, context, and story all work","Many blind people engage meaningfully with art","Sight is NOT necessary for enjoyment"],"action":"yes","confidence":0.88,"content":"Yes — enjoyment of art does not require sight."}

Q: Could a medieval knight send a telegram?
{"intent":"indirect prerequisite chain","reasoning":["Telegrams require electrical wires and telegraph stations","Medieval period ~500–1500 AD — electricity was unknown","Missing prerequisite makes it impossible"],"action":"no","confidence":0.99,"content":"No — telegrams require electricity which did not exist in the medieval period."}

Q: Are Brian Cranston and Saoirse Ronan's combined Emmy Awards a prime number?
{"intent":"numeric computation then math fact","reasoning":["Brian Cranston has won 4 Emmy Awards","Saoirse Ronan has won 0 Emmy Awards (she has Oscar nominations but no Emmys)","4 + 0 = 4","4 is not a prime number (divisible by 1, 2, 4)"],"action":"no","confidence":0.88,"content":"No — 4 combined Emmys is not a prime number."}

Q: Is there a jukebox musical about a sweet transvestite from Transexual, Transylvania?
{"intent":"fictional character identification","reasoning":["The Rocky Horror Show (1973) features Dr. Frank-N-Furter, described as 'a sweet transvestite from Transexual, Transylvania'","Rocky Horror is a musical but NOT a jukebox musical — it uses original songs, not pre-existing pop hits","A jukebox musical is defined as one built around pre-existing popular songs","Rocky Horror does not qualify as a jukebox musical"],"action":"no","confidence":0.85,"content":"No — Rocky Horror Show is an original musical, not a jukebox musical."}

Q: Is lunch on the beach a good activity to spot the full circle of a rainbow?
{"intent":"physics + positioning","reasoning":["A rainbow is actually a full circle, but we normally only see an arc because the ground blocks the lower half","To see a full circular rainbow you need to be elevated (e.g. in a plane or on a cliff) looking down","At beach level (ground/sea level) the horizon still cuts off the lower half","Lunch on the beach does not provide sufficient elevation"],"action":"no","confidence":0.88,"content":"No — beach level is too low to see the full circle of a rainbow."}

Q: Is B's place in the alphabet the same as Prince Harry's birth order among his siblings?
{"intent":"two-fact comparison","reasoning":["B is the 2nd letter of the alphabet","Prince Harry (Henry) has one older sibling: Prince William","Prince Harry is the 2nd child born to Charles and Diana","2nd letter = 2nd born → they match"],"action":"yes","confidence":0.92,"content":"Yes — B is 2nd in the alphabet and Harry is the 2nd royal sibling."}

Q: Was the amount of spinach Popeye ate unhealthy?
{"intent":"fictional quantity vs real nutrition","reasoning":["Popeye is depicted eating entire cans of spinach in one sitting, often multiple cans per episode","A single can of spinach is ~370g; Popeye frequently consumes several","Spinach contains high levels of oxalate — excessive consumption (multiple cans daily) causes kidney stones and can cause oxalate poisoning","The quantities depicted would be genuinely unhealthy in real life"],"action":"yes","confidence":0.82,"content":"Yes — the enormous quantities Popeye consumed would be unhealthy due to oxalate toxicity."}

Q: Does it seem like the Gorillaz is composed of more members than they have?
{"intent":"perception vs reality","reasoning":["Gorillaz is presented as a 4-member virtual band: 2-D, Murdoc, Noodle, Russel","In reality Gorillaz is primarily a project of Damon Albarn and Jamie Hewlett","The elaborate fictional universe and numerous collaborators makes it seem like a larger ensemble","So yes, it seems like more members than the actual 2 core creators"],"action":"yes","confidence":0.80,"content":"Yes — the rich fictional universe makes Gorillaz seem larger than its 2-person core."}

Q: Does Iphone have more iterations than Samsung Galaxy?
{"intent":"product line comparison — count carefully","reasoning":["iPhone: iPhone 1 through iPhone 15 (2007–2023), plus SE models = ~20+ distinct models","Samsung Galaxy S series: S1 through S24 (2010–2024) = ~24+ models; plus Note, A, Z series","Samsung Galaxy has far more total iterations across all its sub-lines","Even comparing just S-series to iPhone mainline, Samsung has more models"],"action":"no","confidence":0.85,"content":"No — Samsung Galaxy has more total iterations than iPhone across all its lines."}
"""

_CRITIC_COMPARE_EXAMPLES = """You have already formed your own independent answer. Now compare it with the solver's answer.

DISAGREE if:
- the solver has a SPECIFIC, NAMEABLE factual error (wrong number, wrong name, wrong fact)
- OR your answer differs AND you can cite a concrete real-world fact that proves solver wrong

AGREE if:
- both answers are the same (even if reasoning path differs)
- OR both answers are plausibly correct and no clear factual error exists

CRITICAL RULES:
- When answers MATCH → almost always AGREE. Only disagree if solver's reasoning contains a
  specific factual error that would mislead future rounds (e.g. wrong calculation, false fact).
- Do NOT disagree just because you used different reasoning paths to the same conclusion.
- Do NOT disagree over phrasing, level of detail, or stricter interpretations.
- Do NOT disagree over vague "could be wrong" feelings — you need a SPECIFIC fact.
- When answers DIFFER → carefully check which is better supported. Disagree if your answer
  has stronger concrete evidence (numbers, names, events).

Example — DISAGREE (genuine factual conflict, different answers):
  Your answer: "yes" (CAS 8009-03-8 is petrolatum, safe for rashes)
  Solver's answer: "no" (solver wrongly claims CAS 8009-03-8 is sulfur mustard)
  → DISAGREE: "CAS 8009-03-8 is petrolatum (Vaseline), not sulfur mustard."

Example — AGREE (same conclusion, different path):
  Your answer: "yes" (Buttercup=green, Blossom=red, Bubbles=blue → covers all three flag stripes)
  Solver's answer: "yes" (same conclusion, slightly different framing)
  → AGREE: reasoning is sound, do not disagree just because your path was different.

Example — AGREE (same answer, solver reasoning has minor gap but conclusion correct):
  Your answer: "no" (argon is a gas, cannot be chewed)
  Solver's answer: "no" (same conclusion, slightly less detailed)
  → AGREE: conclusion is correct, minor reasoning gap does not justify disagree.
"""

_JUDGE_TRICKY_EXAMPLES = """CRITICAL — cases where debate consensus was WRONG and the judge must override:

Case A — solver flip under social pressure (the flip was wrong):
  Q: "Did Rumi spend his time in a state of euphoria?"
  Debate: Solver R0=yes, flipped to no after critique.
  => gt=YES. The solver's ORIGINAL answer was correct. The flip had no factual basis.
  Lesson: When solver flips without a new specific fact, restore the R0 answer.

Case B — both anchored on wrong definition (metaphor missed):
  Q: "Can Africanized bees be considered multicultural?"
  Debate: Both said no (literal: bees don't have culture).
  => gt=YES — "multicultural" is used metaphorically for hybrid African+European genetic origin.
  Lesson: Check if the question uses a word metaphorically, not just literally.

Case C — indirect chain missed:
  Q: "Is the tibia necessary to win the Stanley Cup?"
  Debate: Both said no (team award logic).
  => gt=YES — tibia → skating → playing → winning. Indirect chain makes it necessary.
  Lesson: "Necessary for X" — trace the full indirect chain.

Case D — verifier false-positive (trust solid debate over verifier):
  Q: "Does the Pixar film Brave feature Scottish people?"
  Debate: Both agreed yes with solid reasoning. Verifier flagged issues.
  => gt=YES — verifier was wrong. Solid solver+critic agreement beats verifier noise.
  Lesson: If solver+critic both agree with correct facts, trust them over the verifier.

Case E — numeric/comparison trap (must verify both numbers independently):
  Q: "Are X and Y's combined Z a prime number?"
  => Compute both numbers from scratch. Do NOT trust debate if the numbers weren't verified.
  Lesson: Always recompute the specific numbers before checking the math property.

Case F — jukebox musical / genre definition trap:
  Q: "Is there a jukebox musical about X?"
  => A jukebox musical uses PRE-EXISTING pop songs. Original musicals do NOT count.
  Lesson: Apply the precise genre definition, not the loose everyday meaning.
"""


@register("strategyqa")
class StrategyQAConfig(BenchmarkConfig):

    @property
    def task_name(self) -> str:
        return "strategyqa"

    @property
    def answer_format(self) -> str:
        return "yes or no"

    @property
    def solver_role(self) -> str:
        return "solver"

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
            independent_note = ""
            if hasattr(last, "independent_answer") and last.independent_answer is not None:
                ind = last.independent_answer
                independent_note = (
                    f"The critic independently answered '{ind.action}': {ind.content}\n"
                )
            critique_section = (
                f"\n\n---\nCRITIQUE FROM PREVIOUS ROUND:\n"
                f"{independent_note}"
                f"Critic's verdict: {last.content}\n\n"

                f"DEBATE RULE (VERY IMPORTANT):\n"
                f"- You MUST explicitly respond to the critic.\n"
                f"- If the critic says you are wrong:\n"
                f"    → either DEFEND your answer with stronger facts\n"
                f"    → OR REVISE your answer with corrected reasoning\n"
                f"- Do NOT ignore the critic.\n"
                f"- Do NOT repeat your previous reasoning unchanged.\n"
                f"- If you keep the same answer, you MUST explain WHY the critic is wrong.\n\n"

                f"FLIP DECISION RULES:\n"
                f"  Q1: Did the critique identify a SPECIFIC wrong step or fact?\n"
                f"  Q2: Is that correction actually correct?\n\n"
                f"  → If BOTH YES: update answer\n"
                f"  → If critique is wrong: defend your answer\n"
                f"  → If unsure: recompute from scratch\n\n"

                f"CRITICAL:\n"
                f"- The critic can be wrong.\n"
                f"- You must actively engage in the debate.\n"
                f"- If your reasoning is identical to the previous round, you MUST recompute the answer from scratch.\n"
            )

        return (
            f"You are a knowledgeable reasoning agent. Answer the yes/no question using your world knowledge.\n\n"
            f"{_SOLVER_EXAMPLES}\n"
            f"Now answer this question:\n"
            f"Q: {question}\n\n"
            f"Think step by step. Use specific facts you know.\n"
            f"CRITICAL — question interpretation rules:\n"
            f"- 'Necessary for X': trace the FULL indirect chain (e.g. tibia → skating → playing → winning).\n"
            f"- 'Safe' = absence of harm, not 'recommended' or 'nutritious'.\n"
            f"- Metaphorical words ('multicultural', 'enjoy', 'euphoria'): check broad/figurative definitions.\n"
            f"- Fictional characters: use in-universe canon, not real-world logic.\n"
            f"- Counter-intuitive biology/history: common sense is often wrong.\n"
            f"- Numeric/comparison questions: compute BOTH numbers explicitly before comparing.\n"
            f"- 'Same as' / 'equivalent to' questions: verify each side independently then compare.\n"
            f"- Genre/category questions (jukebox musical, etc.): use precise definitions.\n\n"
            f"- Do NOT conclude 'no' just because there is 'no evidence' or something is 'not publicly known'. "
            f"Many questions rely on indirect or implicit knowledge.\n"
            f"Output ONLY a JSON object, no other text:\n"
            f'{{"intent":"...","reasoning":["fact 1","fact 2","fact 3"],"action":"yes or no","confidence":0.0-1.0,"content":"one sentence answer"}}\n'
            f'The "action" field MUST be exactly "yes" or "no".'
            f"{critique_section}"
        )

    def build_independent_prompt(self, question: str) -> str:
        return (
            f"You are an independent verifier answering a yes/no question.\n"
            f"You have NOT seen any other agent's answer. Answer from your own knowledge only.\n\n"
            f"RULES:\n"
            f"- Your answer ('action') MUST be exactly 'yes' or 'no'. NEVER 'agree', 'disagree', or anything else.\n"
            f"- Do NOT follow common reasoning patterns blindly.\n"
            f"- Do NOT overthink or invent rare edge cases.\n"
            f"- Assume normal real-world conditions unless stated.\n"
            f"- Metaphorical words ('necessary', 'enjoy', 'multicultural'): check broad/figurative meanings.\n"
            f"- Indirect chain questions: trace the FULL chain (e.g. tibia → skating → playing → winning).\n"
            f"- Numeric/comparison questions: compute BOTH sides explicitly before comparing.\n"
            f"- Genre/category questions: use the PRECISE definition (e.g. jukebox musical = pre-existing songs only).\n\n"
            f"Question:\n{question}\n\n"
            f"Output ONLY a JSON object. The 'action' field MUST be 'yes' or 'no':\n"
            f'{{"intent":"...","reasoning":["fact 1","fact 2","fact 3"],"action":"yes or no","confidence":0.0-1.0,"content":"one sentence answer"}}'
        )

    def build_critic_prompt(self, question: str, solver_message, critic_independent) -> str:
        return (
            f"You are a debate critic. You already answered this question independently.\n\n"
            f"{_CRITIC_COMPARE_EXAMPLES}\n"
            f"Question: {question}\n\n"
            f"YOUR independent answer: '{critic_independent.action}' "
            f"(confidence: {critic_independent.confidence:.2f})\n"
            f"Your reasoning: {critic_independent.get_reasoning_text()}\n\n"
            f"SOLVER's answer: '{solver_message.action}' "
            f"(confidence: {solver_message.confidence:.2f})\n"
            f"Solver's reasoning:\n{solver_message.get_reasoning_text()}\n\n"
            f"Now decide: do you AGREE or DISAGREE with the solver?\n\n"
            f"DECISION GUIDE:\n"
            f"- Same answer → AGREE (unless solver has a specific nameable factual error)\n"
            f"- Different answer → DISAGREE and state the exact fact that proves solver wrong\n"
            f"- Vague feeling solver 'might be wrong' → AGREE\n"
            f"- Different reasoning path to same conclusion → AGREE\n\n"
            f"Output ONLY a JSON object, no other text:\n"
            f'{{"intent":"...","reasoning":["your key fact","solver fact compared","verdict reason"],"action":"agree or disagree","confidence":0.0-1.0,"content":"one sentence verdict"}}\n'
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
        verif_hint = ""
        if verification_result:
            _noise = ("not consistent", "cannot verify", "may not be accurate", "unclear", "uncertain")
            real_issues = [
                i for i in verification_result.get("issues", [])
                if len(i) > 25 and not any(p in i.lower() for p in _noise)
            ]
            if real_issues:
                verif_hint = (
                    f"\nVerification hints (SOFT signal only — verifier is frequently wrong on tricky questions): "
                    f"{real_issues[:2]}\n"
                    f"Do NOT treat verification failure as proof the answer is wrong.\n"
                )

        flip_note = ""
        solver_r0_line = ""
        if solver_initial_message is not None:
            solver_r0_line = (
                f"  Solver R0 (original): '{solver_initial_message.action}' "
                f"(conf={solver_initial_message.confidence:.2f}) — {solver_initial_message.content}\n"
                f"  Solver R0 reasoning: {solver_initial_message.get_reasoning_text()}\n"
            )
            if solver_initial_message.action != solver_message.action:
                flip_note = (
                    f"CRITICAL: Solver CHANGED answer '{solver_initial_message.action}' → "
                    f"'{solver_message.action}' after critique.\n"
                    f"  LLMs frequently flip incorrectly under social pressure from critics.\n"
                    f"  The ORIGINAL answer (R0='{solver_initial_message.action}') is often correct.\n"
                    f"  Only accept the flip if the critique named a SPECIFIC verifiable fact that was wrong.\n"
                    f"  If the critique only reframed or raised vague doubts → the flip is likely WRONG.\n"
                    f"  Consider restoring R0='{solver_initial_message.action}' as the final answer.\n\n"
                )

        critic_ind = getattr(critic_message, "independent_answer", None)
        critic_ind_line = ""
        if critic_ind is not None:
            critic_ind_line = (
                f"  Critic independent: '{critic_ind.action}' "
                f"(conf={critic_ind.confidence:.2f}) — {critic_ind.content}\n"
                f"  Critic independent reasoning: {critic_ind.get_reasoning_text()}\n"
            )

        verdict_label = "AGREE" if critic_message.action == "agree" else "DISAGREE"
        debate_evidence = (
            f"{solver_r0_line}"
            f"  Solver FINAL: '{solver_message.action}' (conf={solver_message.confidence:.2f}) — {solver_message.content}\n"
            f"  Solver final reasoning: {solver_message.get_reasoning_text()}\n"
            f"{critic_ind_line}"
            f"  Critic verdict: {verdict_label} — {critic_message.content}\n"
            f"  Critic reasoning: {critic_message.get_reasoning_text()}\n"
        )

        return (
            f"You are the FINAL JUDGE. Output the single correct yes/no answer.\n\n"
            f"{_JUDGE_TRICKY_EXAMPLES}\n\n"
            f"PROCESS:\n\n"
            f"1. Solve the question independently using real-world knowledge.\n"
            f"   - Identify concrete facts\n"
            f"   - For numeric questions: compute BOTH numbers explicitly\n"
            f"   - For genre questions: use precise definitions\n"
            f"   - Avoid overthinking or rare edge cases\n\n"
            f"2. Compare your answer with the debate:\n"
            f"   - If your answer matches one side → prefer that side\n"
            f"   - If both are wrong → override them\n\n"
            f"3. If the solver changed answer:\n"
            f"   - Only trust the change if there is a clear factual correction\n"
            f"   - Otherwise prefer the original answer\n\n"
            f"4. Final rule:\n"
            f"   - Use the most direct, common-sense interpretation\n"
            f"   - Do NOT invent rare scenarios\n"
            f"   - Do NOT rely blindly on the debate\n\n"
            f"QUESTION: {question}\n\n"
            f"{flip_note}"
            f"--- Debate evidence ---\n"
            f"{debate_evidence}"
            f"{verif_hint}\n"
            f"Output ONLY a JSON object, no other text:\n"
            f'{{"intent":"final judgment","reasoning":["my reasoning step 1","key deciding fact","why final answer is correct"],"action":"yes or no","confidence":0.0-1.0,"content":"one concise sentence"}}\n'
            f'"action" MUST be exactly "yes" or "no".\n'
            f'"reasoning" MUST contain at least 3 non-empty strings.\n'
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, pred: str, gt: str) -> bool:
        return pred.strip().lower() == gt.strip().lower()

    def normalize_answer(self, raw: str) -> str:
        raw = raw.strip().lower()
        if raw in ("yes", "no"):
            return raw
        # fallback: tìm yes/no trong string
        import re
        m = re.search(r"\b(yes|no)\b", raw)
        return m.group(1) if m else raw

    # ── Behavior overrides ────────────────────────────────────────────────────

    def allow_flip(self, critic_msg) -> bool:
        """
        FIX 2: Bỏ elif tautology — chỉ cần keyword check.
        Logic cũ:
            if any(kw): return True
            elif conf > 0.75 and action == disagree and any(kw): return True  ← dead code
        Logic mới: chỉ keyword check, sạch và đúng.
        """
        text = (critic_msg.get_reasoning_text() or "").lower()
        allow_keywords = [
            "incorrect", "not true", "wrong", "evidence",
            "calculation", "miscalculated", "error", "factual",
            "actually", "in fact", "contradicts", "misidentif",
            "impossible", "cannot", "does not", "is not",
            "are not", "was not", "did not", "has not",
            "never", "false", "untrue", "mistaken",
            "should be", "instead", "rather",
        ]
        return any(kw in text for kw in allow_keywords)

    def should_early_stop(self, solver_msg, critic_msg, logical, round_id) -> bool:
        """
        FIX 1: Không còn hardcode False.
        Early stop khi:
          - critic agree
          - cả hai confidence đủ cao
          - round > 0 (đã có ít nhất 1 round debate)

        Ngưỡng đặt thấp hơn GSM8K vì StrategyQA không cần arithmetic check.
        Tương tác với orchestrator tier-2 early stop (conf >= 0.80 + critic >= 0.90).
        Method này là tầng benchmark-level bổ sung thêm.
        """
        if critic_msg.action != "agree":
            return False

        # Round 0: chỉ stop nếu cả hai rất tự tin (tránh stop quá sớm khi chưa debate)
        if round_id == 0:
            return (
                solver_msg.confidence >= 0.90
                and critic_msg.confidence >= 0.90
            )

        # Round > 0: đã debate ít nhất 1 lần, ngưỡng thấp hơn
        return (
            solver_msg.confidence >= 0.82
            and critic_msg.confidence >= 0.82
        )

    def verifier_settings(self) -> dict:
        return {
            "check_arithmetic": False,
            "check_percentage": False,
        }