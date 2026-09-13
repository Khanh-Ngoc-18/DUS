# **When Does Adaptive Stopping Pay in Multi-Agent Debate? A Circularity Pitfall, Critic-Derived Signals, and a Fixed-Depth Baseline That Weakly Dominates**

**Abstract** — Multi-Agent Debate (MAD) improves LLM reliability at a cost that scales with agents × rounds, motivating early stopping once a debate's outcome is settled. We show that the standard way of reporting such savings cannot distinguish a rule that removes rounds from one that does not: a score built from four consensus-dynamics signals reports a 33.9% saving over 4,500 debates, yet simulated sequentially against the consensus rule the system already runs, it removes 0.0% of executed rounds — its dominant signal, answer entropy, is zero exactly where consensus already halts. We give a corrected evaluation protocol, of which only sequential simulation against the incumbent proves diagnostic, and apply it to a signal that does survive: seven features read from an independent critic, containing no agreement term, reach ROC-AUC 0.729 against 0.705 for the consensus signals (Δ = +0.025, 95% CI [+0.007, +0.042]). Deployed as a two-stage cascade — a round-0 critic self-consistency check followed by the same score rechecked every round — this cuts token cost against consensus by 39.4% on MMLU at no measurable accuracy loss. Benchmarked against twelve no-debate baselines under the same protocol, the cascade is statistically indistinguishable from unconditional two-round stopping on all three benchmarks, at 8–30% higher cost than that simpler policy, yet it beats six of twelve self-consistency baselines by up to 25 points and carries the lowest worst-case regret (3.7 vs. 4.2 for fixed depth, 15.9 for the best single model). Adaptive stopping buys robustness across benchmarks rather than accuracy on any single one — and fixed depth supplies most of that robustness more cheaply. We give three measurable conditions under which adaptive stopping pays off, and release all debate logs, both protocols, and the analysis pipeline.

**Keywords** — multi-agent debate, large language models, **Debate Uncertainty**, **Critic Self-Consistency**, Adaptive Inference, Critic-Derived Signals, Computational Efficiency, Baseline Evaluation 

---

## **I. INTRODUCTION**

Multi-Agent Debate (MAD) improves the factuality and reasoning reliability of large language models by having several agents answer independently, critique one another over several rounds, and converge on a shared answer \[1\], \[2\], \[3\]. The premise is that idiosyncratic errors cancel across agents while sound reasoning survives scrutiny.

The gains carry a computational cost that scales with the product of agent count and round count: each round here issues four model calls — two solvers and two critic verdicts — plus one at round 0 for the critic's independent answer, so running a question to six rounds costs several times what answering it once does. Two facts measured on our logs show the opportunity is real. Among rounds that are not the last of their debate, **90.6%** commit the same verdict the debate eventually reaches. And debate does not reliably improve anything: over 4,500 debates it corrupts almost as many answers as it rescues, for a net change indistinguishable from zero on every benchmark (TABLE IX). Where a debate stops is therefore an accuracy decision as much as a budget one.

This has motivated proposals that monitor the transcript and terminate once an uncertainty or stability criterion is met \[7\]. What they report as a saving is the reduction against running the protocol to full depth, or the fraction of rounds marked stoppable — neither of which a deployed system would save, because debate systems overwhelmingly halt on consensus already. A new rule removes only the rounds it removes *before* the incumbent would have halted, so one firing at or after the consensus round can mark many rounds and save none. An early-stopping method must therefore be shown to stop earlier than the mechanism already in place; in the closest prior evaluation the reference point is a fixed depth instead (Section II).

Two questions follow, in order. The first is how a debate stopping rule should be evaluated so that the saving it reports is the saving actually obtained against the consensus rule the system already applies. Only once that is settled does the second become meaningful: whether any signal remains that permits earlier termination without sacrificing accuracy, and where in the transcript it comes from.

On the first question, we designed a Debate Uncertainty Score of our own, taking the construction that suggests itself first: the quantities describing how the three agents' answers agree and move — answer entropy, confidence variance, disagreement persistence, answer flip rate — fitted against whether the debate ends up correct, stopping when the score falls below a threshold. Validated as that construction invites, it reads **33.9%** saved. Simulated sequentially against the consensus rule our system already applies, it elides **0.0%** of the rounds that system executes. The reason is an identity we had not anticipated: answer entropy, the term the fitting weights most heavily, is zero precisely where all three agents agree — exactly the rounds at which a consensus-stopping debate has already ended. This failure concerns our own construction and does not re-evaluate any published method.

The correction it forces has four parts: score only rounds that are not the last round of their debate, since only there can stopping elide anything; label each round with the correctness of the answer that round would itself commit; report saving by sequential simulation against the incumbent rather than as a flagged fraction; and compute error rates and confidence intervals per debate, resampling by question. Thresholds are selected on held-out data throughout.

Under that evaluation a usable signal does survive, and it comes from outside the agreement structure, as the first result implies it must: anything built on agreement risks being confounded with the termination condition. Seven features read off an independent critic, containing no agreement term at all, discriminate better than the four consensus-dynamics features we began with. We deploy them as a **cascade** rather than as a score alone: at round 0, before any solver has spoken, the critic's independent answer is itself taken as the majority of three self-consistency samples rather than one, and a debate whose round-0 score already clears the threshold halts there; a debate that does not is handed to the full multi-round protocol, where the same score is rechecked every round. As a policy the cascade cuts token cost against consensus by **39.4%** on MMLU at an accuracy difference indistinguishable from zero. On GSM8K and StrategyQA, where consensus stopping is already close to optimal, it largely reproduces the incumbent.

The contributions of this paper are:

1. **A measurement protocol for stopping rules, and the diagnostic that makes it one.** A rule can report a large saving and deliver none, and the standard reporting practice cannot tell the two apart. We give a protocol that can: score only non-final rounds, label each round by what it would itself commit, simulate sequentially against the incumbent rather than counting flagged rounds, and resample by question. Only the third of these is diagnostic — the others remove optimism without exposing the fault — and we show this by applying each in isolation. Our own first score is the instrument that exposed the need rather than the subject of the finding: four consensus-dynamics signals, dominated by answer entropy, whose reported saving collapses under sequential simulation because entropy is zero exactly where the incumbent already halts. The protocol is released as runnable code alongside the original over identical data.  
2. **DUS-11 and the critic-SC cascade built on it, the stopping rule that protocol yields.** Applying it points away from the agreement structure entirely, since anything built on agreement risks the same confounding. We define seven features read off an independent critic's verdicts and confidences, containing no agreement term, which discriminate better than the consensus signals by a paired margin whose interval excludes zero — and once they are present, adding the consensus features back changes discrimination by an amount indistinguishable from zero. The rule we deploy and evaluate throughout is a two-stage cascade over this score: a self-consistency check on the critic's round-0 answer that can halt a debate before any round is run, and the same DUS-11 score rechecked every round thereafter for debates that continue.  
3. We measure what this cascade is worth against consensus rather than against a fixed depth, reporting where the rule stops relative to the incumbent and what that costs in tokens and accuracy, and we characterise the condition that separates the benchmark where it pays from those where it does not.  
4. Applying that same protocol to policies that read nothing at all, we find the cascade weakly dominated on two benchmarks of three: fixed depth matches it in accuracy under a question-clustered bootstrap corrected for multiple comparisons, at lower cost. We give the three conditions under which adaptive stopping did hold its ground, two of which are measurable on a pilot run before any signal is built, and report that applied to our own pilot they would have advised against building one on two benchmarks of three.  

## **II. RELATED WORK**

**Multi-agent debate.** Debate-style prompting improves factuality and arithmetic reasoning by having several model instances critique and revise one another's answers over multiple rounds \[1\]. Subsequent work explores divergent-thinking prompts that discourage premature agreement \[2\], round-table protocols with confidence-weighted voting across heterogeneous models \[3\], structured cross-model communication topologies \[6\], and debate as a mechanism for eliciting truthful answers from stronger models \[5\]. Debate has also been adopted for evaluation itself, where multiple agents jointly judge candidate responses \[4\]. These systems either fix the number of agents and rounds in advance, or terminate on unanimous agreement — halting as soon as all agents give the same answer — and are assessed on the accuracy the protocol reaches.

**Early termination of debate.** Shortening debate by monitoring the transcript has been studied directly, and adaptive stability detection for debate-based judges \[7\] is the closest work to our setting. It models the number of correct decisions among *k* judges per round as a time-varying mixture of two Beta-Binomial distributions and halts once the Kolmogorov–Smirnov statistic between consecutive rounds' fitted distributions falls below a threshold; with seven judges and a ten-round cap it converges in four to eight rounds while changing accuracy by 0.1 to 0.6 points. Three properties of that setting position this paper. Its criterion is estimated across the evaluation set, yielding one stopping round per run, where ours decides per debate. Its efficiency is measured against the full ten-round run rather than against a rule that would already have halted the debate. And the quantity it monitors is the round-wise distribution of correct judges, which that paper presents as judge consensus dynamics converging toward unanimity — a structure on which a signal is a candidate to coincide with consensus stopping rather than improve on it. Section V-C reports such a coincidence in a score of our own design.

**Verification and self-correction.** A separate line adds a verifier to the loop, either as a dedicated validator agent over a tree of reasoning states \[8\], as verbal self-reflection between attempts \[9\], or as explicit verification of key conditions extracted from the problem \[10\]. A verifier's judgement already serves as a stopping condition in this line: Reflexion \[9\] loops until its evaluator deems the attempt correct, and key-condition verification \[10\] terminates once its verifier judges the answer likely correct. In each the verdict acts as a binary correctness test rather than as a graded quantity that could be traded against a cost budget.

**Self-consistency and sampling.** An alternative to debate is to sample several independent chains of thought and take a majority vote \[15\]. Self-consistency and debate differ in what produces the diversity — independent sampling in the first case, cross-agent critique in the second — which makes self-consistency the natural zero-debate reference point for any claim that debate rounds are worth their cost. It is tempting to read a debate's first round as that reference point, since no agent has yet seen another's output; Section IV-D explains why we run the baseline separately instead, and TABLE 0 reports both.

Each line above has been developed on its own terms: debate protocols are refined and their accuracy measured \[1\]–\[6\], stopping criteria are proposed and the rounds they save reported \[7\], verifiers are added to catch errors the debaters miss \[8\]–\[10\], and surveys \[11\]–\[14\] catalogue all three alongside the cost concern that motivates stopping. Two questions fall between them. The first is how a stopping rule should be evaluated against the termination mechanism the system already runs: efficiency is reported against a fixed depth, so a reported reduction need not correspond to any round actually removed from a debate that halts on consensus. The second is where the signal should come from. The quantities proposed for it are read from the agreement structure itself, while the verification line shows separately that an independent critic carries usable information about correctness; whether a signal from that critic — one not tied to agreement — can support the stopping decision has not, to our knowledge, been examined. This paper takes up the first with a corrected evaluation protocol and the second with critic-derived features containing no agreement term.

## **III. METHODOLOGY**

### **A. Problem Formulation**

A debate on question *q* produces a sequence of rounds *r* \= 0, 1, …, R−1. Each round *r* has a committed answer *a\_r*, obtained by the system's aggregation rule over the three agents' current answers, and a binary outcome *y\_r* \= 1 if *a\_r* matches the ground truth. A **stopping rule** is a function that, observing only rounds 0…*r*, decides whether to halt and commit *a\_r*.

Two rules serve as reference points. **Consensus stopping**, the incumbent, halts at the first round where all three agents give the same answer. **Fixed-*k*** halts unconditionally after *k* rounds. A candidate rule based on an uncertainty score *u\_r* halts at the first round where *u\_r* \< *T*.

The quantity a stopping rule must be judged on is not how many rounds it flags, but how many rounds it removes *relative to the rule the system already uses*. We make this explicit in Section III-E.

### **B. System and Debate Protocol**

Three agents participate: two solvers and one critic. Fig. 1 shows the architecture and the order of processing within a round; it depicts stage 2 of the stopping cascade defined below — the debate proper. Stage 1, the critic self-consistency gate that runs at round 0 before stage 2 is ever entered, is described immediately after it.

![][image1]

**Fig. 1\.** Stage-2 system architecture and the order of the four model calls in a round. The stage-1 critic self-consistency gate that precedes it is not shown; see the cascade definition below.

At round *r*, solver A answers, then solver B answers; each is shown the critic's messages from rounds 0…*r*−1, so at *r* \= 0 neither has seen anything from another agent. The critic then issues a verdict on solver A's current answer and one on solver B's, each with a confidence, which makes four model calls per round. **The critique produced in round *r* is consumed by the solvers in round *r*\+1, not within round *r*** — a solver never revises in the same round in which it is criticised. The critic's own independent answer is generated once, on the first critic call of round 0, and held fixed thereafter, which costs one additional call at round 0 and keeps the critic an independent third opinion rather than a participant drifting toward the majority.

The committed answer *a\_r* is selected from the three current answers by confidence-weighted selection: the majority answer, with ties broken by the highest confidence attached to a tied answer. Consensus at round *r* holds when all three answers are identical.

**The stopping rule is a two-stage cascade,** everywhere the uncertainty score of Section III-C is used to decide whether to halt, it is applied in two stages rather than one. **Stage 1 — critic self-consistency at round 0.** Before the round-0 features are computed, the critic's single cached answer above is replaced by the majority of three independent self-consistency samples of that same critic, taken instead of the one draw, with the mean confidence of the samples agreeing with the majority. If the resulting round-0 score already clears the threshold *T*, the debate halts there, before either solver has produced a second answer. **Stage 2 — the debate protocol above.** A debate whose round-0 score does not clear the threshold proceeds through the solver/critic rounds just described, with the same score recomputed and rechecked at every subsequent round. We refer to this two-stage rule as the **cascade** throughout the paper; "DUS-11" (Section III-C) names the per-round score it evaluates at both stages, not the deployed rule as a whole. Section V-A confirms the stage-1 gate is stable across seeds and sample count before any result that depends on it is reported.

**All experiments are run with consensus-based early stopping disabled**, so every debate executes the full six rounds. This reconstructs the behaviour of the standard system exactly — the standard system's trajectory is the prefix of the logged trajectory up to its first consensus round — while retaining every later round as counterfactual data. Without this, the sequential simulation in Section III-E would be impossible, because the rounds a rule proposes to elide would never have been observed.

### **C. Uncertainty Signals**

**What each round records.** Every round writes one record per agent containing the raw answer, the answer after benchmark-specific normalisation, a self-reported confidence, and the reasoning text. The critic's record additionally carries its verdict on each solver and a confidence attached to each verdict. All eleven signals below are read off these records after the run; none requires an extra model call.

Fix a round *r*. Write *a*ᴬ, *a*ᴮ, *a*ᶜ for the normalised answers of solver A, solver B and the critic, and *c*ᴬ, *c*ᴮ, *c*ᶜ for their confidences. Write *v*ᴬ and *v*ᴮ for the confidence the critic attaches to its verdict on solver A and on solver B. Consensus at round *r* means *a*ᴬ \= *a*ᴮ \= *a*ᶜ.

Two properties of the instrumentation matter for interpreting the signals. First, **confidences are self-reported**: each agent emits one in its structured output, clipped to \[0, 1\], defaulting to 0.5 when absent and capped at 0.5 when the output parses only partially. Second, because **the critic's independent answer is generated once per question and cached**, *a*ᶜ and *c*ᶜ do not move across rounds. Signals 2, 4 and 7 of the critic family therefore change only when the solvers move, and signal 5 (critic\_conf) is **constant across the rounds of a debate by construction**. At round 0 specifically, the cascade's stage-1 gate (Section III-B) substitutes a self-consistency majority for this single cached draw before any of the signals above are computed, so *a*ᶜ and *c*ᶜ at round 0 are the values that gate produces, not the critic's first raw answer.

**Consensus-dynamics features (DUS-4).** These are the four signals we designed first. All are computed from the pattern of agreement among the three answers.

| \# | Signal | Exact definition | Range |
| :---- | :---- | :---- | :---- |
| 1 | answer\_entropy | Shannon entropy (base 2\) of the empirical distribution over the multiset {*a*ᴬ, *a*ᴮ, *a*ᶜ}: −Σ *p* log₂ *p* | {0, 0.918, 1.585} — three agree / two agree / all differ |
| 2 | confidence\_variance | Population variance of {*c*ᴬ, *c*ᴮ, *c*ᶜ} | \[0, 2/9 ≈ 0.222\] |
| 3 | disagreement\_persistence | Running count of rounds 0…*r* in which consensus did **not** hold. Cumulative and non-decreasing, not a rate | {0, 1, …, *r*\+1} |
| 4 | answer\_flip\_rate | Fraction of the three agents whose normalised answer differs from their own answer at round *r*−1; defined as 0 at *r* \= 0 | {0, ⅓, ⅔, 1} |

Signal 1 is the one that turns out to matter, and its range is the reason: it takes **exactly three values**, and the value 0 occurs precisely when consensus holds. A fitted score that weights it heavily therefore inherits the incumbent's stopping condition rather than adding to it, whatever the other three contribute — Section V-C measures what that costs.

**Critic-derived features (CRITIC-7).** These read only the critic's own state and its verdicts. **None contains a term computed from whether the two solvers agree with each other.**

| \# | Signal | Exact definition | Range |
| :---- | :---- | :---- | :---- |
| 1 | verdict\_conf\_mean | (*v*ᴬ \+ *v*ᴮ) / 2 — how sure the critic is of the two judgements it just issued | \[0, 1\] |
| 2 | n\_disagree | Number of solvers whose answer differs from the critic's: |{*k* ∈ {A, B} : *a*ᵏ ≠ *a*ᶜ}| | {0, 1, 2} |
| 3 | verdict\_conf\_min | min(*v*ᴬ, *v*ᴮ) — the weaker of the two judgements | \[0, 1\] |
| 4 | critic\_vs\_majority | 1 if *a*ᶜ occurs exactly once in {*a*ᴬ, *a*ᴮ, *a*ᶜ}, else 0 | {0, 1} |
| 5 | critic\_conf | *c*ᶜ — the critic's confidence in its own independent answer | \[0, 1\] |
| 6 | conf\_gap\_critic\_solvers | *c*ᶜ − (*c*ᴬ \+ *c*ᴮ) / 2 — how far the third opinion outranks the two solvers | \[−1, 1\] |
| 7 | critic\_alone | 1 if *a*ᴬ \= *a*ᴮ and *a*ᶜ ≠ *a*ᴬ, else 0 | {0, 1} |

Signals 4 and 7 are nested by construction: 7 implies 4, and they differ only in the case where all three answers differ, where 4 fires and 7 does not. Signals 1 and 3 are the only ones in either family that read the critic's *judgement*; the remaining five read its answer, or its confidence in that answer.

We refer to the four consensus-dynamics features as **DUS-4**, the seven critic features as **CRITIC-7**, and their union as **DUS-11**. All eleven are of our own design and are computed offline from logs already written during the debate, so they add no model calls at inference time.

### **D. Score Aggregation**

Two aggregation models appear in this paper, and they play different roles.

**Model 1 — the initial DUS, retained only to audit our original approach.** Features are standardised on the training split; a logistic regression is fitted with the target "the debate's *final* answer is wrong"; non-positive coefficients are clipped to zero, since a component of an uncertainty score must increase with uncertainty; and the surviving positive coefficients are normalised to sum to one, yielding a convex combination over DUS-4. This is the construction whose evaluation fails in Section V-C, and **it is not used for any result after that section**; we keep it because reproducing the failure requires reproducing the score that produced it. Its fitted weights are reported with the audit in TABLE II.

**Model 2 — the corrected stopping score, used for every result from Section V-D onward.** A logistic model is fitted over the standardised DUS-11 features with the target "the answer *this round* would commit is wrong", and its linear predictor is the score *u\_r*. No clipping is applied, because the corrected score is not required to be a convex combination and clipping would discard the inverse-signed critic signals that carry much of the discrimination. Fitting is restricted to non-final rounds, matching the rounds at which the rule can act.

### **E. Evaluation Protocols and Threshold Selection**

**Direct protocol (our first design).** The score is the DUS-4 convex combination of Section III-D, whose weights are fitted on the **training split**. Every round of every debate is then scored — all 27,000 of them, training rounds included — and labelled with the debate's final correctness. *T* is swept over the observed score values **on that same set**, taking the value that maximises the fraction of rounds with *u\_r* \< *T* subject to (wrong ∧ flagged)/(all rounds) ≤ 5%; that fraction is reported as compute saved. The sweep and the report therefore run on the same rounds, and on every round rather than only those at which stopping could elide anything — both among the defects the corrected protocol removes.

*On the 5% error budget.* The constraint bounds the fraction of scored rounds at which the rule fires and commits a wrong answer — the rate at which stopping locks in an error. We fixed 5% before the sweep, roughly one seventh of the 32.9% base error rate over non-final rounds. It is a design choice, not a derived quantity, and Section V-C shows its apparent satisfaction turns on a tie-breaking convention, which is why the corrected protocol does not rely on it.

**Corrected protocol.** Four changes:

1. **Score only non-final rounds.** Stopping at the last round of a debate elides nothing; including such rounds inflates the denominator with decisions that cannot save anything.  
2. **Label each round with what that round commits**, *y\_r*, not with the debate's final outcome. The rule's decision is to commit *a\_r*, so *a\_r* is what must be judged.  
3. **Report saving by sequential simulation against the incumbent.** For each debate, compute *i\_c* and *i\_u* as in Section III-A. Rounds actually elided are Σ max(0, *i\_c* − *i\_u*). A rule that fires only at or after *i\_c* saves nothing, however many rounds it flags.   
4. **Compute error rates and confidence intervals per debate**, resampling by question rather than by round, since rounds of the same debate are not independent.

**Threshold selection.** The threshold *T* is a fitted quantity and must not be chosen on the data it is then evaluated on. We therefore separate three roles — the score is fitted on one set of questions, *T* is chosen on a second, and accuracy and token cost are measured on a third — and report results under two instantiations of that separation:

- **Nested cross-validation (primary).** Five outer folds split by question. For each fold, the remaining questions are divided again into an inner-training portion (75%) and an inner-validation portion (25%); the score is fitted on the inner-training portion, *T\_q* is set to the *q*\-quantile of the score over the inner-validation portion's non-final rounds, and the held-out fold is then scored with that frozen threshold. Every debate is evaluated exactly once, out of fold, under a threshold it never contributed to. This preserves all 4,500 debates and therefore the statistical power of the comparison.  
- **Fixed 70/20/10 holdout (confirmatory).** The score is fitted on the training split, *T\_q* is chosen on the validation split, and results are reported on the test split, which is touched only once. This is conceptually the cleanest separation but leaves roughly 150 debates per benchmark, so its intervals are wide.

Thresholds are chosen separately per benchmark, since the score's scale differs across answer spaces. The last paragraph of Section V-F reports what in-sample threshold selection would have added, by re-running the primary protocol with T taken from the evaluation fold itself and changing nothing else.

Both protocols are implemented over identical inputs and released together, so the discrepancy between what we first reported and what the audit found is reproducible in one invocation.

## **IV. EXPERIMENTAL SETUP**

### **A. Models and Benchmarks**

All three agents are open-weight instruction-tuned models served locally, so that token accounting is exact and no API-side caching confounds cost measurement.

| Role | Model | Temperature |
| :---- | :---- | :---- |
| Solver A | qwen2.5:3b-instruct | 0.3 |
| Solver B | llama3.2:3b | 0.3 |
| Critic | gemma3:4b | 0.2 |

The three models are drawn from three different families so that their errors are not identical by construction, since agreement carries information only if the agents can fail independently; the critic is the largest of the three because it is asked to judge rather than to answer.

Three benchmarks are chosen to vary the size of the answer space while holding the protocol fixed, since Section VI-A tracks that quantity across the outcomes we measure. GSM8K contributes an unbounded answer space, where three agents landing on the same integer is near-conclusive; StrategyQA the binary extreme, where two guessers coincide half the time; and MMLU the intermediate four-option case, which is also the standard broad-knowledge reference.

| Benchmark | Reasoning type | Answer space |
| :---- | :---- | :---- |
| GSM8K | Mathematical | Unbounded integers |
| MMLU | General knowledge | 4 options |
| StrategyQA | Multi-hop | Binary |

### **B. Run Configuration**

Three benchmarks × 5 random seeds × 300 questions per seed, with max\_rounds \= 6 and consensus-based early stopping **disabled**. This yields **4,500 debates** and **27,000 rounds**, of which **22,500** are non-final rounds — the only rounds at which a stopping rule can act. The error rate over non-final rounds is **32.9%**, and the base rate of the corrected label over the same rounds is **0.326**.

| Item | Value |
| :---- | :---- |
| Accelerator | NVIDIA Tesla T4, 16 GB VRAM (Google Colab) |
| Serving framework | Ollama, HTTP generate API, one model resident per call |
| Weights / precision | Default GGUF Q4\_K\_M 4-bit quantisation for all three models |
| Decoding | temperature 0.3 (solvers) / 0.2 (critic), top\_p 1.0 |
| Output length cap | 1024 tokens (solvers), 768 tokens (critic) |
| Seed control | The run seed is passed to Ollama's sampler and also fixes question sampling, so a seed reproduces a full rollout |
|  |  |

**Cost in this paper is measured in tokens**, taken from the counts the serving framework reports for every call — not wall-clock time, energy, or monetary price, all of which depend on batching and hardware rather than on the stopping decision. Since the logs record one token total per debate, a policy stopping after *i*\+1 of *R* rounds is charged (*i*\+1)/*R* of that total; the approximation is applied identically to every policy, including the baselines.

### **C. Data Splits and Statistical Treatment**

Splits are 70/20/10 at the **question** level, not the round level, so that all rounds of a question fall in the same split and no debate leaks across the boundary. The training split contains 18,978 rounds; the question-level split sizes are 685/196/97 for GSM8K, 786/225/112 for MMLU and 453/129/65 for StrategyQA. Where cross-validation is used it is 5-fold, grouped by question, for the same reason.

**Which split each result is measured on.** The two protocols use the splits differently, so the table below states this for every reported quantity.

| Reported quantity | Score fitted on | Threshold chosen on | Measured on |
| :---- | :---- | :---- | :---- |
| TABLE III, Fig. 2 — direct protocol | train | swept over all rounds | all rounds |
| TABLE IV, TABLE V — discrimination | train | not applicable | validation + test |
| TABLE VI, VII, VIII — stopping policies | inner-training part of each fold | inner-validation part | held-out fold, pooled over all debates |
| Confirmatory holdout, Section V-F | train | validation | test |
| Pooled AUC in TABLE X | out-of-fold, 5-fold grouped by question | not applicable | all rounds |

All confidence intervals are bootstrap intervals with 2,000 resamples, **clustered by question**. A power analysis over the non-final rounds (22,500 rounds, 2,748 distinct questions) gives power 0.958 at 65 questions and 0.992 at 100 questions, reaching 1.000 from 200 questions upward. This analysis concerns **displacement of ROC-AUC from chance**, and on that axis the 300 questions per seed are comfortably sufficient. It does not transfer to the accuracy comparisons, which are far less well powered: the paired intervals of TABLE VIII span one to three accuracy points at 1,500 debates per benchmark, so differences smaller than that are not resolvable here, and an accuracy difference reported as indistinguishable from zero should be read as *below the resolution of this design*, not as demonstrated equality.

### **D. Baselines**

Every adaptive policy is compared against reference points at matched token budgets. Two are simulated offline over the logged trajectories; two required separate runs.

*Simulated from the debate logs.*

* **always** — run all six rounds. The upper cost bound, normalised to 100% tokens.  
* **consensus** — the incumbent adaptive rule.  
* **fixed\_k** for *k* \= 1…5 — halt unconditionally after *k* rounds.  
* **oracle** — stop at the first round whose committed answer is correct. Not achievable, but it bounds what any signal could deliver.

*Run separately, no debate.*

* **majority voting** — the three models answer the same question independently and a majority vote decides, ties broken by confidence. All three run in the solver role at identical decoding settings and from a single shared prompt.  
* **self-consistency@3** \[15\] — one model answers the same question three times and a majority vote decides. Reported per model.

**On fixed\_k1 and majority voting.** We call fixed\_k1 the **round-0 ensemble vote**: at round 0 no agent has seen another's output, so the committed answer is a majority over three independent answers, and "ensemble" — not "self-consistency" — is the accurate word, since the three votes come from three different models rather than three samples of one. Conceptually it and the separately run majority-voting baseline are the same policy. We report both because in this implementation they are not the same *measurement*, and the discrepancy is larger than several of the effects this paper reports.

Four differences separate them. First, fixed\_k1 is **asymmetric**: the two solvers answer from the solver prompt at temperature 0.3 with a 1024-token budget, while the third vote comes from the critic, which uses a different prompt and runs at temperature 0.2 with a 768-token budget. Second, its **prompt differs by benchmark**: measured as character-level similarity between the solver prompt at round 0 and the shared independent prompt, the two agree closely on GSM8K (0.992) and MMLU (0.994) but almost not at all on StrategyQA (0.027, 8,390 against 1,007 characters). Third, it is **not uniform across benchmarks**: on GSM8K alone, each solver internally samples three times at round 0 and majority-votes before contributing its single vote, so fixed\_k1 costs roughly seven generations there against three elsewhere. Fourth, its **cost is not measured**: the logs record only total tokens per debate, and the per-round figure used throughout is the uniform approximation total⁄6, which the third point makes substantially wrong for GSM8K round 0\. The separately run baselines are symmetric, uniform across benchmarks, share one prompt, and record their own token counts directly.

Two consequences follow, and both are used later. Neither policy is privileged as *the* majority-voting reference, so TABLE 0 reports both and Section V reads the interval between them as the uncertainty this baseline actually carries. And an earlier version of this work called fixed\_k1 "self-consistency@3", which the first and third differences above make wrong in any case: self-consistency varies the *sample*, holding the model fixed, whereas fixed\_k1 varies the *model*. Every claim about self-consistency in this paper rests on the separately run baseline, never on fixed\_k1.

### **E. Evaluation Metrics**

Two families of metric are reported, and they answer different questions.

*Does the score know which rounds are unsafe to stop at?*

* **ROC-AUC** over non-final rounds, with the positive class "the answer this round would commit is wrong". Chance is 0.500; displacement in *either* direction is signal, since some features run inverse. Reported with bootstrap 95% CIs clustered by question.

*Does stopping on that score actually help?*

* **Accuracy**: fraction of debates whose committed answer at the policy's stopping round is correct.  
* **Token cost**: tokens spent by the policy as a percentage of running all six rounds (always \= 100%).  
* **Δ Accuracy and Δ Token against consensus**: computed per seed on the same question set and then averaged across seeds, so each comparison is paired.  
* **Earlier / Same / Later**: the share of debates in which the rule stops strictly before, at, or after the consensus round — the diagnostic that exposes a rule whose stop region lies inside the incumbent's.  
* **Error rate**: the fraction of *debates* in which the rule fires and commits a wrong answer.

The primary metric is the paired comparison against consensus: Δ Token must be negative with an accuracy difference indistinguishable from zero. ROC-AUC is a diagnostic of the signal, not evidence of savings — Section V-C is a case where a respectable AUC accompanied a zero saving, and Section VI-A one where the highest AUC accompanies the smallest.

## **V. RESULTS**

Results open with the comparison that governs how everything after it should be read. TABLE 0 places every policy in this paper — adaptive and non-adaptive alike — on one accuracy/cost plane. It is placed first because the ordering it shows determines what the rest of the paper can claim: on two of three benchmarks a rule that ignores the transcript entirely is not distinguishable in accuracy from the cascade, and costs less.

**TABLE 0 — ALL POLICIES ON ONE PLANE** (4,500 debates; accuracy / tokens as % of *always*)

| Policy | Reads transcript? | GSM8K | MMLU | StrategyQA |
| :---- | :----: | ----: | ----: | ----: |
| **Cost bounds** | | | | |
| always (6 rounds) | no | 0.779 / 100.0% | 0.547 / 100.0% | 0.686 / 100.0% |
| oracle (unachievable) | — | 0.863 / 32.7% | 0.681 / 50.7% | 0.787 / 38.0% |
| **No debate — one round, majority vote** | | | | |
| majority voting, 3 models, symmetric | no | 0.762 / 11.2% | 0.521 / 13.2% | 0.615 / 5.6% |
| self-consistency@3, Qwen2.5-3B | no | 0.638 / 10.9% | 0.582 / 11.1% | 0.548 / 4.1% |
| self-consistency@3, Llama3.2-3B | no | 0.691 / 9.4% | 0.293 / 16.4% | 0.611 / 4.1% |
| self-consistency@3, Gemma3-4B | no | 0.780 / 12.7% | 0.515 / 10.5% | 0.679 / 8.4% |
| **Fixed depth — halt after *k* rounds** | | | | |
| fixed\_k1 *(round-0 ensemble vote)* | no | 0.784 / 16.7% | 0.530 / 16.7% | 0.683 / 16.7% |
| fixed\_k2 | no | 0.797 / 33.3% | 0.540 / 33.3% | 0.698 / 33.3% |
| fixed\_k3 | no | 0.783 / 50.0% | 0.550 / 50.0% | 0.691 / 50.0% |
| fixed\_k4 | no | 0.779 / 66.7% | 0.557 / 66.7% | 0.677 / 66.7% |
| fixed\_k5 | no | 0.793 / 83.3% | 0.563 / 83.3% | 0.689 / 83.3% |
| **Adaptive — reads the transcript** | | | | |
| consensus (incumbent) | yes | 0.785 / 56.0% | 0.546 / 68.2% | 0.691 / 48.5% |
| DUS-11, *q* \= 0.4 | yes | 0.781 / 63.0% | 0.543 / 50.4% | 0.687 / 58.9% |
| DUS-11, *q* \= 0.5 | yes | 0.785 / 52.6% | 0.545 / 41.3% | 0.690 / 48.2% |

Self-consistency@3 is reported per model rather than as a single row, because the three disagree by up to 29 accuracy points on the same benchmark, and no one of them is best on more than one.

Five readings follow, and Section VI-A tests each.

First, **among policies that run the debate at all, no one separates from another by more than about two accuracy points on any benchmark**, while the oracle sits seven to fourteen points above all of them. That family occupies a narrow band; the headroom is real but none of them reaches it.

Second, **among policies that do not run the debate, the spread is an order of magnitude larger** — up to 29 points on a single benchmark, between Qwen2.5-3B at 0.582 on MMLU and Llama3.2-3B at 0.293 on the same questions. The choice of which cheap baseline to run matters far more than the choice of what to do with the transcript once you have it.

Third, **the cheapest policies are competitive only if the right one is chosen, and the right one changes with the benchmark**. Gemma3-4B sampled three times reaches 0.780 on GSM8K and 0.679 on StrategyQA, within about a point of every debate policy at roughly a fifth of the cost — but only 0.515 on MMLU, where Qwen2.5-3B is the best policy in the entire table. No single model is best on more than one benchmark, and the model that wins MMLU is the worst of the three on both others.

Fourth, **MMLU behaves differently from the other two**, and it is the only benchmark on which accuracy rises monotonically with *k* (0.530, 0.540, 0.550, 0.557, 0.563). There debate does something. Elsewhere the fixed-*k* column is a random walk of amplitude comparable to the seed-to-seed spread: 0.784, 0.797, 0.783, 0.779, 0.793 on GSM8K and 0.683, 0.698, 0.691, 0.677, 0.689 on StrategyQA, neither of which is ordered in *k*.

Fifth, and least expected, **the same baseline measured two ways differs by up to 6.7 points**. fixed\_k1 and the symmetric majority vote are the same policy in the only sense that matters conceptually — three models answer independently, a majority decides — yet they report 0.784 against 0.762 on GSM8K, 0.530 against 0.521 on MMLU, and 0.683 against 0.615 on StrategyQA. The gap is implementation, not chance: Section IV-D enumerates the four differences, and on StrategyQA, where the gap is widest, the two prompts share only 2.7% of their content. A "majority-voting baseline" is therefore not a single number, and a paper that reports one without specifying prompt and decoding settings has not reported a reproducible reference point. We give both, and treat the interval between them as the honest uncertainty on this baseline.

### **A. Critic-SC Stability**

The cascade's stage-1 gate — the critic self-consistency check at round 0, defined in Section III-B — replaces the critic's single independent round-0 answer with the majority of three independent self-consistency samples of that same critic, before the score is computed and thresholded. Because the score, the threshold, and every saving reported later in this section rest on this gate firing the same way run to run, its stability is checked first.

**TABLE 0-A — CRITIC-SC GATE STABILITY**

| Benchmark | Flip rate, 1→3 votes | Round-0 gate rate (mean ± SD over 5 seeds) | CV |
| :---- | ----: | ----: | ----: |
| GSM8K | 7.5% | 54.5% ± 5.2pp | 9.5% |
| MMLU | 9.9% | 65.1% ± 1.3pp | 1.9% |
| StrategyQA | 5.2% | 52.0% ± 5.5pp | 10.5% |

The critic-SC majority answer flips in 5.2–9.9% of debates when going from one vote to three, and the fraction of debates the gate fires at round 0 is stable across the five independent seeds on every benchmark, with coefficient of variation between 1.9% and 10.5%. Neither the answer nor the gate rate is a coin flip that happened to land once; the check would have failed had either flip rate or seed-to-seed CV come out above the 15%/25% bar we set in advance, and it did not on any benchmark. Every "DUS-11" and "unc\<q*" entry in the tables below is therefore this cascade end to end — the stage-1 gate checked in this section followed by the stage-2 per-round score — not the score alone.

### **B. Measured Research Gaps**

Two premises usually asserted in this literature are measurable on our logs.

**Fixed rounds are wasteful.** Of the 22,500 non-final rounds, **20,385 (90.6%)** commit the same verdict the debate eventually reaches. Nine of every ten continued rounds cost tokens and change nothing. This bounds what any stopping rule could reclaim; it is not itself reclaimable, since which rounds those are is known only in hindsight.

**Consensus does not imply correctness.** Of the **11,754** rounds in which all three agents agree, **2,020 (17.2%)** agree on a wrong answer. TABLE I shows the rate varies sharply across benchmarks, from 2.9% where the answer space is unbounded to 24–28% where it is small.

**TABLE I — BLIND-SPOT RATE BY BENCHMARK**

| Benchmark | Answer space | Blind-spot rate |
| :---- | :---- | ----: |
| GSM8K | Unbounded integers | **2.9%** |
| StrategyQA | Binary | **24.4%** |
| MMLU | 4 options | **28.0%** |
| **All** | — | **17.2%** (2,020 / 11,754) |

Three agents converging on the same unbounded integer is near-impossible by chance, so on GSM8K agreement is close to proof. Three agents converging on one of two options happens about a quarter of the time among guessers, so on StrategyQA agreement is weak evidence. Chance agreement alone does not order the two bounded spaces, however: three agents selecting one of four options coincide 6.25% of the time, a quarter as often as on a binary task, yet MMLU carries the higher blind spot of the two. What TABLE I establishes is therefore a separation between an unbounded answer space and a small one; within the small spaces the agents are far from independent, and Section VI-A takes up what else could order them. The separation itself governs every later result.

### **C. Auditing Our Own Result: The Direct Protocol Is Circular**

Fitting DUS-4 as described in Section III-D concentrates the score on a single feature (TABLE II): answer entropy takes **0.638** of the weight, and confidence variance is clipped away entirely. Applying our direct protocol selects *T* \= −0.604 and reports a **33.9%** stop rate at a per-round error of **4.85%**, inside the 5% budget.

**TABLE II — LEARNED DUS-4 WEIGHTS**

| Feature | Raw coefficient | Final weight |
| :---- | ----: | ----: |
| answer\_entropy | 0.638 | **0.638** |
| answer\_flip\_rate | 0.194 | 0.194 |
| disagreement\_persistence | 0.167 | 0.168 |
| confidence\_variance | **−0.043** | **0.000** (clipped) | Taken at face value this looked like a result. TABLE III places it beside the sequential accounting we ran afterwards as a check, and Fig. 2 shows where the reported number goes.

**TABLE III — WHAT WE FIRST REPORTED VERSUS WHAT THE SEQUENTIAL AUDIT FOUND**

| Quantity | Direct protocol | Sequential audit |
| :---- | ----: | ----: |
| Rounds flagged "stop" | **33.9%** | — |
| Error rate | 4.85% per round | 5.09% per debate |
| Debates in which the flag fires | — | 1,517 / 4,500 |
| Rounds executed by the standard system | — | 14,987 |
| **Debates stopping earlier than consensus** | — | **0 / 4,500 (0.0%)** |
| **Rounds actually elided** | — | **0 / 14,987 (0.0%)** |

![][image2]

**Fig. 2\.** Uncertainty score by consensus status, with the threshold the direct protocol selects.

Our reported saving was entirely illusory. The mechanism is an exact identity we had not checked for:

**answer\_entropy \= 0 ⟺ consensus**, holding in **27,000 of 27,000** logged rounds.

With answer\_entropy carrying 0.638 of the weight, the region {*u* \< *T*} falls inside {entropy \= 0}, so all **9,149** rounds below the threshold are consensus rounds — the set the debate loop already treats as terminal. This is not an implementation error; we verified the code against its specification.

The general statement is a condition, not a claim about scores in general: **whenever the sublevel set {*u* \< *T*} is contained in the consensus set, the saving against consensus stopping is exactly zero, for every *T* and every debate.** A dominant agreement term is what produces containment, but the weight at which it sets in depends on the other features' spread — so the condition is what to test, not the weight. It is a one-line check on logged rounds, and it is the transferable part of this section.

Three smaller defects run in the same direction, and are worth recording chiefly because they show what a corrected protocol removes for free. The operating point survives only under a strict inequality: exactly **872** rounds sit at *u* \= *T*, and admitting them turns 33.9% at 4.85% into **37.1%** at **5.63%**, outside the stated budget. The error denominator should be debates rather than rounds, which raises 4.85% to **5.09%**, an inflation of **1.05×**. And scoring every round rather than only non-final ones lifts AUC from **0.714** to **0.717**, an inflation of 0.003. Each is small, each is optimistic, and — the point Section VI-B returns to — not one of them would have exposed the circularity.

### **D. Discrimination Under the Corrected Protocol**

We now score only non-final rounds and label each round with what it would itself commit. TABLE IV reports each feature in isolation. AUC displaced from 0.500 in *either* direction is signal; features 1 and 3 of the critic family run inverse, since high verdict confidence predicts a *correct* commit.

**TABLE IV — SINGLE-FEATURE ROC-AUC (22,500 NON-FINAL ROUNDS)**

| Feature | Family | AUC | |AUC − 0.5| |
| :---- | :---- | ----: | ----: |
| answer\_entropy | DUS-4 | 0.694 | 0.194 |
| verdict\_conf\_mean | CRITIC-7 | **0.309** | **0.191** |
| n\_disagree | CRITIC-7 | **0.690** | **0.190** |
| disagreement\_persistence | DUS-4 | 0.664 | 0.164 |
| verdict\_conf\_min | CRITIC-7 | **0.349** | **0.151** |
| critic\_vs\_majority | CRITIC-7 | 0.624 | 0.124 |
| answer\_flip\_rate | DUS-4 | 0.597 | 0.097 |
| critic\_conf | CRITIC-7 | 0.446 | 0.054 |
| confidence\_variance | DUS-4 | 0.533 | 0.033 |
| conf\_gap\_critic\_solvers | CRITIC-7 | 0.522 | 0.022 |
| critic\_alone | CRITIC-7 | 0.517 | 0.017 |

Two observations. First, the second and third strongest individual signals in the entire pool are critic-derived (verdict\_conf\_mean at 0.191 and n\_disagree at 0.190), essentially matching answer\_entropy at 0.194. Second, confidence\_variance — one of the four signals we designed — is nearly uninformative at 0.533, consistent with its elimination by clipping in TABLE II.

**TABLE V — MODEL COMPARISON (BOOTSTRAP CI CLUSTERED BY QUESTION)**

| Model | Features | ROC-AUC | 95% CI |
| :---- | ----: | ----: | :---- |
| DUS-4 — consensus dynamics | 4 | 0.705 | \[0.671, 0.737\] |
| **CRITIC-7 — critic signals alone** | 7 | **0.729** | \[0.703, 0.757\] |
| **DUS-11 — both families** | 11 | **0.736** | \[0.707, 0.764\] |
| DUS-11, 5-fold CV grouped by question | 11 | **0.738** | \[0.720, 0.754\] |

The intervals in TABLE V are marginal — each model bootstrapped separately — and those of DUS-4 and CRITIC-7 overlap, which settles nothing in either direction: two models scored on the same questions have strongly correlated errors, so overlapping marginal intervals are compatible with a clearly non-zero difference. TABLE Va therefore bootstraps the **difference** itself, drawing one set of questions per resample and scoring both models on exactly that set.

**TABLE Va — PAIRED DIFFERENCE IN ROC-AUC** (2,000 resamples, clustered by question, both models scored on each resample)

**Seven signals containing no agreement term outperform the four we designed first**, 0.729 against 0.705, and the paired interval excludes zero. This is the central consequence of Section V-C: because the agreement-based family is confounded with the termination condition, the signal that permits genuinely earlier stopping must come from outside the agreement structure.

The third row closes the argument from the other side. Adding the four consensus-dynamics features back on top of the critic family moves AUC by 0.007, an interval that **includes zero**: once the critic features are present, the agreement features contribute nothing we can detect. They are not merely confounded with the termination condition — on this evidence they are redundant. We carry DUS-11 forward as the stopping score because it is the superset and no result below turns on that 0.007, but a deployment could use CRITIC-7 alone at no measured cost.

Within the critic family, signals 1–3 of TABLE IV carry most of the weight; signals 6 and 7 add almost nothing, and \#7 is a strict subset of \#4.

Discrimination varies sharply by benchmark, and here it is the size of the answer space that orders it — 0.882 on GSM8K, 0.705 on MMLU, 0.590 on StrategyQA, against unbounded, four and two. The blind-spot rates of TABLE I do not order it: StrategyQA has the lower blind spot of the two bounded benchmarks and the lower AUC. Section VI-A returns to this split.

### **E. Stopping Behaviour Relative to Consensus**

Discrimination is necessary but not sufficient: a score can rank rounds well and still stop exactly where consensus already stops. Because every debate was run to six rounds, we can compare *i\_u* with *i\_c* directly on all 4,500 debates. TABLE VI is, to our knowledge, the measurement missing from prior evaluations.

**TABLE VI — WHEN DUS-11 STOPS RELATIVE TO CONSENSUS**

| Benchmark | q | Earlier (*i\_u* \< *i\_c*) | Same round | Later |
| :---- | :---- | ----: | ----: | ----: |
| GSM8K | 0.5 | 200 · 13.3% | **1,155 · 77.0%** | 145 · 9.7% |
| StrategyQA | 0.5 | 149 · 9.9% | **1,133 · 75.5%** | 218 · 14.5% |
| MMLU | 0.4 | **614 · 40.9%** | 809 · 53.9% | 77 · 5.1% |
| **MMLU** | **0.5** | **849 · 56.6%** | 631 · 42.1% | 20 · 1.3% |

The contrast with Section V-C is the result. Our DUS-4 under the direct protocol stopped earlier in **0 of 4,500** debates. DUS-11 stops earlier in **56.6%** of MMLU debates. But on GSM8K and StrategyQA it lands on the consensus round about three quarters of the time, largely reproducing the incumbent. TABLE VI therefore predicts TABLE VIII before we compute it: an adaptive rule departs from consensus where consensus is least trustworthy. 

### **F. Cost Reduction at Unchanged Accuracy**

The question this section answers is not how much accuracy a cheaper policy buys back — it is how much of the cost can be removed *before* accuracy moves at all. TABLE VII gives the full policy comparison under the primary protocol. Accuracy is the mean over 5 seeds; token cost is expressed as a percentage of running all six rounds. Fig. 3 puts the two quantities on separate axes rather than against each other, in the order the problem imposes: the top row is what each policy **spends**, which is the quantity being minimised, and the bottom row is the **constraint** — whether accuracy moved — measured against DUS-11 so that every baseline in this paper can be read off one origin. Oracle is left out of the figure so that the achievable policies stay legible.

**TABLE VII — POLICY COMPARISON (ACCURACY / TOKEN %, MEAN OVER 5 SEEDS, HELD-OUT THRESHOLDS)**

| Policy | GSM8K | MMLU | StrategyQA |
| :---- | :---- | :---- | :---- |
| always (6 rounds) | 0.779 / 100.0% | 0.547 / 100.0% | 0.686 / 100.0% |
| consensus | 0.785 / 56.0% | 0.546 / 68.2% | 0.691 / 48.5% |
| round-0 ensemble vote (fixed\_k1) | 0.784 / 16.7% | 0.530 / 16.7% | 0.683 / 16.7% |
| fixed\_k2 | **0.797** / 33.3% | 0.540 / 33.3% | **0.698** / 33.3% |
| fixed\_k3 | 0.783 / 50.0% | 0.550 / 50.0% | 0.691 / 50.0% |
| unc\<q30 | 0.779 / 70.7% | 0.548 / 63.8% | 0.685 / 68.1% |
| unc\<q40 | 0.781 / 63.0% | 0.543 / **50.4%** | 0.687 / 58.9% |
| **unc\<q50** | 0.785 / **52.6%** | 0.545 / **41.3%** | 0.690 / **48.2%** |
| oracle (unachievable) | 0.863 / 32.7% | 0.681 / 50.6% | 0.787 / 38.0% |

![][image3]

**Fig. 3\.** Every policy on the two axes the problem actually has, by benchmark: (a) GSM8K, (b) MMLU, (c) StrategyQA. **Top row — the objective.** Token cost as a percentage of running all six rounds; the dotted line marks DUS-11 at *q* \= 0.5. **Bottom row — the constraint.** Δ accuracy against that same DUS-11 setting, in accuracy points, with 95% bootstrap intervals clustered by distinct question, on one axis shared across all three benchmarks so that a given gap is the same height everywhere. The shaded band spans zero to the oracle's headroom on that benchmark, so every bar is read against the accuracy actually left on the table rather than against an axis chosen to flatter it.

The three no-debate baselines are drawn separately because they are three different mechanisms, not three settings of one. **Round-0 ensemble vote** takes the three models' round-0 answers from inside the debate harness, where they are asymmetric — two solvers at temperature 0.3 with a 1024-token budget, the critic at 0.2 and 768. **Majority voting** runs the same three models standalone, symmetric, from one shared prompt. **Self-consistency@3** replaces model diversity with sample diversity: one model, three draws. The first two differ only in *how they are measured*, which is why the gap between them — 6.7 accuracy points on StrategyQA — belongs in the figure rather than being averaged away (Section IV-D).

TABLE VII-A tests directly whether depth beyond *k* \= 2 is worth its extra cost, pairing each deeper fixed-*k* against fixed\_k2 with a question-clustered bootstrap.

**TABLE VII-A — FIXED-K (*k* \= 3, 4, 5) PAIRED AGAINST FIXED\_K2** (Bonferroni-corrected over the 9 comparisons)

| Benchmark | fixed\_k3 − fixed\_k2 | *p* | fixed\_k4 − fixed\_k2 | *p* | fixed\_k5 − fixed\_k2 | *p* |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| GSM8K | −0.014 \[−0.028, +0.000\] | 0.058 | −0.017 \[−0.031, −0.003\] | 0.016\* | −0.003 \[−0.016, +0.009\] | 0.642 |
| MMLU | \+0.010 \[−0.007, +0.027\] | 0.247 | \+0.017 \[+0.001, +0.032\] | 0.038\* | \+0.023 \[+0.007, +0.040\] | 0.004\*\* |
| StrategyQA | −0.007 \[−0.021, +0.008\] | 0.404 | −0.021 \[−0.037, −0.005\] | 0.009\* | −0.009 \[−0.024, +0.005\] | 0.231 |

\*\* survives Bonferroni ( *p* \< 0.0056); \* nominal only.

**Fixed depth is pinned to *k* \= 2 on all three benchmarks**, and TABLE VII-A shows why that is a structural choice rather than a result-dependent one: only MMLU buys accuracy from going deeper, at 33–50% more tokens, and every other cell is flat or loses. *k* \= 1 halts at round 0, before any critique has been exchanged, so it is a no-debate vote and already appears as its own bar; *k* \= 2 is therefore the shallowest depth that contains an actual round of debate, making it the cheapest representative of the fixed-depth family in a figure about cost. "The best *k* per benchmark" is not an admissible choice here, since that *k* varies (2, 5, 2) and nothing available in advance identifies it. Self-consistency, by contrast, *is* shown at its best model per benchmark — an asymmetry that favours the no-debate side, so the reading below holds under the assumption least favourable to us; TABLE XII gives the full family.

Read together, the two rows give the paper's result: among policies that run the debate, the bottom row is flat while the top row is not, so the choice among them is a cost decision; outside them, the bottom row is where the variation lives instead. Oracle is omitted (TABLE VII).

TABLE VIII gives the paired comparison against consensus, computed per seed on the same question set and then averaged, which controls for seed-to-seed variation.

**TABLE VIII — PAIRED COMPARISON AGAINST CONSENSUS STOPPING** (nested CV, held-out thresholds)

| Benchmark | q | Δ Accuracy (mean ± SD over seeds) | 95% CI, clustered by question | Δ Token |
| :---- | :---- | ----: | ----- | ----: |
| GSM8K | 0.5 | −0.001 ± 0.004 | \[−0.007, \+0.006\] | **−6.0% ± 5.1%** |
| **MMLU** | 0.4 | −0.003 ± 0.013 | \[−0.015, \+0.008\] | **−26.1% ± 3.3%** |
| **MMLU** | 0.5 | −0.001 ± 0.013 | \[−0.015, \+0.013\] | **−39.4% ± 2.3%** |
| StrategyQA | 0.5 | −0.001 ± 0.008 | \[−0.009, \+0.007\] | −0.4% ± 8.3% |

Read the accuracy column first, because it is the one that does not move. All four differences lie between −0.003 and −0.001, and all four intervals cover zero. The bottom row of Fig. 3 extends that picture to every policy that runs the debate at all: across `always`, `consensus`, all five fixed depths and both DUS settings, on all three benchmarks, no bar reaches two accuracy points and every interval crosses zero. The variation in that row comes entirely from the no-debate family, whose bars run off the shared axis. Within the debate families, then, the choice of stopping policy is barely a choice about accuracy. It is a choice about cost, and that is where the policies separate: consensus already runs at 48.5–68.2% of the six-round budget, and on MMLU the adaptive rule removes a further 39.4% of what consensus spends, at an accuracy difference of −0.001 (CI \[−0.015, \+0.013\]) — 41.3% of the six-round budget for the accuracy consensus buys at 68.2%. On GSM8K the 6.0% reduction is comparable to its own seed-to-seed spread of 5.1%, and on StrategyQA the 0.4% reduction is well inside it; on those two benchmarks the correct reading is that there is no saving to claim, not that a saving was paid for in accuracy.

Two caveats belong in the same reading. fixed\_k2 attains 0.797 on GSM8K and 0.698 on StrategyQA at 33.3% of tokens, beating every adaptive policy we report there on both axes: where the rule does not beat consensus, it does not beat a trivially simple baseline either. And the oracle row shows how much is unclaimed — 0.681 at 50.6% tokens on MMLU against our 0.545 at 41.3% — a gap far wider than the one between our signal and the incumbent.

Under the fixed 70/20/10 holdout the MMLU token reduction reproduces (−39.2% ± 14.6% at q \= 0.5), but the accuracy difference is −0.046, CI \[−0.096, −0.007\] over 151 test debates, which the primary protocol's 1,500 debates do not show. That estimate rests on roughly thirty test debates per seed and on an easier test subset (always 0.587 against 0.547), yet it cannot rule out a real accuracy cost on MMLU. We take the nested-CV figure as primary on grounds of sample size and record the disagreement as the strongest caveat on this result.

Re-selecting T from the evaluation fold instead of held-out data, with score and debates unchanged, shifts Δ Token by at most 3.2 percentage points and Δ Accuracy by at most 0.004 across all fifteen benchmark–quantile combinations, with inconsistent sign: on MMLU at q \= 0.5 it reports −40.9% against the held-out −39.4%. In-sample threshold selection is a real but small distortion here, of a different order from the structural defect of Section V-C.

### **G. Is Debate Worth Its Cost?**

TABLE IX compares the answer committed at round 0 with the answer committed at round 5, following the rescue/hurt framing used to assess whether debate helps individual questions rather than benchmark averages.

**TABLE IX — RESCUE AND HURT OVER SIX ROUNDS**

| Benchmark | n | Rescued | Corrupted | Net | McNemar *p* |
| :---- | ----: | ----: | ----: | ----: | ----: |
| GSM8K | 1,500 | 54 · 3.6% | 62 · 4.1% | **−8** | 0.516 |
| MMLU | 1,500 | 111 · 7.4% | 85 · 5.7% | \+26 | 0.074 |
| StrategyQA | 1,500 | 91 · 6.1% | 86 · 5.7% | \+5 | 0.764 |
| **Total** | **4,500** | **256 · 5.7%** | **233 · 5.2%** | **\+23 (+0.51%)** | **0.320** |

*p* is the two-sided exact McNemar test on the discordant pairs, under the null that debate changes accuracy by zero.

Six rounds of debate over 4,500 questions rescue 256 answers and corrupt 233, a net of **\+23 (+0.51%)**. No net is distinguishable from zero: the exact McNemar test in the last column of TABLE IX returns *p* between 0.074 and 0.764. The negative sign on GSM8K carries no weight — it is 8 questions out of 116 discordant ones, at *p* \= 0.516 — and should not be read as evidence that debate harms that benchmark. What the data support is the stronger and simpler statement that **six rounds of debate change accuracy by an amount indistinguishable from zero on every benchmark**. Combined with the 90.6% no-change rate of Section V-B, debate at this scale is a zero-sum process at six times the single-pass cost. This explains why stopping early costs so little accuracy in TABLE VIII: there is little accuracy left to lose.


## **VI. DISCUSSION**

### **A. Does Adaptive Stopping Beat Not Reading the Transcript?**

TABLE 0 invites a direct question, and it must be answered before any governing quantity is discussed: is the cascade distinguishable from a policy that halts after a fixed two rounds? We test it directly. Because the five seeds draw overlapping question sets — 1,500 draws yield 978 distinct questions on GSM8K, 1,123 on MMLU and 647 on StrategyQA, so any two seeds share 12% to 46% of their questions — a paired *t*-test across five seeds is both underpowered and anti-conservative. TABLE XI instead resamples *questions*, clustering all appearances of a question across seeds.

**TABLE XI — fixed\_k2 MINUS DUS-11** (10,000 bootstrap resamples, clustered by distinct question)

| Benchmark | Opponent | Δ Accuracy | 95% CI | *p* | Δ Token |
| :---- | :---- | ----: | :----: | ----: | ----: |
| GSM8K | DUS-11, *q* \= 0.4 | **\+0.0153** | \[+0.0013, \+0.0293\] | 0.035 | **−29.7%** |
| GSM8K | DUS-11, *q* \= 0.5 | \+0.0120 | \[−0.0013, \+0.0253\] | 0.080 | −19.3% |
| MMLU | DUS-11, *q* \= 0.4 | −0.0027 | \[−0.0182, \+0.0132\] | 0.768 | −17.0% |
| MMLU | DUS-11, *q* \= 0.5 | −0.0053 | \[−0.0207, \+0.0101\] | 0.525 | −8.0% |
| StrategyQA | DUS-11, *q* \= 0.4 | \+0.0107 | \[−0.0046, \+0.0257\] | 0.178 | −25.6% |
| StrategyQA | DUS-11, *q* \= 0.5 | \+0.0080 | \[−0.0066, \+0.0226\] | 0.310 | −14.9% |

Positive Δ Accuracy favours fixed\_k2; negative Δ Token means fixed\_k2 is cheaper.

**On accuracy, no.** One cell reaches nominal significance, GSM8K against *q* \= 0.4 at *p* \= 0.033. But fixed\_k2 was selected by reading the table, and the comparison it was selected from spans five values of *k* across three benchmarks. Under Bonferroni correction for those fifteen comparisons the threshold is *p* \< 0.0033, and no cell approaches it; sweeping all fifteen against DUS-11 at *q* \= 0.5, the smallest *p* is 0.009, and it belongs to a cell in which fixed\_k *loses*. The apparent superiority of fixed\_k2 is the expected behaviour of an argmax over fifteen noisy, near-identical quantities.

**On cost, yes — everywhere.** fixed\_k2 is cheaper in all six comparisons, by 8.0% to 29.7% of full debate cost. That column contains no ambiguity and needs no correction: it is a property of the policies, not an estimate.

The two findings together say that fixed\_k2 does not *beat* DUS-11 but **weakly dominates** it: accuracy statistically indistinguishable on every benchmark, cost consistently lower. Against fixed depth alone, the cascade is not wrong — it is not paid for.

That conclusion does not survive contact with the wider reference class. TABLE XII places DUS-11 against every policy in this paper that does not read the transcript, including self-consistency at three sample budgets matched by model.

**TABLE XII — NO-DEBATE BASELINES MINUS DUS-11 (*q* \= 0.5)** (10,000 resamples, clustered by question; 27 comparisons, Bonferroni threshold *p* \< 0.00185)

| Benchmark | Baseline | Δ Accuracy | 95% CI | *p* | Δ Token |
| :---- | :---- | ----: | :----: | ----: | ----: |
| GSM8K | majority voting | −0.0227 | \[−0.0419, −0.0038\] | 0.024 | −41.5% |
| GSM8K | SC@3, Qwen2.5-3B | **−0.1467** | \[−0.1752, −0.1178\] | **0.0001** | −41.8% |
| GSM8K | SC@3, Llama3.2-3B | **−0.0940** | \[−0.1210, −0.0668\] | **0.0001** | −43.3% |
| GSM8K | SC@3, Gemma3-4B | −0.0047 | \[−0.0224, \+0.0132\] | 0.630 | −40.0% |
| MMLU | majority voting | −0.0247 | \[−0.0493, \+0.0007\] | 0.055 | −28.1% |
| MMLU | SC@3, Qwen2.5-3B | \+0.0367 | \[+0.0072, \+0.0661\] | 0.015 | −30.2% |
| MMLU | SC@3, Llama3.2-3B | **−0.2520** | \[−0.2882, −0.2161\] | **0.0001** | −24.9% |
| MMLU | SC@3, Gemma3-4B | −0.0307 | \[−0.0521, −0.0092\] | 0.006 | −30.8% |
| StrategyQA | majority voting | **−0.0747** | \[−0.1077, −0.0405\] | **0.0002** | −42.6% |
| StrategyQA | SC@3, Qwen2.5-3B | **−0.1420** | \[−0.1880, −0.0961\] | **0.0001** | −44.1% |
| StrategyQA | SC@3, Llama3.2-3B | **−0.0787** | \[−0.1165, −0.0411\] | **0.0002** | −44.1% |
| StrategyQA | SC@3, Gemma3-4B | −0.0113 | \[−0.0317, \+0.0090\] | 0.293 | −39.8% |

Bold marks intervals excluding zero after correction. Negative Δ Accuracy means the baseline is worse than DUS-11.

**Six of twelve no-debate baselines lose to DUS-11 by margins that survive correction**, several of them enormously: 14.7 points on GSM8K, 25.2 on MMLU, 14.2 on StrategyQA. Exactly one beats it — Qwen2.5-3B on MMLU, by 3.7 points at 30.2% fewer tokens, and that cell is nominally significant at *p* \= 0.015 but does not clear the corrected threshold. Two more, Gemma3-4B on GSM8K and StrategyQA, are indistinguishable from DUS-11 at roughly 40% lower cost.

The pattern that matters is not the win column but the **variance**. Which cheap baseline is competitive changes with the benchmark, and changes completely: Gemma3-4B is within a point of DUS-11 on GSM8K and StrategyQA and 3.3 points below on MMLU, while Qwen2.5-3B is the best policy in this paper on MMLU and 14.7 points below DUS-11 on GSM8K. A practitioner choosing one model to sample three times has no way to make that choice correctly in advance, and the cost of choosing wrong is an order of magnitude larger than anything adaptive stopping gains or loses.

Ranked by worst-case regret against the best realisable policy on each benchmark, the ordering inverts the one fixed depth alone suggested: DUS-11 gives up at most 3.7 points, fixed\_k2 at most 4.2, Gemma3-4B at most 6.7, the symmetric majority vote at most 8.3, and Qwen2.5-3B at most 15.9. **The cascade buys robustness across benchmarks, not accuracy on any one of them** — and against fixed depth, which is equally robust and cheaper, it still does not pay for itself.

MMLU remains the single benchmark where a debate policy is not the best choice available, and the policy that beats it there reads nothing at all.

TABLE X aligns the quantities that could govern *where* the remaining difference lives. One relation holds monotonically across all three benchmarks — the saving achieved tracks the consensus token share — and no other column orders with it.

**TABLE X — CANDIDATE GOVERNING QUANTITIES ALONGSIDE THE OUTCOMES**

| Benchmark | Answer space | Accuracy | Blind spot | AUC (pooled) | Consensus token share | Δ Token achieved |
| :---- | :---- | ----: | ----: | ----: | ----: | ----: |
| GSM8K | Unbounded integers | **0.779** | 2.9% | **0.875** | 56.0% | −6.0% |
| StrategyQA | Binary | 0.686 | 24.4% | 0.609 | 48.5% | −0.4% |
| MMLU | 4 options | **0.547** | **28.0%** | 0.685 | **68.2%** | **−39.4%** |

Accuracy is the benchmark's accuracy under *always*, reproduced from TABLE VII.

Discrimination (AUC) and operational value are not the same quantity, and TABLE X separates them cleanly. GSM8K yields the highest AUC, 0.875, yet the smallest realised saving. Savings track the *room the incumbent leaves*, given by the consensus token share: MMLU's consensus rule runs to 68.2% of full cost, leaving the most to reclaim, and yields 39.4%; StrategyQA's runs to only 48.5%, and yields nothing distinguishable from zero.

One account of the blind-spot column is informational. Answer entropy over three agents carries at most log₂3 ≈ 1.58 bits, and on a binary task at most 1 bit. When three agents produce the same unbounded integer, coincidence is nearly impossible and agreement approximates proof — hence a 2.9% blind spot and a consensus rule already near-optimal. When three agents select one of two options, agreement occurs about a quarter of the time among guessers — hence a 24.4% blind spot and a consensus premise that is weak precisely where a better signal would have room.

A second account fits the same three points, and on the one pair that discriminates them it fits better. The blind-spot rate is inversely monotone in the accuracy the protocol reaches: 0.779, 0.686 and 0.547 against blind spots of 2.9%, 24.4% and 28.0%. Where the agents are weak they are also wrong together, so agreement carries less information for a reason unrelated to the size of the answer space. The informational account predicts a *larger* blind spot on the binary task than on the four-option one, since three agents coincide 25% of the time on two options against 6.25% on four; the measurement runs the other way, 24.4% against 28.0%. The accuracy account orders that pair correctly.

The two accounts converge on the same practical recommendation — the blind-spot rate is cheap to measure on a pilot run before committing to build a stopping signal at all — and diverge only in what they predict for a benchmark not yet run.

We cannot separate them here. Three benchmarks differing in domain and difficulty as well as in answer space confound answer-space size with task accuracy, and the two are ordered oppositely on the only pair that would distinguish them, so TABLE X is a set of measurements with two candidate mechanisms rather than a controlled result. The experiment that would separate them holds the questions fixed and varies only the answer format, choosing distractor difficulty so that accuracy does not co-vary with the size of the answer space.

### **B. Recommendations for Evaluating Stopping Rules**

The failure in Section V-C was invisible to every metric our original construction suggested reporting: the stop rate looked healthy, the error rate sat inside budget, and the AUC was respectable. Of the four corrections in Section III-E, only one is a detector: simulating sequentially against the incumbent. The other three remove optimism without exposing the circularity. Scoring only non-final rounds leaves the failure intact, since just 2,037 of the 11,754 consensus rounds are the last of their debate, so some seven thousand rounds would still fall below the threshold and the reported stop rate would still look healthy. Relabelling by what each round commits, and moving the error denominator from rounds to debates, shift the reported numbers by fractions of a point in each case. This asymmetry is itself the lesson: three of the corrections make an evaluation honest, and one makes it *diagnostic*. Two further checks belong in the second category. **Report where the rule stops, not only how well it ranks:** TABLE VI exposes the failure immediately where AUC alone does not. And, specific to this family, **check the identity between the dominant feature and the termination condition** — if the score's leading component is an exact indicator of the incumbent's stopping condition, the evaluation is circular before any threshold is swept.

Two further recommendations follow from Section VI-A, and both concern what to do *before* a signal is built rather than how to evaluate one afterwards.

**Benchmark against policies that read nothing, before claiming any benefit from reading.** A learned stopping rule should be reported alongside fixed\_k for every *k* in range, a self-consistency baseline at matched sample count, and an ensemble vote — all at measured, not assumed, token cost. This is a stricter requirement than it appears. Our own TABLE VII originally compared the cascade against consensus and against fixed depth, and passed: it matched consensus at lower cost. It was only when the full fixed-*k* column was placed on the same plane (TABLE 0\) and tested with adequate resolution (TABLE XI) that fixed\_k2 was seen to dominate weakly on two benchmarks of three. A comparison against the incumbent alone cannot detect this, because the incumbent also reads the transcript. The reference class must include policies that do not.

**Report self-consistency per model, never as one number.** TABLE XII is the reason. The three models spread across 29 accuracy points on MMLU alone, from 0.582 to 0.293, and the ranking between them reverses across benchmarks: the model that is best in this paper on MMLU is 14.7 points behind the cascade on GSM8K. A single "self-consistency baseline" row therefore reports whichever model the authors happened to run, and its value carries a standard error, across model choice, larger than any effect a stopping rule is likely to claim. Reporting the family also prices the selection honestly: taking the best cell per benchmark is an argmax over nine, and the one cell that beats our cascade does not survive correction for it.

That the widened reference class *reversed* our conclusion in both directions is the argument for widening it. Against fixed depth the cascade looked unpaid for; against self-consistency it looked well earned; only with both present is the defensible claim visible, which is that it buys worst-case robustness and fixed depth buys most of that more cheaply.

Reporting these baselines also disciplines the cost column. Policies simulated offline from debate logs inherit whatever token approximation those logs permit — here, a uniform total⁄6 per round, which Section IV-D shows is substantially wrong for GSM8K round 0\. A separately run baseline records its own tokens, and the discrepancy between the two is itself diagnostic.

**Use the blind-spot rate as a first-stage screen, and rescue/hurt as a second.** Building a stopping signal is expensive: it requires full-depth logging, feature extraction, model fitting and per-benchmark threshold calibration. Two measurements available from a pilot run predict, on our data, whether that expense is warranted.

The first is the blind-spot rate of TABLE I — how often consensus commits a wrong answer. It bounds what any consensus-improving rule can reclaim. GSM8K's 2.9% leaves almost nothing to fix, and DUS-11 duly gains nothing there (TABLE 0). This screen is nearly free: it needs only agreement and correctness, no learned model.

It is necessary but not sufficient, and StrategyQA shows why. Its blind spot is 24.4%, so the headroom exists — yet DUS-11 gains nothing there either. The reason appears in TABLE IX: on StrategyQA debate rescues 91 answers and corrupts 86, a net of \+5 in 1,500 (*p* \= 0.76). The headroom is real but unreachable, because the only action a stopping rule can take — continue debating — does not on average improve the answer. Hence the second stage: **estimate rescue and hurt on the pilot, and if their difference is indistinguishable from zero, no stopping policy can convert the blind spot into accuracy.** MMLU is the only benchmark of the three that clears both screens (blind spot 28.0%, net \+26, *p* \= 0.074), and it is the only one on which DUS-11 shows any advantage at all.

Applied honestly to our own pilot, this two-stage screen would have advised against building the cascade on two of three benchmarks. We regard that as the recommendation's principal evidence rather than an objection to it.

A third screen is implied by TABLE XII and is cheaper still, because it needs no debate at all: **run self-consistency on each candidate model first**. On MMLU — the one benchmark that clears both screens above — Qwen2.5-3B sampled three times reaches 0.582 against 0.545 for the cascade, at 30.2% fewer tokens. Had that baseline been measured before the debate system was built rather than after, the case for building anything on MMLU would have had to start by beating it. The general form of the recommendation is that the reference class should be measured *before* the method, not assembled afterwards to defend it.

### **C. Limitations**

**Selection over baseline models.** TABLE XII reports three self-consistency baselines per benchmark and the best differs on each. Any claim of the form "the cheap baseline wins" therefore carries a selection cost we have priced but not eliminated: with nine model-benchmark cells, the single cell that beats DUS-11 does not survive correction for the twenty-seven comparisons in that family.

**Solver scale.** Both solvers are 3B-parameter models and the critic is 4B. Absolute accuracies are correspondingly modest (0.547 on MMLU under always). Whether the entropy–consensus identity and the value of critic signals persist at larger scale is untested; the identity itself is structural and should persist, but the relative value of critic confidence may not.

**Single topology.** One topology — two solvers plus a fixed-answer critic — is studied. Round-table \[3\] and cross-model communication \[6\] topologies may distribute uncertainty differently.

**Threshold transfer.** Thresholds *T\_q* are quantiles chosen on held-out data throughout (Section III-E), and Section V-F measures what in-sample selection would have added: at most 3.2 percentage points of Δ Token. What is untested is transfer *across* settings — a threshold calibrated on one benchmark, model pair or answer format and applied to another. Since the score's scale differs across answer spaces, we expect such transfer to require recalibration, and its cost is unmeasured here.

**Residual mass point.** The 872 rounds tied exactly at *T* under our direct protocol reflect a discrete score. Although the corrected score is continuous, quantile thresholds on a partly discrete feature set can still land on ties.

**Uncalibrated score.** DUS-11 is used only for ranking and thresholding. It is not a calibrated probability and should not be read as one.

**One critic.** The critic's independence is central to CRITIC-7. With multiple critics, or a critic permitted to revise its own answer, the signals would need redefinition.

**Resolution of the accuracy comparisons.** The design is well powered for discrimination but not for accuracy: the paired intervals of TABLE VIII span one to three accuracy points, so a genuine cost of about a point would not be detected. Every accuracy difference we report as indistinguishable from zero should be read within that resolution, and the disagreeing holdout estimate on MMLU is the concrete reason not to read it as equality.

## **VII. CONCLUSION**

We set out to stop multi-agent debate early and found, in sequence, that we could not measure whether we had, that a usable signal nevertheless existed, and that it was not worth its price.

The first finding is a circularity. A score built from four natural measures of consensus dynamics reported a large compute saving; simulated sequentially against the consensus rule the system already applies, it removed essentially no rounds at all, because its dominant term is zero exactly where consensus holds. Of the corrections this forces, only sequential simulation against the incumbent is diagnostic — the others make the evaluation honest without exposing the fault.

The second finding survives that correction. Features read off an independent critic, containing no agreement term, discriminate better than the consensus-dynamics features we began with, by a margin that does not vanish once both are present together. We deploy this score as a two-stage cascade — a self-consistency check on the critic's round-0 answer, ahead of the same score rechecked every round of debate for questions it does not resolve at round 0 — and as a stopping policy it matches consensus accuracy at markedly lower token cost.

The remaining two findings qualify the second, and we report them because the evaluation we recommend is the one that produced them. Against fixed depth, the cascade is weakly dominated: halting unconditionally after two rounds is statistically indistinguishable from it in accuracy on every benchmark, under a question-clustered bootstrap corrected for multiple comparisons, while costing less. Against self-consistency the picture reverses: most no-debate baselines lose to the cascade by wide, corrected margins, and the rare baseline that beats it on one benchmark is the worst of its family on the others, so which cheap baseline is competitive cannot be known in advance. What the cascade buys is robustness across benchmarks rather than accuracy on any one of them; what fixed depth shows is that most of that robustness comes free with the debate, and needs no score on top.

The question in our title therefore has an answer with a narrow scope. Adaptive stopping pays when the incumbent terminates late, when consensus is often wrong, and when continuing the debate actually repairs answers. Those conditions coincided on only one of our three benchmarks, and even there a single model sampled repeatedly still matched it; measured on a pilot run before any signal is built, they would have advised against building the cascade on two benchmarks of three.

Four directions remain open: separating the two accounts of Section VI-A by holding questions fixed and varying only the answer format; re-measuring the critic signals at larger model scale and with multiple critics, where the independence they rely on no longer holds by construction; testing whether the weak dominance of fixed depth persists at larger scale, since debate that genuinely improves answers would restore the headroom adaptive stopping needs; and closing the gap that still separates every realisable policy from the oracle — wide enough that the open question is not whether a stopping rule can match consensus cheaply, which fixed depth already does, but what it would have to read in the transcript to do substantially better than any of them.

## **REPRODUCIBILITY**

Complete debate logs for all 4,500 debates (27,000 rounds, 3 benchmarks × 5 seeds × 300 questions) are released, together with both evaluation protocols implemented over identical inputs, the split definitions, and the analysis pipeline that produces every table in this paper. The direct and corrected accountings of TABLE III reproduce in a single invocation.

## **REFERENCES**

\[1\] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, "Improving factuality and reasoning in language models through multiagent debate," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024\.

\[2\] T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, S. Shi, and Z. Tu, "Encouraging divergent thinking in large language models through multi-agent debate," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2024\.

\[3\] J. C.-Y. Chen, S. Saha, and M. Bansal, "ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs," in *Proc. Annu. Meeting Assoc. Computational Linguistics (ACL)*, 2024\.

\[4\] C.-M. Chan, W. Chen, Y. Su, J. Yu, W. Xue, S. Zhang, J. Fu, and Z. Liu, "ChatEval: Towards better LLM-based evaluators through multi-agent debate," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024\.

\[5\] A. Khan, J. Hughes, D. Valentine, L. Ruis, K. Sachan, A. Radhakrishnan, E. Grefenstette, S. R. Bowman, T. Rocktäschel, and E. Perez, "Debating with more persuasive LLMs leads to more truthful answers," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024\.

\[6\] Z. Yin, Q. Sun, C. Chang, Q. Guo, J. Dai, X. Huang, and X. Qiu, "Exchange-of-Thought: Enhancing large language model capabilities through cross-model communication," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023\.

\[7\] T. Hu, Z. Tan, S. Wang, and H. Qu, "Multi-agent debate for LLM judges with adaptive stability detection," *arXiv preprint*, 2025\.

\[8\] F. Haji, M. Bethany, and M. Tabar, "Improving LLM reasoning with multi-agent tree-of-thought validator agent," *arXiv preprint*, 2024\.

\[9\] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language agents with verbal reinforcement learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023\.

\[10\] Z. Wu, Q. Zeng, Z. Zhang, Z. Tan, C. Shen, and M. Jiang, "Large language models can self-correct with key condition verification," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2024\.

\[11\] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang, "Large language model based multi-agents: A survey of progress and challenges," in *Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2024\.

\[12\] K.-T. Tran, D. Dao, M.-D. Nguyen, Q.-V. Pham, B. O'Sullivan, and H. D. Nguyen, "Multi-agent collaboration mechanisms: A survey of LLMs," *arXiv preprint*, 2025\.

\[13\] S. Chen, Y. Liu, W. Han, W. Zhang, and T. Liu, "A survey on LLM-based multi-agent system: Recent advances and new frontiers in application," *arXiv preprint*, 2024\.

\[14\] W. X. Zhao, K. Zhou, J. Li, and T. Tang, "A survey of large language models," *Frontiers of Computer Science*, 2026\.

\[15\] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou, "Self-consistency improves chain of thought reasoning in language models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2023\.

[image1]: <figures/fig_architecture.png>
[image2]: <figures/fig2_circularity.png>
[image3]: <figures/fig3_cost_accuracy.png>
[image4]: <figures/fig4_cascade_vs_sc_frontier.png>