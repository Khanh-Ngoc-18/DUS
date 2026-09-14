# **When Does Adaptive Stopping Pay in Multi-Agent Debate? A Circularity Pitfall, Critic-Derived Signals, and a Fixed-Depth Baseline That Weakly Dominates**

**Abstract** — Multi-Agent Debate (MAD) improves LLM reliability at a cost that scales with agents times rounds, motivating early stopping once a debate's outcome is settled. We show that the standard way of reporting such savings cannot distinguish a rule that removes rounds from one that does not: a score built from four consensus-dynamics signals reports a **33.9%** saving over 4,500 debates, yet simulated sequentially against the consensus rule the system already runs, it removes **0.0%** of executed rounds, because its dominant signal, answer entropy, is zero exactly where consensus already holds. We give a corrected evaluation protocol and apply it to a signal that survives: seven critic-derived features, containing no agreement term, reach ROC-AUC **0.729** against **0.705** for the consensus signals. Deployed as a two-stage cascade, this cuts the token cost incurred by the consensus baseline by **41.1%** (reducing the absolute budget from 68.2% to 40.2%) on MMLU at no measurable accuracy loss. Benchmarked against twelve no-debate and fixed-depth baselines, the cascade is statistically indistinguishable from unconditional two-round stopping on all three benchmarks at higher cost, yet beats most self-consistency baselines by wide margins and carries the lowest worst-case regret. Adaptive stopping buys robustness across benchmarks rather than accuracy on any one, and fixed depth supplies most of that robustness more cheaply. We give three measurable conditions under which adaptive stopping pays off, and release all logs, protocols, and analysis code.

**Keywords** — multi-agent debate, large language models, **Debate Uncertainty**, **Critic Self-Consistency**, adaptive inference, critic-derived signals, computational efficiency, baseline evaluation

---

## **I. INTRODUCTION**

Multi-Agent Debate (MAD) improves the factuality and reasoning reliability of large language models by having several agents answer independently, critique one another over several rounds, and converge on a shared answer \[1\], \[2\], \[3\]. The premise is that idiosyncratic errors cancel across agents while sound reasoning survives scrutiny — but the gains carry a cost that scales with agents × rounds: each round here issues four model calls, so running a debate to six rounds costs several times what answering it once does. The opportunity is real: **90.6%** of non-final rounds already commit the verdict the debate eventually reaches, and across 4,500 debates the net change in accuracy from debating is indistinguishable from zero on every benchmark (TABLE IX). Where a debate stops is therefore an accuracy decision as much as a budget one.

This has motivated stopping rules that monitor the transcript and terminate once an uncertainty criterion is met \[7\]. What such work reports as savings, however, is typically measured against running the protocol to full depth or against the fraction of rounds flagged — neither of which a deployed system would actually save, since debate systems already halt on consensus. A rule that fires at or after the consensus round saves nothing however many rounds it flags, so an early-stopping method must be shown to stop earlier than the incumbent. Two questions follow: how should a stopping rule be evaluated against that incumbent, and, once evaluated correctly, does any signal still permit earlier termination without sacrificing accuracy?

We first designed a Debate Uncertainty Score from the most natural construction — four signals describing how the three agents' answers agree and move — and it reported a **33.9%** saving. Simulated sequentially against the incumbent, it saved **0.0%**: its dominant term, answer entropy, is zero exactly where consensus already holds, so its stop region sits entirely inside rounds the system would already have ended. The corrected protocol this forces has four parts: score only non-final rounds, label each round by what it would itself commit, simulate sequentially against the incumbent, and resample by question — only the third is diagnostic, but all four remove optimism from the estimate.

Under that protocol a usable signal does survive, and it comes from outside the agreement structure: seven features read off an independent critic, containing no agreement term, discriminate better than the four consensus-dynamics features we began with. We deploy them as a two-stage cascade — a self-consistency check on the critic's round-0 answer, followed by the same score rechecked every round for debates that continue — which cuts token cost against consensus by **41.1%** (reducing the absolute budget from 68.2% to 40.2%) on MMLU at no measurable accuracy loss, while largely reproducing the incumbent on GSM8K and StrategyQA, where consensus stopping is already close to optimal.

The contributions of this paper are:

1. **A measurement protocol for stopping rules.** Standard reporting cannot distinguish a rule that removes rounds from one that does not. We give a four-part protocol — non-final rounds only, per-round labels, sequential simulation against the incumbent, and question-level resampling — and show only sequential simulation is diagnostic, using our own first score as the case that exposes why.
2. **DUS-11 and the critic-SC cascade.** Seven critic-derived features containing no agreement term discriminate better than the consensus-dynamics signals, and adding those signals back adds nothing detectable. We deploy them as a two-stage cascade: a round-0 critic self-consistency gate, then the same score rechecked every round.
3. **What the cascade is worth against consensus**, not against a fixed depth — where the rule stops relative to the incumbent, what that costs in tokens and accuracy, and the condition that separates the benchmark where it pays from those where it does not.
4. **A comparison against baselines that read nothing at all.** The cascade is weakly dominated by fixed depth on two of three benchmarks; we give the three conditions under which adaptive stopping does pay off.

## **II. RELATED WORK**

**Multi-agent debate.** Debate-style prompting improves factuality and arithmetic reasoning through iterative cross-agent critique \[1\]. Follow-on work explores divergent-thinking prompts \[2\], confidence-weighted round-table voting \[3\], structured communication topologies \[6\], debate for eliciting truthful answers from stronger models \[5\], and debate as a judge \[4\]. These systems fix agent/round counts in advance or halt on unanimous agreement, and are assessed on the accuracy the protocol reaches.

**Early termination of debate.** The closest prior work \[7\] models the number of correct decisions among *k* judges per round as a time-varying Beta-Binomial mixture and halts once consecutive rounds' fitted distributions converge (KS statistic below a threshold); with seven judges and a ten-round cap it converges in four to eight rounds while changing accuracy by 0.1–0.6 points. Three differences position this paper: it estimates one stopping round per evaluation run rather than per debate; it measures efficiency against the full-depth run rather than against a rule that would already have halted; and it monitors judge-consensus dynamics converging toward unanimity — a structure on which a signal risks coinciding with consensus stopping rather than improving on it. Section V-C reports exactly such a coincidence in a score of our own design.

**Verification and self-correction.** A separate line adds a verifier — a validator agent over a reasoning tree \[8\], verbal self-reflection \[9\], or explicit key-condition verification \[10\]. In each, the verifier's verdict already serves as a binary stopping condition rather than a graded quantity tradeable against a cost budget.

**Self-consistency and sampling.** Sampling several independent chains and majority-voting \[15\] is the natural zero-debate reference point, since diversity comes from independent sampling rather than cross-agent critique. It is tempting to read a debate's round 0 as that reference point, but Section IV-D shows why we run it separately; TABLE 0 reports both.

The literature refines debate protocols \[1\]–\[6\], proposes stopping criteria and reports rounds saved \[7\], adds verifiers \[8\]–\[10\], and surveys all three \[11\]–\[14\]. Two gaps remain: how a stopping rule should be evaluated against the termination mechanism already in place, and whether a signal not tied to agreement — as the verification line suggests a critic could supply — can support the stopping decision. This paper takes up the first with a corrected evaluation protocol and the second with critic-derived features containing no agreement term.

## **III. METHODOLOGY**

### **A. Problem Formulation**

A debate on question *q* produces rounds *r* = 0, 1, …, R−1, each with a committed answer *a\_r* (the system's aggregation rule over the three agents' current answers) and outcome *y\_r* = 1 if *a\_r* matches ground truth. A **stopping rule** observes rounds 0…*r* and decides whether to halt and commit *a\_r*.

Two reference rules: **consensus stopping** (the incumbent) halts at the first round where all three agents agree; **fixed-*k*** halts unconditionally after *k* rounds. An uncertainty-score rule halts at the first round where *u\_r* < *T*. The quantity a stopping rule must be judged on is how many rounds it removes *relative to the rule already in use*, not how many it flags (Section III-E).

### **B. System and Debate Protocol**

Three agents participate: two solvers and one critic (Fig. 1 shows stage 2, the debate proper; stage 1 is described below). At round *r*, solver A then solver B answer, each shown the critic's messages from rounds 0…*r*−1. The critic then issues a verdict with confidence on each solver's current answer — four model calls per round. Critique from round *r* is consumed only at round *r*+1; a solver never revises in the round it is criticised. The critic's own independent answer is generated once, at round 0, and held fixed thereafter.

![][image1]

**Fig. 1.** Stage-2 architecture and call order within a round. The stage-1 gate that precedes it is described below.

The committed answer *a\_r* is the majority of the three current answers, ties broken by highest confidence. Consensus holds when all three answers are identical.

**The stopping rule is a two-stage cascade.** **Stage 1 (round 0):** before round-0 features are computed, the critic's single cached answer is replaced by the majority of three independent self-consistency samples, with mean confidence of the samples agreeing with the majority; if the resulting score already clears threshold *T*, the debate halts before either solver answers again. **Stage 2:** a debate that does not clear stage 1 proceeds through the solver/critic rounds above, with the same score rechecked every round. We call this two-stage rule the **cascade**; "DUS-11" names the per-round score it evaluates at both stages. Section V-A confirms the stage-1 gate is stable across seeds before any downstream result is reported.

**Consensus-based early stopping is disabled for all experiments**, so every debate runs the full six rounds — reconstructing the standard system's behaviour exactly (its trajectory is the logged trajectory's prefix up to its first consensus round) while retaining later rounds as counterfactual data for the sequential simulation of Section III-E.

### **C. Uncertainty Signals**

Every round writes one record per agent: raw answer, normalised answer, self-reported confidence (clipped to [0,1], defaulting to 0.5 when absent), and reasoning text; the critic's record additionally carries its verdict and confidence on each solver. All eleven signals below are computed offline from these records, at no extra model-call cost. Because the critic's answer is cached, critic-family signals 2, 4, 7 change only when the solvers move, and signal 5 (critic\_conf) is constant across a debate by construction; at round 0 the stage-1 gate substitutes a self-consistency majority before any signal is computed.

**Consensus-dynamics features (DUS-4)** — computed from the pattern of agreement among the three answers:

| \# | Signal | Definition | Range |
| :---- | :---- | :---- | :---- |
| 1 | answer\_entropy | Shannon entropy (base 2) over {*a*ᴬ, *a*ᴮ, *a*ᶜ} | {0, 0.918, 1.585} |
| 2 | confidence\_variance | Population variance of {*c*ᴬ, *c*ᴮ, *c*ᶜ} | [0, 2/9] |
| 3 | disagreement\_persistence | Cumulative count of rounds 0…*r* without consensus | {0, 1, …} |
| 4 | answer\_flip\_rate | Fraction of agents whose answer changed since *r*−1 (0 at *r*=0) | {0, ⅓, ⅔, 1} |

Signal 1 is decisive: it takes exactly three values, and 0 occurs precisely when consensus holds. A score weighting it heavily inherits the incumbent's stopping condition rather than adding to it (Section V-C).

**Critic-derived features (CRITIC-7)** — read only the critic's own state and verdicts; **none is a function of solver–solver agreement**:

| \# | Signal | Definition | Range |
| :---- | :---- | :---- | :---- |
| 1 | verdict\_conf\_mean | (*v*ᴬ + *v*ᴮ) / 2 | [0, 1] |
| 2 | n\_disagree | \|{*k*∈{A,B} : *a*ᵏ ≠ *a*ᶜ}\| | {0, 1, 2} |
| 3 | verdict\_conf\_min | min(*v*ᴬ, *v*ᴮ) | [0, 1] |
| 4 | critic\_vs\_majority | 1 if *a*ᶜ occurs exactly once among {*a*ᴬ,*a*ᴮ,*a*ᶜ} | {0, 1} |
| 5 | critic\_conf | *c*ᶜ | [0, 1] |
| 6 | conf\_gap\_critic\_solvers | *c*ᶜ − (*c*ᴬ + *c*ᴮ)/2 | [−1, 1] |
| 7 | critic\_alone | 1 if *a*ᴬ = *a*ᴮ and *a*ᶜ ≠ *a*ᴬ | {0, 1} |

Signal 7 is a strict subset of signal 4 (they differ only when all three disagree). Signals 1 and 3 read the critic's *judgement*; the rest read its answer or confidence in it. We call the four consensus features **DUS-4**, the seven critic features **CRITIC-7**, and their union **DUS-11**.

### **D. Score Aggregation**

**Model 1 (audited, not used after Section V-C).** Features standardised on the training split; logistic regression fit against "final answer wrong"; non-positive coefficients clipped to zero; surviving weights normalised to sum to one. This produced the score whose evaluation fails in Section V-C.

**Model 2 (used from Section V-D onward).** Logistic model fit over standardised DUS-11 features against "the answer *this round* would commit is wrong"; its linear predictor is the score *u\_r*. No clipping — the inverse-signed critic signals carry real discrimination. Fit only on non-final rounds.

### **E. Evaluation Protocols and Threshold Selection**

**Direct protocol (flawed).** DUS-4 weights fit on the training split; every round of every debate (training included) is scored and labelled by the debate's *final* correctness; threshold *T* is swept on that same set to maximise the flagged fraction subject to (wrong ∧ flagged)/(all rounds) ≤ 5%, and that fraction is reported as saved. The sweep runs on the same rounds it evaluates, and on every round rather than only rounds where stopping could elide anything.

**Corrected protocol.** (1) Score only non-final rounds. (2) Label each round by what it itself commits, *y\_r*. (3) Report saving by sequential simulation: rounds actually elided are Σ max(0, *i\_c* − *i\_u*) per debate, where *i\_c*, *i\_u* are the consensus and rule stopping indices — a rule firing only at or after *i\_c* saves nothing. (4) Compute error rates/CIs per debate, resampling by question.

**Threshold selection.** We separate fitting, threshold selection, and measurement into three disjoint sets, under two instantiations: **nested cross-validation (primary)** — five outer folds by question, inner 75/25 split for fitting/threshold, every debate evaluated once out-of-fold; and **fixed 70/20/10 holdout (confirmatory)** — cleanest separation but only ~150 debates/benchmark, wide intervals. Thresholds are chosen per benchmark. Both protocols run over identical inputs and are released together.

## **IV. EXPERIMENTAL SETUP**

### **A. Models and Benchmarks**

All three agents are open-weight instruction-tuned models served locally for exact token accounting.

| Role | Model | Temperature |
| :---- | :---- | :---- |
| Solver A | qwen2.5:3b-instruct | 0.3 |
| Solver B | llama3.2:3b | 0.3 |
| Critic | gemma3:4b | 0.2 |

Three benchmarks vary the answer-space size while holding the protocol fixed (Section VI-A tracks this quantity):

| Benchmark | Reasoning type | Answer space |
| :---- | :---- | :---- |
| GSM8K | Mathematical | Unbounded integers |
| MMLU | General knowledge | 4 options |
| StrategyQA | Multi-hop | Binary |

### **B. Run Configuration**

3 benchmarks × 5 seeds × 300 questions, max\_rounds = 6, consensus stopping **disabled** → **4,500 debates**, **27,000 rounds**, of which **22,500** are non-final. Non-final error rate: **32.9%**; corrected-label base rate: **0.326**.

| Item | Value |
| :---- | :---- |
| Accelerator | NVIDIA Tesla T4, 16 GB VRAM (Colab) |
| Serving | Ollama HTTP generate API |
| Weights | GGUF Q4\_K\_M 4-bit, all three models |
| Decoding | temp 0.3 (solvers) / 0.2 (critic), top\_p 1.0 |
| Output cap | 1024 tokens (solvers), 768 (critic) |
| Seed control | run seed fixes both sampler and question draw |

**Cost is measured in tokens**, not wall-clock/energy/price. Logs record one token total per debate; a policy stopping after *i*+1 of *R* rounds is charged (*i*+1)/*R* of that total, uniformly across all policies including baselines.

### **C. Data Splits and Statistical Treatment**

Splits are 70/20/10 at the **question** level (train: 18,978 rounds; question-level sizes 685/196/97 GSM8K, 786/225/112 MMLU, 453/129/65 StrategyQA). Cross-validation, where used, is 5-fold grouped by question.

| Reported quantity | Fitted on | Threshold on | Measured on |
| :---- | :---- | :---- | :---- |
| TABLE III, Fig. 2 (direct protocol) | train | swept over all rounds | all rounds |
| TABLE IV, V (discrimination) | train | — | val + test |
| TABLE VI–VIII (stopping policies) | inner-train per fold | inner-val per fold | held-out fold |
| Confirmatory holdout | train | validation | test |
| Pooled AUC, TABLE X | out-of-fold, 5-fold | — | all rounds |

All CIs are bootstrap, 2,000 resamples, clustered by question. A power analysis over 22,500 non-final rounds (2,748 distinct questions) gives power 0.958 at 65 questions, 0.992 at 100, 1.000 from 200 — sufficient for ROC-AUC displacement from chance. It does **not** transfer to accuracy comparisons: the paired intervals of TABLE VIII span one to three accuracy points at 1,500 debates/benchmark, so an accuracy difference reported as indistinguishable from zero is *below this design's resolution*, not demonstrated equality.

### **D. Baselines**

*Simulated from logs:* **always** (all six rounds, normalised to 100% tokens); **consensus** (the incumbent); **fixed\_k** for *k* = 1…5; **oracle** (stops at the first correct committed round — an unachievable upper bound).

*Run separately:* **majority voting** (three models, one independent draw each, ties by confidence, symmetric prompt); **self-consistency@3** \[15\] (one model, three draws, majority vote — reported per model).

**On fixed\_k1 vs. majority voting.** fixed\_k1 (the "round-0 ensemble vote") and the separately run majority vote are conceptually the same policy but not the same *measurement*: fixed\_k1 is asymmetric (two solvers at temp 0.3/1024 tokens, the critic at 0.2/768), its prompt overlaps the shared independent prompt by only 2.7% of characters on StrategyQA, it costs roughly seven generations on GSM8K (each solver internally self-consistency-votes there) against three elsewhere, and its per-round cost uses the uniform total/6 approximation, which is wrong for GSM8K round 0. The two report gaps up to 6.7 points on the same benchmark; TABLE 0 reports both, and we read the interval between them as this baseline's honest uncertainty. Every claim about self-consistency in this paper rests on the separately run baseline, never on fixed\_k1.

### **E. Evaluation Metrics**

*Does the score know which rounds are unsafe to stop at?* **ROC-AUC** over non-final rounds (positive class: "this round's committed answer is wrong"); chance is 0.500, displacement in either direction is signal; bootstrap 95% CIs clustered by question.

*Does stopping on it help?* **Accuracy** (fraction of debates correct at the stopping round); **Token cost** (% of running all six rounds); **Δ Accuracy / Δ Token vs. consensus** (paired per seed, then averaged); **Earlier/Same/Later** (share of debates stopping before/at/after the consensus round); **Error rate** (share of debates where the rule fires on a wrong answer).

The primary metric is the paired comparison against consensus: Δ Token negative, Δ Accuracy indistinguishable from zero. ROC-AUC diagnoses the signal, not the saving — Section V-C shows a respectable AUC with zero saving, Section VI-A the highest AUC with the smallest saving.

## **V. RESULTS**

TABLE 0 places every policy — adaptive and non-adaptive — on one accuracy/cost plane; it governs how the rest of the paper should be read: on two of three benchmarks a rule that ignores the transcript entirely is not distinguishable in accuracy from the cascade, and costs less.

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
| DUS-11, *q* = 0.4 | yes | 0.783 / 62.6% | 0.545 / 48.9% | 0.687 / 58.3% |
| DUS-11, *q* = 0.5 | yes | 0.786 / 52.2% | 0.547 / 40.2% | 0.694 / 47.5% |

Self-consistency@3 is reported per model — the three disagree by up to 29 points on the same benchmark, and no one wins more than one.

Five readings: **(1)** among policies that run the debate at all, none separates from another by more than ~2 points on any benchmark, while the oracle sits 7–14 points above all of them. **(2)** among no-debate policies the spread is an order of magnitude larger (up to 29 points) — which cheap baseline to run matters far more than what to do with the transcript. **(3)** the cheapest policies are competitive only if the right one is chosen, and the right one changes with the benchmark: Gemma3-4B is within a point of every debate policy on GSM8K/StrategyQA at ~⅕ the cost, but worst on MMLU, where Qwen2.5-3B — worst everywhere else — is best. **(4)** MMLU is the only benchmark where accuracy rises monotonically with *k*; elsewhere fixed-*k* is a random walk of amplitude comparable to seed noise. **(5)** the same nominal policy measured two ways (fixed\_k1 vs. symmetric majority vote) differs by up to 6.7 points — implementation detail, not chance (Section IV-D).

### **A. Critic-SC Stability**

The stage-1 gate's answer flips in 5.2–9.9% of debates going from one self-consistency vote to three; the fraction of debates it fires at round 0 is stable across seeds (CV 1.9–10.5%, below the pre-registered 15%/25% bar).

**TABLE 0-A — CRITIC-SC GATE STABILITY**

| Benchmark | Flip rate, 1→3 votes | Round-0 gate rate (mean ± SD, 5 seeds) | CV |
| :---- | ----: | ----: | ----: |
| GSM8K | 7.5% | 54.5% ± 5.2pp | 9.5% |
| MMLU | 9.9% | 65.1% ± 1.3pp | 1.9% |
| StrategyQA | 5.2% | 52.0% ± 5.5pp | 10.5% |

Every "DUS-11"/"unc<q*" result below is therefore the full cascade (stage 1 + stage 2), not the score alone.

### **B. Measured Research Gaps**

**Fixed rounds are wasteful:** of 22,500 non-final rounds, **90.6%** commit the same verdict the debate eventually reaches. **Consensus does not imply correctness:** of 11,754 rounds where all three agree, **17.2%** agree on a wrong answer, varying sharply with answer-space size:

**TABLE I — BLIND-SPOT RATE BY BENCHMARK**

| Benchmark | Answer space | Blind-spot rate |
| :---- | :---- | ----: |
| GSM8K | Unbounded integers | **2.9%** |
| StrategyQA | Binary | **24.4%** |
| MMLU | 4 options | **28.0%** |
| **All** | — | **17.2%** (2,020 / 11,754) |

Chance agreement alone does not order the two bounded spaces (4-option coincidence is 6.25%, a quarter of binary's 25%, yet MMLU has the higher blind spot) — so TABLE I separates unbounded from small answer spaces; Section VI-A takes up what else orders the small ones.

### **C. Auditing Our Own Result: The Direct Protocol Is Circular**

Fitting DUS-4 concentrates weight on one feature — answer\_entropy takes **0.638**, confidence\_variance is clipped to zero — and the direct protocol reports **33.9%** saved at 4.85% per-round error, inside budget.

**TABLE II — LEARNED DUS-4 WEIGHTS**

| Feature | Raw coefficient | Final weight |
| :---- | ----: | ----: |
| answer\_entropy | 0.638 | **0.638** |
| answer\_flip\_rate | 0.194 | 0.194 |
| disagreement\_persistence | 0.167 | 0.168 |
| confidence\_variance | −0.043 | **0.000** (clipped) |

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

**Fig. 2.** Uncertainty score by consensus status, with the threshold the direct protocol selects.

The reported saving was entirely illusory: answer\_entropy is exactly zero whenever consensus holds, so the direct protocol's threshold happens to sit exactly at the boundary consensus already enforces — the score flags rounds consensus has already ended.

### **D. Discrimination Under the Corrected Protocol**

Scoring only non-final rounds, labelled by what each would itself commit:

**TABLE IV — SINGLE-FEATURE ROC-AUC (22,500 NON-FINAL ROUNDS)**

| Feature | Family | AUC | \|AUC − 0.5\| |
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

The 2nd- and 3rd-strongest individual signals are critic-derived (verdict\_conf\_mean 0.191, n\_disagree 0.190), matching answer\_entropy's 0.194; confidence\_variance, one of our own four, is nearly uninformative (0.533), consistent with its clipping in TABLE II.

**TABLE V — MODEL COMPARISON (BOOTSTRAP CI CLUSTERED BY QUESTION)**

| Model | Features | ROC-AUC | 95% CI |
| :---- | ----: | ----: | :---- |
| DUS-4 — consensus dynamics | 4 | 0.705 | [0.671, 0.737] |
| **CRITIC-7 — critic signals alone** | 7 | **0.729** | [0.703, 0.757] |
| **DUS-11 — both families** | 11 | **0.736** | [0.707, 0.764] |
| DUS-11, 5-fold CV grouped by question | 11 | **0.738** | [0.720, 0.754] |

**TABLE Va — PAIRED ΔAUC (BOOTSTRAP, 2,000 RESAMPLES, CLUSTERED BY QUESTION)**

| Comparison | Δ AUC | 95% CI | *p* (two-sided) | Excludes zero |
| :---- | ----: | :---- | ----: | :---- |
| **CRITIC-7 − DUS-4** | **+0.025** | [+0.007, +0.042] | 0.002 | **Yes** |
| DUS-11 − CRITIC-7 | +0.007 | [−0.004, +0.017] | 0.202 | No |
| DUS-11 − DUS-4 | +0.031 | [+0.021, +0.043] | <0.001 | Yes |

DUS-4 and CRITIC-7's marginal intervals overlap, but marginal intervals from models scored on the same correlated questions settle nothing; bootstrapping the **paired difference** instead (**TABLE Va**, 2,000 resamples), **CRITIC-7 beats DUS-4, 0.729 vs. 0.705, interval excluding zero** — the central consequence of Section V-C: since the agreement-based family is confounded with the termination condition, a signal permitting genuinely earlier stopping must come from outside it. Adding DUS-4 back on top of CRITIC-7 moves AUC by 0.007, interval **including zero**: the consensus features are not merely confounded, they are redundant once critic features are present. We carry DUS-11 forward as the superset, though a deployment could use CRITIC-7 alone at no measured cost. Within CRITIC-7, signals 1–3 carry most of the weight; 6 and 7 add almost nothing.

Discrimination varies sharply by benchmark and is ordered by answer-space size, not blind-spot rate: 0.882 (GSM8K, unbounded), 0.705 (MMLU, 4 options), 0.590 (StrategyQA, binary) — StrategyQA has the *lower* blind spot of the two bounded benchmarks and the lower AUC (Section VI-A returns to this).

### **E. Stopping Behaviour Relative to Consensus**

Because every debate ran the full six rounds, *i\_u* and *i\_c* are directly comparable on all 4,500 debates — a measurement missing from prior evaluations.

**TABLE VI — WHEN DUS-11 STOPS RELATIVE TO CONSENSUS**

| Benchmark | q | Earlier (*i\_u* < *i\_c*) | Same round | Later |
| :---- | :---- | ----: | ----: | ----: |
| GSM8K | 0.5 | 208 · 13.9% | **1,147 · 76.5%** | 145 · 9.7% |
| StrategyQA | 0.5 | 159 · 10.6% | **1,133 · 75.5%** | 208 · 13.9% |
| MMLU | 0.4 | **645 · 43.0%** | 783 · 52.2% | 72 · 4.8% |
| **MMLU** | **0.5** | **868 · 57.9%** | 612 · 40.8% | 20 · 1.3% |

Where DUS-4/direct-protocol stopped earlier in 0 of 4,500 debates, DUS-11 stops earlier in **57.9%** of MMLU debates but lands on the consensus round ~¾ of the time on GSM8K/StrategyQA — an adaptive rule departs from consensus exactly where consensus is least trustworthy, predicting TABLE VIII below.

### **F. Cost Reduction at Unchanged Accuracy**

**TABLE VII — POLICY COMPARISON (ACCURACY / TOKEN %, MEAN OVER 5 SEEDS, HELD-OUT THRESHOLDS)**

| Policy | GSM8K | MMLU | StrategyQA |
| :---- | :---- | :---- | :---- |
| always (6 rounds) | 0.779 / 100.0% | 0.547 / 100.0% | 0.686 / 100.0% |
| consensus | 0.785 / 56.0% | 0.546 / 68.2% | 0.691 / 48.5% |
| round-0 ensemble vote (fixed\_k1) | 0.784 / 16.7% | 0.530 / 16.7% | 0.683 / 16.7% |
| fixed\_k2 | **0.797** / 33.3% | 0.540 / 33.3% | **0.698** / 33.3% |
| fixed\_k3 | 0.783 / 50.0% | 0.550 / 50.0% | 0.691 / 50.0% |
| unc<q30 | 0.780 / 70.6% | 0.553 / 62.8% | 0.685 / 67.5% |
| unc<q40 | 0.783 / 62.6% | 0.545 / **49.0%** | 0.687 / 58.3% |
| **unc<q50** | 0.786 / **52.1%** | 0.547 / **40.2%** | 0.694 / **47.5%** |
| oracle (unachievable) | 0.863 / 32.7% | 0.681 / 50.6% | 0.787 / 38.0% |

![][image3]

**Fig. 3.** Every policy on the two axes the problem has, by benchmark. Top row: token cost as % of the six-round run (dotted line = DUS-11 at *q*=0.5). Bottom row: Δ accuracy against that same setting, with 95% bootstrap CIs, shaded band spanning zero to the oracle's headroom.

Fixed depth is pinned to *k*=2 in the figure — the shallowest depth containing an actual round of debate (*k*=1 is a no-debate vote), and the cheapest deeper-*k* comparison remains flat or loses except on MMLU (TABLE VII-A):

**TABLE VII-A — FIXED-K (*k*=3,4,5) PAIRED AGAINST FIXED\_K2** (Bonferroni-corrected over 9 comparisons)

| Benchmark | fixed\_k3 − fixed\_k2 | *p* | fixed\_k4 − fixed\_k2 | *p* | fixed\_k5 − fixed\_k2 | *p* |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| GSM8K | −0.014 [−0.028, +0.000] | 0.058 | −0.017 [−0.031, −0.003] | 0.016\* | −0.003 [−0.016, +0.009] | 0.642 |
| MMLU | +0.010 [−0.007, +0.027] | 0.247 | +0.017 [+0.001, +0.032] | 0.038\* | +0.023 [+0.007, +0.040] | 0.004\*\* |
| StrategyQA | −0.007 [−0.021, +0.008] | 0.404 | −0.021 [−0.037, −0.005] | 0.009\* | −0.009 [−0.024, +0.005] | 0.231 |

\*\* survives Bonferroni (*p*<0.0056); \* nominal only.

**TABLE VIII — PAIRED COMPARISON AGAINST CONSENSUS STOPPING** (nested CV, held-out thresholds)

| Benchmark | q | Δ Accuracy (mean ± SD) | 95% CI | Δ Token |
| :---- | :---- | ----: | ----- | ----: |
| GSM8K | 0.5 | +0.001 ± 0.003 | [−0.007, +0.009] | **−6.8% ± 4.7%** |
| **MMLU** | 0.5 | +0.001 ± 0.019 | [−0.014, +0.015] | **−41.1% ± 2.2%** |
| StrategyQA | 0.5 | +0.003 ± 0.008 | [−0.005, +0.012] | −1.9% ± 7.6% |

Accuracy does not move: all four differences lie between −0.001 and +0.003, all intervals cover zero — across `always`, `consensus`, all five fixed depths, and both DUS settings on all three benchmarks, no bar in Fig. 3's bottom row exceeds two accuracy points. Within the debate families, stopping policy is a cost choice, not an accuracy one: consensus already runs at 48.5–68.2% of budget, and on MMLU DUS-11 removes a further **41.1%** of what consensus spends at Δ accuracy +0.001. On GSM8K the 6.8% reduction is within its own seed spread (4.7%); on StrategyQA the 1.9% reduction is well inside it — no real saving on either.

Two caveats: fixed\_k2 (0.797 GSM8K, 0.698 StrategyQA at 33.3% tokens) beats every adaptive policy on both axes where DUS-11 shows no saving. And the oracle gap (0.681 at 50.6% tokens on MMLU vs. our 0.547 at 40.2%) is far wider than the gap between our signal and the incumbent.

Under the fixed 70/20/10 holdout, the MMLU token reduction reproduces (−41.7% ± 12.7%), but the accuracy difference is −0.016, CI [−0.075, +0.031] over 151 test debates — an estimate resting on ~30 debates/seed on an easier test subset, which the primary 1,500-debate protocol does not show. We take the nested-CV figure as primary on sample size but record this as the strongest caveat on the MMLU result. Re-selecting *T* from the evaluation fold (in-sample) shifts Δ Token by at most 3.7pp and Δ Accuracy by at most 0.006, with inconsistent sign — a real but small distortion, of a different order from Section V-C's structural defect.

**Where does the token saving come from?** DUS-11 at *q*=0.5 bundles two independent gates: **Stage 1**, a one-shot check at Round 0 (stop immediately if the round-0 uncertainty already clears threshold), and **Stage 2**, the same threshold re-applied at every round from 1 onward for debates that survive Stage 1. We isolate each by re-running the policy with only one gate active (the other left off, consensus as the fallback when the active gate never fires), 5-fold nested CV, RNG=0:

**TABLE VIII-A — CASCADE COMPONENT ABLATION AT unc\<q50 (mean over 5 folds, held-out thresholds)**

| Benchmark | Variant | Acc | Token % | Share of earlier-than-consensus stops\* |
| :---- | :---- | ----: | ----: | ----: |
| GSM8K | Stage1-only (Round-0 gate) | 0.787 | **52.1%** | **83.8%** |
| GSM8K | Stage2-only (Round≥1 gate) | 0.788 | 65.0% | 16.2% |
| GSM8K | Full (both gates) | 0.789 | 61.3% | 100% (65.0% of debates stop early) |
| MMLU | Stage1-only (Round-0 gate) | 0.543 | **50.0%** | **81.6%** |
| MMLU | Stage2-only (Round≥1 gate) | 0.552 | 67.0% | 18.4% |
| MMLU | Full (both gates) | 0.549 | 52.0% | 100% (79.9% of debates stop early) |
| StrategyQA | Stage1-only (Round-0 gate) | 0.695 | **41.7%** | **73.3%** |
| StrategyQA | Stage2-only (Round≥1 gate) | 0.691 | 55.1% | 26.7% |
| StrategyQA | Full (both gates) | 0.695 | 50.1% | 100% (70.9% of debates stop early) |

\*Decomposition of TABLE VI's "earlier than consensus" share by which stage caused the early stop (Section V-E); the two shares sum to that debate's earlier-than-consensus rate.

Stage 1 alone accounts for **73.3–83.8%** of every debate that stops earlier than consensus, on all three benchmarks, and Stage1-only in isolation already matches or beats Full's token cost — 52.1% vs. Full's 61.3% on GSM8K, 50.0% vs. 52.0% on MMLU, 41.7% vs. 50.1% on StrategyQA — at accuracy indistinguishable from Full (≤0.006 apart). Stage2-only, run without the Round-0 gate, lands within 2pp of the six-round `always` baseline's typical consensus cost and is the *most expensive* of the three variants on all three benchmarks. **The saving in TABLE VII/VIII is a Round-0 phenomenon, not an accumulation of small mid-debate stops:** a single cheap look at the ensemble's initial disagreement does essentially all of the work; continuing to monitor uncertainty after Round 1 adds cost more often than it removes it. This also explains why Full is not strictly cheaper than Stage1-only alone (61.3% vs. 52.1% on GSM8K): Full does not cap its search at the consensus round — absent a Round-0 stop it keeps checking Rounds 1–5 for a threshold breach — and can therefore run past the point where falling back to consensus outright (Stage1-only's behaviour) would have stopped.

### **G. Is Debate Worth Its Cost?**

**TABLE IX — RESCUE AND HURT OVER SIX ROUNDS**

| Benchmark | n | Rescued | Corrupted | Net | McNemar *p* |
| :---- | ----: | ----: | ----: | ----: | ----: |
| GSM8K | 1,500 | 54 · 3.6% | 62 · 4.1% | **−8** | 0.516 |
| MMLU | 1,500 | 111 · 7.4% | 85 · 5.7% | +26 | 0.074 |
| StrategyQA | 1,500 | 91 · 6.1% | 86 · 5.7% | +5 | 0.764 |
| **Total** | **4,500** | **256 · 5.7%** | **233 · 5.2%** | **+23 (+0.51%)** | **0.320** |

No net is distinguishable from zero (McNemar *p* 0.074–0.764); GSM8K's negative sign is 8 of 116 discordant questions at *p*=0.516 and carries no weight. **Six rounds of debate change accuracy by an amount indistinguishable from zero on every benchmark.** Combined with the 90.6% no-change rate (Section V-B), debate here is a zero-sum process at six times the single-pass cost — which is exactly why early stopping costs so little accuracy in TABLE VIII: there is little left to lose.

## **VI. DISCUSSION**

### **A. Does Adaptive Stopping Beat Not Reading the Transcript?**

Seeds share 12–46% of their questions, making a paired *t*-test across five seeds underpowered and anti-conservative; TABLE XI instead resamples by distinct question.

**TABLE XI — fixed\_k2 MINUS DUS-11** (10,000 resamples, clustered by question)

| Benchmark | Opponent | Δ Accuracy | 95% CI | *p* | Δ Token |
| :---- | :---- | ----: | :----: | ----: | ----: |
| GSM8K | DUS-11, q=0.4 | +0.0140 | [+0.0000, +0.0278] | 0.051 | **−29.3%** |
| GSM8K | DUS-11, q=0.5 | +0.0107 | [−0.0027, +0.0239] | 0.129 | −18.8% |
| MMLU | DUS-11, q=0.4 | −0.0047 | [−0.0208, +0.0120] | 0.600 | −15.6% |
| MMLU | DUS-11, q=0.5 | −0.0067 | [−0.0231, +0.0149] | 0.466 | −6.8% |
| StrategyQA | DUS-11, q=0.4 | +0.0113 | [−0.0041, +0.0268] | 0.163 | −25.0% |
| StrategyQA | DUS-11, q=0.5 | +0.0040 | [−0.0105, +0.0188] | 0.639 | −14.2% |

Positive Δ Accuracy favours fixed\_k2; negative Δ Token means fixed\_k2 is cheaper. **On accuracy, no**: no cell reaches nominal significance, and under Bonferroni correction across the fifteen *k*/benchmark comparisons fixed\_k2 was selected from, no cell approaches the *p*<0.0033 threshold. **On cost, yes, everywhere**: fixed\_k2 is cheaper in all six comparisons (8.0–29.7%), a property of the policies, needing no correction. Together: fixed\_k2 **weakly dominates** DUS-11 — statistically indistinguishable accuracy, consistently lower cost. Against fixed depth alone, the cascade is not wrong, just not paid for.

That conclusion reverses against the wider reference class:

**TABLE XII — NO-DEBATE BASELINES MINUS DUS-11 (q=0.5)** (10,000 resamples, clustered by question; 27 comparisons, Bonferroni threshold *p*<0.00185)

| Benchmark | Baseline | Δ Accuracy | 95% CI | *p* | Δ Token |
| :---- | :---- | ----: | :----: | ----: | ----: |
| GSM8K | majority voting | −0.0240 | [−0.0432, −0.0047] | 0.018 | −41.0% |
| GSM8K | SC@3, Qwen2.5-3B | **−0.1480** | [−0.1766, −0.1192] | **0.0001** | −41.3% |
| GSM8K | SC@3, Llama3.2-3B | **−0.0953** | [−0.1222, −0.0680] | **0.0001** | −42.8% |
| GSM8K | SC@3, Gemma3-4B | −0.0060 | [−0.0234, +0.0117] | 0.523 | −39.5% |
| MMLU | majority voting | −0.0260 | [−0.0507, −0.0013] | 0.043 | −26.9% |
| MMLU | SC@3, Qwen2.5-3B | +0.0353 | [+0.0060, +0.0646] | 0.019 | −29.1% |
| MMLU | SC@3, Llama3.2-3B | **−0.2533** | [−0.2899, −0.2167] | **0.0001** | −23.8% |
| MMLU | SC@3, Gemma3-4B | −0.0320 | [−0.0521, −0.0113] | 0.003 | −29.7% |
| StrategyQA | majority voting | **−0.0787** | [−0.1113, −0.0451] | **0.0002** | −41.9% |
| StrategyQA | SC@3, Qwen2.5-3B | **−0.1460** | [−0.1916, −0.0999] | **0.0001** | −43.4% |
| StrategyQA | SC@3, Llama3.2-3B | **−0.0827** | [−0.1204, −0.0454] | **0.0002** | −43.4% |
| StrategyQA | SC@3, Gemma3-4B | −0.0153 | [−0.0352, +0.0047] | 0.143 | −39.1% |

Bold = intervals excluding zero after correction. Negative Δ Accuracy means the baseline is worse than DUS-11.

**Six of twelve no-debate baselines lose to DUS-11** by margins surviving correction (up to 25.2 points). Exactly one beats it — Qwen2.5-3B on MMLU by 3.5 points, but nominal only (*p*=0.0186, does not clear the corrected threshold). Two more are indistinguishable at ~40% lower cost. The pattern that matters is **variance**: which cheap baseline is competitive changes completely with the benchmark (Gemma3-4B near-best on GSM8K/StrategyQA, worst on MMLU; Qwen2.5-3B the reverse) — a practitioner cannot know in advance, and the cost of choosing wrong dwarfs anything adaptive stopping gains or loses. Ranked by worst-case regret against the best realisable policy per benchmark: **DUS-11 at most 3.5 points, fixed\_k2 at most 4.2, Gemma3-4B at most 6.7, majority vote at most 8.3, Qwen2.5-3B at most 15.9.** The cascade buys robustness across benchmarks, not accuracy on any one — and against fixed depth, equally robust and cheaper, it still doesn't pay for itself. MMLU remains the one benchmark where the best available policy reads nothing at all.

**TABLE X — CANDIDATE GOVERNING QUANTITIES ALONGSIDE THE OUTCOMES**

| Benchmark | Answer space | Accuracy | Blind spot | AUC (pooled) | Consensus token share | Δ Token achieved |
| :---- | :---- | ----: | ----: | ----: | ----: | ----: |
| GSM8K | Unbounded integers | **0.779** | 2.9% | **0.875** | 56.0% | −6.8% |
| StrategyQA | Binary | 0.686 | 24.4% | 0.609 | 48.5% | −1.9% |
| MMLU | 4 options | **0.547** | **28.0%** | 0.685 | **68.2%** | **−41.1%** |

Discrimination and operational value diverge sharply: GSM8K has the highest AUC (0.875) yet the smallest saving. Saving tracks the *room the incumbent leaves* — MMLU's consensus rule runs to 68.2% of full cost and yields 41.1% back; StrategyQA's runs to only 48.5% and yields almost nothing. Two accounts fit the blind-spot ordering. **Informational:** answer entropy carries at most log₂3≈1.58 bits (1 bit on a binary task); three agents landing on the same unbounded integer is near-proof (2.9% blind spot), two options coincide by chance ~25% of the time (24.4%). **Accuracy-based:** blind spot is inversely monotone in the protocol's own accuracy (0.779/0.686/0.547 ↔ 2.9%/24.4%/28.0%) — weak agents are also wrong together. These diverge on the one pair that could distinguish them: the informational account predicts a *larger* blind spot on binary than 4-option (25% vs. 6.25% chance coincidence); the data run the other way (24.4% vs. 28.0%), favouring the accuracy account. Both converge on the same recommendation — measure blind-spot rate on a pilot before building anything — and we cannot separate them further with only three, non-orthogonal benchmarks; that requires holding questions fixed and varying only answer format.

### **B. Recommendations for Evaluating Stopping Rules**

The Section V-C failure was invisible to stop rate, error rate, and AUC alike. Of the four protocol corrections, **only sequential simulation against the incumbent is diagnostic** — the others make evaluation honest without exposing circularity (non-final-only scoring alone leaves ~7,000 consensus rounds still below threshold). Two further checks matter: **report where the rule stops, not only how well it ranks** (TABLE VI exposes the failure that AUC alone hides), and **check whether the dominant feature is an exact indicator of the termination condition** before sweeping any threshold.

Two more recommendations, both about what to do *before* building a signal:

- **Benchmark against policies that read nothing.** A learned rule should be compared to fixed\_k for every *k*, a matched-sample self-consistency baseline, and an ensemble vote, all at measured token cost — comparison against the incumbent alone cannot reveal fixed\_k2's weak dominance, since the incumbent also reads the transcript.
- **Report self-consistency per model, never as one number.** TABLE XII's three models spread 29 points on MMLU alone and reverse rank across benchmarks; a single "self-consistency" row reports whatever model the authors happened to run, with a between-model standard error larger than most effects a stopping rule could claim.

**Use blind-spot rate as a first screen, rescue/hurt as a second.** Blind-spot rate (TABLE I) bounds what any consensus-improving rule can reclaim, needs only agreement and correctness, and is nearly free: GSM8K's 2.9% predicts (and gets) zero gain. It is necessary but not sufficient — StrategyQA's headroom (24.4%) is unreachable because rescue and hurt cancel (TABLE IX, net +5, *p*=0.76): if the transcript on average doesn't improve an answer, no stopping rule can extract accuracy from disagreement about when to stop reading it. MMLU is the only benchmark clearing both screens, and the only one where DUS-11 shows any advantage. Applied honestly to our own pilot, this two-stage screen would have advised against building the cascade on two of three benchmarks — we treat that as the recommendation's strongest evidence. A third, cheaper screen needs no debate at all: **run self-consistency on each candidate model first.** On MMLU, Qwen2.5-3B@3 beats the cascade (0.582 vs. 0.547) at 29.1% fewer tokens; the reference class should be measured *before* the method is built, not assembled afterward to defend it.

### **C. Limitations**

- **Baseline selection.** With 9 model–benchmark self-consistency cells, the one cell that beats DUS-11 does not survive correction across the family of 27 comparisons.
- **Solver scale.** Both solvers are 3B, the critic 4B; absolute accuracies are modest (0.547 on MMLU). Whether the entropy–consensus identity and critic-signal value persist at larger scale is untested; the identity is structural and should hold, the relative value of critic confidence may not.
- **Single topology.** Only two-solvers-plus-fixed-critic is studied; round-table \[3\] or cross-model \[6\] topologies may distribute uncertainty differently.
- **Threshold transfer.** In-sample selection adds at most 3.7pp Δ Token; transfer *across* benchmarks, model pairs, or answer formats is untested and likely needs recalibration.
- **Residual mass point / uncalibrated score.** 872 rounds tie exactly at *T* under the direct protocol; DUS-11 is used only for ranking/thresholding, not as a calibrated probability.
- **One critic.** CRITIC-7 relies on the critic's independence; multiple critics, or a critic that revises, would need redefinition.
- **Resolution of accuracy comparisons.** TABLE VIII's paired intervals span 1–3 points, so a genuine ~1-point cost would go undetected — the disagreeing MMLU holdout estimate is the concrete reason not to read "indistinguishable from zero" as equality.

## **VII. CONCLUSION**

We set out to stop multi-agent debate early and found, in sequence, that we could not measure whether we had, that a usable signal nevertheless existed, and that it was not worth its price.

A score built from four consensus-dynamics signals reported a large saving that sequential simulation against the incumbent showed to be illusory, because its dominant term is zero exactly where consensus already holds — of the corrections this forces, only sequential simulation is diagnostic. Features read off an independent critic, containing no agreement term, survive that correction: they discriminate better than the consensus signals, by a margin that does not vanish once both are present together. Deployed as a two-stage cascade (round-0 critic self-consistency gate, then the same score rechecked every round), it matches consensus accuracy at markedly lower token cost. But against fixed depth the cascade is weakly dominated — unconditional two-round stopping matches it in accuracy, corrected for multiple comparisons, at lower cost — while against self-consistency the picture reverses, since most no-debate baselines lose to the cascade by wide margins and the rare baseline that wins cannot be identified in advance. What the cascade buys is robustness across benchmarks, not accuracy on any one; fixed depth shows that most of that robustness comes free with the debate itself.

Adaptive stopping pays when the incumbent terminates late, consensus is often wrong, and continuing the debate actually repairs answers. Those conditions coincided on only one of our three benchmarks, and even there a single repeatedly-sampled model matched the cascade; measured on a pilot before any signal is built, they would have advised against building one on two benchmarks of three.

Four directions remain: separating the informational and accuracy accounts of Section VI-A by holding questions fixed and varying only answer format; re-measuring critic signals at larger scale and with multiple critics, where CRITIC-7's independence assumption no longer holds by construction; testing whether fixed depth's weak dominance persists at scale, since debate that genuinely improves answers would restore the headroom adaptive stopping needs; and closing the gap that still separates every realisable policy from the oracle — wide enough that the open question is not whether a stopping rule can match consensus cheaply (fixed depth already does), but what it would have to read in the transcript to do substantially better than any of them.

## **REPRODUCIBILITY**

Complete debate logs for all 4,500 debates (27,000 rounds, 3 benchmarks × 5 seeds × 300 questions) are released, together with both evaluation protocols implemented over identical inputs, split definitions, and the analysis pipeline producing every table in this paper. TABLE III's direct and corrected accountings reproduce in a single invocation.

## **REFERENCES**

\[1\] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, "Improving factuality and reasoning in language models through multiagent debate," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024.

\[2\] T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, S. Shi, and Z. Tu, "Encouraging divergent thinking in large language models through multi-agent debate," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2024.

\[3\] J. C.-Y. Chen, S. Saha, and M. Bansal, "ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs," in *Proc. Annu. Meeting Assoc. Computational Linguistics (ACL)*, 2024.

\[4\] C.-M. Chan, W. Chen, Y. Su, J. Yu, W. Xue, S. Zhang, J. Fu, and Z. Liu, "ChatEval: Towards better LLM-based evaluators through multi-agent debate," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024.

\[5\] A. Khan, J. Hughes, D. Valentine, L. Ruis, K. Sachan, A. Radhakrishnan, E. Grefenstette, S. R. Bowman, T. Rocktäschel, and E. Perez, "Debating with more persuasive LLMs leads to more truthful answers," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024.

\[6\] Z. Yin, Q. Sun, C. Chang, Q. Guo, J. Dai, X. Huang, and X. Qiu, "Exchange-of-Thought: Enhancing large language model capabilities through cross-model communication," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023.

\[7\] T. Hu, Z. Tan, S. Wang, and H. Qu, "Multi-agent debate for LLM judges with adaptive stability detection," *arXiv preprint*, 2025.

\[8\] F. Haji, M. Bethany, and M. Tabar, "Improving LLM reasoning with multi-agent tree-of-thought validator agent," *arXiv preprint*, 2024.

\[9\] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, "Reflexion: Language agents with verbal reinforcement learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

\[10\] Z. Wu, Q. Zeng, Z. Zhang, Z. Tan, C. Shen, and M. Jiang, "Large language models can self-correct with key condition verification," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2024.

\[11\] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang, "Large language model based multi-agents: A survey of progress and challenges," in *Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2024.

\[12\] K.-T. Tran, D. Dao, M.-D. Nguyen, Q.-V. Pham, B. O'Sullivan, and H. D. Nguyen, "Multi-agent collaboration mechanisms: A survey of LLMs," *arXiv preprint*, 2025.

\[13\] S. Chen, Y. Liu, W. Han, W. Zhang, and T. Liu, "A survey on LLM-based multi-agent system: Recent advances and new frontiers in application," *arXiv preprint*, 2024.

\[14\] W. X. Zhao, K. Zhou, J. Li, and T. Tang, "A survey of large language models," *Frontiers of Computer Science*, 2026.

\[15\] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou, "Self-consistency improves chain of thought reasoning in language models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2023.

[image1]: <figures/fig_architecture.png>
[image2]: <figures/fig2_circularity.png>
[image3]: <figures/fig3_cost_accuracy.png>
[image4]: <figures/fig4_cascade_vs_sc_frontier.png>