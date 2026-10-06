## P1 — reward–safety conflict
counts: REPORTED=1 DESIGNED=2 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: CRAX builds every task so that high reward needs cost, and Fig. 6 shows PPO-Lagrangian losing reward as the budget tightens in all tasks.\[8\]
- 2503.08241: HASARD's design ties reward and cost so that lowering cost necessarily costs reward (design rationale; full text not read).\[1\]
notes: This is the default design of every safe-RL benchmark (Safety Gym's report says the same, but it has no arXiv id or DOI so it cannot be a row).\[28\] The best curriculum-specific report is Koprulu, Simão, Jansen and Topcu, "Safety-Prioritizing Curricula for Constrained Reinforcement Learning" (ICLR 2025 proceedings; code built on CURROT with OmniSafe PPOLag), whose abstract says existing curricula "may generate tasks that cause RL agents to violate safety constraints during training and behave suboptimally after"; CURROT moves goals over hazards, but no arXiv id or DOI was found, so the id rule excludes it. P1 is supported, but as a property of single contexts, not of how the conflict is spread over Ω.

## P2 — mostly infeasible
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=3
verdict: supported
strongest:
- 2112.12228: an impossible constraint made the multiplier grow without bound and collapsed critic and performance (Arena, Fig. 3).\[11\]
- 2105.10682: theory shows statewise multipliers diverge on infeasible states.\[12\]
notes: The evidence is about one infeasible CMDP or state. No paper reports a family where most ω are infeasible, so the distributional version of P2 is not evidenced. CRAX's "exploration dominated by violations" at Level 3 is the nearest family-level report.\[8\]

## P3 — hard tail
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=2
verdict: argued only
strongest:
- 2509.18648: averaging the constraint over randomised dynamics can underestimate cost in the deployment context.\[15\]
- 2608.30283: an expected-cost budget is met even when a rare fraction of rollouts is very costly.\[23\]
notes: Koprulu, Simão, Jansen and Topcu, "Risk-Aware Curriculum Generation for Heavy-Tailed Task Distributions" (UAI 2023), would be the natural report, but no id was verified and it is unconstrained. A report showing an averaged constraint satisfied while rare hard contexts violate it would change this verdict.

## P4 — context-dependent safety
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: supported
strongest:
- 2506.11033: across randomised physical parameters, constrained baselines fail to balance safety and return even when the parameters are given as inputs.\[5\]
- 2609.15915: argues that the same state has different safety under different task posteriors.\[7\]
notes: The support is thin (one experimental paper). CARL (2110.02102) was not opened, so its context-blind result is not counted here. The 2506.11033 result covers both blind and context-aware policies, so it does not isolate "one behaviour cannot be safe everywhere".

## P5 — dual-variable lag
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=2
verdict: supported
strongest:
- 2412.04426: a multiplier calibrated offline (≈1500) is far from what online finetuning needs (≈0.65), which causes violations or stagnation.\[3\]
- 2007.03964: argues that slow multiplier updates let many constraint-violating iterations pass before the penalty bites.\[9\]
notes: Both reported rows come from one paper, and the shift there is offline→online, not a curriculum moving over Ω.\[3\] No paper reports λ lag at a curriculum stage switch or a change in task distribution. Stooke et al.'s oscillation happens under a fixed distribution and is filed as NEW:multiplier-oscillation.

## P6 — uninformative sub-budget cost
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=2
verdict: argued only
strongest:
- 2206.02675: argues that sparse, unknown cost leads to violations and adds a remaining-budget state to give distance-to-boundary.\[21\]
- 2606.11266: argues that cost is reactive and λ grows only after the budget is exceeded.\[22\]
notes: The Sauté seed (2202.06558) abstract gives no failure report for P6. An ablation showing Lagrangian learners blind to the boundary while under budget would make this supported.

## NEW:multiplier-oscillation — λ/cost oscillation and overshoot around the limit
counts: REPORTED=3 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2007.03964: cost and multiplier oscillate with a 90° phase lag on DoggoButton1, and the authors saw numerous problematic cases.\[9\]
- 2310.12567: Lagrangian methods oscillate more than projection methods in Safety-Gymnasium (full text not read).\[10\]
notes: This is the best-documented failure overall, but it is in-distribution. It matters for situations because any change in context distribution adds a disturbance to a controller that is already badly damped.

## NEW:over-conservatism — feasible but budget-underusing or trivially safe policies
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: direct training on CRAX Pathway/Height leaves budget unused, and PPOLag satisfies every Level 2–3 bound with low reward.\[8\]
- 2606.20376: unconstrained-to-safe transfer fixes this in Pathway, but the reverse holds in Goal.\[8\]
notes: Over-conservatism, not infeasibility, may explain why CRAX's curriculum helps Pathway/Height but not Goal. This is our hypothesis, not a claim by the authors.

## NEW:hard-context-exploration — violation-dominated exploration in the hardest contexts
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: at Level 3, exploration is dominated by constraint violations and sparse progress.\[8\]
notes: Single source. It is close to P2 but differs: the contexts are feasible, just hard to find a safe path in.

## NEW:budget-regime-sensitivity — the same task changes character with the cost limit
counts: REPORTED=2 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2510.17564: constraint restrictiveness varies across cost limits within one task, and λ is highly sensitive.\[4\]\[19\]
- 2606.20376: tightening the bound hurts far more than loosening it helps.\[8\]
notes: This suggests the budget d should be a context dimension in ω.

## NEW:shift-violation — constraint satisfied in training contexts, violated in unseen ones
counts: REPORTED=5 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2601.21094: across eight safe RL algorithms, three diabetes types and three age groups, policies that satisfy the constraint on the training patient frequently violate it on unseen patients; test-time shielding gives "Time-in-Range gains of 13--14%" for PPO-Lag and CPO.
- 2509.18648: domain-randomised constrained training often fails the constraints on target dynamics.\[15\]
notes: This is the most-reported family-level property. 1907.01475 is not a CMDP learner (it counts catastrophes).

## NEW:constraint-type-dependence — method success depends on the kind of constraint
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2606.20376: Saute meets the bound only in Reacher/Pathway and otherwise behaves like unconstrained PPO.\[8\]
notes: Single source.

## NEW:feasibility-first-stall — reward gets no traction until a feasible policy is found
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2112.12228: with normalised multipliers and many constraints, the main task gets almost no weight while feasibility is sought.\[11\]
notes: The Sec. 6.1 multi-constraint experiment (not retrieved) may contain the matching report.\[11\]

## NEW:budget-history-dependence — the safe action depends on budget already spent
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2206.02675: crossing paths need knowledge of the cost already incurred; a policy that ignores it violates the constraint.\[21\]
notes: The authors say experiments confirm it, but no figure-level quote was extracted.\[21\]

## NEW:stale-constraint — a fixed constraint stops describing safety after the context changes
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2605.18842: a policy may satisfy an outdated constraint that no longer captures safety.\[14\]
notes: Single-author preprint, single domain.\[14\]

## NEW:zero-budget-ratchet — with d=0 the multiplier only increases
counts: REPORTED=0 DESIGNED=0 REVIEWED=0 ARGUED=1
verdict: argued only
strongest:
- 2512.23770: once increased, the multiplier cannot decrease, which traps policies in conservative regimes.\[24\]
notes: Snippet only. It is the zero-budget limit of P6, since there is no sub-budget region at all.

## NEW:eval-mode-shift — stochastic-training safety does not carry over to deterministic evaluation
counts: REPORTED=1 DESIGNED=0 REVIEWED=0 ARGUED=0
verdict: supported
strongest:
- 2609.15315: Spoor, Plaat, Moerland and co-authors (submitted 14 Sep 2026; SafeRLEval suite) ask "whether training-time behavior is representative of behavior of the final converged policy"; their snippet shows a converged PPO-Lag policy under budget with noise that violates it when evaluated deterministically on CRAX, but the abstract does not confirm this specific case.
notes: Single-seed illustration according to the snippet; not read in full.\[20\]

## Ambiguities
- 2007.03964 P5? (two rows): the lag argument concerns timescale separation under a fixed distribution, not a shift in the context distribution.
- 2509.18648 P3?: the argument is about averaging over dynamics hiding the target context; that fits a hard tail only if the target is a rare context.
- 2608.30283 P3?: the tail is over trajectories, not contexts.
- 2206.02675 P6?: sparse cost is about the signal being uninformative, but not specifically below the budget.
- 2606.11266 P6?: the argument is about cost arriving too late (reactive), which is related to P6 but not identical.
- Papers excluded only because no arXiv id or DOI was verified: Ray, Achiam, Amodei 2019 Safety Gym (read in full; it supports P1 by design and explicitly randomises layouts for generalisation); Koprulu et al. ICLR 2025 Safety-Prioritizing Curricula for Constrained Reinforcement Learning (read in full; the core by-product A paper and P1 evidence);\[25\] Klink et al. 2022 CURROT; Koprulu et al. UAI 2023; Yao et al. 2023 CCPO.\[25\]\[28\]

## Search log
Date of all searches: 2026-10-06.
Impossible protocol steps:
- arXiv API field-restricted queries (export.arxiv.org) were refused (robots disallowed). Two attempts were made: the Q3 string and the QA1 string, both as abs: field queries.
- Semantic Scholar Graph API returned HTTP 429 on both attempts, so the full reference and citation lists for the seeds could not be pulled.
- OpenReview PDFs were blocked by bot detection, so REVIEWED=0 everywhere reflects a tooling gap, not an absence of evidence.
- Google Scholar forward citations were not available.
What was done instead:
- A general web search engine was used as a proxy for each Q/QA query.
- Snowballing was done manually from reference lists and from citing papers that surfaced in search results.
- The protocol hit counts (69, 137, 8, 60, 10, 31, 6, 35) could not be reproduced.
- Q2 was not screened to 60, because the web search returns about 10 results per query.
- One round of snowballing was done through one delegated sub-search, from 2506.11033/2412.04426. It added 2609.15915 (included), 2605.18842 (included) and 2101.00531 (excluded). A second round was not run, so the stop criterion was not verified.
Web-search strings as executed (each mapped to the nearest protocol query):
1. Safe Curriculum Generation constrained CURROT Klink [QA1]
2. Benchmarking Safe Exploration in Deep Reinforcement Learning Ray Achiam Amodei [seed Safety Gym]
3. arxiv Safety-Prioritizing Curricula for Constrained Reinforcement Learning Koprulu [QA1]
4. CRAX Fast Safe Reinforcement Learning Benchmarking Tomilin curriculum [seed 2606.20376]
5. "Safety-Prioritizing Curricula" arxiv.org abs [QA1]
6. safe reinforcement learning generalization unseen environments constraint violation distribution shift [Q2]
7. constrained reinforcement learning infeasible constraint feasibility Lagrange multiplier diverges [Q4]
8. Safety Gym "Benchmarking Safe Exploration" cdn.openai.com pdf difficulty levels hazards [seed]
9. Lagrange multiplier drops to zero constraint satisfied safe RL cost signal vanishes conservative [Q3/P6]
10. unsupervised environment design safety constraint violations adversarial levels safe RL [QA2]
11. Sauté RL safety budget state augmentation sparse cost Lagrangian fails [seed 2202.06558]
12. OmniSafe benchmark results Lagrangian methods oscillation cost limit Safety-Gymnasium failure [seed 2305.09304/Q3]
13. safe reinforcement learning domain randomization constraint satisfaction varying dynamics contextual CMDP [Q6]
14. multi-task constrained reinforcement learning per-task Lagrange multiplier heterogeneous constraints [P5]
15. safe RL over-conservative policy cost limit zero hard tail rare hazardous scenarios constrained RL [Q5/P3]
16. Koprulu Simão Jansen Topcu arXiv 2024 safe curriculum constrained reinforcement learning [QA1]
17. HASARD vision-based safe RL benchmark difficulty levels agents shortcut hazards reward cost [snowball 2606.20376]
Incidental search-snippet hits not opened and not obviously in scope were not given screening rows.
