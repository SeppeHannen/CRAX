# Synthesis: which properties of a constrained environment family are reported to make training hard

Merged from two independent deep-research runs on `protocol.md` (Claude,
ChatGPT; raw outputs in `runs/`), 2026-10-06. Every `paper_id` cited here
resolved against Semantic Scholar / arXiv (`verify_references.py`: 56 ids, 54
clean; the two flags are a renamed paper and an excluded one). Quotes from
2606.20376 and 2412.04426 were checked against the papers by hand; the rest are
as the tools reported them.

Coverage caveat, stated by both runs: neither could reach the arXiv API or
Semantic Scholar, so the protocol's keyword hit lists were approximated by web
search and snowballing was one manual round. Reviewer evidence (OpenReview) was
not reached at all. This is a decent first pass, not a saturated review.

## Verdict per candidate property

| id | property | Claude | ChatGPT | merged verdict |
|---|---|---|---|---|
| P1 | reward–safety conflict | supported (1 REP, 2 DES) | supported (2 REP, 2 DES) | **supported — but as a property of every single context, by benchmark design.** No paper reports how the conflict is *distributed over* a family. |
| P2 | mostly infeasible | supported (1 REP, 3 ARG) — for *one* infeasible CMDP | no evidence | **argued only, for the family version.** λ blows up on one unsatisfiable constraint (2112.12228 Fig. 3; theory 2105.10682, 2205.07536). Nobody reports a family where most ω are infeasible. |
| P3 | hard tail | argued only (2 ARG, both `?`) | no evidence | **no evidence.** The two arguments are about tails over *trajectories* or *dynamics averaging*, not rare hard contexts. |
| P4 | context-dependent safety | supported (1 REP, 1 ARG) | supported (1 REP, 1 ARG) | **supported, thin.** Each run found one different experimental paper (2506.11033; 2601.21094). See NEW shift-violation, which is P4's empirical face. |
| P5 | dual-variable lag | supported (2 REP, 2 ARG) | no evidence (re-filed as NEW:dual-instability) | **supported for offline→online; never measured at a curriculum switch.** 2412.04426: λ ≈ 1500 offline vs ≈ 0.65 needed online, "Lagrangian mismatch" in the abstract (checked). Our velocity-ant collapse would be the first curriculum-side measurement. |
| P6 | uninformative sub-budget cost | argued only (2 ARG, both `?`) | no evidence | **argued only.** 2206.02675 argues sparse cost → violations; nobody isolates "under budget, no boundary signal". |

## Properties the literature reports that were not on the list

Ordered by how many independent REPORTED items support them. Names unified
across the two runs; the run-specific names are in `runs/*/synthesis.md`.

| property | REPORTED | sources | one-line statement |
|---|---|---|---|
| **shift-violation** | 5 (Claude) + 1 filed as P4 (ChatGPT) | 2601.21094, 2509.18648, 1907.01475, 2301.12593, 2506.11033 | a policy that satisfies the constraint on training contexts violates it on unseen ones. The most-reported family-level property. P4's empirical face; what a curriculum directly manipulates. |
| **multiplier oscillation** (Claude) = **dual-instability** (ChatGPT) | 3 | 2007.03964 §1 Fig. 1, 2310.12567, 2301.10339 | λ and cost oscillate around the limit with overshoot; "we witnessed numerous problematic cases" (Stooke). In-distribution; the controller is badly damped before any shift. |
| **budget-regime sensitivity** (Claude) = **constraint-regime** (ChatGPT) | 2 | 2510.17564 (abstract + Table 1), 2606.20376 §5.4 Fig. 6 | the same task changes character with the cost limit; the best λ-update rule changes with d; tightening d hurts far more than loosening helps. Both runs: *d is itself a context dimension*. |
| **over-conservatism** | 2 | 2606.20376 §5.1 Table 3, §5.2 Fig. 4 | PPO-Lag satisfies every L2–3 bound with low reward; direct training "underutilizes the available safety budget" in Pathway/Height. |
| **hard-context exploration** (Claude) = **constraint-dominated exploration** (ChatGPT) | 1 (both runs, same source) | 2606.20376 §5.2 | "exploration becomes dominated by constraint violations and sparse progress" at Level 3. Verified verbatim. |
| **environment stochasticity** | 2 (ChatGPT only) | 2310.12567 §6, 2305.13681 §6 Fig. 8 | moving hazards / stochastic navigation raise variance and oscillation. Relevant to button (gremlins), not goal. |
| **eval-mode shift** | 1 (Claude) | 2609.15315 | within budget with exploration noise, over budget evaluated deterministically — on CRAX Circle. Methodological; affects how we evaluate, not what we train on. |
| budget-history dependence; feasibility-first stall; stale constraint; zero-budget ratchet; constraint-type dependence; high-dimensional control | ≤ 1 each, mostly ARGUED | see `runs/*/evidence.csv` | recorded, not actionable for the chapter. |

## Reading for the thesis chapter

1. **P1 is universal, so it is not a situation on its own.** Every safe-RL
   benchmark is built so that reward costs safety. The situation that *is*
   new is how the conflict is distributed over Ω — e.g. a layout dimension
   where the unsafe shortcut exists in some contexts and not others — and the
   test is where a reward-driven curriculum (learning progress, regret) puts
   its mass. The literature gives the premise; the Ω-distribution is ours.

2. **P4 ≡ shift-violation is the best-evidenced family property** (six
   REPORTED items from five papers; four groups, since 2601.21094 and
   2506.11033 share first author Kwon). It is also exactly what training
   under q and deploying under w produces. Our "constraint is on deployment,
   not on training" statement (README) has literature behind it.

3. **P5 is the strongest novelty.** λ fragility is documented three ways
   (oscillation in-distribution; value depends on task and d; grossly wrong
   after an offline→online shift). Nobody has measured λ across a *curriculum*
   switch. We have one data point (velocity ant, λ 1.5 → 405). Promote from
   framing to a first-class situation; it is also the one where PPO-Lag vs
   PPO-PID is a controlled comparison.

4. **P2 and P3 should be reframed, not dropped.** The literature does not
   report "most contexts infeasible" or "rare hard tail". It reports two
   *opposite* failure modes on hard contexts: over-conservatism (budget
   unused, low reward) and violation-dominated exploration. CRAX's own data
   has both, in different suites, and its curriculum helped exactly where the
   failure was conservatism (Pathway, Height) and not where it wasn't (Goal).
   That is a mechanism hypothesis for the supervisor's negative result, from
   his own paper — and a better pair of situations than P2/P3: *does the
   curriculum help when the failure on w is conservatism? when it is
   violation-dominated exploration?*

5. **d is a context dimension** (both runs, independently). This contradicts
   the 2026-09-22 decision "Ω is physics and layout, not the cost threshold".
   The decision stands for *this* thesis (one threshold per run keeps the
   constraint well-defined), but the literature says the regime matters;
   note it as a limitation and a follow-up.

6. **P6 has no empirical support.** Keep as a footnote; do not build on it.

## By-product A: curricula or distributions with a constrained learner

| paper_id | paper | how the curriculum touches the constraint |
|---|---|---|
| — (ICLR 2025 proceedings; no arXiv id found) | **Koprulu, Simão, Jansen, Topcu, *Safety-Prioritizing Curricula for Constrained RL*** | CURROT with a constrained objective, on OmniSafe PPO-Lag. Claude read it in full and calls it "the core by-product-A paper". Simão is a CRAX author. **Almost certainly the paper the supervisor meant by "Safe Curriculum Generation" — confirm, and get the proceedings DOI.** |
| 2006.12136 | Turchetta et al. 2020, *Safe RL via Curriculum Induction* | a learned teacher picks reset controllers, not task distributions |
| 2206.02675 | Sootla et al. 2022, *Effects of Safety State Augmentation* | curriculum over the *budget* d (PI controller raises it), not over Ω |
| 2606.20376 | CRAX | hand-made level ladder; helps Reacher/Pathway/Height, not Goal |
| 2503.08241 | HASARD (Tomilin et al. 2025) | level ladder as "implicit curriculum", Lagrangian/PID baselines |
| 2506.11033, 2509.18648, 2301.12593 | adaptive shielding; SPiDR; distributionally robust safe RL | domain randomisation over dynamics with a constrained learner; DR as baseline that *violates* on target dynamics |
| 2609.37070, 2502.15119 | predictive safety curricula (legged); CurricuVLM (driving) | application papers; learner's constraint handling unverified |

Reading: nobody has compared *automatic* curriculum methods on a constrained
learner across *situations*. The nearest is Koprulu et al. (one method, one
objective). The gap the thesis fills is real.

## Disagreements between the runs (resolved as follows)

| paper | Claude | ChatGPT | resolution |
|---|---|---|---|
| 2007.03964 Stooke | NEW:multiplier-oscillation REPORTED + P5? ARGUED ×2 | NEW:dual-instability REPORTED (no quote) | same property, two names → *multiplier oscillation*; the P5? rows stay ARGUED (timescale separation under a fixed distribution is not a shift) |
| 2601.21094 diabetes | NEW:shift-violation | P4 | both: shift-violation is P4's empirical form |
| 2310.12567 Safety-Gymnasium | 1 row (oscillation) | 4 rows incl. P1 and stochasticity | ChatGPT read the full text; take its rows |
| 2206.02675 Sootla | P6? + budget-history ARGUED | NEW:sparse-delayed-cost REPORTED | ChatGPT's REPORTED row stands (Fig. 4 location given); P6 remains argued only |
| 2605.18842 continual safe RL | include (stale constraint) | exclude ("nonstationarity not an environment family") | ChatGPT's exclusion is the stricter reading of the protocol; keep as ARGUED only |
| 2006.12136 Turchetta | include (by-product A) | exclude ("stage 2 unavailable") | tooling, not substance; include |

## What would change these verdicts

- Opening Koprulu et al. ICLR 2025 and the Safety Gym report, and giving them
  rows by title (the protocol's id rule dropped both).
- One round of real forward-citation snowballing on 2606.20376, 2006.12136 and
  Koprulu et al. once Semantic Scholar is reachable.
- Reading CARL (2110.02102) for P4: both runs skipped it as "not safe RL",
  which is correct for the protocol but loses the context-blind-policy
  evidence the README already planned to use.
- Any paper logging λ across a curriculum stage switch. Both runs looked; none
  found. If that holds, it is a contribution.
