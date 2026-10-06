# Literature protocol: what makes a constrained environment family hard to train on?

Status: draft v1, 2026-10-06. Light-weight (one afternoon) by design; written to
be extended to a full systematic pass if the thesis is positioned around it.

## Question

**Which properties of a family of constrained environments $\{M_\omega\}_{\omega\in\Omega}$
make constrained RL training fail or struggle, as reported by people who tried?**

Each property found becomes a candidate *situation* for the thesis chapter
(`docs/acl/README.md`, *The goal*): a distribution over Ω that isolates that one
property, on which curriculum methods are compared. Evidence that practitioners
actually hit the property is what justifies a situation; properties nobody has
reported struggling with are hypotheses, not situations.

The review is **not** about curriculum methods. It is about the constrained-RL
problem side. One by-product is collected, because a supervisor will ask:

- *By-product A:* every paper that combines a curriculum / environment
  distribution with a constrained or safe RL learner. Expected to be <15 papers.
  Recorded in a separate table, no extraction beyond one line.

## Candidate properties (hypotheses going in)

These come from the notebook (`docs/notebook/1-3.xml` p.2) and from reading the
three surveys. The review tests whether each has evidence, and is explicitly
asked to add properties that are **not** on this list.

| id | property | one-line statement |
|---|---|---|
| P1 | reward–safety conflict | the reward-maximising behaviour is the unsafe one (hazardous shortcut); more reward means more cost |
| P2 | mostly infeasible | for most ω no policy satisfies the budget; the feasible region is small |
| P3 | hard tail | most ω are trivially safe; the deployment distribution w is the hard, rare tail |
| P4 | context-dependent safety | the safe behaviour differs by ω (fast is safe here, slow there); one behaviour cannot be safe everywhere |
| P5 | dual-variable lag | the Lagrange multiplier (or equivalent) is calibrated on the training distribution; a shift in that distribution makes it wrong until it re-converges |
| P6 | uninformative sub-budget cost | cost is incurred but the budget is not exceeded, so the learner gets no signal about the constraint boundary |
| P? | *new* | anything reported that is not above |

## Sources and queries

Databases: arXiv (API), Semantic Scholar, OpenReview (NeurIPS/ICLR/ICML
discussions and reviews count: reviewers report failure modes authors omit).
Google Scholar for forward citations only.

Date range: 2017 (CPO, Achiam et al.) to today. Seeds may be older.

### Keyword queries (arXiv, abstract field; counts measured 2026-10-06)

| id | query | hits |
|---|---|---|
| Q1 | `"safe reinforcement learning" AND benchmark` | 69 |
| Q2 | `("constrained reinforcement learning" OR "safe reinforcement learning") AND (generalization OR "distribution shift" OR "unseen environments")` | 137 |
| Q3 | `(Lagrangian OR "Lagrange multiplier") AND "reinforcement learning" AND (oscillat* OR instabil* OR overshoot)` | 8 |
| Q4 | `("constrained MDP" OR CMDP OR "constrained reinforcement learning") AND (infeasib* OR "strictly feasible" OR feasibility)` | 60 |
| Q5 | `"safe reinforcement learning" AND (trade-off OR tradeoff) AND cost AND reward` | 10 |
| Q6 | `(contextual OR "domain randomization") AND ("constrained reinforcement learning" OR "safe reinforcement learning" OR CMDP)` | 31 |
| QA1 | `curriculum AND ("constrained reinforcement learning" OR "safe reinforcement learning" OR CMDP OR "constrained Markov")` | 6 |
| QA2 | `("unsupervised environment design" OR "prioritized level replay" OR "environment design") AND (safe OR safety OR constraint)` | 35 |

~350 abstracts. Q2 is broad; screen its first 60 by relevance, then stop if the
last 20 yielded nothing.

### Seed set for snowballing

Abstract-keyword search will miss most of the evidence: nobody writes "our
method fails when the feasible region is small" in an abstract. It is in
experiment sections, limitations, benchmark design rationale and reviews.
Therefore: for every seed, read backward (its references) and forward (papers
citing it, via Semantic Scholar) and screen those too.

Ids below were resolved against Semantic Scholar on 2026-10-06
(`verify_references.py`); start from the id, not the title.

| id | seed | why |
|---|---|---|
| *(no arXiv id; OpenAI technical report, 2019)* | Ray, Achiam, Amodei, *Benchmarking Safe Exploration in Deep RL* (Safety Gym) | the benchmark whose protocol (reward–cost Pareto, three difficulty levels) every later paper inherits; its design rationale states what the authors thought was hard. Cite by title; find via Semantic Scholar |
| 2310.12567 | Ji et al. 2023, *Safety-Gymnasium* | same, updated; adds vision and multi-agent |
| 2305.09304 | Ji et al. 2023, *OmniSafe* | large empirical comparison; look at what fails where |
| 2606.20376 | Tomilin et al. 2026, *CRAX* | our benchmark; its manual curriculum helps Reacher/Pathway/Height but not Goal — a negative result to explain |
| 2007.03964 | Stooke, Achiam, Abbeel 2020, *Responsive Safety in RL by PID Lagrangian Methods* | exists because λ oscillates; evidence for P5 |
| 2402.02025 | Wachi, Shen, Sui 2024, *A Survey of Constraint Formulations in Safe RL* | formulation-side map; feasibility and "when must the agent be safe" |
| 2006.12136 | Turchetta et al. 2020, *Safe RL via Curriculum Induction* | the one principled ACL × safe-RL paper (by-product A) |
| *(id unknown; search by title)* | Klink et al., *Safe Curriculum Generation* (supervisor's; constrained CURROT) | prior art on constrained curricula (by-product A) |
| 2103.09815 | Romac et al. 2021, *TeachMyAgent* | the structure we borrow; check that its challenges were operationalised the way we think |
| 2110.02102 | Benjamins et al. 2023, *CARL* (contextual RL benchmark) | context-blind policies and physics contexts; evidence for P4 |
| 2202.06558 | Sootla et al. 2022, *Sauté RL* | state augmentation with remaining budget; evidence for or against P6 |
| 2012.02096 | Dennis et al. 2020, *PAIRED* (unsupervised environment design) | the current mainstream for curricula over environment parameters; regret objective is reward-defined (P1) |
| 2010.03934 | Jiang et al. 2021, *Prioritized Level Replay* | same line; forward citations are where any "UED + safety" paper will sit |
| 2203.01302 | Parker-Holder et al. 2022, *ACCEL* | same line |

Stop criterion: one round of snowballing from newly included papers yields no
new inclusion.

## Screening

Two stages, as is standard. Record every decision (the screening table is a
deliverable, not scratch).

**Stage 1 — title + abstract.** Include if **any** of:

- the paper evaluates a constrained / safe RL method across *more than one*
  environment configuration, level, task, or randomised physics (otherwise it
  cannot say anything about properties of a *family*);
- the paper is a benchmark, suite, or large empirical comparison for
  constrained / safe RL;
- the abstract names a failure of constrained training (instability,
  infeasibility, violation under shift, over-conservatism, collapse);
- the paper combines a curriculum, task distribution, environment design, or
  domain randomisation with a constrained learner (→ by-product A).

Exclude if: not RL; constraint is not a safety/cost constraint (e.g. KL
regularisation, budget in bandits, ROI in advertising); single fixed
environment with no variation and no reported failure; offline-only with no
environment variation; pure theory without an experiment *and* without a named
failure mode (theory that names infeasibility or feasibility assumptions **is**
included for P2).

**Stage 2 — full text.** Include if the paper contains at least one
*evidence item* (below). Otherwise exclude with reason "no property evidence".

## Extraction form (one row per evidence item, a paper may have several)

```
paper_id        arXiv id or DOI (mandatory; verified to exist before citing)
title
year
venue
property        P1..P6 or NEW:<short name>
statement       one sentence, in the paper's own terms, of what went wrong or was hard
environment     where it was observed (Safety Gym PointGoal2, custom gridworld, ...)
evidence_type   one of:
                  REPORTED  — the authors ran it and report the failure (figure/table)
                  DESIGNED  — a benchmark/suite built to expose this property (design rationale)
                  REVIEWED  — a reviewer or discussion (OpenReview) raised it
                  ARGUED    — theoretical or informal argument, no experiment
location        section / figure / page
quote           ≤ 40 words verbatim, so the claim can be checked
```

`evidence_type` is the column that matters. REPORTED and DESIGNED are what
"practitioners run into it" means; ARGUED alone does not justify a situation.

## Deliverables

1. `screening.csv` — every title seen, stage reached, decision, criterion.
2. `evidence.csv` — the extraction rows.
3. `by_product_A.md` — the ACL × constrained-RL table, one line each.
4. `synthesis.md` — per property: count of evidence items by type, the two or
   three strongest, and a verdict (*supported* / *argued only* / *no evidence*);
   plus the NEW properties found, with the same.

Two independent runs (Claude, ChatGPT) on this protocol; disagreements in
inclusion or property assignment are listed and resolved by Giuseppe.

## Known limitations of this pass

- Deep-research tools do not expose their real queries; the query log they
  return is self-reported. Mitigation: every `paper_id` is verified against the
  arXiv API / Crossref before use; unverifiable rows are dropped.
- Abstract screening under-detects property evidence; snowballing from the
  seeds is the main detector, keyword search is the safety net.
- One afternoon. Not exhaustive; the stop criterion is saturation on the seed
  snowball, not coverage of the database.
