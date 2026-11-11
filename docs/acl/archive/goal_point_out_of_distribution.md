# Goal-point: the out-of-distribution evaluation target Ω₂

Plan item 3. A policy trained on a distribution over Ω is scored, at every
evaluation, on a third target besides $w$ (the deployment task) and $r$
(Uniform(Ω)): **Uniform(Ω₂)**, where Ω₂ is a set of contexts the environment
can run but Ω never contains.

**Superseded (2026-10-13) by `../research_design.md` §5.3**: both placeholder
directions fail its admissibility rules (`goal_size` never reaches the
observation; density ≥ 21 is confounded by silent overcrowding). The wiring
below stays; the region is to be replaced by the test sets defined there.

**Status: the wiring is done; the values are a placeholder.** The Ω₂ in
`registry.py` (total ≥ 21 hazards, goal radius [0.10, 0.14]) was chosen in an
evening so that the pipeline could be built and tested end to end. It should
not be read as a design. The design question is below; it needs an answer
before Ω₂ appears in an experiment.

## What is forced

Ω is the full per-group box ∩ *total ≤ 20*, with goal radius [0.14, 0.22]
(`registry.py`). The environment can run exactly three kinds of context Ω
excludes, and nothing else without a new dimension:

| direction | what it is | harder or easier than Ω |
|---|---|---|
| total ≥ 21 active hazards | denser than any level | harder; above ~20 the placement overcrowds silently (README, *Known defects*; level 3 itself has 4 % overlap) |
| goal radius < 0.14 | a smaller disc than any level | harder, and clean: no layout defect |
| goal radius > 0.22 | a bigger disc | easier |

The two spaces must be disjoint up to a set of measure zero;
`SuiteContexts.__post_init__` enforces it.

## The question

*What is the gap between the Ω curve and the Ω₂ curve supposed to measure, and
against what reference is that gap readable?* Five decisions hide in it.

1. **Generalisation of what.** Reward, or constraint satisfaction? For a
   constrained curriculum the second is the thesis-relevant reading: a policy
   that met the budget on Ω — does it stay within budget off Ω, or was its
   safety a property of the contexts it saw? The reward gap is what the
   contextual-RL literature measures (Kirk et al. 2023, CARL); the cost gap is
   where the constrained setting adds something. The answer decides which
   curve Ω₂ exists for and which panel it gets.

2. **A harder Ω₂ is not a generalisation gap.** Every available direction on
   the counts is harder. A policy that generalises perfectly still scores lower
   on a harder target, so "reward 21 on Ω, 12 on Ω₂" is unreadable on its own.
   It becomes readable only against (a) the other arms on the same Ω₂ — the
   *ranking* is the result, not the number — or (b) a reference arm for which
   Ω₂ is in-distribution (trained on Uniform(Ω ∪ Ω₂)), giving the achievable
   ceiling. Without (b) a gap cannot be told apart from difficulty. Whether to
   run (b) is a cost decision (one more 500 M arm).

3. **One shift at a time.** The placeholder shifts two things at once (denser
   *and* smaller goal). A drop then cannot be attributed to either. Either Ω₂
   is one shift, or there are two targets, Ω₂ᵃ (density) and Ω₂ᵇ (goal size),
   each a separate panel.

4. **Valid scenes or degenerate ones.** Above 20 hazards the layouts overlap;
   the pseudo-lidar (centre-based, no occlusion) aliases overlapping hazards
   into one reading. Dense Ω₂ therefore tests extrapolation to scenes no level
   was ever meant to produce. That is a legitimate target if the question is
   robustness to broken layouts, and a confound if the question is
   generalisation across valid ones. The overlap rate as a function of total
   count is measurable on CPU today (place N layouts per count, count
   intersections); the decision needs that number.

5. **Extrapolation or interpolation.** The alternative the generalisation
   literature defaults to is a *held-out cell inside Ω*: Ω gets a hole, Ω₂ is
   the hole, and the gap measures interpolation to unseen combinations of
   seen values. It is clean (no difficulty confound, no degenerate layouts) but
   changes the training distribution, so every arm run so far — the 500 M
   baselines included — would be on a different Ω. The trade-off is between a
   clean measurement and comparability with what exists.

What settles these is the research question Ω₂ serves. The README's goal 1
("does a method that copes with a challenge on reward still cope when the
budget binds") points at decision 1 → cost, decision 3 → one shift, and makes
decision 4 the thing to measure first. Nothing here proposes values; the next
step is a short measurement (overlap vs. count) and a one-paragraph answer to
decision 1 from Giuseppe, then the values follow.

## How it is wired

- `ContextSpace.integer_total_floor` (`training/contexts/space.py`): the
  mirror of `integer_total_cap`, applied in `integer_points`, `contains` and
  `describe`.
- `SuiteContexts.evaluation_regions` (`registry.py`): a tuple of
  `EvaluationRegion(name, space, label, relation)` — named regions of Ω's
  dimensions the policy is scored on, each stating its relation to Ω
  (`"disjoint_from"` or `"within"`), checked when the suite is built. Ω₂ is the
  one region so far (`out_of_distribution`, disjoint); goal-point is the only
  suite with one. The README's skill subspaces (goal 2) are further regions,
  `"within"`.
- `training/contexts/setup.py: evaluations_for(suite, deployment_spec, ...)`
  is the one place the rule "which targets does a run get" is written:
  `deployment`, `uniform`, then `Uniform(region.space)` for every region, under
  the region's name and label. No branch per region. `train_kwargs` (one
  evaluator per entry), the W&B config (`evaluations`, name → label, which the
  dashboard reads) and `tests/test_suites.py` all consume its result; none
  re-derives it.
- Metrics: `evaluation/out_of_distribution/*`, the same keys as the other two.
- Changing the values is a one-line edit of the two `_GOAL_OUT_OF_DISTRIBUTION_*`
  constants in `registry.py`; a second target (decision 3) is a second
  `EvaluationRegion` in goal-point's factory and nothing else.
