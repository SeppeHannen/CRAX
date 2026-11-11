# The generalization evaluation: what we test, and why each choice was made

Decided 2026-11-10; the hidden factor and the combinatorial test fixed
2026-11-11. This is the design every current experiment follows. Each decision
is stated with its reason, so that it can be defended or changed.

## The question

Does a policy that stays within its cost budget on the environments it trained
on also stay within budget on environments it has not seen?

Cost is the first outcome and return the second. Cost has a fixed line, the
budget: going over it in a new environment is a violation, not just a lower
score.

## Two questions that are answered separately

1. **Where must the budget hold?** In each environment separately, or only on
   average over environments? This decides what the Lagrange function sees.
2. **What can the policy see?** This decides whether the policy is able to
   behave differently in different environments.

The Lagrange function only sets how expensive cost is during training. It is
gone after training. The policy is what remains, and it can only act on
different prices if it can tell the environments apart.

| Component | Sees a factor when | Exists at deployment |
|---|---|---|
| Lagrange function λ(ω) | the budget must hold for each value of that factor | no |
| Reward and cost critics | always | no |
| Policy | a deployed robot could know or sense that factor | yes |

## The learner

| Decision | Why |
|---|---|
| The budget must hold in each environment, not on average | With one multiplier the budget held on average only: every standard run cost 33 to 78 on level 3 against a budget of 25 (`experiments/2026-10-15_goal_point_learners_500M.md`). If the budget holds only on average, a high cost on a test environment cannot be told apart from the imbalance already present in training. |
| The Lagrange function sees every factor for which the budget must hold separately | It is one multiplier per constraint, so its inputs follow from the constraint. It exists only in training, so it may see what a deployed robot cannot. |
| The policy sees only what a deployed robot could know | A policy told something no robot knows would not be usable. In the code a factor is marked hidden (`Dimension.hidden`), and a `told` policy is told every factor but the hidden ones. |
| The Lagrange function and the critics are built over the contexts of the run's protocol | A factor the protocol keeps fixed is no input to the multiplier, so the first experiment does not depend on the range chosen for a later factor. |
| The critics see the full context | They exist only in training, and the cost critic has to estimate cost per environment. |
| The Lagrange function uses the additive form by default (`--constraint additive`) | It kept more return than the network form at the same cost on all four clean training distributions (2026-10-15). It assumes difficulty rises or falls steadily along each factor. |
| Every run uses the surface lidar (`hazard_lidar=surface_by_kind`) | It removed most of the rare, very costly episodes (`experiments/2026-10-19_goal_point_cost_tail_500M.md`). |
| Goal-point's contacts are solved with MJX's Newton solver, not the stock CG | Under CG a robot pressed against a wall or a solid hazard was kicked away, sometimes out of the arena, which made pressing on a hazard cheap (`known_defects.md`, 2026-11-11). Costs since then are not comparable with October's. |

**The four learners compared** are the two switches crossed:

| `--constraint` | `--policy_context` | What it is | What it shows |
|---|---|---|---|
| `scalar` | `blind` | the published PPO-Lagrange | the baseline |
| `scalar` | `told` | one multiplier, policy sees the context | whether seeing the context helps without a budget per environment |
| `additive` | `blind` | Lagrange function, policy does not see the context | what the budget per environment does on its own |
| `additive` | `told` | Lagrange function, policy sees the context | the full method |

**Expectation for `additive blind`, not yet measured.** The budget holds in
every environment, and the policy is more cautious than needed in easy ones.
The reason is in the update rules. A single multiplier is moved by the cost
averaged over the whole batch, so under-spending on easy environments cancels
over-spending on hard ones (`ScalarConstraint.update`). A multiplier per
environment is moved by the cost of its own environment and cannot go below
zero, so nothing cancels (`integrator_targets`). The hardest environment's
multiplier keeps rising until that environment is on budget.

## What varies between environments

Two kinds of variation, chosen by kind and not one factor at a time: one the
policy can see and one it cannot. With a visible factor the policy can adapt.
With a hidden factor it has to be cautious enough for the worst case.

| Factor | Kind | Status | Why this factor |
|---|---|---|---|
| Total number of hazards | visible, through the lidar | built | ordered; the benchmark's own levels differ by it (12, 16, 20) |
| Drag: how strongly the floor brakes the robot | hidden | built | dynamics is the most common variation in the related work (10 of 28 rows in the literature matrix), motivated by moving from simulation to a real robot |
| Spatial arrangement (clusters, corridors) | visible | if time allows | closer to how real layouts differ than a count |
| The Pathway task | — | if time allows | shows the result is not specific to one environment |

Not varied: the **budget** (the closest related paper, CCPO, Yao et al. 2023,
covers it) and the **goal radius** (the lidar does not show it, so a test on it
would measure difficulty, not generalization; it is sampled the same way in
training and testing).

**Why drag and not the friction coefficient.** The plan said friction. Measured
on CPU on 2026-11-11 (the point robot, no hazards, full thrust then none):

- **The friction coefficients do nothing.** Scaling every surface's friction
  from 0.05 to 2 times left the robot's speed unchanged to three decimals. The
  robot moves on two slide joints and has no vertical freedom, so its contact
  with the floor carries no force. (An earlier note in the task board, "a tenth
  of the friction, fourteen times the distance", was wrong: that run had the
  robot colliding with objects at the origin.)
- **What brakes the robot is the damping of its two slide joints.** The
  context value `drag` multiplies it (`crax/envs/safe_goal.py`). Top speed and
  the time to stop both go with 1/drag:

  | drag | speed after 1.6 s of full thrust | coasting distance after the thrust stops |
  |---|---|---|
  | 0.5 | 2.35 m/s | 2.6 m |
  | 0.75 | 1.80 m/s | 1.4 m |
  | 1 (the benchmark) | 1.43 m/s | 0.74 m |
  | 1.5 | 0.99 m/s | 0.34 m |

- Low drag is the slippery floor: faster and slower to stop. The yaw joint is
  left alone; its speed controller becomes unstable when its damping drops.
- Mass changes only how fast the robot speeds up and stops, not its top speed;
  motor strength only its top speed. Drag changes both, which is what a
  slippery floor does.

| Decision | Why |
|---|---|
| Drag runs from 0.5 to 1.5 | Half to one and a half times the benchmark's braking. Below 0.5 the robot crosses the arena without being able to stop. |
| The Lagrange function treats lower drag as harder (`harder_when="lower"`) | A robot that is slower to stop has more trouble avoiding hazards. This is an assumption; the multiplier's fit error (`lambda_lagr/fit_error` against `target_spread`) shows if it is false. |

## The test sets

The types are those of Kirk et al.'s survey (§4.2,
`literature/zero-shot-generalisation-survey.md`): they say where the test
values lie relative to the training values. Each test set belongs to an
**evaluation protocol** (`training/contexts/registry.py`), and each protocol is
one experiment.

**1. Hazard total** (`--evaluation_protocol hazard_total`). One training set
leaves out two bands, and each band is one test. The drag is the benchmark's.

| Total hazards | Role | Type | Combinations of hazard counts |
|---|---|---|---|
| 0 to 10 | training | | 835 |
| 11 to 13 | held out | interpolation: between trained values | 774 |
| 14 to 16 | training | | 877 |
| 17 to 20 | held out | extrapolation: beyond trained values | 980 |

| Decision | Why |
|---|---|
| Bands of three and four totals, not one or two | The policy has to bridge a real gap, and the extrapolation band shows how cost changes with distance from the trained range. |
| Extrapolation stops at 20 | Above 20 the arena is too crowded and hazards overlap. |
| Levels 1 (total 12) and 3 (total 20) are never trained on; level 2 (total 16) is | A consequence of the bands. Level 3 stays an evaluation target. |
| Extrapolation is included although standard methods are expected to fail there | Whether the policy is still safe in a denser environment than it trained on is the most relevant safety question, and it is where the learners should differ most. |
| One training set serves both tests | Every trained policy is scored on both; no extra runs. |
| Every training distribution samples only from the training set | Otherwise a curriculum could train on the test environments. |

**2. Drag** (`--evaluation_protocol drag`). The same two tests on the hidden
factor. Every hazard total is trained on.

| Drag | Role | Type |
|---|---|---|
| 1.1 to 1.5 | training | |
| 0.95 to 1.1 | held out | interpolation: the benchmark's own drag and a little either side |
| 0.65 to 0.95 | training | |
| 0.5 to 0.65 | held out | extrapolation: more slippery than any trained floor |

| Decision | Why |
|---|---|
| The interpolation band is in the middle of the trained range | Training lies on both sides of it: 0.3 of drag below (twice the band's width, between it and the extrapolation band) and 0.4 above. A band close to the extrapolation band would have little training beyond it and would be half an extrapolation test. |
| The band holds the benchmark's drag (1) | A consequence of putting it in the middle. No level is trained on in this protocol, as levels 1 and 3 are not in the hazard-total protocol; level 3 stays an evaluation target. The interpolation test is then a test on the benchmark's own physics. |
| Extrapolation is towards low drag only | Low drag is the unsafe direction. A stickier floor makes the robot slower, not less safe. |
| The extrapolation band is narrow in drag but not in effect | From top speed the coasting distance goes with 1/drag²: 2.4 to 4 times the benchmark's in the extrapolation band, against at most 2.4 times in training. |

**3. Combination** (`--evaluation_protocol combination`). Every hazard total
and every drag is trained on, but never many hazards on a slippery floor.

| | drag 0.5 to 0.9 (slippery) | drag 0.9 to 1.5 |
|---|---|---|
| totals 0 to 10 | training | training |
| totals 11 to 20 | **held out** | training |

| Decision | Why |
|---|---|
| The two factors are hazard total and drag | One visible, one hidden. Passable against solid hazards was considered and rejected: both add to the total, which is capped at 20, so "many of both" is the extrapolation test again. |
| Its own training set, not the drag protocol's | Holding the corner out of training removes slippery floors with many hazards from training. Drag's own tests would then be confounded with the combination. |

In the code: `--evaluation_protocol <name>` makes the training distribution
sample the protocol's training contexts only, and `--reference` makes it sample
all of the protocol's contexts. The code checks that training set and test sets
do not overlap and together are all of the protocol's contexts. A training
distribution that names a held-out level (`level:3`, `staged:1,2,3` under
`hazard_total`) is refused.

## Measurement

| Decision | Why |
|---|---|
| Cost on a test environment is compared with the budget | The budget is a fixed line; no reference is needed. |
| Return on a test environment is compared with a **reference run**: the same learner trained on all of Ω, test environments included | A harder environment lowers return even when the policy generalizes well. The reference shows what was achievable there. |
| Cost and return are reported per hazard total, not as one average over a test set | An average can hide one total over budget and another under. That is the original finding of this project. |
| The median episode cost and the share of episodes over budget are reported next to the mean | In earlier runs the mean was carried by a few very costly episodes. |
| Return is always read together with cost | A policy that stands still meets the budget. |
| Three seeds per arm | Two runs of the same setup differed as much as different setups did (`experiments/2026-10-18_goal_point_learners_vision_750M.md`). |
| The share of episodes in which the robot left the arena is reported | Under the stock solver a robot could be thrown through the fence (`known_defects.md`). Outside it there is neither goal nor hazard, so such an episode looks cheap. With the solver fixed the share should be 0; it stays in the tables as a check. |

How it is measured. A protocol cuts its contexts into **cells**: one hazard
total (21 cells), one band of drag 0.05 wide (20 cells), or for the combination
a hazard total by a band of drag 0.1 wide (210 cells). After training,
`python -m training.score --group <group>` rolls every run's final policy out
in every cell (128 episodes per cell by default, 32 for the combination so
that its 210 cells fit in one rollout; the same contexts for every run) and
writes one table per run to `runs/scores/<group>/`. The evaluations
logged during training (`evaluation/<region>/*`) are averages over a whole
region; they are for watching a run, not for the result.

The main figure (`scripts/analysis/generalization_figures.py`): one column per
learner, the factor on the horizontal axis, the held-out bands shaded. Top row
cost with the budget as a line, bottom row return. Every panel holds the run
with the bands held out and its reference run. For the combination the figure
is a grid, hazard total by drag, with the held-out corner outlined.

## Curriculum methods

They come after the baseline. The training set and the test sets stay fixed,
so a curriculum method is only a different way of sampling from the same
training set and is scored on the same test sets.

## Open points

- Measure `additive blind`. Its expected behaviour above is a prediction.
- Check that the budget is hard to meet at totals 11 to 13. If it is easy
  there, the interpolation test says little about cost.
- Check that drag changes cost at all. The ranges were chosen from how the
  robot moves, not from how a trained policy's cost responds. The reference
  runs of the drag baseline show it: if their cost is flat over drag, widen
  the range before reading the tests.
- Check `evaluation/*/episode_outside_arena` in the first runs. It should be 0.
- The physics changed on 2026-11-11 (goal-point's solver, `known_defects.md`).
  What is known from October (the four points of the task board) was measured
  under the old solver and is not re-measured. The first baseline shows
  whether a single multiplier still over-spends on level 3.
- Report to the supervisors: levels 1 and 3 are held out of training in the
  hazard-total protocol, and the benchmark's drag in the drag protocol; the
  hidden factor is drag, not friction, and it
  changes the benchmark's physics (`upstream_notes.md`, items 26 to 28).
