# Automated curriculum learning on CRAX

Giuseppe Hannen's graduation project. Start here. This file is the goal, the
plan and the state of the work; the reasons live in the documents it points to.

## Goals

Three goals, each with the idea that serves it and the questions it asks
(presented to the supervisor 2026-10-08). Where each stands is in *The plan*.
Notation: context ω ∈ Ω, training distribution q, deployment distribution w,
return R(π, ω), cost C(π, ω), budget d, policy parameters θ.

| | goal | idea | what we want to explore |
|---|---|---|---|
| 1 | **Which curriculum methods suit which environment spaces.** | One distribution over Ω per *challenge*, a property of the space that makes teaching hard; every method on each. The challenges are TeachMyAgent's, read again with a constraint (table below). | Does a method that copes with a challenge on reward still cope when the budget binds? Where does each method put its mass, and does that predict reward and cost on w? |
| 2 | **Explain the differences, and use the explanation to build a method.** | Subspaces Ω_x, Ω_y, … ⊆ Ω where skill x, y, … is *needed* to do well. They overlap: a context in Ω_x may need y too; what defines Ω_x is that x is required there. Evaluate every method on each subspace at checkpoints through training. | Which skills does each curriculum develop, in what order, and does the order predict performance on w? Does the λ mechanism explain the staged failures? |
| 3 | **Where does generalised learning happen during training.** | Data attribution, TracIn-style (`literature/data_attribution_reading.md`): the gradient a training record induces under PPO-Lag, dotted with the gradient of a target we care about. A return target and a **cost target**, kept separate; the target's expectation over the current buffer, a future policy's buffer, or held-out contexts. | Does a cost-target score predict one-round changes in cost, and better than the cost advantage alone? Where in the state space do reward and safety conflict, and how does that band move? Which contexts produced the data that helped on held-out contexts, and when? |

### Goal 1: the challenges

A thesis chapter that says: *here are the challenges a curriculum faces when
the learner is constrained; here is how each method behaves on each.* The
structure is TeachMyAgent's (Romac et al. 2021): one task distribution per
challenge, each isolating one property of the task space, every method run on
each. The challenges are **TMA's own, asked again with a constraint**, plus
the one thing a constraint adds that TMA has no row for.

The constraint changes the task space in exactly one way: difficulty becomes
two-dimensional. A context is easy or hard for *reward*, and separately easy
or hard for *staying under the budget*. TMA's challenges are about one
landscape; ours are about two.

| TMA challenge | on the reward landscape (TMA's reading) | on the constraint landscape (what the budget adds) | what it needs on `safe_goal_point` |
|---|---|---|---|
| mostly infeasible | most contexts cannot be solved: no policy reaches the goal | most contexts cannot be solved *safely*: a policy can reach the goal, but no policy does so with cost ≤ d | hazard density high enough that the cheapest path to the goal costs ≈ d; current Ω with the run's d set accordingly |
| mostly trivial | most contexts are solved by any policy: the goal is reached without learning | most contexts are safe for any policy: nothing near the path incurs cost, so the budget never binds and λ decays to 0 | low hazard counts; current Ω |
| rugged difficulty | a small change in ω produces a large change in how hard the goal is to reach | a small change in ω produces a large change in how hard it is to stay under d; the two landscapes need not have their cliffs in the same places | a layout dimension with cliffs in cost but not in reward, or the reverse |
| forgetting student | the policy loses a skill it had | the policy loses a skill it had, *or* the multiplier λ loses its calibration: too small, the agent violates; too large, it stops moving | staged q with a large jump between stages; observed twice (Results) |
| diverse students | the method must work for different learners (SAC, PPO; different bodies) | the method must also work for different constraint handlers: Lagrange, PID-Lagrange, Sauté, each with its own failure mode | the `--alg` flag |
| no expert knowledge | the method gets no initial distribution, no target, no mastery threshold | unchanged; the budget d is given to the learner, not to the curriculum | — |
| **(new) the two landscapes disagree** | — (TMA has one landscape, so this cannot arise) | the contexts where reward is easy to earn are the contexts where cost is hard to avoid, or the reverse; a curriculum that selects by reward moves into cost | a layout dimension: hazard density *on the shortest path to the goal* (Koprulu 2025 §4.2 observed CURROT doing this) |

The seventh row is the general form of "reward conflicts with safety"; the
first row is the general form of "tight budget". The concrete things (hazards
on the shortest path, budget near minimum cost) are how a challenge is
*instantiated* on one suite, and belong in the experimental design, not in
the definition. The constraint column is what we test; the reward column is
there so each constrained reading can be checked against the original.

Every test is a question about where the realised curriculum q̂ put its mass,
read against reward and cost on w — both logged every round already. Rows 1,
2 and 4 need nothing new. Rows 3 and 7 need a layout dimension. Any physics
dimension needs a policy that can tell contexts apart (*Later*: context
observation).

Two statements that frame the chapter:

- **The constraint is on deployment, not on training.** Training under q
  enforces $\mathbb{E}_{\omega \sim q}[C] \le d$; that says nothing about w or
  about any single ω. Seen twice: the velocity-ant collapse (λ calibrated to
  level 1), and goal-point's `level:3` arm, at budget on level 3 and at a fifth
  of it on uniform — one policy, one constant margin, two very different costs.
  We state the constraint under w and report the worst bin over Ω next to the
  mean. A definition, not code.
- **Training-violation regret is not our objective.** It counts cost incurred,
  not constraints failed. In CRAX cost is per step and the constraint is on
  the episode sum, so an agent can incur sub-budget cost, learn from it and
  never violate.

**How we got here** (2026-10-05 → 10-08), so the next change of mind has
something to push against:

1. *Notebook 2026-10-05* (`docs/notebook/1-3.xml`): four safety situations
   from intuition — reward vs safety, few contexts feasible, rare hard tail,
   context-dependent safety.
2. *Literature pass 2026-10-06* (`literature/synthesis.md`): two of the four
   had no reported evidence as stated (few feasible, rare hard tail). What
   *is* reported on hard contexts is two opposite failures — over-conservatism
   and violation-dominated exploration (CRAX §5.1–5.2). λ fragility is
   documented in-distribution and offline→online, never at a curriculum
   switch. Prior art: Koprulu, Simão, Jansen, Topcu, ICLR 2025 (SCG,
   confirmed by Giuseppe as "Safe Curriculum Generation").
3. *Reframe 2026-10-07*: five concrete challenges (misaligned reward/cost,
   over-conservatism, violation-dominated exploration, constraint shift, λ at
   a switch). Giuseppe: two of these are *outcomes*, not configurations — you
   cannot set "conservatism" before training. Collapsed to four configurable
   ones; conservatism vs violation became the analysis question asked of every
   result.
4. *After the supervisor meeting 2026-10-08*: TMA's challenges are general
   properties of the difficulty landscape ("mostly infeasible", "rugged") and
   need no defence; ours were concrete and would need a deep review to defend.
   Giuseppe: ask instead *what does a constraint add to TMA's list?* Answer:
   one new row (the two landscapes disagree) and a second reading of each
   existing row. The concrete challenges of step 3 become instantiations.
   "λ at a switch" is a property of a method class, not of a task space; it
   moves to goal 2 (explaining method differences). "3D" is not the axis that
   matters; nothing above depends on it.

## The plan

In this order (Giuseppe, 2026-10-08).

1. **Prioritized Level Replay** (Jiang et al. 2021) as the first curriculum
   method. Chosen because it needs no target distribution, no mastery
   threshold and no context observation. **Code reviewed and committed
   2026-10-09** (242f0a6): `design/prioritized_level_replay.md`;
   `--context_distribution plr`; the trainer ships every transition's reward
   advantage to the round hook (`LearningSignals`); new
   `training_curriculum/{intended,value_loss}/<d>` heatmaps and
   `distribution/*` scalars on every arm. GPU smoke on goal-point passed
   (zero recompiles, 85 k SPS, buffer dynamics as designed). Check: trains on
   goal-point at least as fast as uniform, and q̂ moves where the value loss
   says — being answered by the run below.
   **Done (2026-10-09):**
   `experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md`, plan item
   3's three arms at 500 M (without the OOD target, plan item 2, not built
   yet); W&B group `goal_point_uniform_staged_plr_500M_v2`. Result in *Results
   so far*. Consequence for the plan: PPO-PID first — every arm's end state is
   set by the Lagrange integrator; then PLR with the cost-critic score.
2. **An out-of-distribution context space for goal-point**
   (`design/goal_point_ood.md`): contexts outside the training Ω, as values on
   the existing dimensions (counts above the cap, goal size below the smallest
   level). Registered as a distribution so it is an evaluation target like any
   other.
3. **Uniform vs staged vs PLR, 500 M steps, goal-point**, every arm evaluated
   on the target (uniform over Ω) and the OOD target (uniform over 2) at
   `--num_evals` points through training. Both targets go in via
   `evaluation_wrap_env_fns`, which already exists. Result: the two curves per
   arm, and when a gap between them opens. Write-up in `experiments/`.

Then: PPO-PID on the same arms (λ is calibrated to the stage that just ended,
seen twice); skill probes as extra evaluation targets on the same runs; the
cost-critic pre-check for attribution on any of them.

### Done

- Literature pass on the six candidate properties (2026-10-06) →
  `literature/synthesis.md`.
- Goal-point baseline, uniform vs staged vs level:3 (2026-10-06) →
  `experiments/2026-10-06_goal_point_staged_vs_uniform.md`. Staged is never
  safe: λ decays to 0 on level 1. Only `level:3` meets the budget.
- Slots spread over the episode (2026-10-06): ~655 episodes complete per
  round instead of 8192 every 12.5; reasons in the docstring of
  `training/contexts/wrapper.py`. *Open:* `MetricsLogger` duplicates the
  round hook at our cadence; one `progress_fn` call per round instead.

### Later, with the reason

- **Back the constrained challenges with evidence.** The table above is TMA
  with a constraint added to each row. That is a hypothesis, not a result: for
  each row we need to show that the constrained reading is a difficulty
  someone has reported, or demonstrate it ourselves, or drop the row. The
  review: TMA's own sources, CARL, PAIRED/PLR/ACCEL, the safe-RL benchmarks'
  design sections, Koprulu 2025 in full. Then
  `design/constrained_challenges.md`, one section per surviving row with its
  instantiation on goal-point and a predicted method ordering. After the
  first comparison, so the predictions are made with one curriculum result.
- **Let the learner see its context** (`design/context_observation.md`).
  Needed for physics dimensions of Ω and for SCG, whose Def. 3.1 assumes it.
  Told (ω in the observation) vs inferred (last cost and reward, with or
  without memory). Changes the benchmark's observation space: Tristan first.
- **SCG and CURROT as baselines**, after context observation.
- **Attribute cost to training records.** First the cost critic's explained
  variance on an existing run (if the critic is unfit, every score is noise).
  Then: does the cost-target score predict one-round changes in cost better
  than the cost advantage alone; where in the state space reward and safety
  conflict; which contexts produced the data that helped on held-out
  contexts. Order and reasons in `literature/data_attribution_reading.md`.
  Needs a curriculum result to attribute.

### Deferred, with the reason

- **Ω as the union of level boxes.** Motivated by button's infeasible corner
  and a cleaner "uniform over Ω". Goal-point's corner turned out infeasible
  too, and is handled by the cap on the total count (Decisions) — one
  constraint on the box rather than a union of boxes. Button's corner would be
  handled the same way (a cap, or a cap depending on `placement_extent`). The
  challenges define their own distributions, so this is off the path.
  Design, kept for when a second suite is used: Ω = B₁ ∪ B₂ ∪ B₃; uniform =
  pick a level with probability ⅓, then uniform in its box; widen a box only
  along the dimensions the ladder varies, halfway to the neighbouring level;
  probe every box corner for feasibility. Files that know Ω is one box:
  `training/contexts/space.py`, `distributions/uniform.py`
  (`uniform_log_density`), `registry.py` (15 entries),
  `training_curriculum.py:78` (histogram bounds), `dashboard/view.py:174`
  (Ω text), `tests/test_suites.py`. The velocity uniform arm changes meaning
  and is rerun afterwards.
- **Staged vs uniform on the other suites** (velocity × 5, height, push, lift,
  pathway). CPU-checked and GPU-smoke-tested; commands in
  `experiments/2026-09-22_velocity_suites_staged_vs_uniform.md`. The
  threshold-type Ω is not the research question.
- **750 M velocity-ant repeat** (W&B group `velocity_ant_staged_vs_uniform_750M`):
  read with Tristan, add to the 500 M write-up.

### Ideas parked (2026-09-24, no code)

- A **generative distribution** for goal-point: procedural layout generation,
  Φ = generator weights. Fits `ContextDistribution` as is; layout in ω needs a
  feasibility check or repair at reset.
- A **learning-signal study**: per-context PPO statistics (value loss,
  explained variance, clip fraction, KL) against the next evaluation delta on
  w — *when* does useful learning happen. Logging only; rides any experiment.
- **Skill probes** (Giuseppe, 2026-10-06; the idea behind goal 2, see the
  goals table). Two things the table does not say. The diagnostic is the lag
  between "the curriculum samples $\Omega_s$" and "the agent becomes good on
  $\Omega_s$": it tells whether the curriculum caused the skill, followed it,
  or never trained it. And a subspace is a region of the world, not a skill:
  each probe needs a sentence saying which behaviour it requires and why the
  alternative does not pass. Candidate probes on goal-point: dense hazards
  between start and goal (threading), a far small goal (navigation), many
  collidable blocks (contact avoidance). No code: probes are box distributions
  passed to `evaluation_wrap_env_fns`.

## What is built

All on `main`, CPU-tested: `tests/test_contexts.py`, `test_context_training.py`,
`test_dashboard.py`, `test_suites.py` — 120 tests, ~30 min.

**Contexts** (`training/contexts/`, `crax/envs/context.py`; how it is wired:
`design/contexts_package.md`). A context ω is a value on the `num_envs` axis.
A `ContextDistribution` samples in-program from frozen parameters φ and
updates φ on the host between rounds; one compiled call is one round
(`design/training_round.md`). `ContextualAutoResetWrapper` draws a new ω when
an episode ends. Distributions `uniform`, `level:<n>`, `staged:<n>,<n>,...`,
one file each. Every registered environment reads every knob from
`state.info["context"]`, always; `reset(rng)` is
`reset_with_context(rng, default_context())`; there is no "if a context is
present" branch anywhere. Invariant, tested per suite: a level-1 environment
reset into the level-3 context is step-for-step identical to a level-3
environment, observation included.

**All 15 environments of the nine suites have an Ω**
(`design/context_spaces_by_suite.md`). Value-typed suites read a number
(velocity threshold, `max_height`, `goal_velocity`, `max_gap`, lift's per-foot
mask). Hazard suites use the **union model** (`design/hazard_activation.md`,
`crax/envs/hazard_union.py`): one MuJoCo model with every hazard any level
needs, a per-episode 0/1 activation per hazard; an inactive hazard is parked at
`PARKING_XY` and ignored by cost, lidar, compass and placement. Goal: Ω = four
per-group counts + `goal_size`; every level has 34 hazard bodies (18
collidable). Reach: `active_hazards ∈ {0..10}`. Circle: `active_cylinders ∈
{0,1,2}` + two boundaries. Button: two counts + `gremlin_travel` +
`placement_extent`.

**Logging and dashboard** (`training/dashboard/`, `design/dashboard.md`).
Every logged key must be in the metric registry (`metrics.py`: kept with a
one-sentence provenance, or dropped with a reason; anything else raises); the
registry names no agent. Every run logs the sampled (≈ q) and experienced (q̂)
context distribution per round (`training_curriculum/*`) and is evaluated on
the deployment distribution (`evaluation/deployment/*`, level 3 by default) and
on uniform (`evaluation/uniform/*`). q and q̂ differ because episode length
depends on ω and episodes straddle rounds; always read together
(`design/intended_vs_realised_curriculum.md`). `--wandb_group` creates one
W&B view per experiment group, with generated text, and refuses a run whose
flags differ from the group's.

**Performance** (`training/performance/`, `performance_measurement.md`).
Per-round throughput, compiles and XProf traces go to W&B. Only compiles ≥ 1 s
count (`SUBSTANTIAL_COMPILE_SECONDS`): JAX's few-ms host helpers are not
recompiles. Training is launch-bound; throughput is ~linear in `num_envs` up to
a knee at 8192 on this GPU
(`measurements/2026-09-20_num_envs_sweep_and_launch_bound.md`). A new compiled
program costs ~50 s; a per-slot value costs nothing. Throughput at 8192 envs
(SPS): velocity ant ~370 k, lift ant 438 k, lift spider 342 k, pathway 133 k,
**goal point 85 k** (union model; stock level 1 was 228 k), push 45 k, height
29 k (height is on Brax's `generalized` backend; push's placement loop runs
every step — both stock).

## Results so far

**Goal point, uniform vs staged vs PLR, 500 M steps, one seed**
(`experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md`). PLR ends
best on both targets (level 3: 15.3 return / 73 cost; Ω: 21.3 / 17, under
budget at 20 of 21 evaluations) against uniform (12.5 / 108; 16.0 / 26) and a
staged arm whose policy is motionless from round 470 (2.7 / 22 — a budget met
by doing nothing). No moving policy is under budget on level 3; every arm is
on budget on its own $q$ — the framing statement, a third time. **The Lagrange
integrator sets every end state**: staged's λ climbs 0.3 → 40 after the
level-2 switch (the velocity collapse on a suite whose student sees its
context — λ alone suffices); uniform's λ hits 0 at round 388, then rises to 22
and is still rising at 500 M, costing a third of the return at round 546;
PLR's λ peaks at 10.5 while its mass shifts to denser layouts, then rests at 0
for 120 rounds with cost under budget. PLR tilted every hazard count 6–14 %
towards level 3 and held it, on a reward-critic score that varies only 1.6×
across its buffer. Next: PPO-PID, then PLR on the cost critic.

**Goal point, staged vs uniform vs level 3, 100 M steps, one seed**
(`experiments/2026-10-06_goal_point_staged_vs_uniform.md`). **Its evaluation
numbers are about half the true values** (Known defects, first item; found
2026-10-08): on level 3 after 100 M, as logged, staged 14.5 reward / 135 cost,
uniform 16.9 / 138, level:3 5.1 / 22 (budget 25) — so `level:3` was in truth
at roughly twice the budget, not under it, and the other two at ~10×. The
qualitative reading survives, the budget comparison does not; the re-run is
`experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md`. Staged does not
collapse at either switch — reward holds — but
learns to drive through hazards on level 1 (λ → 0), and after the switches λ
climbs from zero too slowly to ever enforce the budget (75 and rising at the
end). Only training on the deployment distribution is safe; uniform and staged
have the reward. Throughput 85 k SPS in every arm, zero recompiles; the union
model is 2.7× slower than stock level 1 (12 collidable bodies in every slot).

**Velocity ant, staged vs uniform, 500 M steps, one seed**
(`experiments/2026-09-20_uniform_vs_staged_velocity_ant.md`). Uniform solves
level 3 (reward ≈ 20, under budget). Staged learns to run fast on level 1 for a
third of the budget; at the switch to level 2 that gait violates every step, λ
goes 1.5 → 405 within a few rounds, the policy stops moving, and λ decays too
slowly to recover. Two readings: (i) a policy that cannot see ω learns one
behaviour for all of Ω, so an "easier" level is a contradictory task, not a
sub-task; (ii) PPO-Lagrange's dual variable carries the old distribution across
the switch. Not evidence against curricula: one seed, one suite, one algorithm,
and a threshold Ω. *Read with the goal result:* (i) explained the collapse —
goal's student sees its context and does not collapse; (ii) holds on both
suites, with opposite sign.

**What the two together say about the constraint.** Neither curriculum arm
is safe on w at the end of its budget; the arm trained on w is. Training under
q enforces the constraint under q (The goal, first framing statement) — this
is the mechanism, seen twice.

## Decisions

- **Ω is the environment's physics and layout, not its cost threshold** (with
  Tristan, 2026-09-22). A run keeps one threshold. The threshold, `max_height`,
  `max_gap` and foot-mask spaces stay as pipeline smoke tests; experiments use
  layout and physics Ω. Physics dimensions (friction, mass, gravity, limb
  length) are per-slot values via `sys.tree_replace` in `step` — verified,
  `design/what_can_vary_per_slot.md` §2c.
- **First experimental suite: `safe_goal_point`** (Tristan: low training
  variance, so a curriculum effect is detectable with few seeds).
- **Contexts are values, never shapes.** A count becomes pad-and-mask;
  structure becomes a union model (`design/per_slot_constraints.md`).
- **Uniform over Ω for the hazard suites produces layouts no level has** (12
  cylinders *and* 4 cubes). Intended; say so in the thesis.
- **Goal-point has one goal, not two** (Giuseppe, 2026-10-06). Stock
  `difficulty.py` sets `goal_count: 2` at every level, but the observation's
  goal compass reads only the first goal (`safe_goal.py` `TODO handle multiple
  goals`); the second is visible only through the coarse goal lidar. One goal
  is what the agent actually sees. Change: `goal_count: 1` at every level
  (before the first goal-point run; flag to Tristan).
- **The arena walls are a fence, not hazards** (Giuseppe, 2026-10-06). Stock
  CRAX builds the four walls of goal and push through the hazard pipeline
  (`outer_wall` → four `fixed` `rect` hazards), so they were scanned into the
  hazard lidar as four centre-points, competed for the eight compass slots, and
  cost 3 per contact step — verified on the running environment (circle and
  button have no walls). Safety-Gymnasium's Goal has no walls; its placement
  extents bound the arena. Decision: walls are static collidable geometry in the
  model and nothing else — not observed, not costed, not a hazard
  (`crax/envs/arena.py`). Invariant: *every hazard the environment knows is one
  of its hazard specs'*; `fixed` has left the hazard pipeline. Verified: full
  thrust into the fence for 400 steps — robot stays in, cost 0, hazard lidar
  dark. Observation shape unchanged; goal's union model has 30 hazard bodies.
- **Goal's Ω is capped at 20 active hazards in total** (Giuseppe, 2026-10-06).
  The box's corner (30 hazards) does not fit the 5 m square; goal's placement
  nudges instead of raising, so the failure was silent. Measured on CPU: the
  packing quantity is total keepout area — L1 6.0 m², L2 6.3 m², L3 9.1 m²; at
  L3's area 4 % of resets have two hazards overlapping, one kind more and half
  do; `Uniform(box)` hit an overlap in 11.5 % of draws. `ContextSpace` gained
  `integer_total_cap`: Ω = {counts : Σ ≤ 20} × goal_size, so level 3 is the
  densest point of Ω and every level stays inside Ω (the invariant
  `SuiteContexts` enforces; a cap of 16 would have put w outside Ω). Uniform
  samples one admissible integer tuple from an explicit table — which also fixes
  a pre-existing bias: rounding a uniform float gave the endpoints half weight.
  Residual overlap under the cap ≈ 2 %, the same crowding level 3 has.
- **Trainer contract.** `training/rounds.py` `RoundHook` (`extra_fields`,
  `observe`, `on_round_end`) and `evaluation_wrap_env_fns` are the only two
  trainer additions; `ppo_lag` forwards them. Another PPO-family trainer needs
  the same two lines (plan item 3).

## Known defects

- **Every context-run evaluation between 2026-10-06 and 2026-10-08 reported
  about half an episode** (fixed 2026-10-08, uncommitted). The slot
  desynchronisation added on 2026-10-06 gives each slot a random head start on
  its step counter at the first reset; the evaluation environment was built
  through the same stack, so every evaluation episode was cut off after
  `1000 − head start` steps while `avg_episode_length` read 1000. Affects
  `evaluation/*` in the 2026-10-06 goal-point experiment and the first launch
  of the 2026-10-08 one; training-side metrics (`episodic/*`, `training/*`,
  `training_curriculum/*`) and the velocity-ant results (pre-date the spread)
  are unaffected. The evaluation stack (`wrap_for_context_evaluation`) now
  never spreads; `tests/test_contexts.py::test_evaluation_episodes_start_at_step_zero_and_are_summed_whole`
  holds the Evaluator to a step-0 start.

- **The goal-point baseline (2026-10-06) ran with synchronised slots**: 80
  env steps per slot per round, 1000-step episodes, no early termination → all
  8192 slots reset together every 12.5 rounds. Every `training/*` curve has a
  12.5-round sawtooth (the cost critic's error grows with position in the
  episode), `episodic/*` has 12 points, and the stage switches took effect 12
  and 11 rounds late. Fixed for context runs (plan item 2b); stock runs still
  behave this way. Re-run before comparing against PID.

- **Button's Ω has an infeasible region**, and it is not a count cap
  (measured 2026-10-06, 2 m square, 0.5 m orbits, 48 resets each): 12 blocks +
  4 gremlins always fits, 4 blocks + 8 gremlins never does — a gremlin's
  keepout (0.64 m) is five times a block's area (0.28 m). The constructor's
  area bound passes all 117 tuples, so it is not the right check either. The
  constraint is a *weighted* sum (keepout area) depending on `gremlin_travel`
  and `placement_extent` — i.e. a feasibility predicate over the whole ω, not
  a cap on the integer part. `tests/test_suites.py::test_every_key…[safe_button_point]`
  fails on it (the exact-uniform sampler hits the corner reliably where the old
  rounding sampler hit it ~0.7 %). **Do not start a uniform button run**; off
  the path until a second suite is used.
- `safe_push`: 1000-step episodes with no early termination, so almost no
  training episode completes per round at small budgets and sampled /
  experienced stay at the initial stage. Expected, not a bug.
- **The stock `MetricsLogger` drops `episodic/*` for some rounds** (found
  2026-10-09, not fixed — upstream). Its two `jax.debug.callback`s
  (`update_env_metrics`, `update_train_metrics`) run on different threads and
  share unlocked state; when the second flushes while the first is still
  appending, that round's episodic metrics are lost (seen: 3 of 19 keys
  logged at one round, the rest carried into the next). Our
  `training_curriculum/*` metrics are not affected (one callback, host-side
  completion). Tests must not compare the two paths' outputs.

## To raise with Tristan

Upstream bugs found, and places where our changes alter the benchmark:

1. `get_environment("safe_velocity_*", level=n)` ignored the level since
   upstream `583e7fd` (May 2026): stock level-2/3 velocity results since then
   are level 1. Fixed in `crax/envs/difficulty.py`.
2. `crax/fluid.py` `jp.clip(a_min=)` broke the swimmer on the installed JAX.
   Fixed (`min=`).
3. Goal union model: hazard size within a group is the union's max (level 3's
   discs 0.4 not 0.35, solid cylinders 0.3 not 0.25); every level carries 30
   hazard bodies (12 collidable). Measured cost: 85 k SPS vs 228 k for the
   stock level-1 model at 8192 envs (2.7×), the same for every layout — parked
   collidable hazards cost what active ones do.
4. Goal's `goal_type` default changed cube → cylinder (the context's
   `goal_size` needs it).
5. Circle level 1 observes 60-d (hazard blocks present, zero-valued); stock
   level 1 observed 28-d. Levels 2–3 were 60-d already.
6. Stock `training/curriculum.py` warm-starting circle L1 → L2 copies a 28-d
   policy onto a 60-d observation (`ppo/train.py:815-836` checks only the
   normaliser) and raises.
7. Button's cylinders are `cube` groups under MJX (stock; now visible as
   `active_collidable_cubes`).
8. Stock `AutoResetWrapper` never re-samples a layout: each slot's layout is
   frozen for the whole run. Affects the paper's baselines.
9. Stock `train_curriculum.py` restarts `train()` per level, which resets λ at
   every switch — the paper's curriculum results include that reset.
10. The paper says Point control dt 0.008 s; the code has 0.02 × 4 = 0.08 s.
11. Goal has `goal_count: 2` but the compass observes only goal 1 (a `TODO` in
    `safe_goal.py`). We run with one goal (Decisions); is two intended?
12. The arena walls of goal and push were hazards: in the lidar as
    centre-points, compass candidates, 3 per contact step. Now a fence
    (Decisions). Changes the observation contents and cost of both suites.
13. Goal's hazard placement never fails: when no candidate fits it nudges and
    clips, so overcrowded layouts (hazards overlapping) are silent. Level 3 is
    at the limit — 4 % of its resets have one overlapping pair (with the union
    model's 0.4 m discs; stock L3 has 0.35 m).
14. On every suite whose episodes run to the step limit (goal, push, circle,
    button …), all parallel environments reset together and stay in lockstep,
    so each PPO batch is one slice of the episode rather than a sample of it,
    and `episodic/*` is only ever logged at the common reset. Affects the
    paper's baselines on those suites; the fix is a random initial step offset
    per slot.
15. `training/logger.py` `MetricsLogger` has a data race: its two
    `jax.debug.callback`s run concurrently and share unlocked buffers, so
    `episodic/*` is silently dropped on some rounds (Known defects). A
    `threading.Lock` around the three `update_*` bodies fixes it.
16. The compiled training round's only per-round input is `env_state`, so a
    curriculum's parameters φ travel in `env_state.info` and the context
    wrapper has to lift them out before the vmapped stack (they have no slot
    axis). A per-round-parameters argument on `training_step` /
    `generate_unroll` would make this by construction.

## Vocabulary

| term | meaning |
|---|---|
| context ω ∈ Ω | the per-episode parameters of the *world* a suite instantiates — physics (friction, mass, gravity), layout (hazard count, positions). Not the cost threshold: a run keeps one threshold. A value on the `num_envs` axis, never a shape. |
| distribution (not "teacher") | decides which ω each parallel environment gets when its episode starts. `uniform`, `level:3`, `staged:1,2,3`. |
| r | uniform over Ω; the domain-randomisation baseline. |
| w | the deployment distribution; what the agent is evaluated on (level 3 by default). |
| round | one PPO training step; with a context distribution, one compiled call. |
| q vs q̂ | *intended* curriculum (what the distribution samples at round k) vs *realised* (what the gradient came from, per transition). In the logs: `sampled` vs `experienced`. They differ when episode length depends on ω and because episodes straddle rounds. Always shown together. |
| challenge | one row of the table in *The goal*: a property of the task space, TMA's or the constraint's addition, realised as one distribution over Ω that isolates it. Replaces "situation" (2026-10-08). |

## Running

```bash
# manual curriculum
.venv/bin/python -m training.train_env --env_name safe_goal_point --alg ppo_lag \
  --context_distribution staged:1,2,3 --deployment_distribution level:3 \
  --num_envs 8192 --num_timesteps 50_000_000 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --measure_performance --skip_rollout --skip_video --store_model false \
  --wandb_group <experiment_group>

# uniform: same, with --context_distribution uniform; fixed level: level:3; PLR: plr
```

Runs in one `--wandb_group` are one experiment and must share every flag but
the distribution; the first run creates the group's W&B view.

Every context run is evaluated on the deployment distribution
(`evaluation/deployment/*`, level 3 by default) and on uniform (`evaluation/uniform/*`),
and logs the intended, sampled and experienced context distribution and the
reward critic's error per context every round (`training_curriculum/*`, as W&B
histograms). What is logged and what each key
means is `training/dashboard/metrics.py`; the W&B view for an experiment group:

```bash
.venv/bin/python -m training.dashboard --group <wandb_group>
```

CPU tests: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_contexts.py tests/test_context_training.py tests/test_dashboard.py tests/test_suites.py -q -p no:cacheprovider`
(`test_suites.py` runs every check per registered suite; add a suite to the registry and it is covered.)

## Rules

- No GPU runs without asking; the machine is shared.
- W&B is mandatory (`WANDB_API_KEY` in `~/.bashrc`).
- Contexts are values, not shapes: a new compiled program costs ~50 s.
- Anything that changes the upstream benchmark's behaviour is flagged to Tristan.
- Clear names, no abbreviations, one file per distribution.

## Documents

```
docs/acl/
  README.md                          this file: goal, plan, state
  performance_measurement.md         how to profile a run (XProf + W&B)
  profiling_options.md               tool survey (historical)
  literature/
    protocol.md                      (plan item 0) question, candidate properties, queries, seeds, screening rules, extraction form
    deep_research_prompt.md          the exact prompt both models got
    synthesis.md                     merged verdict per property, NEW properties, by-product A, run disagreements — the evidence column for design/constrained_challenges.md
    data_attribution_reading.md      (goal 3) Giuseppe's reading: TracIn, Hu et al. local attribution, lookahead + replay-LOO, RFT-Inf, PIToD, influence functions; our cost-target extension and three experiments
    runs/{claude,chatgpt}/           each run's screening.csv, evidence.csv, by_product_A.csv, synthesis.md, free_text.md
    *-deep-research.md               the raw outputs as pasted
    split_run.py                     raw output → runs/<model>/ files
    verify_references.py             resolves every paper_id (Semantic Scholar / arXiv / Crossref) and compares titles; fabricated ids and real-but-wrong ids both fail
  design/
    prioritized_level_replay.md      (plan 1) PLR as a ContextDistribution: what the paper does, what changes on a continuous Ω with frozen φ, diagnostics
    goal_point_ood.md                (plan 2) the out-of-distribution context space for goal-point, and why those values        [to write]
    constrained_challenges.md        (goal 1, later) TMA's challenges with a constraint, plus the one the constraint adds; each as a distribution over Ω   [to write]
    skill_probes.md                  (goal 2, later) probe subspaces of Ω where one skill is needed; skill-acquisition curves against q̂   [to write]
    data_attribution.md              (goal 3, later) the cost-target attribution: where the gradient is taken, which checkpoint, how records carry their context   [to write]
    context_observation.md           (later) told vs infer: decision and reasons   [to write]
    contexts_package.md              how training/contexts works and how it is wired into the trainer
    intended_vs_realised_curriculum.md   why q ≠ q̂, and why both are reported
    training_round.md                why one compiled call is one training step
    per_slot_constraints.md          what may differ between parallel environments (the rule)
    what_can_vary_per_slot.md        which mjx.Model fields are values vs structure; what a union model costs (probed)
    hazard_activation.md             how a hazard COUNT becomes a value: union model, activation, parking, the four consumers
    context_spaces_by_suite.md       Ω for each of the nine suites
    dashboard.md                     what a reviewer needs from W&B; the metric registry; the generated view
  experiments/                       one file per experiment: question, commands, results
    2026-09-20_uniform_vs_staged_velocity_ant.md      the 50 M / 500 M result
    2026-09-22_velocity_suites_staged_vs_uniform.md   the other five agents (CPU part; deferred)
    2026-10-06_goal_point_staged_vs_uniform.md        plan item 2: the three arms on goal-point, 100 M
  measurements/                      dated measurement logs
    2026-09-19_baseline_goal_point.md
    2026-09-20_gpu_trace_walkthrough.md
    2026-09-20_num_envs_sweep_and_launch_bound.md     the 8192 knee; CUDA-graph flags still to try

docs/notebook/                       Giuseppe's handwritten notes, transcribed (1-3.xml: the four situations)
```
