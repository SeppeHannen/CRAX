# Automated curriculum learning on CRAX

Giuseppe Hannen's graduation project. Start here. This file is the goal, the
plan and the state of the work; the reasons live in the documents it points to.

## The goal

A thesis chapter that says: *here are four safety situations an automated
curriculum can face; here is how each method behaves in each.* The structure
is TeachMyAgent's — fixed scenarios that each isolate one property, methods
compared across them — with safety properties as the scenarios
(notebook 2026-10-05, `docs/notebook/1-3.xml`).

| # | situation | example on `safe_goal_point` | the test |
|---|---|---|---|
| 1 | reward progress conflicts with safety | hazards between start and goal: the shortcut is unsafe | does the curriculum drift toward high-shortcut contexts (more reward) or away from them (more safety)? |
| 2 | few contexts permit safe success | most of Ω is infeasible within the budget | does it find the feasible region and concentrate there? |
| 3 | few contexts are hard but feasible | most of Ω is easy; deployment w is the hard tail | does it find the tail and train on it enough? |
| 4 | safety depends on the context | fast is safe in some contexts, slow in others (physics, dynamics) | does the agent stay good at both ends of Ω at once? |

Every test is a question about where the realised curriculum q̂ put its mass,
read against performance on w — both are logged every round already.
Situations 2 and 3 need nothing new. Situation 1 needs a layout dimension
(hazard density between start and goal). Situation 4 needs a physics dimension
*and* a policy that can tell contexts apart (plan item 5).

Two statements that frame the chapter, from the same notes:

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
  never violate. "How informative is sub-budget cost" is a candidate fifth axis.

## The plan

In order. Each item says what it produces and where it goes.

0. **Ground the situations in the literature** → `literature/`. A one-afternoon
   documented review (protocol, screening table, extraction form, verified
   references) of *which properties of a constrained environment family make
   training hard, as reported*. Candidate properties P1–P6 go in; evidence by
   type (reported / designed / reviewed / argued) comes out. Two independent
   deep-research runs (Claude, ChatGPT) on `literature/deep_research_prompt.md`;
   references checked with `literature/verify_references.py`. Decided
   2026-10-06 after discussing TeachMyAgent: its challenges were problem-side
   properties, operationalised one per unit test; the method-failure analysis
   came afterwards to explain results. We follow the same order.
   **Done 2026-10-06** → `literature/synthesis.md`. 56 papers screened, 54 ids
   clean. Verdicts: P1 universal by benchmark design (the Ω-distribution of the
   conflict is ours); P4 ≡ *shift-violation*, the best-evidenced family
   property (6 REPORTED); P5 (λ lag) documented offline→online, never at a
   curriculum switch — promote to a situation; P2/P3 unevidenced as stated —
   reframe as the two reported failure modes on hard contexts,
   *over-conservatism* vs *violation-dominated exploration* (CRAX's own
   curriculum helped only where the failure was conservatism); P6 no support.
   Both runs: the budget d is a context dimension (noted, not adopted).
   By-product A: Koprulu, Simão, Jansen, Topcu, *Safety-Prioritizing Curricula
   for Constrained RL* (ICLR 2025) is the prior art — **confirm with the
   supervisor that this is "Safe Curriculum Generation"**.
1. **Write the situations down** → `design/safety_challenges.md`. Per
   situation: the property and its evidence (from 0), the Ω dimensions on
   `safe_goal_point`, how the distribution over Ω is skewed, the test
   question, the number that answers it, the *predicted* ordering of method
   classes and why, what is missing. Plus the constraint statement above. No
   code.
2. ~~**Baseline on `safe_goal_point`**~~ **Done 2026-10-06** —
   `experiments/2026-10-06_goal_point_staged_vs_uniform.md`, W&B group
   `goal_point_staged_vs_uniform`. Staged does **not** collapse (the velocity
   collapse was the blind student) but is never safe: λ decays to 0 on level 1
   and cannot climb fast enough after the switches. Only `level:3` meets the
   budget; `uniform` has the reward. See Results.
2b. **Spread the slots over the episode** — **done 2026-10-06**
   (`ContextualAutoResetWrapper.reset_with_parameters`: a uniform random
   initial step offset per slot; verified on goal-point at 8192 slots, offsets
   uniform over 0–999, so ~655 episodes complete every round instead of 8192
   every 12.5). Why: on goal-point every episode runs to 1000 steps, so all
   slots reset together and stayed in lockstep for the whole baseline run.
   What spreading changes: each round's PPO batch is a sample of the episode
   rather than one 80-step slice; a new φ acts from the next round (q̂ ramps
   towards q) instead of taking effect all at once up to 12.5 rounds later; a
   teacher gets completed episodes every round rather than in bursts. What it
   does *not* change: feedback latency — an outcome is known one episode after
   its context was drawn, either way. Decision taken with Giuseppe after the
   fact; a lockstep-vs-spread comparison on the three arms is the check that
   outcomes do not depend on it. Context path only; the stock `AutoResetWrapper`
   keeps lockstep (Tristan, item 14).
   *Still open:* `training/logger.py` `MetricsLogger` buffers `training/*` and
   `episodic/*` and flushes their mean when a step counter (rebuilt from two
   32-bit halves inside a callback) has advanced by `training_metrics_steps`.
   That design serves the stock 1 M-step cadence; we set the cadence to one
   round, so the counter always fires and the buffer averages one value — a
   second door to W&B that does what the round hook already does. Replace by
   one `progress_fn` call per round from the trainer; `MetricsLogger` stays
   only if the stock (non-context) path still needs it — ask Tristan.
3. **The λ fix is PPO-PID, not PPO-Saute.** Repeat 2 with `ppo_pid`, same
   group layout. Two experiments now say the Lagrange multiplier is calibrated
   to the stage that just ended — over-wound on velocity (froze the policy),
   at zero on goal (never enforced). PID's proportional term reacts to the
   *current* violation; hypothesis: within a few rounds of each switch the
   staged arm's cost returns to budget. Saute changes the *constraint*
   (almost-sure per episode; remaining budget in the state) rather than the
   multiplier; for an expected-cost constraint a state-based stochastic policy
   is already optimal (Altman 1999). Footnote on constraint semantics only.
4. **First curriculum method**, chosen so that "where did q go" is legible.
   Baselines: uniform, staged, and the supervisor's constrained CURROT ("Safe
   Curriculum Generation") — the constrained objective is not our novelty.
5. **Let the learner see its context.** Needed for situation 4 and for any
   physics dimension. Two options, both change the benchmark's observation
   space (agree with Tristan first): *told* — ω in the observation, a
   contextual CMDP, the oracle; *infer* — last step's cost and reward in the
   observation, with or without memory. Measure the gap to the oracle before
   building memory. Read first: CARL (how it exposes ω, what its results say
   for both modes); contextual MDPs (Hallak et al. 2015; Modi et al. 2018);
   belief-state / Bayes-adaptive MDPs (Duff 2002; Ghavamzadeh et al. 2015),
   RL² (Duan et al. 2016), VariBAD (Zintgraf et al. 2020); what PLR, ACCEL,
   SPaCE and CURROT do about the observation. With ω hidden the problem is a
   POMDP over (s, ω); a recurrent policy approximates the belief; for the
   velocity suite the sufficient statistic is just the bracket [max v with
   cost, min v without]. Decision → `design/context_observation.md`.
6. **The four situations**, each with its distribution over Ω and the methods
   from 4.

### Deferred, with the reason

- **Ω as the union of level boxes.** Motivated by button's infeasible corner
  and a cleaner "uniform over Ω". Goal-point's corner turned out infeasible
  too, and is handled by the cap on the total count (Decisions) — one
  constraint on the box rather than a union of boxes. Button's corner would be
  handled the same way (a cap, or a cap depending on `placement_extent`). The
  situations define their own distributions, so this is off the path.
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
- **Skill probes over training** (Giuseppe, 2026-10-06). The inverse of the
  TMA design: train every method on the *full* Ω, and evaluate at ~10
  checkpoints on several *probe subspaces* $\Omega_s \subset \Omega$, each
  chosen to force one behaviour (e.g. dense hazards between start and goal →
  threading; large goal far away → navigation; many collidable blocks → contact
  avoidance). Output: per-method skill-acquisition curves (reward, cost, and
  whether the constraint holds *on that subspace*), overlaid with the mass q̂
  placed on $\Omega_s$ per round. The lag between "teacher samples $\Omega_s$"
  and "agent becomes good on $\Omega_s$" says whether the curriculum caused the
  skill, followed it, or never trained it. Answers *how* a method's capability
  came about; the situations answer *whether* it has it. Needs no code:
  `evaluation_wrap_env_fns` already takes any number of evaluation
  distributions and `--num_evals` sets the checkpoints; probes are box
  distributions over Ω. Caveat: a subspace is a region of the world, not a
  skill — each probe needs a sentence saying which behaviour it forces and why
  the alternative does not pass. Can ride the plan-item-2 baseline for free.

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

**Goal point, staged vs uniform vs level 3, 100 M steps, one seed**
(`experiments/2026-10-06_goal_point_staged_vs_uniform.md`). On level 3 after
100 M: staged 14.5 reward / 135 cost, uniform 16.9 / 138, level:3 5.1 / 22
(budget 25). Staged does not collapse at either switch — reward holds — but
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

## Vocabulary

| term | meaning |
|---|---|
| context ω ∈ Ω | the per-episode parameters of the *world* a suite instantiates — physics (friction, mass, gravity), layout (hazard count, positions). Not the cost threshold: a run keeps one threshold. A value on the `num_envs` axis, never a shape. |
| distribution (not "teacher") | decides which ω each parallel environment gets when its episode starts. `uniform`, `level:3`, `staged:1,2,3`. |
| r | uniform over Ω; the domain-randomisation baseline. |
| w | the deployment distribution; what the agent is evaluated on (level 3 by default). |
| round | one PPO training step; with a context distribution, one compiled call. |
| q vs q̂ | *intended* curriculum (what the distribution samples at round k) vs *realised* (what the gradient came from, per transition). In the logs: `sampled` vs `experienced`. They differ when episode length depends on ω and because episodes straddle rounds. Always shown together. |
| situation | one of the four safety scenarios in *The goal*: a distribution over Ω chosen to isolate one property. |

## Running

```bash
# manual curriculum
.venv/bin/python -m training.train_env --env_name safe_goal_point --alg ppo_lag \
  --context_distribution staged:1,2,3 --deployment_distribution level:3 \
  --num_envs 8192 --num_timesteps 50_000_000 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --measure_performance --skip_rollout --skip_video --store_model false \
  --wandb_group <experiment_group>

# uniform: same, with --context_distribution uniform; fixed level: level:3
```

Runs in one `--wandb_group` are one experiment and must share every flag but
the distribution; the first run creates the group's W&B view.

Every context run is evaluated on the deployment distribution
(`evaluation/deployment/*`, level 3 by default) and on uniform (`evaluation/uniform/*`),
and logs the sampled and experienced context distribution every round
(`training_curriculum/*`, as W&B histograms). What is logged and what each key
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
    synthesis.md                     merged verdict per property, NEW properties, by-product A, run disagreements — the evidence column for design/safety_challenges.md
    runs/{claude,chatgpt}/           each run's screening.csv, evidence.csv, by_product_A.csv, synthesis.md, free_text.md
    *-deep-research.md               the raw outputs as pasted
    split_run.py                     raw output → runs/<model>/ files
    verify_references.py             resolves every paper_id (Semantic Scholar / arXiv / Crossref) and compares titles; fabricated ids and real-but-wrong ids both fail
  design/
    safety_challenges.md             (plan item 1) the four situations as distributions over Ω on safe_goal_point
    context_observation.md           (plan item 5) told vs infer: decision and reasons
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
