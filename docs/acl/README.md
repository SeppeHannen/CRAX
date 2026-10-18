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

The thesis studies the link between automated curriculum learning and
generalisation; the scope was set with the supervisors on 2026-10-14 and the
design is `research_design_brief.md`. The supervisor wants first results the
week of 2026-10-21. The tasks follow in order; each heading states what the
task delivers. Every experiment is a `--wandb_group` of arms on
`safe_goal_point` at 500 M steps, on the sensor observation
(`decisions/2026-10-18_vision_set_aside.md`).

**Order as of 2026-10-18.** Task 0 first: everything built since 2026-10-14
lives on one unreviewed branch and `main` is five weeks behind it; nothing new
is decided or built until that is merged. Then task 1b — the cost tail —
before tasks 2 and 3: a constraint on the mean cost is set by the rare
catastrophic episode, so until that episode is understood, every arm's cost
and the caution it buys are statements about the tail, not about the
curriculum. Task 1 is done.

### 0. The work since 2026-10-14 is reviewed and on `main`, so the next step starts from a stable position

**Where things stand.** Branch `contextual-review` (`38b8951`, pushed) holds
every branch's content: `contextual-learner`, `episode-review`, `snellius`,
`experiment/goal_point_learners_500M`, and `replay-checkpoint` (by content;
that branch's first commit rendered MP4s and was reworked). It is 34 commits
and 96 files over `origin/main`, which last merged `episode-review`
(PR #1). One upstream file is touched (`crax/envs/wrappers/pixel_observation_gpu.py`,
11 lines). CPU tests: 12 files, all passing as of 2026-10-17 (`92 passed`); the
entry point has a subprocess test since 2026-10-17. The local checkout `CRAX`
is on `contextual-review`; the four worktrees are removed.

**What is left, in order.**

1. **Giuseppe reviews, by the data flow** (`AGENTS.md`, *Pull-request
   write-ups*). The branch is one PR against `main` but several logical
   changes; the review is organised as walks, one per change, each from its
   CLI flag to its last effect, written when the review of that change starts:
   (i) the contextual learner — `--constraint`, `--policy_context`, the
   `{sensors, context}` observation, λ(ω) in two forms, the Ω-bounds scaling
   (`design/contextual_learner.md` is the design; the walk is still to write);
   (ii) the experiment launcher — `training/experiments`, a definition file,
   `local` and `snellius`, run-directory checkpoints, `run_files.hand_to_run`;
   (iii) the whole-learner checkpoint and the replay — `save_learner_state`,
   `training.replay`, the records with the policy's frames, the review page's
   policy camera; (iv) the vision composition — `training/pixels.py`,
   `contexts/networks.py`'s encoder, `--gpu_memory_fraction`, the per-minibatch
   critic pass and augmentation. Feedback goes into the branch as commits.
2. **The full CPU suite passes at the merge commit**: `JAX_PLATFORMS=cpu
   MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`
   (about 15 minutes; the review's frame test needs the GL context).
3. **Merge to `main`** as one merge commit, push, and delete the five merged
   branches on the remote (`contextual-learner`, `episode-review`, `snellius`,
   `experiment/goal_point_learners_500M`, `replay-checkpoint`) and locally.
   `contextual-review` is then `main`.
4. **The Tristan conversation**: the list below has grown by six items this
   week (19–24); the three that change his files or the benchmark's meaning
   (pixel wrapper dict, augmentation per minibatch, `--lagrangian_coef_rate`
   per round) are raised before anything of task 1b is built, since 1b's
   candidates all change the benchmark too.
5. **Loose ends with a home**: the over-budget fraction logged by the
   evaluator (task 1b, *to build*); the additive constraint's per-round step
   governed by its fit's constants, not the rate flag (`experiments/2026-10-18`,
   plan item 2 — decide, then either fix or document); the Markdown preview
   broken in Giuseppe's VS Code (two preview extensions active; not the
   repository's); the review server and the rendered previews on ports 8790
   and 8791 are stopped when no longer needed.

### 1. The CMDP becomes contextual: the policy sees its context and the constraint holds per context, so a curriculum can no longer move it

PPO-Lagrange's one multiplier λ is updated each round from the cost of that
round's training episodes, so it settles where the policy is on budget for
whatever the curriculum is sampling now. Every run so far shows the result:
cost at the budget in training, three to four times the budget on level 3 for
the same policy (`research_design_brief.md` §6).

**The policy is told its context.** Today it is blind: it sees lidar, compasses
and its own sensors, and nothing of ω. In a contextual CMDP the context is
part of the state, so the policy and both critics receive ω as input; for
goal-point that is the five-number context vector (four hazard counts, goal
radius) appended to the 62-dimensional observation. This is the change
Giuseppe asked for on 2026-10-14: if the constraint is per context, the
policy is contextualised, not blind. Which factors a policy may be told is
decided per factor by what a deployed robot would know (brief §4); layout
factors qualify, dynamics factors later will not. The observation space
changes, so this goes to Tristan.

**The learner** keeps one multiplier per context, λ(ω), each updated from the
cost observed in its own context. The told cost critic estimates cost per
context; the told policy can hold a different reward–cost trade-off per
context. A blind policy under λ(ω) receives conflicting gradients on contexts
it cannot tell apart and collapses to one trade-off; Tristan predicts this,
and it is run once as a control.

**The switches** are two CLI choices, so that every later arm can run under
any combination: `--constraint scalar|network|additive` and
`--policy_context blind|told`. The published methods all use a scalar
constraint, which stays as their baseline. An *arm* is now the training
distribution under a learner (`uniform scalar blind`, `staged:1,2,3
additive told`); the dashboard groups lines by it.

**Result (2026-10-15, 500 M, group `goal_point_learners_500M`, one seed; 15
arms on Snellius): the per-context constraint puts every uniform and PLR arm
on budget on level 3 (cost 16–25) and no scalar arm is (33–78); the told
policy keeps 19–24.5 goals on Ω while doing so; level-3 return is the price,
4.5–13.8 across arms, and `additive` ends above `network` on every
unconfounded distribution. The told arms meet the budget by standing still first (51–128 M)
and learn to move within it afterwards — the 100 M smoke's motionless end state
was a waypoint.** Write-up: `experiments/2026-10-15_goal_point_learners_500M.md`.
The staged-told rows are confounded by the context normaliser (Known defects)
and are rerun after the fix. **Learner settled: `additive told` is the default
for every later arm; `scalar blind` stays as the published baseline row.**

**Follow-up (2026-10-17, same arms from pixels + sensors, group
`goal_point_learners_vision_500M`): the budget still holds under λ(ω) (7 of 10
told arms within 25, 2 of 5 scalar), the clean staged-told arms are the best
policies of either run (19–20 goals on level 3 at cost 20–28, 23–24 on Ω), the
uniform and PLR told arms lost their return (none above 8 goals under budget),
and every arm swings between evaluations: λ oscillates with a 16–43 M-step
period. The run changed the observation and, through 1024 envs, the multiplier
updates per step (8×); it cannot say which did what.** Write-up:
`experiments/2026-10-17_goal_point_learners_vision_500M.md`, with an
interactive return–cost trajectory page.

**Rate-matched rerun (2026-10-18, six arms, 750 M, rate ⅛ per round): the
swings stayed, so they are the pixel policy's, not the multiplier's; two runs
of the same arms disagree at 500 M by as much as the distributions do (no
ranking from one seed); PLR additive told keeps level 3 at the budget with a
moving policy (9.4 goals / cost 18, 8 of 10).** Write-up:
`experiments/2026-10-18_goal_point_learners_vision_750M.md`.

**Vision set aside (2026-10-18).** The sensors control at the pixel runs'
small round (group `goal_point_learners_sensors_small_round`) swings *less*
than at the large round (0.4–0.6 goals on the non-staged arms): the round is
exonerated, the camera is the cause. Vision did not remove the failure it was
adopted for (the agent pressing into a gap closed by two solid hazards, with
both in frame) and adds a variance one seed cannot carry. **Experiments run on
the sensor observation; vision is future work; the tail failure is the next
problem, on sensors.** The chronology and the reasoning:
`decisions/2026-10-18_vision_set_aside.md`.

**Status (2026-10-15): built, CPU-tested, on branch `contextual-learner`;
the 500 M experiment ran from `experiment/goal_point_learners_500M`.** Design and what is logged:
`design/contextual_learner.md`. λ(ω) is a function of the context fitted each
round to the targets PPO-Lagrange's integrator names for every transition, in
one of two classes: `network`, a small MLP, and `additive`, f(Σ h_i(ω_i)) with
monotone per-factor difficulties and a monotone price — Giuseppe's "classical
statistical" form, which makes an unseen context's multiplier a bound from its
neighbours rather than a seed's output (a table over the integer points was
built first and replaced the same day: it left each new stage at λ = 0). The
`network` form needs `--initial_lambda_lagr > 0`. The observation of every
context run is `{sensors, context}` and the critics read both. **Running
(2026-10-15, group `goal_point_lambda_forms_200M`): staged and uniform under
each form, 200 M, `told`.** **Smoke (2026-10-14, group
`smoke_contextual_learner_v2`): three arms at 100 M, `staged:1,2,3` /
`uniform` / `plr`, all `network told` (then named `per_context`), rate 1,
initial λ 0.1; 0 recompiles, 85 k SPS (no throughput cost).** λ_ψ spreads
across Ω as designed: under uniform λ runs 0 at the sparse end to 8–9 at the
dense end (fit error 0.16 against a target spread of 1.75), and the told
policy holds different behaviour per context — return on Ω stays 19 while the
multiplier on level 3 drives level-3 cost 166 → 2 **by driving return 17 →
0.5: the budget on level 3 is met by standing still** (PLR the same: 4 / 3.2).
Staged under λ_ψ ends at level-3 cost 64 / return 15.9 with λ(L3) 3.9 and
still rising, so it is on the same road. Two readings for the 500 M design:
the per-context constraint on level 3 is enforced, which is the claim, and at
this budget the only policy that meets it at 100 M does nothing, which is the
"mostly infeasible" row of the challenges table; the integrator at rate 1
unwinds at 0.023/round, so λ(L3) ≈ 8 takes ~350 rounds to release. At a stage
switch the function's extrapolation to the unseen level is uninformed (one
training context carries no direction): λ(L2) inherited 0.27 of 0.51, λ(L3)
0.0 of 1.37.

**The arms** and what each must show:

| arm | learner | policy | must show | 2026-10-15 |
|---|---|---|---|---|
| `uniform` | scalar | blind | the baseline, rerun at the new λ rate | level 3: 9.9 / 58 |
| `staged:1,2,3` | scalar | blind | cost on level 3 and on uniform leaves the budget | 8.0 / 78, λ 29 and rising |
| `uniform` | per context | told ω | matches or beats its scalar twin | additive 9.2 / 20, network 5.7 / 16: on budget, less return |
| `staged:1,2,3` | per context | told ω | cost on level 3 and on uniform stays near the budget, return holds | confounded (normaliser), rerun |
| `staged:1,2,3` | per context | blind | the multipliers spread across contexts while the behaviour does not | not run yet |

**The multiplier's learning rate** drops from 10 to 1 in every new arm
(`--lagrangian_coef_rate`; Tristan, 2026-10-14: at 10 it overshot 100× after
a stage switch). Old arms are not comparable to new ones, which is why the two
baselines are rerun rather than reused.

### 1b. The cost tail: one failure sets the multiplier, and the policy is made to see it, remember it, or be charged for it differently

**What is known** (`decisions/2026-10-18_vision_set_aside.md` §1–2, 4). On
level 3 the typical episode of the best policy costs 13–15 against a budget of
25; the mean of 17.6 is carried by 0.5 % of episodes over 100 (worst 247; the
scalar baseline 6.8 % over 100, worst 1374). Those episodes are one failure:
the agent drives into a gap closed by two solid hazards that stand between it
and the goal and presses the same path for the rest of the episode, 3 cost per
step. The constraint on the mean is set by that tail, and the policy pays for
it with caution in every other episode. The hazard lidar cannot carry the fact
that decides it (centre distance per sector, no kind, no extent); a camera that
can did not change it — the failure needs the policy to know it has tried, or
the cost to say that it is pressing, not only an observation that shows a wall.

**The question.** Which change to the learner or the task removes the tail
without changing what the constraint means — and does the budget then hold
with a policy that moves, at a return that the tail was costing?

**Candidates, in order; each one arm set against the current learner, same
six arms as the controls** (`uniform`, `plr`, `staged:1,2,3` × `scalar blind`,
`additive told`), judged on three numbers from 384 replayed level-3 episodes:
the fraction over 4× the budget, the median cost, and the return under budget.

| | change | what it gives the policy | changes the benchmark? |
|---|---|---|---|
| a | hazard kind and extent in the observation: the lidar as two arrays (passable, solid) and each compass with its hazard's size | the fact that decides the episode, in the sensor the policy already reads | observation space; Tristan |
| b | memory: a recurrent policy (or the last k actions and costs appended to the observation) | that it has pressed this wall for 200 steps and should turn | policy class; Tristan's trainer |
| c | a cost that rises with sustained contact (the per-step cost multiplied by the length of the current contact run) | pressing distinguished from brushing in the signal the constraint reads | the cost function; changes the CMDP and is flagged as such |

(a) is first: it is the cheapest, it is the direct fix of §2's finding, and if
it fails the failure is not perceptual. **Not** a cost cap and **not** a
stuckness terminator: both end the episodes the measurement is about and
change what the budget means. Every candidate's observation or cost change is
raised with Tristan before it runs; the sensor-only variant of the vision
composition (`{sensors, context}`) is the base.

**To build:** the over-budget fraction and the cost quantiles logged by the
evaluator every evaluation (plan item 6 of 2026-10-15, still owed), so the
tail is a dashboard line and not a replay afterwards; then candidate (a).

### 2. Each arm is scored on a held-out region that tests one type of generalisation

Every run is scored on level 3 and on uniform over Ω, both inside the
training support, so neither measures generalisation to unseen contexts. The
one out-of-distribution region in `registry.py`, Ω₂, was a placeholder and is
retired: it shifted two factors at once, one of them (goal radius) invisible
to the policy and without effect on the optimal action, the other (total
≥ 21) in the range where placement overcrowds silently.

**The survey's rules** (Kirk et al. §4.2–4.3) fix the design. A test set is
of one type: *interpolation*, a region inside the training set's hull that was
removed from training; *combinatorial interpolation*, a combination of seen
values never seen together; *extrapolation*, outside the hull. They are
ordered by difficulty in that order, and extrapolation is not expected to work
without an inductive bias, which a curriculum does not add. An interpolation
set on an ordinal factor is made by training on two disjoint ranges and
testing between them. The factor must be one the policy perceives, or the
test measures difficulty rather than generalisation.

**The axis** is total hazard count: ordinal, visible through lidar density,
and the quantity by which the benchmark's own levels differ. Measured
2026-10-14: Ω has 3,466 admissible count tuples; the levels sit at totals 12,
16 and 20; every total from 12 to 18 has 260–295 tuples.

| set | type | definition | read against |
|---|---|---|---|
| $T_\text{hole}$ | interpolation | total ∈ {14, 15}: 586 tuples between level 1 and level 2; training is Ω without them, two disjoint ranges either side | a uniform arm over all of Ω |
| $T_\text{comb}$ | combinatorial | waits for the corridor factor (*After these*); not faked on the kind counts, which are invisible | a uniform arm over the hull |
| $T_\text{extra}$ | extrapolation | total ∈ {21, 22}, robustness panel only, after an overlap-versus-count measurement shows the layouts are valid | none claimed |

**To build:** a training space, Ω minus the held-out totals, that uniform's
tuple table, PLR's buffer and staged's level points all read; `ContextSpace`
able to exclude a band of totals, by the mechanism its cap and floor already
use; $T_\text{hole}$ as an `EvaluationRegion` whose relation to the training
space is checked; the placeholder values in `registry.py` replaced; and
`design/goal_point_out_of_distribution.md` rewritten under a name that says
what it now is. All CPU-testable; no compiled program changes.

### 3. PLR's twins separate what it covers from how it moves

Of what a curriculum covers and how it moves, which one changes
generalisation? That is the thesis question (`research_design_brief.md` §1),
and the twins of brief §3 are the instrument.

**The twins** are built from a finished PLR run. Its realised distribution in
every round, $\hat q_1 \ldots \hat q_K$, is recorded; one twin samples from
the round-average $\bar q$ in every round (same coverage, no motion), the
other replays the recorded distributions in shuffled order (same motion, no
direction). PLR minus the first twin is the effect of non-stationarity; PLR
minus the second is the effect of its order.

**The learner** is task 1's. Under a scalar λ an arm and its twin have
different λ trajectories, because their rounds sample different contexts, and
the twins would measure that instead of the contexts. Every arm is scored on
task 2's sets.

**To build:** the realised per-round context sequence stored at full
resolution by the round hook (today only 12-bin histograms are logged), and a
`replay` distribution that reads such a sequence, time-averaged or shuffled.

### After these

The corridor factor and with it $T_\text{comb}$; friction and actuator gain,
with a blind or recurrent policy; CURROT and SCG as baselines
(`research_design_brief.md` §4, §5).

### Retired

The Ω₂ placeholder (task 2 replaces it). PPO-PID as a prerequisite: the rate
change addresses the wind-up it was for; PID stays an available learner.

### Built and committed (0ce7d35, bf65448, 2026-10-11/12)

PPO-PID takes the curriculum hooks; PLR scored by the reward critic, the cost
critic or the Lagrangian objective under the learning-signals contract
(`design/learning_signals.md`); `EvaluationRegion`s and
`setup.evaluations_for` as the one place that says which targets a run gets.

### Done

- **PLR scored by the cost critic and by the Lagrangian objective**
  (2026-10-12; code 0ce7d35): `plr:cost`, `plr:lagrangian`; the learning-signals
  contract (`design/learning_signals.md`); evaluation regions. Two new arms
  against the three 2026-10-08 arms at 500 M on goal-point →
  `experiments/2026-10-11_goal_point_plr_scores_500M.md` (group
  `goal_point_plr_scores_500M`, which now holds all five runs; the v2 group is
  empty); result in *Results so far*.
- **Prioritized Level Replay** (Jiang et al. 2021) as the first curriculum
  method (2026-10-09, 242f0a6): `design/prioritized_level_replay.md`;
  `--context_distribution plr`; the trainer ships the reward advantage to the
  round hook (`LearningSignals`); `training_curriculum/{intended,value_loss}/<d>`
  heatmaps and `distribution/*` scalars. Uniform vs staged vs PLR at 500 M on
  goal-point → `experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md`
  (runs moved to group `goal_point_plr_scores_500M`); result in *Results so far*.
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

**Goal point, six arms from pixels with the dual ascent matched per step to the
sensors run, 750 M, one seed** (`experiments/2026-10-18_goal_point_learners_vision_750M.md`).
**The evaluation-to-evaluation swings of 2026-10-17 are not the multiplier's:**
at rate ⅛ per round the scalar arms' λ moves per 655 k steps exactly as with
sensors, and level-3 return still jumps 1.8–4.1 goals between evaluations
(sensors 0.8–1.4; sampling error 0.2–0.4); per-round training cost varies
0.41–1.00 of its mean against 0.09–0.32. **Two runs of the same six arms
differing only in the rate disagree at the 500 M mark by 5–10 goals**, as much
as the distributions differ: no distribution ranking under vision from one
seed. **The per-context constraint holds with a moving policy**: PLR additive
told 9.4 goals / cost 18 on level 3, under budget 8 of the last 10, 17.5 on Ω;
scalar uniform and PLR at 18–20 goals pay cost 57–68. The additive λ(ω)'s
movement per round is set by its fit's Adam constants, not by
`--lagrangian_coef_rate` (its step noise fell 1.2× where the scalar's fell
10×). Candidates for the swings: the shared encoder (DrQ augmentation) or the
1024-environment batch; the sensors-only control at 1024 environments
separates them.

**Goal point, the same fifteen arms from pixels + sensors, 500 M steps, one
seed** (`experiments/2026-10-17_goal_point_learners_vision_500M.md`, Snellius,
5 h 40 per arm). **The per-context constraint still holds the level-3 budget**
(7 of 10 told arms within 25; 2 of 5 scalar, one standing still) **and the
staged curriculum under λ(ω), readable for the first time after the normaliser
fix, gives the best policy of either run** (staged additive told 19.6 goals /
cost 20 on level 3, 23.6 / 5 on Ω; staged network told 18.9 / 28). **The
uniform and PLR told arms lost the return they had** (PLR reward additive 13.8 →
5.7, network 10.8 → 2.4; none of the eight ends above 8 goals under budget; Ω
return 21–24.5 → 10–16). **Every arm swings between evaluations two to five
times more than before** (|Δ return| 1.3–5 goals against a 128-episode sampling
error of 0.1–0.6): λ oscillates with a 16–43 M-step period (sd/mean 0.16–0.60
vs 0.02–0.22) and λ at the hardest context runs to 15–58. The multiplier steps
once per round at the same rate, and this run has 8× the rounds per step (1024
envs); the observation and the dual-ascent clock changed together, so the pixels
are not yet read. Interactive trajectory page:
`experiments/figures/2026-10-17_goal_point_learners_vision_500M/trajectories.html`.

**Goal point, five distributions × three learners, 500 M steps, one seed**
(`experiments/2026-10-15_goal_point_learners_500M.md`, Snellius). **Under the
per-context constraint with a told policy every uniform and PLR arm ends at the
budget on level 3** (cost 16–25, under budget in 6–10 of the last ten
evaluations) **and under the scalar constraint none does** (33–78, 0–3 of ten).
The told policy holds 19–24.5 goals on Ω at the same time; level-3 return is
the price and ranges 4.5–13.8, with PLR reward additive the best policy of the
fifteen on both targets (13.8 / 24.5 on level 3, 24.5 / 20 on Ω). Every told arm
first meets the level-3 budget by standing still (51–128 M, return 0.2–2.7) and
then learns to move within it; λ at the hardest context peaks at 6–10 early and
releases over 300–400 rounds. Scalar PLR reward collapsed to standing still
(λ 8.5, mass drifting to sparse layouts); scalar staged is on the rate-10 road
at a slower clock (λ 29, return falling through the last four evaluations). The
scalar baselines at rate 1 do not reproduce the rate-10 ordering of 2026-10-11
with one seed; the told arms' level-3 cost is consistent across eight arms.

**Goal point, the three PLR scores, 500 M steps, one seed**
(`experiments/2026-10-11_goal_point_plr_scores_500M.md`). **PLR Lagrangian
ends where PLR reward ends** (level 3: 15.2 / 68 vs 15.3 / 73; Ω: 21.1 / 19 vs
21.3 / 17) and had the lowest λ peak of any arm (5.8). **PLR cost was the best
policy of the five at 300 M** (level 3: 16.5 return / 46 cost) **and then
collapsed** — the staged failure with no stage switch and no curriculum
motion: its distribution (16.2 hazards) was fixed from 50 M, but on it the
policy's cost sat 5 % *over* budget where the other PLR arms sat 30 % under;
the leak-free integrator wound λ 0.35 → 2.3 over 160 rounds on that excess, the
update tipped (cost 27 → 47 in 15 rounds, critic error up uniformly across Ω),
λ 2 → 16, the policy gave up its return (end: 4.4 / 30; recovering). **The Lagrangian score is λ in
disguise**: its spread is 10× while λ > 1 and the reward score's 1.6× while λ =
0, so `plr:lagrangian` is a λ-gated switch between the other two scores — which
is why it did not collapse. The cost score has 9× to rank, the reward score
1.5×, throughout. Consequence: PPO-PID first, with PLR cost the arm to rerun; β
< 1 on the cost score the first hyperparameter worth a run.

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

- **Vision is set aside; experiments run on the sensor observation**
  (2026-10-18). Adopted on 2026-10-16 because the lidar cannot tell a passable
  gap from a closed one; the pixel policy kept the same failure with the wall
  in frame, and its evaluations swing 2–5 goals between evaluations where the
  sensor policy's move about one — the round size and the multiplier's rate
  were each ruled out by a control. Future work: the same experiments from
  pixels once that variance is understood. `decisions/2026-10-18_vision_set_aside.md`.
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
  trainer additions; `ppo_lag` forwards them, and `ppo_pid` in the uncommitted
  change (plan item 1). Any other PPO-family trainer needs the same two lines.

## Known defects

- **A told policy under `staged` was blind to any context but the current
  stage's** (found 2026-10-15, fixed 2026-10-16 on `experiment/goal_point_learners_500M`).
  The observation normaliser keeps a running standard deviation per input and
  clips it at 10⁻⁶ (`training/acme/running_statistics.py`, `std_min_value`);
  one context per stage gave the context inputs zero variance, so every other
  context — the level-3 and uniform evaluations, and the next stage's context at
  a switch — normalised to ~10⁶ and saturated the network (reproduced: after 50
  rounds on level 1, level 3 normalised to −6·10⁶, 4·10⁶, 6·10⁶, 4·10⁶). The
  staged-told arms of `goal_point_learners_500M` have evaluation return exactly
  0.0 on both targets for 350 M steps while training return is 23–29, and their
  training return drops at each switch where the blind arm's does not; uniform
  and PLR arms draw a new context every episode and were unaffected. Fix: the
  context part of the observation is scaled by Ω's bounds (`ContextSpace.scale`,
  the function the λ network already used), never by running statistics;
  `design/contextual_learner.md` §2. The two staged-told arms are to be rerun.
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
17. `post_step_fn` (the hook every constrained agent updates its multiplier in)
    now receives the round's `Transition` as a third argument
    (`ppo/train.py`), so PPO-Lagrange reads the batch's cost from the recorded
    transitions rather than from the loss's minibatch metrics; the other four
    agents accept and ignore it. Same numbers; one source.
18. A context run's observation is `{"sensors": o, "context": ω}`, produced by
    a wrapper outside the auto-reset (`training/contexts/observation.py`), and
    its critics always read ω. The environments are untouched; a stock run
    still observes the flat vector. The post-training rollout and video
    (stock environment) cannot feed such a policy, so a context run requires
    `--skip_rollout --skip_video`.
19. `GpuPixelObservationWrapper` in `pixels+state` mode over a dict
    observation collapsed the inner parts under `'state'`; it now keeps them
    and adds `pixels/<camera>` next to them (2026-10-16). A stock run's flat
    observation still becomes `{'state', 'pixels/…}`; only the dict case
    changed. `crax/envs/wrappers/pixel_observation_gpu.py`.
20. The DrQ random translation is applied per minibatch (`minibatch_step`)
    instead of to the whole round's data (`sgd_step`): the same distribution
    of shifts, one per trajectory per pass, and the translated copy in memory
    is one minibatch rather than the round — at 655 360 transitions of pixels
    the round's copy alone was ~48 GB (2026-10-18). `training/agents/ppo/train.py`.
21. `--lagrangian_coef_rate` is applied once per round, so the same value
    moves λ eight times as far per environment step at 1024 environments and
    4 minibatches as at 8192 and 32 (`design/training_round.md`). Should the
    flag be defined per environment step? We matched by hand (rate ⅛) for now.
22. `checkpoint.load` restored a GPU-trained checkpoint with its saved
    `cuda:0` sharding on a machine without a GPU (Brax 0.12.3 maps the
    restore args over the step metadata, which never matches the tree, and
    falls back); fixed by mapping over `metadata.item_metadata`
    (2026-10-16). `training/checkpoint.py`.
23. `requirements-pinned.txt` pins jax 0.6.0; the working environment is
    jax 0.10.1 (`scripts/snellius/requirements.lock` is what Snellius runs).
24. The goal-point hazard lidar is one array over all hazards, closeness of
    the nearest centre per sector: a flat disc and a solid cylinder give the
    same reading, and the episodes that cost 10× the budget are the agent
    pressing into a gap closed by two solids it cannot distinguish from
    passable ones (`decisions/2026-10-18_vision_set_aside.md` §2). A camera
    did not change the failure (§4). Did he see evaluation-to-evaluation
    swings of 2–5 goals in his vision training? Task 1b's candidates (kind
    and extent in the lidar; memory; a contact-duration cost) each change the
    benchmark and are his call.

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
  --lagrangian_coef_rate 1 \
  --measure_performance --skip_rollout --skip_video --store_model false \
  --record_episodes 4 \
  --wandb_group <experiment_group>

# uniform: same, with --context_distribution uniform; fixed level: level:3; PLR: plr
# the contextual learner (design/contextual_learner.md): add --constraint additive --policy_context told (or network)
# (defaults scalar, blind = the published method; a context run needs --skip_rollout --skip_video)
```

`--record_episodes 4` keeps four of every evaluation's episodes per target whole,
with the critics' values, λ and the loss's advantages at every step, as records in
the run's W&B files; watch and compare them with

```bash
.venv/bin/python -m training.review --group <wandb_group>   # http://127.0.0.1:8790/
```

(`design/episode_review.md`: why, what a reviewer can do, what a record carries).
Records of a run on this machine are read live from `wandb/run-*/files/episodes/`.

A context run from pixels adds `--vision --vision_obs_mode pixels+state` (the
pixels join `{sensors, context}`; `pixels` alone would drop the sensors) and
`--gpu_memory_fraction 0.75`: under `--vision` JAX is capped at half the GPU so
MJWarp's renderer has room, and the contextual epoch program at 1024
environments peaks at 19.3 GiB, over half of a 40 GiB A100. The flag states the
share in the run's config instead of an environment variable set by hand.

Runs in one `--wandb_group` are one experiment and must share every flag but
the distribution; the first run creates the group's W&B view.

**An experiment is defined once and launched anywhere** (`training/experiments/`;
branch `snellius`). A definition file exports an `Experiment` — the group, the
common flags and the arms (`scripts/experiments/goal_point_learners_500M.py` is
the model) — and two launchers read it: the local chain and the Snellius
submitter, one SLURM job per arm, all at once. Both create the group's view
before any job starts. A job runs a pushed commit, never a working tree; W&B
records it, and checkpoints go into the run's W&B files (`checkpoints/`).

```bash
# local, sequential, detached
setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/<name>.py \
  > runs/logs/<group>_chain.log 2>&1 < /dev/null & disown
# Snellius: one A100 per arm; --dry-run prints the job scripts. Setup once:
#   bash scripts/snellius/install_environment.sh (on the login node), ~/.netrc for W&B
.venv/bin/python -m training.experiments.snellius scripts/experiments/<name>.py --partition gpu_a100 --time 05:00:00
# watch: ssh snellius squeue -u $USER; logs in ~/crax-logs/<group>/<arm>_<job>.out
```

A 500 M goal-point arm takes 1 h 50 on an A100 (93 k SPS) and costs ~240 SBU
(128 SBU per GPU-hour; H100 192). The budget on account `tesr109122` expires
2026-10-31.

Every context run is evaluated on the deployment distribution
(`evaluation/deployment/*`, level 3 by default) and on uniform (`evaluation/uniform/*`),
and logs the intended, sampled and experienced context distribution and the
reward critic's error per context every round (`training_curriculum/*`, as W&B
histograms). What is logged and what each key
means is `training/dashboard/metrics.py`; the W&B view for an experiment group:

```bash
.venv/bin/python -m training.dashboard --group <wandb_group>
```

CPU tests: `JAX_PLATFORMS=cpu MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_contexts.py tests/test_context_training.py tests/test_dashboard.py tests/test_episode_review.py tests/test_suites.py -q -p no:cacheprovider`
(`test_suites.py` runs every check per registered suite; add a suite to the registry and it is covered;
`MUJOCO_GL` is for the review's frame test, skipped without it.)

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
    prioritized_level_replay.md      (done) PLR as a ContextDistribution: what the paper does, what changes on a continuous Ω with frozen φ, diagnostics
    contextual_learner.md            (plan 1, built) λ(ω) per context, the {sensors, context} observation, blind/told policy, the two CLI switches, what is logged
    episode_review.md                (built 2026-10-15) why the evaluation episodes are recorded with the learner's view, what a reviewer can do, what a record carries, what the first viewing should settle
    goal_point_out_of_distribution.md (plan 3, draft) Ω₂ for goal-point: a first guess at what it is and how it is wired; the values need a design pass
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
  decisions/                         one file per decision that closes a line of work: the chronology and the reasoning
    2026-10-18_vision_set_aside.md   why vision was adopted, what it did and did not change, why sensors for the thesis
  experiments/                       one file per experiment: question, commands, results
    2026-09-20_uniform_vs_staged_velocity_ant.md      the 50 M / 500 M result
    2026-09-22_velocity_suites_staged_vs_uniform.md   the other five agents (CPU part; deferred)
    2026-10-06_goal_point_staged_vs_uniform.md        the three arms on goal-point, 100 M (baseline)
  measurements/                      dated measurement logs
    2026-09-19_baseline_goal_point.md
    2026-09-20_gpu_trace_walkthrough.md
    2026-09-20_num_envs_sweep_and_launch_bound.md     the 8192 knee; CUDA-graph flags still to try

docs/notebook/                       Giuseppe's handwritten notes, transcribed (1-3.xml: the four situations)
```
