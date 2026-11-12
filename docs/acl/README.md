# Generalization of constraint satisfaction on CRAX

Giuseppe Hannen's graduation project. Start here. This file says what the
project asks, what is known, what is being done now, and how to run it. The
reasons live in the documents it points to. The argument as a whole, in the
order a paper would give it, is `thesis.md`; its §4 and §5 propose a plan
(2026-11-12) that differs from the one below and is not yet agreed.

## The question

Does a policy that stays within its cost budget on the environments it trained
on also stay within budget on environments it has not seen?

The design of the evaluation, with the reason for every choice, is
`design/generalization_evaluation.md`. Read it before touching an experiment.

Curriculum methods (PLR and others) come later. They will be compared on the
same training set and the same test sets as the baseline below.

## What is known

1. **One multiplier holds the budget only on average.** PPO-Lagrange with a
   single multiplier, trained on environments of different difficulty, spends
   less than the budget on easy ones and more on hard ones. Every standard run
   cost 33 to 78 on level 3 against a budget of 25
   (`experiments/2026-10-15_goal_point_learners_500M.md`).
2. **A multiplier per environment holds it in each one.** With a Lagrange
   function λ(ω) and a policy that sees the context, every run cost 16 to 25
   on level 3 (same experiment). The design is `design/contextual_learner.md`.
3. **The mean cost was carried by rare, very costly episodes**, caused by a
   lidar that could not tell a passable hazard from a solid one. A lidar that
   reads surfaces by kind removed most of them
   (`experiments/2026-10-19_goal_point_cost_tail_500M.md`). Every run now uses
   it.
4. **A camera did not help** and made evaluations swing. Vision is set aside
   (`decisions/2026-10-18_vision_set_aside.md`).

All four are from one seed each. Two runs of the same setup differed as much
as different setups did, so nothing is ranked on one seed.

All four were also measured under the benchmark's stock contact solver, which
kicked a robot away from a hazard it pressed on and so made that cheaper than
it is. The solver was replaced on 2026-11-11 (`known_defects.md`). Costs from
then on are not comparable with these numbers.

Under the new solver, seed 0 of step 1 (six of eight runs,
`experiments/2026-11-12_goal_point_hazard_total_500M.md`) leans the same way
as points 1 and 2: one multiplier costs about 50 on hazard totals 17 to 20
even when it trains on them, and only the Lagrange function with a told policy
stays under the budget there, at half the return. It also shows that the end
of a run is not a settled state: with totals held out, one multiplier falls to
zero for about a sixth of the run.

## The plan

| # | Step | Status | Read first | Touches |
|---|---|---|---|---|
| 1 | **Baseline on hazard totals.** Four learners (one multiplier or a Lagrange function, policy sees the context or not), each with two bands of hazard totals held out and as a reference trained on everything. Uniform sampling, three seeds. | seed 0: six of eight runs done, two references training (2026-11-12); written up in `experiments/2026-11-12_goal_point_hazard_total_500M.md`, which also lists what to settle before seeds 1 and 2 | `design/generalization_evaluation.md` | `scripts/experiments/goal_point_hazard_total_500M.py`; `training/contexts/{space,protocol,registry,setup}.py` |
| 2 | **Cost and return per hazard total** from the final checkpoints, and the main figure. | drawn for seed 0. Open: score the last five checkpoints and give every cell a standard error (the write-up of step 1, *What this changes*) | same, *Measurement* | `training/score.py`; `scripts/analysis/generalization_figures.py`, `generalization_training_figures.py` |
| 3 | **Drag as a hidden factor.** A context value the Lagrange function and the critics see and the policy does not; an interpolation and an extrapolation band. | built, CPU-tested, not run | same, *What varies* | `scripts/experiments/goal_point_drag_500M.py`; `crax/envs/safe_goal.py`, `training/contexts/{networks,learner,registry}.py` |
| 4 | **The combinatorial test**: many hazards on a slippery floor, never seen together in training. | built, CPU-tested, not run | same, *The test sets* | `scripts/experiments/goal_point_combination_500M.py`; registry |
| 5 | **Curriculum methods** (PLR, two scores) on the training and test sets of step 1. | built, CPU-tested, not run; compared with step 1's runs | `design/prioritized_level_replay.md` | `scripts/experiments/goal_point_hazard_total_curricula_500M.py` |

Steps 1, 3 and 4 are the same experiment under three evaluation protocols:
24 runs each, about 62 hours on this machine (2.6 hours a run with its
evaluations, measured on the six runs of 2026-11-11 and 12). Step 5 is 12
runs, about 31 hours. The chain
that runs all of them was started on 2026-11-11 (*How to run*).

Step 1 also answers an open question: whether a Lagrange function without
telling the policy the context (`additive blind`) is enough. The expectation
is in the design document; it has not been measured.

The hidden factor was planned as friction. It is **drag**: on this robot the
friction coefficients change nothing, and what brakes it is the damping of its
slide joints (measured on CPU, 2026-11-11; the table is in the design
document). The ranges and bands are chosen from how the robot moves. Whether a
trained policy's cost responds to drag is not known until step 3 has run.

The code review of everything since `main` is still open. Two features were
removed to shorten it; see *Code that was removed*.

## How to run

An experiment is a definition file under `scripts/experiments/`. The four of
the plan:

| Step | Definition | Jobs |
|---|---|---|
| 1 | `goal_point_hazard_total_500M.py` | 24 |
| 3 | `goal_point_drag_500M.py` | 24 |
| 4 | `goal_point_combination_500M.py` | 24 |
| 5 | `goal_point_hazard_total_curricula_500M.py` | 12 |

```bash
# the labels of the jobs, in the order they run (seed by seed)
.venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_500M.py --list

# all of them, one after another on this machine, detached (~2.6 h each)
setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_500M.py \
  > runs/logs/goal_point_hazard_total_500M_chain.log 2>&1 < /dev/null & disown

# only some jobs
.venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_500M.py \
  --only additive_blind_seed0 additive_told_seed0
```

A local chain starts each job from the files in its folder at that moment.
**Do not edit code in a folder a chain runs from**; run it from a second
checkout.

All four experiments, seed by seed, each block scored and drawn when it is
done (`scripts/experiments/generalization_chain.sh`). The first complete
comparison, step 1 at seed 0, exists after eight runs:

```bash
git worktree add ../CRAX-run <commit>     # a checkout nobody edits
cd ../CRAX-run && mkdir -p runs/logs
PYTHON=$OLDPWD/.venv/bin/python setsid nohup scripts/experiments/generalization_chain.sh \
  > runs/logs/generalization_chain.log 2>&1 < /dev/null & disown
```

Its progress is in `../CRAX-run/runs/logs/generalization_chain.log`, its
score tables in `../CRAX-run/runs/scores/` and its figures in
`../CRAX-run/docs/acl/experiments/figures/<group>/`. To stop it, kill the
script's process group; the run in progress is lost.

After the runs, the result (step 2). Scoring needs the checkout at the commit
the runs trained on, or one that leaves the policy's code untouched:

```bash
# every finished run of the group, on every cell of its protocol -> runs/scores/<group>/<run>.csv
.venv/bin/python -m training.score --group goal_point_hazard_total_500M

# the combination has 210 cells, all rolled out at once: fewer episodes per cell
.venv/bin/python -m training.score --group goal_point_combination_500M --episodes 32

# the main figure and the numbers per learner and region -> docs/acl/experiments/figures/<group>/
.venv/bin/python -m scripts.analysis.generalization_figures --groups goal_point_hazard_total_500M

# how each run got there: cost per region and the multiplier through training, and the end state
# (the mean of the last five evaluations) -> the same directory. Reads W&B, needs no score tables
.venv/bin/python -m scripts.analysis.generalization_training_figures --group goal_point_hazard_total_500M

# curricula next to the uniform baseline they are compared with
.venv/bin/python -m scripts.analysis.generalization_figures \
  --groups goal_point_hazard_total_curricula_500M goal_point_hazard_total_500M
```

One run by hand:

```bash
.venv/bin/python -m training.train_env --env_name safe_goal_point --alg ppo_lag \
  --env_kwargs '{"hazard_lidar": "surface_by_kind"}' \
  --context_distribution uniform --evaluation_protocol hazard_total \
  --constraint additive --policy_context told \
  --deployment_distribution level:3 --initial_lambda_lagr 0.1 --lagrangian_coef_rate 1 \
  --num_envs 8192 --num_timesteps 500_000_000 --num_evals 21 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --measure_performance --skip_rollout --skip_video --store_model true \
  --wandb_group <experiment_group>
```

| Flag | Meaning |
|---|---|
| `--context_distribution` | how training contexts are sampled: `uniform`, `level:<n>`, `staged:<n>,<n>`, `plr`, `plr:<signal>` |
| `--evaluation_protocol` | which contexts are held out of training: `none` (default), `hazard_total`, `drag` or `combination` |
| `--reference` | train on the held-out contexts too; the run that shows what was achievable on them |
| `--constraint` | `scalar` (one multiplier, the published method), `additive` or `network` (a Lagrange function λ(ω)) |
| `--policy_context` | `blind` (sensors only) or `told` (the context too, except hidden factors such as the drag); the critics always see the context |

While it trains, every run is evaluated on `deployment` (level 3), `uniform`
(all of its protocol's contexts), `training` (the training contexts) and each
held-out region, logged as `evaluation/<name>/*`:

| Protocol | Held-out regions |
|---|---|
| `hazard_total` | `interpolation` (totals 11 to 13), `extrapolation` (totals 17 to 20); the drag is 1 |
| `drag` | `interpolation` (drag 0.95 to 1.1, which holds the benchmark's drag 1), `extrapolation` (drag 0.5 to 0.65) |
| `combination` | `combination` (totals 11 to 20 at drag 0.5 to 0.9) |

`evaluation/*/episode_outside_arena` should be 0 everywhere: under the stock
solver a robot could be thrown out of the arena (`known_defects.md`).
Without a protocol (`none`) a goal-point run now trains on every drag from 0.5
to 1.5; the experiments of October ran at drag 1 and are reproduced from their
tag.

Watch recorded episodes of a group, and build its W&B view:

```bash
.venv/bin/python -m training.review --group <wandb_group>      # http://127.0.0.1:8790/
.venv/bin/python -m training.dashboard --group <wandb_group>
```

Snellius: `.venv/bin/python -m training.experiments.snellius <definition> --partition gpu_a100 --time 03:00:00`
(one job per arm and seed; it runs a pushed commit, never the working folder).

CPU tests, about 45 minutes. Two upstream test files fail on their own
(`known_defects.md`):

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

## Rules

- No GPU run without asking.
- W&B is mandatory.
- Contexts are values, not shapes: a new compiled program costs about 50 s.
- Anything that changes the benchmark's behaviour goes into `upstream_notes.md`.
- A bug that affects results goes into `known_defects.md`.
- Clear names, no abbreviations, one module per concept.

## Code that was removed

Removing code does not remove the evidence it produced. Every experiment up to
2026-10-21 ran from commits contained in the git tag `experiments-2026-10`;
its write-up, figures, W&B runs and checkpoints are unchanged. To rerun or
replay one, check out that tag. To bring a feature back on newer code, revert
its removal commit.

| Removed | Why | Commit |
|---|---|---|
| The memory trace (`--memory_time_constant`) | changed neither the cost tail nor the return (2026-10-19) | `7a9c394` |
| Training a context run from pixels | vision was set aside (2026-10-18) | `7c7d8af` |

Arms added to `goal_point_learners_500M` on 2026-11-05 (`scalar told`,
`additive blind`) were never run. Step 1 replaces them, on the current lidar.

## Vocabulary

| Term | Meaning |
|---|---|
| context ω ∈ Ω | the per-episode parameters of the environment: hazard counts, goal radius, drag. A value per parallel environment, never an array shape. |
| drag | how strongly the floor brakes the robot, as a multiple of the benchmark's (1). Low drag is a slippery floor. Hidden from every policy. |
| distribution | decides which context each parallel environment gets when its episode starts |
| evaluation protocol | which contexts a run trains on and which are held out to test it on |
| region | a named part of a protocol's contexts: `training`, or a held-out test such as `interpolation` |
| cell | a small piece of a protocol's contexts with one value of each factor (one hazard total, one band of drag); the final policies are scored per cell |
| reference run | the same learner trained on the held-out contexts too |
| learner | how the constraint is held (`scalar`, `additive`, `network`) and what the policy sees (`blind`, `told`) |
| arm | everything that differs between the runs of one experiment: distribution, learner, reference or not, environment overrides |
| round | one PPO training step; one compiled call |
| budget | the limit on the mean cost per episode (25) |

## Documents

```
docs/acl/
  README.md                          this file
  thesis.md                          the working paper: question, related work, problem, learners and hypotheses, experiments, results
  known_defects.md                   faults in code or recorded data that a reader of results must know
  upstream_notes.md                  upstream bugs, and where this fork changes the benchmark (numbered; for Tristan)
  performance_measurement.md         how to profile a run
  design/
    generalization_evaluation.md     CURRENT: the question, the learners, the factors, the test sets, the measurement, with reasons
    contextual_learner.md            the Lagrange function λ(ω), the {sensors, context} observation, blind and told policies
    contexts_package.md              how training/contexts works and how it is wired into the trainer (regions, protocols, cells)
    training_round.md                why one compiled call is one training step
    hazard_activation.md             how a hazard count becomes a value
    context_spaces_by_suite.md       Ω for each suite
    per_slot_constraints.md, what_can_vary_per_slot.md   what may differ between parallel environments
    intended_vs_realised_curriculum.md   sampled and experienced contexts differ, and why both are logged
    learning_signals.md              what a curriculum may read from the learner
    prioritized_level_replay.md      PLR as a context distribution (for step 5)
    dashboard.md, episode_review.md  the W&B view and the episode review page
  decisions/                         standing_decisions.md (the environment and trainer decisions every experiment rests on),
                                     and one file per decision that closed a line of work
  experiments/                       one file per finished experiment: question, setup, results
  measurements/                      dated performance measurements
  literature/                        the survey of zero-shot generalization (Kirk et al.), the prior-art search on
                                     per-context constraints, an earlier review of curricula under constraints
  archive/                           not current; kept so older write-ups have something to point to
```
