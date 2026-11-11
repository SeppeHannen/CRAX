# Known defects

Faults in the code or in recorded data that a reader of the results has to know
about. A bug found during an experiment is added here, not narrated in the
experiment's write-up.

- **Goal-point's contact solver did not converge; every cost up to
  2026-10-21 was measured under it** (found and fixed 2026-11-11). The stock
  benchmark solves goal-point's contacts with the CG solver at 4 iterations. A
  robot pressed against a wall or a solid hazard makes tens of constraint rows,
  and CG at 4 iterations does not converge on them. The robot is then kicked:
  one step at 10 to 70 m/s against a top speed of 1.5.
  - *What it did.* Measured on a GPU, no policy: full thrust with a small
    constant turn for 400 steps, level-3 hazards, 4096 episodes. 86 % of the
    episodes had a kick and 5.4 % ended outside the fence (22 % at drag 0.5).
    Outside there is neither goal nor hazard, so such an episode costs nothing.
    A robot that is kicked away from a hazard also stops paying for it: the
    mean cost of this behaviour was 19.7 under the stock solver and 47.4 once
    the solver converges.
  - *Trained policies.* None of the 1178 recorded evaluation episodes on this
    machine (five runs of 2026-10-17) left the arena. 15 of them have a step
    faster than 5 m/s, so trained policies were kicked about once in 80
    episodes. How much that moved their cost, the gap trap of 2026-10-19
    included, is not known.
  - *The fix.* Goal-point now uses MJX's Newton solver at the same 4
    iterations (`crax/envs/safe_goal.py`). MJX's documentation recommends it on
    GPU, and the benchmark's locomotion suites already use it. Same test: a
    kick in 0.2 % of episodes, none faster than 5.6 m/s, 0 of 24 576 episodes
    outside the fence over three drags with and without hazards. It costs 13 %
    of the simulation's steps per second.
  - *Also tried, not kept.* A speed cap (no effect), higher walls (0.8 % still
    left), a frictionless wall contact (0.2 % still left, and the kicks
    stayed), CG at 8 iterations (no escapes, kicks in 5 to 12 % of episodes,
    40 % slower on CPU), Newton at 2 iterations (1.8 % left) and at 8 (no
    kicks at all, less than half the speed).
  - *What follows.* Costs measured from 2026-11-11 on are not comparable with
    the October experiments: the same behaviour costs more when the robot is
    no longer kicked off the hazard. The other point-robot suites (push,
    circle, button) still use CG at 4 iterations (`upstream_notes.md`, item
    28). Every evaluation reports `episode_outside_arena`; it should be 0.

- **Two upstream test files fail** (found 2026-11-10): `tests/test_point_push_goal.py`
  and `tests/test_pixel_observation.py`, 39 failures and 20 errors together.
  Both are older than upstream's own refactors: the first builds `SafePush`
  directly, which is now an abstract class with a subclass per robot; the
  second builds a vision environment without `num_envs`, which the GPU pixel
  wrapper now requires. They fail the same way on the tag
  `experiments-2026-10`, so they do not come from the changes since, and they
  do not test code the experiments run. Upstream's to fix or delete
  (`upstream_notes.md`, item 29). A full test run is read as "nothing else
  fails".

- **A 128-episode cost mean cannot carry a policy with the gap trap** (found
  2026-10-19). One episode in a hundred costs 150–1000 under the centre lidar;
  whether one lands in the evaluation's 128 episodes moves the mean by 5–8 and
  the sd from 14 to 108 between evaluations 25 M apart. Every `evaluation/*/
  episode_cost` of 2026-10-15, 10-17 and 10-18 carries that noise; the medians
  and over-budget shares the evaluator logs since 2026-10-19 do not, and
  replays of 2048 episodes are the numbers to quote. Not a pipeline fault:
  reproduced the 10-15 run's final evaluation from its checkpoint with the
  trainer's key chain (24.8, sd 89, one episode of 1014; fresh keys 17.4–18.6).

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
