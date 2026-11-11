# Notes for the upstream author (Tristan)

Upstream bugs found, and every place where this fork changes the benchmark's
behaviour. Anything that changes a baseline, a wrapper or a model is added
here so that it can be raised with him. The numbers are stable: older
write-ups cite them ("item 25").

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
19. *Withdrawn 2026-11-10.* `GpuPixelObservationWrapper` had been changed to
    keep the parts of a dict observation in `pixels+state` mode, for context
    runs from pixels. Those runs were removed, and the wrapper is the
    benchmark's own again (the change is at the tag `experiments-2026-10`).
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
25. Built for the cost-tail experiment (2026-10-19), opt-in and off by default,
    so the benchmark's runs are unchanged: `hazard_lidar="surface_by_kind"` on
    goal-point (78-dimensional observation, two arrays by kind, surface
    closeness over the subtended arc). Every experiment since uses it. Also:
    the evaluator logs the median episode cost and the share of episodes over
    1× and 4× the budget whenever the agent hands it a budget (`CostTail`);
    PPO-Lagrange does, the others could. (A memory trace built for the same
    experiment was removed; tag `experiments-2026-10`.)
26. **Drag** (2026-11-11): goal-point's context has a sixth value, `drag`,
    which multiplies the damping of the robot's two slide joints
    (`crax/envs/safe_goal.py`, `_system_in`). The default is 1, which is the
    benchmark's physics unchanged (checked on CPU: positions identical bit for
    bit over 100 steps of 256 robots, same steps per second), so stock runs are
    as before. It is the hidden factor of the generalization
    evaluation. `PipelineEnv.pipeline_step` takes an optional system to step
    for it (`crax/envs/base.py`); without one it steps `self.sys` as before.
27. The friction coefficients of the goal model have no effect on the point
    robot: it moves on two slide joints and has no vertical freedom, so its
    contact with the floor carries no force (measured 2026-11-11: speed
    unchanged to three decimals from 0.05 to 2 times the friction). The
    `friction="1 0.01 0.01"` on the robot and the floor is inert. Is that
    intended? The constructor's `max_velocity` is stored and never read.
28. **Goal-point's solver changed from CG to Newton** (2026-11-11,
    `known_defects.md`). At the stock setting, CG with 4 iterations and a
    0.02 s step, a robot pressed against a wall or a solid hazard is kicked at
    10 to 70 m/s: under blind full thrust 86 % of episodes had a kick and 5 %
    ended outside the fence. With Newton at the same 4 iterations: 0.2 % and
    none, for 13 % fewer steps per second on a GPU. The cost of the same
    behaviour more than doubles, because a kicked robot stops paying for the
    hazard it was pressing on. MJX's documentation recommends Newton on GPU,
    and ant, humanoid, height and lift already use it; push, circle and button
    still use CG at 4 iterations and are untested here. Goal-point also
    reports the steps a robot spends outside the fence as the metric
    `outside_arena`. Was CG a leftover from MJX 3.0, which had no Newton?
29. `tests/test_point_push_goal.py` and `tests/test_pixel_observation.py`
    fail on `main`'s own code (59 failures and errors): the first instantiates
    the abstract `SafePush`, the second calls `envs.create(vision=True)`
    without `num_envs`. Stale since the refactors; fix or delete?
