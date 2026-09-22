# Automated curriculum learning on CRAX

Giuseppe Hannen's graduation project. Start here.

## The plan

Done (2026-09-19 → 21):

1. **Profile cleanly.** Per-round throughput, compiles and XProf traces go to
   W&B for every run. `training/performance/`, guide in
   `performance_measurement.md`.
2. **Manual curriculum vs uniform on `safe_velocity_ant`.** PPO-Lagrange,
   staged 1 → 2 → 3 vs uniform over Ω, evaluated on level 3. One seed, 500 M
   steps: uniform solves level 3; the staged curriculum collapses at the first
   switch (Lagrange multiplier blow-up) and never recovers. Write-up:
   `experiments/2026-09-20_uniform_vs_staged_velocity_ant.md`. Repeated at
   750 M on the new dashboard (group `velocity_ant_staged_vs_uniform_750M`),
   to be read with Tristan and written up.
3. **A W&B dashboard one can read.** `design/dashboard.md` derives it from what
   reviewing a run is (Verdict / Mechanism / Trust / Detail);
   `training/dashboard/` implements it: a metric registry every logged key
   must pass (kept with a one-sentence description of where its numbers come
   from, or dropped with a reason; anything else raises), budgets logged next
   to costs, and one W&B view per experiment group whose sections open with
   generated text — the environment in the suite's own words, then cadence,
   population and one line per panel. A run with `--wandb_group` creates the
   group's view at start-up and is refused if its flags differ from the
   group's.

Next:

**Redirection from Tristan (2026-09-22).** Three things, all taken on:

- **Ω should be the environment's physics and layout, not its cost
  threshold.** A run has one cost threshold and keeps it; what varies per
  episode is the world — friction, masses, actuator gains, gravity, the
  number and positions of hazards. This is also what the 500 M ant result
  already said from the other side: a policy that cannot see ω learns one
  behaviour for all of Ω, so a threshold Ω can only teach it *one* speed; a
  physics Ω asks for different behaviour per ω, which is what a curriculum
  orders. Consequence for the code: the `velocity_threshold`, `max_height`,
  `goal_velocity`, `max_gap`, `restrict_<foot>` spaces are the wrong Ω for the
  research question, but the machinery they exercised (`crax/envs/context.py`,
  the wrapper, the registry, the tests) is exactly what a physics Ω needs — a
  `friction` or `body_mass` dimension is one more `tree_replace` in `step`
  (verified: `design/what_can_vary_per_slot.md` §2c). They stay as the smoke
  test of the pipeline; the experiments move to a physics Ω.
- **First experiment on `safe_goal_point`**: Tristan reports it trains with
  low variance, so a curriculum effect is detectable with few seeds. Its
  natural Ω is hazard count and placement (pad-and-mask, item 4b) plus point
  physics (friction, mass). This makes reach/goal the *next* engineering step,
  not the last.
- **Research before designing item 5**: (i) **CARL** (Benjamins et al., the
  contextual RL benchmark) also varies MJX/Brax physics per context — read how
  they expose ω to the agent and what they vary; (ii) the **contextual MDP**
  literature (Hallak et al. 2015; Modi et al. 2018) and the meta-RL line
  (RL², VariBAD, and the belief-state reading already listed under item 5) to
  decide *told* vs *infer* on evidence, not taste; (iii) what the curriculum-
  learning papers we compare against actually do about the observation
  (PLR, ACCEL, SPaCE, CURROT). Output: a short design doc,
  `design/context_observation.md`, with a decision and its reasons.

4. **Is what we built stable across suites?** Everything so far ran on one
   suite. Repeat step 2 — staged vs uniform, one seed, the dashboard — on the
   other value-typed suites in the order `design/context_spaces_by_suite.md`
   recommends: the five remaining velocity agents first (nothing to change but
   the flag), then height and push (a `step`-side context read each), then
   lift and pathway (a mask Ω; the first `reset_with_context`). What we are
   checking, per suite: the context wrapper and round hook run without a
   recompile; every logged key is registered or the run fails at first log
   (new reward components will — that is the point); the generated text is
   right for the suite's task; and the staged-vs-uniform result either
   repeats or does not. Count-typed and structural suites (reach, circle,
   goal, button) wait for pad-and-mask.

   *Velocity agents, CPU part done 2026-09-22* —
   `experiments/2026-09-22_velocity_suites_staged_vs_uniform.md`. A per-suite
   test (`tests/test_suites.py`: every logged key registered, levels reproduce
   `difficulty.py`, env reads its context, task text quotes the env's
   constants) found and fixed four things before any GPU time, two of them
   upstream bugs to raise with Tristan: `get_environment("safe_velocity_*",
   level=n)` has ignored the level since upstream `583e7fd` (stock level-2/3
   velocity runs since May were level 1; our `level:n` runs unaffected), and the
   swimmer could not step under the installed JAX (`jp.clip(a_min=)`). The
   other two were ours: five agents' own reward-term names were unregistered
   (now dropped under one pattern; `metrics.py` names no agent), and the task
   and episode-length descriptions said every agent falls (halfcheetah and
   swimmer cannot; the fact now lives only in the suite's `TaskDescription`,
   tested against the env). GPU runs (commands in the doc) not started — ask
   first.

   *Scope widened 2026-09-22: the full benchmark, all nine suites,* including
   the count-typed and structural ones. Plan, per-suite Ω, order and the three
   facts a line-by-line audit added: `design/context_spaces_by_suite.md`,
   section "Plan for the full benchmark". Order: height, push, lift, pathway
   (value-typed; small), then reach (builds the hazard `active_mask`), then
   circle, goal, button (reuse it).

   *How a suite reads its context is now one pattern* (`crax/envs/context.py`,
   2026-09-22): every knob is read from `state.info["context"]` always; a plain
   `reset` is `reset_with_context(rng, default_context())`; no "if a context is
   present" branch anywhere (the velocity fallback is gone). The invariant's
   test, per registered suite: a level-1 env reset into the level-3 context is
   step-for-step identical to a level-3 env. **All five value-typed suites
   done** (11 registered: 6 velocity, height, push, lift × 2, pathway): one
   read each for height/push/pathway; lift's cost rewritten as all-feet
   contact × per-episode mask (same numbers at every level, tested); push's
   `lax.cond` on a Python float became a per-slot select. **GPU smoke runs of
   all five passed** (10 rounds each, staged 1→2→3, one W&B view per suite;
   deleted afterwards): no error, no unregistered key, the stage switch and
   its lag visible, zero substantial compiles. Throughput: lift ant 438 k,
   spider 342 k, pathway 133 k, push 45 k, height 29 k SPS (height is on Brax's
   `generalized` backend, push's placement loop runs every step — both stock).
   Found while reading the numbers: `performance/epoch_compiles` counted JAX's
   ~9 few-ms host-side helper compiles per round and the dashboard text called
   them recompiles; it now counts only compiles ≥ 1 s (`SUBSTANTIAL_COMPILE_SECONDS`,
   one definition in the tracker) and reads 0.

   **Goal and reach done (2026-09-22, option A: Ω contains the ladder).** The
   hazard activation mechanism is `design/hazard_activation.md`. For goal,
   `difficulty.py` builds every level from the *union* of the three levels'
   hazard specs (`crax/envs/hazard_union.py`: 12 discs + 8 solid cylinders +
   6 squares + 4 solid cubes + walls) and the level is the per-group active
   counts; the env parks inactive hazards at reset (`place_objects(activation=)`)
   and cost, lidar and compass ignore them. Ω = four integer counts +
   `goal_size`; L1/L2/L3 are points of it. Reach: one dimension
   `active_hazards ∈ {0..10}`, every level built with 10. Both pass the
   invariant test (level-1 env reset into the level-3 context ≡ level-3 env,
   observation and all). Two benchmark changes for Tristan: goal's hazard
   *size* within a group is the union's max (level 3's discs 0.4 not 0.35, its
   solid cylinders 0.3 not 0.25), and every goal level now carries 34 hazard
   bodies (18 collidable) — the per-slot cost of that is the first thing to
   measure on the GPU.

   **Circle and button done; all 15 environments of the nine suites registered
   (2026-09-22, pushed).** 120 CPU tests pass (`tests/test_suites.py` × 15 plus
   contexts, training, dashboard; 30 min). Circle: Ω = `active_cylinders ∈ {0,1,2}`,
   `boundary_x`, `boundary_y` ("no y-boundary" encodes as the arena half-width 3 m);
   the boundary cost is one `jp.where` over context values. Button: Ω = two
   active counts (12 solid blocks, 8 gremlins), `gremlin_travel`, `placement_extent`;
   the layout sampler takes per-object keepouts and activation from the context.
   Shared: `HazardGroup.context_name`, `HazardManager.validate_active_counts`.

   *Known defect, fixed by the next task:* button's Ω is a box whose corner
   (12 blocks, 8 gremlins, 0.5 m orbits, 2 m square) has no feasible layout;
   `Uniform(Ω)` samples it ~0.7 % of the time and the reset raises (measured,
   64 seeds per corner: half-width ≥ 2.5 m fits everything, 2.0 m fits nothing
   at the full count). Tristan never hits it because his levels couple the
   square to the crowd. **Do not start a uniform button run until item 4c.**

   Three more things for Tristan: circle level 1 observes 60-d (the union model
   keeps the two hazard blocks, zero-valued) where stock level 1 observed 28-d
   (levels 2–3 were 60-d already); button's cylinders are `cube` groups under
   MJX (stock behaviour, now visible in the context name `active_collidable_cubes`);
   and stock `training/curriculum.py` warm-starting circle L1→L2 copies the
   28-d policy onto a 60-d observation — `ppo/train.py:815-836` checks only the
   normaliser's shape and copies policy params unconditionally, so it raises.

4c. **Ω as the union of level boxes (Giuseppe, 2026-09-22).** Redefine Ω for
   every suite as B₁ ∪ B₂ ∪ B₃, one box of parameters around each level, and
   "uniform over Ω" as: pick a level with probability 1/3, then uniform inside
   its box. This makes Tristan's coupling (more objects ⇒ bigger square) the
   structure of Ω instead of a suite-specific line, removes button's corner,
   and keeps the in-between of the ladder as the boxes' interiors. Design first
   (`design/context_space_as_level_boxes.md`): the box-width rule per level
   (proposal: widen only along the dimensions the ladder varies, halfway to the
   neighbouring level, so the boxes tile the ladder without gaps or overlaps),
   the per-suite boxes, and that each box's corners are probed for feasibility
   the way button's were. Then the code; the files that know Ω is one box are
   exactly: `training/contexts/space.py` (`ContextSpace`, `sample_uniform`,
   `contains`, `describe`, `low`/`high`), `training/contexts/distributions/uniform.py`
   (`uniform_log_density`: becomes ⅓ · 1/vol(B_ℓ), no longer a constant),
   `training/contexts/registry.py` (15 entries become per-level boxes;
   `SuiteContexts.__post_init__`), `training/contexts/training_curriculum.py:78`
   (histogram bounds → the union's envelope), `training/dashboard/view.py:174`
   (Ω text), `tests/test_suites.py` (uniform-differs-per-slot; add: every level
   box's corners reset without raising). Untouched: environments, wrapper,
   `FixedContext`, `StagedContexts`. The velocity staged-vs-uniform experiment's
   uniform arm changes meaning and is rerun afterwards.
5. **Let the learner know its context — research task first (Tristan).** Today
   the policy sees only the state, so it can learn one behaviour for all of Ω
   and nothing else. Two ways to change that, both change the benchmark's
   observation space (agree with Tristan first):
   - **told**: ω in the observation — a contextual CMDP, the oracle;
   - **must infer**: the previous step's cost (and reward) in the observation,
     with or without memory (frame stack; recurrent policy). Measure the gap to
     the oracle before building memory: it is the value of inferring ω.

   Before choosing, read (a) **CARL** — how it varies MJX/Brax physics per
   context and whether/how it exposes ω (it has a flag for both modes; what do
   their results say?); (b) the contextual-MDP papers and (c) what the
   curriculum methods we will compare against do about the observation. Write
   the decision up in `design/context_observation.md`.

   *Reading, before designing this:* with ω hidden, the agent's problem is a
   **POMDP** whose hidden state is (s, ω); it is Markov again only over the
   **belief state** b_t = p(ω | history), and a recurrent policy is a learned
   approximation of that belief. A **sufficient statistic** is any compression
   of the history that preserves the belief — for the velocity suite it is just
   the tightest bracket [max v with cost, min v without cost]. Look up: belief
   MDP / Bayes-adaptive MDP (Duff 2002; Ghavamzadeh et al. 2015 survey),
   RL² (Duan et al. 2016) and VariBAD (Zintgraf et al. 2020) for memory-based
   meta-RL that feeds (s, a, r) back into the policy, and the definition of a
   sufficient statistic (Fisher–Neyman) for why "remember everything" is
   overkill.
6. **Repeat 2 with PPO-Saute.** It augments the state with the remaining safety
   budget and enforces the constraint per episode rather than in expectation,
   and has no dual variable that carries the old distribution across a switch.
7. **Then** actual curriculum methods, and the count-typed and structural
   suites.

## Vocabulary

| term | meaning |
|---|---|
| context ω ∈ Ω | the per-episode parameters of the *world* a suite instantiates — physics (friction, mass, gravity), layout (hazard count, positions). Not the cost threshold: a run keeps one threshold (Tristan, 2026-09-22). A value on the `num_envs` axis, never a shape. |
| distribution | decides which ω each parallel environment gets when its episode starts. `uniform`, `level:3`, `staged:1,2,3`. |
| round | one PPO training step; with a context distribution, one compiled call. |
| sampled vs experienced | which contexts were *chosen* (per episode) vs which the gradient *came from* (per transition). They differ when episode length depends on the context. Always shown together. |

## Running

```bash
# manual curriculum
.venv/bin/python -m training.train_env --env_name safe_velocity_ant --alg ppo_lag \
  --context_distribution staged:1,2,3 --deployment_distribution level:3 \
  --num_envs 8192 --num_timesteps 50_000_000 --num_evals 9 --episode_length 1000 \
  --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
  --measure_performance --skip_rollout --skip_video --store_model false

# uniform: same, with --context_distribution uniform
```

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
  README.md                          this file
  performance_measurement.md         how to profile a run (XProf + W&B)
  measurements/                      dated measurement logs
  experiments/                       one file per experiment
  design/
    contexts_package.md              how training/contexts works and how it is wired into the trainer
    intended_vs_realised_curriculum.md   why sampled ≠ experienced, and why both are reported
    training_round.md                why one compiled call is one training step
    per_slot_constraints.md          what may differ between parallel environments (the rule)
    what_can_vary_per_slot.md        the mechanics: which mjx.Model fields are values vs structure, where a per-slot value can land, what a union model costs (probed)
    hazard_activation.md             how a hazard COUNT becomes a value: the union model, per-hazard activation, parking, the four consumers
    context_spaces_by_suite.md       Ω for each of the nine suites, and the plan for the full benchmark
    dashboard.md                     what a reviewer needs from W&B, key inventory, the eight primary panels
  profiling_options.md               tool survey (historical)
```
