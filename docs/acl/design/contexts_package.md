# `training/contexts` — what exists, what is next

Status (2026-09-20, after review): package implemented, **wired into the
trainer and the CLI**, CPU-tested end to end (17 tests) and smoke-tested on the
GPU (`safe_velocity_ant`, `staged:1,2,3`, 8192 envs, 10 rounds — W&B group
`smoke`). The first experiment can be launched; see
`docs/acl/experiments/2026-09-20_uniform_vs_staged_velocity_ant.md`.

## What exists

```
training/rounds.py             RoundHook (Protocol) — the trainer-side contract, curriculum-agnostic
training/contexts/
  space.py                     ContextSpace, Dimension, box()      — Ω as a typed box; encode/decode/sample/contains
  distribution.py              ContextDistribution (Protocol), EpisodeFeedback, Params
  distributions/{uniform,fixed,staged}.py   r, FixedContext (level / w), StagedContexts (manual curriculum)
  registry.py                  suite_contexts(env_name) -> SuiteContexts {space, level_contexts, parameter_names}
  wrapper.py                   ContextualAutoResetWrapper, make_wrap_env_fn, attach_parameters, current_*
  rollout.py                   RoundRollout (all transitions of a round, host) -> EpisodeFeedback (completed episodes)
  training_curriculum.py       the training_curriculum/* metrics: intended, sampled (≈ q) and experienced (q̂) per round
  round_hook.py                ContextRoundHook(RoundHook): store rollout, update φ, attach it, return the round's metrics
  setup.py                     '--context_distribution' / '--deployment_distribution' specs -> train(**kwargs)
tests/test_contexts.py            8 tests: wrapper mechanics (CPU, ~35 s)
tests/test_context_training.py    9 tests: specs, rollout, metrics, tiny PPO-Lag runs per distribution (CPU, ~2 min)
```

## How it is wired (T1–T3, done)

The problem: a distribution *samples* inside the compiled training step from
frozen φ and *learns* on the host between calls. So one compiled call must be
one training step, the round's transitions must reach the host, and something
must be allowed to edit the environment state before the next call.

The trainer (`training/agents/ppo/train.py`) gained two optional arguments,
forwarded by `ppo_lag` (other PPO-family trainers still need the two lines):

- `round_hook: training.rounds.RoundHook` — three members: `extra_fields`
  (which `state.info` keys to record per transition), `observe(rollout)` (host,
  called from one `jax.debug.callback` next to the existing metrics callback,
  receives every recorded field as NumPy), `on_round_end(round_index, env_state)`
  (host, after each compiled call; may return an edited env state). When set,
  the trainer runs the same training steps **one per compiled call** instead of
  all inside one call (`epochs_per_evaluation = num_training_steps_per_epoch;
  num_training_steps_per_epoch = 1`). Same rounds, same learning, same
  evaluation cadence; only the call granularity changes. The trainer knows
  nothing about contexts; `ContextRoundHook` is the one implementation.
- `evaluation_wrap_env_fns: {name: wrap_env_fn}` — one `Evaluator` per name,
  all wrapping the same raw eval env; metrics under `eval/<name>/...`.

Without either argument the trainer runs exactly as before (test
`test_stock_trainer_path_is_unchanged_without_a_hook`).

CLI (`training/config.py`, used by `training/train_env.py`). Two flags, one
grammar, no suite-specific arguments:

```
--context_distribution    none | uniform | level:<n> | staged:<n>,<n>,...    what the student trains on (default none = stock CRAX)
--deployment_distribution uniform | level:<n> | staged:...                   w, what the policy is for (default level:3)
```

Every context run is evaluated on `deployment` (w) **and** on `uniform` (r):
`evaluation/deployment/*`, `evaluation/uniform/*`. Run names become
`<env>_ctx_<uniform|staged123|level1>_<alg>_seed<s>_<ts>`; W&B config gains
`context_space`, `training_distribution`, `deployment_distribution`,
`num_rounds`, `environment_steps_per_round`. The setup also sets the trainer's
`training_metrics_steps` to one round, so `episodic/*` (per-episode return,
cost, length of the training rollouts) is logged once per round like
everything else.

Per round `ContextRoundHook` logs, via `progress_fn` → W&B (per context
dimension `<d>`, 12 fixed bins over Ω; what is kept and what each key means
is the registry in `training/dashboard/metrics.py`):

| key | meaning |
|---|---|
| `training_curriculum/intended/*` | `distribution.summary(φ_k)` — $q_k$ as the distribution states it (`.../context/velocity_threshold`, `.../stage`) |
| `training_curriculum/sampled/<d>` | `wandb.Histogram`: one count per *completed episode* — the empirical $q_k$ (which contexts were selected) |
| `training_curriculum/experienced/<d>` | `wandb.Histogram`: one count per *transition* — $\hat q_k$ (which contexts the gradient came from) |
| `training_curriculum/{sampled,experienced}/<d>/mean`, `/std` | the one-line summaries that overlay across arms and seeds |
| `training_curriculum/episode_length/<d>` | `wandb.Histogram`: mean completed-episode length per bin — the mechanism behind sampled ≠ experienced |
| `training_curriculum/num_transitions`, `num_completed_episodes` | sample sizes behind the two histograms |

W&B renders a per-step sequence of `wandb.Histogram` as a heatmap over time;
that is the "how does the curriculum evolve" view. Note `sampled` is defined
by completed episodes, so the first round of a stage can be empty (`NaN` mean)
while `experienced` is still 100 % previous-stage data — the lag the design doc
predicts, visible in the smoke run.

The wrapper also writes `state.info["transition_context"]`: the context the
*last transition* was generated in. It differs from `context` only on the step
an episode ends (where `context` is already the next episode's ω, because the
env reads it at the next step). `RoundRollout` is built from
`transition_context`, so both q̂ and the per-episode feedback pair each
episode's return/cost with the context it actually ran in.

### Found in the GPU smoke run

- `training/performance/sinks.py` logged with `run.log(payload)` (no `step=`),
  which advances W&B's step counter past the trainer's; the trainer's own
  `wandb.log(..., step=env_steps)` for the same round was then rejected as out of
  order and **every other round's metrics were silently dropped**. Latent since
  the tracker was written; only fired now that something logs every round. Fixed
  (`step=environment_steps`), verified 0 warnings.
- SPS was 8–54 k and erratic because another process held 7.8 GB and ~11 % of
  the GPU. Throughput numbers from a shared card are meaningless; run the pilot,
  and especially the T4 `level:3` vs `none` pair, on an idle GPU.

Environment side (`crax/envs/context.py`, 2026-09-22 — replaces the earlier
"read the context if present, else the constructor value" fallback):

- **Invariant: every difficulty knob is read from `state.info["context"]`,
  always.** A plain `reset(rng)` is `reset_with_context(rng, self.default_context())`;
  the constructor's knob values exist only to build that default. Stock and
  context runs therefore execute the same `step` code; there is no
  "if a context is present" branch anywhere.
- A suite adopts this with four members: `CONTEXT_PARAMETERS` (names, in Ω
  order), `default_context()`, `reset_with_context(rng, ω)`, and `reset` as
  above; in `step` it reads a knob with `context.parameter(self, state, name)`.
  `crax/envs/context.py` holds the key, the `Protocol` and the two helpers.
- `UnifiedEnvAdapter.reset_with_context` and the wrapper *require*
  `reset_with_context`; nothing attaches a context after a plain reset.
- `tests/test_suites.py::test_the_context_is_the_only_place_the_knob_is_read`
  is the invariant's test, per registered suite: a level-1 env reset into the
  level-3 context must be step-for-step identical to a level-3 env.

Registered environments (15, the whole benchmark): `safe_velocity_{ant,halfcheetah,hopper,humanoid,swimmer,walker2d}`
(Ω = `velocity_threshold ∈ [0.4, 1.0] × baseline`), `safe_height_humanoid`
(`max_height ∈ [0.9, 1.3]`), `safe_push_point` (`goal_velocity ∈ [0, 0.8]`),
`safe_lift_{ant,spider}` (`restrict_<foot> ∈ {0,1}` per foot — a discrete Ω of
16 / 64 masks; the levels are three of them) and `safe_pathway_walker2d`
(`max_gap ∈ [1.5, 8]`, the first knob consumed at reset). Level contexts are
not written down: `level(n)` is `get_environment(env, level=n).default_context()`
(`_level_contexts`), so they cannot drift from `crax/envs/difficulty.py`.
`TaskDescription` carries one typed fact next to its three texts,
`episode_ends_early`, which the dashboard uses to word its episode-length
panels. The count-typed suites — `safe_goal_point` (Ω = four active counts +
`goal_size`), `safe_reacher` (`active_hazards ∈ {0..10}`), `safe_circle_point`
(`active_cylinders ∈ {0,1,2}`, `boundary_x`, `boundary_y`) and `safe_button_point`
(two active counts, `gremlin_travel`, `placement_extent`) — share one mechanism,
`hazard_activation.md`: the model is the union of the levels' hazards and the
context says how many of each group are active. Button's box Ω has an
infeasible corner (README item 4c); Ω as a union of level boxes fixes it.

## The core mechanism, verified

`ContextualAutoResetWrapper` replaces Brax's `AutoResetWrapper` in the stack
`ContextualAutoReset(Episode(Vmap(env)))`. Every step it (1) draws a fresh
context per slot from the frozen parameters in `state.info["distribution_params"]`,
(2) runs the single-slot `reset_with_context` under `vmap`, (3) `where(done, …)`
selects the fresh state, context and per-episode info for slots that finished.
Episode bookkeeping (`steps`, `episode_metrics`, `episode_done`) is kept from
the stepped state exactly as before, so the trainer and PPO-Lagrange see nothing
new.

Tests confirm: contexts differ per slot; the env's threshold equals its context
at every step; contexts change exactly at episode end; `FixedContext` holds;
`StagedContexts` switches level 1 → 2 → 3 with **zero recompilation** of the
step program (only the usual ms-scale host helpers). The stock wrapper's
frozen-layout behaviour is gone: every episode starts from a fresh `reset()`.

Distribution parameters φ are a small pytree; `attach_parameters` broadcasts
them over the slot axis so the pytree *structure* matches what the program was
traced with. Remember to `_strip_weak_type` the state after attaching, as
`train()` already does on every call.

## How the wrapper swap works (no trainer change needed for the swap itself)

Every CRAX trainer (`ppo`, `ppo_lag`, `focops`, `p3o`, `crpo`, `ppo_pid`, `sac*`)
already takes `wrap_env_fn: Optional[Callable]` and, in `_maybe_wrap_env`, calls
`wrap_env_fn(env, episode_length=, action_repeat=, randomization_fn=)` **in place
of** `crax.envs.training.wrap`. That hook is the swap point:

```python
from training import contexts
train(environment=env, wrap_env_fn=contexts.make_wrap_env_fn(distribution), ...)
```

`make_wrap_env_fn` returns a callable with `wrap`'s exact signature that builds
`ContextualAutoReset(Episode(Vmap(env)))` instead of `AutoReset(Episode(Vmap(env)))`.
Nothing in `crax/envs/wrappers/training.py` changes; without a distribution the
benchmark behaves exactly as before. The eval env goes through the same hook, so
evaluation on w or r is `wrap_env_fn=make_wrap_env_fn(FixedContext(w))` etc.
(the trainer currently uses one `wrap_env_fn` for both; a second parameter or a
tiny `eval_wrap_env_fn` is the cleanest way to give eval its own distribution).

## Not yet done

1. **Measure the reset-every-step cost** (T4; `--measure_performance`, Ant @
   8192, `--context_distribution level:3` vs `none --difficulty 3`, idle GPU):
   the one open performance question. `level:3` is the right comparison — same
   task as stock, only the wrapper (and one-call-per-round) differs.
2. **Other PPO-family trainers** (`focops`, `p3o`, `crpo`, `ppo_pid`,
   `ppo_saute`, `ppo_cost`) need `round_hook` and `evaluation_wrap_env_fns`
   forwarded like `ppo_lag` does (two parameters, two kwargs each).
   `train_env.py` refuses `--context_distribution` for algorithms that lack them.
3. **Checkpointing φ**: `ContextRoundHook.parameters` holds φ for the next round
   but is not written to the checkpoint yet (matters for RQ4 branching, not for
   the first experiment).
4. **`train_curriculum.py`** still runs the old per-stage `train()` restart. It
   is superseded by `--context_distribution staged:1,2,3` for registered suites
   and should eventually route through it for them.
5. **Thesis figures**: pull `training_curriculum/*/bin_NN` for all runs via the W&B API
   into a DataFrame and plot heatmap-per-arm + mean-line panel (`results/`).

## First experiment (design)

`safe_velocity_ant`, PPO-Lag, same budget, ≥ 3 seeds, four arms:

| arm | distribution | note |
|---|---|---|
| direct L3 | `FixedContext(level 3)` | paper's "Normal" |
| curriculum | `StagedContexts(levels 1,2,3, equal split)` | paper's "Curriculum", now without recompiles |
| uniform | `UniformDistribution(Ω)` | reference r |
| direct L1 | `FixedContext(level 1)` | sanity anchor |

Evaluate every arm on both w (level 3) and r (uniform over Ω); report R and C
with the 25 cost budget, and q̂ per round.
