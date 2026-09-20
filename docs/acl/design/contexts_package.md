# `training/contexts` — what exists, what is next

Status (2026-09-20): package implemented and CPU-tested; **not yet wired into
the trainer**. First target experiment: uniform vs manual curriculum on
`safe_velocity_ant`.

## What exists

```
training/contexts/
  space.py          ContextSpace, Dimension, box()      — Ω as a typed box; encode/decode/sample/contains
  distribution.py   ContextDistribution (Protocol), EpisodeFeedback, Params
  distributions.py  UniformDistribution (r), FixedContext (one level / target w), StagedContexts (manual curriculum)
  registry.py       suite_contexts(env_name) -> SuiteContexts {space, level_contexts, parameter_names}
  wrapper.py        ContextualAutoResetWrapper, wrap_for_context_training, attach_parameters, current_contexts
tests/test_contexts.py   8 tests, CPU, ~25 s
```

Environment side:

- `crax/envs/safe_velocity.py`: `SafeVelocityBase` reads its threshold from
  `state.info["context"]` when present (`_threshold(state)`), falling back to
  the constructor value otherwise, so plain training is unchanged. It exposes
  `CONTEXT_PARAMETERS = ("velocity_threshold",)` and `reset_with_context(rng, ω)`.
- `crax/envs/__init__.py`: `UnifiedEnvAdapter.reset_with_context` forwards to
  the inner env (or attaches the context after a plain reset).

Registered suites: `safe_velocity_{ant,halfcheetah,hopper,humanoid,swimmer,walker2d}`,
Ω = `velocity_threshold ∈ [0.4, 1.0] × baseline` (level 3 is 0.5 ×, so Ω
extends below the hardest level). Count-typed suites (Goal, Reach, Circle) are
not registered; they need pad-and-mask first.

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

## Not yet done (next session)

1. **Trainer wiring** (`training/agents/ppo/train.py`):
   - accept a `context_distribution` (or a `wrap_env_fn` built from one) and use
     `wrap_for_context_training` instead of `envs.training.wrap`;
   - make one compiled call = one training step (`num_training_steps_per_epoch = 1`,
     evaluation every N iterations) — see `training_round.md`;
   - in the loop body: build `EpisodeFeedback` from the finished episodes of the
     round, call `distribution.update`, `attach_parameters`, log
     `distribution.summary(params)` and the realised curriculum;
   - `extra_fields` must include `"context"` so every transition carries it.
2. **Realised curriculum q̂**: histogram of `data.extras["state_extras"]["context"]`
   per round → W&B (`curriculum/realised_*`), alongside `curriculum/intended_*`
   from `summary()`. See `intended_vs_realised_curriculum.md`.
3. **EpisodeFeedback from rollout data**: finished episodes are the transitions
   where `episode_done == 1`; their `episode_metrics` (`sum_reward`, `cost`,
   `length`) and `context` give returns, costs, lengths, contexts.
4. **Evaluation on w and r**: the evaluator env can use `FixedContext(level 3)`
   and `UniformDistribution` respectively; two evaluators or one with two
   distributions.
5. **Measure the reset-every-step cost** (`--measure_performance`, Ant @ 8192,
   uniform vs the stock wrapper): the one open performance question.
6. **CLI**: `--context_distribution {level,uniform,staged}` plus
   `--context_levels 1 2 3` for staged; `--difficulty` maps to `FixedContext`.

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
