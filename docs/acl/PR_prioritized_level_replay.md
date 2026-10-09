# PR: Prioritized Level Replay as the first curriculum method

Uncommitted, 2026-10-08/09. Plan item 1 of `docs/acl/README.md`.

## What this PR does

It adds Prioritized Level Replay (Jiang et al. 2021) as a context
distribution, selectable with `--context_distribution plr`. PLR keeps a buffer
of contexts, scores each by how wrong the reward critic was on the last episode
run in it, and replays high-scoring contexts more often. To make that possible
the trainer now ships one number per transition — the reward advantage — to the
round hook, and the round hook reconstructs whole episodes from the per-round
pieces. Two diagnostics are added for every distribution: the distribution
actually in force each round, shown by sampling it, and the critic's error as a
function of the context. One defect found on the way is fixed: the evaluation
environments were built with the training stack's random episode-phase offset,
so every evaluation metric since 2026-10-06 covered about half an episode.

Design: `docs/acl/design/prioritized_level_replay.md`. Result:
`docs/acl/experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md`.

## How to read this document

The spine is **the walk**: a call tree from `--context_distribution plr` to a
point on the dashboard, through changed and unchanged files alike. Each entry
is a line in a file and what that line does; what it calls is indented beneath
it. So the answer to "where does this change enter the flow" is always the
parent entry. Each entry gives control flow and meaning in separate sentences.
Tags sit on the callee: **[changed]** this PR edits it, **[new]** the file did
not exist, **[unchanged]** it connects the changed parts and there is nothing
to review in it. Each phase opens with what it does for the flow. Line numbers
are the working tree's and were checked by grep.

After the walk: per-file detail for every changed file, the tests, the
documents, and what is deliberately not in this change.

## Verification

| what | status |
|---|---|
| CPU tests: `test_contexts`, `test_dashboard`, `test_context_training`, `test_suites` on the final code | 133 passed, 1 xfailed (the known button corner); 27 min |
| `EpisodeTracker` against a brute-force reference: 25 episodes over 5 rounds, episodes spanning rounds, several ends per slot per round | matched to 1e-6; a scrambled slot/time layout raised |
| PLR buffer behaviour: top-k kept, scores overwritten on replay, newcomers admitted only when better, `sample` reproduces `replay_weights` to 0.02 with bit-exact rows | verified; the same cases are unit tests |
| GPU smoke on the final code: goal-point, 40 rounds | 0 recompiles; 83–87 k SPS (baselines 85 k); every key registered; buffer full by round 2; staleness ≈ one episode; `replay_mass/top_10` = 0.28, the closed form for β = 1, ρ = 0.3 |
| The experiment: 3 arms × 500 M, group `goal_point_uniform_staged_plr_500M_v2` | done; written up |
| The evaluation-stack defect | located by running one trained policy through both pipelines (training wrapper 14.5 per episode, Evaluator 6.7; the Evaluator's head start averaged 462 of 1000 steps); fixed; pinned by a test; audited (B9b) |

---

## The walk

### A. Start-up

*The two CLI strings become four objects the trainer takes — a training
wrapper, one evaluation wrapper per target, a round hook, a logging cadence —
and the trainer is called. Nothing runs on the GPU yet.*

**`training/train_env.py` `main()`** [changed, one argument]. The entry
point; line 139 now passes `num_envs=config.num_envs` to the setup (below).

- **`train_env.py:26`** — `build_base_parser(…)` declares every CLI flag.
  - **`training/config.py:278` `build_base_parser`** [changed]. Line 113
    declares `--context_distribution`; its help text now lists `plr`. The
    flag's value is a string that nothing interprets until line 135 below.

- **`train_env.py:53`** — `contexts.spec_label(config.context_distribution)`
  builds the run name. The dashboard groups runs by the resulting
  `…_ctx_plr_…`.
  - **`training/contexts/setup.py:54` `spec_label`** [unchanged]. Strips `:`
    and `,` from a spec.

- **`train_env.py:135`** — `contexts.context_training_setup(env_name,
  "plr", "level:3", num_timesteps=, num_envs=, batch_size=, unroll_length=,
  num_minibatches=)` turns the two specs into distribution objects. This is
  the one call through which every context-related object enters the run.
  - **`setup.py:114` `context_training_setup`** [changed]. Takes `num_envs`
    (the slot count the round hook needs, below). Computes `steps_per_round`
    = 655 360 and `total_rounds` = 763, then builds the setup.
    - **`setup.py:129`** — `suite_contexts(env_name)` fetches Ω and the three
      level contexts.
      - **`training/contexts/registry.py`** [unchanged]. Knows nothing of PLR.
        A distribution is handed Ω; it does not choose it.
    - **`setup.py:136`** — `parse_distribution("plr", suite, total_rounds)`
      turns the training spec into a distribution.
      - **`setup.py:36` `parse_distribution`** [changed]. A new branch at line
        42 returns `PrioritizedLevelReplay(suite.space)`. The hyperparameters
        are the class defaults — M = 1000, p = 0.5, β = 1, ρ = 0.3. They are
        not on the CLI: one spec is one configuration.
        - **`training/contexts/distributions/prioritized_level_replay.py:49`
          `PrioritizedLevelReplay`** [new]. A frozen dataclass of the four
          hyperparameters and Ω. Its `__post_init__` (line 66) raises on an
          out-of-range value. Nothing is allocated here; φ does not exist
          until the first environment reset (B9).
    - **`setup.py:137`** — `parse_distribution("level:3", …)` turns the
      deployment spec into `FixedContext(level 3)` [unchanged]. This is $w$,
      the task the policy is evaluated on; no arm trains on it.
    - Returns `ContextTrainingSetup(suite, specs, distribution, deployment,
      num_envs, steps_per_round, total_rounds)`.

- **`train_env.py:150`** — `context_setup.wandb_config()` adds the run's facts
  to the W&B config. The dashboard's text panels are generated from these.
  - **`setup.py:89` `wandb_config`** [unchanged]. Ω, the task text,
    `training_distribution: "plr"`, the round count.

- **`train_env.py:156`** — `dashboard.ensure_view(…)` creates the group's W&B
  view from this run's facts, or refuses the run if the group's view was built
  from different facts. [unchanged]

- **`train_env.py:180`** — `get_algorithm_train_fn("ppo_lag")` picks the
  trainer.
  - **`training/run_utils.py:202`** [unchanged]. A name-to-function table;
    returns `training.agents.ppo_lag.train`.

- **`train_env.py:184`** — `context_setup.train_kwargs()` builds the four
  kwargs the trainer takes. One line inside it is the evaluation-stack fix.
  - **`setup.py:72` `train_kwargs`** [changed].
    - **`setup.py:80`** — `"wrap_env_fn": make_wrap_env_fn(distribution)`.
      - **`training/contexts/wrapper.py:332` `make_wrap_env_fn`** [changed].
        Returns a partial of `wrap_for_context_training`. Nothing is built yet;
        the trainer calls it in B.
    - **`setup.py:81`** — `"round_hook": ContextRoundHook(distribution,
      num_slots=num_envs)`.
      - **`training/contexts/round_hook.py:38` `ContextRoundHook.__init__`**
        [changed]. Stores the distribution and creates the `EpisodeTracker`
        for 8192 slots. That tracker is the hook's only state across rounds;
        φ is not kept here — it lives in the environment state and the hook
        reads it from there every round (D).
    - **`setup.py:83–84`** — `"evaluation_wrap_env_fns": {"deployment":
      make_evaluation_wrap_env_fn(deployment), "uniform":
      make_evaluation_wrap_env_fn(UniformDistribution(space))}`.
      **This is the fix.** Before this PR both entries called
      `make_wrap_env_fn`, so the evaluation environments were built with the
      training stack and inherited its episode-phase offset.
      - **`wrapper.py:344` `make_evaluation_wrap_env_fn`** [new]. Returns a
        partial of `wrap_for_context_evaluation` (B9b).
    - `"training_metrics_steps": steps_per_round` — `episodic/*` is reported
      once per round.

- **`train_env.py:184`** — `filter_kwargs_for_fn(train_fn_base,
  context_kwargs)` checks that the trainer's signature accepts all four. Line
  186 raises if not.
  - **`run_utils.py:226`** [unchanged]. An `inspect.signature` intersection.
    Today every PPO-family trainer except `ppo_lag` lacks `round_hook` and
    `evaluation_wrap_env_fns`; that is why `--alg ppo_pid` fails here, and
    what PPO-PID needs: two forwarding lines.

- **`train_env.py:219`** — `train_fn(environment=env, eval_env=eval_env,
  progress_fn=…)` starts training. Everything else in this PR happens inside
  this call.
  - **`training/agents/ppo_lag/train.py:26` `train`** [unchanged]. Adds
    PPO-Lagrange's constraint handling and forwards our two kwargs.
    - Lines 78–79 accept `round_hook` and `evaluation_wrap_env_fns`; lines
      228–229 pass them to `ppo_train.train`. PLR does not know which
      PPO-family wrapper it came through; any trainer with these two lines
      works.
    - **`training/agents/ppo/train.py` `train`**. Phases B–E are inside this
      function. Its own changes are listed in C.

### B. Environment construction

*The training environment (8192 slots) and one evaluation environment per
target (128 slots each) are built from the partials of A. Both wrapper changes
of this PR live here: φ stored once per run, and an evaluation stack without
the phase offset.*

- **`ppo/train.py:433`** — `_maybe_wrap_env(environment, …,
  wrap_env_fn=wrap_env_fn, …)` builds the training environment.
  - **`ppo/train.py:101` `_maybe_wrap_env`** [unchanged]. Line 136 takes our
    `wrap_env_fn` in place of Brax's `envs.training.wrap`; line 139 calls it
    with `(env, episode_length=, action_repeat=, randomization_fn=)`.
    - **`wrapper.py:297` `wrap_for_context_training`** [changed]. One line:
      `_wrap_with_contexts(…, spread_initial_phase=True)`.
      - **`wrapper.py:282` `_wrap_with_contexts`** [new]. Builds
        `ContextualAutoResetWrapper(EpisodeWrapper(VmapWrapper(env)),
        distribution, spread_initial_phase)` — Brax's stack with our wrapper
        in place of `AutoResetWrapper`.
        - **`crax/envs/wrappers/training.py` `VmapWrapper`, `EpisodeWrapper`**
          [unchanged]. **What the data is.** "The state" is one `State` object
          (`crax/envs/base.py:42`) with six fields: `pipeline_state` (the
          physics: positions, velocities, contacts), `obs`, `reward`, `done`,
          `metrics` (a dict) and `info` (a dict). The dicts hold arrays, or
          further dicts of arrays. This is every array in the state right
          after `reset`, built with 4 slots instead of 8192 so the shapes are
          readable (run on CPU, not inferred):

          ```
          obs                                      float32 (4, 62)
          reward, done                             float32 (4,)
          metrics['cost'], ['reward'], … (7)       float32 (4,)
          pipeline_state.qpos                      float32 (4, 3)
          pipeline_state.qvel                      float32 (4, 3)
          … 102 more physics arrays, all (4, …)
          info['context']                          float32 (4, 5)
          info['transition_context']               float32 (4, 5)
          info['slot_index']                       int32   (4,)
          info['context_rng']                      uint32  (4, 2)
          info['hazard_positions']                 float32 (4, 30, 3)
          info['steps'], ['truncation'], ['episode_done']   float32 (4,)
          info['episode_metrics']['cost'], … (9)   float32 (4,)
          info['cost'], ['step_count'], … (5 more env fields)  (4,)
          info['distribution_parameters'].round_index          int32   ()
          info['distribution_parameters'].contexts             float32 (1000, 5)
          info['distribution_parameters'].scores               float32 (1000,)
          info['distribution_parameters'].last_round           int32   (1000,)
          info['distribution_parameters'].occupied             bool    (1000,)
          info['distribution_parameters'].replay_weights       float32 (1000,)
          info['distribution_parameters'].replay_probability   float32 ()
          ```

          Every array has a first axis of length 4 — slot `i`'s data is row
          `i` of each — except the six under `distribution_parameters`: that
          is φ, one PLR buffer shared by all slots, with no slot axis at all.

          **What `VmapWrapper` does with it.** The environment's own `step`
          is written for a single slot: it expects `obs` of shape `(62,)`,
          `qpos` of shape `(3,)`. `VmapWrapper.step` is
          `jax.vmap(env.step)(state, action)`: run that single-slot `step`
          four times side by side, giving run `i` row `i` of every array in
          the state. For that to be well-defined every array must have the
          slot axis in front. `distribution_parameters.scores` has shape `(1000,)`
          and `round_index` has shape `()`; `vmap` would read the first as
          "1000 slots" and fail on the second. That is why
          `ContextualAutoResetWrapper.step` takes φ out of `info` before
          calling the inner `step` and puts it back after (B9a): nothing below
          our wrapper reads it, and it cannot pass through `vmap` unchanged.

          `EpisodeWrapper` owns `info['steps']`, `['truncation']`,
          `['episode_done']` and `['episode_metrics']`; the phase offset is
          written into its `steps` (B9b).

- **`ppo/train.py:449`** — `reset_fn = jax.jit(jax.vmap(env.reset))`; line
  456 calls it with keys of shape `[1, 8192, 2]`. The `vmap` is over the
  device axis, so every leaf of the returned state carries it.
  - **`wrapper.py:109` `ContextualAutoResetWrapper.reset`** [changed].
    - **`wrapper.py:116`** — `self.distribution.initialise()` makes φ_0. The
      protocol's `initialise` no longer takes a key: none of the four
      distributions used it (φ_0 is deterministic — an empty buffer, stage 0,
      a fixed context).
      - **`prioritized_level_replay.py:78` `initialise`** [new]. An empty
        buffer: `contexts` zeros `[1000, 5]`, `scores` zeros, `last_round`
        −1, `occupied` all False, `replay_weights` zeros,
        `replay_probability` 0. Every later φ has these shapes; that is what
        keeps the compiled program from retracing.
    - **`wrapper.py:116`** also — `self.reset_with_parameters(rng, φ_0)`.
      - **`wrapper.py:118` `reset_with_parameters`** [changed].
        - **Line 122** — `self.distribution.sample(parameters, key, 8192)` draws
          the first context of every slot.
          - **`prioritized_level_replay.py:91` `sample`** [new]. With
            `replay_probability` 0, every slot draws from Uniform(Ω).
        - **Line 123** — `_reset_in_contexts` vmaps the suite's
          `reset_with_context` over slots.
          - **`crax/envs/safe_goal.py:386` `reset_with_context`**
            [unchanged]. Reads the active counts and the goal radius from ω
            and lays out the arena. The environment does not know a
            distribution exists.
        - **Line 124 (B9b)** — `if self.spread_initial_phase:` overwrite
          `info["steps"]` with a random offset in [0, 1000). The condition is
          the change; before, this ran unconditionally. True for the training
          stack, so the 2026-10-06 desynchronisation is kept there.
        - **Line 132** — `state.info["slot_index"] = arange(8192)`. Each
          slot is labelled with its own index, once; the label never changes
          and is recorded with every transition, so the host can lay the
          round out per slot without knowing the trainer's row order (C).
        - **Line 133 (B9a)** — `state.info[DISTRIBUTION_PARAMETERS_KEY] = parameters`. φ is stored
          once. Before, `_broadcast_params(params, (8192,))` gave every slot a
          copy: nothing for staged's two scalars, about 300 MB per round for
          PLR's 36 KB buffer.

- **`ppo/train.py:868–881`** — for each of the two entries of
  `evaluation_wrap_env_fns`: `_maybe_wrap_env(raw_eval_env, …,
  num_eval_envs=128, device_count=1, wrap_env_fn=evaluation_wrap_env_fn)`,
  then `acting.Evaluator(eval_env, …)`. [unchanged]
  - **`wrapper.py:318` `wrap_for_context_evaluation`** [new]. One line:
    `_wrap_with_contexts(…, spread_initial_phase=False)`. The same stack as
    the training one; one flag differs. The module docstring states the
    invariant: *the training and evaluation stacks differ in the context
    distribution and the phase spread, and in nothing else.*
  - **`training/acting.py:108` `Evaluator.__init__`** [unchanged]. Wraps the
    environment in `EvalWrapper`, which sums `metrics` while an episode is
    active and reports `info['steps']` at its first `done` as the length.
    With a phase offset that counter read 1000 while about 540 steps ran —
    the defect's signature.

### C. One round

*Eighty steps per slot are collected; the reward advantage of every transition
is computed and shipped to the host together with the recorded fields; PPO
updates. The three trainer-side changes are here, all inside `if round_hook is
not None`; the stock path is untouched.*

- **`ppo/train.py:709` `training_epoch`** [unchanged since 2026-09-20].
  `jax.jit(jax.vmap(…))` of a scan of length 1 over `training_step`: with a
  round hook, one compiled call is one training step (lines 410–419).
  - **`ppo/train.py:609` `training_step`**.
    - **Line 620** — `acting.generate_unroll(env, state, policy, key,
      unroll_length=20, extra_fields=extra_fields)`, scanned four times
      (`batch_size × num_minibatches / num_envs`).
      - **`training/acting.py:57` `generate_unroll` → `:33` `actor_step`**
        [unchanged]. Line 46 records `nstate.info[x]` for every `x` in
        `extra_fields` into `data.extras['state_extras']`. `extra_fields`
        (train.py:419) is PPO-Lag's four plus the hook's three:
        `transition_context`, `episode_metrics`, `episode_done`.
        - **`wrapper.py:146` `ContextualAutoResetWrapper.step`** [changed, for
          B9a only]. Every environment step.
          - **Line 158** pops φ out of `state.info` before the inner vmapped
            step at 160; **line 195** puts it back. This is B9a's other half.
          - **Line 166** — `self.distribution.sample(parameters, key, 8192)` draws
            a candidate context for every slot; the fresh state replaces the
            stepped one where `done`.
            - **`prioritized_level_replay.py:91` `sample`** [new]. Per slot, a
              Bernoulli with `replay_probability`; a buffer row drawn by
              `categorical(log(replay_weights))` (line 99) or a fresh
              `space.sample_uniform`. Rows are copied bit for bit, so the host
              can match them to the buffer (D22).
          - **Line 193** — `info["transition_context"]` is set to the context
            the transition was generated in. It differs from `context` only on
            the step an episode ends. This is the field the trainer records.
    - **Line 656** — `running_statistics.update(…)`, the observation
      normaliser. [unchanged]
    - **Lines 662–673** [changed]. After the normaliser update, so the
      advantage uses the normaliser the loss will use.
      - **Line 666** — `rounds.LearningSignals(reward_advantage=
        ppo_losses.compute_reward_advantages(training_state.params,
        normalizer_params, data, ppo_network, discounting, reward_scaling,
        gae_lambda))`.
        - **`training/agents/ppo/losses.py:130` `compute_reward_advantages`**
          [new function]. The same `compute_gae` (line 73) the loss calls, over
          the whole `[B, T]` batch, returned instead of consumed. One
          value-network forward.
        - **`training/rounds.py:34` `LearningSignals`** [new type]. A flax
          struct with one field, `reward_advantage`, in the same `[rows, 20]`
          layout as the recorded fields.
      - **Line 673** — `jax.debug.callback(round_hook.observe,
        data.extras['state_extras'], learning_signals)`. Moved here from
        before the normaliser update; gained the second argument.
        - **`rounds.py:53` `RoundHook.observe`** [changed]. The protocol's
          signature now takes `learning_signals`.
        - **`round_hook.py:47` `ContextRoundHook.observe`** [changed]. This is
          the boundary where the trainer's string-keyed dict of recorded
          arrays becomes a typed value; nothing downstream sees the dict.
          Line 50 — `Transitions.from_recorded_fields(rollout,
          learning_signals, num_slots)`, stored for `on_round_end`.
          - **`training/contexts/rollout.py:50` `Transitions.from_recorded_fields`**
            [changed]. Every recorded field arrives as `[rows, 20]`, one row
            per (unroll, slot) in whatever order the trainer flattened them.
            Each row carries the slot it came from — `slot_index`, written
            by the wrapper at reset (B) and recorded like the context — so
            `_rows_by_slot` (line 139) groups rows by that label and
            `_per_slot` (line 156) gathers them into `[8192, 80]` in each
            slot's time order. The trainer's row order is not assumed
            anywhere; a missing field, a label that changes within a row, or
            labels that do not describe 8192 equally-sized slots **raise**
            with a message saying so. `__post_init__` (line 76) then checks
            the invariant *a slot's context changes only on the step after a
            done* as a second line of defence.
    - **Line 694** — `post_step_fn`, then the SGD scan. [unchanged] The one
      interaction with this PR: the advantage at 666 was computed under
      `training_state.params`, the parameters the first SGD minibatch starts
      from, so PLR's score is the value error PPO is about to see.

### D. Between rounds

*The round's recorded arrays become the episodes that ended, those are scored,
PLR's buffer is updated, and φ_{k+1} is written into the state the next
compiled call reads. This is where PLR learns.*

- **`ppo/train.py:940`** — `env_state, round_metrics =
  round_hook.on_round_end(epoch_index, env_state)`. [unchanged since
  2026-09-20]
  - **`round_hook.py:52` `on_round_end`** [changed]. Five statements, no
    branches: take the round's `Transitions` (typed in C), complete the
    episodes, read φ_k from the state, write φ_{k+1} into it, return the
    metrics.
    - **Line 56** — `self.episodes.complete(transitions)`.
      - **`training/contexts/episodes.py:26` `EpisodeTracker.__init__`**
        [new]. Created with the hook (A): two zero arrays per slot,
        Σ|advantage| and step count since the slot's last episode end.
      - **`episodes.py:31` `complete`** [new]. Per slot, a cumulative sum of
        |advantage| along the 80 steps (line 36). At each `episode_done` (line
        43) the segment since the previous done — or since the start of the
        round plus the carry — becomes one `CompletedEpisodes` row whose
        `value_loss` is the mean |advantage| over the whole episode, about
        1000 steps across about 12 rounds. The remainder after the last done
        is carried (line 58). Valid because the trainer never resets slots
        between rounds while a round hook is installed. This is the paper's
        Appendix C machinery for $T$-step rollouts.
    - **Line 56** also — `RoundRollout(round_index, transitions,
      completed_episodes)`.
      - **`rollout.py:131` `RoundRollout`** [changed]. The two views together.
    - **Line 58** — `parameters = current_parameters(env_state)`: φ_k, read
      back from the state the round just ran with. The hook keeps no copy of
      φ; the state is its one home. Before, the hook held a second copy that
      it seeded from the state on the first round and kept in step with it
      afterwards.
      - **`wrapper.py:272` `current_parameters`** [changed]. Strips
        `state.done.ndim − 1` leading axes — the device axis only. Before, it
        stripped device and slot axes.
    - **Line 59** — `self.distribution.update(parameters, rollout)` → φ_{k+1}.
      - **`prioritized_level_replay.py:105` `update`** [new].
        - **Line 109** — `_score_per_context(rollout.completed_episodes)`
          (line 167) groups the ended episodes by exact context with a
          byte-view key and averages their `value_loss` per context.
        - **Line 112** — contexts already in the buffer get the new score and
          `last_round = k`.
        - **Lines 114–116** — newcomers, highest score first: fill an empty
          row; once full, displace the row with the least replay mass if the
          newcomer scores higher; stop at the first rejection (`admit`, line
          216). A rejection leaves the buffer unchanged, so every later,
          lower-scoring candidate would be rejected too.
        - **Line 119** — `to_parameters` (line 235) recomputes `replay_weights`
          with `replay_weights` (line 140: rank^(−1/β) mixed with staleness by
          ρ) and `replay_probability` (p, or 0 while the buffer is empty).
          Same shapes as φ_0.
    - **Line 59** also — `attach_parameters(env_state, φ_{k+1})`.
      - **`wrapper.py:258` `attach_parameters`** [changed]. Broadcasts φ over
        the device axis only and writes it into `state.info[DISTRIBUTION_PARAMETERS_KEY]`. The
        next compiled call reads it at C's line 158. No recompile: 0 in 780
        rounds.
    - **Line 60** — returns the state and `training_curriculum_metrics(
      distribution, parameters, rollout)` with φ_k, the distribution the round
      was sampled from (E).
  - The `update(parameters, rollout)` signature is the protocol's.
    - **`training/contexts/distribution.py`** [changed]. `update` takes a
      `RoundRollout`. `log_probability` is removed: nothing called it, and
      PLR's q — point masses mixed with a density — has no log-density. The
      intended distribution is shown by sampling it instead (E). `Params` is
      renamed `DistributionParameters` (every `params` → `parameters`,
      `*Params` → `*Parameters`, `PARAMS_KEY` → `DISTRIBUTION_PARAMETERS_KEY`,
      the `info` key `"distribution_params"` → `"distribution_parameters"`):
      the trainer already uses `params` for network weights. Its docstring
      now states the contract a method's φ must meet — fixed shapes for the
      run (recompile), no slot axis (`vmap`, B9a) — where the next method's
      author will read it.
    - **`distributions/uniform.py`, `fixed.py`, `staged.py`** [changed,
      mechanically]. Follow the signature; `log_probability` gone; `summary`
      returns `{}`, `{}`, `{"stage"}`.

### E. Metrics leave

*Every round, the hook's metrics and the trainer's go through one callback to
W&B; at 21 points the evaluators run. The new keys, and the registry entries
that admit them, are here.*

- **`round_hook.py:60`** — `training_curriculum_metrics(distribution,
  parameters, rollout)`.
  - **`training/contexts/training_curriculum.py:54`** [changed].
    - **Line 70** — `intended/<d>`: `distribution.sample(φ_k, key, 4096)`,
      binned per dimension, with mean and std. $q_k$ shown by sampling it; for
      PLR the marginals of the replay mixture. New key.
    - `sampled/<d>` from `completed_episodes.contexts`; `experienced/<d>` from
      `transitions`. The same keys as before, from the new structures.
    - **Line 89** — `value_loss/<d>`: mean |advantage| of the round's
      transitions per bin. PLR's score as a function of the context, on every
      arm. New key.
    - **Line 75** — `distribution/<key>` from `distribution.summary(φ_k)`.
      Renamed from `intended/<key>`.
      - **`prioritized_level_replay.py:121` `summary`** [new]. Buffer
        occupancy, score mean and max, mean staleness, and `replay_mass/top_10`
        and `top_100` — the concentration of $P_{replay}$, which the
        per-dimension heatmaps cannot show.

- **`ppo/train.py:942`** — `progress_fn(current_step, {**training_metrics,
  **round_metrics})`. [unchanged]
  - **`run_utils.py:62` `wandb_progress_fn`** [unchanged].
    - **`training/dashboard/metrics.py:429` `select_for_logging`** [unchanged].
      Every key must match a kept `Metric` or a `Dropped` entry; anything else
      raises. Without the registry entries below, the first round would have
      raised — which is how the registry is meant to work.
      - **The `KEPT` table in the same file** [changed]. Entries for
        `intended/{dimension}` with `/mean` and `/std`,
        `value_loss/{dimension}`, `distribution/stage`,
        `distribution/replay_probability`, `score/mean`, `score/max`,
        `replay_mass/top_10`, `replay_mass/top_100`, `buffer_occupancy`,
        `staleness/mean`. `intended/context/*` and `intended/stage` retired.
    - `wandb.log(payload, step=environment_steps)`.

- **`ppo/train.py:890` `run_evaluations`** [unchanged]. At each of 21
  points, each evaluator from B runs.
  - **`acting.py:126` `run_evaluation`** [unchanged]. `eval_env.reset` with
    128 keys, then one unroll of 1000 steps; `EvalWrapper`'s sums, prefixed
    `evaluation/<name>/`. Because of B9b these episodes start at step 0 and
    the sums cover whole episodes.

- **`training/dashboard/view.py`** [changed]. One sentence in the Mechanism
  preamble names the two new heatmaps.

---

## Per-file detail

### `training/contexts/distributions/prioritized_level_replay.py` [new]

- `PrioritizedLevelReplayParameters` is φ: `round_index`, `contexts [M, D]`,
  `scores [M]`, `last_round [M]`, `occupied [M]`, `replay_weights [M]`,
  `replay_probability`. The last two are derived by `update` so that `sample`
  is a table lookup.
- `PrioritizedLevelReplay(space, buffer_size, replay_probability, temperature,
  staleness_coefficient)` is M, p, β, ρ. p is the paper's constant $P_D$
  (Appendix B.3); it is 0 only while the buffer is empty. β = 1 is the design
  doc's argued default; 0.3 is the paper-faithful candidate, to be decided by
  `replay_mass/*` on a run.
- `sample`: `log(0) = −inf` gives unoccupied rows zero probability; with an
  empty buffer the Bernoulli never selects the row branch.
- `update`: `_score_per_context` groups 655 episodes by context in 0.2 s with a
  byte-view key (`np.unique(axis=0)` took 1.6 s on 2.6 M rows).
- `replay_weights`: raises if a row's score is dated in the future.
- `_Buffer`: the mutable host copy `update` edits; raises if two rows hold the
  same context, which is the end-to-end guard on exact matching between device
  and host.

### `training/contexts/episodes.py` [new]

`EpisodeTracker`, as described in D. The only state the round hook keeps
besides φ.

### `training/contexts/rollout.py`

`Transitions` (C), `CompletedEpisodes`, `RoundRollout` (D). The per-slot layout
is recovered from the recorded `slot_index` label, not from the trainer's row
order; the only assumption left is that one slot's rows arrive in the order the
trainer's scan produced them. `TRANSITION_CONTEXT_KEY` and `SLOT_INDEX_KEY` live
here: they name recorded fields, and leaving them in the wrapper would make a
cycle wrapper → distribution → rollout → wrapper now that `distribution.py`
imports `rollout.py`.
`EpisodeFeedback` and `completed_episodes` are gone; `CompletedEpisodes` is the
same rows with the score added and a consumer.

### `training/contexts/round_hook.py`

As described in A, C and D. The hook owns one thing across rounds, the
`EpisodeTracker`; φ is read from and written to the environment state every
round. The constructor takes `num_slots` for the tracker, so there is no
first-round branch. The trainer's recorded dict is turned into typed
`Transitions` in `observe`, the boundary; `on_round_end` never sees it.

### `training/contexts/wrapper.py`

B9a and B9b as described. `_broadcast_params`, `_unbroadcast_params` and the
`Tuple` import are removed; `_device_axes` (line 254) is the one place that
knows how many leading axes the trainer adds. `reset_with_parameters` writes
`slot_index` (line 132), a per-slot label the host uses to lay the recorded
round out per slot.

### `training/contexts/distribution.py`, `distributions/{uniform,fixed,staged}.py`

As described in D.

### `training/contexts/setup.py`, `training/config.py`, `training/train_env.py`, `training/contexts/__init__.py`, `distributions/__init__.py`

As described in A. `context_training_setup` takes `num_envs` and
`train_env.py` passes it (one line each). Exports: added `PrioritizedLevelReplay`, `Transitions`,
`CompletedEpisodes`, `EpisodeTracker`, `make_evaluation_wrap_env_fn`,
`wrap_for_context_evaluation`; removed `EpisodeFeedback`,
`completed_episodes`, `uniform_log_density`.

### `training/rounds.py`, `training/agents/ppo/losses.py`, `training/agents/ppo/train.py`

As described in C. Nothing outside `if round_hook is not None` changed in the
trainer.

### `training/contexts/training_curriculum.py`, `training/dashboard/metrics.py`, `training/dashboard/view.py`

As described in E. In `training_curriculum.py`, `_mean_episode_length_per_bin`
became `_mean_per_bin(contexts, quantity, edges)` (line 115), used for both
episode length and value loss. `metrics.py` imports `NUM_INTENDED_DRAWS` so
the panel text states the number the code uses; this is a new import edge from
the dashboard into the contexts package.

## Tests

### `tests/test_contexts.py`

- `round_rollout(…)`: a round in which every slot's episode ends on the last
  step, so each context has one completed episode with a known score.
- `test_transitions_recover_each_slots_time_series_from_the_trainers_layout`:
  the `u × N + slot` row layout comes back in per-slot time order; a
  non-multiple row count and a mid-episode context change raise.
- `test_episode_tracker_scores_the_whole_episode_across_rounds`: an episode
  spanning two rounds is scored over all six of its steps; two episodes ending
  in one slot in one round are both closed; a slot-count mismatch raises.
- `test_evaluation_episodes_start_at_step_zero_and_are_summed_whole`: on the
  evaluation stack the Evaluator's reported length equals the steps each
  episode actually ran from step 0; on the training stack through the same
  Evaluator the reported length exceeds the steps run. This is the defect,
  pinned.
- Five PLR tests: first round uniform then the top eight kept; replay weights
  are rank mixed with staleness, a cold β concentrates, a future-dated score
  raises; `sample` reproduces `replay_weights` with exact rows; rescoring
  overwrites, a better newcomer displaces the least-replayed row, a worse one
  is rejected, stale rows are protected; hyperparameter ranges.
- Removed: the log-density assertions and `test_log_probability`.

### `tests/test_context_training.py`

`_synthetic_round` is two slots by five steps with one episode ending per
slot, built through `Transitions` and `EpisodeTracker`. The metrics test
covers `intended` as a stage-0 point mass, `distribution/stage`, and
`value_loss` per bin. `plr` is added to the parametrised end-to-end test. The
staged test reads `intended/<d>/mean`. `test_end_to_end_plr_…`: buffer rows
never exceed the episodes completed so far; `replay_probability` is the
constant p from round 1; scores are finite; `value_loss > 0` implies
`experienced > 0`; `intended` is never a point mass; the final φ lies in Ω.

### `tests/test_dashboard.py`, `tests/test_suites.py`

The key list and the merged-panel assertion follow the new keys.
`_logged_keys` builds its Evaluator on the evaluation stack.

## Documents and scripts

- `docs/acl/design/prioritized_level_replay.md`: the design.
- `docs/acl/design/contexts_package.md`: file list, `observe` signature, CLI
  grammar, metric table, φ storage, the two stacks and their invariant.
- `docs/acl/README.md`: plan item 1 status; the 500 M result under *Results
  so far*; a *Known defects* entry for the evaluation-stack defect and its
  effect on the 2026-10-06 numbers.
- `docs/acl/experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md` and
  `figures/2026-10-08_goal_point_500M/`.
- `scripts/analysis/pull_group_history.py` (a W&B group to one parquet per
  arm) and `scripts/analysis/goal_point_500M_figures.py` (the three figures).
- `AGENTS.md`: the pull-request and experiment write-up formats.

## Open, not in this change

- Multiple evaluation targets, for the OOD space: the trainer takes any
  `{name: wrap_env_fn}`; `ContextTrainingSetup.train_kwargs` and
  `dashboard/view.py` `RunFacts.evaluations` hard-code `deployment` and
  `uniform`.
- PPO-PID: `ppo_pid/train.py` needs `round_hook` and
  `evaluation_wrap_env_fns` forwarded, as `ppo_lag/train.py:78–79` and
  `228–229` do.
- PLR score variants (reward, cost, Lagrangian critic): one more field in
  `LearningSignals`, three columns in `CompletedEpisodes`, a `score` choice on
  the class, `plr:cost` and `plr:lagrangian` specs.
- Checkpointing φ and the tracker's carries; PLR hyperparameters on the CLI;
  `MetricsLogger` duplicating the round hook.
- φ travels in `env_state.info` because the compiled round's only
  per-round input is `env_state`, so the wrapper must lift it out before the
  vmapped stack (B9a). The by-construction fix is a per-round-parameters
  argument on the compiled round itself; that changes `ppo/train.py`'s
  `training_step`/`generate_unroll` signatures — upstream, to raise with
  Tristan.
- `ContextualAutoResetWrapper.reset` bypasses the inner stack (vmaps the
  suite's `reset_with_context` itself and rebuilds `EpisodeWrapper`'s fields in
  `_add_episode_fields`), while `step` goes through it. Pre-existing; the two
  relationships with the stack below should become one.
- The PLR end-to-end test does not count recompiles; the staged one does
  (`test_context_training.py:253`). PLR's φ is the one with real shape risk.
