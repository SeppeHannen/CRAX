# Automated curriculum learning on CRAX

Giuseppe Hannen's graduation project. Start here.

## The plan

1. **Profile cleanly.** XProf traces and per-round throughput go to W&B for
   every run, so we always know whether training is still fast. Done:
   `training/performance/`, guide in `performance_measurement.md`.
2. **Manual curriculum vs uniform.** On `safe_velocity_ant`, PPO-Lagrange, level
   1 → 2 → 3 vs contexts drawn uniformly from Ω, compared on level 3. Done, one
   seed at 500 M steps: uniform solves level 3, the curriculum collapses when the
   level switches (Lagrange multiplier blow-up) and never recovers. Write-up:
   `experiments/2026-09-20_uniform_vs_staged_velocity_ant.md`.
3. **A W&B dashboard one can read.** Done and in use. `design/dashboard.md`
   derives it from what reviewing a run is (Verdict / Mechanism / Trust /
   Detail); `training/dashboard/` implements it: a metric registry every
   logged key must pass (39 kept, each with a one-sentence description of
   where its numbers come from; 47 dropped with a reason; anything else
   raises), budgets logged next to costs, and a W&B view per experiment group
   whose sections open with generated text — the environment (agent, reward,
   cost, Ω) in the suite's own words, then cadence, population and one line
   per panel. A run with `--wandb_group` creates the group's view at start-up
   and is **refused** if its flags differ from the group's (`FactsMismatch`).
   First experiment on it: group `velocity_ant_staged_vs_uniform_750M`
   (uniform vs staged:1,2,3, one seed, 750 M steps) — to be read with Tristan
   and written up under `experiments/`.
4. **Let the learner know its context.** Today the policy sees only the state;
   the threshold enters only the cost, so it can learn one behaviour for all of
   Ω and nothing else. Two ways to change that, both change the benchmark's
   observation space (agree with Tristan first):
   - **told**: ω in the observation — a contextual CMDP, the oracle;
   - **must infer**: the previous step's cost (and reward) in the observation,
     with or without memory (frame stack; recurrent policy). Measure the gap to
     the oracle before building memory: it is the value of inferring ω.

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
5. **Repeat 2 with PPO-Saute.** It augments the state with the remaining safety
   budget and enforces the constraint per episode rather than in expectation,
   and has no dual variable that carries the old distribution across a switch.
6. **Then** actual curriculum methods, and more suites.

## Vocabulary

| term | meaning |
|---|---|
| context ω ∈ Ω | the per-episode task parameters of a suite (Ant: the velocity threshold). A value on the `num_envs` axis, never a shape. |
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

CPU tests: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_contexts.py tests/test_context_training.py tests/test_dashboard.py -q -p no:cacheprovider`

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
    per_slot_constraints.md          what may differ between parallel environments
    context_spaces_by_suite.md       Ω for each of the nine suites
    dashboard.md                     what a reviewer needs from W&B, key inventory, the eight primary panels
  profiling_options.md               tool survey (historical)
```
