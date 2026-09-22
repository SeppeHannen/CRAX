# Staged vs uniform on the other five velocity agents

Status: CPU checks done and fixed (2026-09-22); GPU runs not started.

Task board item 4, first batch: repeat the `safe_velocity_ant` experiment
(`2026-09-20_uniform_vs_staged_velocity_ant.md`) on `halfcheetah`, `hopper`,
`humanoid`, `swimmer`, `walker2d`. Nothing in the contexts package is
agent-specific — the registry already covers all six — so what this batch tests
is whether the things we built on one agent hold on the other five: the wrapper
and round hook, the metric registry, the generated dashboard text, and the
staged-vs-uniform result itself.

## What the CPU check found before any GPU time

Everything below came out of running each suite through the training wrapper
stack and the `Evaluator` on CPU (`tests/test_suites.py`, new); none of it
needed a training run.

1. **`get_environment("safe_velocity_*", level=n)` ignored the level.** The
   `_ENV_TO_TASK` map in `crax/envs/difficulty.py` lost its velocity entries in
   upstream commit `583e7fd` ("New naming conventions", 2026-05-06): it dropped
   the old-style `"safe_velocity": "velocity"` key but never added the six
   `safe_velocity_<agent>` names. Since then `--difficulty 2` or `3` on any
   velocity suite printed a "no task mapping" warning and trained on level 1
   (the constructor default), for every algorithm and every script that goes
   through `get_environment`. Our context runs were unaffected (`level:n` goes
   through the registry, not the map), which is why the 50 M / 500 M / 750 M
   experiments are fine; but *stock* CRAX velocity results at level 2 or 3 since
   May are level-1 results. **Flag to Tristan** — this touches the paper's
   velocity baselines. Fixed by restoring the six entries; the test
   `test_levels_reproduce_the_difficulty_module` pins it.

2. **`safe_velocity_swimmer` could not step at all** under the installed
   `jax 0.10.1`: `crax/fluid.py` called `jp.clip(..., a_min=...)`, removed from
   JAX's API (the pinned `requirements.txt` says `jax==0.6.0`, where it still
   existed). Only the swimmer's generalized pipeline reaches the fluid model.
   Fixed (`min=`). Flag to Tristan as a version-compatibility note.

3. **Five agents log reward terms the dashboard did not know.** Brax kept each
   MuJoCo agent's own names for the parts of its reward (`reward_run`,
   `reward_healthy`, `reward_alive`, `reward_linvel`, `reward_fwd`,
   `reward_quadctrl`, …). As designed, a run on any of these would have raised at
   its first log. First instinct was to register each name with a description
   saying which agent logs it; that would have put agent facts into
   `metrics.py`, which is meant to know nothing about agents (commit `381e7fd`).
   Asked instead what the panel is *for*: the question "is the return from
   moving or from surviving?" is answered by `velocity_value ÷ length` (the
   suite's own, agent-neutral speed) and the upright bonus is a constant ×
   episode length, already shown. So the agents' own reward terms are dropped
   under one pattern (`episodic/reward_{reward_term}`, one reason); only
   `forward_reward` is kept, because `results/common.py` reads it for the paper's
   velocity figures. `metrics.py` names no agent.

4. **The task description and two length descriptions were ant-shaped.** They
   said every agent "falls" and "a drop to a few steps = fell"; the halfcheetah
   and swimmer cannot fall. The one agent-dependent fact — can this agent fall —
   now lives in the suite's `TaskDescription` alone; `metrics.py` says "unless
   the environment ends it early (the Environment definition says whether this
   agent can fall)". `test_task_description_says_whether_the_agent_can_fall`
   drives each agent with random torques and checks the text against whether
   `done` ever fired before the step limit.

## Setup

Identical to the ant experiment except for the agent: PPO-Lagrange, 8192 envs,
1 000-step episodes, cost budget 25, 500 M steps, one seed, deployment
`level:3`, two arms (`staged:1,2,3`, `uniform`). One W&B group per agent so each
gets its own generated view (the view stores the suite's task text, which
differs per agent).

Ω per agent (`velocity_threshold`, baseline × [0.4, 1.0]; L1/L2/L3 = 1/0.75/0.5 × baseline):

| agent | baseline | L3 | Ω | falls? |
|---|---|---|---|---|
| halfcheetah | 3.210 | 1.605 | [1.284, 3.210] | no |
| hopper | 0.740 | 0.370 | [0.296, 0.740] | yes |
| humanoid | 1.415 | 0.707 | [0.566, 1.415] | yes |
| swimmer | 0.228 | 0.114 | [0.091, 0.228] | no |
| walker2d | 2.341 | 1.171 | [0.937, 2.341] | yes |

```bash
cd ~/tue/graduation/CRAX
for agent in halfcheetah hopper humanoid swimmer walker2d; do
  COMMON="--env_name safe_velocity_$agent --alg ppo_lag --seeds 0 \
    --num_envs 8192 --num_timesteps 500_000_000 --num_evals 21 --episode_length 1000 \
    --unroll_length 20 --batch_size 1024 --num_minibatches 32 --num_updates_per_batch 4 \
    --safety_bound 25 --deployment_distribution level:3 \
    --store_model false --skip_rollout --skip_video --measure_performance \
    --wandb_group velocity_${agent}_staged_vs_uniform_500M"
  .venv/bin/python -m training.train_env $COMMON --context_distribution staged:1,2,3
  .venv/bin/python -m training.train_env $COMMON --context_distribution uniform
done
```

Budget: the ant took 28 min per 100 M at 371 k SPS; ten runs of 500 M is on the
order of 24 GPU-hours if the other agents are as fast (the humanoid's larger
model will be slower). A 50 M pilot per agent first (≈ 5 min each, `--num_evals 9`)
catches anything the CPU check cannot — throughput, compiles, the generated view
on a real group — for under an hour.

## What to check, per agent

1. `performance/epoch_compiles` is 0 after round 0 (context wrapper and round
   hook run without a recompile) and SPS is steady.
2. The run does not raise at its first log (every key registered — now
   guaranteed by `tests/test_suites.py`, but the GPU run is the proof).
3. The group's view text is right for the agent: the Environment / Reward / Cost
   block, the "cannot fall" sentence for halfcheetah and swimmer, the Ω bounds.
4. Verdict: does uniform solve level 3 and does staged collapse at the first
   switch, as on the ant? The two agents that cannot fall are the interesting
   ones: on the ant the staged run's collapse was "stop moving or fall"; the
   halfcheetah and swimmer cannot fall, so the multiplier has one fewer way to
   zero the cost.

## Results

*(none yet)*
