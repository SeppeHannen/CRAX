# The training round: epoch = training step

Decision (2026-09-20): **one compiled call = one PPO training step**, and
distribution updates happen between calls. This document records why, and a
nuance about what a curriculum distribution actually controls.

## Terms

- **Training step** — one PPO collect-then-learn cycle: `unrolls × unroll_length
  × num_envs` transitions (16 × 20 × 8192 ≈ 2.6 M at our operating point),
  followed by `num_updates_per_batch × num_minibatches` SGD steps. The natural
  unit of on-policy RL; the thesis' round $k$.
- **Epoch** (CRAX/Brax term) — a batch of training steps executed as *one*
  compiled JAX call, `lax.scan(training_step, length=num_training_steps_per_epoch)`.
  Its length is not a learning quantity: it is whatever makes evaluations land
  where `--num_evals` asks (`num_timesteps / (num_evals - 1) / steps_per_training_step`).
  In the paper's 500 M-step, 5-eval config an epoch is ~48 training steps.

Inside a compiled call nothing on the host can act. A context distribution can
therefore only be sampled from **parameters that were handed in before the
call** (a pytree of arrays, $\phi$). It is *frozen* for the whole call and can be
*updated* only between calls.

## Why epoch = training step

The distribution $q_{m,k}$ in the thesis is indexed by round $k$ = training
step. Matching the compiled-call boundary to the training step means:

1. $\phi_k$ is exactly the distribution in force during round $k$ — no
   averaging over an arbitrary number of steps chosen by the eval schedule.
2. Feedback from round $k$ (completed episodes, their contexts and outcomes)
   is available on the host immediately after, so $\phi_{k+1}$ is computed from
   it. This is the update rule as written.
3. The complete state $X_{m,k} = (\Theta_{m,k}, \Phi_{m,k})$ exists as ordinary
   host-visible arrays at every round boundary — checkpointing and branching
   (RQ4) need nothing special.
4. The performance tracker times every training step, not an aggregate.

Cost: one program launch and a handful of host→device copies per training step,
≈ 1–5 ms against ≈ 2.6 s of compute. Negligible. Evaluation is decoupled from
the call structure: `--num_evals` decides how often to evaluate, not how many
steps a compiled call contains.

The alternative — long epochs with `jax.debug.callback` for host-side updates
inside the scan — saves nothing (the callback is a synchronous round trip too)
and puts mutable host state inside a compiled function.

## Nuance: what $q_{m,k}$ controls, and what it does not

Reset-on-done runs *inside* the program: when a slot's episode ends, the next
context is drawn from $\phi_k$ and a fresh episode starts in it, on that same
step. So $q_{m,k}$ controls **which contexts start** during round $k$.

But episodes are longer than a round (≈ 1000 steps vs 320 per round at 8192
envs). An episode started in round $k$ contributes transitions to rounds $k$
through roughly $k+3$. Consequently the PPO batch in round $k$ contains
transitions from contexts drawn under $q_{m,k-3}, \ldots, q_{m,k}$.

Two consequences for the thesis:

- **Update lag.** The feedback used to compute $\phi_{k+1}$ describes contexts
  chosen several rounds earlier. Inherent to any online curriculum whose
  episodes outlast a round; worth one sentence.
- **$\hat q \ne q$ even within a round**, which is why the realised curriculum
  is defined exposure-weighted. Two effects: (a) episodes of different lengths
  contribute different numbers of transitions — a context whose episodes
  terminate early is under-represented relative to its selection frequency;
  (b) the window mixes several rounds' selections. The transition-weighted
  $\hat q_{m,\ell}$ captures both; selection counts capture neither. It is the
  honest description of what the student was trained on, comparable across
  methods, and independent of the optimiser (which is why gradient-magnitude
  weighting is deliberately excluded — that would measure influence, not
  exposure).

Implementation: every transition carries its context in `state.info["context"]`,
so it flows through `extra_fields` into the rollout data; $\hat q_{m,\ell}$ is a
histogram over `data.extras["state_extras"]["context"]`, computed on the host
each round at no cost.

## Implied change to `training/agents/ppo/train.py`

`num_training_steps_per_epoch` becomes 1 (or the scan is removed), the outer
loop runs one training step per iteration, and evaluation triggers every
`N` iterations from `--num_evals`. The context sampler's host-side `update`
runs in that loop body. Not yet implemented.
