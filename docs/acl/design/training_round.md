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

## The round's size is a learning quantity for everything built on the round

Added 2026-10-18, after the vision runs. The round's size in transitions is
`batch_size × num_minibatches × unroll_length`; the number of robots only sets
how many collections of `unroll_length` steps fill it (`batch_size ×
num_minibatches / num_envs`, four at both of our operating points). The three
flags and what each governs:

| flag | governs | sensors runs | vision runs (2026-10-17, -18) |
|---|---|---|---|
| `batch_size` | trajectories per gradient step | 1024 (20 480 transitions) | 1024 |
| `num_minibatches` | pieces the round is cut into = gradient steps per pass, **and** the round's size | 32 → 655 360 transitions | **4 → 81 920** |
| `num_updates_per_batch` | passes (shuffles) over the round; every transition is used this many times | 4 | 4 |
| `num_envs` | robots in parallel; throughput, and contexts assigned per round | 8192 | 1024 (the renderer's memory) |

**What the gradient steps see is set by `batch_size` and the passes, and was
equal in all our runs**: 20 480 transitions per step, four uses per transition,
~97 700 gradient steps per 500 M environment steps. Shrinking `num_minibatches`
from 32 to 4 changed none of that. It changed the round, and **the round is the
unit this project's method computes in**:

| computed once per round, over the round's pool | 655 360-transition round | 81 920-transition round |
|---|---|---|
| the constraint's input, the round's mean cost | ~655 episodes' worth | ~82 episodes' worth: √8 ≈ 2.8× the sampling noise |
| the integrator's step `rate × (cost − budget)`, once per round | 763 steps per 500 M | 6104 steps per 500 M: 8× the drift at the same `rate` |
| the additive λ(ω) fit: 300 Adam steps at a fixed learning rate, once per round | — | 8× the movement, whatever `rate` says |
| the advantage normalisation's mean and standard deviation | over 655 k | over 82 k |
| PLR's scores and the realised curriculum q̂ | from ~655 episodes | from ~82 episodes |
| contexts assigned at the round's start | 8192 | 1024 |
| the on-policy cycle: the policy that collects is refreshed every … | 655 k steps | 82 k steps (shorter: fine or better for PPO itself) |

What the runs showed: at 4 minibatches the pixel policy's level-3 return
moves 2.5–5 goals between evaluations 25 M apart against 0.8–1.4 with sensors
and a sampling error of 0.2–0.4 (`experiments/2026-10-17_…`); matching the
integrator's drift per step (rate ⅛) quieted λ and left the swings
(`experiments/2026-10-18_…`). The round's size and the camera are still
confounded; the two experiments that separate them are the sensors at the
small round, and the pixels at the large one.

**The rule this leaves**: when `num_envs` has to change, hold the round's size
(`batch_size × num_minibatches`) and let the number of collections per round
absorb it; a flag of the published method must keep its meaning per
environment step, not per round. Where the large round does not fit in memory
the compromise is explicit and recorded in the experiment, not hidden in a flag.

A memory fact that bounds the pixel runs: the trainer holds the round's
observations and next observations (64 × 64 × 9 bytes each; 6 GB per 81 920
transitions), and the DrQ augmentation (`sgd_step`) translates the **whole
round's** data once per pass before cutting it into minibatches, so the frames
are held twice. A 655 360-transition round of pixels is ~96 GB before the
networks' activations: it does not fit an 80 GB H100 as the trainer stands.
Translating per minibatch instead (inside `minibatch_step`, the same
distribution of shifts) halves that; a 327 680-transition round (16
minibatches) then fits an H100.

## Implied change to `training/agents/ppo/train.py`

`num_training_steps_per_epoch` becomes 1 (or the scan is removed), the outer
loop runs one training step per iteration, and evaluation triggers every
`N` iterations from `--num_evals`. The context sampler's host-side `update`
runs in that loop body. Not yet implemented.
