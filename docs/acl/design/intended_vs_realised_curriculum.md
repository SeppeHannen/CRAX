# Intended vs realised curriculum: $q_{m,k}$ is not what the student trained on

**Critical for interpreting every result.** The distribution a method holds at
round $k$ and the data the student's optimiser actually saw in round $k$ are
different objects. Conflating them credits (or blames) a method for a
curriculum the student never experienced.

## Definitions

- $q_{m,k}$ — **intended** curriculum. The distribution method $m$ *samples
  from* at round $k$ whenever a slot needs a new context. In code: the frozen
  parameters $\phi_k$ handed to the compiled training step.
- $\hat q_{m,\ell}$ — **realised** curriculum. The empirical distribution over
  contexts of the PPO training transitions in window $\ell$: a context's weight
  is the fraction of transitions in the window that were collected in it.
  Measured, not chosen.

$q$ is per *episode start*. $\hat q$ is per *transition*. Everything below
follows from that.

## Why they differ

### 1. Episode length depends on the context

A context is selected once per episode but trained on once per transition. If
episodes in context A last 1000 steps and episodes in context B terminate at
step 200 (fall, hazard budget exhausted, early success), then equal selection
$q(A) = q(B) = 0.5$ yields exposure
$\hat q(A) : \hat q(B) = 1000 : 200$, i.e. $0.83 : 0.17$.

This is the dominant effect in a **safety** benchmark, where terminations and
constraint behaviour are exactly what varies with difficulty. It has three
consequences:

- The uniform baseline $r = \mathrm{Uniform}(\Omega)$ does **not** produce
  uniform exposure. Its $\hat q$ is skewed toward contexts with long episodes.
  This is a finding in itself, and it means "domain randomisation" is never
  the neutral reference it appears to be.
- Two methods with identical $q$ can have different $\hat q$ if their students
  behave differently (a safer student survives longer in hard contexts and is
  therefore exposed to them more).
- The direction of the skew depends on the task. Where hard = short episodes
  (falls), hard contexts are under-exposed. Where hard = long episodes (goal
  never reached, no early termination), hard contexts are over-exposed.

### 2. Episodes straddle rounds

With ≈ 1000-step episodes and 320-step rounds (8192 envs, 16 unrolls × 20
steps), an episode started in round $k$ contributes transitions to rounds $k$
through roughly $k+3$. Round $k$'s PPO batch therefore mixes contexts chosen
under $q_{m,k-3}, \ldots, q_{m,k}$, weighted by how much of each episode fell
in the window.

Consequences:

- **Lag.** A sharp change in $q$ at round $k$ appears in $\hat q$ gradually
  over the next ~4 rounds. A staged curriculum (level 1 → 2 → 3) does not
  switch the training data at the stage boundary; it blends over it.
- **Update lag.** The feedback used to compute $\phi_{k+1}$ comes from episodes
  that *completed* in round $k$, i.e. were mostly *started* under
  $\phi_{k-3}, \ldots, \phi_{k-1}$. Every online curriculum whose episodes
  outlast a round has this; it is not specific to our implementation.

## What to report, and how

| Question | Object |
|---|---|
| What did method $m$ *decide*? | $q_{m,k}$ — plot its parameters or a summary over $k$ |
| What was the student *trained on*? | $\hat q_{m,\ell}$ — the curriculum as delivered |
| Why did method $m$ outperform $m'$? | compare $\hat q$; identical $q$ with different $\hat q$ is possible and informative |
| Did the baseline give equal exposure? | $\hat q_{r,\ell}$ against uniform — it will not; report the deviation |
| Attribution of gradient to contexts (RQ4) | separate analysis; **not** folded into $\hat q$ |

Always show $\hat q$ alongside $q$ for any curriculum figure. Never present a
$q$ trajectory as "the curriculum the student received".

### Why $\hat q$ is exposure-weighted and *not* gradient-weighted

$\hat q$ describes the **data**: what fraction of the student's training
transitions came from where. That is a property of the environment and the
student's behaviour, comparable across methods, seeds and algorithms, and
independent of network state. Weighting by gradient magnitude would measure
**influence** — how much each context moved the parameters — which depends on
the optimiser state, advantage normalisation, the Lagrange multiplier, and the
current policy. That is a different, legitimate question, handled by the
attribution analysis in RQ4. Mixing the two into one quantity would make $\hat
q$ uninterpretable as a description of the curriculum.

## Implementation

Every transition carries its context: `state.info["context"]` is a
`[num_envs, D]` array, set at reset, and flows through `extra_fields` into
`data.extras["state_extras"]["context"]` in the rollout. $\hat q_{m,\ell}$ is a
histogram (or kernel density, for continuous $\Omega$) over that array on the
host, once per round, at no measurable cost. Log to W&B per round as a
distribution plot plus low-dimensional summaries (mean context, entropy,
distance to $q$).

Both $q$ (via $\phi_k$) and $\hat q$ are part of the run record. $\phi_k$ is
in the checkpointed state $X_{m,k}$; $\hat q$ is derived from logged data.

## For the thesis

This belongs in the preliminaries next to the definition of the realised
curriculum, with the two mechanisms and the "always report both" rule stated
explicitly. It is also the reason the uniform reference $r$ should be
introduced as "uniform *selection*", never "uniform *training*".
