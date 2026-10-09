# Prioritized Level Replay as a `ContextDistribution`

Plan item 1 (`README.md`). PLR (Jiang, Grefenstette, Rocktäschel, ICML 2021,
arXiv 2010.03934) is the first automated curriculum method on CRAX. This
document says what PLR is, what has to change for it to act on a continuous Ω
inside our one-call-per-round trainer, and which of its choices we keep, change
or defer — so the implementation (`training/contexts/distributions/
prioritized_level_replay.py`) can be read against a stated design rather than
against the paper.

## What PLR is

The paper's setting is a procedurally generated game: a *level* $l$ is a seed
that fixes everything about one episode's world, and training normally draws
each episode's level uniformly from a set $\Lambda_{train}$. PLR replaces that
uniform draw. Its premise is that a level is worth revisiting while the
learner's value function is still *wrong* on it — a level the agent has
mastered, and a level it cannot do anything on yet, both produce stable
returns and a small value error; the levels at the edge of its ability produce
a large one.

Notation, all of it the paper's:

| symbol | meaning |
|---|---|
| $l_i$ | the $i$-th level the learner has played |
| $\Lambda_{seen}$ | the levels played at least once — the ones that *have* a score and a timestamp. It exists because $P_{replay}$ is a distribution over scores: it is the domain of $P_{replay}$, nothing more. In our implementation this is the occupied rows of the buffer. |
| $S_i$ | the **score** of $l_i$: how much learning potential it is estimated to have, recomputed from the latest episode played on it |
| $c$ | the number of episodes played so far, in total |
| $C_i$ | the value of $c$ when $l_i$ was last played — its **timestamp** |
| $\gamma, \lambda$ | PPO's discount and GAE parameter, reused as is |
| $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | the one-step TD error at step $t$ of an episode |
| $\beta$ | **temperature**: how sharply scores are turned into probabilities |
| $\rho$ | **staleness coefficient**: how much weight goes to levels not played recently |
| $P_D$ | the probability of *replaying* a seen level instead of trying a new one |

**The score.** After an episode of $T$ steps on $l_i$,

$$S_i = \frac{1}{T}\sum_{t=0}^{T}\Big|\sum_{k=t}^{T}(\gamma\lambda)^{k-t}\delta_k\Big|.$$

The inner sum is the generalised advantage estimate (GAE) $\hat A_t$ at step
$t$ — the same quantity PPO uses as its advantage, before normalisation. It is
computed per transition, in trajectory order: $\delta_k$ needs the value
estimates of the states actually visited, and $\hat A_t$ sums the $\delta$'s
that followed step $t$ in that trajectory, so it depends on the rewards and
states that came *after* $s_t$. PPO's regression target for the value function
is $\hat V_t = V(s_t) + \hat A_t$ (`compute_gae` returns both), hence
$\hat A_t = \hat V_t - V(s_t)$ and $S_i$ is the mean absolute error of the value
function along the episode: the paper calls it the **L1 value loss**. Every
time $l_i$ is played again, $S_i$ is overwritten with the new episode's value
(not averaged).

**The replay distribution.** When an episode needs a level, PLR first decides
whether to replay — a Bernoulli with probability $P_D$. If not, it draws a new
level from $\Lambda_{train}$ as the environment would. If yes, it draws a seen
level from

$$P_{replay}(l_i) = (1-\rho)\,P_S(l_i) + \rho\,P_C(l_i),$$

a mixture of two distributions over $\Lambda_{seen}$:

- $P_S$, **by score**. Sort the seen levels by $S_i$, highest first; let
  $\mathrm{rank}(S_i) \in \{1, 2, \ldots\}$ be $l_i$'s position. Then
  $$P_S(l_i) = \frac{\mathrm{rank}(S_i)^{-1/\beta}}{\sum_j \mathrm{rank}(S_j)^{-1/\beta}}.$$
  Using the rank rather than the score itself ("rank prioritisation") makes
  $P_S$ insensitive to the scale of the scores; $\beta$ sets how top-heavy it
  is — $\beta = 1$ gives weights $1, \tfrac12, \tfrac13, \ldots$, smaller
  $\beta$ concentrates on the top ranks, $\beta \to \infty$ is uniform.
- $P_C$, **by staleness**. $c - C_i$ is how many episodes ago $l_i$ was last
  played;
  $$P_C(l_i) = \frac{c - C_i}{\sum_j (c - C_j)}.$$
  A score was computed under the policy of the time it was last played; the
  older it is, the less it says about the current policy. $P_C$ makes sure
  every seen level is replayed now and then so its score is refreshed. The
  paper shows the method only helps with $0 < \rho < 1$: scores alone or
  staleness alone are both worse than uniform.

**Hyperparameters in the paper.** $\beta = 0.1$; $\rho = 0.1$ on Procgen,
$0.3$ on MiniGrid; $P_D$ grows with the fraction of $\Lambda_{train}$ already
seen.

**When the level set is not finite** (Appendix B.3). With 200 levels,
$\Lambda_{seen}$ can only grow to 200 and the paper never has to drop
anything: once played, a level stays scored forever. When new levels never run
out, $\Lambda_{seen}$ would grow without bound, so the paper caps it at $M$
rows and adds one rule: a new level enters a full set by displacing the level
with the smallest $P_{replay}$, and only if its score is higher. $P_D$ becomes
a constant ($0.5$ or $0.95$ in the paper, by grid search), since "fraction of
the set seen" is undefined. Sampling is otherwise unchanged.

PLR needs no target distribution, no mastery threshold and no observation of
the context — which is why it is first.

## What is different here, and what follows

The paper's $\Lambda_{seen}$ is our buffer; the scores, timestamps,
rank-and-staleness replay and overwrite-on-replay are the paper's. Two facts
about our setting change how the method behaves (1, 3), one is bookkeeping (2),
and one is the paper's own $T$-step-rollout machinery in our shape (4).

**1. A level is a context, not a layout.** PLR's level is a seed that fixes
*everything* about an episode. Our ω fixes the counts and the goal size; the
hazard positions are drawn at reset, inside the environment, from a key the
distribution does not control. Replaying ω therefore means *another layout
with the same counts*. The score of ω is an estimate of the learner's value
error on the *family* of layouts ω describes. This is what we want — the
curriculum is over Ω, the thesis' object — and it makes scores noisier than
PLR's, which is accepted and visible in the diagnostics below.

**2. Ω is not a finite set (bookkeeping).** Every round brings ~655 fresh
contexts, none seen before, so the seen set needs the paper's cap and
replacement rule (Appendix B.3) — that is the only structural difference from
the finite algorithm. "New" is a draw from $r = \mathrm{Uniform}(\Omega)$; $P_D$
is the constant $p$ of Appendix B.3, **0.5** to start: half the data stays a
Uniform(Ω) stream, so the uniform arm is a strict sub-case and the comparison
is clean; the paper's other value, 0.95, is for when fresh draws are nearly
worthless, which we do not know yet. (The finite case's annealing of $P_D$
with coverage has no analogue here: the paper drops it for this case, and so
do we; $P_D$ is 0 only in round 0, when there is nothing to replay.) One
consequence the paper does not have: a buffer row is an
exact point of Ω, and with a continuous `goal_size` a fresh draw never lands on
an existing point. A displaced context is therefore gone for good, and a
near-identical point drawn later is a stranger with no history. The buffer is
$M$ sample points standing in for regions of Ω, not an enumeration of it. The
alternative — a fixed grid over Ω, so neighbours share a score — needs a
resolution choice per suite and, on goal-point, more cells (3466 count tuples
× goal-size bins) than 655 episodes per round can keep scored; see *Not done*.

**3. φ is frozen for a round; ~655 episodes start per round.** PLR refreshes
$P_{replay}$ after *every* episode; the next draw already sees the replayed
level's new (usually lower) score. We draw ≈ 655 contexts per round
(8192 slots × 80 steps / 1000-step episodes) from one $P_{replay}$. With the
paper's $\beta = 0.1$, $P_S \propto \mathrm{rank}^{-10}$ puts 99.9 % of the
*score-driven* part of the mass on the top-ranked context (the staleness part,
$\rho$, still spreads its share over every row — that is what keeps the paper's
method from collapsing onto one level; with $\rho = 0$ it is worse than
uniform, Fig. 16). In the sequential algorithm this is a low-temperature
bandit: replay the current top, its score is overwritten and usually drops,
the next row is top — over hundreds of episodes the top few tens of levels
rotate through. With a frozen φ there is no rotation within a round: at
$\beta = 0.1$, $(1 - \rho) \times 655 \approx 460$ slots land on one ω for a
whole round. The temperature has to stand in for the rotation. What it should
reproduce is "the top few tens of rows share the round's replays": $\beta =
0.3$ ($\mathrm{rank}^{-3.3}$: top 10 of 1000 hold ~94 % of the score mass) is
the paper-faithful candidate; $\beta = 1$ (Zipf: top 10 hold 39 %, top 100
hold 69 %) is warmer. **Default $\beta = 1$**: the warmer of the two is the
safer first run — a too-cold β reproduces the one-ω-per-round failure the
parameter exists to prevent, a too-warm one only dilutes the curriculum towards
uniform, which the diagnostics below make visible. The top-$k$ replay-mass
scalar (below) is what decides between them on a run.

**4. Episodes outlast rounds: 1000 steps vs 80 per slot per round.** The
score is the paper's — the mean |GAE| over **one complete episode**, written
when the episode ends. The paper's $T$-step-rollout form (Appendix C) has the
same problem we do, a rollout being shorter than an episode, and solves it by
keeping per-level partial sums until the episode ends. We do the same per
*slot*: `training/contexts/episodes.py` carries each slot's running
$\sum|\hat A_t|$ and step count across rounds and closes them where
`episode_done` fires. The trainer never resets slots between rounds while a
round hook is installed, so slot $s$ in round $k+1$ continues slot $s$ in round
$k$; the layout invariant "a slot's context changes only after a done" is
checked when the round's transitions are laid out per slot, so a wrong slot/time
order raises instead of producing plausible nonsense. A context drawn in round
$k$ is scored in round $k + 12$ or so, under the policy of then — exactly the
lag the paper has and the staleness term exists for. When several episodes of
one context end in the same round (a replayed row sitting in several slots),
their scores are averaged; the paper has no such case.

Why the window must be the whole episode and not something shorter (say, the
context's transitions in the round, which would make scores at most a round
old): scores have to be *comparable across contexts*, and a partial-episode
score is not. One slot's 80 steps cover one *phase* of a 1000-step episode and
value error varies with phase (the cost critic's error grows along the
episode, seen 2026-10-06), so two contexts with the same true error would rank
by *when in the episode* their slot happened to be. And a fresh context sits in
exactly one slot (its `goal_size` is a new real number), so newcomers would
always carry a one-slot, one-phase score while replayed rows average over
several; the admission rule compares the *maximum* of ~330 newcomer scores
against one incumbent, and would admit lucky newcomers and churn the buffer on
noise. The paper's lag is the price of comparable scores; the staleness term
is what pays it.

What is unchanged from the paper: the score (L1 value loss on the **reward**
critic), rank prioritisation, the staleness mixture, overwrite-on-replay, the
replacement rule for a full buffer.

## The distribution

`PrioritizedLevelReplay(space, buffer_size=1000, replay_probability=0.5,
temperature=1.0, staleness_coefficient=0.3)` — in the paper's symbols,
`buffer_size` = $M$, `replay_probability` = $p$ = $P_D$, `temperature` =
$\beta$, `staleness_coefficient` = $\rho$. Below,
$k$ is the index of the round that just finished and $D$ the number of
dimensions of Ω.

**Parameters φ** (a pytree, frozen for the round, `attach_parameters` as for
every distribution): `contexts [M, D]` (the buffer, one context per row),
`scores [M]` ($S_i$), `last_round [M]` (the round the score was set in — our
$C_i$, counted in rounds instead of episodes), `occupied [M]` (which rows hold
a context), and two values `update` derives from them so that `sample` is a
table lookup: `replay_weights [M]` ($P_{replay}$ for the coming round) and
`replay_probability` ($p$, or 0 while the buffer is empty — round 0 only).

**`sample(φ, key, n)`** (JAX, in-program): for each of the $n$ slots, with
probability `replay_probability` a buffer row drawn by `replay_weights`,
otherwise `space.sample_uniform`. Exact copies of buffer rows, so a replayed
context matches its row bit for bit on the host.

**`update(φ, rollout)`** (host, NumPy, once per round):

1. Take the episodes that *ended* this round (`rollout.completed_episodes`,
   each with its context and its whole-episode mean |reward advantage|); group
   them by exact context, averaging when a context ended several episodes.
2. Contexts already in the buffer: overwrite score, stamp `last_round = k`.
3. Contexts not in the buffer, highest score first: fill an empty row; once
   full, replace the row with the smallest $P_{replay}$ if the candidate's
   score beats that row's score, else stop (every later candidate scores
   lower and $P_{replay}$ has not changed).
4. Recompute $P_{replay}$: $P_S$ by rank with temperature $\beta$; $P_C$ in
   proportion to `k + 1 − last_round`, the number of rounds old a score will
   be when replayed in round $k+1$ (≥ 1, so a round in which every row was
   refreshed is still a distribution).
5. `replay_probability` $= p$ if any row is occupied, else 0.

**`summary(φ)`** → `training_curriculum/distribution/*`: `replay_probability`,
`buffer_occupancy`, `score/mean`, `score/max`, `staleness/mean`,
`replay_mass/top_10`, `replay_mass/top_100`.

The round hook: `observe` stores the round; `on_round_end` lays the recorded
fields out per slot (`Transitions`), closes the episodes that ended through the
`EpisodeTracker` it owns (`CompletedEpisodes`), hands both to `update` as one
`RoundRollout`, attaches φ, returns the metrics. Checkpointing φ — and now the
tracker's per-slot carries — is still open for every distribution
(`contexts_package.md`).

## What the trainer has to provide

The score needs the GAE per transition, which the loss computes per shuffled
minibatch and never returns. The trainer therefore computes, once per round
after the normaliser update and before SGD, the reward advantage of every
transition under the current value function and ships it to the hook next to
the recorded state fields:

```
RoundHook.observe(rollout_fields, learning_signals: LearningSignals)
LearningSignals.reward_advantage   [rows, unroll_length], same layout as the recorded fields
```

`ppo/losses.py: compute_reward_advantages` is the one function; it calls the
same `compute_gae` with the same $\gamma$, $\lambda$ and reward scaling as the
loss, under the round's pre-update policy parameters and the *updated*
observation normaliser — exactly what the loss's first minibatch sees — so the
score is the L1 value loss PPO is about to see on its first pass. (With pixel
augmentation the loss additionally sees translated frames; the score does not.)
Cost: one value-network forward over the batch, next to the 4 × 32 minibatch
passes SGD already does. Computed whenever a round hook is set (every
distribution), so there is one path and the `value_loss/<d>` diagnostic below
exists on every arm.

One thing PLR forced on the wrapper, for every distribution: φ used to be
broadcast over the slot axis so that the inner `VmapWrapper` (which vmaps over
*every* leaf of the state) could step it. For a staged curriculum φ is two
scalars and that was free; PLR's φ is ~36 KB, × 8192 slots ≈ 300 MB per round.
`ContextualAutoResetWrapper.step` now lifts φ out of the state before stepping
the inner stack and puts it back after; φ is stored once per run.

The feedback a distribution learns from is a `RoundRollout` with two views of
the round: `Transitions` (every transition, `[slots, steps]` in each slot's
time order, with context and reward advantage — q̂ and the per-context value
error read from here) and `CompletedEpisodes` (one row per episode that ended,
with its whole-episode value loss — the empirical q and PLR's score read from
here). The protocol has no log-density method: PLR's $q$ is a mixture of point
masses and a density, which has none to report. The intended distribution is
shown by sampling it (below).

## Diagnostics: is it doing what the paper says

Every claim about PLR is a claim about where mass went and why. Three things
are logged every round, for every distribution, under `training_curriculum/`
(`training/contexts/training_curriculum.py`; words in
`training/dashboard/metrics.py`):

| key | what | why |
|---|---|---|
| `intended/<d>` (+ `/mean`, `/std`) | histogram over Ω of 4096 draws from $q_k$ = `sample(φ_k)` — Monte Carlo, the same bins as `sampled` and `experienced` | $q_k$ without the one-episode lag of `sampled`; for PLR the marginals of $P_{replay}$ mixed with $r$ — the paper's Fig. 4 |
| `value_loss/<d>` | histogram: mean \|reward advantage\| of the round's transitions per bin | the score as a function of the context, on every arm — what PLR *would* prioritise; against `intended/<d>` it answers "did q̂ move where the value loss says" |
| `distribution/*` | the scalars above | buffer health (occupancy, staleness, score spread) and **concentration**: `replay_mass/top_10`, `top_100` — the share of $P_{replay}$ on the most-replayed rows. The marginals over one dimension of Ω cannot show concentration: 1000 points ranked in a 5-D Ω have near-uniform marginals even when a few rows hold most of the mass, so this is the number that says whether the curriculum is doing anything, and what $\beta$ is doing (fact 3) |

The check named in the README — *trains on goal-point at least as fast as
uniform, and q̂ moves where the value loss says* — reads: `evaluation/uniform`
and `evaluation/deployment` return curves against the uniform arm; and
`intended/<d>` tracking the high-value-loss region of `value_loss/<d>` with
a lag of about one round.

## Not done, with the reason

- **A score on the cost critic** (`cost_value_loss`): the constrained reading
  of PLR, goal 1's question for this method. One more field in
  `LearningSignals` (PPO-Lagrange computes the cost GAE already) and a choice
  of key in the score. After the reward version has a result.
- **Positive value loss** (Robust PLR, Jiang et al. 2021b): $\max(\text{GAE}, 0)$
  instead of $|\text{GAE}|$. Same place; a variant, not a baseline.
- **A finite grid of levels** (fact 2): would make $P_{replay}$ a vector to
  plot. `intended/<d>` covers the need.
- **Sequential refresh within a round** (fact 3): impossible with a frozen φ
  by design (`training_round.md`); $\beta$ absorbs it.
- **$P_D$ as a function of the buffer's score spread** (replay less when all
  scores are alike): no basis in the paper; the staleness term already spreads
  mass.
