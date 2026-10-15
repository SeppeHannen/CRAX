# The learner becomes contextual: the multiplier is a function of the context, and the networks see the context

2026-10-14, §1 rewritten 2026-10-15. README plan, task 1. Code:
`training/contexts/{observation,networks,learner}.py`, `training/contexts/constraints/`,
`training/agents/ppo_lag/{constraint,train,losses}.py`, `training/agents/ppo/train.py`.

PPO-Lagrange keeps one multiplier λ and updates it from the mean cost of each
round's training batch, so λ settles where the policy is on budget for whatever
the curriculum is sampling now; the same policy then costs three to four times
the budget on level 3 (`research_design_brief.md` §6). This change gives every
context run two switches. The **constraint** is held either with that one
scalar λ or with a multiplier that is a function of the context, λ(ω), moved
at each context by the cost observed there, in one of two function classes: a
small network, or a monotone additive model of difficulty. The **policy** either stays blind to ω or is told
it, as the extra input it would need to hold a different reward–cost trade-off
per context. The critics see ω in every context run. Nothing below changes an
array shape between rounds, so no arm recompiles.

## 1. The multiplier is a function of the context that the integrator moves; two function classes carry it

**A multiplier is a nondecreasing function of difficulty.** At equilibrium
PPO-Lagrange's λ is the shadow price of the budget: 0 wherever the policy is
under budget on its own, positive where the budget binds, and larger the harder
it is to stay under. So the ideal per-context multiplier is λ*(ω) = f(D(ω))
for some measure D of how hard the budget is to meet at ω and some
nondecreasing price f that is 0 on the slack range. This is what a multiplier
*is*, not a modelling choice; it fixes what any function class for λ(ω) must
be able to say.

**The agent asks a `LagrangeConstraint` three things**
(`training/agents/ppo_lag/constraint.py`) and knows nothing else about where λ lives:

| question | `ScalarConstraint` | the per-context forms (`training/contexts/constraints/`) |
|---|---|---|
| the λ each transition is weighted by in the loss | the one value, broadcast | `λ(transition_context)`, one per transition |
| the update after a round, from the round's transitions | `λ ← relu(λ + η (mean cost − d))` over the whole batch | every transition names a target `λ*_t = relu(λ(ω_t) + η (c_t − d))`; the function is fitted to the targets by least squares |
| which `state.info` keys the trainer must record | none beyond `cost` | `transition_context` |

`d` is the budget per step, `η` is `--lagrangian_coef_rate`, as before. The
target form is shared by both classes (`fitted.py`), so they differ only in
the function and its fit, and `η` keeps the meaning it has for the scalar. The
textbook alternative, gradient ascent on the dual objective, was measured and
rejected: its step is in units of the parameters, not of λ, and at the
integrator's rate a network diverged within thirty rounds.

**`network`: λ_ψ(ω) = softplus(MLP_ψ(ω̃))** (`network.py`), a two-layer MLP over
the context normalised to Ω's box, through a positive map; fitted by 32 Adam
steps per round. It assumes nothing about the cost landscape. The price: what
it says about a context it has not seen is whatever the random hidden layers
happened to produce (at a stage switch, measured: level 2 inherited 0.69 of
level 1's 2.5, level 3 0.01), it depends on a seed, and softplus never reaches
0, so `--initial_lambda_lagr` must be positive and slack contexts carry a small
λ forever. It is kept as the unconstrained comparison.

**`additive`: λ(ω) = f(Σ_i h_i(ω_i))** (`additive.py`), the form that follows
from two assumptions Giuseppe stated on 2026-10-15 and from the fact above.
Each step is forced by the one before it, or is a knob named as such:

| step | what it says | forced by |
|---|---|---|
| λ = f(D), f nondecreasing | the price of difficulty | what a multiplier is |
| D = Σ_i h_i(ω_i), each h_i nondecreasing in factor i's difficulty direction | difficulty is additive over the factors and monotone in each | the two assumptions: *monotone per factor* (more hazards never make the budget easier to meet; a smaller goal never makes it harder to miss) and *no interactions* (a cylinder's contribution does not depend on the cubes). Which way difficulty runs is a fact of the suite, `Dimension.harder_when` in the registry |
| h_i is a staircase: nonnegative step heights at every value of a count | the complete description of a monotone function on a finite set | nothing is assumed beyond monotonicity; there is no simpler and no more general form. A continuous factor (goal size) is cut into `CONTINUOUS_BINS` = 8 equal bins — **knob** |
| D scaled to [0, 1] by the sum of all heights | 0 at the easiest corner of Ω, 1 at the hardest | so f lives on one interval whatever the suite |
| f piecewise linear over `PRICE_SEGMENTS` = 16 equal segments with nonnegative slopes, plus a baseline b ≥ 0 | any nondecreasing price curve, differentiable in D | the fit moves the heights *through* f, so f must have a gradient in D; a staircase f would not. 16 — **knob** (the fit is indifferent to 8) |
| every parameter ≥ 0 | monotonicity of h_i and f at every round, by construction | each parameter is a difference of a monotone function; the shape constraint becomes a box constraint, enforced by clipping at 0 after each step (isotonic regression's device) |

On goal-point that is 4 × 12 + 7 = 55 heights, 16 slopes and one baseline: 72
nonnegative numbers. At the start every slope is 0 and every height is equal,
so λ is the constant `--initial_lambda_lagr` (which may be 0) and D is the
share of all possible factor steps a context has taken. The fit is 300
projected Adam steps per round, warm-started (**knob**, numerical: the heights,
slopes and baseline differ in scale by orders of magnitude, and one step size
for all either stalled the slopes or drove the baseline to 0 — measured).

**The sum-of-factors form was built first and rejected.** λ = b + Σ_i g_i(ω_i),
the literal reading of "additive", cannot be 0 on a slack context while
raising λ for one factor elsewhere: g_i depends on ω_i alone, so a rise at some
cylinder count applies at every value of the other factors, slack or not. Its
slack region is a box; a shadow price's is a level set of difficulty. On a cost
linear in the hazard total it missed the integrator by 0.19 where f(Σ h_i)
reproduces it to 0.001. The literal form is the special case f(D) = D.

**Measured against the integrator** (simulated costs on goal-point's Ω, the
reference being PPO-Lagrange's own rule applied per context with the same η):

| | `network` | `additive` |
|---|---|---|
| one context over budget for 40 rounds: λ there vs the scalar | agree to 0.06 | agree to 0.01 |
| uniform sampling, cost linear in the hazard total, 60 rounds: λ per total vs the integrator's | mean error 0.02 | mean error 0.001 |
| what an unseen harder level inherits at a stage switch (level 1 at λ = 2.5) | 0.69, 0.01 — seeded | 2.40, 2.31 — at least the hardest seen, by monotonicity |
| cost with an interaction between two factors | fitted | fitted too |

The last row is a stated limit: on this Ω both forms fit an interaction, so
`fit_error` does **not** detect a violated no-interaction assumption; whether
interactions matter has to be read from behaviour. And neither form rescues a
staged curriculum's switches: one context per stage carries no direction on
any factor, so the best a monotone model can do is "at least as hard as what I
have seen" — a lower bound, not an estimate.

**What the curriculum changes.** How often a region of Ω contributes targets,
not what the multiplier there converges to: a context's λ rises while its own
transitions are over budget and falls while they are under, whatever the rest
of the batch does, up to what the function class can hold apart. Under the
additive form the fit of h_i needs only the *marginal* coverage of factor i, so
where a curriculum tilts its joint mass stops mattering for λ. The remaining
gap is logged as `training/lambda_lagr/fit_error`.

**The cost of a transition reaches the update from the round's data, not from
the loss's metrics.** The base trainer's `post_step_fn` now receives the
round's `Transition` next to the training state and the loss metrics, and the
scalar rule reads `mean(cost)` from it: the same number the loss's
`mean_cost` averaged over minibatches used to give, now from the one place the
per-context rule has to read anyway. The other constrained agents received the
same signature and ignore the new argument.

**The learning signals use the same per-transition λ.** `plr:lagrangian`
scores by `A_r − λ A_c`; `compute_learning_signals` takes λ per transition from
the constraint, so under λ(ω) the score is computed with each transition's own
multiplier, as the loss is.

## 2. The context enters the observation as a labelled part, and each network reads the parts it is allowed

**The observation of a context run is a dict** `{"sensors": o, "context": ω}`
(`training/contexts/observation.py`, `ContextObservationWrapper`), produced
outermost in both the training and the evaluation stack from
`state.info["context"]`; the environment is untouched and a stock run still
observes the flat vector. On the step an episode ends the inner wrapper already
replaces the observation with the fresh episode's and the context with the
fresh episode's ω, so the two parts of the dict always describe the same
episode. Labelling the parts at the source is what lets a network declare what
it reads instead of slicing a concatenation by position.

**The networks read keys** (`training/contexts/networks.py`,
`make_contextual_ppo_networks`): each MLP concatenates the normalised parts it
is given. The observation normaliser already runs per key.

| network | reads | why |
|---|---|---|
| policy, `--policy_context blind` | `sensors` | the baseline every published method uses; under λ(ω) it is the control Tristan predicted — a scalar λ in disguise, since one behaviour cannot hold a trade-off per context |
| policy, `--policy_context told` | `sensors`, `context` | can hold a different trade-off per context; for goal-point's layout factors this is what a deployed robot could be told (brief §4) |
| reward critic, cost critic | `sensors`, `context` | always: the cost critic must estimate cost per context for λ(ω) to be about something; the critic exists only in training, so what a robot knows is no concern |

Which factors a policy may be told will be decided per factor when dynamics
factors arrive (friction, actuator gain cannot be told). Today every dimension
of goal-point's Ω is a layout factor, so `told` is all of ω. The next instance
is a `told` subset chosen by the wrapper, not a change to the networks.

## 3. Two CLI switches name the learner; every arm can run under any combination

```
--constraint      scalar | network | additive     default scalar
--policy_context  blind | told                    default blind
```

Both require a context distribution; with `--context_distribution none` any
non-default value is refused at the CLI. A per-context `--constraint` needs
`--alg ppo_lag` (the only agent with a `LagrangeConstraint`); the existing
"this algorithm does not accept these train() parameters" check refuses the
others. `--policy_context` works with every PPO-family agent, because every
context run now uses the contextual networks.

**The arm** of an experiment is the training distribution plus the learner:
the W&B config gains `arm = "<training_distribution> <constraint> <policy_context>"`,
and the dashboard groups lines by it (`view.GROUP_RUNS_BY`). Run names carry
the same three words. Runs before this change have no `arm` in their config;
their groups' saved views stand as built, and rebuilding one would need the
key added to each run (the 2026-10-14 smokes ran under the name `per_context`,
which became `network` when the additive form arrived). The first comparison of
the two function classes, 200 M steps on goal-point (group
`goal_point_lambda_forms_200M`, 2026-10-15):

| arm | `--context_distribution` | `--constraint` | `--policy_context` |
|---|---|---|---|
| `staged:1,2,3 network told` | staged:1,2,3 | network | told |
| `uniform network told` | uniform | network | told |
| `staged:1,2,3 additive told` | staged:1,2,3 | additive | told |
| `uniform additive told` | uniform | additive | told |

The scalar baselines and the blind control of README task 1 follow once the
function class is settled.

## 4. What is logged

`training/lambda_lagr` keeps its meaning for a scalar arm. For a per-context
arm it is the mean over the round's transitions of λ_ψ (after the update) at
each transition's context: the λ the batch was weighted by on average, so the
panel compares across arms. Next to it a per-context arm logs, over the same
population, `training/lambda_lagr/std`, `/min`, `/max` — the spread the blind
control must show while its behaviour does not change — and
`training/lambda_lagr/fit_error`, the root mean squared distance between the
fitted function and the integrator's targets, next to
`training/lambda_lagr/target_spread`, the targets' standard deviation over the
batch. Targets vary between contexts, whose multipliers differ, and within a
context, because a target is one transition's cost; the fit resolves the first
and lands on the mean of the second. Fit error well below the spread means the
function holds the contexts apart (uniform smoke: 0.16 against 1.75); near it
means it does not (staged: 0.3–0.6 against a one-context batch, the
within-context scatter alone). The additive form adds `training/lambda_lagr/baseline`,
λ at the easiest corner of Ω, and `training/lambda_lagr/difficulty_share/<factor>`,
each factor's share of the learnt difficulty range — which factors the
multiplier has learnt to price. `training/cost_violation` stays the batch mean
cost per step minus the budget per step: what moves a scalar λ, descriptive
under λ(ω).

A heatmap of λ over each dimension of Ω (`training_curriculum/…`) is not in
this change: the round hook would need each transition's λ, and the hook's
signals are the critics' quantities. Open, below.

## 5. Limits, stated

- **Each function class is an inductive bias, chosen, not derived.** The
  network assumes λ varies smoothly with ω; the additive form assumes it is
  monotone in each factor and that factors do not interact. Both are claims
  about goal-point's cost landscape that the fit metric cannot test (§1: both
  fit an interaction). A suite where the budget's price is not monotone in a
  factor, or where kinds interact (cubes that block routes so disc contact
  becomes unavoidable), breaks the additive form; a suite where cost jumps
  between neighbouring contexts breaks the network's smoothness.
- **An unseen context's multiplier is an extrapolation.** Under the network it
  is whatever the hidden layers produce; under the additive form it is a bound,
  at least the hardest-seen λ when the context is harder on every factor. At a
  stage switch neither is an estimate: one context per stage carries no
  direction.
- **Within a factor, between observed values, the additive form is not
  identified.** Only the sum of the step heights between two observed values is
  fixed by the data; where the steps sit between them is the solver's choice.
  λ at an unobserved value lies between its observed neighbours', no more.
- **One round's targets are fragments.** A round is 80 steps per slot and an
  episode 1,000, so a context's target comes from fragments of its episodes; the
  fit over 655k transitions and the rounds average this out, as the scalar's
  batch mean does.
- **The fits' constants are tuned on one Ω.** Network: 32 Adam steps at 3e-3;
  additive: 300 projected Adam steps at 1e-2, 8 bins per continuous factor, 16
  price segments. Another suite may want them re-measured (the comments at the
  constants in `network.py` and `additive.py`).
- **Not for PPO-PID.** The PID controller's state is scalar; a per-context PID
  would need its three terms as functions of ω and a design pass of its own.

## Open, not in this change

- λ_ψ over each dimension of Ω as a heatmap over rounds, the way the value
  losses are: the function can be read anywhere (`multipliers_at`), so this is
  a round-hook metric away.
- A `told` subset of ω chosen per factor, for dynamics factors.
