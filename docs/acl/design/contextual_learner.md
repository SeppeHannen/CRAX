# The learner becomes contextual: the multiplier is a function of the context, and the networks see the context

2026-10-14. README plan, task 1. Code: `training/contexts/{observation,networks,learner,constraint}.py`,
`training/agents/ppo_lag/{constraint,train,losses}.py`, `training/agents/ppo/train.py`.

PPO-Lagrange keeps one multiplier λ and updates it from the mean cost of each
round's training batch, so λ settles where the policy is on budget for whatever
the curriculum is sampling now; the same policy then costs three to four times
the budget on level 3 (`research_design_brief.md` §6). This change gives every
context run two switches. The **constraint** is held either with that one
scalar λ or with a multiplier that is a function of the context, λ(ω), moved
at each context by the cost observed there. The **policy** either stays blind to ω or is told
it, as the extra input it would need to hold a different reward–cost trade-off
per context. The critics see ω in every context run. Nothing below changes an
array shape between rounds, so no arm recompiles.

## 1. The multiplier is a function λ_ψ(ω) that the integrator moves and a small network carries

**λ is a function of the context**, `λ_ψ(ω) = softplus(f_ψ(ω̃))`: a two-layer
MLP `f_ψ` over the context normalised to Ω's box, through a positive map
(`training/contexts/constraint.py`). Its parameters ψ are kept where the scalar
λ was kept, `TrainingState.aux_state`, with the optimiser state of the fit. The
output layer starts at zero weights with its bias at `softplus⁻¹(initial λ)`,
so the function is the constant `--initial_lambda_lagr` for every ω until the
first round shapes it; that constant must be positive, since softplus never
reaches 0, and the agent refuses 0 before anything compiles. Being a function,
λ is defined on every ω of Ω, continuous coordinates included, and on every
suite; it needs nothing of Ω beyond its bounds.

**The agent asks a `LagrangeConstraint` three things**
(`training/agents/ppo_lag/constraint.py`) and knows nothing else about where λ lives:

| question | `ScalarConstraint` | `PerContextConstraint(space)` |
|---|---|---|
| the λ each transition is weighted by in the loss | the one value, broadcast | `λ_ψ(transition_context)`, one per transition |
| the update after a round, from the round's transitions | `λ ← relu(λ + η (mean cost − d))` over the whole batch | every transition names a target `λ*_t = relu(λ_ψ(ω_t) + η (c_t − d))`; ψ is fitted to the targets by 32 Adam steps on the squared error |
| which `state.info` keys the trainer must record | none beyond `cost` | `transition_context` |

`d` is the budget per step, `η` is `--lagrangian_coef_rate`, as before.

**The update is the integrator projected onto the function class.** For one
context with one cost the targets are all equal and the fit moves the function's
value there by the scalar's step: measured against `ScalarConstraint` over 40
rounds, the two agree to 0.06. For many contexts each moves by its own
transitions' step, and the network's smoothness spreads the movement to
contexts nearby: under uniform sampling with a cost that grows with the hazard
total, the function reproduced the integrator's value per total to 0.02 after
60 rounds. A context the curriculum has not sampled therefore has a multiplier
before its first episode, inherited from its neighbours, which is what
Giuseppe asked for on 2026-10-14 ("a table is ridiculous"): a stage switch
starts from what the earlier stages taught, not from zero.

**Why fitted targets rather than gradient ascent on the dual.** Dual ascent,
`ψ ← ψ + η ∇_ψ mean_t λ_ψ(ω_t)(c_t − d)`, is the textbook form, but its step
is in units of the hidden weights, not of λ: at the integrator's rate the
function diverged within thirty rounds (measured). The target form keeps `η`
meaning what it means for the scalar, and the fit's own learning rate and step
count are constants of the module, tuned once against the integrator.

**What the curriculum changes.** How often a region of Ω contributes targets,
not what the multiplier there converges to: a context's λ rises while its own
transitions are over budget and falls while they are under, whatever the rest
of the batch does, up to what one smooth function can hold apart. The gap is
logged as `training/lambda_lagr/fit_error`.

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
--constraint      scalar | per_context     default scalar
--policy_context  blind | told             default blind
```

Both require a context distribution; with `--context_distribution none` any
non-default value is refused at the CLI. `--constraint per_context` needs
`--alg ppo_lag` (the only agent with a `LagrangeConstraint`); the existing
"this algorithm does not accept these train() parameters" check refuses the
others. `--policy_context` works with every PPO-family agent, because every
context run now uses the contextual networks.

**The arm** of an experiment is the training distribution plus the learner:
the W&B config gains `arm = "<training_distribution> <constraint> <policy_context>"`,
and the dashboard groups lines by it (`view.GROUP_RUNS_BY`). Run names carry
the same three words. Runs before this change have no `arm` in their config;
their groups' saved views stand as built, and rebuilding one would need the
key added to each run. The five arms of README task 1 are, in this vocabulary:

| arm | `--context_distribution` | `--constraint` | `--policy_context` |
|---|---|---|---|
| `uniform scalar blind` | uniform | scalar | blind |
| `staged:1,2,3 scalar blind` | staged:1,2,3 | scalar | blind |
| `uniform per_context told` | uniform | per_context | told |
| `staged:1,2,3 per_context told` | staged:1,2,3 | per_context | told |
| `staged:1,2,3 per_context blind` | staged:1,2,3 | per_context | blind |

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
within-context scatter alone). `training/cost_violation` stays the batch mean
cost per step minus the budget per step: what moves a scalar λ, descriptive
under λ(ω).

A heatmap of λ over each dimension of Ω (`training_curriculum/…`) is not in
this change: the round hook would need each transition's λ, and the hook's
signals are the critics' quantities. Open, below.

## 5. Limits, stated

- **The function's smoothness is an inductive bias, chosen, not derived.** Two
  contexts that need very different multipliers but lie close in Ω share one
  smooth λ_ψ; the fit then settles between them and `fit_error` stays up. On
  goal-point the levels differ by 4 hazards, and cost varies smoothly with
  density, so the bias is the right one; a suite where cost jumps between
  neighbouring contexts would need a larger network or another input encoding.
- **An unseen context's multiplier is an extrapolation.** It is what the
  neighbours taught, which is right when λ varies smoothly with ω and wrong
  otherwise; the smoke run will show whether level 3's λ at the switch is
  closer to its eventual value than the zero a table gave it.
- **One round's targets are fragments.** A round is 80 steps per slot and an
  episode 1,000, so a context's target comes from fragments of its episodes; the
  fit over 655k transitions and the rounds average this out, as the scalar's
  batch mean does.
- **The fit's constants are tuned on one Ω.** 32 Adam steps at 3e-3 tracked the
  integrator on goal-point's five-dimensional box; another suite may want them
  re-measured (`constraint.py`, the comment at `FIT_STEPS`).
- **Not for PPO-PID.** The PID controller's state is scalar; a per-context PID
  would need its three terms as functions of ω and a design pass of its own.

## Open, not in this change

- λ_ψ over each dimension of Ω as a heatmap over rounds, the way the value
  losses are: the function can be read anywhere (`multipliers_at`), so this is
  a round-hook metric away.
- A `told` subset of ω chosen per factor, for dynamics factors.
