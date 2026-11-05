# Deep-research prompt: prior art for a contextwise constraint in constrained RL

Status: draft 2026-11-04. Paste everything below the rule into ChatGPT Deep
Research. The output comes back into `docs/acl/literature/runs/<tool>/`.

---

You are doing an adversarial prior-art search for a master's thesis. The thesis
has built a specific formulation and needs to know whether it already exists,
under what name, and whether the problem it solves has been named before.
**Finding prior work is a success, not a failure.** Do not argue for novelty;
argue against it as hard as the literature allows, then report what is left.

## 1. The object under test

A family of constrained environments $\{M_\omega\}_{\omega \in \Omega}$. The
*context* $\omega$ is a vector of environment parameters (hazard counts,
object sizes, friction). Within one context the initial state is still random.
Reward return $R$ and cost return $C$ are episodic and undiscounted; $d$ is
one cost budget shared by all contexts.

**The constraint is held per context, in expectation within the context:**

$$\max_\pi\ \mathbb{E}_{\omega \sim q}\,\mathbb{E}_{s_0 \mid \omega}[R]
\quad \text{s.t.} \quad \mathbb{E}_{s_0 \mid \omega}[C] \le d \ \ \text{for every } \omega \in \Omega .$$

**The multiplier is a function of the context**, $\lambda_\psi(\omega) \ge 0$
(a small neural network, or a monotone additive form). It replaces the scalar
$\lambda$ of PPO-Lagrangian in the policy loss
$\mathbb{E}[\,A_R - \lambda_\psi(\omega)\,A_C\,]$. Each training round, the
PPO-Lagrangian integrator rule is applied per context to the costs observed in
that context, and $\psi$ is fitted to those targets. The policy and both
critics observe $\omega$. The training distribution $q_k$ over $\Omega$
**changes during training** (a curriculum or domain randomisation), and the
policy is evaluated on a fixed deployment distribution and on held-out contexts.

The formulation sits between two published granularities:

| granularity | constraint | multiplier | expectation taken over | known instance |
|---|---|---|---|---|
| expectation-based CMDP | $\mathbb{E}_{\omega,s_0}[C] \le d$ | one scalar | everything | Altman 1999; PPO-Lagrangian |
| **contextwise (under test)** | $\mathbb{E}_{s_0\mid\omega}[C] \le d\ \ \forall \omega$ | $\lambda(\omega)$ | initial states within a context | ? |
| statewise | $v_C^\pi(s) \le d\ \ \forall s$ | $\lambda(s)$ | nothing; discounted cost-to-go per state | Feasible Actor-Critic, arXiv 2105.10682 |

A third axis matters: *robust* CMDPs (Russel et al. 2010.04870, Mankowitz et
al. 2010.10644, Wang et al. 2209.06866) keep a **scalar** multiplier and take
the **worst case** over an **unobserved** uncertainty set. The formulation
under test takes an expectation inside an **observed** context and holds the
budget for **every** context. Keep these three axes separate in everything
you report: (a) what the multiplier is a function of, (b) what the constraint
quantifies over, (c) whether the context is observed by the policy.

## 2. The questions

Answer each with evidence rows (format in §5), then a verdict.

**Q1 — Does the formulation exist?** A constrained RL / CMDP paper whose
constraint must hold for each member of a context, task, goal, or environment
family, in expectation within the member, and whose Lagrange multiplier (or
equivalent dual variable, penalty weight, or scalarisation weight) is indexed
by or a function of that member. Includes: per-task multipliers in multi-task
or multi-goal safe RL; task-conditioned multipliers in safe meta-RL; multipliers
indexed by domain-randomisation parameters in safe sim-to-real; constrained
contextual MDPs or constrained contextual bandits with per-context duals;
multi-objective RL where a preference or scalarisation weight is conditioned
on a task variable and used as a constraint. For every hit, record *exactly*
what the multiplier is a function of and quote the update equation's location.

**Q2 — What is it called?** Collect the names used for constraints that hold
for every member of a family: "robust", "statewise", "per-task", "uniform",
"pointwise", "almost-sure over contexts", "contextual CMDP", "multi-constraint
CMDP", "semi-infinite", anything else. For each name, one sentence on what the
authors mean by it (worst case vs. per-member expectation; observed vs.
unobserved). The thesis currently calls it "robust" and suspects that is
wrong.

**Q3 — Theory.** Which results cover a CMDP with one constraint per context
(a continuum of constraints) and a dual variable that is a function over that
continuum? Altman's multi-constraint CMDP, semi-infinite programming applied to
CMDPs, Lagrangian duality with function-approximated multipliers, FAC's
Theorems 7–9. State what each result assumes (finite constraints? convexity?
tabular?) and whether it transfers to a function $\lambda(\omega)$.

**Q4 — Has the scalar-multiplier problem under a moving training distribution
been named?** The thesis's motivating observation: a scalar $\lambda$ driven by
the batch cost converges to "on budget for the current training distribution";
when a curriculum or domain randomisation moves that distribution, the
constraint is enforced on a moving target, and anchoring $\lambda$ to a
reference distribution instead makes the constraint and the policy gradient
speak about different data. Find any paper that reports, designs for, or
argues this: Lagrangian safe RL combined with curricula, domain randomisation,
non-stationary or continual CMDPs, or online CMDPs with changing dynamics. The
Safe Curriculum Generation line (constrained CURROT; find its id) is believed
to use a scalar multiplier from the batch — verify from its method section.

**Q5 — Nearest adjacent constructs, and why they differ.** Threshold-conditioned
policies (CCPO, 2310.03718), statewise multipliers (FAC, 2105.10682), robust
CMDPs (the three above), safe in-context RL (SCARED, 2509.25582), Pareto- or
preference-conditioned policies, cost-conditioned "versatile" safe RL. For each,
one row stating the axis on which it differs from the object under test.

**Q6 — Empirical.** Any paper that evaluates a per-member constraint by
reporting cost *per member* (per task, per level, per context) and shows the
expectation-based baseline meeting the budget on average while violating it on
some members. This is the pattern the thesis measured; find who has shown it.

## 3. Where to look

Keyword lanes (arXiv abstract search, Semantic Scholar, OpenReview, Google
Scholar forward citations). Run each, screen by title and abstract, then read
the **method section** of every candidate: the classification needs the
multiplier update equation, not the abstract.

| lane | query idea |
|---|---|
| L1 | ("constrained" OR "safe") AND ("multi-task" OR "multi-goal" OR "goal-conditioned") AND ("reinforcement learning") AND (Lagrang* OR "dual variable" OR multiplier) |
| L2 | ("safe" OR "constrained") AND ("meta-reinforcement learning" OR "meta-RL" OR "in-context") |
| L3 | ("constrained" OR "safe") AND ("contextual MDP" OR "contextual Markov" OR "contextual bandit") |
| L4 | ("constrained" OR "safe") AND ("domain randomization" OR "sim-to-real") AND (Lagrang* OR multiplier) |
| L5 | ("Lagrange multiplier" OR Lagrangian) AND (curriculum OR "non-stationary" OR "continual" OR "changing dynamics") AND ("constrained" OR "safe") AND "reinforcement learning" |
| L6 | ("statewise" OR "state-wise" OR "state-dependent") AND ("Lagrange multiplier" OR Lagrangian) AND "reinforcement learning" |
| L7 | ("multi-constraint" OR "multiple constraints" OR "semi-infinite") AND ("constrained Markov decision" OR CMDP) |
| L8 | ("constraint-conditioned" OR "threshold-conditioned" OR "cost-conditioned" OR "versatile safe") AND "reinforcement learning" |
| L9 | ("multi-objective reinforcement learning") AND (conditioned OR "preference") AND (constraint OR safe) |

Seeds for forward and backward snowballing (one round each; a second round
only if the first added something): 2105.10682 (FAC), 2310.03718 (CCPO),
2010.04870, 2010.10644, 2209.06866 (robust CMDPs), 2509.25582 (SCARED),
Altman 1999 *Constrained Markov Decision Processes* (book; cite by title),
Stooke et al. 2020 PID-Lagrangian (2007.03964), Safe Curriculum Generation
(constrained CURROT; locate the id), SPiDR (2509.18648), Safety Gym report
(Ray, Achiam, Amodei 2019; no arXiv id, cite by title).

Expect the answer to Q1 to hide in multi-task and meta safe RL papers whose
abstracts never say "multiplier". Read their algorithm boxes.

## 4. Hard rules

- **Every paper needs an arXiv id or a DOI** in `paper_id`, as `2105.10682`
  (no prefix, no version) or `10.xxxx/...`. No id, no row. Ids are checked by
  a script; one that does not resolve, or resolves to a different title, is
  reported as a fabrication.
- `title` is the full title, not an acronym.
- `quote` is verbatim, ≤ 40 words, with section, equation or algorithm number
  in `location`. If the full text was not accessible, write
  `full text not accessed` in `location`, leave `quote` empty, and lower your
  confidence in that row's classification.
- Classify from the **method section**, never from the abstract. If the
  multiplier update is not shown, say `update not shown`.
- Record search strings and dates as executed; if your tooling hides them,
  say so.
- When unsure which value a field takes, give your best value followed by `?`.

## 5. Output

Return **four fenced code blocks in this order, each starting with a comment
line naming the file**, then a short free-text section. CSV: one header row,
exact column names and order, commas inside fields quoted.

### Block 1 — `neighbours.csv`

```
paper_id,title,year,venue,question,multiplier_argument,constraint_quantifier,context_observed_by,training_distribution,learner,name_used,distance,location,quote
```

- `question`: `Q1`, `Q4`, `Q5`, `Q6` (a paper may have several rows)
- `multiplier_argument`: `scalar` | `state` | `context` | `task_id` |
  `threshold` | `preference` | `none` | `other:<what>`
- `constraint_quantifier`: `expectation_over_all` | `per_member_expectation` |
  `per_state` | `worst_case_over_set` | `chance` | `other:<what>`
- `context_observed_by`: comma-free list from `policy` `critic` `multiplier`
  `none`, joined with `+` (e.g. `critic+multiplier`)
- `training_distribution`: `fixed` | `domain_randomisation` | `curriculum` |
  `meta_task_distribution` | `other:<what>`
- `learner`: the base algorithm (PPO-Lagrangian, SAC-Lagrangian, CPO, MPO, …)
- `name_used`: the authors' own term for their constraint, ≤ 5 words
- `distance`: `same` (Q1 answered yes) | `special_case` (theirs is a special
  case of the object under test, e.g. finite task ids) | `generalisation`
  (theirs contains it, e.g. statewise) | `adjacent` (shares a device but a
  different axis)

### Block 2 — `terminology.csv`

```
term,paper_id,meaning,quantifier,observed
```

One row per (term, paper) for Q2. `quantifier` and `observed` use the Block 1
vocabularies.

### Block 3 — `theory.csv`

```
paper_id,title,year,result,assumes,transfers_to_function_multiplier,location
```

For Q3. `transfers_to_function_multiplier`: `yes` | `with:<extra assumption>` |
`no:<why>`.

### Block 4 — `verdict.md`

For Q1 to Q6, in this fixed shape:

```
## Q<n> — <question in five words>
verdict: <one sentence; for Q1 one of: exists | exists as special case | not found>
strongest:
- <paper_id>: <one sentence>
- <paper_id>: <one sentence>
what would change it: <≤ 2 sentences>
```

Then `## Naming`: a recommendation for what the thesis should call the
constraint, with the one-sentence reason and the papers whose usage it
follows or avoids. Then `## Search log`: queries and dates as executed.

### Free text (after the blocks, ≤ 300 words)

What surprised you; which lane was empty and whether that is a real absence
or a vocabulary problem; what you would read next with another hour.
