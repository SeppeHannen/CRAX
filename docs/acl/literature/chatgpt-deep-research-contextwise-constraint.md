# Adversarial Prior-Art Search: Contextwise Constrained Reinforcement Learning

```csv
# neighbours.csv
paper_id,title,year,venue,question,multiplier_argument,constraint_quantifier,context_observed_by,training_distribution,learner,name_used,distance,location,quote
"2405.02456","Natural Policy Gradient and Actor Critic Methods for Constrained Multi-Task Reinforcement Learning",2024,"arXiv","Q1","task_id","per_member_expectation","critic+multiplier","fixed","MT-PDNAC","constrained multi-task formulation","special_case","§2 Eq. (3); §3 Eq. (4); Algorithm 2 Eq. (15)","For each task i, initialize dual variables λ_i^0,ν_i^0... and critic parameters Q_i^0."
"2405.02456","Natural Policy Gradient and Actor Critic Methods for Constrained Multi-Task Reinforcement Learning",2024,"arXiv","Q6","task_id","per_member_expectation","critic+multiplier","fixed","MT-PDNPG / MT-PDNAC","performance in each task","special_case","§1; §2 Eq. (3); §7","the constrained formulation provides a way to mitigate this task imbalance."
"2406.14047","Constrained Meta Agnostic Reinforcement Learning",2024,"arXiv","Q1","task_id","per_member_expectation","multiplier","meta_task_distribution","TRPO-Lagrangian + MAML","task-specific constraints","special_case","§4.1 Eq. (6)–(7)","This formulation utilizes dual variables, λ_p, for each task p."
"2406.14047","Constrained Meta Agnostic Reinforcement Learning",2024,"arXiv","Q5","scalar","expectation_over_all","none","meta_task_distribution","FoMAML + Lagrangian","universal safety constraint","adjacent","§4.2 Eq. (8); §4.3","C-MAML incorporates task-specific constraints in the inner loop alongside a universal safety constraint in the outer loop"
"2105.10682","Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety",2021,"NeurIPS","Q5","state","per_state","critic+multiplier","fixed","Feasible Actor-Critic","statewise safety constraint","generalisation","§3.2, original statewise Lagrangian (O-SL)","As each state in I_F has a constraint, there exists the corresponding Lagrange multiplier at each state"
"2105.10682","Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety",2021,"NeurIPS","Q6","state","per_state","critic+multiplier","fixed","Feasible Actor-Critic","statewise safety","adjacent","§3.1; Theorem 8 discussion","a feasible policy under expectation-based constraints may become 'unsafe' in certain states"
"2310.03718","Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning",2023,"NeurIPS 2023 Workshop / arXiv","Q5","threshold","per_member_expectation","policy+critic","other:sampled_thresholds","CVPO-based EM","versatile safe RL","adjacent","§3.2 Eq. (2); §4 Eq. (8)–(9)","the dual variables η_i* and λ_i* are the solutions of the following convex optimization problem"
"2209.06866","Robust Constrained Reinforcement Learning",2022,"NeurIPS 2022 / arXiv","Q5","scalar","worst_case_over_set","none","fixed","robust primal-dual policy gradient","robust constrained RL","adjacent","§3 Eq. (3); Appendix Eq. (55)–(56)","Any feasible solution... can guarantee that under any MDP in the uncertainty set, its accumulative discounted utility is always no less than b_i"
"2010.10644","Robust Constrained Reinforcement Learning for Continuous Control with Model Misspecification",2020,"ICLR 2021 / arXiv","Q5","scalar","worst_case_over_set","none","fixed","R3C / RC-D4PG / RC-DMPO","robust constrained RL","adjacent","§2.1; Appendix A.1 Eq. (6)","min_{λ≥0} max_{π∈Π} inf_{p∈P}... − λ(sup_{p∈P}...−β)"
"2509.25582","Safe In-Context Reinforcement Learning",2026,"ICML 2026","Q5","scalar","other:per_adaptation_episode","policy","meta_task_distribution","SCARED actor-critic","safe ICRL","adjacent","§3 Eq. (4); §4 Eq. (5)–(6)","These motivate the use of a single multiplier for all episodes."
"2509.25582","Safe In-Context Reinforcement Learning",2026,"ICML 2026","Q1","other:episode_index","other:per_adaptation_episode","policy+multiplier","meta_task_distribution","SCARED theoretical dual","safe ICRL","adjacent","§4 Eq. (5)","where λ≽0 denotes componentwise nonnegativity of the Lagrangian multipliers."
"2509.18648","SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer",2025,"arXiv","Q5","scalar","expectation_over_all","none","domain_randomisation","generic CMDP solver; SAC/PPO primal-dual among implementations","pessimistic domain randomization","adjacent","§4.1 Eq. (2); Algorithm 1; Appendix M","Domain randomization... solving max E_{ξ∼μ}J... s.t. E_{ξ∼μ}C...≤d."
"2509.18648","SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer",2025,"arXiv","Q6","scalar","expectation_over_all","none","domain_randomisation","generic CMDP solver","constraint underestimation","adjacent","§5.1 Fig. 3; Appendix D Fig. 13","domain randomization dramatically violates the safety constraints."
"2201.11965","Provably Efficient Primal-Dual Reinforcement Learning for CMDPs with Non-stationary Objectives and Constraints",2022,"AAAI 2023","Q4","other:time-varying_dual","other:time-varying_constraints","none","other:non-stationary CMDP sequence","PROPD-PPO","non-stationary objectives and constraints","adjacent","full text not accessed",""
"2007.03964","Responsive Safety in Reinforcement Learning by PID Lagrangian Methods",2020,"ICML 2020","Q4","scalar","expectation_over_all","none","fixed","PID-Lagrangian","PID Lagrangian","adjacent","full text not accessed",""
```

```csv
# terminology.csv
term,paper_id,meaning,quantifier,observed
"constrained multi-task formulation","2405.02456","One shared policy maximizes average task return subject to separate lower/upper expected-return bounds for every finite task.","per_member_expectation","critic+multiplier"
"task-specific constraints","2406.14047","Each meta-training task has its own inner-loop expected-cost constraint and task-indexed dual λ_p; the paper's outer safety constraint is instead averaged over tasks.","per_member_expectation","multiplier"
"statewise safety constraint","2105.10682","Expected discounted cost-to-go must satisfy the budget for every feasible initial state.","per_state","critic+multiplier"
"semi-infinitely constrained MDP","2305.00254","A continuum of expected-value constraints V_c_y^π(μ)≤u_y indexed by y in a compact set Y.","per_member_expectation","none"
"robust constrained RL","2209.06866","Reward and constraint values use worst-case transition dynamics from an uncertainty set; the uncertainty realization is not a policy input.","worst_case_over_set","none"
"robust constrained RL","2010.10644","Worst-case reward is combined with worst-case constraint cost over model uncertainty and a scalar Lagrange multiplier.","worst_case_over_set","none"
"versatile safe RL","2310.03718","A policy is conditioned on the requested safety threshold and is intended to satisfy the corresponding expected-cost constraint for every threshold.","per_member_expectation","policy+critic"
"safe ICRL","2509.25582","A fixed pretrained contextual policy must meet an expected cost budget in each successive test-time adaptation episode; the context is interaction history rather than an observed environment parameter.","other:per_adaptation_episode","policy"
"pessimistic domain randomization","2509.18648","Training still constrains expected cost averaged across randomized simulator domains, but augments cost pessimistically to protect real-world deployment.","expectation_over_all","none"
"non-stationary objectives and constraints","2201.11965","The CMDP reward, utility/constraint functions, and transitions may change over episodes subject to variation budgets.","other:time-varying_constraints","none"
```

```csv
# theory.csv
paper_id,title,year,result,assumes,transfers_to_function_multiplier,location
"10.1007/s001860050035","Constrained Markov decision processes with total cost criteria: Lagrangian approach and dual linear program",1998,"Lagrangian/LP duality and no duality gap for classes of total-cost CMDPs under stated regularity conditions.","Countable state space; total expected cost criteria; transient MDPs or uniform-Lyapunov conditions; a finite collection of constraint costs.","no:the published Lagrange multiplier is a finite vector over finitely many constraints; an uncountable index set requires infinite-dimensional/semi-infinite duality.","Paper abstract and Lagrangian/dual-LP results; DOI metadata"
"2405.02456","Natural Policy Gradient and Actor Critic Methods for Constrained Multi-Task Reinforcement Learning",2024,"For finitely many task constraints, a task-indexed dual vector and primal-dual natural actor-critic converge to the constrained multi-task optimum under the paper's assumptions.","Finite N; in main analysis common finite state space and transition kernel, task-dependent rewards; Slater condition; tabular softmax, with a later linear-function-approximation extension; uniform ergodicity for sample-based result.","no:directly covers only finite λ_i,ν_i; λ(ω) over a continuum needs a separate approximation/duality argument.","§2 Assumption 1; §3 Eq. (4)–(11); §4 Corollary 2"
"2305.00254","Semi-Infinitely Constrained Markov Decision Processes and Efficient Reinforcement Learning",2023,"Directly formulates a CMDP with a continuum of expected constraints indexed by y∈Y and reduces known-model planning to linear semi-infinite programming; gives sample-complexity results for SI-CRL and SI-CPO.","Compact Y⊂R^m; finite S,A for the base SICMDP; feasibility/optimal-solution assumption; regularity including Lipschitz dependence on y; SI-CPO additionally assumes differentiable/Lipschitz/smooth policy class, Fisher condition and bounded approximation errors.","with:the continuum constraint set transfers directly, but a neural λψ(ω) is not the dual representation analyzed; one needs assumptions ensuring the infinite-dimensional dual admits a density/function and that the chosen function class approximates it.","§3 Eq. (M), Eq. (5), Assumption 1; §5.2 Assumptions 6–11"
"2105.10682","Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety",2021,"Theorems 7–9 connect a state-indexed multiplier mapping to the statewise constrained problem and compare its feasible set with expectation-based safety.","Statewise constraints on feasible initial states; existence of optimal policy/multiplier mapping for Theorem 7; Theorem 9 additionally invokes convexity/concavity and a per-state Slater-type condition.","with:replace state by observed context only if the resulting optimization has analogous value functions, feasibility, convexity/duality assumptions; FAC does not prove neural function-approximation consistency for arbitrary λ(ω).","§4 Theorems 7–9"
"2209.06866","Robust Constrained Reinforcement Learning",2022,"Strong-duality-based primal-dual analysis for robust reward/constraint values uses a scalar multiplier for a robust constraint.","Rectangular model-uncertainty set; discounted setting; Slater-type feasibility and smoothness/parameterization assumptions in algorithmic results.","no:its scalar dual multiplies a worst-case robust value constraint, not a continuum of separately dualized observed-context constraints.","§2–§3; Appendix Eq. (46), (55)–(63)"
```

```markdown
# verdict.md
## Q1 — Does the formulation exist?
verdict: exists as special case — finite per-task expected constraints with one dual variable per task are explicit prior art, but I did not find the full combination of an observed continuous environment context, one shared context-conditioned policy, and a learned function λψ(ω) enforcing every context separately.
strongest:
- 2405.02456: gives a single multi-task policy, a separate expected-value constraint for every task, a vector of per-task dual variables, and explicit per-task dual updates.
- 2406.14047: gives task-specific CMDP constraints and λ_p in safe meta-RL, although this occurs in the task-adaptation inner loop and its outer-loop constraint is averaged across tasks.
what would change it: A paper with π(a|s,ω) and λ(ω), or a neural/context-parametric dual over environment parameters, would upgrade the verdict from special case to exists. A finite-task paper that merely stores λ_i is already enough to defeat novelty claims framed broadly as “member-indexed Lagrange multipliers.”

## Q2 — What is it called?
verdict: The closest precise names are “constrained multi-task,” “per-task/task-specific constraints,” and, for a continuum, “semi-infinitely constrained”; “robust” usually denotes worst-case model uncertainty and is misleading here.
strongest:
- 2305.00254: uses “semi-infinitely constrained MDP” for a continuum of expected constraints indexed by a compact set.
- 2405.02456: uses “constrained multi-task formulation” for a finite family with a separate constraint for each task.
what would change it: A contextual-RL paper explicitly standardizing “contextwise” or “pointwise-in-context” constraints would provide a better field-specific label. “Statewise” should remain reserved for conditioning on states, and “robust” for worst-case uncertainty unless carefully qualified.

## Q3 — Which theory covers this?
verdict: Finite per-task duality is standard CMDP theory; the continuum of constraints is directly covered by SICMDP/semi-infinite programming, but the extra step from an infinite-dimensional dual to a neural function λψ(ω) is not supplied by the cited CMDP results.
strongest:
- 2305.00254: directly handles V_c_y^π(μ)≤u_y for every y in a compact continuum and supplies tabular and policy-optimization theory.
- 2105.10682: provides the closest RL precedent for a function-valued multiplier, λ(s), under statewise constraints.
what would change it: A theorem showing that the semi-infinite dual has an absolutely-continuous functional representative λ(ω), plus approximation and optimization guarantees for λψ, would close the theoretical gap. Otherwise the mathematically natural dual object may be more general than a pointwise neural function.

## Q4 — Has the moving-scalar problem been named?
verdict: not found — I found non-stationary CMDPs and a curriculum paper exhibiting the ingredients, but no established name or direct analysis of “a scalar Lagrange multiplier tracks the currently sampled context mixture, so changing q_k moves the constraint it represents.”
strongest:
- 2201.11965: explicitly studies primal-dual learning when objectives, constraints, and dynamics change over time, but its non-stationarity is in the CMDP sequence rather than a curriculum-induced mixture over observed contexts.
- 2007.03964: diagnoses undesirable scalar-Lagrange dynamics such as oscillation and overshoot, but not distribution-shift-induced movement of the constraint target.
what would change it: A curriculum/domain-randomization Lagrangian paper that derives λ_{k+1} from E_{ω∼q_k}C and discusses q_k drift as dual-target drift would answer this directly. “Safety-Prioritizing Curricula for Constrained Reinforcement Learning” is highly relevant, but no arXiv ID or DOI was found, so the hard ID rule prevents a CSV row.

## Q5 — Which adjacent constructs differ?
verdict: The nearest constructs each coincide on one axis but differ on another: FAC indexes the multiplier by state; CCPO indexes behavior/duals by budget threshold; robust CMDPs optimize worst-case unobserved models; SCARED indexes adaptation episodes/history; SPiDR retains an average-over-domains constraint.
strongest:
- 2105.10682: λ(s) is a genuine function-valued multiplier, but the quantified variable is initial state rather than environment context.
- 2310.03718: has per-threshold feasible problems and λ_i* while the policy observes ε, but ε changes the desired budget rather than the environment.
what would change it: A method that simply replaces FAC's s with an exogenous observed ω while retaining an expectation over stochastic s0|ω would become extremely close. Likewise, a CCPO-style method conditioning λ as well as π on environment parameters rather than thresholds would materially narrow the remaining gap.

## Q6 — Has the empirical pattern appeared?
verdict: not found in the exact per-context form; closely analogous evidence exists for statewise safety, while SPiDR demonstrates an average-domain training constraint that is safe in simulation yet unsafe in deployment rather than memberwise violations inside the same evaluated family.
strongest:
- 2105.10682: explicitly motivates statewise safety by noting that an expectation-safe policy can be unsafe at individual states.
- 2509.18648: shows standard domain randomization can satisfy its averaged simulation constraint while substantially violating the deployment constraint, and identifies constraint underestimation.
what would change it: A plot/table reporting costs separately for task/context members while an aggregate baseline lies below d would answer Q6 exactly. Multi-task safe-RL evaluations are the most promising place for a second snowballing pass.

## Naming
Recommendation: call the thesis constraint a “per-context expected-cost constraint” or “contextwise expected-cost constraint”; when Ω is continuous, describe the mathematical program as a “semi-infinitely constrained contextual CMDP.” This follows the per-task semantics of 2405.02456 and the continuum terminology of 2305.00254 while avoiding the worst-case, generally unobserved-model meaning of “robust” in 2010.10644 and 2209.06866.

## Search log
- 2026-11-04 — ("constrained" OR "safe") AND ("multi-task" OR "multi-goal" OR "goal-conditioned") AND ("reinforcement learning") AND (Lagrang* OR "dual variable" OR multiplier)
- 2026-11-04 — ("safe" OR "constrained") AND ("meta-reinforcement learning" OR "meta-RL" OR "in-context")
- 2026-11-04 — ("constrained" OR "safe") AND ("contextual MDP" OR "contextual Markov" OR "contextual bandit")
- 2026-11-04 — ("constrained" OR "safe") AND ("domain randomization" OR "sim-to-real") AND (Lagrang* OR multiplier)
- 2026-11-04 — ("Lagrange multiplier" OR Lagrangian) AND (curriculum OR "non-stationary" OR "continual" OR "changing dynamics") AND ("constrained" OR "safe") AND "reinforcement learning"
- 2026-11-04 — ("statewise" OR "state-wise" OR "state-dependent") AND ("Lagrange multiplier" OR Lagrangian) AND "reinforcement learning"
- 2026-11-04 — ("multi-constraint" OR "multiple constraints" OR "semi-infinite") AND ("constrained Markov decision" OR CMDP)
- 2026-11-04 — ("constraint-conditioned" OR "threshold-conditioned" OR "cost-conditioned" OR "versatile safe") AND "reinforcement learning"
- 2026-11-04 — ("multi-objective reinforcement learning") AND (conditioned OR preference) AND (constraint OR safe)
- 2026-11-04 — "constrained contextual MDP" reinforcement learning 2024 Lin arXiv
- 2026-11-04 — "per-task" Lagrange multiplier safe reinforcement learning multi-task arXiv
- 2026-11-04 — "task-specific" "Lagrange multiplier" meta reinforcement learning safety
- 2026-11-04 — "Provably Efficient Primal-Dual Reinforcement Learning for CMDPs with Non-stationary Objectives and Constraints" arxiv
- 2026-11-04 — site:arxiv.org safe multi-task reinforcement learning "dual variables" task constraints Lagrangian
- 2026-11-04 — site:arxiv.org constrained multi-task reinforcement learning "Lagrange" task-specific constraints
- 2026-11-04 — site:arxiv.org safe meta reinforcement learning "task-specific" constraints Lagrangian multiplier
- 2026-11-04 — site:arxiv.org context-conditioned Lagrange multiplier constrained reinforcement learning context task
- 2026-11-04 — forward/backward screening from 2105.10682, 2310.03718, 2010.10644, 2209.06866, 2509.25582, 2007.03964, 2509.18648, and Altman's CMDP work.
- Tooling exposed executed web-search strings but did not expose an actual Google Scholar interface; searches used the available web indexes plus arXiv/OpenReview/venue pages.
```

The most damaging prior art for a broad novelty claim is **Zeng, Doan & Romberg (2405.02456)**: its mathematical program is explicitly “average objective across tasks, subject to a bound in each task,” its Lagrangian carries a separate dual coordinate for every task, and its sampled actor-critic performs local dual updates per task. The remaining differences are substantive but narrower: finite tasks rather than a continuous observed context, a common policy not explicitly conditioned on task identity, and a stored dual vector rather than a learned λψ(ω). citeturn502436view0turn383898view0

**C-MAML (2406.14047)** independently removes much of the conceptual novelty of “task-specific multiplier”: its method writes a task-specific constrained problem and states explicitly that λ_p is used for each task. Its outer loop, however, averages constraint violations over the task distribution, so it also cleanly illustrates the distinction the thesis wants to make. citeturn153980view0

For a continuous Ω, the unexpectedly close theoretical vocabulary is **semi-infinite**, not robust. SICMDP already formalizes a continuum \(V_{c_y}^\pi(\mu)\le u_y,\forall y\in Y\) and converts the known-model case to linear semi-infinite programming. But its \(y\) is a constraint index, not necessarily an observed policy context, and its algorithms do not train a neural dual function. citeturn400318view1turn400318view2 FAC supplies the complementary precedent: a genuinely indexed multiplier mapping λ(s), with one constraint per initial state. citeturn400318view0

The “robust” label is therefore especially risky: the robust-CMDP papers use worst-case uncertainty and scalar dualization rather than an observed-context expectation constrained separately for every context. citeturn153980view3turn623597view1

The weakest lane was Q6. I found strong statewise and sim-to-real analogues, but not a method-section/experiment combination showing exactly “aggregate budget satisfied, some observed context members violated.” SPiDR gets close empirically: it explicitly formulates standard domain randomization with an **average-over-domains** constraint and documents constraint underestimation and unsafe transfer. citeturn178406view2

With another hour, the highest-value target would be forward citations of 2405.02456 and its multi-task predecessors, plus papers citing SICMDP that introduce neural or parametric dual representations. citeturn400318view1