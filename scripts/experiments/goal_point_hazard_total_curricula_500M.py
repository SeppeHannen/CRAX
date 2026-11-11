"""Does a curriculum change how a policy holds the budget on hazard totals it never saw?

Step 5 of the plan (``docs/acl/README.md``): Prioritized Level Replay on the training set and
the test sets of the hazard-total baseline. A curriculum here is only another way of sampling
the same training contexts (totals 0–10 and 14–16); it never draws a held-out total. The runs
are compared with the baseline's uniform runs of the same learner and seed
(``goal_point_hazard_total_500M``), which is why there is no uniform arm and no reference here.

Two scores, the two that held up on 2026-10-11 (``docs/acl/experiments/2026-10-11_goal_point_plr_scores_500M.md``):
the reward critic's error (the paper's) and the error of the Lagrangian objective the policy
optimises. The cost score is left out: it parked the policy at the budget and the multiplier
wound up. Two learners, the ends of the baseline: the published method (``scalar blind``) and
the full one (``additive told``). Four arms, three seeds, ~23 h on this machine.

    .venv/bin/python -m training.score --group goal_point_hazard_total_curricula_500M
    .venv/bin/python -m scripts.analysis.generalization_figures \\
        --groups goal_point_hazard_total_curricula_500M goal_point_hazard_total_500M

    .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_curricula_500M.py --list
    setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_curricula_500M.py \\
        > runs/logs/goal_point_hazard_total_curricula_500M_chain.log 2>&1 < /dev/null & disown
    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_hazard_total_curricula_500M.py \\
        --partition gpu_a100 --time 03:00:00
"""
from scripts.experiments.generalization import COMMON, LEARNERS, SEEDS
from training.experiments import Arm, Experiment

CURRICULA = {"plr": "plr", "plr_lagrangian": "plr:lagrangian"}

EXPERIMENT = Experiment(
    group="goal_point_hazard_total_curricula_500M",
    common=(*COMMON, "--evaluation_protocol", "hazard_total"),
    arms=tuple(
        Arm(f"{curriculum}_{learner}", ("--context_distribution", spec, *LEARNERS[learner]))
        for curriculum, spec in CURRICULA.items()
        for learner in ("scalar_blind", "additive_told")
    ),
    seeds=SEEDS,
)
