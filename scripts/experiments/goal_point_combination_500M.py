"""Does a policy that has seen many hazards, and has seen a slippery floor, meet the budget when
it meets both at once?

Step 4 of the plan (``docs/acl/README.md``): the combinatorial test. The evaluation protocol
``combination`` trains on every hazard total and on every drag, but on hazard totals 11–20 only
at drag 0.9–1.5; totals 11–20 at drag 0.5–0.9 are held out. Nothing in the test is outside the
trained range of either factor; only the combination is new.

The same four learners, references, seeds and budget as the other two baselines
(``scripts/experiments/generalization.py``). Afterwards (the figure is a grid, hazard total by drag):

    .venv/bin/python -m training.score --group goal_point_combination_500M --episodes 32   # 210 cells at once
    .venv/bin/python -m scripts.analysis.generalization_figures --groups goal_point_combination_500M

    .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_combination_500M.py --list
    setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_combination_500M.py \\
        > runs/logs/goal_point_combination_500M_chain.log 2>&1 < /dev/null & disown
    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_combination_500M.py \\
        --partition gpu_a100 --time 03:00:00
"""
from scripts.experiments.generalization import baseline

EXPERIMENT = baseline("goal_point_combination_500M", protocol="combination")
