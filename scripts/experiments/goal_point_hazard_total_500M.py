"""Does a policy that meets the budget on the hazard totals it trained on meet it on totals it never saw?

Step 1 of the plan (``docs/acl/README.md``): the baseline on the visible factor. Uniform
sampling, no curriculum. The evaluation protocol ``hazard_total`` holds two bands of the total
number of hazards out of training: 11–13, between trained totals (interpolation), and 17–20,
beyond them (extrapolation). Training samples totals 0–10 and 14–16; the drag is the benchmark's.

Four learners (``scripts/experiments/generalization.py``), each run twice: with the bands held
out, and as its **reference**, trained on every total. On the held-out bands cost is read against
the budget and return against the reference. Eight arms, three seeds.

Every run is scored during training on the benchmark's level 3 (total 20, held out), on all of
the protocol's contexts, on the training totals, and on each held-out band
(``evaluation/<name>/*``). Checkpoints are stored; cost and return per hazard total, for the
main figure, are measured afterwards:

    .venv/bin/python -m training.score --group goal_point_hazard_total_500M
    .venv/bin/python -m scripts.analysis.generalization_figures --groups goal_point_hazard_total_500M

The jobs run seed by seed, so the first eight are the whole comparison at seed 0:

    .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_500M.py --list
    # one after another on this machine (~2.6 h each with its evaluations: 21 h per seed, 62 h in all), detached:
    setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_hazard_total_500M.py \\
        > runs/logs/goal_point_hazard_total_500M_chain.log 2>&1 < /dev/null & disown
    # or one A100 per job on Snellius:
    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_hazard_total_500M.py \\
        --partition gpu_a100 --time 03:00:00
"""
from scripts.experiments.generalization import baseline

EXPERIMENT = baseline("goal_point_hazard_total_500M", protocol="hazard_total")
