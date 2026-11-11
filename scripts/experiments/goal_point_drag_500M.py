"""Does a policy that meets the budget at the drags it trained on meet it at drags it never felt?

Step 3 of the plan (``docs/acl/README.md``): the baseline on the hidden factor. The drag is how
strongly the floor brakes the robot; the Lagrange function and the critics see it, no policy
does. The evaluation protocol ``drag`` trains on every hazard total at drag 0.65–0.95 and
1.1–1.5 and holds out 0.95–1.1, in the middle of the trained values (interpolation; it holds
the benchmark's own drag, 1), and 0.5–0.65, below every trained value (extrapolation): a floor
more slippery than any the policy trained on.

The same four learners, references, seeds and budget as the hazard-total baseline
(``scripts/experiments/generalization.py``). With a hidden factor ``told`` means told the hazard
counts and the goal radius only. Afterwards:

    .venv/bin/python -m training.score --group goal_point_drag_500M
    .venv/bin/python -m scripts.analysis.generalization_figures --groups goal_point_drag_500M

    .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_drag_500M.py --list
    setsid nohup .venv/bin/python -m training.experiments.local scripts/experiments/goal_point_drag_500M.py \\
        > runs/logs/goal_point_drag_500M_chain.log 2>&1 < /dev/null & disown
    .venv/bin/python -m training.experiments.snellius scripts/experiments/goal_point_drag_500M.py \\
        --partition gpu_a100 --time 03:00:00
"""
from scripts.experiments.generalization import baseline

EXPERIMENT = baseline("goal_point_drag_500M", protocol="drag")
