#!/usr/bin/env bash
# Build the Python environment every Snellius job uses, from the lock file next to
# this script. Run ON a Snellius login node, from any checkout of the repository:
#
#   bash scripts/snellius/install_environment.sh
#
# Idempotent: rerunning brings the environment to the lock and changes nothing else.
# The environment holds only the dependencies; the code a job runs is the per-commit
# checkout the job puts on PYTHONPATH (see job.sbatch), so one environment serves
# every commit that has the same dependencies.
set -euo pipefail

VENV="$HOME/crax-venv"
LOCK="$(cd "$(dirname "$0")" && pwd)/requirements.lock"

module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0

if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --requirement "$LOCK"

"$VENV/bin/python" - <<'EOF'
import jax, mujoco, wandb
print(f"python {__import__('sys').version.split()[0]}, jax {jax.__version__}, mujoco {mujoco.__version__}, wandb {wandb.__version__}")
EOF
echo "environment ready at $VENV"
