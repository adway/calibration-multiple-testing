#!/bin/bash
#SBATCH --job-name=calibration_splines
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00

module load gcc/11.2.0

cd "$HOME/calibration-multiple-testing"

mkdir -p logs run_logs calibration_results

export PYTHONPATH="$HOME/calibration-multiple-testing/src:$PYTHONPATH"

echo "Project: $(pwd)"
echo "uv: $(which uv)"
echo "uv version:"
uv --version

echo "Syncing environment from pyproject.toml..."
uv sync

echo "Python version:"
uv run python --version

echo "Installed packages:"
uv run python -m pip list | head

echo "Python executable:"
uv run python -c "import sys; print(sys.executable)"

echo "Submitting submitit jobs..."

PYTHONPATH="$HOME/calibration-multiple-testing/src:$PYTHONPATH" \
uv run python -u src/calibration_sims.py

echo "Submission complete."
