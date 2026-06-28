#!/bin/bash
#SBATCH --job-name=qmaml_complete
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/qmaml_%j.out
#SBATCH --error=logs/qmaml_%j.err

set -e

echo "=== QMAML Complete on Hercules ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

# Robust Python activation
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate qmaml 2>/dev/null || true
fi

cd /home/quantum-nas/papers/drafts/Papers/qmaml-experiments

# Run everything
bash run_all.sh

echo "Completed: $(date)"
