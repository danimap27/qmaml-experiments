#!/bin/bash
#SBATCH --job-name=qmaml_noise
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-60%20
#SBATCH --output=logs/qmaml_noise_%A_%a.out
#SBATCH --error=logs/qmaml_noise_%A_%a.err

# Robust Python activation - tries multiple methods
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate qmaml 2>/dev/null || true
fi

if ! python -c "import torch" 2>/dev/null; then
    for PY in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v $PY &> /dev/null; then
            if $PY -c "import torch, pennylane, yaml" 2>/dev/null; then
                alias python="$PY"
                break
            fi
        fi
    done
fi

# Change to code directory
cd /home/quantum-nas/papers/drafts/Papers/qmaml-experiments/code

# Get command for this array task
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cmds_noise.txt)

# Run experiment
echo "[$(date)] Running: $CMD"
eval $CMD
echo "[$(date)] Completed"
