#!/bin/bash
#SBATCH --job-name=qmaml_breast_cancer
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/qmaml_%j.out
#SBATCH --error=logs/qmaml_%j.err

set -e

echo "=== QMAML Breast Cancer on Hercules ==="
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

# Fallback: find Python with dependencies
if ! python -c "import torch, qiskit, yaml" 2>/dev/null; then
    for PY in python3.12 python3.11 python3.10 python3; do
        if $PY -c "import torch, qiskit, yaml" 2>/dev/null; then
            alias python="$PY"
            break
        fi
    done
fi

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

cd /home/quantum-nas/papers/drafts/Papers/qmaml-experiments

# Run experiments with checkpointing
python run_qmaml_ibm.py

echo "Completed: $(date)"
