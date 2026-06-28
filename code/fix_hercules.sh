#!/bin/bash
# fix_hercules.sh — Fix Qiskit compatibility issues on Hercules

set -e

echo "=== QMAML Hercules Fix ==="

# Find Python 3.9+ or use conda
PYTHON=""
for PY in python3.12 python3.11 python3.10 python3.9; do
    if command -v $PY &> /dev/null; then
        VER=$($PY --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON=$PY
            echo "Found compatible Python: $PY ($VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "No Python 3.9+ found. Trying conda..."
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
        if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
        fi
        conda activate qmaml 2>/dev/null || conda create -n qmaml python=3.10 -y
        conda activate qmaml
        PYTHON=python
        echo "Using conda environment 'qmaml'"
    else
        echo "ERROR: No Python 3.9+ and no conda available"
        exit 1
    fi
fi

echo "Python: $($PYTHON --version)"

# Uninstall conflicting versions
echo "Removing old Qiskit versions..."
pip uninstall -y qiskit qiskit-machine-learning qiskit-aer qiskit-ibm-runtime 2>/dev/null || true

# Install compatible versions
echo "Installing compatible Qiskit versions..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install qiskit==0.46.0
pip install qiskit-machine-learning==0.7.2
pip install qiskit-aer
pip install qiskit-ibm-runtime
pip install pyyaml numpy scikit-learn

echo "Verifying installation..."
$PYTHON -c "
import qiskit
print(f'Qiskit: {qiskit.__version__}')
from qiskit_machine_learning.neural_networks import EstimatorQNN
print('EstimatorQNN: OK')
from qiskit.primitives import Estimator
print('Estimator: OK')
"

echo "=== Fix complete ==="
