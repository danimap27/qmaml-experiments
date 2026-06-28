#!/bin/bash
# fix_hercules.sh — Fix Qiskit compatibility issues on Hercules
# Creates a completely fresh conda environment

set -e

echo "=== QMAML Hercules Fix ==="

# Remove old environment completely
echo "Removing old qmaml environment..."
conda deactivate 2>/dev/null || true
conda env remove -n qmaml -y 2>/dev/null || true

# Create fresh environment with Python 3.10
echo "Creating fresh conda environment 'qmaml' with Python 3.10..."
conda create -n qmaml python=3.10 -y

# Activate
echo "Activating environment..."
conda activate qmaml

echo "Python: $(python --version)"

# Install everything in correct order
echo "Installing dependencies..."
pip install --upgrade pip

# PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Qiskit 0.46 (old stable) and compatible packages
pip install qiskit==0.46.0
pip install qiskit-machine-learning==0.7.2
pip install qiskit-aer

# DO NOT install qiskit-ibm-runtime with qiskit 0.46
# They conflict. IBM runtime requires qiskit >=1.0

# Other dependencies
pip install pyyaml numpy scikit-learn matplotlib seaborn scipy pandas

echo ""
echo "Verifying installation..."
python -c "
import qiskit
print(f'Qiskit: {qiskit.__version__}')
from qiskit_machine_learning.neural_networks import EstimatorQNN
print('EstimatorQNN: OK')
from qiskit.primitives import Estimator
print('Estimator: OK')
import torch
print(f'PyTorch: {torch.__version__}')
print('')
print('NOTE: IBM Quantum hardware not available with qiskit 0.46')
print('Simulator mode will be used automatically')
"

echo ""
echo "=== Fix complete ==="
echo ""
echo "To use:"
echo "  conda activate qmaml"
echo "  python run_qmaml_ibm.py"
