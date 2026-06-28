#!/bin/bash
# fix_hercules.sh — Fix Qiskit compatibility issues on Hercules
# WARNING: This completely removes and recreates the conda environment

set -e

echo "=== QMAML Hercules Fix ==="

# Step 1: Completely remove the old environment
echo "Step 1: Removing old qmaml environment..."
conda deactivate 2>/dev/null || true
conda env remove -n qmaml -y 2>/dev/null || true

# Step 2: Clean conda cache to avoid stale packages
echo "Step 2: Cleaning conda cache..."
conda clean --all -y 2>/dev/null || true

# Step 3: Create fresh environment with Python 3.10
echo "Step 3: Creating fresh conda environment 'qmaml' with Python 3.10..."
conda create -n qmaml python=3.10 -y

# Step 4: Activate
echo "Step 4: Activating environment..."
conda activate qmaml

echo "Python: $(python --version)"

# Step 5: Completely remove any residual pip packages
echo "Step 5: Removing any residual Qiskit packages..."
pip uninstall -y qiskit qiskit-terra qiskit-aer qiskit-ibm-runtime qiskit-machine-learning qiskit-algorithms 2>/dev/null || true

# Step 6: Upgrade pip
echo "Step 6: Upgrading pip..."
pip install --upgrade pip

# Step 7: Install PyTorch first (separately to avoid conflicts)
echo "Step 7: Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 8: Install Qiskit 0.46 and compatible packages ONLY
echo "Step 8: Installing Qiskit 0.46 and compatible packages..."
pip install qiskit==0.46.0
pip install qiskit-machine-learning==0.7.2
pip install qiskit-aer

# Step 9: Install other dependencies (NO qiskit-ibm-runtime!)
echo "Step 9: Installing other dependencies..."
pip install pyyaml numpy scikit-learn matplotlib seaborn scipy pandas

# Step 10: Verify
echo ""
echo "Step 10: Verifying installation..."
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
print('✅ All packages installed correctly!')
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
