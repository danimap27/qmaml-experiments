#!/bin/bash
# fix_hercules.sh — Fix Qiskit compatibility issues on Hercules
# Uses SAME versions as QTCL project

set -e

echo "=== QMAML Hercules Fix ==="

# Step 1: Completely remove the old environment
echo "Step 1: Removing old qmaml environment..."
conda deactivate 2>/dev/null || true
conda env remove -n qmaml -y 2>/dev/null || true

# Step 2: Clean conda cache
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

# Step 7: Install PyTorch first
echo "Step 7: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 8: Install Qiskit SAME versions as QTCL
echo "Step 8: Installing Qiskit (same versions as QTCL)..."
pip install qiskit>=1.2.0
pip install qiskit-machine-learning>=0.8.0
pip install qiskit-aer>=0.15.0
pip install qiskit-algorithms>=0.3.0
pip install qiskit-ibm-runtime

# Step 9: Install other dependencies (same as QTCL)
echo "Step 9: Installing other dependencies..."
pip install numpy>=1.26.0 scikit-learn>=1.4.0 scipy>=1.11.0 pandas>=2.2.0
pip install pyyaml>=6.0 tqdm>=4.66.0 matplotlib>=3.8.0 seaborn>=0.13.0 codecarbon>=2.4.0

# Step 10: Verify
echo ""
echo "Step 10: Verifying installation..."
python -c "
import qiskit
print(f'Qiskit: {qiskit.__version__}')
from qiskit_machine_learning.neural_networks import EstimatorQNN
print('EstimatorQNN: OK')
from qiskit.primitives import StatevectorEstimator
print('StatevectorEstimator: OK')
import torch
print(f'PyTorch: {torch.__version__}')
print('')
print('✅ All packages installed correctly!')
"

echo ""
echo "=== Fix complete ==="
echo ""
echo "To use:"
echo "  conda activate qmaml"
echo "  python run_qmaml_ibm.py"
