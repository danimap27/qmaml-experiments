#!/bin/bash
# setup_clean.sh — Completely clean QMAML environment setup
# Removes old environment and creates fresh one from scratch

set -e

echo "=========================================="
echo "QMAML Clean Environment Setup"
echo "=========================================="

# Step 1: Deactivate and remove old conda environment
echo ""
echo "Step 1: Removing old conda environment 'qmaml'..."
conda deactivate 2>/dev/null || true
conda env remove -n qmaml -y 2>/dev/null || true
echo "✅ Old environment removed"

# Step 2: Clean conda cache
echo ""
echo "Step 2: Cleaning conda cache..."
conda clean --all -y 2>/dev/null || true
echo "✅ Cache cleaned"

# Step 3: Create fresh environment with Python 3.10
echo ""
echo "Step 3: Creating fresh conda environment 'qmaml' with Python 3.10..."
conda create -n qmaml python=3.10 -y
echo "✅ Environment created"

# Step 4: Activate environment
echo ""
echo "Step 4: Activating environment..."
eval "$(conda shell.bash hook)" 2>/dev/null || true
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate qmaml
echo "✅ Environment activated"

# Verify Python
PYTHON="python"
echo "Python: $($PYTHON --version)"

# Step 5: Upgrade pip
echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip
echo "✅ Pip upgraded"

# Step 6: Install PyTorch (CPU version for Hercules)
echo ""
echo "Step 6: Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "✅ PyTorch installed"

# Step 7: Install Qiskit 1.2+ (same as QTCL)
echo ""
echo "Step 7: Installing Qiskit 1.2+..."
pip install qiskit>=1.2.0
pip install qiskit-machine-learning>=0.8.0
pip install qiskit-aer>=0.15.0
pip install qiskit-algorithms>=0.3.0
pip install qiskit-ibm-runtime
echo "✅ Qiskit installed"

# Step 8: Install other dependencies
echo ""
echo "Step 8: Installing other dependencies..."
pip install numpy>=1.26.0
pip install scikit-learn>=1.4.0
pip install scipy>=1.11.0
pip install pandas>=2.2.0
pip install pyyaml>=6.0
pip install tqdm>=4.66.0
pip install matplotlib>=3.8.0
pip install seaborn>=0.13.0
pip install codecarbon>=2.4.0
echo "✅ Dependencies installed"

# Step 9: Verify installation
echo ""
echo "Step 9: Verifying installation..."
$PYTHON -c "
import qiskit
print(f'Qiskit: {qiskit.__version__}')
from qiskit.primitives import StatevectorEstimator
print('StatevectorEstimator: OK')
from qiskit_machine_learning.neural_networks import EstimatorQNN
print('EstimatorQNN: OK')
import torch
print(f'PyTorch: {torch.__version__}')
print()
print('✅ All packages installed correctly!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate:"
echo "  conda activate qmaml"
echo ""
echo "To run experiments:"
echo "  python manager.py"
echo ""
echo "Or directly:"
echo "  QMAML_MODE=ideal python run_qmaml_ibm.py"
echo "  QMAML_MODE=heron_r2 python run_qmaml_ibm.py"
echo "  QMAML_MODE=real IBM_BACKEND=ibm_fez python run_qmaml_ibm.py"
