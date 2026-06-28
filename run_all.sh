#!/bin/bash
# run_all.sh — Complete QMAML experiment runner for Hercules
# Updates code, fixes environment, and runs all experiments

set -e

echo "=========================================="
echo "QMAML Complete Runner for Hercules"
echo "=========================================="

# Step 1: Update code from GitHub
echo ""
echo "Step 1: Updating code from GitHub..."
git fetch origin
git reset --hard origin/master
echo "✅ Code updated"

# Step 2: Check Python version
echo ""
echo "Step 2: Checking Python version..."
PYTHON=""
for PY in python3.12 python3.11 python3.10 python3.9; do
    if command -v $PY &> /dev/null; then
        VER=$($PY --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON=$PY
            echo "Found Python: $PY ($VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "No Python 3.9+ found. Using conda..."
    if ! command -v conda &> /dev/null; then
        echo "ERROR: No Python 3.9+ and no conda available"
        exit 1
    fi
    
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    
    if ! conda env list | grep -q "^qmaml"; then
        echo "Creating conda environment 'qmaml' with Python 3.10..."
        conda create -n qmaml python=3.10 -y
    fi
    
    conda activate qmaml
    PYTHON=python
    echo "Using conda environment 'qmaml'"
fi

echo "Python: $($PYTHON --version)"

# Step 3: Check and fix Qiskit installation
echo ""
echo "Step 3: Checking Qiskit installation..."

QISKIT_OK=false
if $PYTHON -c "import qiskit; from qiskit.primitives import StatevectorEstimator; from qiskit_machine_learning.neural_networks import EstimatorQNN" 2>/dev/null; then
    QISKIT_VER=$($PYTHON -c "import qiskit; print(qiskit.__version__)" 2>/dev/null)
    echo "Qiskit found: $QISKIT_VER"
    
    # Check if version is 1.x
    if [[ "$QISKIT_VER" == 1.* ]]; then
        QISKIT_OK=true
        echo "✅ Qiskit version is compatible"
    else
        echo "⚠️ Qiskit version $QISKIT_VER is not 1.x, needs update"
    fi
else
    echo "⚠️ Qiskit not installed or incompatible"
fi

if [ "$QISKIT_OK" = false ]; then
    echo ""
    echo "Fixing Qiskit installation..."
    
    # Remove old packages
    echo "Removing old Qiskit packages..."
    pip uninstall -y qiskit qiskit-terra qiskit-aer qiskit-ibm-runtime qiskit-machine-learning qiskit-algorithms 2>/dev/null || true
    
    # Install correct versions (same as QTCL)
    echo "Installing Qiskit 1.2+..."
    pip install --upgrade pip
    pip install qiskit>=1.2.0
    pip install qiskit-machine-learning>=0.8.0
    pip install qiskit-aer>=0.15.0
    pip install qiskit-algorithms>=0.3.0
    pip install qiskit-ibm-runtime
    
    # Install other dependencies
    echo "Installing other dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install numpy>=1.26.0 scikit-learn>=1.4.0 scipy>=1.11.0 pandas>=2.2.0
    pip install pyyaml>=6.0 tqdm>=4.66.0 matplotlib>=3.8.0 seaborn>=0.13.0 codecarbon>=2.4.0
    
    echo ""
    echo "Verifying installation..."
    $PYTHON -c "
import qiskit
print(f'Qiskit: {qiskit.__version__}')
from qiskit.primitives import StatevectorEstimator
print('StatevectorEstimator: OK')
from qiskit_machine_learning.neural_networks import EstimatorQNN
print('EstimatorQNN: OK')
import torch
print(f'PyTorch: {torch.__version__}')
print('✅ All packages installed correctly!')
"
fi

# Step 4: Clean old results and checkpoints
echo ""
echo "Step 4: Cleaning old results..."
rm -rf results/*.json checkpoints/*.pkl
mkdir -p results checkpoints logs
echo "✅ Cleaned"

# Step 5: Run experiments
echo ""
echo "=========================================="
echo "Step 5: Running experiments..."
echo "=========================================="
echo ""

$PYTHON run_qmaml_ibm.py

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results: ./results/"
echo "Check progress: python check_progress.py"
