#!/bin/bash
# setup_hercules.sh — Setup QMAML environment on Hercules

set -e

echo "=== QMAML Hercules Setup ==="

# Find the best Python available
PYTHON=""
for PY in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v $PY &> /dev/null; then
        PYTHON=$PY
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python 3.9+ found on system"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# Check Python version (need 3.9+)
PYVER=$($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.9" "$PYVER" | sort -V | head -n1)" != "3.9" ]; then
    echo "ERROR: Python $PYVER is too old. Need 3.9+ for PennyLane"
    exit 1
fi

# Try conda first, fallback to venv
if command -v conda &> /dev/null; then
    echo "Using conda..."
    module load Miniconda3 2>/dev/null || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "qmaml"; then
        echo "Creating conda environment 'qmaml'..."
        conda create -n qmaml python=$PYVER -y
    fi
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qmaml
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # Install compatible Qiskit versions
    pip install qiskit==0.46.0
    pip install qiskit-machine-learning==0.7.2
    pip install qiskit-aer
    pip install qiskit-ibm-runtime
    pip install numpy pandas pyyaml matplotlib seaborn scipy scikit-learn
    
else
    echo "Conda not found, using venv..."
    $PYTHON -m venv /home/quantum-nas/qmaml-venv
    source /home/quantum-nas/qmaml-venv/bin/activate
    
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install qiskit qiskit-machine-learning qiskit-algorithms
    pip install numpy pandas pyyaml matplotlib seaborn scipy scikit-learn
fi

# Verify installations
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"
python -c "import yaml; print(f'PyYAML: OK')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "To activate:"
if command -v conda &> /dev/null; then
    echo "  conda activate qmaml"
else
    echo "  source /home/quantum-nas/qmaml-venv/bin/activate"
fi
