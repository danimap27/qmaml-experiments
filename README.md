# QMAML: Quantum MAML for Real-World Medical Diagnosis

> **Teaching quantum computers to learn like doctors do** — by studying a few examples and quickly adapting to new patients.

This repository contains the official implementation of **QMAML (Quantum Model-Agnostic Meta-Learning)**, a hybrid quantum-classical approach to few-shot learning that runs on real IBM quantum hardware. We demonstrate it on breast cancer diagnosis, where the model learns to distinguish malignant from benign tumors using just a handful of examples.

---

## What is QMAML?

Imagine a doctor who has seen thousands of patients. When a new patient arrives with an unusual case, the doctor doesn't start from scratch — they adapt their existing knowledge based on just a few observations. That's exactly what **MAML** (Model-Agnostic Meta-Learning) does for machine learning.

Now imagine that doctor has access to a quantum computer that can explore multiple possibilities simultaneously. **QMAML** combines:

- **Classical neural networks** for feature extraction
- **Variational quantum circuits (VQCs)** for the learning adaptation step
- **IBM Quantum hardware** for real quantum execution

The quantum part uses the **Quantum Fisher Information Metric (QFIM)** to guide the learning process, potentially offering advantages in optimization landscapes that classical methods struggle with.

---

## The Breast Cancer Dataset

We use the classic **Wisconsin Breast Cancer Dataset** from scikit-learn:

| Property | Value |
|----------|-------|
| **Features** | 30 (cell nucleus measurements) |
| **Classes** | 2 (malignant / benign) |
| **Samples** | 569 |
| **Task** | Binary classification |

The challenge: train a model that can adapt to new classification tasks after seeing only **1 or 5 examples** (few-shot learning).

---

## Architecture

```
Input (30 features)
    ↓
Classical Encoder (MLP)
    ↓
Variational Quantum Circuit (6 qubits, IBM hardware)
    ↓
Classifier (2 outputs: malignant / benign)
```

### Three Variants Compared

| Variant | Inner Loop | Hardware |
|---------|-----------|----------|
| **Classical MAML** | SGD (classical) | CPU/GPU |
| **QMAML-Euclidean** | SGD (quantum params) | IBM Quantum |
| **QMAML-QNG** | SPSA + QFIM (quantum) | IBM Quantum |

---

## Key Features

### 🔄 Automatic Checkpointing

Experiments take hours on real quantum hardware. If your job gets interrupted (time limits, queue eviction, network issues), it **automatically resumes** from the last completed experiment.

```python
# First run
python run_qmaml_ibm.py
# → Completes 5/18 experiments, saves checkpoint

# Job interrupted (out of IBM minutes!)
# ... later, with fresh minutes ...

# Resume automatically
python run_qmaml_ibm.py
# → Loads checkpoint, starts from experiment 6/18
```

### 🚀 Batching for Efficient QPU Use

IBM's Heron processors have 156 qubits. Our circuits use only 6 qubits. We automatically batch multiple circuits into a single QPU execution:

| Circuit Size | Circuits per Batch | Speedup |
|-------------|-------------------|---------|
| 6 qubits | 26 | 26× |

### 📊 Live Progress Tracking

```bash
python check_progress.py
```

```
======================================================================
QMAML EXPERIMENT PROGRESS
======================================================================

✅ Completed experiments: 6/18
  classical_maml_k1_s0: Acc=0.5333, Time=146.2s
  classical_maml_k1_s42: Acc=0.6267, Time=90.5s
  classical_maml_k5_s123: Acc=0.7833, Time=21.7s

💾 Checkpoint: Experiment 6/18
🔄 Process running (PID: 746330)
======================================================================
```

---

## Quick Start

### Local Machine (Simulator)

```bash
# Install dependencies
pip install torch qiskit qiskit-machine-learning pyyaml numpy scikit-learn

# Run with simulator (fast, no IBM minutes needed)
cd qmaml-experiments
python run_qmaml_ibm.py
```

### IBM Quantum Hardware

```bash
# Set credentials (already configured in code for this repo)
export IBMQ_TOKEN="your-token"
export IBMQ_CRN="your-crn"

# Run on real quantum hardware
python run_qmaml_ibm.py
```

### HPC Cluster (SLURM)

```bash
# Submit to Hercules HPC
sbatch code/slurm_breast_cancer.sh

# Monitor
squeue -u $USER
tail -f logs/qmaml_*.out
```

---

## Results

### Classical MAML Baseline

| k-shot | Seed 0 | Seed 42 | Seed 123 | Average |
|--------|--------|---------|----------|---------|
| **1** | 0.533 | 0.627 | 0.553 | **0.571** |
| **5** | 0.537 | 0.533 | **0.783** | **0.618** |

### QMAML-Euclidean (IBM Quantum)

*Running on IBM Fez (156 qubits)*

| k-shot | Seed 0 | Seed 42 | Seed 123 | Average |
|--------|--------|---------|----------|---------|
| **1** | TBD | TBD | TBD | TBD |
| **5** | TBD | TBD | TBD | TBD |

### QMAML-QNG (IBM Quantum + SPSA)

*Running on IBM Fez (156 qubits)*

| k-shot | Seed 0 | Seed 42 | Seed 123 | Average |
|--------|--------|---------|----------|---------|
| **1** | TBD | TBD | TBD | TBD |
| **5** | TBD | TBD | TBD | TBD |

---

## Repository Structure

```
qmaml-experiments/
├── code/
│   ├── qmaml_trainer.py          # Core QMAML implementation
│   ├── runner.py                  # Experiment orchestrator
│   ├── manager.py                 # Interactive control center
│   ├── config.yaml                # Experiment configurations
│   ├── setup_hercules.sh          # HPC environment setup
│   ├── slurm_breast_cancer.sh   # SLURM job script
│   └── requirements.txt           # Python dependencies
├── run_qmaml_ibm.py               # Main experiment runner
├── check_progress.py              # Progress monitor
├── results/                       # Experiment outputs (JSON)
├── checkpoints/                   # Resume checkpoints
├── logs/                          # SLURM logs
├── README.md                      # This file
└── HERCULES.md                   # HPC-specific guide
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{martin2026qmaml,
  title={QMAML: Quantum Model-Agnostic Meta-Learning with the Quantum Fisher Information Metric},
  author={Martín-Pérez, Daniel and Gutiérrez-Avilés, David and Martínez-Álvarez, Francisco and Troncoso, Alicia},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Acknowledgments

- **IBM Quantum** for providing access to Heron R2 processors (156 qubits)
- **Qiskit** team for the quantum computing framework
- **PennyLane** inspiration for the original QML implementation
- **U.S. National Cancer Institute** for the Wisconsin Breast Cancer Dataset

---

## License

MIT License — see LICENSE file for details.

---

*Built with ❤️ and quantum superposition by the COGNAC DataLab at UPO.*
