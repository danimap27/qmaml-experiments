# QMAML on Hercules HPC

## Quick Start

```bash
# 1. SSH to Hercules
ssh hercules@hercules.upo.es

# 2. Setup environment (once)
cd /home/quantum-nas/papers/drafts/Papers/qmaml-experiments/code
bash setup_hercules.sh

# 3. Submit job
sbatch slurm_breast_cancer.sh

# 4. Monitor
squeue -u $USER
tail -f logs/qmaml_*.out
```

## Job Scripts

| Script | Purpose | Time | Memory |
|--------|---------|------|--------|
| `slurm_breast_cancer.sh` | Main experiments (18 runs) | 24h | 16GB |
| `slurm_noise.sh` | Noise study (60 runs) | 24h | 16GB |
| `slurm_qfim.sh` | QFIM analysis (3 runs) | 12h | 16GB |

## Checkpointing

If job is interrupted (time limit, node failure), it automatically resumes from last checkpoint:

```bash
# Checkpoint is saved every experiment
./checkpoints/checkpoint.pkl

# Results saved incrementally
./results/*.json
```

## Output

```
results/
├── classical_maml_k1_s0.json
├── qmaml_qng_k5_s42.json
└── ... (18 files total)

checkpoints/
└── checkpoint.pkl

logs/
└── qmaml_12345.out
```

## IBM Quantum Credentials

The token is already configured in the code. No manual setup needed on Hercules.
