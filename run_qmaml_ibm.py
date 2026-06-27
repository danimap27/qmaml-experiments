#!/usr/bin/env python3
"""
run_qmaml_ibm.py — Execute QMAML experiments on IBM Fez with checkpointing.

This script runs the full QMAML experiment suite from the paper:
- Classical MAML baseline
- QMAML-Euclidean (VQC + SGD inner loop)
- QMAML-QNG (VQC + SPSA inner loop)

With checkpointing to resume if interrupted (e.g., IBM minutes run out).
"""

import os
import sys
import json
import time
import pickle
from datetime import datetime
from pathlib import Path

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config

# Configuration
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
IBM_BACKEND = "ibm_fez"
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N episodes

# Experiment configurations from paper - using Breast Cancer dataset
EXPERIMENTS = [
    # Main experiments: Classical MAML
    {"run_id": "classical_maml_k1_s0", "arch": "classical_maml", "k_shot": 1, "seed": 0},
    {"run_id": "classical_maml_k1_s42", "arch": "classical_maml", "k_shot": 1, "seed": 42},
    {"run_id": "classical_maml_k1_s123", "arch": "classical_maml", "k_shot": 1, "seed": 123},
    {"run_id": "classical_maml_k5_s0", "arch": "classical_maml", "k_shot": 5, "seed": 0},
    {"run_id": "classical_maml_k5_s42", "arch": "classical_maml", "k_shot": 5, "seed": 42},
    {"run_id": "classical_maml_k5_s123", "arch": "classical_maml", "k_shot": 5, "seed": 123},
    
    # Main experiments: QMAML-Euclidean
    {"run_id": "qmaml_euclidean_k1_s0", "arch": "qmaml_euclidean", "k_shot": 1, "seed": 0},
    {"run_id": "qmaml_euclidean_k1_s42", "arch": "qmaml_euclidean", "k_shot": 1, "seed": 42},
    {"run_id": "qmaml_euclidean_k1_s123", "arch": "qmaml_euclidean", "k_shot": 1, "seed": 123},
    {"run_id": "qmaml_euclidean_k5_s0", "arch": "qmaml_euclidean", "k_shot": 5, "seed": 0},
    {"run_id": "qmaml_euclidean_k5_s42", "arch": "qmaml_euclidean", "k_shot": 5, "seed": 42},
    {"run_id": "qmaml_euclidean_k5_s123", "arch": "qmaml_euclidean", "k_shot": 5, "seed": 123},
    
    # Main experiments: QMAML-QNG
    {"run_id": "qmaml_qng_k1_s0", "arch": "qmaml_qng", "k_shot": 1, "seed": 0},
    {"run_id": "qmaml_qng_k1_s42", "arch": "qmaml_qng", "k_shot": 1, "seed": 42},
    {"run_id": "qmaml_qng_k1_s123", "arch": "qmaml_qng", "k_shot": 1, "seed": 123},
    {"run_id": "qmaml_qng_k5_s0", "arch": "qmaml_qng", "k_shot": 5, "seed": 0},
    {"run_id": "qmaml_qng_k5_s42", "arch": "qmaml_qng", "k_shot": 5, "seed": 42},
    {"run_id": "qmaml_qng_k5_s123", "arch": "qmaml_qng", "k_shot": 5, "seed": 123},
]


def save_checkpoint(experiment_idx, episode, results):
    """Save checkpoint to resume later."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "experiment_idx": experiment_idx,
        "episode": episode,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"💾 Checkpoint saved: {path}")


def load_checkpoint():
    """Load checkpoint if exists."""
    path = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"📂 Checkpoint loaded: {path}")
        print(f"   Resuming from experiment {checkpoint['experiment_idx']}, episode {checkpoint['episode']}")
        return checkpoint
    return None


def run_experiment(exp_config, base_config, checkpoint=None):
    """Run a single experiment with checkpoint support."""
    run_id = exp_config["run_id"]
    arch = exp_config["arch"]
    k_shot = exp_config["k_shot"]
    seed = exp_config["seed"]
    
    print(f"\n{'='*70}")
    print(f"Running: {run_id}")
    print(f"Architecture: {arch}, k_shot: {k_shot}, seed: {seed}")
    print(f"{'='*70}")
    
    # Configure for IBM hardware with Breast Cancer dataset
    config = base_config.copy()
    config["meta"]["n_meta_train"] = 2000
    config["meta"]["inner_steps"] = 5
    config["meta"]["k_shot"] = k_shot
    config["meta"]["n_way"] = 2  # Binary classification (malignant/benign)
    config["hardware"]["backend_type"] = "ibm"
    config["hardware"]["backend_name"] = IBM_BACKEND
    
    # Use Breast Cancer dataset
    config["datasets"] = [{
        "name": "breast_cancer",
        "root": "./data/datasets",
        "n_tasks": 2,
        "data_type": "tabular",
        "source": "sklearn",
        "description": "Breast Cancer Wisconsin (30 features, 2 classes)",
        "n_features": 30,
        "n_classes": 2,
        "n_samples": 569
    }]
    
    # Set seed
    config["seeds"] = [seed]
    
    run = RunConfig(
        run_id=run_id,
        architecture=arch,
        seed=seed,
        k_shot=k_shot,
        study="main"
    )
    
    # Create trainer
    trainer = QMAMLTrainer(config, run)
    
    # Run training
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    results["elapsed_time"] = elapsed
    results["backend"] = IBM_BACKEND
    
    print(f"\n✅ Completed: {run_id}")
    print(f"   Final accuracy: {results['final_accuracy']:.4f}")
    print(f"   Elapsed time: {elapsed:.1f}s")
    
    return results


def main():
    """Main execution with checkpointing."""
    print("="*70)
    print("QMAML EXPERIMENTS ON IBM QUANTUM")
    print("="*70)
    print(f"Backend: {IBM_BACKEND}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} episodes")
    print()
    
    # Load base config
    base_config = load_config("code/config.yaml")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    start_idx = 0
    all_results = []
    
    if checkpoint:
        start_idx = checkpoint["experiment_idx"]
        all_results = checkpoint["results"]
        print(f"Resuming from experiment {start_idx}/{len(EXPERIMENTS)}")
    
    # Run experiments
    for i, exp_config in enumerate(EXPERIMENTS[start_idx:], start=start_idx):
        try:
            results = run_experiment(exp_config, base_config, checkpoint)
            all_results.append(results)
            
            # Save checkpoint after each experiment
            save_checkpoint(i + 1, 0, all_results)
            
            # Save individual result
            os.makedirs(RESULTS_DIR, exist_ok=True)
            result_path = os.path.join(RESULTS_DIR, f"{exp_config['run_id']}.json")
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Error in {exp_config['run_id']}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save checkpoint on error
            save_checkpoint(i, 0, all_results)
            print(f"Checkpoint saved. You can resume when IBM minutes are available.")
            raise
    
    # Save final results
    final_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")
    
    # Print summary
    print("\nSummary:")
    for r in all_results:
        print(f"  {r['run_id']}: Acc={r['final_accuracy']:.4f}, Time={r.get('elapsed_time', 0):.1f}s")


if __name__ == "__main__":
    main()
