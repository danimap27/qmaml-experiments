#!/usr/bin/env python3
"""
check_progress.py — Check QMAML experiment progress.

Usage:
    python check_progress.py
"""

import os
import json
from pathlib import Path

RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"

def main():
    print("=" * 70)
    print("QMAML EXPERIMENT PROGRESS")
    print("=" * 70)
    
    # Count completed experiments
    if os.path.exists(RESULTS_DIR):
        results = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
        print(f"\n✅ Completed experiments: {len(results)}")
        
        for result_file in results:
            path = os.path.join(RESULTS_DIR, result_file)
            with open(path) as f:
                data = json.load(f)
            print(f"  {data['run_id']}: Acc={data['final_accuracy']:.4f}, Time={data.get('elapsed_time', 0):.1f}s")
    else:
        print("\n⏳ No results yet")
    
    # Check checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        import pickle
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"\n💾 Checkpoint:")
        print(f"  Experiment: {checkpoint['experiment_idx']}/18")
        print(f"  Timestamp: {checkpoint['timestamp']}")
    
    # Check for running process
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'run_qmaml_ibm.py'], capture_output=True, text=True)
        if result.stdout.strip():
            print(f"\n🔄 Process running (PID: {result.stdout.strip()})")
        else:
            print(f"\n⏹️  No process running")
    except:
        pass
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
