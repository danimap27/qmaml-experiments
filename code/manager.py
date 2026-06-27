#!/usr/bin/env python3
"""
manager.py — QMAML Experiment Control Center & SLURM Monitor

Interactive menu to:
1. Refresh command lists from config.yaml.
2. Launch experiments in phases (Main, Noise, QFIM).
3. Monitor live progress by scanning results/ folders.
4. Generate LaTeX tables from aggregated JSON results.

Usage:
    python manager.py
"""

import os
import sys
import subprocess
import glob
import time
from pathlib import Path

# Always run from the directory where manager.py lives (code/)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CODE_DIR)

PYTHON  = sys.executable                          # full path to current python interpreter
RUNNER  = os.path.join(CODE_DIR, "runner.py")     # absolute path to runner.py
CONFIG  = os.path.join(CODE_DIR, "config.yaml")   # absolute path to config
from typing import Dict, List, Optional, Set, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(CODE_DIR, "results")

def _cmd_path(name: str) -> str:
    return os.path.join(CODE_DIR, name)

COMMAND_FILES = {
    "1": (_cmd_path("cmds_main.txt"),         "Phase 1: Main Experiments (Classical + QMAML)"),
    "2": (_cmd_path("cmds_noise.txt"),        "Phase 2: Noise Robustness Study"),
    "3": (_cmd_path("cmds_qfim.txt"),         "Phase 3: QFIM Spectrum Analysis"),
    "A": (_cmd_path("cmds_all.txt"),          "All Phases Combined"),
}

# Expected runs
# Main: 3 architectures × 2 k_shot × 3 seeds = 18
# Noise: 2 quantum architectures × 2 k_shot × 3 seeds × 5 noise levels = 60
# QFIM: 1 architecture × 3 seeds = 3
# Total = 81 expected runs
EXPECTED_RUNS  = 81
SLURM_PARTITION = "standard"   # Hercules CICA: standard (CPU) or gpu
CHUNK_SIZE     = 20            # Max concurrent array tasks per chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_arch_from_cmd(line: str) -> Optional[str]:
    parts = line.split()
    for i in range(len(parts) - 1):
        if parts[i] == "--arch":
            return parts[i+1].replace('"', '').replace("'", "")
    return None


def classify_arch_group(arch_name: str) -> str:
    """Classify architecture into 'classical', 'euclidean', or 'qng'."""
    if not arch_name:
        return "unknown"
    an = arch_name.lower()
    if "classical" in an:
        return "classical"
    if "euclidean" in an:
        return "euclidean"
    if "qng" in an:
        return "qng"
    return "unknown"


# ── Menu system ───────────────────────────────────────────────────────────────

def print_menu():
    print("\n" + "=" * 60)
    print("   QMAML Experiment Manager")
    print("=" * 60)
    print("  [1] Generate commands from config.yaml")
    print("  [2] Launch experiments (local or SLURM)")
    print("  [3] Monitor progress")
    print("  [4] Generate LaTeX tables")
    print("  [5] Check results completeness")
    print("  [0] Exit")
    print("-" * 60)


def get_choice() -> str:
    try:
        return input("Select option: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "0"


# ── 1. Generate commands ────────────────────────────────────────────────────

def generate_commands():
    print("\n--- Generating command lists ---")
    
    # Main experiments
    print("  → Phase 1: Main experiments")
    subprocess.run([PYTHON, RUNNER, "--config", CONFIG, "--study", "main", "--export-commands", _cmd_path("cmds_main.txt")], check=True)
    
    # Noise study
    print("  → Phase 2: Noise robustness")
    subprocess.run([PYTHON, RUNNER, "--config", CONFIG, "--study", "noise", "--export-commands", _cmd_path("cmds_noise.txt")], check=True)
    
    # QFIM analysis
    print("  → Phase 3: QFIM spectrum")
    subprocess.run([PYTHON, RUNNER, "--config", CONFIG, "--study", "qfim", "--export-commands", _cmd_path("cmds_qfim.txt")], check=True)
    
    # All combined
    print("  → All phases combined")
    subprocess.run([PYTHON, RUNNER, "--config", CONFIG, "--study", "all", "--export-commands", _cmd_path("cmds_all.txt")], check=True)
    
    print("\nDone. Command files created:")
    for key, (path, desc) in COMMAND_FILES.items():
        if os.path.exists(path):
            count = sum(1 for _ in open(path))
            print(f"  [{key}] {desc}: {count} commands")


# ── 2. Launch experiments ────────────────────────────────────────────────────

def launch_experiments():
    print("\n--- Launch experiments ---")
    print("Available phases:")
    for key, (path, desc) in COMMAND_FILES.items():
        if os.path.exists(path):
            count = sum(1 for _ in open(path))
            print(f"  [{key}] {desc} ({count} commands)")
    
    phase = input("\nSelect phase [1/2/3/A]: ").strip().upper()
    if phase not in COMMAND_FILES:
        print("Invalid selection.")
        return
    
    cmd_file, desc = COMMAND_FILES[phase]
    if not os.path.exists(cmd_file):
        print(f"Command file not found: {cmd_file}")
        print("Run 'Generate commands' first.")
        return
    
    print(f"\nSelected: {desc}")
    print("Launch mode:")
    print("  [L] Local execution (sequential)")
    print("  [P] Local parallel (multiprocessing)")
    print("  [S] SLURM array job (Hercules)")
    
    mode = input("Select mode [L/P/S]: ").strip().upper()
    
    if mode == "L":
        launch_local(cmd_file, parallel=1)
    elif mode == "P":
        n_workers = input("Number of workers [4]: ").strip()
        n_workers = int(n_workers) if n_workers else 4
        launch_local(cmd_file, parallel=n_workers)
    elif mode == "S":
        launch_slurm(cmd_file)
    else:
        print("Invalid mode.")


def launch_local(cmd_file: str, parallel: int = 1):
    """Launch experiments locally."""
    print(f"\nLaunching locally (parallel={parallel})...")
    
    with open(cmd_file) as f:
        commands = [line.strip() for line in f if line.strip()]
    
    if parallel > 1:
        # Use runner.py with --parallel
        subprocess.run([PYTHON, RUNNER, "--config", CONFIG, "--parallel", str(parallel), "--resume"])
    else:
        # Execute commands sequentially
        for cmd in commands:
            print(f"  Running: {cmd}")
            subprocess.run(cmd, shell=True)


def launch_slurm(cmd_file: str):
    """Launch SLURM array job on Hercules."""
    print("\n--- SLURM launch ---")
    
    with open(cmd_file) as f:
        commands = [line.strip() for line in f if line.strip()]
    
    n_commands = len(commands)
    print(f"Total commands: {n_commands}")
    
    # Split into chunks if necessary
    chunks = [commands[i:i+CHUNK_SIZE] for i in range(0, n_commands, CHUNK_SIZE)]
    print(f"Split into {len(chunks)} chunks (max {CHUNK_SIZE} per chunk)")
    
    for i, chunk in enumerate(chunks):
        chunk_file = f"{cmd_file}.chunk_{i+1}"
        with open(chunk_file, "w") as f:
            f.write("\n".join(chunk))
        
        # Create SLURM script
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=qmaml_{i+1}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-{len(chunk)}
#SBATCH --output=logs/qmaml_{i+1}_%A_%a.out
#SBATCH --error=logs/qmaml_{i+1}_%A_%a.err

# Load environment
module load Miniconda3
conda activate qmaml

# Get command for this array task
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {chunk_file})
echo "Running: $CMD"
eval $CMD
"""
        
        script_path = f"slurm_qmaml_chunk_{i+1}.sh"
        with open(script_path, "w") as f:
            f.write(slurm_script)
        
        print(f"  Chunk {i+1}: {len(chunk)} commands → {script_path}")
        
        # Submit
        submit = input(f"  Submit chunk {i+1}? [Y/n]: ").strip().lower()
        if submit in ("", "y", "yes"):
            os.makedirs("logs", exist_ok=True)
            result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
            print(f"    {result.stdout.strip()}")


# ── 3. Monitor progress ───────────────────────────────────────────────────────

def monitor_progress():
    print("\n--- Monitoring progress ---")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return
    
    # Count result files
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    n_completed = len(result_files)
    
    print(f"Completed runs: {n_completed} / {EXPECTED_RUNS}")
    print(f"Progress: {n_completed / EXPECTED_RUNS * 100:.1f}%")
    
    # Group by architecture
    arch_counts: Dict[str, int] = {}
    for f in result_files:
        basename = os.path.basename(f).replace(".json", "")
        parts = basename.split("_")
        if len(parts) >= 1:
            arch = parts[0]
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
    
    print("\nBy architecture:")
    for arch, count in sorted(arch_counts.items()):
        print(f"  {arch}: {count}")
    
    # Check for errors
    log_files = glob.glob(os.path.join(CODE_DIR, "logs", "*.err"))
    error_logs = [f for f in log_files if os.path.getsize(f) > 0]
    if error_logs:
        print(f"\nWarning: {len(error_logs)} jobs have error logs")


# ── 4. Generate LaTeX tables ──────────────────────────────────────────────────

def generate_tables():
    print("\n--- Generating LaTeX tables ---")
    
    if not HAS_PANDAS:
        print("pandas not installed. Install with: pip install pandas")
        return
    
    # Import and run table generator
    try:
        from generate_tables import main as generate_main
        generate_main()
    except ImportError:
        print("generate_tables.py not found. Creating basic table...")
        generate_basic_table()


def generate_basic_table():
    """Generate a basic LaTeX table from results."""
    import json
    
    results = []
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(f) as fh:
            data = json.load(fh)
            results.append(data)
    
    if not results:
        print("No results found.")
        return
    
    # Create simple table
    table = """\\begin{table}
\\centering
\\caption{QMAML Results Summary}
\\label{tab:qmaml_results}
\\begin{tabular}{llcc}
\\toprule
Architecture & k-shot & Accuracy & Convergence Epoch \\\\
\\midrule
"""
    
    for r in results:
        arch = r.get("architecture", "unknown")
        k_shot = r.get("k_shot", "-")
        acc = r.get("final_accuracy", "-")
        epoch = r.get("convergence_epoch", "-")
        table += f"{arch} & {k_shot} & {acc:.4f} & {epoch} \\\\\n"
    
    table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    output_path = os.path.join(CODE_DIR, "..", "tables", "results_table.tex")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)
    
    print(f"Table saved to: {output_path}")


# ── 5. Check completeness ───────────────────────────────────────────────────────

def check_completeness():
    print("\n--- Checking completeness ---")
    
    # Check which runs are missing
    expected_ids = set()
    for key, (path, desc) in COMMAND_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Extract run_id from command
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "--export-commands":
                                break
                        else:
                            # This is a run command, generate ID
                            arch = ""
                            seed = ""
                            k_shot = ""
                            for i, part in enumerate(parts):
                                if part == "--arch":
                                    arch = parts[i+1]
                                elif part == "--seed":
                                    seed = parts[i+1]
                                elif part == "--k_shot":
                                    k_shot = parts[i+1]
                            if arch and seed and k_shot:
                                expected_ids.add(f"{arch}_k{k_shot}_s{seed}")
    
    # Check existing results
    existing_ids = set()
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        basename = os.path.basename(f).replace(".json", "")
        existing_ids.add(basename)
    
    missing = expected_ids - existing_ids
    
    print(f"Expected runs: {len(expected_ids)}")
    print(f"Completed: {len(existing_ids)}")
    print(f"Missing: {len(missing)}")
    
    if missing:
        print("\nMissing runs:")
        for run_id in sorted(missing):
            print(f"  {run_id}")


# ── Main loop ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  QMAML Experiment Manager")
    print("  Quantum Model-Agnostic Meta-Learning")
    print("=" * 60)
    
    while True:
        print_menu()
        choice = get_choice()
        
        if choice == "1":
            generate_commands()
        elif choice == "2":
            launch_experiments()
        elif choice == "3":
            monitor_progress()
        elif choice == "4":
            generate_tables()
        elif choice == "5":
            check_completeness()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
