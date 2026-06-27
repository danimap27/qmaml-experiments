#!/usr/bin/env python3
"""
runner.py — QMAML Experiment Orchestrator.

Generates all (architecture, seed, k_shot) combinations from config.yaml,
applies CLI filters, checks resumability, and executes sequentially.

Usage:
    python runner.py --config config.yaml --study main --export-commands cmds.txt
    python runner.py --config config.yaml --arch classical_maml --seed 0 --k_shot 1
"""

import argparse
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Check required dependencies
missing = []
try:
    import yaml
except ImportError:
    missing.append("pyyaml")

try:
    import torch
except ImportError:
    missing.append("torch")

try:
    from qiskit import QuantumCircuit
except ImportError:
    missing.append("qiskit")

try:
    from qiskit_machine_learning.neural_networks import EstimatorQNN
except ImportError:
    missing.append("qiskit-machine-learning")

if missing:
    print(f"ERROR: Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

_csv_lock = Lock()

_log_handler = logging.StreamHandler(sys.stdout)
if hasattr(_log_handler.stream, "reconfigure"):
    _log_handler.stream.reconfigure(encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Single experiment configuration."""
    run_id: str
    architecture: str
    seed: int
    k_shot: int
    study: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_commands(config: Dict[str, Any], study: str) -> List[RunConfig]:
    """Generate all experiment combinations."""
    commands = []
    seeds = config.get("seeds", [0])
    k_shots = config["meta"]["k_shot"]
    if isinstance(k_shots, int):
        k_shots = [k_shots]
    
    for arch in config["architectures"]:
        if study == "noise" and arch["type"] != "quantum":
            continue
        for k_shot in k_shots:
            for seed in seeds:
                run_id = f"{arch['name']}_k{k_shot}_s{seed}"
                commands.append(RunConfig(
                    run_id=run_id,
                    architecture=arch["name"],
                    seed=seed,
                    k_shot=k_shot,
                    study=study
                ))
    return commands


def execute_run(config: Dict[str, Any], run: RunConfig) -> Dict[str, Any]:
    """Execute a single experiment."""
    from qmaml_trainer import QMAMLTrainer
    
    logger.info(f"Starting: {run.run_id}")
    trainer = QMAMLTrainer(config, run)
    results = trainer.train()
    return results


def main():
    parser = argparse.ArgumentParser(description="QMAML Experiment Runner")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--study", default="main", choices=["main", "noise", "qfim"], help="Study type")
    parser.add_argument("--arch", help="Filter by architecture")
    parser.add_argument("--seed", type=int, help="Filter by seed")
    parser.add_argument("--k_shot", type=int, help="Filter by k_shot")
    parser.add_argument("--export-commands", help="Export commands to file")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    commands = generate_commands(config, args.study)
    
    # Apply filters
    if args.arch:
        commands = [c for c in commands if c.architecture == args.arch]
    if args.seed is not None:
        commands = [c for c in commands if c.seed == args.seed]
    if args.k_shot is not None:
        commands = [c for c in commands if c.k_shot == args.k_shot]
    
    logger.info(f"Generated {len(commands)} commands for study '{args.study}'")
    
    if args.export_commands:
        with open(args.export_commands, "w") as f:
            for cmd in commands:
                f.write(f"python runner.py --config {args.config} --study {cmd.study} --arch {cmd.architecture} --seed {cmd.seed} --k_shot {cmd.k_shot}\n")
        logger.info(f"Exported {len(commands)} commands to {args.export_commands}")
        return
    
    if args.dry_run:
        for cmd in commands:
            print(f"Would run: {cmd.run_id}")
        return
    
    # Execute runs
    for cmd in commands:
        try:
            results = execute_run(config, cmd)
            logger.info(f"Completed: {cmd.run_id}, Acc={results['final_accuracy']:.4f}")
        except Exception as e:
            logger.error(f"Failed: {cmd.run_id}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
