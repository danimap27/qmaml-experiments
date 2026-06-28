#!/usr/bin/env python3
"""
manager.py — QMAML Experiment Control Center

Interactive menu to manage three execution modes:
1. IDEAL: Simulator (no noise)
2. HERON R2: Simulator with realistic Heron R2 noise model
3. REAL: IBM Quantum hardware (requires token + CRN)

Usage:
    python manager.py
"""

import os
import sys
import json
import subprocess
import getpass
from pathlib import Path
from typing import Dict, List, Optional

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from runner import load_config

# ANSI colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

def print_menu():
    print_header("QMAML EXPERIMENT MANAGER")
    print(f"""
{Colors.CYAN}Select execution mode:{Colors.ENDC}

  {Colors.GREEN}[1]{Colors.ENDC} IDEAL      — Simulator (no noise, fastest)
  {Colors.GREEN}[2]{Colors.ENDC} HERON R2   — Simulator with Heron R2 noise model
  {Colors.GREEN}[3]{Colors.ENDC} REAL IBM   — IBM Quantum hardware (requires credentials)
  
  {Colors.GREEN}[4]{Colors.ENDC} Check progress
  {Colors.GREEN}[5]{Colors.ENDC} View results
  {Colors.GREEN}[6]{Colors.ENDC} Clean results
  {Colors.GREEN}[7]{Colors.ENDC} Install/Fix environment
  
  {Colors.GREEN}[0]{Colors.ENDC} Exit
""")

def ask_yes_no(question, default=True):
    """Ask yes/no question."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        response = input(f"{question}{suffix}").strip().lower()
        if not response:
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'")

def get_ibm_credentials():
    """Get IBM Quantum credentials from user."""
    print_header("IBM QUANTUM CREDENTIALS")
    print(f"{Colors.CYAN}Please enter your IBM Quantum credentials.{Colors.ENDC}")
    print(f"{Colors.WARNING}These will be saved to ~/.qiskit/qiskit-ibm.json{Colors.ENDC}\n")
    
    # Check if already saved
    token_file = Path.home() / ".qiskit" / "qiskit-ibm.json"
    if token_file.exists():
        with open(token_file) as f:
            saved = json.load(f)
        print(f"{Colors.GREEN}Saved credentials found:{Colors.ENDC}")
        print(f"  Token: {saved.get('token', 'N/A')[:20]}...")
        print(f"  CRN: {saved.get('instance', 'N/A')[:50]}...")
        if ask_yes_no("Use saved credentials?", default=True):
            return saved.get('token', ''), saved.get('instance', '')
    
    # Ask for token
    print(f"\n{Colors.CYAN}IBM Quantum Token:{Colors.ENDC}")
    print("  Get it from: https://quantum.ibm.com/")
    token = input("  Token: ").strip()
    
    if not token:
        print(f"{Colors.FAIL}ERROR: Token is required for IBM hardware{Colors.ENDC}")
        return None, None
    
    # Ask for CRN
    print(f"\n{Colors.CYAN}IBM Cloud CRN (Resource Name):{Colors.ENDC}")
    print("  Format: crn:v1:bluemix:public:quantum-computing:us-east:a/...")
    print("  Get it from: IBM Cloud → Resource List → Quantum Service")
    crn = input("  CRN: ").strip()
    
    if not crn:
        print(f"{Colors.FAIL}ERROR: CRN is required for IBM hardware{Colors.ENDC}")
        return None, None
    
    # Save credentials
    if ask_yes_no("Save credentials for future use?", default=True):
        os.makedirs(Path.home() / ".qiskit", exist_ok=True)
        with open(token_file, 'w') as f:
            json.dump({
                "channel": "ibm_cloud",
                "token": token,
                "instance": crn
            }, f, indent=2)
        print(f"{Colors.GREEN}✅ Credentials saved to {token_file}{Colors.ENDC}")
    
    return token, crn

def run_ideal():
    """Run experiments with ideal simulator (no noise)."""
    print_header("MODE: IDEAL SIMULATOR")
    print(f"{Colors.CYAN}Running with no noise — fastest mode{Colors.ENDC}\n")
    
    # Set environment variables
    os.environ['QMAML_MODE'] = 'ideal'
    os.environ['IBMQ_TOKEN'] = ''
    os.environ['IBMQ_CRN'] = ''
    
    # Modify config for ideal mode
    config = load_config("code/config.yaml")
    config['hardware']['backend_type'] = 'simulator'
    
    # Run
    print(f"{Colors.GREEN}Starting experiments...{Colors.ENDC}")
    subprocess.run([sys.executable, "run_qmaml_ibm.py"], check=False)

def run_heron_r2():
    """Run experiments with Heron R2 noise model."""
    print_header("MODE: HERON R2 NOISE MODEL")
    print(f"{Colors.CYAN}Running with realistic Heron R2 noise{Colors.ENDC}")
    print(f"{Colors.WARNING}This uses AerSimulator with noise model{Colors.ENDC}\n")
    
    # Set environment variables
    os.environ['QMAML_MODE'] = 'heron_r2'
    os.environ['IBMQ_TOKEN'] = ''
    os.environ['IBMQ_CRN'] = ''
    
    # Run
    print(f"{Colors.GREEN}Starting experiments...{Colors.ENDC}")
    subprocess.run([sys.executable, "run_qmaml_ibm.py"], check=False)

def run_real():
    """Run experiments on real IBM Quantum hardware."""
    print_header("MODE: REAL IBM QUANTUM HARDWARE")
    print(f"{Colors.CYAN}Running on actual IBM Quantum hardware{Colors.ENDC}")
    print(f"{Colors.WARNING}This uses IBM Quantum minutes{Colors.ENDC}\n")
    
    # Get credentials
    token, crn = get_ibm_credentials()
    if not token or not crn:
        print(f"{Colors.FAIL}Cannot run without credentials{Colors.ENDC}")
        return
    
    # Set environment variables
    os.environ['QMAML_MODE'] = 'real'
    os.environ['IBMQ_TOKEN'] = token
    os.environ['IBMQ_CRN'] = crn
    
    # Ask for backend
    print(f"\n{Colors.CYAN}Select IBM backend:{Colors.ENDC}")
    print("  [1] ibm_fez (156 qubits)")
    print("  [2] ibm_marrakesh (156 qubits)")
    print("  [3] ibm_kingston (156 qubits)")
    backend_choice = input("  Choice [1]: ").strip() or "1"
    
    backends = {
        "1": "ibm_fez",
        "2": "ibm_marrakesh",
        "3": "ibm_kingston"
    }
    backend = backends.get(backend_choice, "ibm_fez")
    os.environ['IBM_BACKEND'] = backend
    
    print(f"\n{Colors.GREEN}Using backend: {backend}{Colors.ENDC}")
    
    # Confirm
    if not ask_yes_no(f"This will use IBM Quantum minutes. Continue?", default=True):
        print(f"{Colors.WARNING}Cancelled{Colors.ENDC}")
        return
    
    # Run
    print(f"{Colors.GREEN}Starting experiments on {backend}...{Colors.ENDC}")
    subprocess.run([sys.executable, "run_qmaml_ibm.py"], check=False)

def check_progress():
    """Check experiment progress."""
    print_header("EXPERIMENT PROGRESS")
    
    if os.path.exists("check_progress.py"):
        subprocess.run([sys.executable, "check_progress.py"])
    else:
        print(f"{Colors.WARNING}check_progress.py not found{Colors.ENDC}")

def view_results():
    """View experiment results."""
    print_header("EXPERIMENT RESULTS")
    
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"{Colors.WARNING}No results directory found{Colors.ENDC}")
        return
    
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"{Colors.WARNING}No results yet{Colors.ENDC}")
        return
    
    print(f"{Colors.GREEN}Found {len(result_files)} result files:{Colors.ENDC}\n")
    
    for f in result_files:
        with open(f) as fp:
            data = json.load(fp)
        acc = data.get('final_accuracy', 0)
        time = data.get('elapsed_time', 0)
        print(f"  {f.stem:40s} Acc={acc:.4f}  Time={time:.1f}s")

def clean_results():
    """Clean old results and checkpoints."""
    print_header("CLEAN RESULTS")
    
    if not ask_yes_no("Delete all results and checkpoints?", default=False):
        print(f"{Colors.WARNING}Cancelled{Colors.ENDC}")
        return
    
    import shutil
    for dir_name in ['results', 'checkpoints', 'logs']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            os.makedirs(dir_name, exist_ok=True)
    
    print(f"{Colors.GREEN}✅ Cleaned all results{Colors.ENDC}")

def fix_environment():
    """Fix/install environment."""
    print_header("FIX ENVIRONMENT")
    
    if os.path.exists("code/fix_hercules.sh"):
        print(f"{Colors.CYAN}Running fix_hercules.sh...{Colors.ENDC}")
        subprocess.run(["bash", "code/fix_hercules.sh"])
    else:
        print(f"{Colors.WARNING}fix_hercules.sh not found{Colors.ENDC}")
        print("Running pip install...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "code/requirements.txt"])

def main():
    """Main interactive menu."""
    while True:
        print_menu()
        choice = input(f"{Colors.CYAN}Enter choice [0-7]: {Colors.ENDC}").strip()
        
        if choice == '1':
            run_ideal()
        elif choice == '2':
            run_heron_r2()
        elif choice == '3':
            run_real()
        elif choice == '4':
            check_progress()
        elif choice == '5':
            view_results()
        elif choice == '6':
            clean_results()
        elif choice == '7':
            fix_environment()
        elif choice == '0':
            print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}")
            break
        else:
            print(f"{Colors.FAIL}Invalid choice. Please enter 0-7.{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        sys.exit(0)
