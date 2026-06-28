#!/bin/bash
# diagnose_hercules.sh — Diagnose git issues on Hercules

echo "=== Git Diagnosis ==="
echo ""
echo "1. Git status:"
git status
echo ""
echo "2. Current branch:"
git branch -vv
echo ""
echo "3. Remote branches:"
git branch -r
echo ""
echo "4. Latest local commit:"
git log --oneline -1
echo ""
echo "5. Latest remote commit:"
git log --oneline -1 origin/main 2>/dev/null || echo "No origin/main"
echo ""
echo "6. Files modified locally:"
git diff --name-only
echo ""
echo "7. Untracked files:"
git ls-files --others --exclude-standard
echo ""
echo "8. Python version:"
python3 --version 2>/dev/null || echo "python3 not found"
echo ""
echo "9. Qiskit version:"
python3 -c "import qiskit; print(qiskit.__version__)" 2>/dev/null || echo "qiskit not installed"
echo ""
echo "10. qiskit-machine-learning version:"
python3 -c "import qiskit_machine_learning; print(qiskit_machine_learning.__version__)" 2>/dev/null || echo "qiskit-machine-learning not installed"
