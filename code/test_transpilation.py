import sys
sys.path.insert(0, '.')
from qmaml_trainer import QiskitVQC_IBM, analyze_transpilation, get_backend_info
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import json

print("=" * 70)
print("TRANSPILATION TEST FOR IBM HARDWARE")
print("=" * 70)

# Test with simulator (no IBM credentials needed)
print("\n1. Creating VQC for IBM backend (simulator fallback)...")

vqc = QiskitVQC_IBM(
    n_qubits=4,
    n_layers=2,
    input_dim=4,
    backend_name="ibm_brisbane",
    optimize_for_hardware=True
)

print("\n2. Circuit info:")
info = vqc.get_circuit_info()
print(json.dumps(info, indent=2))

# Test transpilation analysis
print("\n3. Transpilation analysis:")
if hasattr(vqc, 'qc_transpiled'):
    analysis = analyze_transpilation(vqc.qc, vqc.qc_transpiled, "ibm_brisbane")
    print(json.dumps(analysis, indent=2))
else:
    print("No transpiled circuit (using simulator fallback)")

# Test backend info
print("\n4. Backend info (if available):")
backend_info = get_backend_info("ibm_brisbane")
print(json.dumps(backend_info, indent=2))

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
