import sys
sys.path.insert(0, '.')
from qmaml_trainer import QiskitVQC_IBM
import torch

print("=" * 70)
print("IBM HARDWARE CONNECTION TEST")
print("=" * 70)

# Create VQC with IBM backend
vqc = QiskitVQC_IBM(
    n_qubits=4,
    n_layers=2,
    input_dim=4,
    backend_name="ibm_fez",
    optimize_for_hardware=True,
    max_batch_size=4
)

print(f"\nConnected to: {vqc.backend_name}")
print(f"Total qubits: {vqc.total_qubits}")
print(f"Circuits per batch: {vqc.circuits_per_batch}")

# Test circuit info
info = vqc.get_circuit_info()
print(f"\nCircuit info:")
for key, value in info.items():
    print(f"  {key}: {value}")

# Test single execution
print(f"\nTesting single execution on IBM hardware...")
x = torch.randn(1, 4)
try:
    out = vqc(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output: {out}")
    print("  ✅ SUCCESS!")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
