import sys
sys.path.insert(0, '.')
from qmaml_trainer import QiskitVQC_IBM
import torch

print("=" * 70)
print("BATCHING TEST FOR IBM HARDWARE")
print("=" * 70)

# Test with simulator (no IBM credentials needed)
print("\n1. Creating VQC with batching support...")

vqc = QiskitVQC_IBM(
    n_qubits=4,  # Small circuit
    n_layers=2,
    input_dim=4,
    backend_name="ibm_brisbane",
    optimize_for_hardware=True,
    max_batch_size=4
)

print(f"\n2. Circuit info:")
info = vqc.get_circuit_info()
for key, value in info.items():
    print(f"   {key}: {value}")

print(f"\n3. Testing single input (batch_size=1):")
x_single = torch.randn(1, 4)
out_single = vqc(x_single)
print(f"   Input shape: {x_single.shape}")
print(f"   Output shape: {out_single.shape}")

print(f"\n4. Testing batch input (batch_size=3):")
x_batch = torch.randn(3, 4)
out_batch = vqc(x_batch)
print(f"   Input shape: {x_batch.shape}")
print(f"   Output shape: {out_batch.shape}")

print(f"\n5. Testing large batch (batch_size=5):")
x_large = torch.randn(5, 4)
out_large = vqc(x_large)
print(f"   Input shape: {x_large.shape}")
print(f"   Output shape: {out_large.shape}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
