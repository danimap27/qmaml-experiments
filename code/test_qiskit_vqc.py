import sys
sys.path.insert(0, '.')
from qmaml_trainer import QiskitVQC
import torch

print("Testing QiskitVQC...")

config = {
    "encoder": {"input_dim": 4, "hidden_dim": 64, "output_dim": 4},
    "vqc": {"n_qubits": 4, "n_shared_layers": 1, "n_task_layers": 1},
    "classifier": {"input_dim": 4, "output_dim": 3}
}

try:
    model = QiskitVQC(n_qubits=4, n_layers=2, input_dim=4)
    print("Model created successfully")
    
    x = torch.randn(2, 4)
    print("Testing forward pass...")
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")
    
    print("All tests passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
