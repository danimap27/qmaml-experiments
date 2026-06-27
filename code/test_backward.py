import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLModel
import torch

config = {
    "encoder": {"input_dim": 784, "hidden_dim": 64, "output_dim": 4},
    "vqc": {"n_qubits": 4, "n_shared_layers": 1, "n_task_layers": 1, "device": "default.qubit"},
    "classifier": {"input_dim": 4, "output_dim": 5}
}

print("Creating model...")
model = QMAMLModel(config)
print("Model created")

x = torch.randn(5, 784)
y = torch.randint(0, 5, (5,))

print("Forward pass...")
out = model(x)
print(f"Output: {out.shape}")

print("Loss...")
loss = torch.nn.functional.cross_entropy(out, y)
print(f"Loss: {loss.item():.4f}")

print("Backward...")
loss.backward()
print("Backward done!")

print("All tests passed!")
