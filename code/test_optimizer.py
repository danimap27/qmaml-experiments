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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(5, 784)
y = torch.randint(0, 5, (5,))

print("Training step...")
for i in range(3):
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}: loss={loss.item():.4f}")

print("All tests passed!")
