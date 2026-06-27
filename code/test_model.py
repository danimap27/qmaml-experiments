import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLModel, VQC
import torch

print("Testing QMAMLModel creation...")

config = {
    "encoder": {"input_dim": 784, "hidden_dim": 64, "output_dim": 6},
    "vqc": {"n_qubits": 6, "n_shared_layers": 1, "n_task_layers": 1, "device": "lightning.qubit"},
    "classifier": {"input_dim": 6, "output_dim": 5}
}

try:
    model = QMAMLModel(config)
    print("Model created successfully")
    
    # Test forward pass
    x = torch.randn(2, 784)
    print("Testing forward pass...")
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Test clone
    print("Testing _clone_model_safe...")
    import copy
    new_model = QMAMLModel(config)
    with torch.no_grad():
        for new_param, old_param in zip(new_model.parameters(), model.parameters()):
            new_param.copy_(old_param)
    print("Clone successful")
    
    # Test inner loop
    print("Testing inner loop...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(5, 784)
    y = torch.randint(0, 5, (5,))
    
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    print("Inner loop successful")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
