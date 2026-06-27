import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 2
config['meta']['inner_steps'] = 1

# Add noise
config['noise'] = {'depolarizing': 0.01}

run = RunConfig(
    run_id='test_qiskit_noisy',
    architecture='qmaml_qng',
    seed=42,
    k_shot=5,
    study='noise'
)

print('Creating trainer with Qiskit VQC (NOISY)...')
trainer = QMAMLTrainer(config, run)
print('Starting train...')
results = trainer.train()
print('Done!')
print(json.dumps(results, indent=2))
