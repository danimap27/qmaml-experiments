import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 2
config['meta']['inner_steps'] = 1

# Use Heron R2 noise model
config['noise'] = {'depolarizing': 0.01}  # Not used with heron_r2
config['hardware']['backend_type'] = 'simulator'

# Set noise model type in VQC config
for arch in config['architectures']:
    if arch['name'] == 'qmaml_qng':
        arch['vqc']['noise_model_type'] = 'heron_r2'

run = RunConfig(
    run_id='test_heron_r2',
    architecture='qmaml_qng',
    seed=42,
    k_shot=5,
    study='noise'
)

print('Creating trainer with Heron R2 noise model...')
trainer = QMAMLTrainer(config, run)
print('Starting train...')
results = trainer.train()
print('Done!')
print(json.dumps(results, indent=2))
