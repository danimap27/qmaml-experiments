import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 1
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='quick_test_qng',
    architecture='qmaml_qng',
    seed=42,
    k_shot=5,
    study='main'
)

print('Initializing QMAML-QNG trainer...')
trainer = QMAMLTrainer(config, run)
print('Starting train (1 episode)...')
results = trainer.train()
print(json.dumps(results, indent=2))
