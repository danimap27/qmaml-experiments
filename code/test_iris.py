import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
# Use Iris dataset
config['datasets'] = [{
    'name': 'iris',
    'root': './data/datasets',
    'n_tasks': 3,
    'data_type': 'tabular',
    'source': 'sklearn',
    'n_features': 4,
    'n_classes': 3,
    'n_samples': 150
}]
config['meta']['n_meta_train'] = 2
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='test_iris',
    architecture='classical_maml',
    seed=0,
    k_shot=5,
    study='main'
)

print('Creating trainer with Iris dataset...')
trainer = QMAMLTrainer(config, run)
print('Starting train...')
results = trainer.train()
print('Done!')
print(json.dumps(results, indent=2))
