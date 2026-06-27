import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
# Use tabular dataset
config['datasets'] = [{
    'name': 'tabular_classification',
    'root': './data/datasets',
    'n_tasks': 5,
    'data_type': 'tabular',
    'n_samples': 500,
    'n_features': 10,
    'n_classes': 5
}]
config['meta']['n_meta_train'] = 2
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='test_tabular',
    architecture='classical_maml',
    seed=0,
    k_shot=5,
    study='main'
)

print('Creating trainer with tabular dataset...')
trainer = QMAMLTrainer(config, run)
print('Starting train...')
results = trainer.train()
print('Done!')
print(json.dumps(results, indent=2))
