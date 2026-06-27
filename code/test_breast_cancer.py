import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
# Use Breast Cancer dataset
config['datasets'] = [{
    'name': 'breast_cancer',
    'root': './data/datasets',
    'n_tasks': 2,
    'data_type': 'tabular',
    'source': 'sklearn',
    'n_features': 30,
    'n_classes': 2,
    'n_samples': 569
}]
config['meta']['n_meta_train'] = 2
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='test_breast_cancer',
    architecture='classical_maml',
    seed=0,
    k_shot=5,
    study='main'
)

print('Creating trainer with Breast Cancer dataset...')
trainer = QMAMLTrainer(config, run)
print('Starting train...')
results = trainer.train()
print('Done!')
print(json.dumps(results, indent=2))
