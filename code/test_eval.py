import sys
sys.path.insert(0, '.')
import torch
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 1
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='debug_eval',
    architecture='qmaml_qng',
    seed=42,
    k_shot=5,
    study='main'
)

print('Creating trainer...')
trainer = QMAMLTrainer(config, run)
print('Running evaluate (5 tasks)...')
acc = trainer.evaluate(n_tasks=5)
print(f'Accuracy: {acc:.4f}')
print('Done!')
