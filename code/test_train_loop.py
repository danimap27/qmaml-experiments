import sys
sys.path.insert(0, '.')
from qmaml_trainer import QMAMLTrainer, OmniglotTask, load_omniglot
from runner import RunConfig, load_config
import torch

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 1
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='debug_qng',
    architecture='qmaml_qng',
    seed=42,
    k_shot=5,
    study='main'
)

print('Creating trainer...')
trainer = QMAMLTrainer(config, run)
print('Sampling task...')
task = trainer._sample_task()
print(f'Task: support={task.support_x.shape}, query={task.query_x.shape}')

print('Testing inner loop...')
import copy
inner_model = trainer._clone_model_safe()
print('Model cloned')

inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.05)
print('Running inner step...')
inner_optimizer.zero_grad()
support_logits = inner_model(task.support_x)
support_loss = torch.nn.functional.cross_entropy(support_logits, task.support_y)
print(f'Support loss: {support_loss.item():.4f}')
support_loss.backward()
inner_optimizer.step()
print('Inner step done')

print('Testing query...')
query_logits = inner_model(task.query_x)
query_loss = torch.nn.functional.cross_entropy(query_logits, task.query_y)
print(f'Query loss: {query_loss.item():.4f}')

print('All tests passed!')
