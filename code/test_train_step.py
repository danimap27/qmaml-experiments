import sys
sys.path.insert(0, '.')
import torch
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config
import json

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 1
config['meta']['inner_steps'] = 1
config['meta']['n_meta_test'] = 1

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

print('Saving params...')
original_params = {name: param.clone().detach() for name, param in trainer.model.named_parameters()}
print('Running inner loop...')
inner_optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
inner_optimizer.zero_grad()
support_logits = trainer.model(task.support_x)
support_loss = torch.nn.functional.cross_entropy(support_logits, task.support_y)
print(f'Support loss: {support_loss.item():.4f}')
support_loss.backward()
inner_optimizer.step()
print('Inner loop done')

print('Computing query loss...')
with torch.no_grad():
    query_logits = trainer.model(task.query_x)
    query_loss = torch.nn.functional.cross_entropy(query_logits, task.query_y)
    print(f'Query loss: {query_loss.item():.4f}')

print('Restoring params...')
with torch.no_grad():
    for name, param in trainer.model.named_parameters():
        param.copy_(original_params[name])
print('Params restored')

print('Running outer step...')
trainer.outer_optimizer.zero_grad()
support_logits_meta = trainer.model(task.support_x)
support_loss_meta = torch.nn.functional.cross_entropy(support_logits_meta, task.support_y)
support_loss_meta.backward()
trainer.outer_optimizer.step()
print('Outer step done')

print('All tests passed!')
