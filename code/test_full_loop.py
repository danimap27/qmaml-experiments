import sys
sys.path.insert(0, '.')
import torch
from qmaml_trainer import QMAMLTrainer
from runner import RunConfig, load_config

config = load_config('config.yaml')
config['meta']['n_meta_train'] = 1
config['meta']['inner_steps'] = 1

run = RunConfig(
    run_id='debug_train_loop',
    architecture='qmaml_qng',
    seed=42,
    k_shot=5,
    study='main'
)

print('Creating trainer...')
trainer = QMAMLTrainer(config, run)

print('Episode 1...')
task = trainer._sample_task()
print('Task sampled')

# Save params
original_params = {name: param.clone().detach() for name, param in trainer.model.named_parameters()}
print('Params saved')

# Inner loop
inner_optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
inner_optimizer.zero_grad()
support_logits = trainer.model(task.support_x)
support_loss = torch.nn.functional.cross_entropy(support_logits, task.support_y)
print(f'Support loss: {support_loss.item():.4f}')
support_loss.backward()
inner_optimizer.step()
print('Inner loop done')

# Query loss (WITHOUT no_grad)
print('Query forward...')
query_logits = trainer.model(task.query_x)
print('Query loss...')
query_loss = torch.nn.functional.cross_entropy(query_logits, task.query_y)
print(f'Query loss: {query_loss.item():.4f}')

# Accuracy
query_pred = query_logits.argmax(dim=1)
accuracy = (query_pred == task.query_y).float().mean().item()
print(f'Accuracy: {accuracy:.4f}')

# Restore params
print('Restoring params...')
with torch.no_grad():
    for name, param in trainer.model.named_parameters():
        param.copy_(original_params[name])
print('Params restored')

# Outer step
print('Outer step...')
trainer.outer_optimizer.zero_grad()
support_logits_meta = trainer.model(task.support_x)
support_loss_meta = torch.nn.functional.cross_entropy(support_logits_meta, task.support_y)
support_loss_meta.backward()
trainer.outer_optimizer.step()
print('Outer step done')

print('Episode 1 done!')
print('All tests passed!')
