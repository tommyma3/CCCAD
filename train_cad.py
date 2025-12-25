"""
Fine-tuning script for Compressed Algorithm Distillation (CAD).

This script fine-tunes the full CAD system (compression + AD) after pre-training
the compression transformer. Supports multi-GPU training and curriculum learning.

Usage:
    accelerate launch train_cad.py
    
For multi-GPU:
    accelerate launch --multi_gpu --num_processes=N train_cad.py
    
With config:
    accelerate config  # First time setup
    accelerate launch train_cad.py
"""

from datetime import datetime
import os
import os.path as path
from glob import glob
import argparse

from accelerate import Accelerator
from accelerate.utils import set_seed

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import CompressedADDataset, ADDataset
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_env
import numpy as np
import torch.nn.functional as F
from functools import partial


def cad_collate_fn(batch, grid_size):
    """
    Collate function for variable-length CAD dataset.
    Handles sequences of different lengths by padding.
    """
    # Find max context length in batch
    max_context_len = max(item['states'].shape[0] for item in batch)
    
    batch_size = len(batch)
    dim_state = batch[0]['states'].shape[1]
    num_actions = 5  # Darkroom actions
    
    # Initialize padded arrays
    states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    actions = np.zeros((batch_size, max_context_len), dtype=np.int64)
    rewards = np.zeros((batch_size, max_context_len), dtype=np.float32)
    next_states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    
    query_states = []
    target_actions = []
    context_lengths = []
    
    for i, item in enumerate(batch):
        ctx_len = item['states'].shape[0]
        states[i, :ctx_len] = item['states']
        actions[i, :ctx_len] = item['actions']
        rewards[i, :ctx_len] = item['rewards']
        next_states[i, :ctx_len] = item['next_states']
        
        query_states.append(item['query_states'])
        target_actions.append(item['target_actions'])
        context_lengths.append(ctx_len)
    
    res = {
        'query_states': torch.tensor(np.array(query_states), requires_grad=False, dtype=torch.float),
        'target_actions': torch.tensor(np.array(target_actions), requires_grad=False, dtype=torch.long),
        'states': torch.tensor(states, requires_grad=False, dtype=torch.float),
        'actions': F.one_hot(torch.tensor(actions, requires_grad=False, dtype=torch.long), num_classes=num_actions),
        'rewards': torch.tensor(rewards, dtype=torch.float, requires_grad=False),
        'next_states': torch.tensor(next_states, requires_grad=False, dtype=torch.float),
        'context_lengths': torch.tensor(context_lengths, dtype=torch.long),  # For masking
    }
    
    return res


def get_cad_data_loader(dataset, batch_size, config, shuffle=True):
    """Data loader for CAD with variable-length collate function."""
    from torch.utils.data import DataLoader
    
    collate_fn = partial(cad_collate_fn, grid_size=config['grid_size'])
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=config['num_workers'], 
        persistent_workers=True
    )


# Curriculum schedule: (step, max_compressions)
DEFAULT_CURRICULUM = [
    (0, 1),       # Start with max 1 compression
    (10000, 2),   # Allow 2 compressions
    (25000, 3),   # Allow 3 compressions
    (40000, None), # Unlimited
]


def get_curriculum_max_compressions(step, curriculum):
    """Get max compressions allowed at current step."""
    max_comp = curriculum[0][1]
    for threshold, comp in curriculum:
        if step >= threshold:
            max_comp = comp
    return max_comp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                       help='Path to pre-trained compression checkpoint')
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum learning')
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load configs
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/cad_dr.yaml'))

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    log_dir = path.join('./runs', f"CAD-{config['env']}-seed{config['env_split_seed']}")
    
    # Check if already exists
    config_save_path = path.join(log_dir, 'config.yaml')
    try:
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)
    
    config['log_dir'] = log_dir
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp32'

    # Curriculum settings
    use_curriculum = not args.no_curriculum
    curriculum = DEFAULT_CURRICULUM if use_curriculum else [(0, None)]

    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
    )
    
    config['device'] = accelerator.device
    
    # Only main process prints and logs
    is_main = accelerator.is_main_process
    
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        print(f'Using Device: {config["device"]}')
        print(f'Number of processes: {accelerator.num_processes}')
        print(f'Curriculum enabled: {use_curriculum}')

    # Create model
    model = MODEL[config['model']](config)

    # Load pre-trained compression if available
    if args.pretrain_ckpt:
        if is_main:
            print(f'Loading pre-trained compression from {args.pretrain_ckpt}')
        model.load_pretrained_compression(args.pretrain_ckpt)
    else:
        # Try to find pre-trained checkpoint automatically
        pretrain_dir = path.join('./runs', f"CAD-pretrain-{config['env']}-seed{config['env_split_seed']}")
        pretrain_path = path.join(pretrain_dir, 'pretrain-final.pt')
        if path.exists(pretrain_path):
            if is_main:
                print(f'Found pre-trained compression at {pretrain_path}')
            model.load_pretrained_compression(pretrain_path)
        elif is_main:
            print('WARNING: No pre-trained compression found. Training from scratch.')

    if is_main:
        load_start_time = datetime.now()
        print(f'Data loading started at {load_start_time}')

    # Create datasets
    train_dataset = CompressedADDataset(
        config, 
        config['traj_dir'], 
        'train', 
        config['train_n_stream'], 
        config['train_source_timesteps']
    )
    
    # Use standard AD dataset for testing (fixed length)
    test_dataset = ADDataset(
        config, 
        config['traj_dir'], 
        'test', 
        1, 
        config['train_source_timesteps']
    )

    train_dataloader = get_cad_data_loader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        config=config, 
        shuffle=True
    )
    train_dataloader = next_dataloader(train_dataloader)

    # Standard data loader for test
    from utils import get_data_loader
    test_dataloader = get_data_loader(
        test_dataset, 
        batch_size=config['test_batch_size'], 
        config=config, 
        shuffle=False
    )
    
    if is_main:
        load_end_time = datetime.now()
        print(f'Data loading ended at {load_end_time}')
        print(f'Elapsed time: {load_end_time - load_start_time}')

    # Optimizer for all parameters
    optimizer = AdamW(
        model.parameters(), 
        lr=config['lr'], 
        betas=(config['beta1'], config['beta2']), 
        weight_decay=config['weight_decay']
    )
    
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, 
        config['num_warmup_steps'], 
        config['train_timesteps']
    )
    
    step = 0

    # Load checkpoint if exists
    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=config['device'])
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        if is_main:
            print(f'Checkpoint loaded from {ckpt_path}')

    # Setup evaluation environments
    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args    

    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    else:
        raise NotImplementedError('Environment not supported')

    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )

    if is_main:
        start_time = datetime.now()
        print(f'Training started at {start_time}')

    # Training loop
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)

        while step < config['train_timesteps']:
            batch = next(train_dataloader)
            
            step += 1
            
            # Update curriculum
            if use_curriculum:
                max_comp = get_curriculum_max_compressions(step, curriculum)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.set_curriculum(max_comp)
            
            with accelerator.autocast():
                output = model(batch)
            
            # Use total loss (action + reconstruction regularization)
            loss = output['loss_total']

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(
                loss=loss.item(), 
                acc=output['acc_action'].item(),
                n_comp=output['num_compressions']
            )

            # Logging
            if is_main and step % config['summary_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'].item(), step)
                writer.add_scalar('train/loss_recon', output['loss_recon'].item(), step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)
                writer.add_scalar('train/num_compressions', output['num_compressions'], step)
                
                if use_curriculum:
                    curr_max = get_curriculum_max_compressions(step, curriculum)
                    writer.add_scalar('train/curriculum_max_compressions', 
                                    curr_max if curr_max is not None else -1, step)

            # Evaluation
            if is_main and step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'\nEvaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_cnt = 0

                    for j, test_batch in enumerate(test_dataloader):
                        output = model(test_batch)
                        cnt = len(test_batch['states'])
                        test_loss_action += output['loss_action'].item() * cnt
                        test_acc_action += output['acc_action'].item() * cnt
                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)

                eval_end_time = datetime.now()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()
                torch.cuda.empty_cache()

            # In-context evaluation (less frequent)
            if is_main and step % config['gen_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(model)
                    eval_output = unwrapped.evaluate_in_context(
                        vec_env=envs, 
                        eval_timesteps=config['horizon'] * 100
                    )
                    
                    mean_reward = eval_output['reward_episode'].mean()
                    total_compressions = eval_output['total_compressions']
                    
                    writer.add_scalar('eval/mean_reward', mean_reward, step)
                    writer.add_scalar('eval/total_compressions', total_compressions, step)
                    
                    print(f'\nIn-context eval: mean_reward={mean_reward:.3f}, compressions={total_compressions}')
                
                model.train()
                torch.cuda.empty_cache()

            pbar.update(1)

            # Save checkpoint
            if is_main and step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for old_ckpt_path in ckpt_paths:
                    os.remove(old_ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')
                
                # Get unwrapped model state dict
                unwrapped_model = accelerator.unwrap_model(model)
                
                torch.save({
                    'step': step,
                    'config': config,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')

    # Cleanup
    if is_main:
        writer.flush()
    
    envs.close()

    if is_main:
        end_time = datetime.now()
        print(f'\nTraining ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
