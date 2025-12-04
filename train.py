from datetime import datetime
import os
import os.path as path
from modulefinder import ModuleFinder
from glob import glob
import shutil

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset, ADCompressedDataset, collate_compressed_batch
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_env

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/ad_compressed_dr.yaml'))  # Use CompressedAD config

    log_dir = path.join('./runs', f"{config['model']}-{config['env']}-seed{config['env_split_seed']}")
    
    writer = SummaryWriter(log_dir, flush_secs=15)

    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)        

    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp16'

    # Initialize Accelerator for multi-GPU training
    # find_unused_parameters=True needed for CompressedAD because encoder isn't used when compression_depth=0
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=log_dir,
        kwargs_handlers=[ddp_kwargs]
    )
    config['device'] = accelerator.device
    
    # Print distributed training info
    if accelerator.is_main_process:
        print(f'Distributed training: {accelerator.distributed_type}')
        print(f'Number of processes: {accelerator.num_processes}')
        print(f'Mixed precision: {config["mixed_precision"]}')
    print(f'Process {accelerator.process_index} using device: {config["device"]}')

    model_name = config['model']
    model = MODEL[model_name](config)

    # Load pretrained encoder weights if specified
    pretrained_encoder_path = config.get('pretrained_encoder_path', None)
    if pretrained_encoder_path and model_name == 'CompressedAD':
        if path.exists(pretrained_encoder_path):
            if accelerator.is_main_process:
                print(f'\nLoading pretrained encoder from {pretrained_encoder_path}')
            
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            
            # Load encoder weights
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            # Load embedding weights (shared between encoder and decoder)
            model.embed_context.load_state_dict(checkpoint['embed_context_state_dict'])
            model.embed_ln.load_state_dict(checkpoint['embed_ln_state_dict'])
            
            if accelerator.is_main_process:
                print(f'Successfully loaded pretrained encoder from step {checkpoint.get("global_step", "unknown")}')
            
            # Optionally freeze encoder
            if config.get('freeze_encoder', False):
                if accelerator.is_main_process:
                    print('Freezing encoder and shared embedding weights')
                # Freeze encoder
                for param in model.encoder.parameters():
                    param.requires_grad = False
                # Freeze shared embeddings (they were trained with encoder during pre-training)
                for param in model.embed_context.parameters():
                    param.requires_grad = False
                for param in model.embed_ln.parameters():
                    param.requires_grad = False
        else:
            if accelerator.is_main_process:
                print(f'WARNING: Pretrained encoder path specified but not found: {pretrained_encoder_path}')

    load_start_time = datetime.now()
    print(f'Data loading started at {load_start_time}')

    # Use compressed dataset for CompressedAD model
    if model_name == 'CompressedAD':
        train_dataset = ADCompressedDataset(config, config['traj_dir'], 'train', config['train_n_stream'], config['train_source_timesteps'])
        test_dataset = ADCompressedDataset(config, config['traj_dir'], 'test', 1, config['train_source_timesteps'])
        
        # Use custom collate function and bucket sampler for compressed dataset
        from torch.utils.data import DataLoader
        from dataset import BucketSampler
        
        # BucketSampler groups samples by compression depth
        train_sampler = BucketSampler(
            train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True,
            drop_last=False
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,  # Use 0 workers with batch_sampler to avoid multiprocessing issues
            collate_fn=collate_compressed_batch,
            pin_memory=True
        )
        train_dataloader = next_dataloader(train_dataloader)
        
        test_sampler = BucketSampler(
            test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False,
            drop_last=False
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=0,  # Use 0 workers with batch_sampler to avoid multiprocessing issues
            collate_fn=collate_compressed_batch,
            pin_memory=True
        )
    else:
        # Standard AD dataset
        train_dataset = ADDataset(config, config['traj_dir'], 'train', config['train_n_stream'], config['train_source_timesteps'])
        test_dataset = ADDataset(config, config['traj_dir'], 'test', 1, config['train_source_timesteps'])

        train_dataloader = get_data_loader(train_dataset, batch_size=config['train_batch_size'], config=config, shuffle=True)
        train_dataloader = next_dataloader(train_dataloader)

        test_dataloader = get_data_loader(test_dataset, batch_size=config['test_batch_size'], config=config, shuffle=False)
    
    load_end_time = datetime.now()
    print()
    print(f'Data loading ended at {load_end_time}')
    print(f'Elapsed time: {load_end_time - load_start_time}')

    optimizer = AdamW(model.parameters(), lr = config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    lr_sched = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0

    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        print(f'Checkpoint loaded from {ckpt_path}')

    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args    

    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    else:
        raise NotImplementedError('Environment not supported')
    
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(model, optimizer, train_dataloader, lr_sched)

    start_time = datetime.now()
    if accelerator.is_main_process:
        print(f'Training started at {start_time}')

    # Only show progress bar on main process
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not accelerator.is_local_main_process) as pbar:
        pbar.update(step)

        while True:
            batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = model(batch)
            
            loss = output['loss_action']

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss=loss.item())

            # Logging only on main process
            if step % config['summary_interval'] == 0 and accelerator.is_main_process:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'], step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)
                if 'loss_reconstruct' in output:
                    writer.add_scalar('train/loss_reconstruct', output['loss_reconstruct'].item(), step)


            # Eval - synchronize all processes before evaluation
            if step % config['eval_interval'] == 0:
                accelerator.wait_for_everyone()  # Synchronize all processes
                torch.cuda.empty_cache()
                model.eval()
                if accelerator.is_main_process:
                    eval_start_time = datetime.now()
                    print(f'Evaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_loss_reward = 0.0
                    test_acc_reward = 0.0
                    test_loss_next_state = 0.0
                    test_acc_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        output = model(batch)
                        
                        # Handle both standard AD and CompressedAD batch formats
                        if 'states' in batch:
                            cnt = len(batch['states'])
                        else:
                            cnt = len(batch['query_states'])
                        
                        test_loss_action += output['loss_action'].item() * cnt
                        test_acc_action += output['acc_action'].item() * cnt
                            
                        if config['dynamics']:
                            test_loss_reward += output['loss_reward'].item() * cnt
                            test_acc_reward += output['acc_reward'].item() * cnt
                            test_loss_next_state += output['loss_next_state'].item() * cnt
                            test_acc_next_state += output['acc_next_state'].item() * cnt
                            
                        test_cnt += cnt

                # Only log on main process
                if accelerator.is_main_process:
                    writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                    writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)              

                    eval_end_time = datetime.now()
                    print()
                    print(f'Evaluating ended at {eval_end_time}')
                    print(f'Elapsed time: {eval_end_time - eval_start_time}')
                
                model.train()
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()  # Synchronize after evaluation

            pbar.update(1)

            # LOGGING - only save on main process
            if step % config['ckpt_interval'] == 0:
                accelerator.wait_for_everyone()  # Synchronize before saving
                
                if accelerator.is_main_process:
                    # Remove old checkpoints
                    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                    for ckpt_path in ckpt_paths:
                        os.remove(ckpt_path)

                    new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')

                    # Unwrap model to get original state dict
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    torch.save({
                        'step': step,
                        'config': config,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_sched': lr_sched.state_dict(),
                    }, new_ckpt_path)
                    print(f'\nCheckpoint saved to {new_ckpt_path}')
                
                accelerator.wait_for_everyone()  # Wait for checkpoint to be saved

            if step >= config['train_timesteps']:
                break

    # Final synchronization
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        writer.flush()
    
    envs.close()

    if accelerator.is_main_process:
        end_time = datetime.now()
        print()
        print(f'Training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')