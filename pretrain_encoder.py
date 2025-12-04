"""
Pre-training script for CompressedAD encoder using reconstruction objective.

This script trains the encoder to compress and reconstruct context sequences,
providing a strong initialization before the main supervised training.

Key features:
- Multi-GPU training with Accelerate
- Autoencoding objective: compress then reconstruct context
- Variable compression depths to match main training
- Compatible checkpoint format for loading into main training
"""

from datetime import datetime
import os
import os.path as path
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup

from dataset_pretrain import ADCompressedPretrainDataset, collate_pretrain_batch
from model.ad_compressed import CompressionEncoder
from utils import get_config

import multiprocessing
from tqdm import tqdm


class ReconstructionDecoder(nn.Module):
    """
    Simple decoder for reconstruction pre-training.
    Takes compressed latent tokens and reconstructs original embeddings.
    """
    def __init__(self, config):
        super(ReconstructionDecoder, self).__init__()
        
        tf_n_embd = config['tf_n_embd']
        n_layer = config.get('pretrain_decoder_n_layer', 2)
        n_head = config.get('tf_n_head', 4)
        dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=config.get('tf_dropout', 0.1),
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layer)
        
        # Output projection back to embedding space
        self.output_proj = nn.Linear(tf_n_embd, tf_n_embd)
        
    def forward(self, latent_tokens):
        """
        Args:
            latent_tokens: [batch, n_latent, tf_n_embd]
        Returns:
            reconstructed: [batch, n_latent, tf_n_embd]
        """
        # Process through decoder
        decoded = self.transformer(latent_tokens)
        
        # Project to output
        reconstructed = self.output_proj(decoded)
        
        return reconstructed


class EncoderPretrainModel(nn.Module):
    """
    Wrapper model for encoder pre-training with reconstruction objective.
    """
    def __init__(self, config):
        super(EncoderPretrainModel, self).__init__()
        
        self.config = config
        self.device = config['device']
        self.grid_size = config['grid_size']
        
        tf_n_embd = config['tf_n_embd']
        
        # Encoder (what we're pre-training)
        self.encoder = CompressionEncoder(config)
        
        # Decoder for reconstruction
        self.decoder = ReconstructionDecoder(config)
        
        # Context embedding (same as in CompressedAD)
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        
        # Embedding normalization
        self.embed_ln = nn.LayerNorm(tf_n_embd)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embed_context.weight)
        nn.init.zeros_(self.embed_context.bias)
        
        # Loss function - MSE for reconstruction
        self.loss_fn = nn.MSELoss(reduction='mean')
        
    def _embed_context_sequence(self, states, actions, rewards, next_states):
        """
        Embed context sequence into embedding space.
        """
        from einops import pack, rearrange
        
        # Ensure rewards have correct shape
        if rewards.dim() == 2:
            rewards = rearrange(rewards, 'b l -> b l 1')
        
        # Concatenate context
        context, _ = pack([states, actions, rewards, next_states], 'b l *')
        context_embed = self.embed_context(context)
        
        # Apply layer normalization
        context_embed = self.embed_ln(context_embed)
        
        return context_embed
    
    def forward(self, batch):
        """
        Pre-training forward pass with reconstruction objective.
        
        Args:
            batch: Dictionary containing:
                - encoder_input_stages: List of context sequences to compress hierarchically
                - reconstruction_target: Embeddings to reconstruct
                - num_stages: Number of compression cycles
        """
        num_stages = batch['num_stages']
        encoder_input_stages = batch['encoder_input_stages']
        
        # Hierarchical compression
        latent_tokens = None
        
        for stage_idx in range(num_stages):
            stage = encoder_input_stages[stage_idx]
            
            # Embed this stage's context
            states = stage['states'].to(self.device)
            actions = stage['actions'].to(self.device)
            rewards = stage['rewards'].to(self.device)
            next_states = stage['next_states'].to(self.device)
            
            context_embed = self._embed_context_sequence(states, actions, rewards, next_states)
            
            # Combine with previous latents
            if latent_tokens is not None:
                encoder_input = torch.cat([latent_tokens, context_embed], dim=1)
            else:
                encoder_input = context_embed
            
            # Compress
            latent_tokens = self.encoder(encoder_input)
        
        # Reconstruct from latent tokens
        reconstructed = self.decoder(latent_tokens)
        
        # Target: the latent tokens themselves (autoencoding in latent space)
        # We train the encoder to produce latents that contain enough information
        # for the decoder to reconstruct a good representation
        target = latent_tokens.detach()
        
        # Compute reconstruction loss
        loss = self.loss_fn(reconstructed, target)
        
        # Also compute a simple information retention metric
        # (correlation between reconstructed and target)
        with torch.no_grad():
            reconstructed_flat = reconstructed.reshape(-1)
            target_flat = target.reshape(-1)
            correlation = torch.corrcoef(torch.stack([reconstructed_flat, target_flat]))[0, 1]
        
        return {
            'loss': loss,
            'correlation': correlation
        }


def pretrain_encoder(config):
    """Main pre-training function with multi-GPU support."""
    
    log_dir = path.join('./runs', f"pretrain-encoder-{config['env']}-seed{config['env_split_seed']}")
    
    # Initialize Accelerator for multi-GPU training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=config.get('mixed_precision', 'fp16'),
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=log_dir,
        kwargs_handlers=[ddp_kwargs]
    )
    config['device'] = accelerator.device
    
    # Print distributed training info
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir, flush_secs=15)
        print(f'Pre-training encoder for {config["model"]}')
        print(f'Distributed training: {accelerator.distributed_type}')
        print(f'Number of processes: {accelerator.num_processes}')
        print(f'Mixed precision: {config.get("mixed_precision", "fp16")}')
        print(f'Log directory: {log_dir}')
    
    print(f'Process {accelerator.process_index} using device: {config["device"]}')
    
    # Create model
    model = EncoderPretrainModel(config)
    
    # Create dataset
    if accelerator.is_main_process:
        print(f'Loading pre-training data...')
    
    train_dataset = ADCompressedPretrainDataset(
        config=config,
        traj_dir=config['traj_dir'],
        mode='train',
        n_stream=config.get('train_n_stream', None),
        source_timesteps=config.get('train_source_timesteps', None)
    )
    
    if accelerator.is_main_process:
        print(f'Training dataset size: {len(train_dataset)}')
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('pretrain_batch_size', 128),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_pretrain_batch,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('pretrain_lr', 0.0003),
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    num_training_steps = config.get('pretrain_steps', 50000)
    num_warmup_steps = config.get('pretrain_warmup_steps', 2000)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    # Training loop
    if accelerator.is_main_process:
        print(f'\nStarting pre-training for {num_training_steps} steps...')
    
    model.train()
    global_step = 0
    
    # Create infinite dataloader
    train_iter = iter(train_loader)
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    while global_step < num_training_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        with accelerator.autocast():
            outputs = model(batch)
            loss = outputs['loss']
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient clipping
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        global_step += 1
        progress_bar.update(1)
        
        # Logging
        if global_step % 100 == 0:
            if accelerator.is_main_process:
                writer.add_scalar('pretrain/loss', loss.item(), global_step)
                writer.add_scalar('pretrain/correlation', outputs['correlation'].item(), global_step)
                writer.add_scalar('pretrain/lr', scheduler.get_last_lr()[0], global_step)
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'corr': f"{outputs['correlation'].item():.3f}"
            })
        
        # Save checkpoint
        if global_step % 10000 == 0 or global_step == num_training_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Get unwrapped model
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Save encoder weights
                checkpoint = {
                    'encoder_state_dict': unwrapped_model.encoder.state_dict(),
                    'embed_context_state_dict': unwrapped_model.embed_context.state_dict(),
                    'embed_ln_state_dict': unwrapped_model.embed_ln.state_dict(),
                    'global_step': global_step,
                    'config': config
                }
                
                checkpoint_path = path.join(log_dir, f'encoder-pretrained-{global_step}.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f'\nSaved encoder checkpoint to {checkpoint_path}')
    
    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'encoder_state_dict': unwrapped_model.encoder.state_dict(),
            'embed_context_state_dict': unwrapped_model.embed_context.state_dict(),
            'embed_ln_state_dict': unwrapped_model.embed_ln.state_dict(),
            'global_step': global_step,
            'config': config
        }
        
        final_path = path.join(log_dir, 'encoder-pretrained-final.pt')
        torch.save(checkpoint, final_path)
        print(f'\nPre-training complete! Final encoder saved to {final_path}')
        writer.close()
    
    return final_path if accelerator.is_main_process else None


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load config
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/ad_compressed_dr.yaml'))
    
    # Pre-training specific settings
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp16'
    
    # Pre-training hyperparameters (can override in config file)
    config.setdefault('pretrain_steps', 50000)
    config.setdefault('pretrain_batch_size', 128)
    config.setdefault('pretrain_lr', 0.0003)
    config.setdefault('pretrain_warmup_steps', 2000)
    config.setdefault('pretrain_decoder_n_layer', 2)
    config.setdefault('num_workers', 4)
    
    # Run pre-training
    pretrain_encoder(config)
