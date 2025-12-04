"""
Pre-training dataset for CompressedAD encoder.

Generates samples for autoencoding/reconstruction training by simulating
hierarchical compression on trajectory data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import random
from utils import get_traj_file_name


class ADCompressedPretrainDataset(Dataset):
    """
    Dataset for encoder pre-training with reconstruction objective.
    
    Generates compression samples at various depths (0-max_compression_depth)
    where the encoder learns to compress context sequences into latent representations
    that can be reconstructed.
    """
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_latent = config.get('n_latent', 60)
        self.max_compression_depth = config.get('max_compression_depth', 3)
        
        # Compression parameters
        self.min_compress_length = config.get('min_compress_length', 10)
        self.max_compress_length = config.get('max_compress_length', 50)
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        else:
            raise ValueError('Invalid env')
        
        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')
        
        states = []
        actions = []
        rewards = []
        next_states = []
        
        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
        
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        
        # Pre-compute valid samples
        self.samples = self._generate_sample_indices()
    
    def _generate_sample_indices(self):
        """
        Generate pre-training sample configurations.
        
        Each sample represents a compression scenario with:
        - compression_depth: number of compression cycles (1 to max_depth)
        - segment_lengths: lengths of segments to compress at each stage
        """
        samples = []
        
        n_histories = len(self.states)
        traj_length = self.states.shape[1]
        
        # Weight towards higher compression depths
        depth_weights = {1: 1, 2: 2, 3: 3}
        
        for history_idx in range(n_histories):
            # Generate samples with different compression depths
            # Note: depth 0 (no compression) not useful for pre-training
            for depth in range(1, self.max_compression_depth + 1):
                n_samples_for_depth = depth_weights.get(depth, 1) * 5
                
                for _ in range(n_samples_for_depth):
                    segment_lengths = []
                    total_length = 0
                    
                    # Generate compression stage lengths
                    for stage in range(depth):
                        seg_len = random.randint(self.min_compress_length, self.max_compress_length)
                        segment_lengths.append(seg_len)
                        total_length += seg_len
                    
                    # Check if valid
                    if total_length < traj_length - 5:
                        samples.append({
                            'history_idx': history_idx,
                            'compression_depth': depth,
                            'segment_lengths': segment_lengths,
                            'total_length': total_length
                        })
        
        random.shuffle(samples)
        print(f"\nGenerated {len(samples)} pre-training samples")
        depth_counts = {}
        for s in samples:
            d = s['compression_depth']
            depth_counts[d] = depth_counts.get(d, 0) + 1
        print(f"Samples per depth: {depth_counts}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _get_context_dict(self, history_idx, start_idx, length):
        """
        Extract context dictionary for a given segment.
        """
        end_idx = start_idx + length
        
        actions_data = self.actions[history_idx, start_idx:end_idx]
        
        # Check if actions are already one-hot encoded
        if len(actions_data.shape) == 1:
            num_actions = self.config.get('num_actions', 5)
            actions_onehot = np.zeros((length, num_actions), dtype=np.float32)
            actions_onehot[np.arange(length), actions_data.astype(int)] = 1.0
            actions_tensor = torch.tensor(actions_onehot, dtype=torch.float32)
        else:
            actions_tensor = torch.tensor(actions_data, dtype=torch.float32)
        
        return {
            'states': torch.tensor(self.states[history_idx, start_idx:end_idx], dtype=torch.float32),
            'actions': actions_tensor,
            'rewards': torch.tensor(self.rewards[history_idx, start_idx:end_idx], dtype=torch.float32),
            'next_states': torch.tensor(self.next_states[history_idx, start_idx:end_idx], dtype=torch.float32)
        }
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        history_idx = sample_info['history_idx']
        depth = sample_info['compression_depth']
        segment_lengths = sample_info['segment_lengths']
        
        # Build compression stages
        encoder_input_stages = []
        current_idx = 0
        
        for seg_len in segment_lengths:
            stage_dict = self._get_context_dict(history_idx, current_idx, seg_len)
            encoder_input_stages.append(stage_dict)
            current_idx += seg_len
        
        return {
            'encoder_input_stages': encoder_input_stages,
            'num_stages': depth
        }


def collate_pretrain_batch(batch):
    """
    Collate function for pre-training batches.
    
    Handles variable-depth compression by padding stages to match batch.
    """
    # Find max number of stages in this batch
    max_stages = max(item['num_stages'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize lists for each stage
    batched_stages = []
    
    for stage_idx in range(max_stages):
        # Collect all items that have this stage
        stage_states = []
        stage_actions = []
        stage_rewards = []
        stage_next_states = []
        
        max_len = 0
        
        # First pass: find max length for this stage
        for item in batch:
            if stage_idx < item['num_stages']:
                stage = item['encoder_input_stages'][stage_idx]
                max_len = max(max_len, stage['states'].shape[0])
        
        # If no items have this stage, use length 1 for dummy tensors
        if max_len == 0:
            max_len = 1
        
        # Second pass: pad and collect
        for item in batch:
            if stage_idx < item['num_stages']:
                stage = item['encoder_input_stages'][stage_idx]
                seq_len = stage['states'].shape[0]
                
                # Pad if needed
                if seq_len < max_len:
                    pad_len = max_len - seq_len
                    states = torch.cat([stage['states'], torch.zeros(pad_len, *stage['states'].shape[1:])], dim=0)
                    actions = torch.cat([stage['actions'], torch.zeros(pad_len, *stage['actions'].shape[1:])], dim=0)
                    rewards = torch.cat([stage['rewards'], torch.zeros(pad_len, *stage['rewards'].shape[1:])], dim=0)
                    next_states = torch.cat([stage['next_states'], torch.zeros(pad_len, *stage['next_states'].shape[1:])], dim=0)
                else:
                    states = stage['states']
                    actions = stage['actions']
                    rewards = stage['rewards']
                    next_states = stage['next_states']
                
                stage_states.append(states)
                stage_actions.append(actions)
                stage_rewards.append(rewards)
                stage_next_states.append(next_states)
            else:
                # This item doesn't have this stage, add dummy with correct max_len
                # Get shapes from any valid item in batch
                sample_item = next((b for b in batch if b['num_stages'] > 0), batch[0])
                sample_stage = sample_item['encoder_input_stages'][0]
                
                stage_states.append(torch.zeros(max_len, *sample_stage['states'].shape[1:]))
                stage_actions.append(torch.zeros(max_len, *sample_stage['actions'].shape[1:]))
                stage_rewards.append(torch.zeros(max_len, *sample_stage['rewards'].shape[1:]))
                stage_next_states.append(torch.zeros(max_len, *sample_stage['next_states'].shape[1:]))
        
        # Stack into batch
        batched_stages.append({
            'states': torch.stack(stage_states),
            'actions': torch.stack(stage_actions),
            'rewards': torch.stack(stage_rewards),
            'next_states': torch.stack(stage_next_states)
        })
    
    return {
        'encoder_input_stages': batched_stages,
        'num_stages': max_stages
    }
