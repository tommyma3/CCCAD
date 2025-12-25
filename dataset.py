"""
Datasets for Algorithm Distillation and Compressed Algorithm Distillation.

Contains:
1. ADDataset - Original AD dataset with fixed context length
2. CompressedADDataset - Variable-length dataset for CAD fine-tuning
3. CompressionPretrainDataset - Dataset for compression pre-training
"""

from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat


class ADDataset(Dataset):
    """Original AD dataset with fixed context length."""
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
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
    
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
            
        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1],
            })
        
        return traj


class CompressedADDataset(Dataset):
    """
    Dataset for Compressed AD fine-tuning with variable-length sequences.
    
    Samples sequences of varying lengths to ensure the model learns to handle:
    - Short sequences (no compression)
    - Medium sequences (1 compression)
    - Long sequences (2-3 compressions)
    - Very long sequences (4+ compressions)
    """
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']  # Max AD sequence length
        self.n_compress_tokens = config.get('n_compress_tokens', 40)
        self.dynamics = config['dynamics']
        
        # Context length distribution settings
        self.min_context = config.get('min_context_length', 50)
        self.max_context = config.get('max_context_length', 800)
        
        # Length distribution (can be configured)
        self.length_distribution = config.get('length_distribution', {
            'short': 0.2,      # No compression: 50-200
            'medium': 0.3,     # 1 compression: 250-400  
            'long': 0.3,       # 2-3 compressions: 450-700
            'very_long': 0.2,  # 4+ compressions: 750-1000
        })
        
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
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
    
    def __len__(self):
        # Approximate: we can sample many different windows
        return self.n_histories * (self.seq_length - self.min_context)
    
    def _sample_context_length(self):
        """Sample context length from distribution."""
        r = random.random()
        cumulative = 0
        
        for category, prob in self.length_distribution.items():
            cumulative += prob
            if r < cumulative:
                if category == 'short':
                    return random.randint(50, min(200, self.n_transit - 1))
                elif category == 'medium':
                    return random.randint(250, min(400, self.seq_length - 1))
                elif category == 'long':
                    return random.randint(450, min(700, self.seq_length - 1))
                else:  # very_long
                    return random.randint(750, min(self.max_context, self.seq_length - 1))
        
        return random.randint(self.min_context, min(self.max_context, self.seq_length - 1))
    
    def __getitem__(self, i):
        # Use index for reproducibility but also allow randomness
        history_idx = i % self.n_histories
        
        # Sample context length from distribution
        context_length = self._sample_context_length()
        
        # Sample a valid end position (where we predict action)
        max_end = self.seq_length - 1
        min_end = context_length
        
        if min_end >= max_end:
            end_idx = max_end
        else:
            end_idx = random.randint(min_end, max_end)
        
        start_idx = end_idx - context_length
        
        traj = {
            'query_states': self.states[history_idx, end_idx],
            'target_actions': self.actions[history_idx, end_idx],
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
            'context_length': context_length,  # For logging/debugging
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, end_idx],
                'target_rewards': self.rewards[history_idx, end_idx],
            })
        
        return traj


class CompressionPretrainDataset(Dataset):
    """
    Dataset for pre-training the compression transformer.
    
    Samples fixed-length windows of transitions for reconstruction loss training.
    No query states or target actions needed - just sequences to compress and reconstruct.
    """
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.window_size = config['n_transit'] - 1  # Size of sequences to compress
        self.dynamics = config['dynamics']
        
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
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
    
    def __len__(self):
        return self.n_histories * (self.seq_length - self.window_size)
    
    def __getitem__(self, i):
        history_idx = i % self.n_histories
        
        # Random window start
        max_start = self.seq_length - self.window_size
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + self.window_size
        
        traj = {
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
        }
        
        return traj