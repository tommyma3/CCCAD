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
        elif self.env == 'dark_key_to_door':
            n_total_envs = min(200, config['grid_size'] ** 4)  # Limited to 200 tasks
        else:
            raise ValueError(f'Invalid env: {self.env}')

        # Note: collect.py saves data with train tasks first (indices 0 to n_train-1),
        # then test tasks (indices n_train to n_total-1). No shuffle needed here.
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = list(range(n_train_envs))
        elif mode == 'test':
            env_idx = list(range(n_train_envs, n_total_envs))
        elif mode == 'all':
            env_idx = list(range(n_total_envs))
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                grp = f.get(f'{i}')
                if grp is None:
                    print(f'Warning: trajectory group "{i}" not found in {traj_dir}/{get_traj_file_name(config)}.hdf5; skipping')
                    continue

                states.append(grp['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(grp['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(grp['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(grp['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        if len(states) == 0:
            raise RuntimeError(f'No trajectory groups found for mode="{mode}" in {traj_dir}/{get_traj_file_name(config)}.hdf5')

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
    
    Supports curriculum-aware sampling where length distribution can be 
    dynamically updated during training to prevent catastrophic forgetting.
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
        
        # Length distribution (can be updated dynamically for curriculum)
        default_distribution = {
            'short': 0.2,      # No compression: 50-200
            'medium': 0.3,     # 1 compression: 250-400  
            'long': 0.3,       # 2-3 compressions: 450-700
            'very_long': 0.2,  # 4+ compressions: 750-1000
        }
        self.length_distribution = config.get('length_distribution', default_distribution).copy()
        
        # Validate distribution sums to 1.0
        self._validate_distribution(self.length_distribution)
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'dark_key_to_door':
            n_total_envs = config['grid_size'] ** 4  # All possible key/goal combinations
        else:
            raise ValueError(f'Invalid environment: {self.env}')

        # Note: collect.py saves data with train tasks first (indices 0 to n_train-1),
        # then test tasks (indices n_train to n_total-1). No shuffle needed here.
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = list(range(n_train_envs))
        elif mode == 'test':
            env_idx = list(range(n_train_envs, n_total_envs))
        elif mode == 'all':
            env_idx = list(range(n_total_envs))
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                grp = f.get(f'{i}')
                if grp is None:
                    print(f'Warning: trajectory group "{i}" not found in {traj_dir}/{get_traj_file_name(config)}.hdf5; skipping')
                    continue

                states.append(grp['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(grp['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(grp['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(grp['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        if len(states) == 0:
            raise RuntimeError(f'No trajectory groups found for mode="{mode}" in {traj_dir}/{get_traj_file_name(config)}.hdf5')

        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
    
    def _validate_distribution(self, dist):
        """Validate that distribution sums to 1.0."""
        total_prob = sum(dist.values())
        assert abs(total_prob - 1.0) < 1e-6, f"Length distribution must sum to 1.0, got {total_prob}"
    
    def update_length_distribution(self, new_distribution):
        """
        Update the length distribution for curriculum-aware sampling.
        
        This allows the training script to dynamically change what sequence
        lengths are sampled based on the current curriculum stage.
        
        Args:
            new_distribution: dict with keys like 'short', 'medium', 'long', 'very_long'
                             and values summing to 1.0
        """
        self._validate_distribution(new_distribution)
        self.length_distribution = new_distribution.copy()
    
    def __len__(self):
        # Approximate: we can sample many different windows
        return self.n_histories * (self.seq_length - self.min_context)
    
    def _sample_context_length(self):
        """Sample context length from distribution supporting multi-compression training."""
        r = random.random()
        cumulative = 0
        
        # Maximum available context (leave room for query state)
        max_available = self.seq_length - 1
        
        # Calculate compression trigger point: when buffer fills up
        # Available space = n_transit - n_compress_tokens (if has latent) - 1 (query)
        # First compression triggers at ~(n_transit - 1) transitions
        compression_trigger = self.n_transit - 1
        
        for category, prob in self.length_distribution.items():
            cumulative += prob
            if r < cumulative:
                if category == 'short':
                    # No compression needed (fits in n_transit - 1)
                    low = max(20, self.min_context)
                    high = min(compression_trigger - 10, compression_trigger - 1)  # Stay below compression
                elif category == 'medium':
                    # 1-2 compressions - spans 1x to 2.5x compression trigger
                    low = compression_trigger
                    high = min(int(compression_trigger * 2.5), self.max_context)
                elif category == 'long':
                    # 3-5 compressions - spans 2.5x to 5x compression trigger
                    low = min(int(compression_trigger * 2.5), self.max_context)
                    high = min(int(compression_trigger * 5), self.max_context)
                elif category == 'very_long':
                    # 5+ compressions - spans 5x+ compression trigger
                    low = min(int(compression_trigger * 5), self.max_context)
                    high = self.max_context
                elif category == 'extended':
                    # Maximum compressions (requires longer dataset)
                    low = min(int(compression_trigger * 8), self.max_context)
                    high = self.max_context
                else:
                    # Fallback for unknown categories
                    low, high = self.min_context, self.max_context
                
                # Clamp to available data
                high = min(high, max_available)
                low = min(low, high)  # Ensure low <= high
                
                return random.randint(low, high)
        
        # Fallback
        return random.randint(self.min_context, min(self.max_context, max_available))
    
    def __getitem__(self, i):
        # Use index for reproducibility but also allow randomness
        history_idx = i % self.n_histories
        
        # CRITICAL: Sample a random END position within the trajectory
        # This ensures the model sees the LEARNING PROGRESSION:
        # - Early positions: agent is still exploring, making mistakes
        # - Middle positions: agent is improving
        # - Late positions: agent has learned good policy
        # Without this, the model only sees expert behavior and can't learn in-context!
        
        # We need at least min_context transitions before end_idx
        min_end = self.min_context
        max_end = self.seq_length - 1
        
        # Sample random end position
        end_idx = random.randint(min_end, max_end)
        
        # Calculate context length (from start to end_idx, exclusive of end_idx which is query)
        # We return full available context; collate_fn will truncate based on curriculum
        context_length = min(end_idx, self.max_context)
        start_idx = end_idx - context_length
        
        traj = {
            'query_states': self.states[history_idx, end_idx],
            'target_actions': self.actions[history_idx, end_idx],
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
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
        elif self.env == 'dark_key_to_door':
            n_total_envs = config['grid_size'] ** 4  # All possible key/goal combinations
        else:
            raise ValueError(f'Invalid env: {self.env}')

        # Note: collect.py saves data with train tasks first (indices 0 to n_train-1),
        # then test tasks (indices n_train to n_total-1). No shuffle needed here.
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = list(range(n_train_envs))
        elif mode == 'test':
            env_idx = list(range(n_train_envs, n_total_envs))
        elif mode == 'all':
            env_idx = list(range(n_total_envs))
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