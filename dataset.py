from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat
import torch


class ADDataset(Dataset):
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


class ADCompressedDataset(Dataset):
    """
    Dataset for training compressed AD model with hierarchical compression.
    
    This dataset generates training samples with variable compression depths (0-3 cycles),
    simulating the hierarchical compression that occurs during evaluation.
    
    Key feature: Maintains cross-episode history by sampling from trajectories within
    the same environment (history_idx), ensuring the model learns from in-context learning.
    """
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        self.n_latent = config.get('n_latent', 60)
        self.max_compression_depth = config.get('max_compression_depth', 3)
        
        # Compression parameters
        self.min_compress_length = config.get('min_compress_length', 10)
        self.max_compress_length = config.get('max_compress_length', 50)
        self.min_uncompressed_length = config.get('min_uncompressed_length', 5)
        self.max_uncompressed_length = config.get('max_uncompressed_length', 30)
        
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
        
        # Pre-compute valid samples based on compression scenarios
        self.samples = self._generate_sample_indices()
    
    def _generate_sample_indices(self):
        """
        Pre-generate all valid sample configurations.
        
        Each sample specifies:
        - history_idx: which trajectory (maintains cross-episode history)
        - compression_depth: 0-3 compression cycles
        - segment_lengths: list of segment lengths for each compression stage
        - target_idx: which timestep to predict
        """
        samples = []
        
        n_histories = len(self.states)
        traj_length = self.states.shape[1]
        
        for history_idx in range(n_histories):
            # Generate samples with different compression depths
            for depth in range(self.max_compression_depth + 1):
                if depth == 0:
                    # No compression: simple case
                    min_len = self.min_uncompressed_length + 1
                    for context_len in range(min_len, min(self.max_uncompressed_length + 1, traj_length)):
                        for target_idx in range(context_len, min(context_len + 20, traj_length)):
                            samples.append({
                                'history_idx': history_idx,
                                'compression_depth': 0,
                                'segment_lengths': [],
                                'uncompressed_start': target_idx - context_len,
                                'target_idx': target_idx
                            })
                else:
                    # Multiple compression stages
                    # Generate random valid segmentations
                    for _ in range(5):  # 5 random samples per depth per history
                        segment_lengths = []
                        total_length = 0
                        
                        # Generate compression stage lengths
                        for stage in range(depth):
                            if stage == 0:
                                seg_len = random.randint(self.min_compress_length, self.max_compress_length)
                            else:
                                seg_len = random.randint(self.min_compress_length, 
                                                        min(self.max_compress_length, 40))
                            segment_lengths.append(seg_len)
                            total_length += seg_len
                        
                        # Add uncompressed segment
                        uncomp_len = random.randint(self.min_uncompressed_length, self.max_uncompressed_length)
                        total_length += uncomp_len
                        
                        # Check if valid
                        if total_length < traj_length:
                            samples.append({
                                'history_idx': history_idx,
                                'compression_depth': depth,
                                'segment_lengths': segment_lengths,
                                'uncompressed_length': uncomp_len,
                                'total_length': total_length
                            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _get_context_dict(self, history_idx, start_idx, length):
        """
        Extract context dictionary for a given segment.
        
        Args:
            history_idx: which trajectory
            start_idx: starting timestep
            length: number of timesteps
        
        Returns:
            dict with 'states', 'actions', 'rewards', 'next_states'
        """
        end_idx = start_idx + length
        
        return {
            'states': torch.tensor(self.states[history_idx, start_idx:end_idx], dtype=torch.float32),
            'actions': torch.tensor(self.actions[history_idx, start_idx:end_idx], dtype=torch.float32),
            'rewards': torch.tensor(self.rewards[history_idx, start_idx:end_idx], dtype=torch.float32),
            'next_states': torch.tensor(self.next_states[history_idx, start_idx:end_idx], dtype=torch.float32)
        }
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        history_idx = sample_info['history_idx']
        depth = sample_info['compression_depth']
        
        if depth == 0:
            # No compression case
            uncompressed_start = sample_info['uncompressed_start']
            target_idx = sample_info['target_idx']
            uncompressed_length = target_idx - uncompressed_start
            
            return {
                'compression_stages': [],
                'uncompressed_context': self._get_context_dict(history_idx, uncompressed_start, uncompressed_length),
                'query_states': torch.tensor(self.states[history_idx, target_idx], dtype=torch.float32),
                'target_actions': torch.tensor(self.actions[history_idx, target_idx], dtype=torch.long),
                'num_compression_stages': 0
            }
        else:
            # Multiple compression stages
            segment_lengths = sample_info['segment_lengths']
            uncomp_len = sample_info['uncompressed_length']
            
            compression_stages = []
            current_idx = 0
            
            # Build compression stages
            for seg_len in segment_lengths:
                compression_stages.append(self._get_context_dict(history_idx, current_idx, seg_len))
                current_idx += seg_len
            
            # Uncompressed context
            uncompressed_context = self._get_context_dict(history_idx, current_idx, uncomp_len)
            current_idx += uncomp_len
            
            # Query state and target
            query_states = torch.tensor(self.states[history_idx, current_idx], dtype=torch.float32)
            target_actions = torch.tensor(self.actions[history_idx, current_idx], dtype=torch.long)
            
            return {
                'compression_stages': compression_stages,
                'uncompressed_context': uncompressed_context,
                'query_states': query_states,
                'target_actions': target_actions,
                'num_compression_stages': depth
            }


def collate_compressed_batch(batch):
    """
    Custom collate function for ADCompressedDataset.
    
    Handles variable-length compression stages by batching them appropriately.
    """
    batch_size = len(batch)
    max_stages = max(item['num_compression_stages'] for item in batch)
    
    # Initialize lists for each compression stage
    compression_stages = [[] for _ in range(max_stages)] if max_stages > 0 else []
    
    uncompressed_contexts = []
    query_states_list = []
    target_actions_list = []
    num_stages_list = []
    
    for item in batch:
        num_stages = item['num_compression_stages']
        num_stages_list.append(num_stages)
        
        # Collect compression stages
        for stage_idx in range(num_stages):
            stage_data = item['compression_stages'][stage_idx]
            compression_stages[stage_idx].append(stage_data)
        
        # Pad with None for samples with fewer stages
        for stage_idx in range(num_stages, max_stages):
            compression_stages[stage_idx].append(None)
        
        uncompressed_contexts.append(item['uncompressed_context'])
        query_states_list.append(item['query_states'])
        target_actions_list.append(item['target_actions'])
    
    # Stack tensors for each stage
    batched_stages = []
    for stage_idx in range(max_stages):
        stage_batch = [s for s in compression_stages[stage_idx] if s is not None]
        if len(stage_batch) > 0:
            # Find max length in this stage
            max_len = max(s['states'].shape[0] for s in stage_batch)
            
            # Pad and stack
            batched_stage = {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': []
            }
            
            for s in stage_batch:
                seq_len = s['states'].shape[0]
                if seq_len < max_len:
                    pad_len = max_len - seq_len
                    batched_stage['states'].append(torch.cat([
                        s['states'], 
                        torch.zeros(pad_len, *s['states'].shape[1:])
                    ]))
                    batched_stage['actions'].append(torch.cat([
                        s['actions'],
                        torch.zeros(pad_len, *s['actions'].shape[1:])
                    ]))
                    batched_stage['rewards'].append(torch.cat([
                        s['rewards'],
                        torch.zeros(pad_len)
                    ]))
                    batched_stage['next_states'].append(torch.cat([
                        s['next_states'],
                        torch.zeros(pad_len, *s['next_states'].shape[1:])
                    ]))
                else:
                    batched_stage['states'].append(s['states'])
                    batched_stage['actions'].append(s['actions'])
                    batched_stage['rewards'].append(s['rewards'])
                    batched_stage['next_states'].append(s['next_states'])
            
            # Stack into batch dimension
            batched_stage = {
                k: torch.stack(v) for k, v in batched_stage.items()
            }
            batched_stages.append(batched_stage)
    
    # Batch uncompressed context
    max_uncomp_len = max(c['states'].shape[0] for c in uncompressed_contexts)
    batched_uncompressed = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': []
    }
    
    for c in uncompressed_contexts:
        seq_len = c['states'].shape[0]
        if seq_len < max_uncomp_len:
            pad_len = max_uncomp_len - seq_len
            batched_uncompressed['states'].append(torch.cat([
                c['states'],
                torch.zeros(pad_len, *c['states'].shape[1:])
            ]))
            batched_uncompressed['actions'].append(torch.cat([
                c['actions'],
                torch.zeros(pad_len, *c['actions'].shape[1:])
            ]))
            batched_uncompressed['rewards'].append(torch.cat([
                c['rewards'],
                torch.zeros(pad_len)
            ]))
            batched_uncompressed['next_states'].append(torch.cat([
                c['next_states'],
                torch.zeros(pad_len, *c['next_states'].shape[1:])
            ]))
        else:
            batched_uncompressed['states'].append(c['states'])
            batched_uncompressed['actions'].append(c['actions'])
            batched_uncompressed['rewards'].append(c['rewards'])
            batched_uncompressed['next_states'].append(c['next_states'])
    
    batched_uncompressed = {
        k: torch.stack(v) for k, v in batched_uncompressed.items()
    }
    
    return {
        'compression_stages': batched_stages,
        'uncompressed_context': batched_uncompressed,
        'query_states': torch.stack(query_states_list),
        'target_actions': torch.stack(target_actions_list),
        'num_compression_stages': max_stages
    }