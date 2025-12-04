from torch.utils.data import Dataset, Sampler
from collections import defaultdict
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
        Pre-generate all valid sample configurations with better coverage.
        
        Each sample specifies:
        - history_idx: which trajectory (maintains cross-episode history)
        - compression_depth: 0-3 compression cycles
        - segment_lengths: list of segment lengths for each compression stage
        - target_idx: which timestep to predict
        
        Strategy: Generate more samples with higher compression depths to match
        the distribution seen during long evaluation episodes.
        """
        samples = []
        
        n_histories = len(self.states)
        traj_length = self.states.shape[1]
        
        # Weight distribution towards higher compression depths
        # This ensures model learns to handle compressed contexts
        depth_weights = {0: 1, 1: 2, 2: 3, 3: 4}  # More samples for higher depths
        
        for history_idx in range(n_histories):
            # Generate samples with different compression depths
            for depth in range(self.max_compression_depth + 1):
                n_samples_for_depth = depth_weights.get(depth, 1)
                
                if depth == 0:
                    # No compression: sample from various context lengths
                    min_len = self.min_uncompressed_length + 1
                    max_len = min(self.max_uncompressed_length + 1, traj_length)
                    
                    # Sample multiple context lengths
                    for _ in range(n_samples_for_depth * 3):
                        context_len = random.randint(min_len, max_len)
                        if context_len < traj_length:
                            target_idx = random.randint(context_len, min(context_len + 30, traj_length - 1))
                            samples.append({
                                'history_idx': history_idx,
                                'compression_depth': 0,
                                'segment_lengths': [],
                                'uncompressed_start': target_idx - context_len,
                                'target_idx': target_idx
                            })
                else:
                    # Multiple compression stages - generate varied configurations
                    for _ in range(n_samples_for_depth * 10):  # More samples for compression training
                        segment_lengths = []
                        total_length = 0
                        
                        # Generate compression stage lengths
                        for stage in range(depth):
                            # Vary segment lengths more to expose model to diverse compressions
                            seg_len = random.randint(self.min_compress_length, self.max_compress_length)
                            segment_lengths.append(seg_len)
                            total_length += seg_len
                        
                        # Add uncompressed segment with varied length
                        uncomp_len = random.randint(self.min_uncompressed_length, 
                                                   min(self.max_uncompressed_length, 80))
                        total_length += uncomp_len
                        
                        # Check if valid
                        if total_length < traj_length - 5:  # Leave room for target
                            samples.append({
                                'history_idx': history_idx,
                                'compression_depth': depth,
                                'segment_lengths': segment_lengths,
                                'uncompressed_length': uncomp_len,
                                'total_length': total_length
                            })
        
        random.shuffle(samples)
        print(f"\nGenerated {len(samples)} training samples")
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
        
        Args:
            history_idx: which trajectory
            start_idx: starting timestep
            length: number of timesteps
        
        Returns:
            dict with 'states', 'actions', 'rewards', 'next_states'
        """
        end_idx = start_idx + length
        
        actions_data = self.actions[history_idx, start_idx:end_idx]
        
        # Check if actions are already one-hot encoded (2D) or need encoding (1D)
        if len(actions_data.shape) == 1:
            # Actions are indices, need to one-hot encode
            num_actions = self.config.get('num_actions', 5)
            actions_onehot = np.zeros((length, num_actions), dtype=np.float32)
            actions_onehot[np.arange(length), actions_data.astype(int)] = 1.0
            actions_tensor = torch.tensor(actions_onehot, dtype=torch.float32)
        else:
            # Actions are already one-hot encoded
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
        
        if depth == 0:
            # No compression case
            uncompressed_start = sample_info['uncompressed_start']
            target_idx = sample_info['target_idx']
            uncompressed_length = target_idx - uncompressed_start
            
            # Handle target action - convert one-hot to index if needed
            target_action_data = self.actions[history_idx, target_idx]
            if len(target_action_data.shape) == 1 and target_action_data.shape[0] > 1:
                # One-hot encoded, convert to index
                target_action = torch.tensor(np.argmax(target_action_data), dtype=torch.long)
            else:
                # Already an index
                target_action = torch.tensor(target_action_data, dtype=torch.long)
            
            return {
                'compression_stages': [],
                'uncompressed_context': self._get_context_dict(history_idx, uncompressed_start, uncompressed_length),
                'query_states': torch.tensor(self.states[history_idx, target_idx], dtype=torch.float32),
                'target_actions': target_action,
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
            
            # Query state and target action
            query_states = torch.tensor(self.states[history_idx, current_idx], dtype=torch.float32)
            
            # Handle target action - convert one-hot to index if needed
            target_action_data = self.actions[history_idx, current_idx]
            if len(target_action_data.shape) == 1 and target_action_data.shape[0] > 1:
                # One-hot encoded, convert to index
                target_action = torch.tensor(np.argmax(target_action_data), dtype=torch.long)
            else:
                # Already an index
                target_action = torch.tensor(target_action_data, dtype=torch.long)
            
            return {
                'compression_stages': compression_stages,
                'uncompressed_context': uncompressed_context,
                'query_states': query_states,
                'target_actions': target_action,
                'num_compression_stages': depth
            }


def collate_compressed_batch(batch):
    """
    Custom collate function for ADCompressedDataset.
    
    Assumes all samples in batch have the SAME compression depth (enforced by BucketSampler).
    Only pads sequences within each stage to the max length in that stage.
    """
    batch_size = len(batch)
    num_stages = batch[0]['num_compression_stages']
    
    # Verify all samples have same compression depth
    depths = [item['num_compression_stages'] for item in batch]
    if not all(d == num_stages for d in depths):
        print(f"ERROR: Mixed compression depths in batch: {depths}")
        print(f"Expected all to be {num_stages}")
        raise AssertionError(
            f"BucketSampler should ensure all samples in batch have same compression depth. "
            f"Got depths: {depths}"
        )
    
    query_states_list = []
    target_actions_list = []
    all_compression_stages = []
    all_uncompressed = []
    
    for item in batch:
        all_compression_stages.append(item['compression_stages'])
        all_uncompressed.append(item['uncompressed_context'])
        query_states_list.append(item['query_states'])
        target_actions_list.append(item['target_actions'])
    
    # Batch each compression stage
    batched_stages = []
    for stage_idx in range(num_stages):
        # Collect all samples' data for this stage
        stage_samples = [all_compression_stages[i][stage_idx] for i in range(batch_size)]
        
        # Find max length in this stage
        max_len = max(s['states'].shape[0] for s in stage_samples)
        
        # Pad each sample to max_len
        batched_stage = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': []
        }
        
        for sample in stage_samples:
            seq_len = sample['states'].shape[0]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                batched_stage['states'].append(torch.cat([
                    sample['states'], 
                    torch.zeros(pad_len, *sample['states'].shape[1:])
                ]))
                batched_stage['actions'].append(torch.cat([
                    sample['actions'],
                    torch.zeros(pad_len, *sample['actions'].shape[1:])
                ]))
                batched_stage['rewards'].append(torch.cat([
                    sample['rewards'],
                    torch.zeros(pad_len)
                ]))
                batched_stage['next_states'].append(torch.cat([
                    sample['next_states'],
                    torch.zeros(pad_len, *sample['next_states'].shape[1:])
                ]))
            else:
                batched_stage['states'].append(sample['states'])
                batched_stage['actions'].append(sample['actions'])
                batched_stage['rewards'].append(sample['rewards'])
                batched_stage['next_states'].append(sample['next_states'])
        
        # Stack into batch dimension
        batched_stage = {
            k: torch.stack(v) for k, v in batched_stage.items()
        }
        batched_stages.append(batched_stage)
    
    # Batch uncompressed context
    max_uncomp_len = max(c['states'].shape[0] for c in all_uncompressed)
    batched_uncompressed = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': []
    }
    
    for c in all_uncompressed:
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
        'num_compression_stages': num_stages
    }


class BucketSampler(Sampler):
    """
    Sampler that groups samples by compression depth into buckets.
    Each batch contains only samples with the same compression depth.
    
    This eliminates the need for dummy padding across different compression depths.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group sample indices by compression depth
        self.buckets = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            depth = sample['compression_depth']
            self.buckets[depth].append(idx)
        
        # Print bucket statistics for debugging
        print(f"\nBucketSampler initialized:")
        print(f"  Total samples: {len(dataset.samples)}")
        print(f"  Batch size: {batch_size}")
        for depth in sorted(self.buckets.keys()):
            num_samples = len(self.buckets[depth])
            num_batches = (num_samples + batch_size - 1) // batch_size if drop_last else \
                         (num_samples // batch_size) + (1 if num_samples % batch_size else 0)
            print(f"  Depth {depth}: {num_samples} samples -> ~{num_batches} batches")
        
        # Precompute batches
        self._create_batches()
        print(f"  Created {len(self.batches)} total batches\n")
    
    def _create_batches(self):
        self.batches = []
        
        for depth, indices in self.buckets.items():
            # Create a copy to avoid modifying the original
            depth_indices = indices.copy()
            
            if self.shuffle:
                random.shuffle(depth_indices)
            
            # Create batches for this depth
            for i in range(0, len(depth_indices), self.batch_size):
                batch_indices = depth_indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    # Store both indices and expected depth for verification
                    self.batches.append({
                        'indices': batch_indices,
                        'depth': depth
                    })
        
        if self.shuffle:
            random.shuffle(self.batches)
    
    def __iter__(self):
        # Recreate batches each epoch if shuffling
        if self.shuffle:
            self._create_batches()
        
        for batch_info in self.batches:
            yield batch_info['indices']  # Yield list of indices for this batch
    
    def __len__(self):
        return len(self.batches)  # Number of batches, not total samples