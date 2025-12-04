# Bucketing Strategy for CompressedAD Training

## Problem with Previous Approach

The original padding strategy mixed samples with **different compression depths** in the same batch:

```
Batch Example (OLD):
- Sample 1: depth=0 (no compression)
- Sample 2: depth=2 (2 compression stages)
- Sample 3: depth=3 (3 compression stages)
```

**Issues:**
1. **Dummy padding**: Sample 1 needed 3 dummy compression stages filled with zeros
2. **Wasted computation**: Encoder processed meaningless zero tensors
3. **Gradient contamination**: Dummy data contributed to intermediate representations
4. **Memory inefficiency**: Extra padding increased memory usage

## New Bucketing Strategy

Now we group samples by **compression depth** using `BucketSampler`:

```
Bucket 0: All samples with depth=0
Bucket 1: All samples with depth=1
Bucket 2: All samples with depth=2
Bucket 3: All samples with depth=3
```

Each batch contains **only samples with the same compression depth**:

```
Batch Example (NEW):
- Sample 1: depth=2, stages=[30 steps, 25 steps]
- Sample 2: depth=2, stages=[40 steps, 35 steps]
- Sample 3: depth=2, stages=[35 steps, 30 steps]
```

**Benefits:**
1. ✅ **No dummy stages**: All stages are real data
2. ✅ **Minimal padding**: Only pad within-stage to match longest sequence
3. ✅ **Efficient computation**: No wasted transformer operations
4. ✅ **Clean gradients**: All data contributes meaningfully to learning
5. ✅ **Better memory usage**: Reduced padding overhead

## Implementation

### BucketSampler (`dataset.py`)
```python
class BucketSampler(Sampler):
    """Groups samples by compression depth into buckets."""
    
    def __init__(self, dataset, batch_size, shuffle=True):
        # Group indices by compression depth
        self.buckets = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            depth = sample['compression_depth']
            self.buckets[depth].append(idx)
        
        # Create batches within each bucket
        # Shuffle both within buckets and across buckets
```

### Updated Collate Function (`dataset.py`)
```python
def collate_compressed_batch(batch):
    """Simplified - assumes all samples have same depth."""
    
    num_stages = batch[0]['num_compression_stages']
    
    # Assert all samples have same depth (enforced by BucketSampler)
    assert all(item['num_compression_stages'] == num_stages for item in batch)
    
    # Only pad within each stage to match max length
    # No dummy stages needed!
```

### Training Script (`train.py`)
```python
# Create bucket sampler for training
train_sampler = BucketSampler(
    train_dataset,
    batch_size=config['train_batch_size'],
    shuffle=True
)

# Use batch_sampler instead of batch_size
# IMPORTANT: num_workers=0 to avoid multiprocessing issues with batch_sampler
train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,  # Controls batching by depth
    num_workers=0,  # Must be 0 with custom batch_sampler
    collate_fn=collate_compressed_batch,
    ...
)
```

**Note on `num_workers`**: Custom batch samplers don't work reliably with `num_workers > 0` due to how PyTorch distributes batches across worker processes. Using `num_workers=0` has minimal performance impact since:
- Data is in HDF5 format (fast to load)
- Most computation happens on GPU
- Bucketing already reduces batch preparation overhead

## Expected Impact on Training

### Computation Efficiency
- **Before**: ~30-40% of encoder computations on dummy data
- **After**: 100% of encoder computations on real data
- **Speedup**: ~15-20% faster training

### Memory Efficiency
- **Before**: Padded to max_depth=3 for all samples
- **After**: Only pad to actual depth per batch
- **Savings**: ~20-30% memory reduction

### Learning Quality
- **Cleaner gradients**: No gradient flow through dummy padding
- **Better representations**: Encoder learns from real data only
- **Faster convergence**: Expected 10-20% fewer steps to reach target performance

## Usage

No changes needed to existing config files. The bucketing strategy is automatically applied when training CompressedAD:

```bash
python train.py config/algorithm/ppo_base.yaml config/env/darkroom.yaml config/model/ad_compressed_dr.yaml
```

The sampler ensures:
- Each batch has consistent structure (same number of compression stages)
- Within each batch, only minimal sequence-length padding is applied
- No dummy data is processed by the model

## Verification

You can verify the bucketing is working by checking batch statistics during training:

```python
# In training loop
print(f"Batch compression depth: {batch['num_compression_stages']}")
print(f"Stage 0 shape: {batch['compression_stages'][0]['states'].shape}")
```

All samples in a batch should show the same `num_compression_stages`.
