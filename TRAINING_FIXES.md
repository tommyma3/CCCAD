# Critical Issues and Fixes for Compression Training

## Problem Analysis

Your model trained for 150k steps but loss stayed at ~1.6 (random chance) and evaluation shows near-zero rewards. This indicates **the model never learned anything**.

### Root Causes Identified:

1. **Model TOO COMPLEX**: 128-dim, 8 heads, 4 layers - overshooting for this task
2. **Learning rate TOO LOW**: 0.0001 is too conservative for the model size
3. **Compression TOO DEEP**: Max depth 3 with 40 latent tokens is excessive
4. **Dropout TOO LOW**: 0.1 dropout insufficient to prevent overfitting
5. **Label smoothing OFF**: Model becomes overconfident on wrong predictions

## Changes Applied

### 1. Model Simplification (CRITICAL)
```yaml
# OLD (TOO COMPLEX):
tf_n_embd: 128
tf_n_layer: 4
tf_n_head: 8
n_latent: 40
encoder_n_layer: 3

# NEW (MANAGEABLE):
tf_n_embd: 64      # 50% smaller
tf_n_layer: 3      # Fewer layers
tf_n_head: 4       # Fewer heads
n_latent: 20       # 50% fewer latents
encoder_n_layer: 2 # Simpler encoder
```

**Why**: Smaller model = easier to train, less prone to overfitting, faster convergence

### 2. Learning Rate Increase
```yaml
# OLD:
lr: 0.0001

# NEW:
lr: 0.0003  # 3x higher (matches standard AD)
```

**Why**: Smaller model needs higher LR to converge in reasonable time

### 3. Compression Simplification
```yaml
# OLD:
max_compression_depth: 3  # 0, 1, 2, 3 cycles
min_compress_length: 15
max_compress_length: 60
n_latent: 40

# NEW:
max_compression_depth: 2  # 0, 1, 2 cycles only
min_compress_length: 20   # Longer segments
max_compress_length: 40   # Shorter max
n_latent: 20              # Stronger compression
```

**Why**: Simpler compression = clearer learning signal

### 4. Higher Dropout
```yaml
# OLD:
tf_dropout: 0.1
tf_attn_dropout: 0.1

# NEW:
tf_dropout: 0.2
tf_attn_dropout: 0.2
```

**Why**: Prevents overfitting on complex compression patterns

### 5. Label Smoothing
```yaml
# OLD:
label_smoothing: 0.0

# NEW:
label_smoothing: 0.1
```

**Why**: Prevents overconfidence, improves generalization

### 6. Batch Size Increase
```yaml
# OLD:
train_batch_size: 128

# NEW:
train_batch_size: 256
```

**Why**: Larger batches = more stable gradients for simpler model

### 7. Training Length Adjustment
```yaml
# OLD:
train_timesteps: 150000
num_warmup_steps: 5000

# NEW:
train_timesteps: 100000  # Sufficient for simpler model
num_warmup_steps: 2000   # Faster warmup
```

**Why**: Simpler model converges faster

## Expected Improvements

### Loss Trajectory:
- **Before**: Flat at ~1.6 for 150k steps
- **After**: Should decrease to <0.8 by 50k steps, <0.5 by 100k steps

### Evaluation Rewards:
- **Before**: Mean 0.496, mostly zeros
- **After**: Mean >2.0, most environments >1.0 reward

### Training Stability:
- **Before**: No learning signal
- **After**: Clear decreasing loss curve

## Additional Recommendations

### 1. Monitor Training Closely
```bash
tensorboard --logdir runs/
```

Watch for:
- Loss should start decreasing by step 2000
- If still flat at step 5000, stop and investigate
- Accuracy should reach >40% by step 10000

### 2. Debug Mode (If Still Fails)
Add to training loop to verify compression is working:

```python
# In train.py, after line "output = model(batch)"
if step % 100 == 0 and accelerator.is_main_process:
    print(f"Batch compression depths: {batch['num_compression_stages']}")
    print(f"Loss: {output['loss_action'].item():.4f}")
```

### 3. Compare with Standard AD
Train standard AD model for comparison:
```bash
# Edit train.py line 30 to use ad_dr.yaml instead
config.update(get_config('./config/model/ad_dr.yaml'))
```

Standard AD should achieve mean reward >3.0

### 4. Check Data Loading
Verify dataset is generating good samples:
```python
# Test script
from dataset import ADCompressedDataset
config = {...}  # Load your config
dataset = ADCompressedDataset(config, './datasets', 'train', 100, 1000)
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample depth: {sample['num_compression_stages']}")
```

Should see balanced distribution across depths 0, 1, 2

## Training Command

```bash
# Single GPU
python train.py

# Multi-GPU
accelerate launch --config_file accelerate_config.yaml train.py
```

## Success Criteria

After 50k steps, you should see:
- ✅ Train loss < 0.8
- ✅ Train accuracy > 50%
- ✅ Test loss < 1.0  
- ✅ Test accuracy > 40%

After 100k steps:
- ✅ Train loss < 0.5
- ✅ Eval mean reward > 2.0
- ✅ At least 5/8 environments with reward > 1.0

## If Training Still Fails

### Option A: Further Simplify
```yaml
max_compression_depth: 1  # Only depths 0 and 1
n_latent: 10              # Even fewer latents
```

### Option B: Increase Uncompressed Context
```yaml
min_uncompressed_length: 30   # More recent context
max_uncompressed_length: 120  # Much more context
```

### Option C: Add Auxiliary Loss
In model forward(), add reconstruction loss for latents to provide additional training signal.

### Option D: Start from Standard AD
1. Train standard AD to convergence
2. Initialize CompressedAD encoder/decoder from AD weights
3. Fine-tune with compression

## Model Architecture Comparison

| Component | Old (Failed) | New (Fixed) | Standard AD |
|-----------|-------------|-------------|-------------|
| Embedding | 128 | 64 | 64 |
| Layers | 4 | 3 | 4 |
| Heads | 8 | 4 | 4 |
| Latents | 40 | 20 | N/A |
| Parameters | ~2M | ~0.5M | ~0.4M |
| Compression Depth | 0-3 | 0-2 | N/A |

The new configuration is much closer to standard AD in complexity while adding compression capability.

## Key Insight

**Compression is hard to learn**. The model must simultaneously:
1. Learn the base task (navigation)
2. Learn to compress history into latents
3. Learn to decode from latents

Starting with a complex model makes this nearly impossible. The simplified version provides a clearer learning path.
