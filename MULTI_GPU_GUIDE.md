# Multi-GPU Training Guide

## Setup

### 1. Configure Accelerate

You have two options:

#### Option A: Use the provided config (Recommended)
```bash
# The accelerate_config.yaml is already configured for 2 GPUs
# Edit it to match your GPU count:
# Change `num_processes: 2` to your number of GPUs
```

#### Option B: Interactive configuration
```bash
accelerate config
# Follow the prompts:
# - Choose "This machine"
# - Choose "multi-GPU"
# - Enter number of GPUs
# - Choose "fp16" for mixed precision
# - Choose "all" for GPU IDs
```

### 2. Check Your Configuration
```bash
accelerate env
```

## Running Training

### Single Command Launch (Recommended)
```bash
# Use the provided config file
accelerate launch --config_file accelerate_config.yaml train.py

# Or if you used interactive config:
accelerate launch train.py
```

### Manual Launch (Advanced)
```bash
# For 2 GPUs
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 train.py

# For 4 GPUs
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 train.py

# For all available GPUs
accelerate launch --multi_gpu --num_processes=auto --mixed_precision=fp16 train.py
```

## What Changed

### Code Updates:
1. **Distributed initialization** - Accelerator now handles multi-GPU setup
2. **Process synchronization** - Added `wait_for_everyone()` before/after eval and checkpoints
3. **Main process logging** - Only rank 0 writes to tensorboard and saves checkpoints
4. **Model unwrapping** - Properly saves model state from distributed wrapper
5. **Progress bars** - Only shown on main process to avoid clutter

### Expected Behavior:

#### With 2 GPUs:
- **Effective batch size**: 128 × 2 = 256
- **Speed**: ~1.8-1.9x faster than single GPU
- **Memory**: Distributed across GPUs

#### With 4 GPUs:
- **Effective batch size**: 128 × 4 = 512
- **Speed**: ~3.5-3.8x faster than single GPU
- **Memory**: Distributed across GPUs

## Monitoring

### Check GPU Usage:
```bash
# Windows
nvidia-smi

# Continuous monitoring (every 2 seconds)
nvidia-smi -l 2
```

### Tensorboard:
```bash
tensorboard --logdir runs/
```

You should see the same metrics as single-GPU training, but training will be faster.

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config
```yaml
train_batch_size: 64  # Instead of 128
```

### Issue: Only one GPU is being used
**Solution**: Check your accelerate config
```bash
accelerate env
# Verify num_processes matches your GPU count
```

### Issue: Duplicate tensorboard logs
**Solution**: Already fixed - only main process logs

### Issue: Multiple checkpoints saved
**Solution**: Already fixed - only main process saves

## Performance Tips

### 1. Optimize Batch Size
- Larger batches = better GPU utilization
- But: may need to adjust learning rate
- Rule of thumb: If you double batch size, multiply LR by √2

### 2. Gradient Accumulation (if needed)
If you want larger effective batch size but hit memory limits:

```python
# In train.py, change Accelerator initialization:
accelerator = Accelerator(
    mixed_precision=config['mixed_precision'],
    gradient_accumulation_steps=4,  # Accumulate over 4 steps
    ...
)

# This gives effective batch size = 128 × num_gpus × 4
```

### 3. Monitor Efficiency
```python
# Add to your tensorboard logs (in summary_interval block):
if accelerator.is_main_process:
    writer.add_scalar('train/samples_per_sec', 
                      config['train_batch_size'] * accelerator.num_processes / time_per_step, 
                      step)
```

## Scaling Recommendations

### For CompressedAD Model:

| GPUs | Batch Size | Effective Batch | Expected Speedup |
|------|------------|----------------|------------------|
| 1    | 128        | 128            | 1.0x (baseline)  |
| 2    | 128        | 256            | ~1.85x           |
| 4    | 96         | 384            | ~3.6x            |
| 8    | 64         | 512            | ~6.5x            |

Note: With more GPUs, you may need to reduce per-GPU batch size due to model size.

## Verification

After starting training, you should see:
```
Distributed training: MULTI_GPU
Number of processes: 2
Mixed precision: fp16
Process 0 using device: cuda:0
Process 1 using device: cuda:1
```

Then only process 0 will show the progress bar and logging messages.

## Converting Single-GPU Checkpoints

If you have old checkpoints from single-GPU training, they're compatible!
Just load them normally - Accelerate handles the conversion automatically.

## Emergency: Fallback to Single GPU

If multi-GPU has issues, just run normally:
```bash
python train.py
```

The code automatically detects and works with single GPU.
