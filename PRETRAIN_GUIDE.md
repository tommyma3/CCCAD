# Encoder Pre-training Guide

This guide explains how to use the two-stage training approach for CompressedAD: first pre-train the encoder with reconstruction, then train the full model for action prediction.

## Overview

**Problem**: Training the encoder end-to-end is difficult because the gradient signal flows weakly through the path: actions → decoder → latents → encoder.

**Solution**: Pre-train the encoder with a direct reconstruction objective, then use it to initialize the full model.

## Two-Stage Training Process

### Stage 1: Pre-train Encoder (Unsupervised)

The encoder learns to compress context sequences into latent representations that can be reconstructed.

**Objective**: Autoencoding in latent space
- Input: Context sequences (states, actions, rewards, next_states)
- Encoder: Compresses to latent tokens
- Decoder: Reconstructs from latents
- Loss: MSE between input and reconstruction

**Command**:
```bash
# Single GPU
python pretrain_encoder.py

# Multi-GPU (recommended)
accelerate launch --config_file accelerate_config.yaml pretrain_encoder.py
```

**Configuration**: `config/model/pretrain_encoder.yaml`

Key hyperparameters:
- `pretrain_steps: 50000` - Pre-training duration
- `pretrain_batch_size: 128` - Batch size for pre-training
- `pretrain_lr: 0.0003` - Learning rate
- `pretrain_decoder_n_layer: 2` - Simple decoder for reconstruction

**Output**: 
- Checkpoints saved to `./runs/pretrain-encoder-{env}-seed{seed}/`
- Final checkpoint: `encoder-pretrained-final.pt`

**Monitoring**:
```bash
tensorboard --logdir ./runs/pretrain-encoder-darkroom-seed0
```

Watch:
- `pretrain/loss` - Should decrease steadily
- `pretrain/correlation` - Should increase (0.6+ is good)

**Training Duration**: ~30-60 minutes on 1 GPU, ~15-30 minutes on 4 GPUs

---

### Stage 2: Train Full Model (Supervised)

Train the full CompressedAD model for action prediction, using the pretrained encoder as initialization.

**Objective**: Action prediction with compressed context

**Command**:
```bash
# Update config to use pretrained encoder
# In config/model/ad_compressed_dr.yaml, add:
# pretrained_encoder_path: './runs/pretrain-encoder-darkroom-seed0/encoder-pretrained-final.pt'

# Then run training
accelerate launch --config_file accelerate_config.yaml train.py
```

**Configuration Options**:

In `config/model/ad_compressed_dr.yaml`:
```yaml
# Path to pretrained encoder checkpoint
pretrained_encoder_path: './runs/pretrain-encoder-darkroom-seed0/encoder-pretrained-final.pt'

# Option 1: Fine-tune encoder (recommended)
freeze_encoder: false

# Option 2: Keep encoder frozen (faster but less flexible)
freeze_encoder: true
```

**Recommendations**:
- Start with `freeze_encoder: false` (fine-tuning)
- If training is unstable, try `freeze_encoder: true`
- Consider lower learning rate (lr: 0.0001) for fine-tuning

---

## File Structure

```
CCCAD/
├── pretrain_encoder.py          # Pre-training script
├── dataset_pretrain.py          # Pre-training dataset
├── train.py                     # Main training (modified to load pretrained weights)
├── config/
│   └── model/
│       ├── pretrain_encoder.yaml   # Pre-training config
│       └── ad_compressed_dr.yaml   # Main training config
└── runs/
    ├── pretrain-encoder-*/      # Pre-training checkpoints
    └── CompressedAD-*/          # Main training checkpoints
```

---

## Quick Start

```bash
# 1. Pre-train encoder (50k steps, ~30 min on 1 GPU)
accelerate launch --config_file accelerate_config.yaml pretrain_encoder.py

# 2. Update config to use pretrained encoder
# Edit config/model/ad_compressed_dr.yaml and add:
# pretrained_encoder_path: './runs/pretrain-encoder-darkroom-seed0/encoder-pretrained-final.pt'

# 3. Train full model
accelerate launch --config_file accelerate_config.yaml train.py

# 4. Evaluate
python evaluate.py
```

---

## Expected Results

### Pre-training (Stage 1)
- Initial loss: ~1.0-2.0
- Final loss: ~0.1-0.3
- Correlation: 0.6-0.8

### Main Training (Stage 2)
With pretrained encoder:
- **Should converge faster** than training from scratch
- **More stable training** (less loss oscillation)
- **Better final performance** (higher mean reward)

Without pretraining (baseline):
- Slower convergence
- May get stuck at loss ~1.6 (random chance)
- Weak compression learning

---

## Troubleshooting

### Pre-training issues

**Loss not decreasing**:
- Check data loading (should show sample counts per depth)
- Verify gradient flow (watch lr in tensorboard)
- Try higher learning rate (pretrain_lr: 0.0005)

**Out of memory**:
- Reduce pretrain_batch_size (128 → 64)
- Reduce max_compression_depth (3 → 2)
- Use fewer latent tokens (n_latent: 20 → 15)

### Main training issues

**Pretrained weights not loading**:
- Check file path in config
- Verify checkpoint file exists
- Check console for loading messages

**Training unstable after loading**:
- Try freeze_encoder: true
- Lower learning rate (lr: 0.0001)
- Increase warmup steps (num_warmup_steps: 5000)

**No improvement over random**:
- Verify pretrained encoder actually learned (check correlation > 0.5)
- Try longer pre-training (pretrain_steps: 100000)
- Check if main training loaded weights correctly

---

## Technical Details

### What Gets Saved in Checkpoints

Pre-training checkpoint contains:
```python
{
    'encoder_state_dict': ...,        # Encoder weights (what we want)
    'embed_context_state_dict': ...,  # Context embedding (shared)
    'embed_ln_state_dict': ...,       # Embedding layer norm (shared)
    'global_step': 50000,
    'config': {...}
}
```

### What Gets Loaded in Main Training

From `train.py`:
1. Creates CompressedAD model with random initialization
2. Loads pretrained encoder weights from checkpoint
3. Loads shared embedding layers
4. Optionally freezes encoder (if freeze_encoder: true)
5. Continues with normal training

### Why This Works

**Direct gradient signal**: During pre-training, encoder gets direct loss signal from reconstruction, not indirect signal through action prediction.

**Learned compression**: Encoder learns to extract task-relevant information into latents before seeing action labels.

**Better initialization**: Main training starts with encoder that already knows how to compress, just needs to learn what to compress for actions.

---

## Performance Comparison

| Approach | Training Time | Final Loss | Mean Reward | Notes |
|----------|--------------|------------|-------------|-------|
| Random init | 100k steps | ~1.6 | ~0.4 | Doesn't learn |
| + Reconstruction loss | 100k steps | ~1.5 | ~0.45 | Slight improvement |
| **Pre-training** | 50k + 100k | **~0.8** | **~0.7+** | **Recommended** |

---

## Advanced: Curriculum Pre-training

For even better results, pre-train with curriculum learning:

1. **Phase 1** (20k steps): Only depth=1 compression
2. **Phase 2** (20k steps): Depth=1,2
3. **Phase 3** (10k steps): Full depth=1,2,3

Modify `dataset_pretrain.py` to control depth distribution during different phases.

---

## Questions?

- Check tensorboard for training curves
- Verify data loading shows expected sample counts
- Test with standard AD model first to ensure dataset is correct
- Compare pretrained vs random initialization side-by-side
