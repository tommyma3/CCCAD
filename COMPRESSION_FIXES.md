# Compression Model Training Fixes

## Issues Identified and Fixed

### 1. **Architecture Stability Issues**

#### Problem:
- Missing layer normalization in encoder and decoder
- Poor initialization of embeddings
- No pre-normalization in transformer layers

#### Solution:
- Added `input_ln` and `output_ln` to `CompressionEncoder`
- Added `input_ln` to decoder in `CompressedAD`
- Added `embed_ln` for embedding normalization
- Changed to `norm_first=True` in TransformerEncoderLayer (pre-norm architecture)
- Improved initialization: Xavier for linear layers, proper scaling for embeddings

**Impact**: Better gradient flow, more stable training, reduced risk of vanishing/exploding gradients

---

### 2. **Attention Masking Problems**

#### Problem:
- Applied causal masking uniformly across all tokens
- Latent tokens (compressed history) were being causally masked
- This prevented proper information flow from compressed context

#### Solution:
- Implemented hybrid attention mask in `_get_causal_mask(seq_len, n_prefix_tokens=0)`
- Latent tokens (prefix) can attend to each other bidirectionally
- Context tokens use causal masking (only attend to past)
- Query token uses causal masking relative to context

**Impact**: Latent tokens can now properly aggregate information, decoder can leverage compressed history effectively

---

### 3. **Dataset Sampling Issues**

#### Problem:
- Too few samples with high compression depth
- Imbalanced training distribution (mostly depth-0 samples)
- Small random sample count per configuration
- Model never learned to handle compressed contexts properly

#### Solution:
- Weighted sampling: `depth_weights = {0: 1, 1: 2, 2: 3, 3: 4}`
- More samples for higher compression depths
- Increased samples per depth: depth-0 gets 3x samples, depth-3 gets 40x samples
- Better variation in segment lengths
- Added debug logging for sample distribution

**Impact**: Model now trains on compression extensively, learns to use latent tokens

---

### 4. **Hyperparameter Issues**

#### Problem:
- Learning rate too high (0.0003) for complex model
- Insufficient warmup steps
- Wrong batch size (too large)
- Insufficient training steps
- Embedding dimension too small
- Too many latent tokens (information dilution)

#### Solution:
```yaml
# Critical parameter changes:
lr: 0.0001              # Reduced from 0.0003
train_batch_size: 128   # Reduced from 256
train_timesteps: 150000 # Increased from 100000
num_warmup_steps: 5000  # Increased from 2000

# Architecture improvements:
tf_n_embd: 128          # Increased from 64
tf_n_head: 8            # Increased from 4
tf_dim_feedforward: 512 # Increased from 256
n_latent: 40            # Reduced from 60 (better compression ratio)

# Better compression parameters:
min_compress_length: 15      # Increased from 10
max_compress_length: 60      # Increased from 50
min_uncompressed_length: 10  # Increased from 5
max_uncompressed_length: 100 # Reduced from 160
```

**Impact**: More stable training, better capacity, proper convergence

---

### 5. **Gradient Flow Improvements**

#### Problem:
- No normalization of embeddings
- Abrupt transitions between encoder and decoder
- Information bottleneck at compression stage

#### Solution:
- Added `embed_ln` after context embedding
- Added `input_ln` before encoder input
- Added `output_ln` after encoder output
- Added `input_ln` before decoder
- Proper initialization with Xavier/truncated normal

**Impact**: Smooth gradient propagation through the compression bottleneck

---

## Training Recommendations

### 1. Monitor These Metrics:
- **Loss trajectory**: Should decrease steadily after warmup
- **Accuracy by compression depth**: Check if model learns all depths
- **Gradient norms**: Should stabilize after warmup (not explode/vanish)
- **Evaluation returns**: Should improve steadily

### 2. Expected Behavior:
- First 5k steps: Warmup phase, loss may be volatile
- 5k-30k steps: Rapid improvement phase
- 30k-100k steps: Continued improvement, slower
- 100k+ steps: Fine-tuning, diminishing returns

### 3. Debug Commands:
```bash
# Monitor training
tensorboard --logdir runs/

# Check sample distribution
# Look for "Generated X training samples" and "Samples per depth: {...}"
# in training logs

# Evaluate checkpoint
python evaluate.py
```

### 4. If Training Still Fails:
- Reduce learning rate further (try 5e-5)
- Increase warmup steps to 10000
- Check for NaN losses (indicates instability)
- Verify dataset is loading correctly
- Check GPU memory usage (model is larger now)

---

## Key Architecture Changes Summary

### CompressionEncoder:
```python
# Added:
- self.input_ln = nn.LayerNorm(tf_n_embd)
- self.output_ln = nn.LayerNorm(tf_n_embd)
- norm_first=True in TransformerEncoderLayer

# Forward pass now:
1. Apply input normalization
2. Concatenate with query tokens
3. Add positional embeddings
4. Transform
5. Apply output normalization
```

### CompressedAD Decoder:
```python
# Added:
- self.input_ln = nn.LayerNorm(tf_n_embd)
- self.embed_ln = nn.LayerNorm(tf_n_embd)
- norm_first=True in TransformerEncoderLayer

# Forward pass now:
1. Embed and normalize contexts
2. Hierarchically compress stages
3. Normalize decoder input
4. Apply hybrid attention mask
5. Decode with proper masking
```

### Attention Mechanism:
```
Before: [latents | context | query] -> all causal masking
After:  [latents: bidirectional | context: causal | query: causal]
```

---

## Configuration Files Updated

1. **config/model/ad_compressed_dr.yaml**: Main darkroom config
2. **config/model/ad_compressed_dktd.yaml**: Darkroom key-to-door config

Both now have:
- Larger model capacity (128 dim, 8 heads)
- Lower learning rate (0.0001)
- Better compression parameters
- Longer training (150k steps)
- Proper warmup (5k steps)

---

## Testing the Fixes

### Quick Test:
```bash
# Train for a few steps and check loss decreases
python train.py
# Watch for "loss" in progress bar - should decrease from ~1.6 to <1.0
```

### Full Evaluation:
```bash
# After training completes
python evaluate.py
# Check mean reward per environment
# Should see positive rewards (>0) for most environments
```

---

## Expected Performance Improvements

### Before Fixes:
- Loss: Not decreasing or very slow
- Eval rewards: Poor (near 0 or negative)
- Accuracy: Random chance (~20% for 5 actions)

### After Fixes:
- Loss: Steady decrease to <0.5
- Eval rewards: Positive, improving with training
- Accuracy: >60% after sufficient training
- Model learns to compress and use context effectively

---

## Additional Notes

### Why These Fixes Work:

1. **Layer Normalization**: Keeps activations in a reasonable range, prevents saturation
2. **Pre-norm Architecture**: Better gradient flow, standard in modern transformers
3. **Hybrid Attention Mask**: Allows latents to aggregate, context stays causal
4. **Better Sampling**: Model actually sees compressed examples during training
5. **Lower LR**: Complex model with bottleneck needs gentle optimization
6. **Proper Warmup**: Prevents early divergence from bad initialization
7. **Larger Capacity**: More parameters to learn compression function
8. **Fewer Latents**: Higher information density per token

### Model Capacity:
- **Old**: ~0.5M parameters
- **New**: ~2M parameters
- Still small enough for fast training but large enough for compression

### GPU Memory:
- Estimated memory: ~4-6 GB
- Batch size 128 is safe for most modern GPUs
- Can reduce to 64 if needed

---

## Verification Checklist

- [x] Encoder has input/output layer norm
- [x] Decoder has input layer norm
- [x] Embeddings are normalized
- [x] Attention mask handles latents correctly
- [x] Dataset generates diverse compression samples
- [x] Learning rate is reduced
- [x] Warmup is sufficient
- [x] Model capacity is increased
- [x] Training steps are increased
- [x] Gradient clipping is in place

All critical issues have been addressed. The model should now train properly!
