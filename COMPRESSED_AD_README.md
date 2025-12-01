# Compressed Algorithm Distillation (CompressedAD)

## Overview

This implementation extends Algorithm Distillation with a **hierarchical compression mechanism** to handle arbitrarily long context windows during evaluation.

### Problem
Standard AD uses simple truncation when the maximum sequence length is reached, discarding the oldest tokens. This loses potentially important historical information.

### Solution
**CompressedAD** uses a two-model architecture:
1. **Compression Encoder**: Compresses old context into fixed-size latent tokens
2. **Decoder**: Processes latent tokens + recent context to predict actions

When the sequence length exceeds the maximum, the encoder compresses `[old_latents + some_context]` → `new_latents`, maintaining a hierarchical compression of the entire history.

---

## Architecture

### Compression Encoder
- **Input**: Context embeddings (from previous latents and/or new transitions) + learnable query tokens
- **Output**: `n_latent` compressed tokens (default: 60 tokens)
- **Purpose**: Extract and compress relevant information from history

The encoder uses learnable "query tokens" that act as compression slots, learning to extract the most relevant information for future action prediction.

### Decoder
- **Input**: `[latent_tokens (if exist)] + [recent_context] + [query_state]`
- **Output**: Action prediction
- **Purpose**: Autoregressive action prediction using compressed + uncompressed context

---

## Hierarchical Compression Process

### During Training
Training data simulates different compression scenarios with 0-3 compression cycles:

**Example with 3 compression cycles:**
```
Stage 1: [t0...t19] → Encoder → L1 (60 latents)
Stage 2: [L1, t20...t39] → Encoder → L2 (60 latents)  
Stage 3: [L2, t40...t59] → Encoder → L3 (60 latents)
Decoder: [L3, t60...t89, query_t90] → predict action_t90
```

Each sample randomly selects:
- Compression depth (0-3 cycles)
- Segment lengths for each stage
- Recent uncompressed context length

This trains the model to handle various compression scenarios.

### During Evaluation
When `sequence_length > decoder_max_seq_length`:
```python
# Compress old context
keep_recent = max_seq_length - n_latent - 1
context_to_compress = [latent_tokens, context[:old]]
context_to_keep = context[old:]

# Generate new latents
new_latents = encoder(context_to_compress)

# Continue with compressed history
decoder_input = [new_latents, context_to_keep, query_state]
```

The model maintains memory of the entire episode through hierarchical compression.

---

## Training

### Configuration

Edit `config/model/ad_compressed_dr.yaml`:

```yaml
model: CompressedAD

# Decoder settings
decoder_max_seq_length: 100  # Max before compression
tf_n_embd: 64
tf_n_layer: 4
tf_n_head: 4

# Encoder settings
encoder_n_layer: 3  # Can differ from decoder
encoder_n_head: 4
n_latent: 60  # Number of latent tokens

# Compression parameters
max_compression_depth: 3  # 0-3 compression cycles
min_compress_length: 10
max_compress_length: 50
min_uncompressed_length: 5
max_uncompressed_length: 30

# Training
train_batch_size: 256
train_timesteps: 100000
lr: 0.0003
```

### Running Training

Train the compressed model:
```bash
python train.py
```

The script automatically detects `CompressedAD` and uses `ADCompressedDataset` with the custom collate function.

**Multi-GPU Training:**
The implementation uses Hugging Face Accelerator, which automatically handles multi-GPU training:
```bash
# Single GPU
python train.py

# Multi-GPU (automatic detection)
python train.py
```

### Key Training Features

1. **Cross-Episode History**: Dataset maintains `history_idx` to ensure samples come from the same trajectory, preserving cross-episode learning

2. **Variable Compression Depth**: Each batch contains mixed samples (0-3 compression cycles) to train robust compression

3. **Data Augmentation**: Single 50-step trajectory generates ~10-20 training samples with different compression configurations

4. **End-to-End Training**: Encoder and decoder trained jointly with action prediction loss

---

## Evaluation

```bash
python evaluate.py
```

During evaluation:
- Model starts with no latents
- As context grows, compression automatically triggers when `seq_length > max_seq_length`
- Hierarchical compression maintains full episode history
- No manual intervention needed

### Monitoring Compression
To track compression behavior, you can add logging:
```python
# In CompressedAD.evaluate_in_context()
if current_seq_len > self.max_seq_length:
    print(f"Step {step}: Compressing {current_seq_len} → {self.n_latent + keep_recent}")
```

---

## Key Design Decisions

### 1. Latent Token Nature
- **Latent tokens are abstract learned representations**, not interpretable (s,a,r,s') tokens
- Similar to memory tokens in Perceiver or BERT's [CLS] token
- The decoder learns to interpret these during joint training

### 2. Query Tokens
- Learnable parameters: `nn.Parameter(torch.randn(1, n_latent, tf_n_embd))`
- Act as "compression slots" asking: "What information should I remember?"
- Initialized randomly and learned end-to-end

### 3. Hierarchical Compression
- **Approach 2**: Compress `[old_latents + new_context]` → `new_latents`
- Maintains memory of entire episode through multiple compression cycles
- Better for long-horizon tasks requiring distant history

### 4. Training Without Long Trajectories
- Don't need trajectories longer than current episodes
- Simulate compression by splitting existing trajectories
- Random compression points create diverse training scenarios

---

## Comparison with Standard AD

| Feature | Standard AD | CompressedAD |
|---------|------------|--------------|
| **Context Length** | Fixed (n_transit) | Unlimited |
| **Memory** | Simple truncation | Hierarchical compression |
| **History Retention** | Only recent `n_transit` steps | Entire episode (compressed) |
| **Complexity** | Single transformer | Encoder + Decoder |
| **Training Data** | Simple sliding window | Multi-stage compression |
| **Evaluation** | Fixed context window | Dynamic compression |

---

## Expected Benefits

1. **Long-Horizon Learning**: Maintain memory of early episode events
2. **Better In-Context Learning**: Don't lose information about learning trajectory
3. **Flexible Context**: Handle varying episode lengths without fixed window
4. **Learned Compression**: Model learns what to remember vs. forget

---

## Hyperparameter Tuning

### Compression Parameters

**`n_latent` (default: 60)**
- Larger → More information retained, but larger computation
- Smaller → More aggressive compression, potential information loss
- Rule of thumb: `n_latent ≈ 0.5 * decoder_max_seq_length`

**`decoder_max_seq_length` (default: 100)**
- Trigger point for compression
- Larger → Less frequent compression, more uncompressed context
- Smaller → More frequent compression, more reliance on encoder

**`max_compression_depth` (default: 3)**
- Maximum nested compression cycles in training
- Higher → More robust to long episodes, but harder to train
- Start with 2-3 for most tasks

### Encoder vs. Decoder Size

**Option 1: Equal Size** (default)
```yaml
encoder_n_layer: 4
tf_n_layer: 4
```
Balanced approach, encoder and decoder have similar capacity.

**Option 2: Smaller Encoder**
```yaml
encoder_n_layer: 2
tf_n_layer: 4
```
Faster compression, but may lose information.

**Option 3: Larger Encoder**
```yaml
encoder_n_layer: 6
tf_n_layer: 4
```
Better compression quality, but slower training.

---

## Troubleshooting

### Issue: Model not learning to compress
**Symptoms**: Low accuracy, compression seems to hurt performance
**Solutions**:
1. Start with `max_compression_depth: 1` for simpler learning
2. Increase `n_latent` to retain more information
3. Pre-train encoder with reconstruction loss (see below)

### Issue: Out of memory during training
**Solutions**:
1. Reduce `train_batch_size`
2. Reduce `max_compress_length` to shorten sequences
3. Reduce `encoder_n_layer` or `tf_n_layer`

### Issue: Slow training
**Solutions**:
1. Reduce `max_compression_depth` (fewer nested compressions)
2. Use smaller encoder: `encoder_n_layer: 2`
3. Reduce `max_uncompressed_length`

---

## Advanced: Pre-training Encoder

For better initialization, you can pre-train the encoder with reconstruction loss:

```python
# Add to CompressedAD.__init__()
self.reconstruction_head = nn.Linear(tf_n_embd, tf_n_embd)

# Pre-training forward pass
def pretrain_step(self, context_dict):
    context_embed = self._embed_context_dict(context_dict)
    latent_tokens = self.encoder(context_embed)
    
    # Try to reconstruct original embeddings from latents
    reconstructed = self.reconstruction_head(latent_tokens.mean(dim=1))
    original = context_embed.mean(dim=1)
    
    loss = F.mse_loss(reconstructed, original)
    return loss
```

Pre-train for ~10k steps, then switch to end-to-end training.

---

## Future Extensions

1. **Adaptive Compression**: Learn when to compress vs. keep more context
2. **Selective Compression**: Compress only less important transitions
3. **Multi-resolution**: Multiple compression levels (e.g., 60, 30, 15 latents)
4. **Attention Visualization**: Analyze what information latents capture

---

## File Structure

```
model/
  ad.py                    # Original AD model
  ad_compressed.py         # New CompressedAD model
  __init__.py             # Model registry

dataset.py                # ADDataset + ADCompressedDataset

config/
  model/
    ad_dr.yaml            # Original AD config
    ad_compressed_dr.yaml # CompressedAD config

train.py                  # Training script (supports both models)
evaluate.py              # Evaluation script (supports both models)
```

---

## Citation

Based on:
- **Algorithm Distillation**: [arXiv:2210.14215](https://arxiv.org/abs/2210.14215)
- **Compression Mechanism**: Custom hierarchical compression with learnable query tokens

---

## Questions?

For implementation details, see:
- `model/ad_compressed.py`: Model architecture
- `dataset.py`: `ADCompressedDataset` class
- `config/model/ad_compressed_dr.yaml`: Configuration options
