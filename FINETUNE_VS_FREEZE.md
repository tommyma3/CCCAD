# Stage 2 Training: Fine-tune vs Freeze Encoder

## Summary

**Stage 2 can do BOTH - you choose via config:**
- **Fine-tune** (`freeze_encoder: false`) - Encoder continues learning ✅ **Recommended**
- **Freeze** (`freeze_encoder: true`) - Only decoder trains

---

## Detailed Comparison

### Option 1: Fine-tune Encoder (Default) ✅

**Configuration:**
```yaml
freeze_encoder: false  # or omit (default is false)
```

**What Gets Updated:**
```
✅ encoder.* (compression encoder)
✅ embed_context.* (shared context embeddings)
✅ embed_ln.* (shared layer norm)
✅ transformer_decoder.* (action decoder)
✅ pred_action.* (action prediction head)
```

**Gradient Flow:**
```
Action Loss → Decoder → Latents → Encoder
    ↑                               ↑
    └─── All components learn ──────┘
```

**Pros:**
- ✅ **Better final performance** - Encoder adapts to task
- ✅ **Task-aware compression** - Learns what info matters for actions
- ✅ **End-to-end optimization** - All components work together
- ✅ **Standard practice** - How pre-training works in deep learning

**Cons:**
- ⚠️ Slightly slower (more parameters to update)
- ⚠️ Needs careful learning rate (risk of forgetting compression)
- ⚠️ May need lower LR than training from scratch

**When to use:**
- Default choice for best performance
- When you have sufficient compute
- When pre-training gave good correlation (>0.6)

---

### Option 2: Freeze Encoder

**Configuration:**
```yaml
freeze_encoder: true
```

**What Gets Updated:**
```
❌ encoder.* (frozen)
❌ embed_context.* (frozen with encoder)
❌ embed_ln.* (frozen with encoder)
✅ transformer_decoder.* (action decoder)
✅ pred_action.* (action prediction head)
```

**Gradient Flow:**
```
Action Loss → Decoder → [Latents] ❌ Encoder (frozen)
    ↑                       ↑
    └─── Only decoder learns ┘
```

**Pros:**
- ✅ **Faster training** - Fewer parameters to optimize
- ✅ **Preserves compression** - Encoder stays exactly as pre-trained
- ✅ **More stable** - No risk of encoder drift or collapse
- ✅ **Simpler** - Decoder only learns to use fixed representations

**Cons:**
- ⚠️ **Lower final performance** - Encoder can't adapt to task
- ⚠️ **Fixed compression** - May not capture task-relevant features
- ⚠️ Decoder must work with whatever encoder provides

**When to use:**
- Training is unstable with fine-tuning
- Limited compute budget
- Pre-training already gives great results
- Debugging (isolate encoder vs decoder issues)

---

## Recommended Workflow

### Step 1: Try Fine-tuning First (Default)
```yaml
# config/model/ad_compressed_dr.yaml
pretrained_encoder_path: './runs/pretrain-encoder-darkroom-seed0/encoder-pretrained-final.pt'
freeze_encoder: false  # or omit this line
lr: 0.0003  # Standard learning rate
```

**Monitor:**
- Loss should decrease steadily
- No wild oscillations
- Better than baseline by 20k steps

### Step 2: If Unstable, Try Frozen
```yaml
freeze_encoder: true
lr: 0.0003  # Can keep same LR
```

**Monitor:**
- More stable loss curve
- May converge faster (fewer parameters)
- Check if final performance is acceptable

### Step 3: If Frozen Works Well, Try Lower LR Fine-tuning
```yaml
freeze_encoder: false
lr: 0.0001  # Much lower LR for careful fine-tuning
```

**Monitor:**
- Should be more stable than default LR
- May take longer but reach higher performance

---

## What Happens During Training

### Fine-tuning (freeze_encoder: false)

**Batch 1:**
```
1. Input trajectory → Embed (trainable)
2. Encoder compresses → Latents (encoder updating)
3. Decoder processes → Actions (decoder updating)
4. Loss computed → Gradients flow to ALL components
```

**Batch 2:**
```
Encoder has changed slightly from Batch 1
→ Produces slightly different latents
→ Decoder adapts to work with evolving encoder
→ Both learn together
```

**Result:** Encoder learns task-aware compression

---

### Frozen (freeze_encoder: true)

**Batch 1:**
```
1. Input trajectory → Embed (frozen)
2. Encoder compresses → Latents (frozen, same every time)
3. Decoder processes → Actions (decoder updating)
4. Loss computed → Gradients only to decoder
```

**Batch 2:**
```
Encoder unchanged from Batch 1
→ Produces identical latents for same input
→ Only decoder learns
→ Encoder provides consistent compression
```

**Result:** Decoder learns to interpret fixed latents

---

## Implementation Details

The code automatically handles both cases:

```python
# From train.py
if config.get('freeze_encoder', False):
    if accelerator.is_main_process:
        print('Freezing encoder and shared embedding weights')
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Freeze shared embeddings (trained with encoder)
    for param in model.embed_context.parameters():
        param.requires_grad = False
    for param in model.embed_ln.parameters():
        param.requires_grad = False
```

**Important:** When freezing encoder, we also freeze `embed_context` and `embed_ln` because they were trained together during pre-training. Keeping them frozen maintains the input representation the encoder was trained on.

---

## Analogy: Pre-trained Language Models

This is exactly how pre-trained models work in NLP:

### BERT Example:
- **Pre-training:** Learn language understanding on massive text
- **Fine-tuning:** Adapt to specific task (sentiment, QA)
  - **Option A:** Fine-tune all layers (better performance)
  - **Option B:** Freeze base, train head only (faster, simpler)

### Your CompressedAD:
- **Pre-training:** Learn compression on trajectories
- **Fine-tuning:** Adapt to action prediction
  - **Option A:** Fine-tune encoder + decoder (better performance)
  - **Option B:** Freeze encoder, train decoder only (faster, simpler)

---

## Performance Expectations

### Fine-tuning (freeze_encoder: false)
```
Expected Results @ 100k steps:
- Loss: 0.6-0.8
- Mean Reward: 0.7-0.8
- Training time: ~2-3 hours (4 GPUs)
```

### Frozen (freeze_encoder: true)
```
Expected Results @ 100k steps:
- Loss: 0.8-1.0  (slightly worse)
- Mean Reward: 0.6-0.7  (slightly worse)
- Training time: ~1.5-2 hours (4 GPUs, faster)
```

### Without Pre-training (baseline)
```
Expected Results @ 100k steps:
- Loss: ~1.6  (random)
- Mean Reward: ~0.4  (random)
- Training time: Doesn't matter, doesn't learn
```

---

## Common Questions

**Q: Won't fine-tuning destroy the pre-trained compression?**

A: No, because:
1. Pre-training gives a strong initialization
2. The compression task (minimizing reconstruction loss) is still implicit in the action prediction task
3. The learning rate during fine-tuning is typically low enough to prevent catastrophic forgetting
4. If concerned, use lower LR (e.g., 0.0001 instead of 0.0003)

**Q: Which is more common in deep learning?**

A: **Fine-tuning is more common**. Examples:
- ImageNet pre-training → fine-tune on specific datasets
- BERT pre-training → fine-tune on downstream tasks
- GPT pre-training → fine-tune on specific applications

Freezing is used mainly when:
- Very small downstream dataset (prevent overfitting)
- Very limited compute
- Transfer learning to very different domain

**Q: Can I freeze some layers but not others?**

A: Yes! You could modify the code to:
```python
# Freeze only first half of encoder layers
for i, layer in enumerate(model.encoder.transformer_encoder.layers):
    if i < len(model.encoder.transformer_encoder.layers) // 2:
        for param in layer.parameters():
            param.requires_grad = False
```

This is called "gradual unfreezing" and is used in ULMFiT and other methods.

---

## Recommendation: Start with Fine-tuning

**Default configuration (recommended):**
```yaml
# config/model/ad_compressed_dr.yaml
pretrained_encoder_path: './runs/pretrain-encoder-darkroom-seed0/encoder-pretrained-final.pt'
freeze_encoder: false  # Fine-tune everything
lr: 0.0003
```

**If unstable, try this:**
```yaml
freeze_encoder: false
lr: 0.0001  # Lower LR for stability
num_warmup_steps: 5000  # More warmup
```

**Last resort (if still unstable):**
```yaml
freeze_encoder: true  # Give up on encoder adaptation
lr: 0.0003
```

---

## Summary Table

| Aspect | Fine-tune | Freeze |
|--------|-----------|--------|
| **Performance** | Best | Good |
| **Training Time** | Slower | Faster |
| **Stability** | May need tuning | Very stable |
| **Complexity** | Higher | Lower |
| **Adaptability** | Encoder adapts | Encoder fixed |
| **Recommended?** | ✅ Yes (default) | For debugging/fast training |
| **Learning Rate** | 0.0001-0.0003 | 0.0003 |
| **Use Case** | Best results | Quick experiments |

---

## Bottom Line

**Stage 2 trains both encoder and decoder by default**, which gives the best performance. This is the standard approach in pre-training literature.

You can optionally freeze the encoder for faster/simpler training, but you'll likely get slightly worse final performance.

**My recommendation: Start with the default (fine-tuning), only freeze if you encounter instability.**
