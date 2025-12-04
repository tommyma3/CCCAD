# Retraining Guide for CompressedAD

## Why Retrain?

The previous checkpoint (`ckpt-100000.pt`) was trained with **buggy code** where actions were incorrectly encoded:
- **Bug**: One-hot vectors `[0,0,0,1]` were cast to long tensors as `[0,0,0,1]` instead of extracting index `3`
- **Impact**: Model received completely wrong training labels for 100,000 steps
- **Result**: Random performance (mean reward 1.23 instead of 5-10)

## Fixes Applied

✅ **Action Encoding Fix** (dataset.py):
```python
# Now correctly converts one-hot to index
if len(target_action_data.shape) == 1 and target_action_data.shape[0] > 1:
    target_action = torch.tensor(np.argmax(target_action_data), dtype=torch.long)
```

✅ **BucketSampler** (dataset.py):
- Groups samples by compression depth into separate batches
- Eliminates dummy padding for missing stages
- Reduces wasted computation by ~30-40%

✅ **Multiprocessing Fix** (train.py):
```python
num_workers=0  # Required with custom batch_sampler
```

✅ **Simplified Collate** (dataset.py):
- No longer creates dummy stages for samples with different depths
- Only minimal within-stage padding

## Retraining Steps

### 1. Clear Old Checkpoint

```powershell
# Remove buggy checkpoint
Remove-Item -Recurse -Force runs\CompressedAD-darkroom-seed0

# Or rename to keep as backup
Move-Item runs\CompressedAD-darkroom-seed0 runs\CompressedAD-darkroom-seed0-buggy-backup
```

### 2. Start Fresh Training

```powershell
# Start training with all fixes
python train.py config/algorithm/ppo_base.yaml config/env/darkroom.yaml config/model/ad_compressed_dr.yaml

# In another terminal, monitor with TensorBoard
tensorboard --logdir=runs
```

### 3. Monitor Training Progress

Open TensorBoard at `http://localhost:6006` and watch these metrics:

**Expected Training Curves:**

| Metric | Initial | Target (100k steps) | Status |
|--------|---------|---------------------|--------|
| `train/loss_action` | ~1.6 | 0.3-0.5 | Should decrease steadily |
| `train/acc_action` | ~20% | 70-90% | Should increase steadily |
| `test/loss_action` | ~1.6 | 0.4-0.6 | Should track train loss |
| `test/acc_action` | ~20% | 65-85% | Should track train accuracy |

**Warning Signs:**
- ❌ Loss stays above 1.0 after 20k steps → something wrong
- ❌ Accuracy plateaus below 40% → check data or model
- ❌ Loss increases or spikes → learning rate too high

### 4. Evaluate After Training

```powershell
# After ~50k steps, checkpoint should start working
python evaluate.py

# Expected results:
# Mean reward per environment: 5-10 per episode
# Overall mean reward: 5-10
# Actions should show clear progress toward goals
```

## Training Time Estimates

- **With GPU**: ~2-3 hours for 100k steps
- **With CPU**: ~8-12 hours for 100k steps
- **First useful checkpoint**: ~30-50k steps
- **Recommended final checkpoint**: 80-100k steps

## Troubleshooting

### If Training Still Fails

**Option A: Verify Baseline AD Works**

Train standard AD first to ensure data/setup is correct:

```powershell
python train.py config/algorithm/ppo_base.yaml config/env/darkroom.yaml config/model/ad_dr.yaml
```

If standard AD works but CompressedAD doesn't, the issue is in the compression logic.

**Option B: Check Data Quality**

```powershell
python test_bucketing.py
```

Should show:
- Buckets created for each depth (0-3)
- Samples properly distributed
- No assertion errors

**Option C: Reduce Batch Size**

If memory issues occur:

Edit `config/model/ad_compressed_dr.yaml`:
```yaml
train_batch_size: 128  # Reduce from 256
```

### If Evaluation Shows Poor Performance After 100k Steps

Check TensorBoard metrics:
1. **If loss is low (~0.3) but evaluation poor**: Model may be overfitting or eval setup issue
2. **If loss is high (~1.0+)**: Training didn't converge, train longer or check learning rate
3. **If accuracy is high (~80%) but rewards low**: Model predicts well on training data but doesn't generalize

## Expected Evaluation Output

After successful training, evaluation should look like:

```
Mean reward per environment: [8.2  7.5  9.1  6.8  8.5  7.2  8.9  7.6]
Overall mean reward: 8.0
```

With action sequences showing clear goal-directed behavior (moving consistently toward target positions).

## Next Steps After Successful Training

1. **Compare with baseline AD**: Train standard AD and compare performance
2. **Analyze compression**: Check if compression is actually helping or hurting
3. **Tune hyperparameters**: Adjust `n_latent`, `max_uncompressed_length`, etc.
4. **Test on longer episodes**: See if compression maintains performance over time
