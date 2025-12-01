# Quick Start: Training CompressedAD

## 1. Verify Configuration

Check that the config file exists:
```bash
cat config/model/ad_compressed_dr.yaml
```

Key settings:
- `model: CompressedAD`
- `n_latent: 60` (number of compression tokens)
- `decoder_max_seq_length: 100` (when to trigger compression)

## 2. Train the Model

Start training with the compressed model:
```bash
python train.py
```

Make sure to update `train.py` line 31 to use the compressed config:
```python
config.update(get_config('./config/model/ad_compressed_dr.yaml'))  # Changed from ad_dr.yaml
```

The training will automatically:
- Load `ADCompressedDataset` 
- Use custom `collate_compressed_batch`
- Train encoder and decoder end-to-end
- Support multi-GPU via Accelerator

## 3. Monitor Training

Check TensorBoard logs:
```bash
tensorboard --logdir=./runs
```

Look for:
- `train/loss_action`: Should decrease over time
- `train/acc_action`: Should increase over time
- Compression stages are handled automatically

## 4. Evaluate the Model

After training completes:
```bash
python evaluate.py
```

Update the checkpoint directory in `evaluate.py` line 23:
```python
ckpt_dir = './runs/CompressedAD-darkroom-seed0'  # Match your log_dir
```

## 5. Expected Behavior

During evaluation:
- First 100 steps: No compression (building context)
- Step 101+: Automatic compression triggers
- Hierarchical: Old latents + new context → new latents
- Model should maintain performance across long episodes

## Training Time Estimates

On single GPU (RTX 3090):
- ~2-3 hours for 100k steps with batch_size=256
- ~4-6 hours for full training

Multi-GPU:
- Scales linearly with number of GPUs
- 2 GPUs → ~1-1.5 hours

## Key Differences from Standard AD

1. **Config**: Use `ad_compressed_dr.yaml` instead of `ad_dr.yaml`
2. **Model**: `CompressedAD` instead of `AD`
3. **Dataset**: Automatically uses `ADCompressedDataset`
4. **Evaluation**: Automatic compression during long rollouts

## Troubleshooting

**Issue: "KeyError: 'n_latent'"**
- Solution: Make sure you're using `config/model/ad_compressed_dr.yaml`

**Issue: Out of memory**
- Solution: Reduce `train_batch_size` in config (try 128 or 64)

**Issue: Model not loading**
- Solution: Check `model/__init__.py` includes `CompressedAD` in `MODEL` dict

**Issue: Slow training**
- Solution: Reduce `max_compression_depth` to 2 or reduce `encoder_n_layer` to 2

## Next Steps

After successful training:
1. Compare performance with standard AD
2. Experiment with `n_latent` (try 30, 60, 90)
3. Try different `decoder_max_seq_length` (50, 100, 150)
4. Visualize compression behavior by adding logging

See `COMPRESSED_AD_README.md` for detailed documentation.
