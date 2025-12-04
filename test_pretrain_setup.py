# Quick test script to verify pre-training setup
# This does NOT run full training, just checks if everything loads correctly

import sys
import torch
from pathlib import Path

print("=" * 60)
print("Pre-training Setup Verification")
print("=" * 60)

# Test 1: Import modules
print("\n[Test 1] Checking imports...")
try:
    from dataset_pretrain import ADCompressedPretrainDataset, collate_pretrain_batch
    print("✓ dataset_pretrain imports OK")
except Exception as e:
    print(f"✗ dataset_pretrain import failed: {e}")
    sys.exit(1)

try:
    from model.ad_compressed import CompressionEncoder
    print("✓ CompressionEncoder import OK")
except Exception as e:
    print(f"✗ CompressionEncoder import failed: {e}")
    sys.exit(1)

try:
    import pretrain_encoder
    print("✓ pretrain_encoder imports OK")
except Exception as e:
    print(f"✗ pretrain_encoder import failed: {e}")
    sys.exit(1)

# Test 2: Load config
print("\n[Test 2] Checking configuration...")
try:
    from utils import get_config
    
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/ad_compressed_dr.yaml'))
    
    config['device'] = 'cpu'
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'no'  # Required for model initialization
    
    print(f"✓ Config loaded")
    print(f"  - n_latent: {config.get('n_latent', 'NOT SET')}")
    print(f"  - encoder_n_layer: {config.get('encoder_n_layer', 'NOT SET')}")
    print(f"  - max_compression_depth: {config.get('max_compression_depth', 'NOT SET')}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# Test 3: Create encoder model
print("\n[Test 3] Creating encoder model...")
try:
    encoder = CompressionEncoder(config)
    print(f"✓ CompressionEncoder created")
    print(f"  - Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
except Exception as e:
    print(f"✗ Encoder creation failed: {e}")
    sys.exit(1)

# Test 4: Create pre-training model
print("\n[Test 4] Creating pre-training model...")
try:
    model = pretrain_encoder.EncoderPretrainModel(config)
    print(f"✓ EncoderPretrainModel created")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  - Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
except Exception as e:
    print(f"✗ Pre-training model creation failed: {e}")
    sys.exit(1)

# Test 5: Check dataset
print("\n[Test 5] Checking dataset...")
try:
    # Check if dataset files exist
    from utils import get_traj_file_name
    dataset_path = Path(f'./datasets/{get_traj_file_name(config)}.hdf5')
    
    if not dataset_path.exists():
        print(f"⚠ Dataset file not found: {dataset_path}")
        print(f"  This is OK if you haven't collected data yet.")
        print(f"  Run collect.py first to generate training data.")
    else:
        print(f"✓ Dataset file found: {dataset_path}")
        
        # Try loading a small sample
        dataset = ADCompressedPretrainDataset(
            config=config,
            traj_dir='./datasets',
            mode='train',
            n_stream=1,
            source_timesteps=100
        )
        print(f"✓ Dataset created successfully")
        print(f"  - Total samples: {len(dataset)}")
        
        # Try getting one sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample retrieved")
            print(f"  - Compression depth: {sample['num_stages']}")
            print(f"  - Number of stages: {len(sample['encoder_input_stages'])}")
except Exception as e:
    print(f"⚠ Dataset check failed: {e}")
    print(f"  This is expected if you haven't run collect.py yet")

# Test 6: Forward pass
print("\n[Test 6] Testing forward pass...")
try:
    # Create dummy input
    batch_size = 2
    seq_len = 10
    
    dummy_batch = {
        'encoder_input_stages': [
            {
                'states': torch.randn(batch_size, seq_len, config['dim_states']),
                'actions': torch.randn(batch_size, seq_len, config['num_actions']),
                'rewards': torch.randn(batch_size, seq_len, 1),
                'next_states': torch.randn(batch_size, seq_len, config['dim_states'])
            }
        ],
        'num_stages': 1
    }
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_batch)
    
    print(f"✓ Forward pass successful")
    print(f"  - Loss: {outputs['loss'].item():.4f}")
    print(f"  - Correlation: {outputs['correlation'].item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Checkpoint format
print("\n[Test 7] Testing checkpoint format...")
try:
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'embed_context_state_dict': model.embed_context.state_dict(),
        'embed_ln_state_dict': model.embed_ln.state_dict(),
        'global_step': 0,
        'config': config
    }
    
    # Try saving and loading
    test_path = './test_checkpoint.pt'
    torch.save(checkpoint, test_path)
    loaded_checkpoint = torch.load(test_path, map_location='cpu')
    
    # Verify keys
    required_keys = ['encoder_state_dict', 'embed_context_state_dict', 'embed_ln_state_dict']
    for key in required_keys:
        if key not in loaded_checkpoint:
            raise ValueError(f"Missing key: {key}")
    
    # Clean up
    Path(test_path).unlink()
    
    print(f"✓ Checkpoint format valid")
    print(f"  - All required keys present")
except Exception as e:
    print(f"✗ Checkpoint test failed: {e}")
    sys.exit(1)

# Test 8: Check if main training can load
print("\n[Test 8] Testing checkpoint loading in main model...")
try:
    from model import MODEL
    
    main_model = MODEL['CompressedAD'](config)
    
    # Load checkpoint
    main_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    main_model.embed_context.load_state_dict(checkpoint['embed_context_state_dict'])
    main_model.embed_ln.load_state_dict(checkpoint['embed_ln_state_dict'])
    
    print(f"✓ Main model successfully loaded pretrained weights")
except Exception as e:
    print(f"✗ Main model loading failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now:")
print("1. Run pre-training:")
print("   accelerate launch --config_file accelerate_config.yaml pretrain_encoder.py")
print("\n2. Or use the automated script:")
print("   .\\train_two_stage.ps1")
print("\n3. Make sure you have training data first:")
print("   python collect.py")
print("=" * 60)
