"""
Parallel training script for latent token ablation study.

This script launches multiple ablation experiments in parallel, 
each on a separate GPU. Useful for multi-GPU systems.

Usage:
    python train_latent_ablation_parallel.py --gpus 0 1 2 3
    python train_latent_ablation_parallel.py --gpus 0 1 --configs latent8 latent16 latent24
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# Ablation configurations: (suffix, n_compress_tokens)
ABLATION_CONFIGS = {
    'latent8': 8,
    'latent16': 16,
    'latent24': 24,
    'latent32': 32,
    'latent48': 48,
}


def run_single_experiment(config_suffix, seed, gpu_id, exp_dir='ablation_latent', env='darkroom'):
    """
    Run a single ablation experiment on a specific GPU.
    
    This function is called in a separate process.
    """
    config_name = f'cad_dr_{config_suffix}'
    run_name = f'{config_name}-seed{seed}'
    log_dir = os.path.join('./runs', exp_dir, run_name)
    
    # Set CUDA device for this process
    env_vars = os.environ.copy()
    env_vars['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, 'train_cad.py',
        '--config', config_name,
        '--env', env,
        '--log_dir', log_dir,
        '--seed', str(seed),
    ]
    
    print(f"[GPU {gpu_id}] Starting {run_name}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env_vars,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ {run_name} completed successfully")
            return (config_suffix, seed, gpu_id, True, None)
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"[GPU {gpu_id}] ✗ {run_name} failed: {error_msg}")
            return (config_suffix, seed, gpu_id, False, error_msg)
            
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ {run_name} exception: {e}")
        return (config_suffix, seed, gpu_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Run latent token ablation in parallel on multiple GPUs')
    parser.add_argument('--gpus', type=int, nargs='+', required=True,
                       help='GPU IDs to use (e.g., --gpus 0 1 2 3)')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configs to run: latent8, latent16, latent24, latent32, latent48')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--exp_dir', type=str, default='ablation_latent',
                       help='Experiment directory under ./runs/')
    parser.add_argument('--env', type=str, default='darkroom',
                       help='Environment name')
    args = parser.parse_args()
    
    # Determine which configs to run
    if args.configs:
        configs = [c for c in args.configs if c in ABLATION_CONFIGS]
    else:
        configs = list(ABLATION_CONFIGS.keys())
    
    n_gpus = len(args.gpus)
    n_configs = len(configs)
    
    print(f"=" * 70)
    print(f"Latent Token Ablation Study - Parallel Execution")
    print(f"=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPUs: {args.gpus}")
    print(f"Configs: {configs}")
    print(f"Seed: {args.seed}")
    print(f"Environment: {args.env}")
    print(f"=" * 70)
    
    # Create experiment directory
    os.makedirs(os.path.join('./runs', args.exp_dir), exist_ok=True)
    
    # Assign configs to GPUs (round-robin if more configs than GPUs)
    tasks = []
    for i, config_suffix in enumerate(configs):
        gpu_id = args.gpus[i % n_gpus]
        tasks.append((config_suffix, args.seed, gpu_id, args.exp_dir, args.env))
    
    print(f"\nTask assignments:")
    for config_suffix, seed, gpu_id, exp_dir, env in tasks:
        print(f"  {config_suffix} -> GPU {gpu_id}")
    print()
    
    # Run experiments in parallel
    # Use min(n_gpus, n_configs) workers to avoid oversubscription
    max_workers = min(n_gpus, n_configs)
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        gpu_in_use = set()
        task_queue = list(tasks)
        
        # Submit initial batch of tasks (one per GPU)
        while task_queue and len(futures) < max_workers:
            config_suffix, seed, gpu_id, exp_dir, env = task_queue.pop(0)
            if gpu_id not in gpu_in_use:
                future = executor.submit(
                    run_single_experiment,
                    config_suffix, seed, gpu_id, exp_dir, env
                )
                futures[future] = (config_suffix, gpu_id)
                gpu_in_use.add(gpu_id)
        
        # Process completed tasks and submit new ones
        while futures:
            # Wait for any task to complete
            done_futures = []
            for future in as_completed(futures):
                done_futures.append(future)
                break  # Process one at a time
            
            for future in done_futures:
                config_suffix, gpu_id = futures.pop(future)
                gpu_in_use.discard(gpu_id)
                
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Task {config_suffix} raised exception: {e}")
                    results.append((config_suffix, args.seed, gpu_id, False, str(e)))
                
                # Submit next task for this GPU if available
                for i, task in enumerate(task_queue):
                    t_config, t_seed, t_gpu, t_exp, t_env = task
                    if t_gpu == gpu_id or t_gpu not in gpu_in_use:
                        task_queue.pop(i)
                        new_future = executor.submit(
                            run_single_experiment,
                            t_config, t_seed, t_gpu, t_exp, t_env
                        )
                        futures[new_future] = (t_config, t_gpu)
                        gpu_in_use.add(t_gpu)
                        break
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()
    
    successful = 0
    for config_suffix, seed, gpu_id, success, error in results:
        n_tokens = ABLATION_CONFIGS[config_suffix]
        if success:
            print(f"  ✓ {config_suffix} (n_compress_tokens={n_tokens}) - GPU {gpu_id}")
            successful += 1
        else:
            print(f"  ✗ {config_suffix} (n_compress_tokens={n_tokens}) - GPU {gpu_id}")
            if error:
                print(f"      Error: {error[:100]}...")
    
    print(f"\nTotal: {successful}/{len(results)} experiments completed successfully")
    
    # Run analysis if all succeeded
    if successful == len(results):
        print("\nRunning analysis...")
        subprocess.run([sys.executable, 'analyze_latent_ablation.py', '--exp_dir', f'./runs/{args.exp_dir}'])


if __name__ == '__main__':
    main()
