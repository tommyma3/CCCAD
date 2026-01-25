"""
Analysis script for n_transit ablation study.

Reads results from all ablation runs and generates comparison plots/tables.

Usage:
    python analyze_transit_ablation.py --exp_dir ./runs/ablation_transit
"""

import argparse
import os
import os.path as path
from glob import glob
import yaml
import numpy as np
import torch
from datetime import datetime


ABLATION_CONFIGS = {
    'transit30': 30,
    'transit45': 45,
    'transit60': 60,
    'transit90': 90,
    'transit120': 120,
}


def load_results(exp_dir):
    """Load results from all ablation runs."""
    results = {}
    
    for config_suffix, n_transit in ABLATION_CONFIGS.items():
        config_name = f'cad_dr_{config_suffix}'
        
        # Find all seed runs for this config
        run_dirs = glob(path.join(exp_dir, f'{config_name}-seed*'))
        
        for run_dir in run_dirs:
            if not path.isdir(run_dir):
                continue
                
            seed = int(run_dir.split('seed')[-1])
            
            # Load config
            config_path = path.join(run_dir, 'config.yaml')
            if path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.full_load(f)
            else:
                config = None
            
            # Load best model info
            best_model_path = path.join(run_dir, 'best-model.pt')
            if path.exists(best_model_path):
                ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                best_reward = ckpt.get('eval_reward', None)
                best_step = ckpt.get('step', None)
            else:
                best_reward = None
                best_step = None
            
            # Load eval result if exists
            eval_result_path = path.join(run_dir, 'eval_result.npy')
            if path.exists(eval_result_path):
                eval_result = np.load(eval_result_path)
            else:
                eval_result = None
            
            if config_suffix not in results:
                results[config_suffix] = {}
            
            results[config_suffix][seed] = {
                'config': config,
                'n_transit': n_transit,
                'best_reward': best_reward,
                'best_step': best_step,
                'eval_result': eval_result,
                'run_dir': run_dir,
            }
    
    return results


def print_summary(results):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("n_transit ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"n_compress_tokens: 16 (fixed)")
    print("-" * 80)
    print(f"{'Config':<15} {'n_transit':<12} {'Seeds':<8} {'Best Reward':<20} {'Best Step':<12}")
    print("-" * 80)
    
    for config_suffix in sorted(ABLATION_CONFIGS.keys(), key=lambda x: ABLATION_CONFIGS[x]):
        if config_suffix not in results:
            print(f"{config_suffix:<15} {ABLATION_CONFIGS[config_suffix]:<12} {'N/A':<8} {'N/A':<20} {'N/A':<12}")
            continue
        
        seeds = list(results[config_suffix].keys())
        rewards = [results[config_suffix][s]['best_reward'] for s in seeds if results[config_suffix][s]['best_reward'] is not None]
        steps = [results[config_suffix][s]['best_step'] for s in seeds if results[config_suffix][s]['best_step'] is not None]
        
        if rewards:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) if len(rewards) > 1 else 0
            reward_str = f"{mean_reward:.4f} ± {std_reward:.4f}"
        else:
            reward_str = "N/A"
        
        if steps:
            mean_step = np.mean(steps)
            step_str = f"{mean_step:.0f}"
        else:
            step_str = "N/A"
        
        n_transit = ABLATION_CONFIGS[config_suffix]
        print(f"{config_suffix:<15} {n_transit:<12} {len(seeds):<8} {reward_str:<20} {step_str:<12}")
    
    print("=" * 80)


def save_summary(results, exp_dir):
    """Save summary to YAML file."""
    summary = {
        'experiment': 'n_transit_ablation',
        'n_compress_tokens': 16,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {}
    }
    
    for config_suffix in sorted(ABLATION_CONFIGS.keys(), key=lambda x: ABLATION_CONFIGS[x]):
        if config_suffix not in results:
            continue
        
        seeds = list(results[config_suffix].keys())
        rewards = [results[config_suffix][s]['best_reward'] for s in seeds if results[config_suffix][s]['best_reward'] is not None]
        steps = [results[config_suffix][s]['best_step'] for s in seeds if results[config_suffix][s]['best_step'] is not None]
        
        summary['results'][config_suffix] = {
            'n_transit': ABLATION_CONFIGS[config_suffix],
            'seeds': seeds,
            'best_rewards': rewards,
            'mean_reward': float(np.mean(rewards)) if rewards else None,
            'std_reward': float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            'mean_best_step': float(np.mean(steps)) if steps else None,
        }
    
    summary_path = path.join(exp_dir, 'ablation_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze n_transit ablation results')
    parser.add_argument('--exp_dir', type=str, default='./runs/ablation_transit',
                       help='Experiment directory')
    args = parser.parse_args()
    
    if not path.exists(args.exp_dir):
        print(f"Error: Directory {args.exp_dir} does not exist")
        return
    
    results = load_results(args.exp_dir)
    
    if not results:
        print("No results found!")
        return
    
    print_summary(results)
    save_summary(results, args.exp_dir)


if __name__ == '__main__':
    main()
