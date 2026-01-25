"""
Analyze and compare results from the latent token ablation study.

This script loads evaluation results from all ablation experiments,
generates comparison plots, and produces a summary report.

Usage:
    python analyze_latent_ablation.py [--exp_dir ablation_latent] [--output_dir ablation_results]
"""

import argparse
import os
import os.path as path
from glob import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch


def load_experiment_results(exp_dir):
    """Load results from all experiments in the ablation directory."""
    results = []
    
    run_dirs = sorted(glob(path.join(exp_dir, '*')))
    
    for run_dir in run_dirs:
        if not path.isdir(run_dir):
            continue
            
        run_name = path.basename(run_dir)
        
        # Load config
        config_path = path.join(run_dir, 'config.yaml')
        if not path.exists(config_path):
            print(f"Warning: No config found in {run_dir}")
            continue
        
        # Try to load config, handling torch objects that may be serialized
        try:
            with open(config_path, 'r') as f:
                config = yaml.full_load(f)  # Can handle torch.device and other Python objects
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            # Try to extract key info from checkpoint instead
            config = None
            ckpt_paths = sorted(glob(path.join(run_dir, 'ckpt-*.pt')))
            if ckpt_paths:
                try:
                    ckpt = torch.load(ckpt_paths[-1], map_location='cpu', weights_only=False)
                    config = ckpt.get('config', {})
                except:
                    pass
            if config is None:
                print(f"Warning: Skipping {run_dir} - could not load config")
                continue
        
        # Load evaluation results
        eval_path = path.join(run_dir, 'eval_result.npy')
        if path.exists(eval_path):
            eval_result = np.load(eval_path, allow_pickle=True).item()
        else:
            eval_result = None
        
        # Find checkpoints
        ckpt_paths = sorted(glob(path.join(run_dir, 'ckpt-*.pt')))
        best_model_path = path.join(run_dir, 'best-model.pt')
        
        # Try to load best model info
        best_info = None
        if path.exists(best_model_path):
            try:
                ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                best_info = {
                    'step': ckpt.get('step', 'N/A'),
                    'eval_reward': ckpt.get('eval_reward', 'N/A'),
                }
            except Exception as e:
                print(f"Warning: Could not load best model from {best_model_path}: {e}")
        
        results.append({
            'run_name': run_name,
            'run_dir': run_dir,
            'config': config,
            'n_compress_tokens': config.get('n_compress_tokens', 16),
            'n_transit': config.get('n_transit', 60),
            'eval_result': eval_result,
            'num_checkpoints': len(ckpt_paths),
            'best_info': best_info,
            'has_best_model': path.exists(best_model_path),
        })
    
    return results


def extract_training_curves(results):
    """Extract training curves from tensorboard logs if available."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        for result in results:
            run_dir = result['run_dir']
            event_files = glob(path.join(run_dir, 'events.out.tfevents.*'))
            
            if not event_files:
                continue
            
            ea = event_accumulator.EventAccumulator(event_files[0])
            ea.Reload()
            
            # Extract scalars
            scalars = {}
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                scalars[tag] = {
                    'steps': [e.step for e in events],
                    'values': [e.value for e in events],
                }
            
            result['training_curves'] = scalars
    except ImportError:
        print("Warning: tensorboard not installed, skipping training curve extraction")


def plot_comparison(results, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by n_compress_tokens
    results = sorted(results, key=lambda x: x['n_compress_tokens'])
    
    # Extract data for plotting
    n_tokens = [r['n_compress_tokens'] for r in results]
    
    # Plot 1: Final evaluation reward vs latent tokens
    if any(r['best_info'] is not None for r in results):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rewards = []
        valid_tokens = []
        for r in results:
            if r['best_info'] and r['best_info']['eval_reward'] != 'N/A':
                rewards.append(r['best_info']['eval_reward'])
                valid_tokens.append(r['n_compress_tokens'])
        
        if rewards:
            ax.bar(range(len(valid_tokens)), rewards, tick_label=valid_tokens)
            ax.set_xlabel('Number of Latent Tokens (n_compress_tokens)')
            ax.set_ylabel('Best Evaluation Reward')
            ax.set_title('Latent Token Ablation: Best Evaluation Reward')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(path.join(output_dir, 'reward_vs_latent_tokens.png'), dpi=150)
            plt.close()
    
    # Plot 2: Compression ratio analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_transit = results[0]['n_transit'] if results else 60
    available_space = [n_transit - n - 1 for n in n_tokens]
    compression_ratios = [n_transit / n for n in n_tokens]
    
    ax.bar(range(len(n_tokens)), compression_ratios, tick_label=n_tokens, color='steelblue')
    ax.set_xlabel('Number of Latent Tokens (n_compress_tokens)')
    ax.set_ylabel(f'Compression Ratio (n_transit={n_transit} / n_tokens)')
    ax.set_title('Latent Token Ablation: Compression Ratios')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations for available space
    for i, (n, space) in enumerate(zip(n_tokens, available_space)):
        ax.annotate(f'{space} free', (i, compression_ratios[i]), 
                   textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(path.join(output_dir, 'compression_ratios.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_report(results, output_dir):
    """Generate a markdown summary report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = path.join(output_dir, 'ablation_report.md')
    
    # Sort by n_compress_tokens
    results = sorted(results, key=lambda x: x['n_compress_tokens'])
    
    with open(report_path, 'w') as f:
        f.write("# Latent Token Ablation Study Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This ablation study varies the number of latent tokens (`n_compress_tokens`) ")
        f.write("while keeping `n_transit` constant to understand the effect of compression ")
        f.write("aggressiveness on in-context learning performance.\n\n")
        
        f.write("## Configuration Comparison\n\n")
        f.write("| Config | n_compress_tokens | n_transit | Available Space | Compression Ratio |\n")
        f.write("|--------|-------------------|-----------|-----------------|-------------------|\n")
        
        for r in results:
            n_tokens = r['n_compress_tokens']
            n_transit = r['n_transit']
            available = n_transit - n_tokens - 1
            ratio = n_transit / n_tokens
            f.write(f"| {r['run_name']} | {n_tokens} | {n_transit} | {available} | {ratio:.2f}x |\n")
        
        f.write("\n## Results\n\n")
        f.write("| Config | Best Step | Best Reward | Status |\n")
        f.write("|--------|-----------|-------------|--------|\n")
        
        for r in results:
            if r['best_info']:
                step = r['best_info']['step']
                reward = r['best_info']['eval_reward']
                if isinstance(reward, float):
                    reward = f"{reward:.4f}"
                status = "✓ Complete"
            elif r['num_checkpoints'] > 0:
                step = "N/A"
                reward = "N/A"
                status = f"In Progress ({r['num_checkpoints']} ckpts)"
            else:
                step = "N/A"
                reward = "N/A"
                status = "Not Started"
            
            f.write(f"| {r['run_name']} | {step} | {reward} | {status} |\n")
        
        f.write("\n## Analysis\n\n")
        f.write("### Key Observations\n\n")
        f.write("- **Available Space**: After compression, the number of tokens available for ")
        f.write("new transitions before the next compression is `n_transit - n_compress_tokens - 1`.\n")
        f.write("- **Compression Ratio**: Higher compression ratio means more information is ")
        f.write("compressed into fewer tokens.\n")
        f.write("- Lower `n_compress_tokens` = more aggressive compression = less information preserved\n")
        f.write("- Higher `n_compress_tokens` = less aggressive compression = more information preserved, ")
        f.write("but less room for new transitions\n\n")
        
        f.write("### Trade-offs\n\n")
        f.write("1. **Very few latent tokens (8)**: Maximum new context space (51 tokens), ")
        f.write("but may lose important historical information.\n")
        f.write("2. **Many latent tokens (48)**: Preserves more history, but only 11 tokens ")
        f.write("for new transitions before next compression.\n")
        
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze latent token ablation results')
    parser.add_argument('--exp_dir', type=str, default='./runs/ablation_latent',
                       help='Directory containing ablation experiments')
    parser.add_argument('--output_dir', type=str, default='./runs/ablation_latent/analysis',
                       help='Output directory for analysis results')
    args = parser.parse_args()
    
    print(f"Analyzing experiments in: {args.exp_dir}")
    
    # Load results
    results = load_experiment_results(args.exp_dir)
    
    if not results:
        print("No experiment results found!")
        return
    
    print(f"Found {len(results)} experiments")
    
    # Extract training curves if possible
    extract_training_curves(results)
    
    # Generate plots
    plot_comparison(results, args.output_dir)
    
    # Generate report
    generate_report(results, args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
