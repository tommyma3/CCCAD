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
import matplotlib.pyplot as plt


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
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.full_load(f)
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
                    config = None
            else:
                config = None
            
            # Load best model info
            best_model_path = path.join(run_dir, 'best-model.pt')
            if path.exists(best_model_path):
                try:
                    ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
                    best_reward = ckpt.get('eval_reward', None)
                    best_step = ckpt.get('step', None)
                except Exception as e:
                    print(f"Warning: Could not load best model from {best_model_path}: {e}")
                    best_reward = None
                    best_step = None
            else:
                best_reward = None
                best_step = None
            
            # Load eval result if exists
            eval_result_path = path.join(run_dir, 'eval_result.npy')
            if path.exists(eval_result_path):
                try:
                    eval_result = np.load(eval_result_path, allow_pickle=True)
                except:
                    eval_result = None
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


def extract_training_curves(results):
    """Extract training curves from tensorboard logs if available."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        for config_suffix in results:
            for seed in results[config_suffix]:
                run_dir = results[config_suffix][seed]['run_dir']
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
                
                results[config_suffix][seed]['training_curves'] = scalars
    except ImportError:
        print("Warning: tensorboard not installed, skipping training curve extraction")


def plot_comparison(results, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by n_transit
    sorted_configs = sorted(ABLATION_CONFIGS.keys(), key=lambda x: ABLATION_CONFIGS[x])
    
    # Extract data for plotting
    n_transits = [ABLATION_CONFIGS[c] for c in sorted_configs]
    
    # Plot 1: Final evaluation reward vs n_transit
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rewards = []
    reward_stds = []
    valid_transits = []
    
    for config_suffix in sorted_configs:
        if config_suffix not in results:
            continue
        
        seeds = list(results[config_suffix].keys())
        seed_rewards = [results[config_suffix][s]['best_reward'] for s in seeds 
                       if results[config_suffix][s]['best_reward'] is not None]
        
        if seed_rewards:
            rewards.append(np.mean(seed_rewards))
            reward_stds.append(np.std(seed_rewards) if len(seed_rewards) > 1 else 0)
            valid_transits.append(ABLATION_CONFIGS[config_suffix])
    
    if rewards:
        ax.bar(range(len(valid_transits)), rewards, yerr=reward_stds, 
               tick_label=valid_transits, capsize=5, color='steelblue')
        ax.set_xlabel('n_transit (context window size)')
        ax.set_ylabel('Best Evaluation Reward')
        ax.set_title('n_transit Ablation: Best Evaluation Reward (n_compress_tokens=16)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path.join(output_dir, 'reward_vs_n_transit.png'), dpi=150)
        plt.close()
        print(f"Saved: {path.join(output_dir, 'reward_vs_n_transit.png')}")
    
    # Plot 2: Available space analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_compress = 16  # Fixed
    available_space = [n - n_compress - 1 for n in n_transits]
    compression_capacity = [(n - n_compress - 1) / n_compress for n in n_transits]
    
    ax.bar(range(len(n_transits)), available_space, tick_label=n_transits, color='coral')
    ax.set_xlabel('n_transit (context window size)')
    ax.set_ylabel(f'Available Space for New Transitions')
    ax.set_title(f'n_transit Ablation: Available Space (n_compress_tokens={n_compress})')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, (n, space) in enumerate(zip(n_transits, available_space)):
        ax.annotate(f'{space}', (i, space), textcoords="offset points", 
                   xytext=(0, 5), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(path.join(output_dir, 'available_space.png'), dpi=150)
    plt.close()
    print(f"Saved: {path.join(output_dir, 'available_space.png')}")
    
    # Plot 3: Reward vs Available Space (scatter)
    if rewards:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid_spaces = [n - n_compress - 1 for n in valid_transits]
        
        ax.scatter(valid_spaces, rewards, s=100, c='steelblue', edgecolors='black')
        
        # Add labels
        for i, (space, reward, n_transit) in enumerate(zip(valid_spaces, rewards, valid_transits)):
            ax.annotate(f'n={n_transit}', (space, reward), textcoords="offset points", 
                       xytext=(5, 5), fontsize=9)
        
        ax.set_xlabel('Available Space (n_transit - n_compress_tokens - 1)')
        ax.set_ylabel('Best Evaluation Reward')
        ax.set_title('Reward vs Available Space')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path.join(output_dir, 'reward_vs_space.png'), dpi=150)
        plt.close()
        print(f"Saved: {path.join(output_dir, 'reward_vs_space.png')}")
    
    # Plot 4: Training curves if available
    has_curves = any(
        'training_curves' in results[c][s] 
        for c in results for s in results[c]
    )
    
    if has_curves:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['train/loss', 'train/acc_action', 'eval/reward', 'train/num_compressions']
        titles = ['Training Loss', 'Training Accuracy', 'Evaluation Reward', 'Number of Compressions']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            for config_suffix in sorted_configs:
                if config_suffix not in results:
                    continue
                
                n_transit = ABLATION_CONFIGS[config_suffix]
                
                # Average across seeds
                all_steps = []
                all_values = []
                
                for seed in results[config_suffix]:
                    curves = results[config_suffix][seed].get('training_curves', {})
                    if metric in curves:
                        all_steps.append(curves[metric]['steps'])
                        all_values.append(curves[metric]['values'])
                
                if all_values:
                    # Simple plot of first seed (or could average)
                    ax.plot(all_steps[0], all_values[0], label=f'n_transit={n_transit}')
            
            ax.set_xlabel('Step')
            ax.set_ylabel(metric.split('/')[-1])
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path.join(output_dir, 'training_curves.png'), dpi=150)
        plt.close()
        print(f"Saved: {path.join(output_dir, 'training_curves.png')}")
    
    print(f"\nAll plots saved to {output_dir}")


def print_summary(results):
    """Print summary table of results."""
    print("\n" + "=" * 90)
    print("n_transit ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"n_compress_tokens: 16 (fixed)")
    print("-" * 90)
    print(f"{'Config':<15} {'n_transit':<12} {'Avail Space':<12} {'Seeds':<8} {'Best Reward':<20} {'Best Step':<12}")
    print("-" * 90)
    
    n_compress = 16
    for config_suffix in sorted(ABLATION_CONFIGS.keys(), key=lambda x: ABLATION_CONFIGS[x]):
        n_transit = ABLATION_CONFIGS[config_suffix]
        avail_space = n_transit - n_compress - 1
        
        if config_suffix not in results:
            print(f"{config_suffix:<15} {n_transit:<12} {avail_space:<12} {'N/A':<8} {'N/A':<20} {'N/A':<12}")
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
        
        print(f"{config_suffix:<15} {n_transit:<12} {avail_space:<12} {len(seeds):<8} {reward_str:<20} {step_str:<12}")
    
    print("=" * 90)


def save_summary(results, exp_dir):
    """Save summary to YAML file."""
    summary = {
        'experiment': 'n_transit_ablation',
        'n_compress_tokens': 16,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {}
    }
    
    n_compress = 16
    for config_suffix in sorted(ABLATION_CONFIGS.keys(), key=lambda x: ABLATION_CONFIGS[x]):
        if config_suffix not in results:
            continue
        
        n_transit = ABLATION_CONFIGS[config_suffix]
        seeds = list(results[config_suffix].keys())
        rewards = [results[config_suffix][s]['best_reward'] for s in seeds if results[config_suffix][s]['best_reward'] is not None]
        steps = [results[config_suffix][s]['best_step'] for s in seeds if results[config_suffix][s]['best_step'] is not None]
        
        summary['results'][config_suffix] = {
            'n_transit': n_transit,
            'available_space': n_transit - n_compress - 1,
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


def generate_report(results, output_dir):
    """Generate a markdown summary report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = path.join(output_dir, 'ablation_report.md')
    
    # Sort by n_transit
    sorted_configs = sorted(ABLATION_CONFIGS.keys(), key=lambda x: ABLATION_CONFIGS[x])
    n_compress = 16
    
    with open(report_path, 'w') as f:
        f.write("# n_transit Ablation Study Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This ablation study varies `n_transit` (context window size) ")
        f.write(f"while keeping `n_compress_tokens` fixed at {n_compress} to understand ")
        f.write("the effect of context length on in-context learning performance.\n\n")
        
        f.write("## Configuration Comparison\n\n")
        f.write("| Config | n_transit | Available Space | Compression Capacity |\n")
        f.write("|--------|-----------|-----------------|---------------------|\n")
        
        for config_suffix in sorted_configs:
            n_transit = ABLATION_CONFIGS[config_suffix]
            available = n_transit - n_compress - 1
            capacity = available / n_compress
            f.write(f"| {config_suffix} | {n_transit} | {available} | {capacity:.2f}x |\n")
        
        f.write("\n## Results\n\n")
        f.write("| Config | n_transit | Best Reward | Best Step | Status |\n")
        f.write("|--------|-----------|-------------|-----------|--------|\n")
        
        for config_suffix in sorted_configs:
            n_transit = ABLATION_CONFIGS[config_suffix]
            
            if config_suffix not in results:
                f.write(f"| {config_suffix} | {n_transit} | N/A | N/A | Not Started |\n")
                continue
            
            seeds = list(results[config_suffix].keys())
            rewards = [results[config_suffix][s]['best_reward'] for s in seeds 
                      if results[config_suffix][s]['best_reward'] is not None]
            steps = [results[config_suffix][s]['best_step'] for s in seeds 
                    if results[config_suffix][s]['best_step'] is not None]
            
            if rewards:
                mean_reward = np.mean(rewards)
                reward_str = f"{mean_reward:.4f}"
                step_str = f"{np.mean(steps):.0f}" if steps else "N/A"
                status = "✓ Complete"
            else:
                reward_str = "N/A"
                step_str = "N/A"
                status = f"In Progress ({len(seeds)} seeds)"
            
            f.write(f"| {config_suffix} | {n_transit} | {reward_str} | {step_str} | {status} |\n")
        
        f.write("\n## Analysis\n\n")
        f.write("### Key Observations\n\n")
        f.write(f"- **Fixed compression**: All experiments use n_compress_tokens={n_compress}\n")
        f.write("- **Available Space**: Tokens available for new transitions = `n_transit - n_compress_tokens - 1`\n")
        f.write("- **Compression Capacity**: How many compressions worth of new data fits = `available_space / n_compress_tokens`\n\n")
        
        f.write("### Trade-offs\n\n")
        f.write("1. **Small n_transit (30)**: Frequent compressions, faster forgetting of old context\n")
        f.write("2. **Large n_transit (120)**: Longer context before compression, more compute per forward pass\n\n")
        
        f.write("### Figures\n\n")
        f.write("![Reward vs n_transit](reward_vs_n_transit.png)\n\n")
        f.write("![Available Space](available_space.png)\n\n")
        f.write("![Reward vs Space](reward_vs_space.png)\n\n")
        
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze n_transit ablation results')
    parser.add_argument('--exp_dir', type=str, default='./runs/ablation_transit',
                       help='Experiment directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: exp_dir/analysis)')
    args = parser.parse_args()
    
    if not path.exists(args.exp_dir):
        print(f"Error: Directory {args.exp_dir} does not exist")
        return
    
    output_dir = args.output_dir if args.output_dir else path.join(args.exp_dir, 'analysis')
    
    print(f"Analyzing experiments in: {args.exp_dir}")
    
    results = load_results(args.exp_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} configurations")
    
    # Extract training curves if possible
    extract_training_curves(results)
    
    # Print summary table
    print_summary(results)
    
    # Generate plots
    plot_comparison(results, output_dir)
    
    # Generate report
    generate_report(results, output_dir)
    
    # Save summary YAML
    save_summary(results, args.exp_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
