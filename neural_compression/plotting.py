"""
Plotting utilities for NeurLZ training visualization.

This module provides functions to visualize training curves, model comparisons,
and performance metrics from NeurLZ compression experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import re


def plot_training_curves(stats, error_bound, relative_error_bound, output_path=None, title_suffix=""):
    """
    Plot training and validation loss curves.
    
    Args:
        stats: Statistics dictionary from compress()
        error_bound: Error bound used for compression
        relative_error_bound: Relative error bound used for compression
        output_path: Path to save the plot (optional)
        title_suffix: Additional text for plot title
    """
    train_losses = stats.get('train_losses', [])
    val_losses = stats.get('val_losses', [])
    
    if not train_losses:
        print("No training losses to plot. Set track_losses=True during compression.")
        return
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'Training Curves - Error Bound: {error_bound:.2e} Relative Error Bound: {relative_error_bound:.2e} {title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for loss curves
    
    # Add final loss values as text
    final_train = train_losses[-1]
    text = f'Final Train Loss: {final_train:.6f}'
    if val_losses:
        final_val = val_losses[-1]
        text += f'\nFinal Val Loss: {final_val:.6f}'
    
    plt.text(0.98, 0.98, text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(stats_list, model_names, metric='loss', output_path=None, title="Model Comparison"):
    """
    Plot comparison of multiple models' training curves.
    
    Args:
        stats_list: List of stats dictionaries from compress()
        model_names: List of model names (e.g., ['TinyResidualPredictor', 'FrequencyTinyResidualPredictor'])
        metric: 'loss' for training loss, 'psnr' for PSNR history
        output_path: Path to save the plot (optional)
        title: Plot title
    """
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    if metric == 'loss':
        # Plot training and validation losses
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, (stats, name) in enumerate(zip(stats_list, model_names)):
            train_losses = stats.get('train_losses', [])
            val_losses = stats.get('val_losses', [])
            
            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                color = colors[i % len(colors)]
                
                ax1.plot(epochs, train_losses, color=color, linestyle='-', 
                        label=f'{name}', linewidth=2, alpha=0.8)
                
                if val_losses:
                    ax2.plot(epochs, val_losses, color=color, linestyle='-',
                            label=f'{name}', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax1.set_title('Training Loss Comparison', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Loss (MSE)', fontsize=12)
        ax2.set_title('Validation Loss Comparison', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
    elif metric == 'psnr':
        # Plot PSNR history
        plt.figure(figsize=(12, 6))
        for i, (stats, name) in enumerate(zip(stats_list, model_names)):
            psnr_history = stats.get('psnr_history', [])
            
            if psnr_history:
                epochs, psnrs = zip(*psnr_history)
                color = colors[i % len(colors)]
                
                plt.plot(epochs, psnrs, color=color, linestyle='-', marker='o',
                        label=f'{name} (Final: {psnrs[-1]:.2f} dB)', 
                        linewidth=2, markersize=6, alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_psnr_vs_epochs_from_logs(log_files, model_names, output_path=None, title="PSNR vs Epochs Comparison"):
    """
    Parse log files and plot validation loss vs epochs for different models.
    
    Args:
        log_files: List of paths to log files
        model_names: List of model names corresponding to each log file
        output_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for i, (log_file, name) in enumerate(zip(log_files, model_names)):
        epochs = []
        train_losses = []
        val_losses = []
        
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Parse epoch logs: "Epoch  10: Train Loss = 0.912829, Val Loss = 0.904888"
            pattern = r'Epoch\s+(\d+):\s+Train Loss\s*=\s*([\d.]+),?\s*Val Loss\s*=\s*([\d.]+)'
            matches = re.findall(pattern, content)
            
            for match in matches:
                epoch, train_loss, val_loss = int(match[0]), float(match[1]), float(match[2])
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
            # Get final PSNR improvement
            psnr_pattern = r'PSNR:\s*[\d.]+\s*dB\s*→\s*[\d.]+\s*dB\s*\(\+?([\d.-]+)\s*dB\)'
            psnr_matches = re.findall(psnr_pattern, content)
            final_psnr_improvement = float(psnr_matches[0]) if psnr_matches else None
        
        if epochs:
            color = colors[i % len(colors)]
            label = f'{name}'
            if final_psnr_improvement is not None:
                label += f' (ΔPSNR: +{final_psnr_improvement:.2f} dB)'
            
            plt.plot(epochs, val_losses, color=color, linestyle='-', marker='o',
                    label=label, linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss (MSE)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_run_comparison(log_file, output_path=None):
    """
    Parse a single log file with multiple runs and create a box plot showing PSNR distribution.
    
    Args:
        log_file: Path to log file with multiple test runs
        output_path: Path to save the plot (optional)
    
    Returns:
        Dictionary with statistics
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract all PSNR improvements
    psnr_pattern = r'PSNR:\s*[\d.]+\s*dB\s*→\s*([\d.]+)\s*dB\s*\(\+?([\d.-]+)\s*dB\)'
    matches = re.findall(psnr_pattern, content)
    
    final_psnrs = [float(m[0]) for m in matches]
    psnr_improvements = [float(m[1]) for m in matches]
    
    # Extract model name
    model_pattern = r'Model:\s*(\w+)'
    model_match = re.search(model_pattern, content)
    model_name = model_match.group(1) if model_match else "Unknown"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot for final PSNR
    bp1 = ax1.boxplot([final_psnrs], tick_labels=[model_name], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#2E86AB')
    ax1.set_ylabel('Final PSNR (dB)', fontsize=12)
    ax1.set_title(f'Final PSNR Distribution\n(n={len(final_psnrs)} runs)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_psnr = np.mean(final_psnrs)
    ax1.axhline(y=mean_psnr, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_psnr:.2f} dB')
    ax1.legend()
    
    # Box plot for PSNR improvement
    bp2 = ax2.boxplot([psnr_improvements], tick_labels=[model_name], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#A23B72')
    ax2.set_ylabel('PSNR Improvement (dB)', fontsize=12)
    ax2.set_title(f'PSNR Improvement Distribution\n(n={len(psnr_improvements)} runs)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_improvement = np.mean(psnr_improvements)
    ax2.axhline(y=mean_improvement, color='red', linestyle='--', alpha=0.7, label=f'Mean: +{mean_improvement:.2f} dB')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Multi-run comparison saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'model': model_name,
        'n_runs': len(final_psnrs),
        'mean_psnr': mean_psnr,
        'std_psnr': np.std(final_psnrs),
        'mean_improvement': mean_improvement,
        'std_improvement': np.std(psnr_improvements),
    }


def compare_two_model_logs(simp_log_file, freq_log_file, output_path=None):
    """
    Compare TinyResidualPredictor vs FrequencyTinyResidualPredictor from log files.
    
    Args:
        simp_log_file: Path to simple model log file
        freq_log_file: Path to frequency model log file
        output_path: Path to save the plot (optional)
    """
    def parse_log(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract all PSNR improvements
        psnr_pattern = r'PSNR:\s*[\d.]+\s*dB\s*→\s*([\d.]+)\s*dB\s*\(\+?([\d.-]+)\s*dB\)'
        matches = re.findall(psnr_pattern, content)
        
        final_psnrs = [float(m[0]) for m in matches]
        psnr_improvements = [float(m[1]) for m in matches]
        
        # Extract training losses per run
        runs_data = []
        # Split by test sections
        sections = re.split(r'#{50,}', content)
        
        for section in sections:
            epoch_pattern = r'Epoch\s+(\d+):\s+Train Loss\s*=\s*([\d.]+),?\s*Val Loss\s*=\s*([\d.]+)'
            matches = re.findall(epoch_pattern, section)
            if matches:
                epochs = [int(m[0]) for m in matches]
                val_losses = [float(m[2]) for m in matches]
                runs_data.append({'epochs': epochs, 'val_losses': val_losses})
        
        return final_psnrs, psnr_improvements, runs_data
    
    simp_psnrs, simp_improvements, simp_runs = parse_log(simp_log_file)
    freq_psnrs, freq_improvements, freq_runs = parse_log(freq_log_file)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Box plot comparison of PSNR improvements
    ax1 = axes[0, 0]
    bp = ax1.boxplot([simp_improvements, freq_improvements], 
                     tick_labels=['TinyResidual\nPredictor', 'Frequency\nResidualPredictor'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    ax1.set_ylabel('PSNR Improvement (dB)', fontsize=12)
    ax1.set_title('PSNR Improvement Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    ax1.annotate(f'Mean: +{np.mean(simp_improvements):.2f} dB', 
                xy=(1, np.mean(simp_improvements)), fontsize=10, color='#2E86AB')
    ax1.annotate(f'Mean: +{np.mean(freq_improvements):.2f} dB', 
                xy=(2, np.mean(freq_improvements)), fontsize=10, color='#A23B72')
    
    # 2. Validation loss curves (sample first run)
    ax2 = axes[0, 1]
    if simp_runs:
        ax2.plot(simp_runs[0]['epochs'], simp_runs[0]['val_losses'], 
                'o-', color='#2E86AB', label='TinyResidualPredictor', linewidth=2)
    if freq_runs:
        ax2.plot(freq_runs[0]['epochs'], freq_runs[0]['val_losses'], 
                's-', color='#A23B72', label='FrequencyResidualPredictor', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss vs Epochs (Sample Run)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Histogram of PSNR improvements
    ax3 = axes[1, 0]
    ax3.hist(simp_improvements, bins=15, alpha=0.6, color='#2E86AB', 
             label=f'TinyResidual (μ={np.mean(simp_improvements):.2f})')
    ax3.hist(freq_improvements, bins=15, alpha=0.6, color='#A23B72',
             label=f'Frequency (μ={np.mean(freq_improvements):.2f})')
    ax3.set_xlabel('PSNR Improvement (dB)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('PSNR Improvement Distribution', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    COMPARISON SUMMARY                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Metric                    │  TinyResidual  │  Frequency     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Number of Runs            │  {len(simp_improvements):>10}    │  {len(freq_improvements):>10}    ║
    ║  Mean PSNR Improvement     │  +{np.mean(simp_improvements):>8.2f} dB │  +{np.mean(freq_improvements):>8.2f} dB ║
    ║  Std PSNR Improvement      │  {np.std(simp_improvements):>9.2f} dB │  {np.std(freq_improvements):>9.2f} dB ║
    ║  Min PSNR Improvement      │  +{np.min(simp_improvements):>8.2f} dB │  +{np.min(freq_improvements):>8.2f} dB ║
    ║  Max PSNR Improvement      │  +{np.max(simp_improvements):>8.2f} dB │  +{np.max(freq_improvements):>8.2f} dB ║
    ║  Mean Final PSNR           │  {np.mean(simp_psnrs):>9.2f} dB │  {np.mean(freq_psnrs):>9.2f} dB ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Winner: {'TinyResidualPredictor' if np.mean(simp_improvements) > np.mean(freq_improvements) else 'FrequencyResidualPredictor'}
    Advantage: {abs(np.mean(simp_improvements) - np.mean(freq_improvements)):.2f} dB
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('TinyResidualPredictor vs FrequencyResidualPredictor', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary to console
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"TinyResidualPredictor ({len(simp_improvements)} runs):")
    print(f"  Mean PSNR Improvement: +{np.mean(simp_improvements):.2f} dB (±{np.std(simp_improvements):.2f})")
    print(f"  Range: [{np.min(simp_improvements):.2f}, {np.max(simp_improvements):.2f}] dB")
    print(f"\nFrequencyResidualPredictor ({len(freq_improvements)} runs):")
    print(f"  Mean PSNR Improvement: +{np.mean(freq_improvements):.2f} dB (±{np.std(freq_improvements):.2f})")
    print(f"  Range: [{np.min(freq_improvements):.2f}, {np.max(freq_improvements):.2f}] dB")
    print(f"\nWinner: {'TinyResidualPredictor' if np.mean(simp_improvements) > np.mean(freq_improvements) else 'FrequencyResidualPredictor'}")
    print(f"Advantage: {abs(np.mean(simp_improvements) - np.mean(freq_improvements)):.2f} dB")
    print("="*70)


__all__ = [
    'plot_training_curves',
    'plot_model_comparison',
    'plot_psnr_vs_epochs_from_logs',
    'plot_multi_run_comparison',
    'compare_two_model_logs',
]