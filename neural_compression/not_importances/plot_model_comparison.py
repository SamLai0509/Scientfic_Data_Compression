#!/usr/bin/env python3
"""
Script to compare TinyResidualPredictor vs FrequencyResidualPredictor
from log files and generate comparison plots.

Usage:
    python plot_model_comparison.py
    
Or import and use the functions:
    from plot_model_comparison import compare_two_model_logs
    compare_two_model_logs(simp_log, freq_log, output_path='comparison.png')
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os


def parse_log_file(log_file):
    """
    Parse a log file and extract training metrics.
    
    Returns:
        final_psnrs: List of final PSNR values
        psnr_improvements: List of PSNR improvements
        runs_data: List of dicts with epochs and val_losses per run
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract all PSNR improvements
    psnr_pattern = r'PSNR:\s*[\d.]+\s*dB\s*→\s*([\d.]+)\s*dB\s*\(\+?([\d.-]+)\s*dB\)'
    matches = re.findall(psnr_pattern, content)
    
    final_psnrs = [float(m[0]) for m in matches]
    psnr_improvements = [float(m[1]) for m in matches]
    
    # Extract training losses per run
    runs_data = []
    sections = re.split(r'#{50,}', content)
    
    for section in sections:
        epoch_pattern = r'Epoch\s+(\d+):\s+Train Loss\s*=\s*([\d.]+),?\s*Val Loss\s*=\s*([\d.]+)'
        matches = re.findall(epoch_pattern, section)
        if matches:
            epochs = [int(m[0]) for m in matches]
            train_losses = [float(m[1]) for m in matches]
            val_losses = [float(m[2]) for m in matches]
            runs_data.append({
                'epochs': epochs, 
                'train_losses': train_losses,
                'val_losses': val_losses
            })
    
    return final_psnrs, psnr_improvements, runs_data


def compare_two_model_logs(simp_log_file, freq_log_file, output_path=None):
    """
    Compare TinyResidualPredictor vs FrequencyTinyResidualPredictor from log files.
    
    Args:
        simp_log_file: Path to simple model log file
        freq_log_file: Path to frequency model log file
        output_path: Path to save the plot (optional)
    """
    simp_psnrs, simp_improvements, simp_runs = parse_log_file(simp_log_file)
    freq_psnrs, freq_improvements, freq_runs = parse_log_file(freq_log_file)
    
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
        print(f"✓ Model comparison saved to: {output_path}")
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
    
    return {
        'simp': {
            'n_runs': len(simp_improvements),
            'mean_improvement': np.mean(simp_improvements),
            'std_improvement': np.std(simp_improvements),
            'mean_psnr': np.mean(simp_psnrs),
        },
        'freq': {
            'n_runs': len(freq_improvements),
            'mean_improvement': np.mean(freq_improvements),
            'std_improvement': np.std(freq_improvements),
            'mean_psnr': np.mean(freq_psnrs),
        }
    }


def plot_all_runs_loss_curves(log_file, model_name, output_path=None):
    """
    Plot validation loss curves for all runs in a log file.
    
    Args:
        log_file: Path to log file
        model_name: Name of the model
        output_path: Path to save the plot (optional)
    """
    _, _, runs_data = parse_log_file(log_file)
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs_data)))
    
    for i, run in enumerate(runs_data):
        alpha = 0.3 if len(runs_data) > 10 else 0.6
        plt.plot(run['epochs'], run['val_losses'], 
                color=colors[i], alpha=alpha, linewidth=1)
    
    # Plot mean
    if runs_data:
        all_epochs = runs_data[0]['epochs']
        all_losses = np.array([run['val_losses'] for run in runs_data if len(run['val_losses']) == len(all_epochs)])
        if len(all_losses) > 0:
            mean_losses = np.mean(all_losses, axis=0)
            std_losses = np.std(all_losses, axis=0)
            plt.plot(all_epochs, mean_losses, 'r-', linewidth=3, label='Mean')
            plt.fill_between(all_epochs, mean_losses - std_losses, mean_losses + std_losses,
                           color='red', alpha=0.2, label='±1 Std')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(f'{model_name} - All Runs Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ All runs plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Default log file paths
    base_dir = "/Users/923714256/Data_compression/neural_compression"
    
    simp_log = os.path.join(base_dir, "5e-4_freq3d_zxy_mseloss_new_freq_model_v1_+2.42dB_64patches.out")
    freq_log = os.path.join(base_dir, "5e-4_simp3d_zxy_mseloss_10K +3.07dB.out")
    
    # Check if files exist
    if not os.path.exists(simp_log):
        print(f"Warning: {simp_log} not found")
        simp_log = None
    if not os.path.exists(freq_log):
        print(f"Warning: {freq_log} not found")
        freq_log = None
    
    if simp_log and freq_log:
        print("="*70)
        print("Generating Model Comparison Plot")
        print("="*70)
        print(f"Simple model log: {simp_log}")
        print(f"Frequency model log: {freq_log}")
        print("="*70)
        
        # Generate comparison
        output_path = os.path.join(base_dir, "model_comparison_simp_vs_freq.png")
        compare_two_model_logs(simp_log, freq_log, output_path=output_path)
        
        # Generate individual all-runs plots
        print("\nGenerating individual model plots...")
        plot_all_runs_loss_curves(simp_log, "TinyResidualPredictor", 
                                  os.path.join(base_dir, "simp_all_runs.png"))
        plot_all_runs_loss_curves(freq_log, "FrequencyResidualPredictor",
                                  os.path.join(base_dir, "freq_all_runs.png"))
        
        print("\n✓ All plots generated successfully!")
    else:
        print("Please provide valid log file paths.")
        print("\nUsage:")
        print("  python plot_model_comparison.py")
        print("\nOr modify the file paths in this script.")

