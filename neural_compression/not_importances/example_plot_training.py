"""
Example script demonstrating validation split and training loss plotting
for NeurLZ compression with different error bounds and configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from neurlz_correct import NeurLZCompressor, plot_training_curves

def example_single_plot():
    """Example: Compress data and plot training curves for a single error bound."""
    print("="*70)
    print("Example 1: Single Error Bound with Training Curve")
    print("="*70)
    
    # Load or create test data
    data = np.random.randn(128, 128, 128).astype(np.float32) * 1000
    
    # Initialize compressor
    compressor = NeurLZCompressor(device='cuda')
    
    # Compress with validation tracking
    error_bound = 10.0
    package, stats = compressor.compress(
        data,
        error_bound=error_bound,
        mode='strict',
        spatial_dims=2,          # Use 2D sliced processing
        slice_order='zxy',       # Process along Z dimension
        online_epochs=100,
        model_channels=4,
        val_split=0.1,           # 10% for validation
        track_losses=True,       # Track losses
        verbose=True
    )
    
    # Plot training curves
    plot_training_curves(
        stats, 
        error_bound=error_bound,
        output_path=f'training_curve_eb{error_bound:.0e}.png',
        title_suffix='(2D sliced mode)'
    )
    
    print(f"\n✓ Plot saved!")
    print(f"  Training loss: {stats['train_losses'][0]:.6f} → {stats['train_losses'][-1]:.6f}")
    print(f"  Validation loss: {stats['val_losses'][0]:.6f} → {stats['val_losses'][-1]:.6f}")


def example_multiple_error_bounds():
    """Example: Compare training curves across multiple error bounds."""
    print("\n" + "="*70)
    print("Example 2: Multiple Error Bounds Comparison")
    print("="*70)
    
    # Load or create test data
    data = np.random.randn(128, 128, 128).astype(np.float32) * 1000
    
    # Initialize compressor
    compressor = NeurLZCompressor(device='cuda')
    
    # Test multiple error bounds
    error_bounds = [10.0, 50.0, 100.0, 500.0]
    all_stats = []
    
    for eb in error_bounds:
        print(f"\nProcessing error bound: {eb:.1e}")
        package, stats = compressor.compress(
            data,
            error_bound=eb,
            mode='strict',
            spatial_dims=2,
            slice_order='zxy',
            online_epochs=50,
            model_channels=4,
            val_split=0.1,
            track_losses=True,
            verbose=False
        )
        all_stats.append(stats)
        print(f"  Final train loss: {stats['train_losses'][-1]:.6f}")
        print(f"  Final val loss: {stats['val_losses'][-1]:.6f}")
        print(f"  Compression ratio: {stats['overall_ratio']:.2f}x")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (eb, stats) in enumerate(zip(error_bounds, all_stats)):
        ax = axes[idx]
        epochs = range(1, len(stats['train_losses']) + 1)
        
        ax.plot(epochs, stats['train_losses'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, stats['val_losses'], 'r-', label='Val', linewidth=2)
        
        ax.set_title(f'Error Bound: {eb:.1e} (Ratio: {stats["overall_ratio"]:.1f}x)', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final loss text
        final_text = f'Train: {stats["train_losses"][-1]:.4f}\nVal: {stats["val_losses"][-1]:.4f}'
        ax.text(0.98, 0.98, final_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    plt.suptitle('Training Curves Comparison Across Error Bounds', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: training_curves_comparison.png")


def example_2d_vs_3d():
    """Example: Compare 2D sliced vs 3D volume processing."""
    print("\n" + "="*70)
    print("Example 3: 2D Sliced vs 3D Volume Processing")
    print("="*70)
    
    # Load or create test data
    data = np.random.randn(128, 128, 128).astype(np.float32) * 1000
    
    # Initialize compressor
    compressor = NeurLZCompressor(device='cuda')
    
    error_bound = 50.0
    
    # Test 2D mode
    print("\nProcessing with 2D sliced mode...")
    package_2d, stats_2d = compressor.compress(
        data,
        error_bound=error_bound,
        spatial_dims=2,
        slice_order='zxy',
        online_epochs=50,
        val_split=0.1,
        track_losses=True,
        verbose=False
    )
    
    # Test 3D mode
    print("Processing with 3D volume mode...")
    package_3d, stats_3d = compressor.compress(
        data,
        error_bound=error_bound,
        spatial_dims=3,
        online_epochs=50,
        val_split=0.1,
        track_losses=True,
        verbose=False
    )
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D plot
    epochs = range(1, len(stats_2d['train_losses']) + 1)
    ax1.plot(epochs, stats_2d['train_losses'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, stats_2d['val_losses'], 'r-', label='Val', linewidth=2)
    ax1.set_title(f'2D Sliced Mode\n(Ratio: {stats_2d["overall_ratio"]:.2f}x, '
                  f'Params: {stats_2d["model_params"]:,})', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D plot
    epochs = range(1, len(stats_3d['train_losses']) + 1)
    ax2.plot(epochs, stats_3d['train_losses'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, stats_3d['val_losses'], 'r-', label='Val', linewidth=2)
    ax2.set_title(f'3D Volume Mode\n(Ratio: {stats_3d["overall_ratio"]:.2f}x, '
                  f'Params: {stats_3d["model_params"]:,})', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'2D vs 3D Processing - Error Bound: {error_bound:.1e}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_2d_vs_3d.png', dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Comparison plot saved: training_2d_vs_3d.png")
    print(f"\n2D Mode:")
    print(f"  Compression ratio: {stats_2d['overall_ratio']:.2f}x")
    print(f"  Model params: {stats_2d['model_params']:,}")
    print(f"  Final train loss: {stats_2d['train_losses'][-1]:.6f}")
    print(f"  Final val loss: {stats_2d['val_losses'][-1]:.6f}")
    
    print(f"\n3D Mode:")
    print(f"  Compression ratio: {stats_3d['overall_ratio']:.2f}x")
    print(f"  Model params: {stats_3d['model_params']:,}")
    print(f"  Final train loss: {stats_3d['train_losses'][-1]:.6f}")
    print(f"  Final val loss: {stats_3d['val_losses'][-1]:.6f}")


if __name__ == "__main__":
    # Run examples (comment out the ones you don't need)
    
    # Example 1: Single error bound with plot
    example_single_plot()
    
    # Example 2: Multiple error bounds comparison
    example_multiple_error_bounds()
    
    # Example 3: 2D vs 3D comparison
    example_2d_vs_3d()
    
    print("\n" + "="*70)
    print("✓ All examples complete!")
    print("="*70)

