"""
NeurLZ: Correct Implementation Following the Paper

Reference: https://arxiv.org/abs/2409.05785

Key differences from typical neural compression:
1. SZ3/ZFP is PRIMARY compressor (not neural)
2. Tiny DNN (~3k params) trained ONLINE during compression
3. DNN predicts residuals from SZ3-decompressed data
4. Storage: {SZ3_bytes, DNN_weights, outliers}

This is the main entry point for NeurLZ compression.
All core functionality is now modularized:
- compressor.py: NeurLZCompressor class
- plotting.py: Visualization functions
- utils.py: Utility functions
"""

# Re-export main classes and functions for backward compatibility
# Handle both relative and absolute imports
try:
    from .compressor import NeurLZCompressor
except ImportError:
    from compressor import NeurLZCompressor

try:
    from .plotting import (
        plot_training_curves,
        plot_model_comparison,
        plot_psnr_vs_epochs_from_logs,
        plot_multi_run_comparison,
        compare_two_model_logs
    )
except ImportError:
    from plotting import (
        plot_training_curves,
        plot_model_comparison,
        plot_psnr_vs_epochs_from_logs,
        plot_multi_run_comparison,
        compare_two_model_logs
    )

try:
    from .utils import setup_multi_gpu_model, get_available_gpus
except ImportError:
    from utils import setup_multi_gpu_model, get_available_gpus

# Also export loss functions for convenience (if needed)
try:
    import sys
    import importlib.util
    
    sys.path.append('/Users/923714256/Data_compression/neural_compression/Loss')
    
    # Use importlib to handle filenames with + symbols
    spec1 = importlib.util.spec_from_file_location(
        "spatial_1_mag_1_phase_loss",
        "/Users/923714256/Data_compression/neural_compression/Loss/spatial+1_mag+1_phase_loss.py"
    )
    spatial_1_module = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(spatial_1_module)
    SpatialFrequencyLoss = spatial_1_module.SpatialFrequencyLoss
    
    spec2 = importlib.util.spec_from_file_location(
        "spatial_3_mag_3_phase_loss",
        "/Users/923714256/Data_compression/neural_compression/Loss/spatial+3_mag+3_phase_loss.py"
    )
    spatial_3_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(spatial_3_module)
    BandedFrequencyLoss_3_mag_3_phase = spatial_3_module.BandedFrequencyLoss_3_mag_3_phase
    
    spec3 = importlib.util.spec_from_file_location(
        "spatial_3_freq_spaitial_loss",
        "/Users/923714256/Data_compression/neural_compression/Loss/spatial+3_freq_spaitial_loss.py"
    )
    spatial_3_freq_module = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(spatial_3_freq_module)
    BandWeightedSpectralLoss = spatial_3_freq_module.BandWeightedSpectralLoss
    
    # Create aliases for compatibility
    BandedFrequencyLoss = BandedFrequencyLoss_3_mag_3_phase
    SpatialEnergyBandLoss = BandWeightedSpectralLoss
except ImportError:
    # If loss functions can't be imported, set to None
    SpatialFrequencyLoss = None
    BandedFrequencyLoss = None
    SpatialEnergyBandLoss = None


__all__ = [
    'NeurLZCompressor',
    'plot_training_curves',
    'plot_model_comparison',
    'plot_psnr_vs_epochs_from_logs',
    'plot_multi_run_comparison',
    'compare_two_model_logs',
    'setup_multi_gpu_model',
    'get_available_gpus',
    'SpatialFrequencyLoss',
    'BandedFrequencyLoss',
    'SpatialEnergyBandLoss',
]


if __name__ == "__main__":
    import numpy as np
    
    print("="*70)
    print("NeurLZ: Correct Implementation Following the Paper")
    print("="*70)
    print("Pipeline: SZ3 (primary) → Train tiny DNN online → Enhance")
    print("Storage: {SZ3_bytes, DNN_weights, outliers}")
    print("="*70)
    
    # Quick test
    compressor = NeurLZCompressor()
    
    # Test data
    test_data = np.random.randn(64, 64, 64).astype(np.float32) * 1000
    
    print("\n" + "="*70)
    print("Test 1: 3D mode with validation and loss tracking")
    print("="*70)
    package_3d, stats_3d = compressor.compress(
        test_data,
        eb_mode=1,  # REL mode
        absolute_error_bound=0.0,
        relative_error_bound=5e-3,
        pwr_error_bound=0.0,
        spatial_dims=3,
        online_epochs=20,
        model_channels=4,
        val_split=0.2,
        track_losses=True,
        model='tiny_residual_predictor',
        num_res_blocks=1,
    )
    
    print("\nTest decompression (3D)...")
    reconstructed_3d = compressor.decompress(package_3d)
    
    print("\nTest verification (3D)...")
    metrics_3d = compressor.verify_reconstruction(
        test_data, 
        reconstructed_3d, 
        eb_mode=1,
        absolute_error_bound=0.0, 
        relative_error_bound=5e-3,
        pwr_error_bound=0.0,
        model='tiny_residual_predictor'
    )
    
    print("\n" + "="*70)
    print("Test 2: 2D sliced mode with validation and loss tracking")
    print("="*70)
    package_2d, stats_2d = compressor.compress(
        test_data,
        eb_mode=1,  # REL mode
        absolute_error_bound=0.0,
        relative_error_bound=5e-3,
        pwr_error_bound=0.0,
        spatial_dims=2,
        slice_order='zxy',
        online_epochs=20,
        model_channels=4,
        val_split=0.2,
        track_losses=True,
        model='tiny_residual_predictor',
        num_res_blocks=1,
    )
    
    print("\nTest decompression (2D)...")
    reconstructed_2d = compressor.decompress(package_2d)
    
    print("\nTest verification (2D)...")
    metrics_2d = compressor.verify_reconstruction(
        test_data, 
        reconstructed_2d, 
        eb_mode=1,
        absolute_error_bound=0.0, 
        relative_error_bound=5e-3,
        pwr_error_bound=0.0,
        model='tiny_residual_predictor'
    )
    
    print("\n" + "="*70)
    print("✓ All tests complete!")
    print("="*70)
    print(f"3D Mode:")
    print(f"  Compression ratio: {stats_3d['overall_ratio']:.2f}x")
    print(f"  Max error: {metrics_3d['max_error']:.3e}")
    print(f"  Within bound: {metrics_3d['within_bound']}")
    print(f"  Training losses: {len(stats_3d['train_losses'])} epochs")
    print(f"  Validation losses: {len(stats_3d['val_losses'])} epochs")
    
    print(f"\n2D Mode:")
    print(f"  Compression ratio: {stats_2d['overall_ratio']:.2f}x")
    print(f"  Max error: {metrics_2d['max_error']:.3e}")
    print(f"  Within bound: {metrics_2d['within_bound']}")
    print(f"  Training losses: {len(stats_2d['train_losses'])} epochs")
    print(f"  Validation losses: {len(stats_2d['val_losses'])} epochs")
    
    print("\n" + "="*70)
