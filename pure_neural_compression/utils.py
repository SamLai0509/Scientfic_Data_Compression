"""
Utility functions for Pure Neural Compressor.
"""

import numpy as np
import torch
import json
from pathlib import Path


def load_volume(filepath, shape, dtype=np.float32):
    """
    Load 3D volume from binary file.
    
    Args:
        filepath: Path to binary file
        shape: Tuple of (D, H, W)
        dtype: Data type
    
    Returns:
        volume: Numpy array of shape (D, H, W)
    """
    data = np.fromfile(filepath, dtype=dtype)
    expected_size = np.prod(shape)
    
    if data.size != expected_size:
        raise ValueError(
            f"File size mismatch: expected {expected_size} elements "
            f"for shape {shape}, got {data.size}"
        )
    
    return data.reshape(shape)


def save_volume(volume, filepath, dtype=np.float32):
    """
    Save 3D volume to binary file.
    
    Args:
        volume: Numpy array
        filepath: Output path
        dtype: Data type for saving
    """
    volume = volume.astype(dtype)
    volume.tofile(filepath)


def compute_data_stats(data):
    """
    Compute comprehensive statistics for data.
    
    Args:
        data: Numpy array
    
    Returns:
        stats: Dictionary with statistics
    """
    stats = {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'size_bytes': data.nbytes,
        'size_mb': data.nbytes / (1024**2),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'nonzero_count': int(np.count_nonzero(data)),
        'nonzero_ratio': float(np.count_nonzero(data) / data.size),
    }
    
    # Percentiles
    stats['percentiles'] = {
        'p01': float(np.percentile(data, 1)),
        'p10': float(np.percentile(data, 10)),
        'p25': float(np.percentile(data, 25)),
        'p50': float(np.percentile(data, 50)),
        'p75': float(np.percentile(data, 75)),
        'p90': float(np.percentile(data, 90)),
        'p99': float(np.percentile(data, 99)),
    }
    
    return stats


def print_data_stats(data, name="Data"):
    """Print data statistics in a readable format."""
    stats = compute_data_stats(data)
    
    print(f"\n{name} Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Size: {stats['size_mb']:.2f} MB")
    print(f"  Range: [{stats['min']:.6e}, {stats['max']:.6e}]")
    print(f"  Mean: {stats['mean']:.6e}")
    print(f"  Std: {stats['std']:.6e}")
    print(f"  Median: {stats['median']:.6e}")
    print(f"  Nonzero: {stats['nonzero_ratio']*100:.2f}%")


def verify_model_checkpoint(checkpoint_path):
    """
    Verify model checkpoint integrity.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        info: Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'path': str(checkpoint_path),
        'keys': list(checkpoint.keys()),
    }
    
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        info['num_parameters'] = sum(p.numel() for p in state_dict.values())
        info['parameter_keys'] = list(state_dict.keys())
    
    return info


def estimate_compression_ratio(original_size, latent_dim, n_levels=256, 
                                entropy_efficiency=0.7):
    """
    Estimate expected compression ratio.
    
    Args:
        original_size: Size of original data in bytes
        latent_dim: Latent dimension
        n_levels: Number of quantization levels
        entropy_efficiency: Entropy coding efficiency (0-1)
    
    Returns:
        estimated_ratio: Estimated compression ratio
    """
    # Calculate bits per symbol after quantization
    bits_per_symbol = np.log2(n_levels)
    
    # Estimate compressed latent size
    quantized_size = latent_dim * np.ceil(bits_per_symbol / 8)
    
    # Apply entropy coding efficiency
    compressed_size = quantized_size * entropy_efficiency
    
    # Add overhead for metadata (rough estimate)
    overhead = 1024  # 1 KB
    compressed_size += overhead
    
    estimated_ratio = original_size / compressed_size
    
    return estimated_ratio


def create_training_plan(data_dir, output_dir, config):
    """
    Create a training plan with recommended settings.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        config: Dictionary with user preferences
    
    Returns:
        plan: Dictionary with training plan
    """
    data_dir = Path(data_dir)
    
    # Find available data files
    data_files = sorted(data_dir.glob("*.f32"))
    
    if not data_files:
        raise ValueError(f"No .f32 files found in {data_dir}")
    
    # Split into train/val
    n_files = len(data_files)
    n_train = int(n_files * 0.8)
    train_files = [f.name for f in data_files[:n_train]]
    val_files = [f.name for f in data_files[n_train:]]
    
    # Load one file to get stats
    sample_file = data_files[0]
    shape = config.get('data_shape', (512, 512, 512))
    
    try:
        sample_data = load_volume(sample_file, shape)
        data_stats = compute_data_stats(sample_data)
    except:
        data_stats = None
    
    # Determine if patch-based training is needed
    if data_stats:
        data_size_mb = data_stats['size_mb']
        use_patches = data_size_mb > 1000  # Use patches if > 1GB per volume
    else:
        use_patches = True  # Default to patches for safety
    
    plan = {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'train_files': train_files,
        'val_files': val_files,
        'data_shape': shape,
        'use_patches': use_patches,
        'patch_shape': (128, 128, 128) if use_patches else shape,
        'spatial_channels': config.get('spatial_channels', 16),
        'freq_channels': config.get('freq_channels', 8),
        'latent_dim': config.get('latent_dim', 2048),
        'num_epochs': config.get('num_epochs', 100),
        'batch_size': config.get('batch_size', 1),
        'learning_rate': config.get('learning_rate', 1e-4),
        'error_bound': config.get('error_bound', None),
        'data_stats': data_stats,
    }
    
    return plan


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_bytes(num_bytes):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_time(seconds):
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test data generation
    test_data = np.random.randn(100, 100, 100).astype(np.float32)
    
    # Test stats
    print_data_stats(test_data, "Test Data")
    
    # Test compression ratio estimation
    original_size = test_data.nbytes
    for n_levels in [64, 256, 1024]:
        ratio = estimate_compression_ratio(original_size, latent_dim=2048, n_levels=n_levels)
        print(f"\nEstimated ratio with {n_levels} levels: {ratio:.2f}x")
    
    # Test formatting
    print(f"\nFormatted size: {format_bytes(original_size)}")
    print(f"Formatted time: {format_time(123.45)}")
    
    print("\n✓ Utility tests passed!")

