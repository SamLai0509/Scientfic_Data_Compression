"""
Evaluation script for Pure Neural Compressor.

Tests compression performance on held-out data:
- Compression ratios
- Reconstruction quality (PSNR, NRMSE)
- Error bound compliance
- Compression/decompression speed
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from neural_compressor import NeuralCompressor


def load_volume(filepath, shape):
    """Load 3D volume from file."""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"Data size {data.size} doesn't match expected shape {shape}")
    return data.reshape(shape)


def compute_metrics(original, reconstructed, error_bound=None):
    """
    Compute comprehensive quality metrics.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
        error_bound: Expected error bound
    
    Returns:
        metrics: Dictionary with all metrics
    """
    # Basic error metrics
    error = original - reconstructed
    abs_error = np.abs(error)
    
    max_error = float(np.max(abs_error))
    mean_error = float(np.mean(abs_error))
    std_error = float(np.std(error))
    
    # MSE and RMSE
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    
    # PSNR
    data_range = float(np.max(original) - np.min(original))
    if mse > 0:
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # NRMSE
    nrmse = rmse / data_range if data_range > 0 else 0.0
    
    # Error bound compliance
    within_bound = None
    violation_count = None
    violation_ratio = None
    
    if error_bound is not None:
        within_bound = bool(max_error <= error_bound)
        violation_count = int(np.sum(abs_error > error_bound))
        violation_ratio = float(violation_count / abs_error.size)
    
    # Percentiles
    error_percentiles = {
        'p50': float(np.percentile(abs_error, 50)),
        'p90': float(np.percentile(abs_error, 90)),
        'p95': float(np.percentile(abs_error, 95)),
        'p99': float(np.percentile(abs_error, 99)),
    }
    
    metrics = {
        'max_error': max_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'nrmse': nrmse,
        'data_range': data_range,
        'within_bound': within_bound,
        'violation_count': violation_count,
        'violation_ratio': violation_ratio,
        'error_percentiles': error_percentiles,
    }
    
    return metrics


def evaluate_file(compressor, filepath, shape, error_bound, output_dir, quantization_levels=None):
    """
    Evaluate compression on a single file.
    
    Args:
        compressor: NeuralCompressor instance
        filepath: Path to data file
        shape: Data shape
        error_bound: Error bound
        output_dir: Directory for temporary files
        quantization_levels: Number of quantization levels
    
    Returns:
        results: Dictionary with all results
    """
    filename = Path(filepath).name
    print(f"\n{'='*60}")
    print(f"Evaluating: {filename}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    original = load_volume(filepath, shape)
    
    print(f"Data shape: {original.shape}")
    print(f"Data range: [{original.min():.6e}, {original.max():.6e}]")
    print(f"Data size: {original.nbytes / (1024**2):.2f} MB")
    
    # Compress
    print(f"\nCompressing with error bound {error_bound}...")
    compressed_path = output_dir / f"{filename}.compressed"
    
    compress_stats = compressor.compress(
        original,
        output_path=compressed_path,
        error_bound=error_bound,
        n_levels=quantization_levels
    )
    
    print(f"Compression time: {compress_stats['total_time']:.2f}s")
    print(f"Compression ratio: {compress_stats['compression_ratio']:.2f}x")
    print(f"Compressed size: {compress_stats['compressed_size'] / (1024**2):.2f} MB")
    
    # Decompress
    print("\nDecompressing...")
    reconstructed, decompress_stats = compressor.decompress(
        input_path=compressed_path,
        error_bound=error_bound
    )
    
    print(f"Decompression time: {decompress_stats['total_time']:.2f}s")
    
    # Compute metrics
    print("\nComputing quality metrics...")
    quality_metrics = compute_metrics(original, reconstructed, error_bound)
    
    print(f"Max error: {quality_metrics['max_error']:.6e}")
    print(f"Mean error: {quality_metrics['mean_error']:.6e}")
    print(f"PSNR: {quality_metrics['psnr']:.2f} dB")
    print(f"NRMSE: {quality_metrics['nrmse']:.6f}")
    
    if error_bound is not None:
        if quality_metrics['within_bound']:
            print(f"✓ Error bound satisfied ({quality_metrics['max_error']:.6e} <= {error_bound:.6e})")
        else:
            print(f"✗ Error bound violated ({quality_metrics['max_error']:.6e} > {error_bound:.6e})")
        print(f"Violation ratio: {quality_metrics['violation_ratio']:.6f}")
    
    # Clean up temporary file
    if compressed_path.exists():
        compressed_path.unlink()
    
    # Combine all results
    results = {
        'filename': filename,
        'error_bound': error_bound,
        'compression': compress_stats,
        'decompression': decompress_stats,
        'quality': quality_metrics,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure Neural Compressor")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--test_files', type=str, nargs='+', required=True,
                        help='List of test file names')
    parser.add_argument('--data_shape', type=int, nargs=3, default=[512, 512, 512],
                        help='Shape of data volumes')
    
    # Compression parameters
    parser.add_argument('--error_bounds', type=float, nargs='+', 
                        default=[1e-2, 1e-4, 1e-6],
                        help='List of error bounds to test')
    parser.add_argument('--quantization_levels', type=int, default=None,
                        help='Number of quantization levels (auto if None)')
    parser.add_argument('--quantization_method', type=str, default='uniform',
                        choices=['uniform', 'adaptive'],
                        help='Quantization method')
    parser.add_argument('--entropy_coding', type=str, default='auto',
                        choices=['auto', 'arithmetic', 'zlib'],
                        help='Entropy coding method')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create compressor
    print("Initializing compressor...")
    compressor = NeuralCompressor(
        device=args.device,
        quantization=args.quantization_method,
        entropy_coding=args.entropy_coding
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    compressor.load_model(args.model_path)
    
    # Prepare data files
    data_dir = Path(args.data_dir)
    data_shape = tuple(args.data_shape)
    
    # Evaluate each file with each error bound
    all_results = []
    
    for error_bound in args.error_bounds:
        print(f"\n{'#'*60}")
        print(f"Testing with error bound: {error_bound:.1e}")
        print(f"{'#'*60}")
        
        for filename in args.test_files:
            filepath = data_dir / filename
            
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping...")
                continue
            
            try:
                results = evaluate_file(
                    compressor,
                    filepath,
                    data_shape,
                    error_bound,
                    output_dir,
                    args.quantization_levels
                )
                all_results.append(results)
            except Exception as e:
                print(f"Error evaluating {filename}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    table_data = []
    for result in all_results:
        row = [
            result['filename'],
            f"{result['error_bound']:.1e}",
            f"{result['compression']['compression_ratio']:.2f}",
            f"{result['quality']['psnr']:.2f}",
            f"{result['quality']['nrmse']:.6f}",
            f"{result['quality']['max_error']:.2e}",
            "✓" if result['quality']['within_bound'] else "✗",
            f"{result['compression']['total_time']:.2f}",
            f"{result['decompression']['total_time']:.2f}",
        ]
        table_data.append(row)
    
    headers = [
        "File", "Error Bound", "Ratio", "PSNR", "NRMSE", 
        "Max Error", "Within Bound", "Comp Time(s)", "Decomp Time(s)"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nSummary saved to {summary_file}")
    
    # Compute averages per error bound
    print("\n" + "="*80)
    print("AVERAGES PER ERROR BOUND")
    print("="*80)
    
    avg_table = []
    for eb in args.error_bounds:
        eb_results = [r for r in all_results if r['error_bound'] == eb]
        if not eb_results:
            continue
        
        avg_ratio = np.mean([r['compression']['compression_ratio'] for r in eb_results])
        avg_psnr = np.mean([r['quality']['psnr'] for r in eb_results if np.isfinite(r['quality']['psnr'])])
        avg_nrmse = np.mean([r['quality']['nrmse'] for r in eb_results])
        avg_comp_time = np.mean([r['compression']['total_time'] for r in eb_results])
        avg_decomp_time = np.mean([r['decompression']['total_time'] for r in eb_results])
        success_rate = np.mean([1.0 if r['quality']['within_bound'] else 0.0 for r in eb_results])
        
        avg_table.append([
            f"{eb:.1e}",
            f"{avg_ratio:.2f}",
            f"{avg_psnr:.2f}",
            f"{avg_nrmse:.6f}",
            f"{avg_comp_time:.2f}",
            f"{avg_decomp_time:.2f}",
            f"{success_rate*100:.1f}%"
        ])
    
    avg_headers = ["Error Bound", "Avg Ratio", "Avg PSNR", "Avg NRMSE", 
                   "Avg Comp Time", "Avg Decomp Time", "Success Rate"]
    print(tabulate(avg_table, headers=avg_headers, tablefmt="grid"))
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

