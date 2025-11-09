"""
Evaluation script for NeurLZ (Correct Implementation)

Compares:
1. Baseline: SZ3 only
2. NeurLZ: SZ3 + Online-trained tiny DNN

Following the actual NeurLZ paper approach.
"""

import sys
import os
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from tabulate import tabulate
import time

sys.path.append('/Users/923714256/Data_compression/SZ3/tools/pysz')
from pysz import SZ

from neurlz_correct import NeurLZCompressor


def evaluate_baseline_sz3(sz, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound):
    """Evaluate baseline SZ3 compression."""
    print(f"\n{'─'*70}")
    print("BASELINE: SZ3 Only")
    print(f"{'─'*70}")
    
    # Compress
    compress_start = time.time()
    sz3_compressed, sz3_ratio = sz.compress(data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound)
    compress_time = time.time() - compress_start
    
    sz3_size = len(sz3_compressed)
    
    # Decompress
    decompress_start = time.time()
    reconstructed = sz.decompress(sz3_compressed, data.shape, np.float32)
    decompress_time = time.time() - decompress_start
    
    # Metrics
    error = np.abs(data - reconstructed)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    data_range = np.max(data) - np.min(data)
    mse = np.mean((data - reconstructed) ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
    nrmse = np.sqrt(mse) / data_range
    
    within_bound = max_error <= absolute_error_bound
    violation_ratio = np.sum(error > absolute_error_bound) / error.size * 100
    
    print(f"Compressed size: {sz3_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {sz3_ratio:.2f}x")
    print(f"Compress time: {compress_time:.2f}s")
    print(f"Decompress time: {decompress_time:.2f}s")
    print(f"\nQuality:")
    print(f"  Max error: {max_error:.3e} (bound: {absolute_error_bound:.3e})")
    print(f"  Mean error: {mean_error:.3e}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  NRMSE: {nrmse:.6e}")
    print(f"  Within bound: {'✓ YES' if within_bound else '✗ NO'}")
    if not within_bound:
        print(f"  Violations: {violation_ratio:.2f}%")
    
    return {
        'compressed_size': sz3_size,
        'ratio': sz3_ratio,
        'max_error': max_error,
        'mean_error': mean_error,
        'psnr': psnr,
        'nrmse': nrmse,
        'within_bound': within_bound,
        'violation_ratio': violation_ratio,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
    }


def evaluate_neurlz(compressor, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                    online_epochs=50, 
                    learning_rate=1e-3, 
                    model_channels=4, 
                    model='tiny_residual_predictor', 
                    num_res_blocks=1, 
                    spatial_dims=3, 
                    slice_order='zxy', 
                    val_split=0.1, 
                    track_losses=True):
    """Evaluate NeurLZ compression."""
    print(f"\n{'─'*70}")
    print(f"NeurLZ: SZ3 + Online DNN Enhancement")
    print(f"{'─'*70}")
    
    # Compress
    package, compress_stats = compressor.compress(
        data,
        eb_mode=eb_mode,
        absolute_error_bound=absolute_error_bound,
        relative_error_bound=relative_error_bound,
        pwr_error_bound=pwr_error_bound,
        online_epochs=online_epochs,
        learning_rate=learning_rate,
        model_channels=model_channels,
        model=model,
        verbose=True,
        spatial_dims=spatial_dims,
        slice_order=slice_order,
        val_split=0.1,
        track_losses=True,
        num_res_blocks=num_res_blocks
    )
    
    # Decompress
    decompress_start = time.time()
    reconstructed = compressor.decompress(package, verbose=True)
    decompress_time = time.time() - decompress_start
    
    # Verify
    metrics = compressor.verify_reconstruction(data, reconstructed, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound, model=model)
    
    # Combine stats
    result = {
        'compressed_size': int(compress_stats['total_size_mb'] * 1024**2),
        'sz3_size': int(compress_stats['sz3_size_kb'] * 1024),
        'weights_size': int(compress_stats['weights_size_kb'] * 1024),
        'ratio': compress_stats['overall_ratio'],
        'sz3_ratio': compress_stats['sz3_ratio'],
        'max_error': metrics['max_error'],
        'mean_error': metrics['mean_error'],
        'psnr': metrics['psnr'],
        'nrmse': metrics['nrmse'],
        'within_bound': metrics['within_bound'],
        'violation_ratio': metrics['violation_ratio'],
        'compress_time': compress_stats['compress_time'],
        'decompress_time': decompress_time,
        'training_time': compress_stats['training_time'],
        'model_params': compress_stats['model_params'],
        'base_max_error': compress_stats['base_max_error'],
        'base_mean_error': compress_stats['base_mean_error'],
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeurLZ (Correct Implementation)")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing data files')
    parser.add_argument('--sz_lib', type=str,
                       default='/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so',
                       help='Path to SZ3 library')
    parser.add_argument('--test_files', type=str, nargs='+',
                       default=['velocity_x.f32', 'velocity_y.f32', 'velocity_z.f32'],
                       help='Test files')
    parser.add_argument('--eb_modes', type=int, nargs='+',
                       default=[0, 1, 2, 3, 4, 5, 10],
                       help='Error bound modes to test (0: ABS, 1: REL, 2: ABS_AND_REL, 3: ABS_OR_REL, 4: PSNR, 5: NORM, 10: PW_REL)')
    parser.add_argument('--absolute_error_bounds', type=float, nargs='+',
                       default=[300.0],
                       help='Error bounds to test')
    parser.add_argument('--relative_error_bounds', type=float, nargs='+',
                       default=[5e-3],
                       help='Relative error bounds to test')
    parser.add_argument('--pwr_error_bounds', type=float, nargs='+',
                       default=[0],
                       help='Power error bounds to test')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results_correct',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--online_epochs', type=int, default=50,
                       help='Epochs for online DNN training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for online training')
    parser.add_argument('--model_channels', type=int, default=4,
                       help='Base channels for tiny DNN (4 → ~3k params)')
    parser.add_argument('--model', type=str, default='tiny_residual_predictor',
                       choices=['tiny_residual_predictor', 'tiny_frequency_residual_predictor','tiny_physics_residual_predictor'],
                       help='Model to use')
    parser.add_argument('--num_res_blocks', type=int, default=1,
                       help='Number of residual blocks for the model')
    parser.add_argument('--spatial_dims', type=int, default=3,
                       help='Spatial dimensions')
    parser.add_argument('--slice_order', type=str, default='zxy',
                       help='Slice order')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split')
    parser.add_argument('--track_losses', type=bool, default=True,
                       help='Track losses')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"{'='*70}")
    print(f"NeurLZ Evaluation (Correct Implementation)")
    print(f"{'='*70}")
    print(f"Data directory: {args.data_dir}")
    print(f"Test files: {args.test_files}")
    print(f"Error bound modes: {args.eb_modes}")
    print(f"Absolute error bounds: {args.absolute_error_bounds}")
    print(f"Relative error bounds: {args.relative_error_bounds}")
    print(f"Power error bounds: {args.pwr_error_bounds}")
    print(f"Device: {device}")
    print(f"Online epochs: {args.online_epochs}")
    print(f"Model channels: {args.model_channels} (~{4*args.model_channels*args.model_channels:,} params)")
    print(f"Model: {args.model}")
    print(f"Spatial dimensions: {args.spatial_dims}")
    print(f"Slice order: {args.slice_order}")
    print(f"Validation split: {args.val_split}")
    print(f"Track losses: {args.track_losses}")
    print(f"Number of residual blocks: {args.num_res_blocks}")
    print(f"{'='*70}")
    
    # Initialize
    sz = SZ(args.sz_lib)
    compressor = NeurLZCompressor(sz_lib_path=args.sz_lib, device=device)
    
    # Results storage
    all_results = []
    
    # Evaluate each combination
    import itertools
    for eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound, model in itertools.product(args.eb_modes, args.absolute_error_bounds, args.relative_error_bounds, args.pwr_error_bounds, args.model):
        print(f"\n\n{'#'*70}")
        print(f"TESTING WITH ERROR BOUND MODE: {eb_mode}, ERROR BOUND: {absolute_error_bound}, RELATIVE ERROR BOUND: {relative_error_bound}, POWER ERROR BOUND: {pwr_error_bound}, MODEL: {model}")
        print(f"{'#'*70}")
        
        for filename in args.test_files:
            filepath = os.path.join(args.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found, skipping...")
                continue
            
            print(f"\n{'='*70}")
            print(f"File: {filename}")
            print(f"{'='*70}")
            
            # Load data
            data = np.fromfile(filepath, dtype=np.float32).reshape(512, 512, 512)
            print(f"Shape: {data.shape}")
            print(f"Range: [{np.min(data):.3e}, {np.max(data):.3e}]")
            print(f"Size: {data.nbytes / (1024**2):.2f} MB")
            
            # Baseline SZ3
            baseline_stats = evaluate_baseline_sz3(sz, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound)
            
            # NeurLZ
            neurlz_stats = evaluate_neurlz(
                compressor, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                online_epochs=args.online_epochs,
                learning_rate=args.learning_rate,
                model_channels=args.model_channels,
                model=args.model,
                num_res_blocks=args.num_res_blocks,
                spatial_dims=args.spatial_dims,
                slice_order=args.slice_order,
                val_split=args.val_split,
                track_losses=args.track_losses,
            )
            
            # Comparison
            ratio_improvement = (neurlz_stats['ratio'] / baseline_stats['ratio'] - 1) * 100
            psnr_improvement = neurlz_stats['psnr'] - baseline_stats['psnr']
            mean_error_improvement = ((baseline_stats['mean_error'] - neurlz_stats['mean_error']) 
                                     / baseline_stats['mean_error'] * 100)
            
            print(f"\n{'─'*70}")
            print(f"COMPARISON: Baseline vs NeurLZ")
            print(f"{'─'*70}")
            print(f"Compression ratio: {baseline_stats['ratio']:.2f}x → {neurlz_stats['ratio']:.2f}x "
                  f"({ratio_improvement:+.1f}%)")
            print(f"PSNR: {baseline_stats['psnr']:.2f} dB → {neurlz_stats['psnr']:.2f} dB "
                  f"({psnr_improvement:+.2f} dB)")
            print(f"Mean error: {baseline_stats['mean_error']:.3e} → {neurlz_stats['mean_error']:.3e} "
                  f"({mean_error_improvement:+.1f}%)")
            print(f"Base SZ3 max error: {neurlz_stats['base_max_error']:.3e}")
            print(f"Enhanced max error: {neurlz_stats['max_error']:.3e}")
            
            # Store results
            all_results.append({
                'file': filename,
                'eb_mode': eb_mode,
                'absolute_error_bound': absolute_error_bound,
                'relative_error_bound': relative_error_bound,
                'pwr_error_bound': pwr_error_bound,
                'spatial_dims': args.spatial_dims,
                'slice_order': args.slice_order,
                'val_split': args.val_split,
                'track_losses': args.track_losses,
                'num_res_blocks': args.num_res_blocks,
                'model': args.model,
                'baseline': baseline_stats,
                'neurlz': neurlz_stats,
                'improvements': {
                    'ratio_pct': ratio_improvement,
                    'psnr_db': psnr_improvement,
                    'mean_error_pct': mean_error_improvement,
                }
            })
    
    # Summary table
    print(f"\n\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}\n")
    
    table_data = []
    for r in all_results:
        table_data.append([
            r['file'],
            r['eb_mode'],
            r['absolute_error_bound'],
            r['relative_error_bound'],
            r['pwr_error_bound'],
            r['model'],
            r['num_res_blocks'],
            f"{r['baseline']['ratio']:.2f}",
            f"{r['neurlz']['ratio']:.2f}",
            f"{r['improvements']['ratio_pct']:+.1f}%",
            f"{r['baseline']['psnr']:.2f}",
            f"{r['neurlz']['psnr']:.2f}",
            f"{r['improvements']['psnr_db']:+.2f}",
            f"{r['improvements']['mean_error_pct']:+.1f}%",
            '✓' if r['neurlz']['within_bound'] else '✗',
        ])
    
    headers = ['File', 'EB Mode', 'Absolute Error Bound', 'Relative Error Bound', 'Power Error Bound', 'Model', 'Number of Residual Blocks', 'Base Ratio', 'NeurLZ Ratio', 'ΔRatio',
               'Base PSNR', 'NeurLZ PSNR', 'ΔPSNR', 'ΔMeanErr%', 'Bound OK']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Calculate averages
    print(f"\n{'='*70}")
    print("AVERAGES")
    print(f"{'='*70}")
    
    avg_ratio_improvement = np.mean([r['improvements']['ratio_pct'] for r in all_results])
    avg_psnr_improvement = np.mean([r['improvements']['psnr_db'] for r in all_results])
    avg_mean_error_improvement = np.mean([r['improvements']['mean_error_pct'] for r in all_results])
    
    print(f"Average ratio improvement: {avg_ratio_improvement:+.2f}%")
    print(f"Average PSNR improvement: {avg_psnr_improvement:+.2f} dB")
    print(f"Average mean error reduction: {avg_mean_error_improvement:+.2f}%")
    
    # Convert numpy types to Python types
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types for JSON."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    # Save results
    output_file = os.path.join(args.output_dir, 'neurlz_correct_results.json')
    with open(output_file, 'w') as f:
        serializable_data = {
            'results': convert_to_json_serializable(all_results),
            'averages': convert_to_json_serializable({
                'ratio_improvement_pct': float(avg_ratio_improvement),
                'psnr_improvement_db': float(avg_psnr_improvement),
                'mean_error_improvement_pct': float(avg_mean_error_improvement),
            }),
            'config': vars(args),
        }
        json.dump(serializable_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

