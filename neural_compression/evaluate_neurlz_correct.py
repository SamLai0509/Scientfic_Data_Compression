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
from skimage.metrics import structural_similarity as ssim

sys.path.append('/Users/923714256/Data_compression/SZ3/tools/pysz')
from pysz import SZ

# Import NeurLZCompressor - try both import methods for compatibility
try:
    from compressor import NeurLZCompressor
except ImportError:
    try:
        from neurlz_correct import NeurLZCompressor
    except ImportError:
        from .compressor import NeurLZCompressor
def compute_spectral_metrics(original, reconstructed):
    """
    Computes Log-Magnitude MSE and Phase MSE.
    Crucial for wave-based simulations (Cosmology, Seismic).
    """
    # Compute FFT (n-dimensional)
    fft_orig = np.fft.fftn(original)
    fft_recon = np.fft.fftn(reconstructed)
    
    # 1. Log-Magnitude Error (Energy Fidelity)
    # We use log1p to dampen the huge dynamic range of scientific data
    mag_orig = np.log1p(np.abs(fft_orig))
    mag_recon = np.log1p(np.abs(fft_recon))
    log_mag_mse = np.mean((mag_orig - mag_recon) ** 2)
    
    # 2. Phase Error (Structural Alignment)
    # We care about phase difference vectors
    # A simple subtraction of angles is risky due to wrapping (-pi to pi)
    # Instead, we measure the distance on the unit circle: |e^(i*phi1) - e^(i*phi2)|
    phase_orig = np.angle(fft_orig)
    phase_recon = np.angle(fft_recon)
    
    # Vectorized phase distance (0.0 to 2.0)
    # Equivalent to 2 * sin(|phi1 - phi2| / 2)
    phase_diff = np.abs(np.exp(1j * phase_orig) - np.exp(1j * phase_recon))
    phase_mse = np.mean(phase_diff ** 2)
    
    return log_mag_mse, phase_mse

def compute_ssim_3d(original, reconstructed):
    """
    Computes SSIM for 3D volumes by averaging 2D SSIM over slices.
    (Skimage SSIM is 2D by default, 3D support is experimental/slow)
    """
    # Normalize data to [0, 1] for SSIM calculation
    d_min, d_max = original.min(), original.max()
    rng = d_max - d_min + 1e-9
    
    orig_norm = (original - d_min) / rng
    recon_norm = (reconstructed - d_min) / rng
    
    # Compute SSIM slice-by-slice along the first dimension (usually Depth)
    ssim_accum = 0.0
    valid_slices = 0
    
    # Iterate through slices
    for i in range(original.shape[0]):
        # Slice
        s1 = orig_norm[i]
        s2 = recon_norm[i]
        
        # Calculate SSIM for this slice
        # data_range=1.0 because we normalized
        try:
            val = ssim(s1, s2, data_range=1.0)
            ssim_accum += val
            valid_slices += 1
        except ValueError:
            pass # Skip tiny slices if any
            
    if valid_slices == 0:
        return 0.0
        
    return ssim_accum / valid_slices


# =========================================================
# UPDATED EVALUATION FUNCTIONS
# =========================================================

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
    
    # --- Standard Metrics ---
    error = np.abs(data - reconstructed)
    max_error = np.max(error)
    mean_error = np.mean(error)
    data_range = np.max(data) - np.min(data)
    mse = np.mean((data - reconstructed) ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
    nrmse = np.sqrt(mse) / data_range
    
    # --- NEW: Spectral Metrics ---
    log_mag_mse, phase_mse = compute_spectral_metrics(data, reconstructed)
    
    # --- NEW: Structural Metrics ---
    # Check if 3D or 2D
    if data.ndim == 3:
        ssim_val = compute_ssim_3d(data, reconstructed)
    else:
        # Normalize for 2D SSIM
        d_min, d_max = data.min(), data.max()
        rng = d_max - d_min + 1e-9
        ssim_val = ssim((data-d_min)/rng, (reconstructed-d_min)/rng, data_range=1.0)

    # FFT PSNR (Legacy)
    fft_original = np.fft.fftn(data)
    fft_reconstructed = np.fft.fftn(reconstructed)
    mag_original = np.abs(fft_original)
    mag_reconstructed = np.abs(fft_reconstructed)
    fft_mag_mse = np.mean((mag_original - mag_reconstructed) ** 2)
    fft_range = np.max(mag_original) - np.min(mag_original)
    fft_psnr = 20 * np.log10(fft_range) - 10 * np.log10(fft_mag_mse) if fft_mag_mse > 0 else float('inf')

    within_bound = max_error <= absolute_error_bound
    
    print(f"Compressed size: {sz3_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {sz3_ratio:.2f}x")
    print(f"Compress time: {compress_time:.2f}s")
    print(f"\nQuality Metrics:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f} (Higher is better)")
    print(f"  Max Error: {max_error:.3e}")
    print(f"  Log-Mag MSE: {log_mag_mse:.4e} (Lower is better)")
    print(f"  Phase MSE: {phase_mse:.4e} (Lower is better)")
    
    return {
        'compressed_size': sz3_size,
        'ratio': sz3_ratio,
        'max_error': max_error,
        'mean_error': mean_error,
        'psnr': psnr,
        'ssim': ssim_val,
        'log_mag_mse': log_mag_mse,
        'phase_mse': phase_mse,
        'fft_psnr': fft_psnr,
        'within_bound': within_bound,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
    }


def evaluate_neurlz(compressor, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                    online_epochs=50, learning_rate=1e-3, model_channels=4, model='tiny_residual_predictor', 
                    num_res_blocks=1, spatial_dims=3, slice_order='zxy', val_split=0.1, track_losses=True, 
                    evaluate_per_slice=True, enable_post_process=True, Patch_size=256, Batch_size=512,
                    save_components=False, components_dir='./compressed_components', filename='data'):
    """Evaluate NeurLZ compression."""
    print(f"\n{'─'*70}")
    print(f"NeurLZ: SZ3 + Online DNN Enhancement")
    print(f"{'─'*70}")
    
    # Compress
    package, compress_stats = compressor.compress(
        data, eb_mode=eb_mode, absolute_error_bound=absolute_error_bound,
        relative_error_bound=relative_error_bound, pwr_error_bound=pwr_error_bound,
        online_epochs=online_epochs, learning_rate=learning_rate,
        model_channels=model_channels, model=model, verbose=True,
        spatial_dims=spatial_dims, slice_order=slice_order, val_split=val_split,
        track_losses=track_losses, num_res_blocks=num_res_blocks, Patch_size=Patch_size,
        Batch_size=Batch_size, save_components=save_components, components_dir=components_dir, filename=filename
    )
    
    # Save components if requested
    if save_components:
        base_name = os.path.splitext(filename)[0]
        compressor.save_components(package, components_dir, base_name, verbose=True)
    
    # Decompress
    decompress_start = time.time()
    reconstructed = compressor.decompress(package, verbose=True, enable_post_process=enable_post_process)
    decompress_time = time.time() - decompress_start
    
    # --- Compute Metrics ---
    # 1. Standard
    error = np.abs(data - reconstructed)
    max_error = np.max(error)
    mean_error = np.mean(error)
    data_range = np.max(data) - np.min(data)
    mse = np.mean((data - reconstructed) ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
    nrmse = np.sqrt(mse) / data_range

    # 2. Spectral
    log_mag_mse, phase_mse = compute_spectral_metrics(data, reconstructed)
    
    # 3. Structural (SSIM)
    if data.ndim == 3:
        ssim_val = compute_ssim_3d(data, reconstructed)
    else:
        d_min, d_max = data.min(), data.max()
        rng = d_max - d_min + 1e-9
        ssim_val = ssim((data-d_min)/rng, (reconstructed-d_min)/rng, data_range=1.0)
        
    # 4. FFT PSNR (Legacy)
    fft_original = np.fft.fftn(data)
    fft_reconstructed = np.fft.fftn(reconstructed)
    mag_original = np.abs(fft_original)
    mag_reconstructed = np.abs(fft_reconstructed)
    fft_mag_mse = np.mean((mag_original - mag_reconstructed) ** 2)
    fft_range = np.max(mag_original) - np.min(mag_original)
    fft_psnr = 20 * np.log10(fft_range) - 10 * np.log10(fft_mag_mse) if fft_mag_mse > 0 else float('inf')
    
    within_bound = max_error <= absolute_error_bound # Approximation for strict mode

    # Combine stats
    result = {
        'compressed_size': int(compress_stats['total_size_mb'] * 1024**2),
        'ratio': compress_stats['overall_ratio'],
        'sz3_ratio': compress_stats['sz3_ratio'],
        'max_error': max_error,
        'mean_error': mean_error,
        'psnr': psnr,
        'ssim': ssim_val,
        'log_mag_mse': log_mag_mse,
        'phase_mse': phase_mse,
        'fft_psnr': fft_psnr,
        'within_bound': within_bound,
        'compress_time': compress_stats['compress_time'],
        'decompress_time': decompress_time,
        'base_max_error': compress_stats['base_max_error'],
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
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cpu or cuda:0)')
    parser.add_argument('--online_epochs', type=int, default=50,
                       help='Epochs for online DNN training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for online training')
    parser.add_argument('--model_channels', type=int, default=4,
                       help='Base channels for tiny DNN (4 → ~3k params)')
    parser.add_argument('--model', type=str, default='tiny_residual_predictor',
                       choices=['tiny_residual_predictor', 
                       'tiny_frequency_residual_predictor_1_input', 
                       'tiny_frequency_residual_predictor_with_energy', 
                       'tiny_frequency_residual_predictor_7_inputs', 
                       'tiny_frequency_residual_predictor_4_inputs'],
                       help='Model to use')
    parser.add_argument('--num_res_blocks', type=int, default=1,
                       help='Number of residual blocks for the model')
    parser.add_argument('--spatial_dims', type=int, default=3,
                       choices=[2, 3],
                       help='Spatial dimensions (2 or 3)')
    parser.add_argument('--slice_order', type=str, default='zxy',
                       choices=['xyz', 'zxy', 'yxz'],
                       help='Slice order (for 2D mode)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split')
    parser.add_argument('--track_losses', action='store_true', default=True,
                       help='Track losses')
    # parser.add_argument('--no_track_losses', dest='track_losses', action='store_false',
    #                    help='Disable loss tracking')
    parser.add_argument('--evaluate_per_slice', action='store_true', default=True,
                       help='Evaluate reconstruction quality per slice')
    # parser.add_argument('--no_evaluate_per_slice', dest='evaluate_per_slice', action='store_false',
    #                    help='Disable per-slice evaluation')
    parser.add_argument('--enable_post_process', action='store_true',
                       help='Enable post-process (default: False)')
    parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs for statistical evaluation (default: 1)')
    parser.add_argument('--Patch_size', type=int, default=256,
                       help='Patch size for local decompression and training')
    parser.add_argument('--Batch_size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--save_components', action='store_true',
                       help='Save SZ3 bytes, model weights, and metadata separately')
    parser.add_argument('--components_dir', type=str, default='./compressed_components',
                       help='Directory to save compressed components')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle device string properly
    if args.device.startswith('cuda'):
        device = args.device if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
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
    print(f"Number of runs: {args.num_runs}")
    print(f"Track losses: {args.track_losses}")
    print(f"Spatial dimensions: {args.spatial_dims}")
    print(f"Slice order: {args.slice_order}")
    print(f"Validation split: {args.val_split}")
    print(f"Track losses: {args.track_losses}")
    print(f"Number of residual blocks: {args.num_res_blocks}")
    print(f"Evaluate per slice: {args.evaluate_per_slice}")
    print(f"Enable post-process: {args.enable_post_process}")
    print(f"Patch size: {args.Patch_size}")
    print(f"Batch size: {args.Batch_size}")
    print(f"Save components: {args.save_components}")
    if args.save_components:
        print(f"Components directory: {args.components_dir}")
    print(f"{'='*70}")
    
    # Initialize
    sz = SZ(args.sz_lib)
    compressor = NeurLZCompressor(sz_lib_path=args.sz_lib, device=device)
    
    # Results storage
    all_results = []
    
    # Evaluate each combination
    import itertools
    for eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound in itertools.product(
        args.eb_modes, args.absolute_error_bounds, args.relative_error_bounds, args.pwr_error_bounds
    ):
        print(f"\n\n{'#'*70}")
        print(f"TESTING WITH ERROR BOUND MODE: {eb_mode}, ERROR BOUND: {absolute_error_bound}, "
              f"RELATIVE ERROR BOUND: {relative_error_bound}, POWER ERROR BOUND: {pwr_error_bound}, MODEL: {args.model}")
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
            baseline_stats = evaluate_baseline_sz3(
                sz, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound
            )
            # NeurLZ - Multiple runs for statistical evaluation
            print(f"\n{'='*70}")
            print(f"Running NeurLZ evaluation {args.num_runs} times...")
            print(f"{'='*70}")

            # NeurLZ
            neurlz_runs = []
            for run_idx in range(args.num_runs):
                print(f"\n--- Run {run_idx + 1}/{args.num_runs} ---")
                neurlz = evaluate_neurlz(
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
                    evaluate_per_slice=args.evaluate_per_slice,
                    enable_post_process=args.enable_post_process,
                    Patch_size=args.Patch_size,
                    Batch_size=args.Batch_size,
                    save_components=args.save_components,
                    components_dir=args.components_dir,
                    filename=filename,
                )
                neurlz_runs.append(neurlz)

                # Print individual run result
                diff_psnr = neurlz['psnr'] - baseline_stats['psnr']
                print(f"  Run {run_idx + 1}: PSNR = {neurlz['psnr']:.2f} dB (Δ = {diff_psnr:+.2f} dB)")

            # Calculate statistics across runs
            if args.num_runs > 1:
                psnr_values = [r['psnr'] for r in neurlz_runs]
                ssim_values = [r['ssim'] for r in neurlz_runs]
                mag_mse_values = [r['log_mag_mse'] for r in neurlz_runs]
                phase_mse_values = [r['phase_mse'] for r in neurlz_runs]
                
                avg_psnr = np.mean(psnr_values)
                std_psnr = np.std(psnr_values)
                avg_ssim = np.mean(ssim_values)
                std_ssim = np.std(ssim_values)
                avg_mag_mse = np.mean(mag_mse_values)
                std_mag_mse = np.std(mag_mse_values)
                avg_phase_mse = np.mean(phase_mse_values)
                std_phase_mse = np.std(phase_mse_values)
                
                # Create averaged result dictionary
                neurlz_avg = {
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'log_mag_mse': avg_mag_mse,
                    'phase_mse': avg_phase_mse,
                    'psnr_std': std_psnr,
                    'ssim_std': std_ssim,
                    'mag_mse_std': std_mag_mse,
                    'phase_mse_std': std_phase_mse,
                    'num_runs': args.num_runs,
                    'all_runs': neurlz_runs  # Store all individual runs
                }
                
                print(f"\n{'='*70}")
                print(f"STATISTICS ACROSS {args.num_runs} RUNS:")
                print(f"{'='*70}")
                print(f"PSNR:      {avg_psnr:.2f} ± {std_psnr:.2f} dB")
                print(f"SSIM:      {avg_ssim:.4f} ± {std_ssim:.4f}")
                print(f"Mag MSE:   {avg_mag_mse:.2e} ± {std_mag_mse:.2e}")
                print(f"Phase MSE: {avg_phase_mse:.2e} ± {std_phase_mse:.2e}")
                avg_diff_psnr = avg_psnr - baseline_stats['psnr']
                print(f"Average ΔPSNR: {avg_diff_psnr:+.2f} ± {std_psnr:.2f} dB")
            else:
                neurlz_avg = neurlz_runs[0]
                neurlz_avg['num_runs'] = 1
            # Calculate Differences
            res = {
                'file': filename,
                'baseline': baseline_stats,
                'neurlz': neurlz,
                'diff_psnr': neurlz['psnr'] - baseline_stats['psnr'],
                'diff_ssim': neurlz['ssim'] - baseline_stats['ssim'],
                'diff_mag': neurlz['log_mag_mse'] - baseline_stats['log_mag_mse'],   # Negative is good
                'diff_phase': neurlz['phase_mse'] - baseline_stats['phase_mse']      # Negative is good
            }
            all_results.append(res)
            
            # Print Comparison Block
            print(f"\n{'='*40}")
            print(f"RESULTS: {filename}")
            print(f"{'='*40}")
            if args.num_runs > 1:
                print(f"PSNR:      {baseline_stats['psnr']:.2f} -> {neurlz_avg['psnr']:.2f} ± {neurlz_avg['psnr_std']:.2f} ({res['diff_psnr']:+.2f} dB)")
                print(f"SSIM:      {baseline_stats['ssim']:.4f} -> {neurlz_avg['ssim']:.4f} ± {neurlz_avg['ssim_std']:.4f} ({res['diff_ssim']:+.4f})")
                print(f"Mag MSE:   {baseline_stats['log_mag_mse']:.2e} -> {neurlz_avg['log_mag_mse']:.2e} ± {neurlz_avg['mag_mse_std']:.2e} ({res['diff_mag']:.2e})")
                print(f"Phase MSE: {baseline_stats['phase_mse']:.2e} -> {neurlz_avg['phase_mse']:.2e} ± {neurlz_avg['phase_mse_std']:.2e} ({res['diff_phase']:.2e})")
            else:   
                print(f"PSNR:      {baseline_stats['psnr']:.2f} -> {neurlz_avg['psnr']:.2f} ({res['diff_psnr']:+.2f} dB)")
                print(f"SSIM:      {baseline_stats['ssim']:.4f} -> {neurlz_avg['ssim']:.4f} ({res['diff_ssim']:+.4f})")
                print(f"Mag MSE:   {baseline_stats['log_mag_mse']:.2e} -> {neurlz_avg['log_mag_mse']:.2e} ({res['diff_mag']:.2e})")
                print(f"Phase MSE: {baseline_stats['phase_mse']:.2e} -> {neurlz_avg['phase_mse']:.2e} ({res['diff_phase']:.2e})")
            # Final Summary Table (Updated Columns)
            print(f"\n\n{'='*70}")
            print(f"SUMMARY OF NEW METRICS")
            print(f"{'='*70}")
            table_data = []
            for r in all_results:
                table_data.append([
                    r['file'],
                    f"{r['neurlz']['psnr']:.2f}",
                    f"{r['neurlz']['ssim']:.4f}",
                    f"{r['neurlz']['log_mag_mse']:.2e}",
                    f"{r['neurlz']['phase_mse']:.2e}"
                ])
            print(tabulate(table_data, headers=['File', 'PSNR', 'SSIM', 'MagMSE', 'PhaseMSE'], tablefmt='grid'))
            # Comparison
            # ratio_improvement = (neurlz_stats['ratio'] / baseline_stats['ratio'] - 1) * 100
    #         psnr_improvement = neurlz_stats['psnr'] - baseline_stats['psnr']
    #         mean_error_improvement = ((baseline_stats['mean_error'] - neurlz_stats['mean_error']) 
    #                                  / baseline_stats['mean_error'] * 100)
    #         fft_psnr_improvement = neurlz_stats['fft_psnr'] - baseline_stats['fft_psnr']

    #         print(f"\n{'─'*70}")
    #         print(f"COMPARISON: Baseline vs NeurLZ")
    #         print(f"{'─'*70}")
    #         print(f"Compression ratio: {baseline_stats['ratio']:.2f}x → {neurlz_stats['ratio']:.2f}x "
    #               f"({ratio_improvement:+.1f}%)")
    #         print(f"PSNR: {baseline_stats['psnr']:.2f} dB → {neurlz_stats['psnr']:.2f} dB "
    #               f"({psnr_improvement:+.2f} dB)")
    #         print(f"Mean error: {baseline_stats['mean_error']:.3e} → {neurlz_stats['mean_error']:.3e} "
    #               f"({mean_error_improvement:+.1f}%)")
    #         print(f"FFT PSNR: {baseline_stats['fft_psnr']:.2f} dB → {neurlz_stats['fft_psnr']:.2f} dB "
    #               f"({fft_psnr_improvement:+.2f} dB)")
    #         print(f"Base SZ3 max error: {neurlz_stats['base_max_error']:.3e}")
    #         print(f"Enhanced max error: {neurlz_stats['max_error']:.3e}")
            
    #         # Store results
    #         all_results.append({
    #             'file': filename,
    #             'eb_mode': eb_mode,
    #             'absolute_error_bound': absolute_error_bound,
    #             'relative_error_bound': relative_error_bound,
    #             'pwr_error_bound': pwr_error_bound,
    #             'spatial_dims': args.spatial_dims,
    #             'slice_order': args.slice_order,
    #             'val_split': args.val_split,
    #             'track_losses': args.track_losses,
    #             'num_res_blocks': args.num_res_blocks,
    #             'model': args.model,
    #             'baseline': baseline_stats,
    #             'neurlz': neurlz_stats,
    #             'improvements': {
    #                 'ratio_pct': ratio_improvement,
    #                 'psnr_db': psnr_improvement,
    #                 'mean_error_pct': mean_error_improvement,
    #                 'fft_psnr_db': fft_psnr_improvement,
    #             }
    #         })
    
    # # Summary table
    # print(f"\n\n{'='*70}")
    # print(f"OVERALL SUMMARY")
    # print(f"{'='*70}\n")
    
    # table_data = []
    # for r in all_results:
    #     table_data.append([
    #         r['file'],
    #         r['eb_mode'],
    #         r['absolute_error_bound'],
    #         r['relative_error_bound'],
    #         r['pwr_error_bound'],
    #         r['model'],
    #         r['num_res_blocks'],
    #         f"{r['baseline']['ratio']:.2f}",
    #         f"{r['neurlz']['ratio']:.2f}",
    #         f"{r['improvements']['ratio_pct']:+.1f}%",
    #         f"{r['baseline']['psnr']:.2f}",
    #         f"{r['neurlz']['psnr']:.2f}",
    #         f"{r['improvements']['psnr_db']:+.2f}",
    #         f"{r['baseline']['fft_psnr']:.2f}",
    #         f"{r['neurlz']['fft_psnr']:.2f}",
    #         f"{r['improvements']['fft_psnr_db']:+.2f}",
    #         f"{r['improvements']['mean_error_pct']:+.1f}%",
    #         '✓' if r['neurlz']['within_bound'] else '✗',
    #     ])
    
    # headers = ['File', 'EB Mode', 'Abs EB', 'Rel EB', 'Pwr EB', 'Model', 'ResBlks', 
    #            'Base Ratio', 'NeurLZ Ratio', 'ΔRatio',
    #            'Base PSNR', 'NeurLZ PSNR', 'ΔPSNR', 
    #            'Base FFT PSNR', 'NeurLZ FFT PSNR', 'ΔFFT PSNR',
    #            'ΔMeanErr%', 'Bound OK']
    # print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # # Calculate averages
    # print(f"\n{'='*70}")
    # print("AVERAGES")
    # print(f"{'='*70}")
    
    # avg_ratio_improvement = np.mean([r['improvements']['ratio_pct'] for r in all_results])
    # avg_psnr_improvement = np.mean([r['improvements']['psnr_db'] for r in all_results])
    # avg_fft_psnr_improvement = np.mean([r['improvements']['fft_psnr_db'] for r in all_results])
    # avg_mean_error_improvement = np.mean([r['improvements']['mean_error_pct'] for r in all_results])
    
    # print(f"Average ratio improvement: {avg_ratio_improvement:+.2f}%")
    # print(f"Average PSNR improvement: {avg_psnr_improvement:+.2f} dB")
    # print(f"Average FFT PSNR improvement: {avg_fft_psnr_improvement:+.2f} dB")
    # print(f"Average mean error reduction: {avg_mean_error_improvement:+.2f}%")
    
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
    output_file = os.path.join(args.output_dir, 'neurlz_correct_results_7inputfreq2d_multiloss_postprocess.json')
    with open(output_file, 'w') as f:
        serializable_data = {
            'results': convert_to_json_serializable(all_results),
            'config': vars(args),
        }
        json.dump(serializable_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()