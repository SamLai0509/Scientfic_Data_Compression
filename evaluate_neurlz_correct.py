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

# Import functions from compression_function.py
try:
    from compression_function import (
        compute_spectral_metrics,
        compute_ssim_3d,
        normalize_roi_to_3d_boxes,
    )
except ImportError:
    from .compression_function import (
        compute_spectral_metrics,
        compute_ssim_3d,
        normalize_roi_to_3d_boxes,
    )


# =========================================================
# ROI parsing utilities
# =========================================================

def parse_roi_string(s: str):
    """
    Parse ROI definition strings.

    Supported formats:
      1) 3D box: "x0,x1,y0,y1,z0,z1"
      2) 2D ROI: "z,k,x0,x1,y0,y1"  (also supports "x,k,..." or "y,k,...")
         meaning a rectangle on a single slice; internally treated as a 3D box with thickness=1

    Returns:
      - 3D box tuple: (x0,x1,y0,y1,z0,z1)
      - or 2D spec tuple: (axis, k, u0,u1,v0,v1)
    """
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 6:
        raise ValueError(f"ROI string must have 6 fields, got {len(parts)}: {s}")

    # detect axis-form
    if parts[0] in ("x", "y", "z"):
        axis = parts[0]
        k = int(parts[1])
        u0, u1, v0, v1 = map(int, parts[2:])
        return (axis, k, u0, u1, v0, v1)
    else:
        # 3D box
        x0, x1, y0, y1, z0, z1 = map(int, parts)
        return (x0, x1, y0, y1, z0, z1)


# Note: normalize_roi_to_3d_boxes, compute_spectral_metrics, and compute_ssim_3d
# are now imported from compression_function.py

def compute_standard_metrics(original, reconstructed):
    """
    Returns psnr, mse, max_error, mean_error, nrmse.
    """
    error = np.abs(original - reconstructed)
    max_error = float(np.max(error))
    mean_error = float(np.mean(error))
    data_range = float(np.max(original) - np.min(original))
    mse = float(np.mean((original - reconstructed) ** 2))
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
    nrmse = (np.sqrt(mse) / data_range) if data_range > 0 else float('nan')
    return psnr, mse, max_error, mean_error, nrmse


def compute_roi_metrics(data, recon, roi_boxes_3d, ssim_slice_axis=0):
    """
    Compute metrics on concatenated ROI voxels (simple) + per-ROI report.

    Returns:
      roi_summary: dict with aggregated ROI metrics
      per_roi: list of dict metrics for each ROI box
    """
    if not roi_boxes_3d:
        return None, []

    per_roi = []
    # aggregated by weighted average (by voxel count)
    total_vox = 0
    agg_mse = 0.0
    agg_abs_err_sum = 0.0
    agg_abs_err_max = 0.0

    # SSIM / spectral: average across ROI boxes (unweighted, or weighted). Here use unweighted average.
    ssim_list = []
    mag_list = []
    phase_list = []

    for idx, (x0, x1, y0, y1, z0, z1) in enumerate(roi_boxes_3d):
        o = data[x0:x1, y0:y1, z0:z1]
        r = recon[x0:x1, y0:y1, z0:z1]

        psnr, mse, max_error, mean_error, nrmse = compute_standard_metrics(o, r)

        # SSIM
        if o.ndim == 3:
            roi_ssim = compute_ssim_3d(o, r, slice_axis=ssim_slice_axis)
        else:
            d_min, d_max = o.min(), o.max()
            rng = d_max - d_min + 1e-9
            roi_ssim = ssim((o - d_min) / rng, (r - d_min) / rng, data_range=1.0)

        # Spectral (ROI volume small -> OK)
        log_mag_mse, phase_mse = compute_spectral_metrics(o, r)

        per_roi.append({
            "roi_index": idx,
            "box": (x0, x1, y0, y1, z0, z1),
            "psnr": float(psnr),
            "mse": float(mse),
            "max_error": float(max_error),
            "mean_error": float(mean_error),
            "ssim": float(roi_ssim),
            "log_mag_mse": float(log_mag_mse),
            "phase_mse": float(phase_mse),
        })

        vox = o.size
        total_vox += vox
        agg_mse += mse * vox
        agg_abs_err_sum += mean_error * vox
        agg_abs_err_max = max(agg_abs_err_max, max_error)

        ssim_list.append(roi_ssim)
        mag_list.append(log_mag_mse)
        phase_list.append(phase_mse)

    # aggregated metrics
    agg_mse /= max(total_vox, 1)
    agg_mean_abs = agg_abs_err_sum / max(total_vox, 1)

    # For PSNR on aggregated ROI, use aggregated MSE and aggregated ROI range.
    # Use the overall range across all ROI voxels.
    roi_vals = []
    for (x0, x1, y0, y1, z0, z1) in roi_boxes_3d:
        roi_vals.append(data[x0:x1, y0:y1, z0:z1].ravel())
    roi_vals = np.concatenate(roi_vals, axis=0)
    roi_range = float(np.max(roi_vals) - np.min(roi_vals))

    roi_psnr = 20 * np.log10(roi_range) - 10 * np.log10(agg_mse) if agg_mse > 0 and roi_range > 0 else float('inf')

    roi_summary = {
        "roi_count": len(roi_boxes_3d),
        "roi_total_voxels": int(total_vox),
        "roi_psnr": float(roi_psnr),
        "roi_mse": float(agg_mse),
        "roi_mean_abs_error": float(agg_mean_abs),
        "roi_max_error": float(agg_abs_err_max),
        "roi_ssim_mean": float(np.mean(ssim_list)) if ssim_list else None,
        "roi_log_mag_mse_mean": float(np.mean(mag_list)) if mag_list else None,
        "roi_phase_mse_mean": float(np.mean(phase_list)) if phase_list else None,
    }

    return roi_summary, per_roi

def compute_bg_metrics(data, recon, roi_boxes_3d):
    """
    Compute metrics on background (non-ROI) regions.
    
    Returns:
      bg_summary: dict with aggregated BG metrics
    """
    if not roi_boxes_3d:
        # If no ROI, entire volume is BG
        psnr, mse, max_error, mean_error, nrmse = compute_standard_metrics(data, recon)
        return {
            "bg_psnr": float(psnr),
            "bg_mse": float(mse),
            "bg_mean_abs_error": float(mean_error),
            "bg_max_error": float(max_error),
            "bg_total_voxels": int(data.size),
        }
    
    # Create ROI mask
    from compression_function import create_roi_mask
    roi_mask = create_roi_mask(data.shape, roi_boxes_3d)
    bg_mask = ~roi_mask  # Background is inverse of ROI mask
    
    # Extract BG regions
    data_bg = data[bg_mask]
    recon_bg = recon[bg_mask]
    
    if data_bg.size == 0:
        return {
            "bg_psnr": float('inf'),
            "bg_mse": 0.0,
            "bg_mean_abs_error": 0.0,
            "bg_max_error": 0.0,
            "bg_total_voxels": 0,
        }
    
    # Compute metrics on BG
    error = np.abs(data_bg - recon_bg)
    max_error = float(np.max(error))
    mean_error = float(np.mean(error))
    mse = float(np.mean((data_bg - recon_bg) ** 2))
    
    # PSNR using BG data range
    bg_range = float(np.max(data_bg) - np.min(data_bg))
    bg_psnr = 20 * np.log10(bg_range) - 10 * np.log10(mse) if mse > 0 and bg_range > 0 else float('inf')
    
    return {
        "bg_psnr": float(bg_psnr),
        "bg_mse": float(mse),
        "bg_mean_abs_error": float(mean_error),
        "bg_max_error": float(max_error),
        "bg_total_voxels": int(data_bg.size),
    }
# =========================================================
# UPDATED EVALUATION FUNCTIONS
# =========================================================

def evaluate_baseline_sz3(sz, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                            roi_boxes_3d=None,
                            verbose_print: bool = True):
    """Evaluate baseline SZ3 compression."""
    if verbose_print:
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
    psnr, mse, max_error, mean_error, nrmse = compute_standard_metrics(data, reconstructed)
    
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
    
    roi_summary, roi_details = compute_roi_metrics(data, reconstructed, roi_boxes_3d or [], ssim_slice_axis=0)
    
    # BG metrics
    bg_summary = compute_bg_metrics(data, reconstructed, roi_boxes_3d or [])

    if verbose_print:
        print(f"Compressed size: {sz3_size / (1024**2):.2f} MB")
        print(f"Compression ratio: {sz3_ratio:.2f}x")
        print(f"Compress time: {compress_time:.2f}s")
        print(f"\nQuality Metrics:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim_val:.4f} (Higher is better)")
        print(f"  Max Error: {max_error:.3e}")
        print(f"  Log-Mag MSE: {log_mag_mse:.4e} (Lower is better)")
        print(f"  Phase MSE: {phase_mse:.4e} (Lower is better)")

    if verbose_print and roi_summary is not None:
        print(f"\nROI Metrics (Baseline SZ3):")
        print(f"  ROI count: {roi_summary['roi_count']}")
        print(f"  ROI PSNR: {roi_summary['roi_psnr']:.2f} dB")
        print(f"  ROI SSIM(mean): {roi_summary['roi_ssim_mean']:.4f}")
        print(f"  ROI Log-Mag MSE(mean): {roi_summary['roi_log_mag_mse_mean']:.2e}")
        print(f"  ROI Phase MSE(mean): {roi_summary['roi_phase_mse_mean']:.2e}")
    
    if verbose_print and bg_summary is not None:
        print(f"\nBG Metrics (Baseline SZ3):")
        print(f"  BG PSNR: {bg_summary['bg_psnr']:.2f} dB")
        print(f"  BG MSE: {bg_summary['bg_mse']:.2e}")
        print(f"  BG Mean Abs Error: {bg_summary['bg_mean_abs_error']:.3e}")
        print(f"  BG Max Error: {bg_summary['bg_max_error']:.3e}")
        print(f"  BG Total Voxels: {bg_summary['bg_total_voxels']:,}")
    
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
        'roi_summary': roi_summary,
        'roi_details': roi_details,
        'bg_summary': bg_summary,
    }


def evaluate_neurlz(compressor, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                    online_epochs=50, learning_rate=1e-3, model_channels=4, model='tiny_residual_predictor', 
                    num_res_blocks=1, spatial_dims=3, slice_order='zxy', val_split=0.1, track_losses=True, 
                    evaluate_per_slice=True, enable_post_process=True, Patch_size=256, Batch_size=512,
                    save_components=False, components_dir='./compressed_components', filename='data',
                    roi_boxes_3d=None,
                    use_dual_models=False,
                    auto_select_roi=False,
                    roi_percentage=0.05,  # Percentage of dataset as ROI (default: 0.05 for 5%, 95% as BG)
                    model_bg=None,        # Model type for BG (if None, uses model)
                    model_roi=None,       # Model type for ROI (if None, uses model)
                    verbose_print: bool = True):      # Control printing in this function
    """Evaluate NeurLZ compression."""
    if verbose_print:
        print(f"\n{'─'*70}")
        print(f"NeurLZ: SZ3 + Online DNN Enhancement (+ optional ROI layer)")
        print(f"{'─'*70}")
    
    # Dual-only policy (matches compressor.py dual-only)
    if not use_dual_models:
        raise ValueError("Dual-only: 请传入 use_dual_models=True")
    if not auto_select_roi and not roi_boxes_3d:
        raise ValueError("Dual-only: 必须提供 ROI（--roi 或 --auto_select_roi）")
    # Compress (compat: if compressor doesn't accept ROI args, fall back)
    compress_kwargs = dict(
        eb_mode=eb_mode,
        absolute_error_bound=absolute_error_bound,
        relative_error_bound=relative_error_bound,
        pwr_error_bound=pwr_error_bound,
        online_epochs=online_epochs,
        learning_rate=learning_rate,
        model_channels=model_channels,
        model=model,
        verbose=bool(verbose_print),
        spatial_dims=spatial_dims,
        slice_order=slice_order,
        val_split=val_split,
        track_losses=track_losses,
        num_res_blocks=num_res_blocks,
        Patch_size=Patch_size,
        Batch_size=Batch_size,
        save_components=save_components,
        components_dir=components_dir,
        filename=filename
    )

    # ROI boxes and Dual Models (Scheme B)
    if roi_boxes_3d:
        compress_kwargs.update({
            "roi_boxes": roi_boxes_3d,
        })
    
    if auto_select_roi:
        compress_kwargs.update({
            "auto_select_roi": True,
            "roi_percentage": roi_percentage,
        })

    # Dual models (Scheme B) - requires ROI boxes
    if use_dual_models:
        compress_kwargs.update({
            "use_dual_models": True,
        })
        # Add separate model types if provided
        if model_bg is not None:
            compress_kwargs["model_bg"] = model_bg
        if model_roi is not None:
            compress_kwargs["model_roi"] = model_roi
    # Compress (dual-only: no fallback)
    package, compress_stats = compressor.compress(data, **compress_kwargs)
    
    # Save components if requested
    if save_components:
        base_name = os.path.splitext(filename)[0]
        compressor.save_components(package, components_dir, base_name, verbose=True)
    
    # Decompress
    decompress_start = time.time()
    reconstructed = compressor.decompress(package, verbose=bool(verbose_print), enable_post_process=enable_post_process)
    decompress_time = time.time() - decompress_start
    
    # --- Compute Metrics ---
    # 1. Standard
    psnr, mse, max_error, mean_error, nrmse = compute_standard_metrics(data, reconstructed)

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
    
    # Get ROI boxes from package metadata if not provided (e.g., when auto_select_roi is enabled)
    if not roi_boxes_3d:
        roi_boxes_3d = package.get('metadata', {}).get('roi_boxes_3d')
        if verbose_print and roi_boxes_3d:
            print(f"\n从压缩包 metadata 中获取到 {len(roi_boxes_3d)} 个 ROI boxes")
    
    # ROI metrics (evaluate on final reconstructed)
    roi_summary, roi_details = compute_roi_metrics(data, reconstructed, roi_boxes_3d or [], ssim_slice_axis=0)
    
    # BG metrics (evaluate on final reconstructed)
    bg_summary = compute_bg_metrics(data, reconstructed, roi_boxes_3d or [])

    # Save reconstructed file if components_dir is provided
    if save_components and components_dir:
        base_name = os.path.splitext(filename)[0]
        reconstructed_path = os.path.join(components_dir, f"{base_name}_reconstructed.f32")
        os.makedirs(components_dir, exist_ok=True)
        reconstructed.astype(np.float32).tofile(reconstructed_path)
        print(f"\n  Reconstructed data saved to: {reconstructed_path}")
        print(f"  Size: {reconstructed.nbytes / (1024**2):.2f} MB")

    # Combine stats
    result = {
        'compressed_size': int(compress_stats['total_size_mb'] * 1024**2) if 'total_size_mb' in compress_stats else None,
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
        'roi_summary': roi_summary,
        'roi_details': roi_details,
        'bg_summary': bg_summary,
    }
    if verbose_print and roi_summary is not None:
        print(f"\nROI Metrics (NeurLZ Final):")
        print(f"  ROI count: {roi_summary['roi_count']}")
        print(f"  ROI PSNR: {roi_summary['roi_psnr']:.2f} dB")
        print(f"  ROI SSIM(mean): {roi_summary['roi_ssim_mean']:.4f}")
        print(f"  ROI Log-Mag MSE(mean): {roi_summary['roi_log_mag_mse_mean']:.2e}")
        print(f"  ROI Phase MSE(mean): {roi_summary['roi_phase_mse_mean']:.2e}")
    
    if verbose_print and bg_summary is not None:
        print(f"\nBG Metrics (NeurLZ Final):")
        print(f"  BG PSNR: {bg_summary['bg_psnr']:.2f} dB")
        print(f"  BG MSE: {bg_summary['bg_mse']:.2e}")
        print(f"  BG Mean Abs Error: {bg_summary['bg_mean_abs_error']:.3e}")
        print(f"  BG Max Error: {bg_summary['bg_max_error']:.3e}")
        print(f"  BG Total Voxels: {bg_summary['bg_total_voxels']:,}")
    
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
    parser.add_argument('--print_level', type=str, default='summary',
                        choices=['summary', 'verbose'],
                        help='Console output verbosity. summary=compact tables, verbose=full step-by-step logs.')
    parser.add_argument('--online_epochs', type=int, default=50,
                       help='Epochs for online DNN training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for online training')
    parser.add_argument('--model_channels', type=int, default=4, nargs='+',
                       help='Base channels for tiny DNN (4 → ~3k params). Can specify multiple values, e.g., --model_channels 2 4 6 8 10')
    parser.add_argument('--model', type=str, default='tiny_residual_predictor',
                       choices=['tiny_residual_predictor', 
                       'tiny_frequency_residual_predictor_1_input', 
                       'tiny_frequency_residual_predictor_with_energy',  
                       'tiny_frequency_residual_predictor_4_inputs',
                       'tiny_frequency_residual_predictor_7_attn_roi'],
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


    # =========================
    # ROI arguments (for Scheme B dual models)
    # =========================
    parser.add_argument('--roi', type=str, nargs='*', default=None,
                        help=(
                            "ROI definitions for Scheme B dual models. Examples:\n"
                            "  3D ROI: 100,164,200,264,300,364\n"
                            "  2D ROI: z,128,100,164,200,264\n"
                            "You can pass multiple ROI strings.\n"
                            "If not provided and --auto_select_roi is enabled, ROI will be auto-selected."
                        ))
    parser.add_argument('--auto_select_roi', action='store_true', default=False,
                        help='Auto-select ROI from error distribution based on percentage (requires use_dual_models)')
    parser.add_argument('--roi_percentage', type=float, default=0.05,
                        help='Percentage of dataset as ROI (default: 0.05 for 5%%, 95%% as BG)')
    # =========================
    # Dual Model arguments (Scheme B)
    # =========================
    parser.add_argument('--use_dual_models', action='store_true', default=False,
                        help='Enable dual-UNET architecture (Scheme B): separate BG and ROI models')
    parser.add_argument('--model_bg', type=str, default=None,
                        choices=['tiny_residual_predictor', 
                                'tiny_frequency_residual_predictor_1_input', 
                                'tiny_frequency_residual_predictor_with_energy',  
                                'tiny_frequency_residual_predictor_4_inputs',
                                'tiny_frequency_residual_predictor_7_attn_roi'],
                        help='Model type for BG (background) model in dual-UNET. If not specified, uses --model')
    parser.add_argument('--model_roi', type=str, default=None,
                        choices=['tiny_residual_predictor', 
                                'tiny_frequency_residual_predictor_1_input', 
                                'tiny_frequency_residual_predictor_with_energy',  
                                'tiny_frequency_residual_predictor_4_inputs',
                                'tiny_frequency_residual_predictor_7_attn_roi'],
                        help='Model type for ROI model in dual-UNET. If not specified, uses --model')


    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle device string properly
    if args.device.startswith('cuda'):
        device = args.device if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    verbose_print = (args.print_level == 'verbose')

    if verbose_print:
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
    if verbose_print:
        if isinstance(args.model_channels, list):
            print(f"Model channels: {args.model_channels} (will test each)")
            for ch in args.model_channels:
                print(f"  - Channels {ch}: ~{4*ch*ch:,} params")
        else:
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
            print(f"Components base directory: {args.components_dir}")
            if isinstance(args.model_channels, list):
                print(f"  Components will be saved in subdirectories for each channels:")
                for ch in args.model_channels:
                    if args.num_runs > 1:
                        print(f"    - channels_{ch}/run_1/ ... channels_{ch}/run_{args.num_runs}/")
                    else:
                        print(f"    - channels_{ch}/")

    # ROI parse
    roi_specs = []
    if args.roi:
        for s in args.roi:
            roi_specs.append(parse_roi_string(s))
        if verbose_print:
            print(f"\nROI enabled. Raw specs: {args.roi}")
    else:
        if verbose_print:
            print("\nROI disabled (no --roi provided).")
    
    # Dual models info
    if args.use_dual_models:
        if verbose_print:
            print(f"\nDual-UNET (Scheme B) enabled: separate BG and ROI models")
        # Dual-only with auto_select_roi support: no need to require manual --roi
        if (not args.roi) and (not args.auto_select_roi):
            raise ValueError("Dual-only: use_dual_models=True 时必须提供 --roi 或开启 --auto_select_roi")
    else:
        if verbose_print:
            print(f"\nSingle model mode (default)")
    if verbose_print:
        print(f"{'='*70}")
    
    # Initialize
    sz = SZ(args.sz_lib)
    compressor = NeurLZCompressor(sz_lib_path=args.sz_lib, device=device)
    
    # Results storage
    all_results = []
    
    # Evaluate each combination
    import itertools
    # Ensure model_channels is a list
    if not isinstance(args.model_channels, list):
        model_channels_list = [args.model_channels]
    else:
        model_channels_list = args.model_channels
    
    for eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound, model_channels in itertools.product(
        args.eb_modes, args.absolute_error_bounds, args.relative_error_bounds, args.pwr_error_bounds, model_channels_list
    ):
        if verbose_print:
            print(f"\n\n{'#'*70}")
            print(f"TESTING CONFIGURATION:")
            print(f"  Error Bound Mode: {eb_mode}")
            print(f"  Absolute Error Bound: {absolute_error_bound}")
            print(f"  Relative Error Bound: {relative_error_bound}")
            print(f"  Power Error Bound: {pwr_error_bound}")
            print(f"  Model: {args.model}")
            print(f"  Model Channels: {model_channels} (~{4*model_channels*model_channels:,} params)")
            print(f"  Channels Progress: {model_channels_list.index(model_channels) + 1}/{len(model_channels_list)}")
            print(f"{'#'*70}")
        
        for filename in args.test_files:
            filepath = os.path.join(args.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found, skipping...")
                continue
            
            if verbose_print:
                print(f"\n{'='*70}")
                print(f"File: {filename}")
                print(f"{'='*70}")
            
            # Load data
            data = np.fromfile(filepath, dtype=np.float32).reshape(512, 512, 512)
            if verbose_print:
                print(f"Shape: {data.shape}")
                print(f"Range: [{np.min(data):.3e}, {np.max(data):.3e}]")
                print(f"Size: {data.nbytes / (1024**2):.2f} MB")
        
                        # normalize ROI to boxes
            roi_boxes_3d = normalize_roi_to_3d_boxes(roi_specs, data.shape)
            if verbose_print and roi_boxes_3d:
                print("\nNormalized ROI boxes (X,Y,Z):")
                for i, b in enumerate(roi_boxes_3d):
                    print(f"  ROI[{i}]: {b}")
            # Baseline SZ3
            baseline_stats = evaluate_baseline_sz3(
                sz, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                roi_boxes_3d=roi_boxes_3d,
                verbose_print=verbose_print
            )
            # NeurLZ - Multiple runs for statistical evaluation
            if verbose_print:
                print(f"\n{'='*70}")
                print(f"Running NeurLZ evaluation {args.num_runs} times...")
                print(f"{'='*70}")

            # NeurLZ
            neurlz_runs = []
            for run_idx in range(args.num_runs):
                if verbose_print:
                    print(f"\n--- Run {run_idx + 1}/{args.num_runs} (Channels: {model_channels}) ---")
                
                # Prepare components directory for this channels and run
                if args.save_components:
                    # Create subdirectory structure: channels_{model_channels}/run_{run_idx + 1}/
                    if args.num_runs > 1:
                        components_dir_for_run = os.path.join(
                            args.components_dir, 
                            f"channels_{model_channels}",
                            f"run_{run_idx + 1}"
                        )
                    else:
                        # Single run: just use channels subdirectory
                        components_dir_for_run = os.path.join(
                            args.components_dir, 
                            f"channels_{model_channels}"
                        )
                    # Create directory if it doesn't exist
                    os.makedirs(components_dir_for_run, exist_ok=True)
                    if verbose_print:
                        print(f"  Components will be saved to: {components_dir_for_run}/")
                else:
                    components_dir_for_run = args.components_dir
                
                neurlz = evaluate_neurlz(
                    compressor, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound,
                    online_epochs=args.online_epochs,
                    learning_rate=args.learning_rate,
                    model_channels=model_channels,
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
                    components_dir=components_dir_for_run,
                    filename=filename,
                    roi_boxes_3d=roi_boxes_3d,
                    use_dual_models=args.use_dual_models,
                    auto_select_roi=args.auto_select_roi,  
                    roi_percentage=args.roi_percentage,
                    model_bg=args.model_bg,
                    model_roi=args.model_roi,
                    verbose_print=verbose_print,
                )
                neurlz_runs.append(neurlz)

                # Print individual run result
                diff_psnr = neurlz['psnr'] - baseline_stats['psnr']
                if verbose_print:
                    print(f"  Run {run_idx + 1}: PSNR = {neurlz['psnr']:.2f} dB (Δ = {diff_psnr:+.2f} dB)")

            # Calculate statistics across runs
            if args.num_runs > 1:
                psnr_values = [r['psnr'] for r in neurlz_runs]
                ssim_values = [r['ssim'] for r in neurlz_runs]
                mag_mse_values = [r['log_mag_mse'] for r in neurlz_runs]
                phase_mse_values = [r['phase_mse'] for r in neurlz_runs]
                
                avg_psnr = float(np.mean(psnr_values)); std_psnr = float(np.std(psnr_values))
                avg_ssim = float(np.mean(ssim_values)); std_ssim = float(np.std(ssim_values))
                avg_mag_mse = float(np.mean(mag_mse_values)); std_mag_mse = float(np.std(mag_mse_values))
                avg_phase_mse = float(np.mean(phase_mse_values)); std_phase_mse = float(np.std(phase_mse_values))

                
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
                if roi_boxes_3d and all(r.get("roi_summary") for r in neurlz_runs):
                    roi_psnr_vals = [r["roi_summary"]["roi_psnr"] for r in neurlz_runs]
                    roi_ssim_vals = [r["roi_summary"]["roi_ssim_mean"] for r in neurlz_runs]
                    neurlz_avg["roi_psnr"] = float(np.mean(roi_psnr_vals))
                    neurlz_avg["roi_psnr_std"] = float(np.std(roi_psnr_vals))
                    neurlz_avg["roi_ssim"] = float(np.mean(roi_ssim_vals))
                    neurlz_avg["roi_ssim_std"] = float(np.std(roi_ssim_vals))
                
                # BG metrics statistics
                if all(r.get("bg_summary") for r in neurlz_runs):
                    bg_psnr_vals = [r["bg_summary"]["bg_psnr"] for r in neurlz_runs]
                    bg_mse_vals = [r["bg_summary"]["bg_mse"] for r in neurlz_runs]
                    neurlz_avg["bg_summary"] = {
                        "bg_psnr": float(np.mean(bg_psnr_vals)),
                        "bg_psnr_std": float(np.std(bg_psnr_vals)),
                        "bg_mse": float(np.mean(bg_mse_vals)),
                        "bg_mse_std": float(np.std(bg_mse_vals)),
                        "bg_total_voxels": neurlz_runs[0]["bg_summary"]["bg_total_voxels"],  # Same for all runs
                    }
                if verbose_print:
                    print(f"\n{'='*70}")
                    print(f"STATISTICS ACROSS {args.num_runs} RUNS:")
                    print(f"{'='*70}")
                    print(f"PSNR:      {avg_psnr:.2f} ± {std_psnr:.2f} dB")
                    print(f"SSIM:      {avg_ssim:.4f} ± {std_ssim:.4f}")
                    if "roi_psnr" in neurlz_avg:
                        print(f"ROI PSNR:  {neurlz_avg['roi_psnr']:.2f} ± {neurlz_avg['roi_psnr_std']:.2f} dB")
                        print(f"ROI SSIM:  {neurlz_avg['roi_ssim']:.4f} ± {neurlz_avg['roi_ssim_std']:.4f}")
                    if "bg_summary" in neurlz_avg:
                        print(f"BG PSNR:   {neurlz_avg['bg_summary']['bg_psnr']:.2f} ± {neurlz_avg['bg_summary']['bg_psnr_std']:.2f} dB")
                        print(f"BG MSE:    {neurlz_avg['bg_summary']['bg_mse']:.2e} ± {neurlz_avg['bg_summary']['bg_mse_std']:.2e}")
                    print(f"Mag MSE:   {avg_mag_mse:.2e} ± {std_mag_mse:.2e}")
                    print(f"Phase MSE: {avg_phase_mse:.2e} ± {std_phase_mse:.2e}")
                    avg_diff_psnr = avg_psnr - baseline_stats['psnr']
                    print(f"Average ΔPSNR: {avg_diff_psnr:+.2f} ± {std_psnr:.2f} dB")
            else:
                neurlz_avg = neurlz_runs[0]
                neurlz_avg['num_runs'] = 1
                # Ensure bg_summary is preserved for single run
                if neurlz_avg.get("bg_summary") is None and neurlz_runs[0].get("bg_summary"):
                    neurlz_avg["bg_summary"] = neurlz_runs[0]["bg_summary"]

             # Differences
            diff_psnr = neurlz_avg['psnr'] - baseline_stats['psnr']
            diff_ssim = neurlz_avg['ssim'] - baseline_stats['ssim']
            diff_mag = neurlz_avg['log_mag_mse'] - baseline_stats['log_mag_mse']
            diff_phase = neurlz_avg['phase_mse'] - baseline_stats['phase_mse']

            # ROI diffs
            diff_roi_psnr = None
            diff_roi_ssim = None
            if baseline_stats.get("roi_summary") and neurlz_avg.get("roi_summary"):
                diff_roi_psnr = neurlz_avg["roi_summary"]["roi_psnr"] - baseline_stats["roi_summary"]["roi_psnr"]
                diff_roi_ssim = neurlz_avg["roi_summary"]["roi_ssim_mean"] - baseline_stats["roi_summary"]["roi_ssim_mean"]
            elif "roi_psnr" in neurlz_avg and baseline_stats.get("roi_summary"):
                diff_roi_psnr = neurlz_avg["roi_psnr"] - baseline_stats["roi_summary"]["roi_psnr"]
                diff_roi_ssim = neurlz_avg["roi_ssim"] - baseline_stats["roi_summary"]["roi_ssim_mean"]
            
            # BG diffs
            diff_bg_psnr = None
            if baseline_stats.get("bg_summary") and neurlz_avg.get("bg_summary"):
                diff_bg_psnr = neurlz_avg["bg_summary"]["bg_psnr"] - baseline_stats["bg_summary"]["bg_psnr"]
            elif "bg_summary" in neurlz_avg and baseline_stats.get("bg_summary"):
                diff_bg_psnr = neurlz_avg["bg_summary"]["bg_psnr"] - baseline_stats["bg_summary"]["bg_psnr"]

            res = {
                'file': filename,
                'model_channels': model_channels,
                'eb_mode': eb_mode,
                'absolute_error_bound': absolute_error_bound,
                'relative_error_bound': relative_error_bound,
                'pwr_error_bound': pwr_error_bound,
                'baseline': baseline_stats,
                'neurlz': neurlz_avg,
                'diff_psnr': neurlz_avg['psnr'] - baseline_stats['psnr'],
                'diff_ssim': neurlz_avg['ssim'] - baseline_stats['ssim'],
                'diff_mag': neurlz_avg['log_mag_mse'] - baseline_stats['log_mag_mse'],   # Negative is good
                'diff_phase': neurlz_avg['phase_mse'] - baseline_stats['phase_mse'],    # Negative is good
                'diff_roi_psnr': diff_roi_psnr,
                'diff_roi_ssim': diff_roi_ssim,
                'diff_bg_psnr': diff_bg_psnr,
                'roi_bg_psnr_diff': diff_roi_psnr - diff_bg_psnr if (diff_roi_psnr is not None and diff_bg_psnr is not None) else None,
                'roi_boxes': roi_boxes_3d,
                'use_dual_models': args.use_dual_models,
            }
            all_results.append(res)
            
            # In summary mode, avoid noisy per-channel narrative prints.
            # The compact per-file and final summary tables printed later include ΔPSNR and ΔROI PSNR.
            
            # Summary Table for this file across all tested channels (up to this point)
            # Group results by file to show channels comparison
            current_file_results = [r for r in all_results if r['file'] == filename]
            if len(current_file_results) > 1:  # Multiple channels tested for this file
                print(f"\n\n{'='*70}")
                print(f"CHANNELS COMPARISON FOR {filename}")
                print(f"{'='*70}")
                table_data = []
                for r in sorted(current_file_results, key=lambda x: x.get('model_channels', 0)):
                    ch = r.get('model_channels', 'N/A')
                    ne = r['neurlz']
                    table_row = [
                        ch,
                        f"{ne['psnr']:.2f}",
                        f"{ne['ssim']:.4f}",
                        f"{ne['log_mag_mse']:.2e}",
                        f"{ne['phase_mse']:.2e}",
                        f"{r['diff_psnr']:+.2f}",
                        f"{r['diff_ssim']:+.4f}"
                    ]
                    # append ROI columns if enabled
                    if roi_boxes_3d and r.get("diff_roi_psnr") is not None:
                        # if neurlz contains roi_summary
                        if ne.get("roi_summary"):
                            roi_psnr = ne["roi_summary"]["roi_psnr"]
                            roi_ssim = ne["roi_summary"]["roi_ssim_mean"]
                        else:
                            roi_psnr = ne.get("roi_psnr", np.nan)
                            roi_ssim = ne.get("roi_ssim", np.nan)
                        table_row += [
                            f"{roi_psnr:.2f}",
                            f"{roi_ssim:.4f}",
                            f"{r['diff_roi_psnr']:+.2f}",
                            f"{r['diff_roi_ssim']:+.4f}"
                        ]
                    # append BG columns if available
                    if r.get("diff_bg_psnr") is not None:
                        if ne.get("bg_summary"):
                            bg_psnr = ne["bg_summary"]["bg_psnr"]
                        else:
                            bg_psnr = np.nan
                        table_row += [
                            f"{bg_psnr:.2f}",
                            f"{r['diff_bg_psnr']:+.2f}"
                        ]
                    table_data.append(table_row)
                headers = ['Channels', 'PSNR (dB)', 'SSIM', 'MagMSE', 'PhaseMSE', 'ΔPSNR', 'ΔSSIM']
                if roi_boxes_3d:
                    headers += ['ROI PSNR', 'ROI SSIM', 'ΔROI PSNR', 'ΔROI SSIM']
                # Add BG headers if any result has BG data
                if any(r.get("diff_bg_psnr") is not None for r in current_file_results):
                    headers += ['BG PSNR', 'ΔBG PSNR']
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                
                # Find best performing channels
                best_psnr_idx = max(range(len(current_file_results)), 
                                   key=lambda i: current_file_results[i]['neurlz']['psnr'])
                best_ssim_idx = max(range(len(current_file_results)), 
                                   key=lambda i: current_file_results[i]['neurlz']['ssim'])
                best_mag_idx = min(range(len(current_file_results)), 
                                  key=lambda i: current_file_results[i]['neurlz']['log_mag_mse'])
                best_phase_idx = min(range(len(current_file_results)), 
                                    key=lambda i: current_file_results[i]['neurlz']['phase_mse'])
                
                print(f"\nBest Performance:")
                print(f"  PSNR:      Channels {current_file_results[best_psnr_idx]['model_channels']} ({current_file_results[best_psnr_idx]['neurlz']['psnr']:.2f} dB)")
                print(f"  SSIM:      Channels {current_file_results[best_ssim_idx]['model_channels']} ({current_file_results[best_ssim_idx]['neurlz']['ssim']:.4f})")
                print(f"  Mag MSE:   Channels {current_file_results[best_mag_idx]['model_channels']} ({current_file_results[best_mag_idx]['neurlz']['log_mag_mse']:.2e})")
                print(f"  Phase MSE: Channels {current_file_results[best_phase_idx]['model_channels']} ({current_file_results[best_phase_idx]['neurlz']['phase_mse']:.2e})")
                print(f"{'='*70}")
            else:
                # Single channels tested, show simple summary
                print(f"\n{'='*70}")
                print(f"SUMMARY FOR {filename}")
                print(f"{'='*70}")
                r = current_file_results[0]
                table_data = [[
                    r.get('model_channels', 'N/A'),
                    f"{r['neurlz']['psnr']:.2f}",
                    f"{r['neurlz']['ssim']:.4f}",
                    f"{r['neurlz']['log_mag_mse']:.2e}",
                    f"{r['neurlz']['phase_mse']:.2e}",
                    f"{r['diff_psnr']:+.2f}",
                    f"{r['diff_ssim']:+.4f}"
                ]]
                print(tabulate(table_data, headers=['Channels', 'PSNR', 'SSIM', 'MagMSE', 'PhaseMSE', 'ΔPSNR', 'ΔSSIM'], tablefmt='grid'))
                print(f"{'='*70}")
                
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
    
    # Final Summary: Show performance comparison across all channels for each file
    if len(all_results) > 0:
        print(f"\n\n{'='*70}")
        print(f"FINAL PERFORMANCE SUMMARY: ALL CHANNELS COMPARISON")
        print(f"{'='*70}")
        
        # Group results by file
        files_tested = set(r['file'] for r in all_results)
        for file in sorted(files_tested):
            file_results = [r for r in all_results if r['file'] == file]
            if len(file_results) > 1:  # Multiple channels tested for this file
                print(f"\n{'-'*70}")
                print(f"File: {file}")
                print(f"{'-'*70}")
                table_data = []
                baseline_for_file = file_results[0]['baseline']
                has_roi = any(r.get("diff_roi_psnr") is not None for r in file_results)
                has_bg = any(r.get("diff_bg_psnr") is not None for r in file_results)
                
                for r in sorted(file_results, key=lambda x: x.get('model_channels', 0)):
                    ch = r.get('model_channels', 'N/A')
                    if args.num_runs > 1:
                        psnr_val = f"{r['neurlz']['psnr']:.2f} ± {r['neurlz'].get('psnr_std', 0):.2f}"
                        ssim_val = f"{r['neurlz']['ssim']:.4f} ± {r['neurlz'].get('ssim_std', 0):.4f}"
                    else:
                        psnr_val = f"{r['neurlz']['psnr']:.2f}"
                        ssim_val = f"{r['neurlz']['ssim']:.4f}"
                    
                    # Calculate improvement percentage
                    psnr_improvement_pct = ((r['neurlz']['psnr'] - baseline_for_file['psnr']) / baseline_for_file['psnr'] * 100) if baseline_for_file['psnr'] > 0 else 0
                    ssim_improvement_pct = ((r['neurlz']['ssim'] - baseline_for_file['ssim']) / baseline_for_file['ssim'] * 100) if baseline_for_file['ssim'] > 0 else 0
                    
                    row = [
                        ch,
                        f"{baseline_for_file['psnr']:.2f}",
                        psnr_val,
                        f"{r['diff_psnr']:+.2f}",
                        f"{psnr_improvement_pct:+.2f}%",
                        f"{baseline_for_file['ssim']:.4f}",
                        ssim_val,
                        f"{r['diff_ssim']:+.4f}",
                        f"{ssim_improvement_pct:+.2f}%",
                        f"{r['neurlz']['log_mag_mse']:.2e}",
                        f"{r['neurlz']['phase_mse']:.2e}"
                    ]

                    # --- ROI columns (key fix: include ΔROI PSNR in final summary) ---
                    if has_roi:
                        base_roi = baseline_for_file.get("roi_summary", {}) or {}
                        base_roi_psnr = base_roi.get("roi_psnr", np.nan)
                        # neurlz roi can be stored in roi_summary (single run) or roi_psnr (multi-run avg)
                        ne = r.get("neurlz", {}) or {}
                        if ne.get("roi_summary"):
                            ne_roi_psnr = ne["roi_summary"].get("roi_psnr", np.nan)
                        else:
                            ne_roi_psnr = ne.get("roi_psnr", np.nan)
                        row += [
                            f"{base_roi_psnr:.2f}" if np.isfinite(base_roi_psnr) else "N/A",
                            f"{ne_roi_psnr:.2f}" if np.isfinite(ne_roi_psnr) else "N/A",
                            f"{(r.get('diff_roi_psnr') if r.get('diff_roi_psnr') is not None else np.nan):+.2f}" if r.get('diff_roi_psnr') is not None else "N/A",
                        ]

                    # --- BG columns ---
                    if has_bg:
                        base_bg = baseline_for_file.get("bg_summary", {}) or {}
                        base_bg_psnr = base_bg.get("bg_psnr", np.nan)
                        ne = r.get("neurlz", {}) or {}
                        ne_bg = ne.get("bg_summary", {}) or {}
                        ne_bg_psnr = ne_bg.get("bg_psnr", np.nan)
                        row += [
                            f"{base_bg_psnr:.2f}" if np.isfinite(base_bg_psnr) else "N/A",
                            f"{ne_bg_psnr:.2f}" if np.isfinite(ne_bg_psnr) else "N/A",
                            f"{(r.get('diff_bg_psnr') if r.get('diff_bg_psnr') is not None else np.nan):+.2f}" if r.get('diff_bg_psnr') is not None else "N/A",
                        ]

                    table_data.append(row)
                
                headers = ['Channels', 'Base PSNR', 'NeurLZ PSNR', 'ΔPSNR', 'ΔPSNR%', 
                          'Base SSIM', 'NeurLZ SSIM', 'ΔSSIM', 'ΔSSIM%', 'MagMSE', 'PhaseMSE']
                if has_roi:
                    headers += ['Base ROI PSNR', 'NeurLZ ROI PSNR', 'ΔROI PSNR']
                if has_bg:
                    headers += ['Base BG PSNR', 'NeurLZ BG PSNR', 'ΔBG PSNR']
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                
                # Highlight best performing channels
                best_psnr = max(file_results, key=lambda x: x['neurlz']['psnr'])
                best_ssim = max(file_results, key=lambda x: x['neurlz']['ssim'])
                print(f"\nBest Channels for {file}:")
                print(f"  Highest PSNR: Channels {best_psnr['model_channels']} ({best_psnr['neurlz']['psnr']:.2f} dB, Δ={best_psnr['diff_psnr']:+.2f} dB)")
                print(f"  Highest SSIM: Channels {best_ssim['model_channels']} ({best_ssim['neurlz']['ssim']:.4f}, Δ={best_ssim['diff_ssim']:+.4f})")
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: Tested {len(files_tested)} file(s) with {len(model_channels_list)} channels configuration(s)")
        print(f"  Total results: {len(all_results)}")
        print(f"{'='*70}\n")
    
    # Save results - include channels in filename if multiple channels tested
    if isinstance(args.model_channels, list) and len(args.model_channels) > 1:
        channels_str = "_".join(map(str, args.model_channels))
        output_file = os.path.join(args.output_dir, f'neurlz_correct_results_channels_{channels_str}.json')
    else:
        ch_val = args.model_channels[0] if isinstance(args.model_channels, list) else args.model_channels
        output_file = os.path.join(args.output_dir, f'neurlz_correct_results_channels_{ch_val}.json')
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