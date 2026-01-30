"""
Compression utility functions for NeurLZ compressor.

This module contains reusable functions for:
- Quality metrics computation (spectral, SSIM)
- ROI (Region of Interest) operations
- Model creation and management
- Post-processing
- Verification
"""

import numpy as np
import torch
import torch.nn as nn
import sys

# Add neural_compression to path for Model imports
sys.path.insert(0, '/Users/923714256/Data_compression/neural_compression')

try:
    from Model import (
        TinyResidualPredictor, 
        TinyFrequencyResidualPredictorWithEnergy,
        TinyFrequencyResidualPredictor_1_input,
        TinyFrequencyResidualPredictor_4_inputs,
        TinyFrequencyResidualPredictor7_AttnROI,
    )
except ImportError:
    from Model import (
        TinyResidualPredictor, 
        TinyFrequencyResidualPredictorWithEnergy,
        TinyFrequencyResidualPredictor_1_input,
        TinyFrequencyResidualPredictor_4_inputs,
        TinyFrequencyResidualPredictor7_AttnROI,
    )

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


# =============================================================================
# GroupNorm Helper
# =============================================================================

def compute_gn_groups(channels, preferred=4):
    """
    Compute appropriate GroupNorm groups that divides channels evenly.
    
    Args:
        channels: Number of channels
        preferred: Preferred number of groups (default: 4)
    
    Returns:
        Number of groups that divides channels evenly, choosing the largest
        divisor that is <= preferred, or the largest divisor overall if all
        divisors are > preferred
    """
    if channels < preferred:
        return channels
    elif channels % preferred == 0:
        return preferred
    
    # Find all divisors of channels
    divisors = []
    for i in range(1, channels + 1):
        if channels % i == 0:
            divisors.append(i)
    
    # Find the largest divisor <= preferred
    valid_divisors = [d for d in divisors if d <= preferred]
    if valid_divisors:
        return max(valid_divisors)
    else:
        return min(divisors)


# =============================================================================
# Quality Metrics
# =============================================================================

def compute_spectral_metrics(original, reconstructed):
    """
    Computes Log-Magnitude MSE and Phase MSE.
    Crucial for wave-based simulations (Cosmology, Seismic).
    Memory-optimized version for large 3D datasets.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
    
    Returns:
        tuple: (log_mag_mse, phase_mse)
    """
    try:
        # Compute FFT (n-dimensional) - use float32 to save memory
        fft_orig = np.fft.fftn(original.astype(np.float32))
        fft_recon = np.fft.fftn(reconstructed.astype(np.float32))
        
        # 1. Log-Magnitude Error (Energy Fidelity)
        mag_orig = np.log1p(np.abs(fft_orig))
        del fft_orig
        mag_recon = np.log1p(np.abs(fft_recon))
        del fft_recon
        
        log_mag_mse = np.mean((mag_orig - mag_recon) ** 2)
        
        # 2. Phase Error (Structural Alignment)
        if original.size > 100_000_000:  # > 100M elements
            phase_mse = np.nan
        else:
            fft_orig = np.fft.fftn(original.astype(np.float32))
            fft_recon = np.fft.fftn(reconstructed.astype(np.float32))
            phase_orig = np.angle(fft_orig)
            del fft_orig
            phase_recon = np.angle(fft_recon)
            del fft_recon
            phase_diff = np.abs(np.exp(1j * phase_orig) - np.exp(1j * phase_recon))
            del phase_orig, phase_recon
            phase_mse = np.mean(phase_diff ** 2)
            del phase_diff
        
        del mag_orig, mag_recon
        return log_mag_mse, phase_mse
    except MemoryError as e:
        print(f"Warning: Memory error in spectral metrics calculation: {e}")
        return np.nan, np.nan
    except Exception as e:
        print(f"Warning: Error in spectral metrics calculation: {e}")
        return np.nan, np.nan


def compute_ssim_3d(original, reconstructed, slice_axis=0):
    """
    Computes SSIM for 3D volumes by averaging 2D SSIM over slices.
    (Skimage SSIM is 2D by default, 3D support is experimental/slow)
    
    Args:
        original: Original 3D data
        reconstructed: Reconstructed 3D data
        slice_axis: Axis along which to compute slices (default: 0)
    
    Returns:
        float: Average SSIM across slices
    """
    if ssim is None:
        print("Warning: skimage not available, SSIM calculation skipped")
        return 0.0
    
    # Normalize data to [0, 1] for SSIM calculation
    d_min, d_max = original.min(), original.max()
    rng = d_max - d_min + 1e-9
    
    orig_norm = (original - d_min) / rng
    recon_norm = (reconstructed - d_min) / rng
    
    # Compute SSIM slice-by-slice along the specified axis
    ssim_accum = 0.0
    valid_slices = 0
    
    n_slices = original.shape[slice_axis]
    for i in range(n_slices):
        # Slice along the specified axis
        if slice_axis == 0:
            s1 = orig_norm[i]
            s2 = recon_norm[i]
        elif slice_axis == 1:
            s1 = orig_norm[:, i, :]
            s2 = recon_norm[:, i, :]
        else:  # slice_axis == 2
            s1 = orig_norm[:, :, i]
            s2 = recon_norm[:, :, i]
        
        try:
            val = ssim(s1, s2, data_range=1.0)
            ssim_accum += val
            valid_slices += 1
        except ValueError:
            pass
    
    if valid_slices == 0:
        return 0.0
    
    return ssim_accum / valid_slices


# =============================================================================
# ROI (Region of Interest) Functions
# =============================================================================
def create_precise_roi_mask_from_error(
    err_abs: np.ndarray,
    roi_percentage: float = 0.1,
) -> np.ndarray:
    """
    创建精确的ROI mask，直接选择top-k误差voxels，不创建boxes。
    
    Args:
        err_abs: Absolute error array of shape (X, Y, Z), dtype float32
        roi_percentage: Percentage of voxels to select as ROI (default: 0.1 for 10%)
    
    Returns:
        numpy.ndarray: Boolean mask of shape (X, Y, Z), True表示ROI voxel
    """
    X, Y, Z = err_abs.shape
    total_voxels = X * Y * Z
    num_roi_voxels = int(total_voxels * roi_percentage)
    num_roi_voxels = max(1, min(num_roi_voxels, total_voxels - 1))
    
    # Flatten error and find top-k voxels
    err_flat = err_abs.flatten()
    threshold_idx = total_voxels - num_roi_voxels
    threshold = np.partition(err_flat, threshold_idx)[threshold_idx]
    
    # Create precise binary mask
    roi_mask = err_abs >= threshold
    
    # Verify the actual percentage
    actual_roi_voxels = np.sum(roi_mask)
    actual_percentage = actual_roi_voxels / total_voxels
    
    return roi_mask

def create_roi_boxes_from_mask(
    roi_mask: np.ndarray,
    min_box_size: int = 8,
    max_boxes: int = 100,
    grid_size: int = None,
) -> list:
    """
    从精确的ROI mask创建boxes，只包含mask中的voxels。
    
    Args:
        roi_mask: Boolean mask of shape (X, Y, Z), True表示ROI voxel
        min_box_size: Minimum size for each box (default: 8)
        max_boxes: Maximum number of boxes to create (default: 100)
        grid_size: Grid size for grouping (if None, auto-calculate)
    
    Returns:
        list: List of ROI box tuples [(x0, x1, y0, y1, z0, z1), ...]
    """
    X, Y, Z = roi_mask.shape
    total_voxels = X * Y * Z
    num_roi_voxels = np.sum(roi_mask)
    
    if num_roi_voxels == 0:
        return []
    
    # Auto-calculate grid size if not provided
    if grid_size is None:
        avg_voxels_per_box = max(1, num_roi_voxels // max_boxes)
        grid_size = max(min_box_size, int(np.cbrt(avg_voxels_per_box * 8)))
        grid_size = min(grid_size, min(X, Y, Z) // 2)
    
    # Create grid and group ROI voxels
    grid_x = (X + grid_size - 1) // grid_size
    grid_y = (Y + grid_size - 1) // grid_size
    grid_z = (Z + grid_size - 1) // grid_size
    
    grid_voxels = {}
    roi_coords = np.where(roi_mask)
    
    for i in range(len(roi_coords[0])):
        x, y, z = roi_coords[0][i], roi_coords[1][i], roi_coords[2][i]
        gx, gy, gz = x // grid_size, y // grid_size, z // grid_size
        key = (gx, gy, gz)
        if key not in grid_voxels:
            grid_voxels[key] = []
        grid_voxels[key].append((x, y, z))
    
    # Create boxes from grid cells
    roi_boxes = []
    for (gx, gy, gz), voxels in grid_voxels.items():
        if len(voxels) < min_box_size:
            continue
        
        voxels_arr = np.array(voxels)
        x0 = max(0, int(voxels_arr[:, 0].min()))
        x1 = min(X, int(voxels_arr[:, 0].max()) + 1)
        y0 = max(0, int(voxels_arr[:, 1].min()))
        y1 = min(Y, int(voxels_arr[:, 1].max()) + 1)
        z0 = max(0, int(voxels_arr[:, 2].min()))
        z1 = min(Z, int(voxels_arr[:, 2].max()) + 1)
        
        # Ensure minimum size
        if (x1 - x0) < min_box_size:
            center_x = (x0 + x1) // 2
            x0 = max(0, center_x - min_box_size // 2)
            x1 = min(X, x0 + min_box_size)
        if (y1 - y0) < min_box_size:
            center_y = (y0 + y1) // 2
            y0 = max(0, center_y - min_box_size // 2)
            y1 = min(Y, y0 + min_box_size)
        if (z1 - z0) < min_box_size:
            center_z = (z0 + z1) // 2
            z0 = max(0, center_z - min_box_size // 2)
            z1 = min(Z, z0 + min_box_size)
        
        roi_boxes.append((x0, x1, y0, y1, z0, z1))
    
    # Sort by number of ROI voxels in each box (descending)
    if roi_boxes:
        box_roi_counts = []
        for (x0, x1, y0, y1, z0, z1) in roi_boxes:
            # Count actual ROI voxels in this box
            roi_count = np.sum(roi_mask[x0:x1, y0:y1, z0:z1])
            box_roi_counts.append(roi_count)
        
        # Sort by ROI voxel count and take top boxes
        sorted_indices = np.argsort(box_roi_counts)[::-1]
        roi_boxes = [roi_boxes[i] for i in sorted_indices[:max_boxes]]
    
    return roi_boxes

def auto_select_roi_boxes_from_error(
    err_abs: np.ndarray,
    roi_percentage: float = 0.1,
    min_box_size: int = 8,
    max_boxes: int = 100,
    return_mask: bool = False,
) -> list:
    """
    选择ROI boxes，使用精确的ROI mask确保精确的百分比。
    先创建精确的ROI mask（精确到roi_percentage），然后从mask创建boxes。
    
    Args:
        err_abs: Absolute error array of shape (X, Y, Z), dtype float32
        roi_percentage: Percentage of voxels to select as ROI (default: 0.1 for 10%)
        min_box_size: Minimum size for each box (default: 8)
        max_boxes: Maximum number of boxes to create (default: 100)
        return_mask: If True, return both boxes and mask as tuple (boxes, mask)
    
    Returns:
        list or tuple: ROI boxes list, or (roi_boxes, roi_mask) if return_mask=True
    """
    # Step 1: Create precise ROI mask
    roi_mask = create_precise_roi_mask_from_error(err_abs, roi_percentage)
    
    # Step 2: Create boxes from mask
    roi_boxes = create_roi_boxes_from_mask(roi_mask, min_box_size, max_boxes)
    
    if return_mask:
        return roi_boxes, roi_mask
    else:
        return roi_boxes


def auto_select_roi_boxes_from_error_precise(
    err_abs: np.ndarray,
    roi_percentage: float = 0.1,
    min_box_size: int = 8,
    max_boxes: int = 100,
    use_precise_mask: bool = True,
) -> tuple:
    """
    改进版本：先创建精确的ROI mask，然后从mask创建boxes。
    可以选择返回mask或boxes，或两者都返回。
    
    Args:
        err_abs: Absolute error array of shape (X, Y, Z), dtype float32
        roi_percentage: Percentage of voxels to select as ROI (default: 0.1 for 10%)
        min_box_size: Minimum size for each box (default: 8)
        max_boxes: Maximum number of boxes to create (default: 100)
        use_precise_mask: If True, boxes only contain ROI voxels from mask
    
    Returns:
        tuple: (roi_boxes, roi_mask) if use_precise_mask=True, else (roi_boxes, None)
    """
    # Step 1: Create precise ROI mask
    roi_mask = create_precise_roi_mask_from_error(err_abs, roi_percentage)
    
    # Step 2: Create boxes from mask
    roi_boxes = create_roi_boxes_from_mask(roi_mask, min_box_size, max_boxes)
    
    if use_precise_mask:
        return roi_boxes, roi_mask
    else:
        return roi_boxes, None


def normalize_roi_to_3d_boxes(roi_specs, volume_shape):
    """
    Normalize ROI specifications to 3D boxes.
    
    roi_specs supports:
    1) 3D box: (x0,x1,y0,y1,z0,z1)
    2) 2D ROI: ("z", k, x0,x1,y0,y1) -> z in [k,k+1)
       also supports ("x", k, y0,y1,z0,z1) or ("y", k, x0,x1,z0,z1)
    
    Args:
        roi_specs: List of ROI specifications
        volume_shape: (X, Y, Z) volume shape
    
    Returns:
        list: List of valid clipped 3D boxes (x0,x1,y0,y1,z0,z1)
    """
    if roi_specs is None:
        return []

    X, Y, Z = volume_shape
    boxes = []

    for r in roi_specs:
        # 3D box
        if isinstance(r, (tuple, list)) and len(r) == 6 and isinstance(r[0], (int, np.integer)):
            x0, x1, y0, y1, z0, z1 = r
            boxes.append((x0, x1, y0, y1, z0, z1))
            continue

        # 2D ROI specification
        if isinstance(r, (tuple, list)) and len(r) == 6 and isinstance(r[0], str):
            axis, k, u0, u1, v0, v1 = r
            if axis == "z":
                boxes.append((u0, u1, v0, v1, k, k + 1))
            elif axis == "y":
                boxes.append((u0, u1, k, k + 1, v0, v1))
            elif axis == "x":
                boxes.append((k, k + 1, u0, u1, v0, v1))
            else:
                raise ValueError(f"Unknown ROI axis: {axis}")
            continue

        raise ValueError(f"Unsupported ROI spec: {r}")

    # Clip to bounds, drop invalid
    out = []
    for (x0, x1, y0, y1, z0, z1) in boxes:
        x0 = max(0, min(X, int(x0))); x1 = max(0, min(X, int(x1)))
        y0 = max(0, min(Y, int(y0))); y1 = max(0, min(Y, int(y1)))
        z0 = max(0, min(Z, int(z0))); z1 = max(0, min(Z, int(z1)))
        if (x1 > x0) and (y1 > y0) and (z1 > z0):
            out.append((x0, x1, y0, y1, z0, z1))

    return out


def create_roi_mask(volume_shape, roi_boxes_3d):
    """
    Create ROI mask: ROI region is 1, non-ROI region is 0.
    
    Args:
        volume_shape: (X, Y, Z) original data shape
        roi_boxes_3d: list of (x0, x1, y0, y1, z0, z1) tuples
    
    Returns:
        numpy.ndarray: Mask array of shape volume_shape, dtype=bool
    """
    mask = np.zeros(volume_shape, dtype=bool)
    for (x0, x1, y0, y1, z0, z1) in roi_boxes_3d:
        mask[x0:x1, y0:y1, z0:z1] = True
    return mask


# =============================================================================
# Post-Processing
# =============================================================================

def error_bounded_post_process(x_enhanced, x_prime, absolute_error_bound,
                               relative_error_bound=0.0, verbose=False, a=0.5):
    """
    Error-bounded post-processing using ML-predicted values over the full volume.
    
    This function applies post-processing to all points (not just block boundaries)
    while ensuring the error bound is not violated. The method uses ML-predicted
    values from x_enhanced (x_prime + predicted_residuals) and clamps them around
    x_prime with the effective error bound.
    
    Args:
        x_enhanced: Enhanced reconstruction (x_prime + predicted_residuals)
        x_prime: Base SZ3 reconstruction
        absolute_error_bound: Absolute error bound
        relative_error_bound: Relative error bound
        verbose: Print progress
        a: Scaling factor for error bound
    
    Returns:
        numpy.ndarray: Post-processed reconstruction
    """
    if relative_error_bound > 0:
        data_range = np.max(x_prime) - np.min(x_prime)
        effective_bound = relative_error_bound * data_range
    else:
        effective_bound = absolute_error_bound

    d = x_enhanced
    base = x_prime 
    upper = base + a * effective_bound
    lower = base - a * effective_bound
    d_prime = np.maximum(np.minimum(d, upper), lower)

    if verbose:
        max_delta = np.max(np.abs(d_prime - d))
        print(f"  Post-processing applied (full volume):")
        print(f"    a: {a}")
        print(f"    Effective bound: {effective_bound:.3e}")
        print(f"    Max delta: {max_delta:.3e}")

    return d_prime


# =============================================================================
# Model Creation
# =============================================================================

def create_model_for_decompress(model_type, metadata, spatial_dims, device):
    """
    Create model instance for decompression.
    
    Args:
        model_type: Model type string
        metadata: Metadata dictionary
        spatial_dims: Spatial dimensions (2 or 3)
        device: PyTorch device
    
    Returns:
        torch.nn.Module: Model instance or None if unsupported
    """
    if model_type == 'tiny_residual_predictor':
        metadata_channels = metadata['model_channels']
        metadata_rb = metadata.get('num_res_blocks', 2)
        return TinyResidualPredictor(
            channels=metadata_channels,
            spatial_dims=spatial_dims,
            num_res_blocks=metadata_rb
        ).to(device)
    elif model_type == 'tiny_frequency_residual_predictor_1_input':
        metadata_channels = metadata.get('model_channels', 2)
        return TinyFrequencyResidualPredictor_1_input(
            channels=metadata_channels,
            spatial_dims=spatial_dims,
            num_res_blocks=metadata.get('num_res_blocks', 2)
        ).to(device)
    elif model_type == 'tiny_frequency_residual_predictor_4_inputs':
        metadata_channels = metadata.get('model_channels', 2)
        return TinyFrequencyResidualPredictor_4_inputs(
            channels=metadata_channels,
            spatial_dims=spatial_dims,
            num_res_blocks=metadata.get('num_res_blocks', 2)
        ).to(device)
    elif model_type == 'tiny_frequency_residual_predictor_7_inputs':
        # Note: This model may not be imported, handle gracefully
        try:
            from Model import TinyFrequencyResidualPredictor_7_inputs
            metadata_channels = metadata.get('model_channels', 2)
            return TinyFrequencyResidualPredictor_7_inputs(
                channels=metadata_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 2)
            ).to(device)
        except ImportError:
            print(f"Model {model_type} not available")
            return None
    elif model_type == 'tiny_frequency_residual_predictor_with_energy':
        metadata_channels = metadata.get('model_channels', 2)
        return TinyFrequencyResidualPredictorWithEnergy(
            channels=metadata_channels,
            spatial_dims=spatial_dims,
            num_res_blocks=metadata.get('num_res_blocks', 2)
        ).to(device)
    elif model_type == 'tiny_frequency_residual_predictor_7_attn_roi':
        metadata_channels = metadata.get('model_channels', 2)
        # Support dual model with different prefixes (bg_ or roi_)
        # Check if this is for BG or ROI model by checking metadata keys
        prefix = ''
        if 'bg_low_cutoff' in metadata and 'roi_low_cutoff' in metadata:
            # Determine prefix based on model_type_bg/roi comparison
            model_type_bg = metadata.get('model_type_bg', '')
            model_type_roi = metadata.get('model_type_roi', '')
            # This will be called separately for BG and ROI, so we need to check which one
            # For now, use default keys and let caller specify via a parameter
            pass
        
        # Try to get model-specific parameters (for dual models)
        low_cutoff = metadata.get('bg_low_cutoff') or metadata.get('roi_low_cutoff') or metadata.get('low_cutoff', 0.15)
        mid_cutoff = metadata.get('bg_mid_cutoff') or metadata.get('roi_mid_cutoff') or metadata.get('mid_cutoff', 0.40)
        use_phase_sincos = metadata.get('bg_use_phase_sincos') if 'bg_use_phase_sincos' in metadata else (metadata.get('roi_use_phase_sincos') if 'roi_use_phase_sincos' in metadata else metadata.get('use_phase_sincos', True))
        gn_groups = metadata.get('bg_gn_groups') or metadata.get('roi_gn_groups') or metadata.get('gn_groups', 4)
        
        return TinyFrequencyResidualPredictor7_AttnROI(
            channels=metadata_channels,
            spatial_dims=spatial_dims,
            num_res_blocks=metadata.get('num_res_blocks', 2),
            low_cutoff=low_cutoff,
            mid_cutoff=mid_cutoff,
            use_phase_sincos=use_phase_sincos,
            return_roi_mask=False,
            gn_groups=gn_groups
        ).to(device)
    else:
        print(f"Model {model_type} not supported")
        return None


# =============================================================================
# Verification
# =============================================================================

def verify_reconstruction(original, reconstructed, eb_mode, absolute_error_bound,
                          relative_error_bound, pwr_error_bound=0, verbose=True,
                          compute_spectral_fn=None):
    """
    Verify reconstruction quality and error bound compliance.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
        eb_mode: Error bound mode
        absolute_error_bound: Absolute error bound
        relative_error_bound: Relative error bound
        pwr_error_bound: Power error bound
        verbose: Print progress
        compute_spectral_fn: Function to compute spectral metrics (optional)
    
    Returns:
        dict: Metrics dictionary
    """
    # Standard metrics
    error = np.abs(original - reconstructed)
    max_error = np.max(error)
    mean_error = np.mean(error)
    data_range = np.max(original) - np.min(original)
    mse = np.mean((original - reconstructed) ** 2)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
    nrmse = np.sqrt(mse) / data_range

    # Spectral metrics
    if compute_spectral_fn:
        try:
            log_mag_mse, phase_mse = compute_spectral_fn(original, reconstructed)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute spectral metrics: {e}")
            log_mag_mse, phase_mse = np.nan, np.nan
    else:
        log_mag_mse, phase_mse = np.nan, np.nan
    
    # FFT PSNR (Legacy)
    try:
        fft_original = np.fft.fftn(original.astype(np.float32))
        fft_reconstructed = np.fft.fftn(reconstructed.astype(np.float32))
        mag_original = np.abs(fft_original)
        del fft_original
        mag_reconstructed = np.abs(fft_reconstructed)
        del fft_reconstructed
        fft_mag_mse = np.mean((mag_original - mag_reconstructed) ** 2)
        del mag_original, mag_reconstructed
        fft_range = np.max(mag_original) - np.min(mag_original) if 'mag_original' in locals() else 1.0
        fft_psnr = 20 * np.log10(fft_range) - 10 * np.log10(fft_mag_mse) if fft_mag_mse > 0 else float('inf')
    except MemoryError:
        if verbose:
            print("Warning: Could not compute FFT PSNR due to memory constraints")
        fft_psnr = np.nan
    except Exception as e:
        if verbose:
            print(f"Warning: Error computing FFT PSNR: {e}")
        fft_psnr = np.nan
    
    # Error bound compliance
    within_bound = max_error <= absolute_error_bound
    violation_count = np.sum(error > absolute_error_bound)
    violation_ratio = violation_count / error.size * 100
    
    metrics = {
        'max_error': max_error,
        'mean_error': mean_error,
        'psnr': psnr,
        'nrmse': nrmse,
        'log_mag_mse': log_mag_mse,
        'phase_mse': phase_mse,
        'fft_psnr': fft_psnr,
        'within_bound': within_bound,
        'violation_count': int(violation_count),
        'violation_ratio': violation_ratio,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Reconstruction Quality:")
        print(f"  Max error: {max_error:.3e} (bound: {absolute_error_bound:.3e})")
        print(f"  Mean error: {mean_error:.3e}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  NRMSE: {nrmse:.6e}")
        if not np.isnan(log_mag_mse):
            print(f"  Log-Mag MSE: {log_mag_mse:.6e}")
        if not np.isnan(phase_mse):
            print(f"  Phase MSE: {phase_mse:.6e}")
        if not np.isnan(fft_psnr):
            print(f"  FFT PSNR: {fft_psnr:.2f} dB")
        print(f"  Within bound: {'✓ YES' if within_bound else '✗ NO'}")
        if not within_bound:
            print(f"  Violations: {violation_count} ({violation_ratio:.4f}%)")
        print(f"{'='*70}\n")
    
    return metrics


def verify_reconstruction_per_slice(original, reconstructed, eb_mode, absolute_error_bound,
                                    relative_error_bound, pwr_error_bound=0,
                                    slice_axis=2, verbose=True):
    """
    Verify reconstruction quality per slice.
    
    Args:
        original: Original 3D data
        reconstructed: Reconstructed 3D data
        eb_mode: Error bound mode
        absolute_error_bound: Absolute error bound
        relative_error_bound: Relative error bound
        pwr_error_bound: Power error bound
        slice_axis: Axis to slice along (0=X, 1=Y, 2=Z)
        verbose: Print detailed per-slice metrics
    
    Returns:
        dict: Dictionary with per-slice metrics and statistics
    """
    n_slices = original.shape[slice_axis]
    
    slice_metrics = {
        'max_error': [],
        'mean_error': [],
        'psnr': [],
        'nrmse': [],
        'fft_psnr': [],
        'within_bound': [],
        'violation_ratio': [],
    }
    
    axis_names = ['X', 'Y', 'Z']
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Per-Slice Reconstruction Quality (Axis: {axis_names[slice_axis]}, {n_slices} slices)")
        print(f"{'='*70}")
    
    slices_processed = 0
    for i in range(n_slices):
        # Extract slice
        if slice_axis == 0:
            orig_slice = original[i, :, :]
            recon_slice = reconstructed[i, :, :]
        elif slice_axis == 1:
            orig_slice = original[:, i, :]
            recon_slice = reconstructed[:, i, :]
        else:  # slice_axis == 2
            orig_slice = original[:, :, i]
            recon_slice = reconstructed[:, :, i]
        
        # Compute metrics for this slice
        error = np.abs(orig_slice - recon_slice)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        # PSNR
        data_range = np.max(orig_slice) - np.min(orig_slice)
        mse = np.mean((orig_slice - recon_slice) ** 2)
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
        
        # NRMSE
        nrmse = np.sqrt(mse) / (data_range + 1e-10)
        
        # FFT metrics (2D)
        fft_orig = np.fft.fft2(orig_slice)
        fft_recon = np.fft.fft2(recon_slice)
        mag_orig = np.abs(fft_orig)
        mag_recon = np.abs(fft_recon)
        fft_mag_mse = np.mean((mag_orig - mag_recon) ** 2)
        fft_range = np.max(mag_orig) - np.min(mag_orig)
        fft_psnr = 20 * np.log10(fft_range) - 10 * np.log10(fft_mag_mse) if fft_mag_mse > 0 else float('inf')
        
        # Error bound compliance
        within_bound = max_error <= absolute_error_bound
        violation_count = np.sum(error > absolute_error_bound)
        violation_ratio = violation_count / error.size * 100
        
        # Store metrics
        slice_metrics['max_error'].append(float(max_error))
        slice_metrics['mean_error'].append(float(mean_error))
        slice_metrics['psnr'].append(float(psnr))
        slice_metrics['nrmse'].append(float(nrmse))
        slice_metrics['fft_psnr'].append(float(fft_psnr))
        slice_metrics['within_bound'].append(bool(within_bound))
        slice_metrics['violation_ratio'].append(float(violation_ratio))
        
        slices_processed += 1
    
    # Verify all slices were processed
    num_metrics = len(slice_metrics['max_error'])
    
    if verbose:
        if slices_processed != n_slices or num_metrics != n_slices:
            print(f"  [WARNING] Mismatch! Expected {n_slices} slices, got {slices_processed} processed and {num_metrics} metrics")
    
    # Compute statistics across slices
    stats = {}
    for key in ['max_error', 'mean_error', 'psnr', 'nrmse', 'fft_psnr', 'violation_ratio']:
        values = np.array(slice_metrics[key])
        stats[key] = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
        }
    
    stats['slices_within_bound'] = int(sum(slice_metrics['within_bound']))
    stats['total_slices'] = int(n_slices)
    stats['bound_compliance_ratio'] = float(stats['slices_within_bound'] / n_slices * 100)
    
    if verbose:
        print(f"\nStatistics across {n_slices} slices:")
        print(f"  PSNR:       min={stats['psnr']['min']:.2f}, max={stats['psnr']['max']:.2f}, "
              f"mean={stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB, median={stats['psnr']['median']:.2f} dB")
        print(f"  Max Error:  min={stats['max_error']['min']:.3e}, max={stats['max_error']['max']:.3e}, "
              f"mean={stats['max_error']['mean']:.3e}, median={stats['max_error']['median']:.3e}")
        print(f"  Mean Error: min={stats['mean_error']['min']:.3e}, max={stats['mean_error']['max']:.3e}, "
              f"mean={stats['mean_error']['mean']:.3e}, median={stats['mean_error']['median']:.3e}")
        print(f"  FFT PSNR:   min={stats['fft_psnr']['min']:.2f}, max={stats['fft_psnr']['max']:.2f}, "
              f"mean={stats['fft_psnr']['mean']:.2f} ± {stats['fft_psnr']['std']:.2f} dB, median={stats['fft_psnr']['median']:.2f} dB")
        print(f"  Slices within bound: {stats['slices_within_bound']}/{n_slices} "
              f"({stats['bound_compliance_ratio']:.1f}%)")
        print(f"{'='*70}\n")
    
    return {
        'per_slice': slice_metrics,
        'statistics': stats,
        'slice_axis': slice_axis,
        'n_slices': n_slices,
    }

