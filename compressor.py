"""
NeurLZ Compressor Implementation.

Reference: https://arxiv.org/abs/2409.05785

Key differences from typical neural compression:
1. SZ3/ZFP is PRIMARY compressor (not neural)
2. Tiny DNN (~3k params) trained ONLINE during compression
3. DNN predicts residuals from SZ3-decompressed data
4. Storage: {SZ3_bytes, DNN_weights, outliers}
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import sys
import os
import json
import pickle
import time
import importlib.util

# =============================================================================
# Third-Party Imports
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Local Imports - Models (for decompress only)
# =============================================================================
# Add neural_compression to path for Model imports
sys.path.insert(0, '/Users/923714256/Data_compression/neural_compression')

# =============================================================================
# Local Imports - Compression Functions
# =============================================================================
try:
    from .compression_function import (
        compute_gn_groups,
        compute_spectral_metrics,
        compute_ssim_3d,
        normalize_roi_to_3d_boxes,
        create_roi_mask,
        auto_select_roi_boxes_from_error,
        error_bounded_post_process,
        create_model_for_decompress,
        verify_reconstruction,
        verify_reconstruction_per_slice,
    )
    from .train import (
        train_dual_models,
        predict_residuals_dual,
    )
except ImportError:
    from compression_function import (
        compute_gn_groups,
        compute_spectral_metrics,
        compute_ssim_3d,
        normalize_roi_to_3d_boxes,
        create_roi_mask,
        auto_select_roi_boxes_from_error,
        error_bounded_post_process,
        create_model_for_decompress,
        verify_reconstruction,
        verify_reconstruction_per_slice,
    )
    from train import (
        train_dual_models,
        predict_residuals_dual,
    )

# =============================================================================
# External Library - SZ3 Compressor
# =============================================================================
sys.path.append('/Users/923714256/Data_compression/SZ3/tools/pysz')
from pysz import SZ

class NeurLZCompressor:
    """
    NeurLZ: SZ3/ZFP + Online-trained tiny DNN.
    
    Correct implementation following the paper.
    """
    
    def __init__(self, sz_lib_path="/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so",
                 device=None):
        self.sz = SZ(sz_lib_path)
        # Auto-detect device: use CUDA if available, otherwise CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
                print("Use cpu when no gpu detected")
        else:
            # If device is specified, validate it
            if device.startswith('cuda') and not torch.cuda.is_available():
                print(f"Warning: CUDA not available, using CPU instead")
                self.device = 'cpu'
            else:
                self.device = device

    ### Region of Interest (ROI) functions
    @staticmethod
    def normalize_roi_to_3d_boxes(roi_specs, volume_shape):
        """Wrapper for normalize_roi_to_3d_boxes from compression_function."""
        return normalize_roi_to_3d_boxes(roi_specs, volume_shape)


    def _create_roi_mask(self, volume_shape, roi_boxes_3d):
        """Wrapper for create_roi_mask from compression_function."""
        return create_roi_mask(volume_shape, roi_boxes_3d)

    def compress(self, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound, output_path=None,
                online_epochs=50, learning_rate=1e-3, model='tiny_residual_predictor', num_res_blocks=1,
                model_channels=4, verbose=True, 
                spatial_dims=3, slice_order='zxy',
                val_split=0.1, track_losses=True,
                Patch_size=256,Batch_size=512,
                save_components=False, components_dir='./compressed_components', filename='data',
                roi_boxes=None,           # list of 3D boxes or 2D ROI specs (for Scheme B dual models)
                use_dual_models=False,    # Enable dual model (Scheme B): separate BG and ROI models
                auto_select_roi=False,    # Auto-select ROI from error
                roi_percentage=0.05,      # Percentage of dataset as ROI (default: 0.05 for 5%, 95% as BG)
                model_bg=None,           # Model type for BG (if None, uses model)
                model_roi=None):         # Model type for ROI (if None, uses model)
        """
        NeurLZ compression pipeline.
        
        Args:
            data: Input 3D volume (numpy array)
            eb_mode: Error bound mode (0: ABS, 1: REL, 2: ABS_AND_REL, 3: ABS_OR_REL, 4: PSNR, 5: NORM, 10: PW_REL)
            absolute_error_bound: Target absolute error bound (ε)
            relative_error_bound: Relative error bound (δ)
            pwr_error_bound: Power error bound
            output_path: Path to save compressed file
            online_epochs: Number of epochs for online DNN training
            learning_rate: Learning rate for online training
            num_res_blocks: Number of residual blocks for the model
            model_channels: Base channels for tiny DNN (4 → ~3k params)
            model: Model to use (tiny_residual_predictor or tiny_frequency_residual_predictor)
            verbose: Print progress
            spatial_dims: 2 for 2D sliced processing, 3 for full 3D volume
            slice_order: 'xyz', 'zxy', or 'yxz' (only used when spatial_dims=2)
            val_split: Fraction of data to use for validation (0.0-1.0)
            track_losses: Whether to track training and validation losses
            Patch_size: Patch size for local decompression and training
            Batch_size: Batch size for training
            save_components: Whether to save components
            components_dir: Directory to save components
            filename: Filename to save components

        Note: NeurLZ compression pipeline with optional dual model support (Scheme B).
        
        If use_dual_models=True and roi_boxes is provided:
        - BG model: trained on non-ROI regions
        - ROI model: trained on ROI regions
        - Final residual: R_hat = R_hat_bg * (1 - M_roi) + R_hat_roi * M_roi
        Returns:
            compressed_package: Dict with {sz_bytes, model_weights, outliers, metadata}
            stats: Compression statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"NeurLZ Compression (spatial_dims={spatial_dims})")
            if spatial_dims == 2:
                print(f"  Processing as 2D slices (order={slice_order})")
                print(f"  Note: Using 2D model for training, but ROI boxes are defined in 3D space.")
                print(f"       ROI boxes will be mapped to 2D slices during training/prediction.")
            print(f"{'='*70}")
            print(f"Data: {data.shape}, range=[{np.min(data):.3e}, {np.max(data):.3e}]")
            print(f"Error bound: {absolute_error_bound}")
            if roi_boxes:
                print(f"ROI boxes (3D coordinates): {roi_boxes}")
            if use_dual_models:
                print(f"Dual models (Scheme B): Enabled")
        
        original_size = data.nbytes
        compress_start = time.time()
        
        # ================================================================
        # STEP 1: SZ3 Compression (PRIMARY)
        # ================================================================
        if verbose:
            print(f"\nStep 1: SZ3 compression (primary)...")
        
        data_f32 = data.astype(np.float32)
        sz3_compressed, sz3_ratio = self.sz.compress(data_f32, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound)
        sz3_size = len(sz3_compressed)
        
        if verbose:
            print(f"  SZ3 size: {sz3_size / 1024:.2f} KB (ratio: {sz3_ratio:.2f}x)")
        
        # ================================================================
        # STEP 2: Local Decompression (to get X')
        # ================================================================
        if verbose:
            print(f"\nStep 2: Decompressing SZ3 locally (for training)...")
        
        x_prime = self.sz.decompress(sz3_compressed, data.shape, np.float32)
        
        # Base quality
        base_error = np.abs(data - x_prime)
        base_max_error = np.max(base_error)
        base_mean_error = np.mean(base_error)
        
        if verbose:
            print(f"  SZ3 base quality:")
            print(f"    Max error: {base_max_error:.3e}")
            print(f"    Mean error: {base_mean_error:.3e}")
        
        # ================================================================
        # STEP 3: Compute True Residuals (R = X - X')
        # ================================================================
        if verbose:
            print(f"\nStep 3: Computing true residuals (R = original - SZ3_decompressed)...")
        
        true_residuals = data - x_prime
        
        if verbose:
            print(f"  Residual stats:")
            print(f"    Mean: {np.mean(true_residuals):.3e}")
            print(f"    Std: {np.std(true_residuals):.3e}")
            print(f"    Range: [{np.min(true_residuals):.3e}, {np.max(true_residuals):.3e}]")
        # ================================================================
        # AUTO-SELECT ROI if enabled and not manually provided
        # ================================================================
        if auto_select_roi and roi_boxes is None:
            if verbose:
                print(f"\nStep 3.5: Auto-selecting ROI ({roi_percentage*100:.1f}% of dataset) from error distribution...")
            error_abs = np.abs(true_residuals)
            # Get both boxes and precise mask
            auto_roi_boxes, roi_mask_precise = auto_select_roi_boxes_from_error(
                error_abs, 
                roi_percentage=roi_percentage,
                return_mask=True
            )
            roi_boxes = auto_roi_boxes
            if verbose:
                total_voxels = error_abs.size
                # Calculate precise ROI voxels from mask
                roi_voxels_precise = np.sum(roi_mask_precise)
                # Calculate ROI voxels covered by boxes (may be larger than precise)
                roi_voxels_from_boxes = sum((x1-x0)*(y1-y0)*(z1-z0) for (x0, x1, y0, y1, z0, z1) in roi_boxes)
                bg_voxels = total_voxels - roi_voxels_precise
                print(f"  Auto-selected {len(roi_boxes)} ROI box(es):")
                for i, box in enumerate(roi_boxes):
                    box_size = (box[1]-box[0]) * (box[3]-box[2]) * (box[5]-box[4])
                    print(f"    ROI[{i}]: {box} (size: {box_size:,} voxels)")
                print(f"  Precise ROI mask: {roi_voxels_precise:,} voxels ({roi_voxels_precise/total_voxels*100:.2f}%)")
                print(f"  ROI boxes cover: {roi_voxels_from_boxes:,} voxels ({roi_voxels_from_boxes/total_voxels*100:.2f}%)")
                print(f"  BG:  {bg_voxels:,} voxels ({bg_voxels/total_voxels*100:.2f}%)")            


        # ================================================================
        # STEP 4: Train Tiny DNN Online (DUAL MODELS ONLY)
        # ================================================================
        roi_boxes_3d = normalize_roi_to_3d_boxes(roi_boxes, data.shape)

        # Dual-only: refuse silent fallback to single model
        if not use_dual_models:
            raise ValueError("This compressor is Dual-Model only. Please set use_dual_models=True.")
        if not roi_boxes_3d:
            raise ValueError(
                "use_dual_models=True 但未获得任何 ROI。请提供 roi_boxes，或启用 auto_select_roi 以自动选择 ROI。"
            )
        use_dual = True

        if verbose:
            print(f"\nStep 4: Training DUAL models (Scheme B)...")
            print(f"  BG model: non-ROI regions")
            print(f"  ROI model: ROI regions")

        model_type = model

        # Requested behavior: BG model uses the same architecture as ROI model by default
        model_type_roi = model_roi if model_roi is not None else model_type
        model_type_bg = model_bg if model_bg is not None else model_type_roi
        
        # ============================================================
        # Handle 2D vs 3D Processing
        # ============================================================
        if spatial_dims == 2:
            # 2D mode: Transpose data to process as slices
            if slice_order == 'zxy':
                # (X, Y, Z) → (Z, X, Y): Z slices of (X, Y)
                x_prime_for_training = x_prime.transpose(2, 0, 1)
                true_residuals_for_training = true_residuals.transpose(2, 0, 1)
            elif slice_order == 'yxz':
                # (X, Y, Z) → (Y, X, Z): Y slices of (X, Z)
                x_prime_for_training = x_prime.transpose(1, 0, 2)
                true_residuals_for_training = true_residuals.transpose(1, 0, 2)
            else:  # 'xyz'
                # Keep as is: X slices of (Y, Z)
                x_prime_for_training = x_prime
                true_residuals_for_training = true_residuals
            
            if verbose:
                print(f"  2D mode: Reshaped to {x_prime_for_training.shape} "
                    f"({x_prime_for_training.shape[0]} slices of {x_prime_for_training.shape[1]}×{x_prime_for_training.shape[2]})")
                if roi_boxes_3d:
                    print(f"  ROI boxes (3D) will be mapped to 2D slices:")
                    if slice_order == 'zxy':
                        print(f"    - ROI box (x0, x1, y0, y1, z0, z1) maps to slices [z0:z1] with region [x0:x1, y0:y1] in each slice")
                    elif slice_order == 'yxz':
                        print(f"    - ROI box (x0, x1, y0, y1, z0, z1) maps to slices [y0:y1] with region [x0:x1, z0:z1] in each slice")
                    else:  # 'xyz'
                        print(f"    - ROI box (x0, x1, y0, y1, z0, z1) maps to slices [x0:x1] with region [y0:y1, z0:z1] in each slice")
        else:
            # 3D mode: Keep original shape
            x_prime_for_training = x_prime
            true_residuals_for_training = true_residuals

        # Map precise ROI mask (in original XYZ) into training space (if available)
        roi_mask_for_training = None
        if auto_select_roi and 'roi_mask_precise' in locals() and roi_mask_precise is not None:
            if spatial_dims == 2:
                if slice_order == 'zxy':
                    roi_mask_for_training = roi_mask_precise.transpose(2, 0, 1)  # (X,Y,Z)->(Z,X,Y)
                elif slice_order == 'yxz':
                    roi_mask_for_training = roi_mask_precise.transpose(1, 0, 2)  # (X,Y,Z)->(Y,X,Z)
                else:  # 'xyz'
                    roi_mask_for_training = roi_mask_precise
            else:
                roi_mask_for_training = roi_mask_precise
        
        # Normalize residuals for better training
        residual_mean = float(np.mean(true_residuals_for_training))
        residual_std = float(np.std(true_residuals_for_training)) + 1e-8
        x_prime_norm = (x_prime_for_training - np.mean(x_prime_for_training)) / (np.std(x_prime_for_training) + 1e-8)
        residuals_norm = (true_residuals_for_training - residual_mean) / residual_std
        input_mean = float(np.mean(x_prime_for_training))
        input_std = float(np.std(x_prime_for_training)) + 1e-8

        # Train dual models
        result = train_dual_models(
            x_prime_norm, residuals_norm, roi_boxes_3d, model_type, model_channels,
            spatial_dims, num_res_blocks, slice_order, online_epochs, learning_rate,
            Patch_size, Batch_size, val_split, track_losses, verbose,
            model_type_bg=model_type_bg, model_type_roi=model_type_roi,
            roi_mask_training=roi_mask_for_training,
        )
        if result[0] is None:
            return None, None
        (model_bg, model_roi, train_losses_bg, train_losses_roi, val_losses_bg, val_losses_roi,
         training_time, num_params_bg, num_params_roi, batch_size_2d, batch_size_3d, primary_device) = result

        # ================================================================
        # STEP 5: Generate Enhanced Reconstruction (X_enh = X' + R_hat)
        # ================================================================
        if verbose:
            print(f"\nStep 5: Generating enhanced reconstruction...")

        pred_residuals_np = predict_residuals_dual(
            model_bg, model_roi, x_prime_norm, roi_boxes_3d,
            residual_mean, residual_std, spatial_dims, slice_order,
            batch_size_2d, batch_size_3d, primary_device,
            roi_mask_pred=roi_mask_for_training,
        )
        
        # Use x_prime (not normalized) for final reconstruction
        x_enhanced = x_prime + pred_residuals_np
        
        # Compute enhanced quality
        enhanced_error = np.abs(data - x_enhanced)
        enhanced_max_error = np.max(enhanced_error)
        enhanced_mean_error = np.mean(enhanced_error)
        
        if verbose:
            print(f"  Enhanced quality:")
            print(f"    Max error: {enhanced_max_error:.3e}")
            print(f"    Mean error: {enhanced_mean_error:.3e}")
            print(f"    Improvement: {(base_mean_error - enhanced_mean_error):.3e}")

        # ================================================================
        # STEP 6: Error Bound Enforcement
        # ================================================================
        if verbose:
            print(f"\nStep 6: Computing final quality metrics...")
            print(f"  Enhanced max error: {enhanced_max_error:.3e}")
        
        # ================================================================
        # STEP 7: Package for Storage (Following Paper's Archive Layout)
        # ================================================================
        if verbose:
            print(f"\nStep 7: Packaging compressed data...")
        
        # Extract model weights (dual-only)
        if isinstance(model_bg, nn.DataParallel):
            model_weights_bg = {k: v.cpu().numpy() for k, v in model_bg.module.state_dict().items()}
        else:
            model_weights_bg = {k: v.cpu().numpy() for k, v in model_bg.state_dict().items()}

        if isinstance(model_roi, nn.DataParallel):
            model_weights_roi = {k: v.cpu().numpy() for k, v in model_roi.module.state_dict().items()}
        else:
            model_weights_roi = {k: v.cpu().numpy() for k, v in model_roi.state_dict().items()}

        model_weights = {
            'bg': model_weights_bg,
            'roi': model_weights_roi
        }
        num_params = num_params_bg + num_params_roi
        
        metadata_dict = {
            'original_shape': data.shape,
            'original_dtype': str(data.dtype),
            'absolute_error_bound': absolute_error_bound,
            'relative_error_bound': relative_error_bound,
            'pwr_error_bound': pwr_error_bound,
            'model_channels': model_channels,
            'num_res_blocks': num_res_blocks,
            'model_params': num_params,
            'spatial_dims': spatial_dims,
            'slice_order': slice_order if spatial_dims == 2 else None, 
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'input_mean': input_mean,
            'input_std': input_std,
            'model_type': model_type,
            # inference settings to keep decompression consistent
            'batch_size_2d': int(batch_size_2d),
            'batch_size_3d': int(batch_size_3d),
            # ROI meta (for Scheme B dual models)
            'roi_boxes_3d': roi_boxes_3d if roi_boxes_3d else None,
            'use_dual_models': True,
        }
        
        # Dual-only: save both model types + model-specific metadata
        metadata_dict['model_type_bg'] = model_type_bg
        metadata_dict['model_type_roi'] = model_type_roi
        # Add model-specific metadata for BG model
        if model_type_bg == 'tiny_frequency_residual_predictor_7_attn_roi':
            metadata_dict['bg_low_cutoff'] = 0.15
            metadata_dict['bg_mid_cutoff'] = 0.40
            metadata_dict['bg_use_phase_sincos'] = True
            gn_groups_value = compute_gn_groups(model_channels, preferred=4)
            metadata_dict['bg_gn_groups'] = gn_groups_value
        # Add model-specific metadata for ROI model
        if model_type_roi == 'tiny_frequency_residual_predictor_7_attn_roi':
            metadata_dict['roi_low_cutoff'] = 0.15
            metadata_dict['roi_mid_cutoff'] = 0.40
            metadata_dict['roi_use_phase_sincos'] = True
            gn_groups_value = compute_gn_groups(model_channels, preferred=4)
            metadata_dict['roi_gn_groups'] = gn_groups_value
        
        compressed_package = {
            'backend': 'SZ3',
            'sz_bytes': sz3_compressed,
            'model_weights': model_weights,
            'metadata': metadata_dict,
        }
        
        # Calculate sizes
        sz_size = len(sz3_compressed)
        weights_size = sum(w.nbytes for w in model_weights_bg.values()) + sum(w.nbytes for w in model_weights_roi.values())
        total_size = sz_size + weights_size
        
        overall_ratio = original_size / total_size
        
        compress_time = time.time() - compress_start
        
        if verbose:
            print(f"\n{'='*70}")
            print("Compression Summary:")
            print(f"  Original:    {original_size / (1024**2):.2f} MB")
            print(f"  SZ3 bytes:   {sz_size / 1024:.2f} KB")
            print(f"  DNN weights: {weights_size / 1024:.2f} KB ({num_params:,} params)")
            print(f"  Dual models: BG + ROI (Scheme B)")
            print(f"  Total:       {total_size / (1024**2):.2f} MB")
            print(f"  Overall CR:  {overall_ratio:.2f}x")
            print(f"  SZ3-only CR: {sz3_ratio:.2f}x")
            print(f"  Time:        {compress_time:.2f}s (training {training_time:.2f}s)")
            print(f"{'='*70}\n")
        
        # Save if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(compressed_package, f, protocol=4)
            if verbose:
                print(f"Saved to: {output_path}\n")
            
            # Also save individual components to the same directory
            output_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            self.save_components(compressed_package, output_dir, base_name, verbose)
        
        # Package training losses (dual models have separate BG / ROI losses)
        if track_losses:
            train_losses = {
                'bg': train_losses_bg,
                'roi': train_losses_roi,
            }
            val_losses = {
                'bg': val_losses_bg,
                'roi': val_losses_roi,
            }
        else:
            train_losses = []
            val_losses = []

        stats = {
            'original_size_mb': original_size / (1024**2),
            'sz3_size_kb': sz_size / 1024,
            'weights_size_kb': weights_size / 1024,
            'total_size_mb': total_size / (1024**2),
            'sz3_ratio': float(sz3_ratio),
            'overall_ratio': float(overall_ratio),
            'base_max_error': base_max_error,
            'base_mean_error': base_mean_error,
            'enhanced_max_error': enhanced_max_error,
            'enhanced_mean_error': enhanced_mean_error,
            'compress_time': float(compress_time),
            'training_time': float(training_time),
            'model_params': int(num_params),
            'train_losses': train_losses if track_losses else [],
            'val_losses': val_losses if track_losses else [],
        }
        
        return compressed_package, stats

    def _error_bounded_post_process_residual(
        self,
        pred_residuals,
        x_prime,
        absolute_error_bound,
        relative_error_bound=0.0,
        verbose=False,
        a=1.0,
        data_range=None,
        mode="residual",
        sanitize_nonfinite=True,
    ):
        """
        Error-bounded post-processing applied to predicted residuals (delta from x_prime).

        Goal:
        x_post = x_prime + delta'
        delta' is bounded so that x_post stays within a safe window around x_prime:
            x_post in [x_prime - a*B, x_prime + a*B]
        equivalently:
            delta' in [-a*B, +a*B]

        Notes:
        - If REL mode is used, B = relative_error_bound * data_range.
        - data_range should ideally be saved from compression time (original range).
            If not provided, we fallback to range(x_prime).
        - This bounds |x_post - x_prime|, not strictly |x_post - original|.
        """

        # ---- 1) Determine effective bound B ----
        rel = float(relative_error_bound) if relative_error_bound is not None else 0.0
        absb = float(absolute_error_bound)

        if rel > 0.0:
            if data_range is None:
                # fallback: use x_prime range (decompress side availability)
                xr = float(np.max(x_prime) - np.min(x_prime))
            else:
                xr = float(data_range)
            # avoid degenerate range
            xr = max(xr, 1e-12)
            effective_bound = rel * xr
        else:
            effective_bound = absb

        # guard: negative/NaN bounds
        if not np.isfinite(effective_bound) or effective_bound < 0:
            if verbose:
                print(f"  WARNING: invalid effective_bound={effective_bound}, fallback to abs bound={absb}")
            effective_bound = max(absb, 0.0)

        residual_bound = float(a) * float(effective_bound)

        # ---- 2) Sanitize residuals (optional) ----
        post_in = pred_residuals
        if sanitize_nonfinite:
            # replace NaN/Inf with 0 so clipping is well-defined
            if not np.isfinite(post_in).all():
                post_in = np.where(np.isfinite(post_in), post_in, 0.0)

        # ---- 3) Apply postprocess ----
        mode = str(mode).lower().strip()
        if mode == "residual":
            # delta' = clip(delta, [-rb, +rb])
            post_residuals = np.clip(post_in, -residual_bound, residual_bound)
        elif mode == "value":
            # clamp final value, then convert back to residual form
            x_post = np.clip(x_prime + post_in, x_prime - residual_bound, x_prime + residual_bound)
            post_residuals = x_post - x_prime
        else:
            raise ValueError(f"mode must be 'residual' or 'value', got {mode}")

        # ---- 4) Verbose logging ----
        if verbose:
            max_before = float(np.max(np.abs(post_in))) if post_in.size else 0.0
            max_after = float(np.max(np.abs(post_residuals))) if post_residuals.size else 0.0
            clipped = int(np.sum(np.abs(post_in) > residual_bound)) if post_in.size else 0
            total = int(post_in.size)

            print(f"  Residual post-process:")
            print(f"    Mode:             {mode}")
            print(f"    Effective bound B:{effective_bound:.3e} (abs={absb:.3e}, rel={rel:.3e})")
            if rel > 0.0:
                used_range = (float(np.max(x_prime) - np.min(x_prime)) if data_range is None else float(data_range))
                print(f"    data_range used:  {used_range:.3e}")
            print(f"    a:                {float(a):.3f}")
            print(f"    Residual bound:   {residual_bound:.3e}")
            print(f"    Max |res| before: {max_before:.3e}")
            print(f"    Max |res| after:  {max_after:.3e}")
            if total > 0:
                print(f"    Clipped points:   {clipped}/{total} ({100.0*clipped/total:.2f}%)")

        return post_residuals

    def verify_residual_clipping(self, pred_residuals, postprocessed_residuals, 
                            x_prime, absolute_error_bound, relative_error_bound=0.0, 
                            a=1.0, verbose=True):
        """
        Verify that postprocessed residuals are correctly clipped within the bound.
        
        Args:
            pred_residuals: Original predicted residuals before post-processing
            postprocessed_residuals: Residuals after post-processing
            x_prime: SZ3 decompressed base reconstruction
            absolute_error_bound: Absolute error bound
            relative_error_bound: Relative error bound
            a: Scaling factor used in post-processing
            verbose: Print detailed verification results
        
        Returns:
            dict: Verification results with statistics
        """
        # Calculate effective bound (same logic as post-processing function)
        # IMPORTANT: Use the exact same logic as _error_bounded_post_process_residual
        rel = float(relative_error_bound) if relative_error_bound is not None else 0.0
        absb = float(absolute_error_bound)
        
        if rel > 0.0:
            data_range = np.max(x_prime) - np.min(x_prime)
            data_range = max(data_range, 1e-12)  # avoid degenerate range
            effective_bound = rel * data_range
        else:
            effective_bound = absb
        
        # guard: negative/NaN bounds
        if not np.isfinite(effective_bound) or effective_bound < 0:
            effective_bound = max(absb, 0.0)
        
        residual_bound = float(a) * float(effective_bound)
        
        # Check if all postprocessed residuals are within bounds
        abs_postprocessed = np.abs(postprocessed_residuals)
        violations = np.sum(abs_postprocessed > residual_bound)
        total_points = postprocessed_residuals.size
        
        # Calculate statistics
        max_abs_before = np.max(np.abs(pred_residuals))
        max_abs_after = np.max(abs_postprocessed)
        mean_abs_before = np.mean(np.abs(pred_residuals))
        mean_abs_after = np.mean(abs_postprocessed)
        
        # Check clipping behavior
        clipped_mask = np.abs(pred_residuals) > residual_bound
        clipped_count = np.sum(clipped_mask)
        
        # For clipped points, check if they were correctly clipped
        if clipped_count > 0:
            clipped_original = pred_residuals[clipped_mask]
            clipped_postprocessed = postprocessed_residuals[clipped_mask]
            clipped_abs_original = np.abs(clipped_original)
            clipped_abs_postprocessed = np.abs(clipped_postprocessed)
            
            # Check if clipped points are exactly at the boundary
            at_boundary = np.sum(np.abs(clipped_abs_postprocessed - residual_bound) < 1e-10)
        else:
            at_boundary = 0
        
        results = {
            'residual_bound': residual_bound,
            'effective_bound': effective_bound,
            'max_abs_before': max_abs_before,
            'max_abs_after': max_abs_after,
            'mean_abs_before': mean_abs_before,
            'mean_abs_after': mean_abs_after,
            'violations': violations,
            'violation_ratio': violations / total_points if total_points > 0 else 0.0,
            'clipped_count': clipped_count,
            'clipped_ratio': clipped_count / total_points if total_points > 0 else 0.0,
            'at_boundary': at_boundary,
            'all_within_bound': violations == 0
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Residual Clipping Verification")
            print(f"{'='*70}")
            print(f"Residual bound (a*eb): {residual_bound:.6e}")
            print(f"Effective bound (eb):   {effective_bound:.6e}")
            print(f"\nBefore post-processing:")
            print(f"  Max |residual|:       {max_abs_before:.6e}")
            print(f"  Mean |residual|:      {mean_abs_before:.6e}")
            print(f"\nAfter post-processing:")
            print(f"  Max |residual|:       {max_abs_after:.6e}")
            print(f"  Mean |residual|:      {mean_abs_after:.6e}")
            print(f"\nClipping statistics:")
            print(f"  Points clipped:       {clipped_count}/{total_points} ({100*results['clipped_ratio']:.2f}%)")
            print(f"  Points at boundary:   {at_boundary}/{clipped_count} ({100*at_boundary/clipped_count:.2f}% of clipped)" if clipped_count > 0 else "  Points at boundary:   0")
            print(f"\nVerification:")
            print(f"  Violations (> bound): {violations}/{total_points} ({100*results['violation_ratio']:.2f}%)")
            if violations == 0:
                print(f"  ✓ All residuals are within bound!")
            else:
                print(f"  ✗ WARNING: {violations} residuals exceed the bound!")
                max_violation = np.max(abs_postprocessed[abs_postprocessed > residual_bound])
                print(f"  Max violation: {max_violation:.6e} (bound: {residual_bound:.6e})")
            print(f"{'='*70}\n")
        
        return results
    
    def decompress(self, compressed_package, verbose=True, enable_post_process=False):
        """
        NeurLZ decompression pipeline.
        
        Args:
            compressed_package: Package from compress() or loaded from file
            verbose: Print progress
        
        Returns:
            reconstructed: Decompressed data (numpy array)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"NeurLZ Decompression")
            print(f"{'='*70}")
        
        decompress_start = time.time()
        metadata = compressed_package['metadata']
        absolute_error_bound = metadata['absolute_error_bound']
        
        # ================================================================
        # STEP 1: Decompress SZ3 (Base Reconstruction X')
        # ================================================================
        if verbose:
            print(f"\nStep 1: SZ3 decompression...")
        
        x_prime = self.sz.decompress(
            compressed_package['sz_bytes'],
            metadata['original_shape'],
            np.dtype(metadata['original_dtype'])
        )
        
        if verbose:
            print(f"  Base reconstruction: {x_prime.shape}")
        
        # ================================================================
        # STEP 2: Load DNN and Predict Residuals
        # ================================================================
        # Dual-only decompression: refuse silent fallback
        use_dual = metadata.get('use_dual_models', False)
        if not use_dual:
            raise ValueError(
                "This compressor is Dual-Model only, but metadata['use_dual_models'] is False. "
                "请使用 dual 模式生成的 compressed_package，或重新压缩开启 use_dual_models。"
            )
        
        if verbose:
            print(f"\nStep 2: Loading DUAL DNN weights and predicting residuals...")
            
        # Recreate model with same architecture
        spatial_dims = metadata.get('spatial_dims', 2)
        slice_order = metadata.get('slice_order', 'zxy')
        model_type = metadata.get('model_type', metadata.get('model', 'tiny_residual_predictor'))

        # Load dual models
        model_weights_dict = compressed_package['model_weights']
        model_weights_bg = model_weights_dict['bg']
        model_weights_roi = model_weights_dict['roi']

        # Get model types for BG and ROI (may be different)
        model_type_bg = metadata.get('model_type_bg', model_type)
        model_type_roi = metadata.get('model_type_roi', model_type)

        # Create metadata copies for BG and ROI models to handle different parameters
        metadata_bg = metadata.copy()
        metadata_roi = metadata.copy()

        # For BG model, prefer bg_* keys, fallback to default keys
        if 'bg_low_cutoff' in metadata:
            metadata_bg['low_cutoff'] = metadata['bg_low_cutoff']
        if 'bg_mid_cutoff' in metadata:
            metadata_bg['mid_cutoff'] = metadata['bg_mid_cutoff']
        if 'bg_use_phase_sincos' in metadata:
            metadata_bg['use_phase_sincos'] = metadata['bg_use_phase_sincos']
        if 'bg_gn_groups' in metadata:
            metadata_bg['gn_groups'] = metadata['bg_gn_groups']

        # For ROI model, prefer roi_* keys, fallback to default keys
        if 'roi_low_cutoff' in metadata:
            metadata_roi['low_cutoff'] = metadata['roi_low_cutoff']
        if 'roi_mid_cutoff' in metadata:
            metadata_roi['mid_cutoff'] = metadata['roi_mid_cutoff']
        if 'roi_use_phase_sincos' in metadata:
            metadata_roi['use_phase_sincos'] = metadata['roi_use_phase_sincos']
        if 'roi_gn_groups' in metadata:
            metadata_roi['gn_groups'] = metadata['roi_gn_groups']

        # Create BG model
        model_bg = create_model_for_decompress(model_type_bg, metadata_bg, spatial_dims, self.device)
        if model_bg is None:
            return None
        state_dict_bg = {k: torch.from_numpy(v) for k, v in model_weights_bg.items()}
        model_bg.load_state_dict(state_dict_bg)
        model_bg.eval()

        # Create ROI model (may have different architecture)
        model_roi = create_model_for_decompress(model_type_roi, metadata_roi, spatial_dims, self.device)
        if model_roi is None:
            return None
        state_dict_roi = {k: torch.from_numpy(v) for k, v in model_weights_roi.items()}
        model_roi.load_state_dict(state_dict_roi)
        model_roi.eval()

        if verbose:
            print(f"  BG model: {model_type_bg}, parameters: {int(metadata.get('model_params', 0)) // 2:,}")
            print(f"  ROI model: {model_type_roi}, parameters: {int(metadata.get('model_params', 0)) // 2:,}")
            if model_type_bg != model_type_roi:
                print(f"  Using different architectures: BG={model_type_bg}, ROI={model_type_roi}")

        # Get ROI boxes for mask creation (from metadata for Scheme B)
        roi_boxes_3d = metadata.get('roi_boxes_3d', [])
        if not roi_boxes_3d:
            raise ValueError("Dual decompression requires non-empty metadata['roi_boxes_3d'], but it is empty.")

        model = model_bg  # For compatibility

        residual_mean = float(metadata['residual_mean'])
        residual_std = float(metadata['residual_std'])
        input_mean = float(metadata['input_mean'])
        input_std = float(metadata['input_std'])

        batch_size_2d = int(metadata.get('batch_size_2d', 256))
        batch_size_3d = int(metadata.get('batch_size_3d', 256))
        
        # Predict residuals R_hat = f(X')
        # Prepare data for prediction
        if spatial_dims == 2:
            # Transpose for 2D processing
            if slice_order == 'zxy':
                x_prime_for_pred = x_prime.transpose(2, 0, 1)
            elif slice_order == 'yxz':
                x_prime_for_pred = x_prime.transpose(1, 0, 2)
            else:  # 'xyz'
                x_prime_for_pred = x_prime
        else:
            x_prime_for_pred = x_prime

        with torch.no_grad():
            # Normalize input (same as during training!)
            x_prime_norm = (x_prime_for_pred - input_mean) / input_std

            # Dual model prediction (dual-only)
            # Create ROI mask in prediction coordinate space
            roi_mask_pred = np.zeros(x_prime_norm.shape, dtype=bool)
            for (x0, x1, y0, y1, z0, z1) in roi_boxes_3d:
                if spatial_dims == 2:
                    if slice_order == 'zxy':
                        for z in range(z0, z1):
                            if z < x_prime_norm.shape[0]:
                                roi_mask_pred[z, x0:x1, y0:y1] = True
                    elif slice_order == 'yxz':
                        for y in range(y0, y1):
                            if y < x_prime_norm.shape[0]:
                                roi_mask_pred[y, x0:x1, z0:z1] = True
                    else:
                        for x in range(x0, x1):
                            if x < x_prime_norm.shape[0]:
                                roi_mask_pred[x, y0:y1, z0:z1] = True
                else:
                    roi_mask_pred[x0:x1, y0:y1, z0:z1] = True

            pred_residuals_bg_list = []
            pred_residuals_roi_list = []

            if spatial_dims == 2:
                n_slices = x_prime_norm.shape[0]
                for batch_start in range(0, n_slices, batch_size_2d):
                    batch_end = min(batch_start + batch_size_2d, n_slices)
                    x_batch = torch.from_numpy(x_prime_norm[batch_start:batch_end]).float().unsqueeze(1).to(self.device)

                    pred_bg = model_bg(x_batch)
                    if isinstance(pred_bg, tuple):
                        pred_bg = pred_bg[0]
                    pred_residuals_bg_list.append(pred_bg.cpu().numpy().squeeze(1))

                    pred_roi = model_roi(x_batch)
                    if isinstance(pred_roi, tuple):
                        pred_roi = pred_roi[0]
                    pred_residuals_roi_list.append(pred_roi.cpu().numpy().squeeze(1))

                    del x_batch, pred_bg, pred_roi
                    torch.cuda.empty_cache()

                pred_residuals_bg_norm = np.concatenate(pred_residuals_bg_list, axis=0)
                pred_residuals_roi_norm = np.concatenate(pred_residuals_roi_list, axis=0)
            else:
                z_dim = x_prime_norm.shape[2]
                for batch_start in range(0, z_dim, batch_size_3d):
                    batch_end = min(batch_start + batch_size_3d, z_dim)
                    x_batch = x_prime_norm[:, :, batch_start:batch_end]
                    x_batch_tensor = torch.from_numpy(x_batch).float().unsqueeze(0).unsqueeze(0).to(self.device)

                    pred_bg = model_bg(x_batch_tensor)
                    if isinstance(pred_bg, tuple):
                        pred_bg = pred_bg[0]
                    pred_residuals_bg_list.append(pred_bg.cpu().numpy().squeeze())

                    pred_roi = model_roi(x_batch_tensor)
                    if isinstance(pred_roi, tuple):
                        pred_roi = pred_roi[0]
                    pred_residuals_roi_list.append(pred_roi.cpu().numpy().squeeze())

                    del x_batch_tensor, pred_bg, pred_roi
                    torch.cuda.empty_cache()

                pred_residuals_bg_norm = np.concatenate(pred_residuals_bg_list, axis=2)
                pred_residuals_roi_norm = np.concatenate(pred_residuals_roi_list, axis=2)

            # Denormalize
            pred_residuals_bg = pred_residuals_bg_norm * residual_std + residual_mean
            pred_residuals_roi = pred_residuals_roi_norm * residual_std + residual_mean

            # Transpose back if 2D
            if spatial_dims == 2:
                if slice_order == 'zxy':
                    pred_residuals_bg = pred_residuals_bg.transpose(1, 2, 0)
                    pred_residuals_roi = pred_residuals_roi.transpose(1, 2, 0)
                    roi_mask_pred = roi_mask_pred.transpose(1, 2, 0)
                elif slice_order == 'yxz':
                    pred_residuals_bg = pred_residuals_bg.transpose(1, 0, 2)
                    pred_residuals_roi = pred_residuals_roi.transpose(1, 0, 2)
                    roi_mask_pred = roi_mask_pred.transpose(1, 0, 2)

            # Combine: R_hat = R_hat_bg * (1 - M_roi) + R_hat_roi * M_roi
            roi_mask_float = roi_mask_pred.astype(np.float32)
            pred_residuals_np = pred_residuals_bg * (1 - roi_mask_float) + pred_residuals_roi * roi_mask_float

        if verbose:
            print(f"  Predicted residuals: mean={np.mean(pred_residuals_np):.3e}, "
                  f"std={np.std(pred_residuals_np):.3e}")

        # Save original residuals for verification
        pred_residuals_original = pred_residuals_np.copy()

        # ================================================================
        # STEP 3: Post-process residuals (if enabled)
        # ================================================================
        if enable_post_process:
            if verbose:
                print(f"\nStep 3: Post-processing predicted residuals...")

            relative_error_bound = metadata.get("relative_error_bound", 0.0)
            postprocess_a = metadata.get("postprocess_a", 0.1) # 0.5, 0.75, 1.0
            
            postprocessed_residuals = self._error_bounded_post_process_residual(
                pred_residuals=pred_residuals_np,
                x_prime=x_prime,
                absolute_error_bound=absolute_error_bound,
                relative_error_bound=relative_error_bound,
                a=postprocess_a,
                data_range=metadata.get("data_range", None),   # 强烈建议 compression 时存
                mode="residual",  # 或 "value"
                verbose=verbose
            )        

            if verbose:
                verification_results = self.verify_residual_clipping(
                    pred_residuals=pred_residuals_original,
                    postprocessed_residuals=postprocessed_residuals,
                    x_prime=x_prime,
                    absolute_error_bound=absolute_error_bound,
                    relative_error_bound=relative_error_bound,
                    a=postprocess_a,
                    verbose=True
                )
        else:
            postprocessed_residuals = pred_residuals_np

        # ================================================================
        # STEP 4: Enhance Reconstruction (X_post = X' + R_postprocessed)
        # ================================================================
        if verbose:
            print(f"\nStep 4: Computing final reconstruction...")

        x_enhanced = x_prime + postprocessed_residuals        
        decompress_time = time.time() - decompress_start       

        if verbose:
            print(f"\nDecompression complete: {x_enhanced.shape}")
            print(f"Time: {decompress_time:.2f}s")
            print(f"{'='*70}\n")
        
        return x_enhanced             

    def compress_file(self, input_path, output_path, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound=0.0, **kwargs):
        """Compress a file."""
        data = np.fromfile(input_path, dtype=np.float32).reshape(512, 512, 512)
        package, stats = self.compress(
            data, 
            eb_mode=eb_mode,
            absolute_error_bound=absolute_error_bound, 
            relative_error_bound=relative_error_bound,
            pwr_error_bound=pwr_error_bound,
            output_path=output_path, 
            **kwargs
        )
        return stats
    
    def decompress_file(self, input_path, output_path=None):
        """Decompress a file."""
        with open(input_path, 'rb') as f:
            package = pickle.load(f)
        
        reconstructed = self.decompress(package)
        
        if output_path:
            reconstructed.astype(np.float32).tofile(output_path)
        
        return reconstructed
    
    def save_components(self, compressed_package, save_dir, base_name, verbose=True):
        """
        Save compressed components to separate files.
        
        Args:
            compressed_package: Package from compress()
            save_dir: Directory to save files
            base_name: Base name for output files
            verbose: Print progress
        
        Saves:
            - {base_name}.sz3: SZ3 compressed bytes
            - {base_name}_model.pt: Model weights (PyTorch format) [single model]
            - {base_name}_model_bg.pt: BG model weights (PyTorch format) [dual model]
            - {base_name}_model_roi.pt: ROI model weights (PyTorch format) [dual model]
            - {base_name}_metadata.json: Metadata (JSON format)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save SZ3 compressed bytes
        sz3_path = os.path.join(save_dir, f"{base_name}.sz3")
        with open(sz3_path, 'wb') as f:
            f.write(compressed_package['sz_bytes'])
        
        # 2. Save model weights (PyTorch format)
        model_weights = compressed_package['model_weights']
        use_dual = compressed_package['metadata'].get('use_dual_models', False)
        
        if use_dual:
            # Dual models: save BG and ROI separately
            model_path_bg = os.path.join(save_dir, f"{base_name}_model_bg.pt")
            state_dict_bg = {k: torch.from_numpy(v) for k, v in model_weights['bg'].items()}
            torch.save(state_dict_bg, model_path_bg)
            
            model_path_roi = os.path.join(save_dir, f"{base_name}_model_roi.pt")
            state_dict_roi = {k: torch.from_numpy(v) for k, v in model_weights['roi'].items()}
            torch.save(state_dict_roi, model_path_roi)
            
            model_path = model_path_bg  # For size calculation
        else:
            # Single model
            model_path = os.path.join(save_dir, f"{base_name}_model.pt")
            state_dict = {k: torch.from_numpy(v) for k, v in model_weights.items()}
            torch.save(state_dict, model_path)
        
        # 3. Save metadata (JSON format)
        metadata_path = os.path.join(save_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(compressed_package['metadata'], f, indent=2)
        
        if verbose:
            sz3_size = len(compressed_package['sz_bytes'])
            print(f"\n  Components saved to: {save_dir}/")
            print(f"    - {base_name}.sz3: {sz3_size / 1024:.2f} KB")
            if use_dual:
                model_size_bg = os.path.getsize(model_path_bg)
                model_size_roi = os.path.getsize(model_path_roi)
                print(f"    - {base_name}_model_bg.pt: {model_size_bg / 1024:.2f} KB")
                print(f"    - {base_name}_model_roi.pt: {model_size_roi / 1024:.2f} KB")
            else:
                model_size = os.path.getsize(model_path)
                print(f"    - {base_name}_model.pt: {model_size / 1024:.2f} KB")
            print(f"    - {base_name}_metadata.json")
    
    def load_components(self, save_dir, base_name, verbose=True):
        """
        Load compressed components from separate files.
        
        Args:
            save_dir: Directory containing saved files
            base_name: Base name of the files
            verbose: Print progress
        
        Returns:
            compressed_package: Reconstructed package for decompress()
        """
        import os
        
        # 1. Load SZ3 compressed bytes (as numpy array for ctypes compatibility)
        sz3_path = os.path.join(save_dir, f"{base_name}.sz3")
        with open(sz3_path, 'rb') as f:
            sz_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 2. Load metadata first to check if dual models
        metadata_path = os.path.join(save_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        use_dual = metadata.get('use_dual_models', False)
        
        # 3. Load model weights
        if use_dual:
            # Dual models: load BG and ROI separately
            model_path_bg = os.path.join(save_dir, f"{base_name}_model_bg.pt")
            state_dict_bg = torch.load(model_path_bg, map_location='cpu')
            model_weights_bg = {k: v.numpy() for k, v in state_dict_bg.items()}
            
            model_path_roi = os.path.join(save_dir, f"{base_name}_model_roi.pt")
            state_dict_roi = torch.load(model_path_roi, map_location='cpu')
            model_weights_roi = {k: v.numpy() for k, v in state_dict_roi.items()}
            
            model_weights = {
                'bg': model_weights_bg,
                'roi': model_weights_roi
            }
        else:
            # Single model
            model_path = os.path.join(save_dir, f"{base_name}_model.pt")
            state_dict = torch.load(model_path, map_location='cpu')
            model_weights = {k: v.numpy() for k, v in state_dict.items()}
        
        # Reconstruct package
        compressed_package = {
            'backend': 'SZ3',
            'sz_bytes': sz_bytes,
            'model_weights': model_weights,
            'metadata': metadata,
        }
        
        if verbose:
            print(f"\n  Components loaded from: {save_dir}/")
            print(f"    - {base_name}.sz3: {len(sz_bytes) / 1024:.2f} KB")
            if use_dual:
                print(f"    - {base_name}_model_bg.pt")
                print(f"    - {base_name}_model_roi.pt")
            else:
                print(f"    - {base_name}_model.pt")
            print(f"    - {base_name}_metadata.json")
        
        return compressed_package
    
    def reconstruct_from_components(self, save_dir, base_name, output_path=None, 
                                    verbose=True, enable_post_process=False):
        """
        Reconstruct original data from saved components.
        
        Args:
            save_dir: Directory containing saved files
            base_name: Base name of the files
            output_path: Optional path to save reconstructed data as .f32 file
            verbose: Print progress
            enable_post_process: Apply post-processing
        
        Returns:
            reconstructed: Decompressed data (numpy array)
        """
        # Load components
        package = self.load_components(save_dir, base_name, verbose)
        
        # Decompress
        reconstructed = self.decompress(package, verbose=verbose, enable_post_process=enable_post_process)
        

        # Decompress file with SZ3 + Neural Network enhancement 
        
        # Save if path provided
        if output_path:
            reconstructed.astype(np.float32).tofile(output_path)
            if verbose:
                print(f"  Reconstructed data saved to: {output_path}")
                print(f"  Size: {reconstructed.nbytes / (1024**2):.2f} MB")
        
        return reconstructed
    
    def verify_reconstruction(self, original, reconstructed, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound=0, verbose=True):
        """Wrapper for verify_reconstruction from compression_function."""
        return verify_reconstruction(
            original, reconstructed, eb_mode, absolute_error_bound,
            relative_error_bound, pwr_error_bound, verbose,
            compute_spectral_fn=compute_spectral_metrics
        )
    
    def verify_reconstruction_per_slice(self, original, reconstructed, eb_mode, absolute_error_bound, 
                                      relative_error_bound, pwr_error_bound=0, 
                                      slice_axis=2, verbose=True):
        """Wrapper for verify_reconstruction_per_slice from compression_function."""
        return verify_reconstruction_per_slice(
            original, reconstructed, eb_mode, absolute_error_bound,
            relative_error_bound, pwr_error_bound, slice_axis, verbose
        )


__all__ = ['NeurLZCompressor']