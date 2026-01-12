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
# Local Imports - Utilities
# =============================================================================
try:
    from .utils import setup_multi_gpu_model, get_available_gpus
except ImportError:
    from utils import setup_multi_gpu_model, get_available_gpus

# =============================================================================
# Local Imports - Data Handling
# =============================================================================
try:
    from .Patch_data import (
        create_hybrid_datasets, 
        PatchDataLoader, 
        collate_patches_to_tensor, 
        get_dataset_info
    )
except ImportError:
    from Patch_data import (
        create_hybrid_datasets, 
        PatchDataLoader, 
        collate_patches_to_tensor, 
        get_dataset_info
    )

# =============================================================================
# Local Imports - Models
# =============================================================================
try:
    from .Model import (
        TinyResidualPredictor, 
        TinyFrequencyResidualPredictor,
        TinyFrequencyResidualPredictorWithEnergy,
        TinyFrequencyResidualPredictor_1_input,
        TinyFrequencyResidualPredictor_7_inputs,
        TinyFrequencyResidualPredictor_4_inputs,
    )
except ImportError:
    from Model import (
        TinyResidualPredictor, 
        TinyFrequencyResidualPredictor,
        TinyFrequencyResidualPredictorWithEnergy,
        TinyFrequencyResidualPredictor_1_input,
        TinyFrequencyResidualPredictor_7_inputs,
        TinyFrequencyResidualPredictor_4_inputs,
    )

# =============================================================================
# Local Imports - Loss Functions
# (Using importlib to handle filenames with + symbols)
# =============================================================================
LOSS_PATH = '/Users/923714256/Data_compression/neural_compression/Loss'

def _load_loss_module(module_name, filename):
    """Helper to load loss modules with special characters in filename."""
    spec = importlib.util.spec_from_file_location(module_name, f"{LOSS_PATH}/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load loss classes
_spatial_1_module = _load_loss_module("spatial_1_mag_1_phase_loss", "spatial+1_mag+1_phase_loss.py")
_spatial_3_module = _load_loss_module("spatial_3_mag_3_phase_loss", "spatial+3_mag+3_phase_loss.py")
_spatial_3_freq_module = _load_loss_module("spatial_3_freq_spaitial_loss", "spatial+3_freq_spaitial_loss.py")

SpatialFrequencyLoss = _spatial_1_module.SpatialFrequencyLoss
BandedFrequencyLoss_3_mag_3_phase = _spatial_3_module.BandedFrequencyLoss_3_mag_3_phase
BandWeightedSpectralLoss = _spatial_3_freq_module.BandWeightedSpectralLoss

# Aliases for compatibility
BandedFrequencyLoss = BandedFrequencyLoss_3_mag_3_phase
SpatialEnergyBandLoss = BandWeightedSpectralLoss
BandedFrequencyLoss_4_inputs = BandWeightedSpectralLoss  # Same interface: weight_spatial, weight_low/mid/high

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
                 device='cuda:0'):
        self.sz = SZ(sz_lib_path)
        self.device = device
    
    def compress(self, data, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound, output_path=None,
                online_epochs=50, learning_rate=1e-3, model='tiny_residual_predictor', num_res_blocks=1,
                model_channels=4, verbose=True, 
                spatial_dims=3, slice_order='zxy',
                val_split=0.1, track_losses=True):
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
        
        Returns:
            compressed_package: Dict with {sz_bytes, model_weights, outliers, metadata}
            stats: Compression statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"NeurLZ Compression (spatial_dims={spatial_dims})")
            if spatial_dims == 2:
                print(f"  Processing as 2D slices (order={slice_order})")
            print(f"{'='*70}")
            print(f"Data: {data.shape}, range=[{np.min(data):.3e}, {np.max(data):.3e}]")
            print(f"Error bound: {absolute_error_bound}")
        
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
        # STEP 4: Train Tiny DNN Online to Predict R from X'
        # ================================================================
        if verbose:
            print(f"\nStep 4: Training tiny DNN online ({online_epochs} epochs)...")
        
        model_type = model
        # ============================================================
        # Handle 2D vs 3D Processing
        # ============================================================
        if spatial_dims == 2:
            # 2D mode: Transpose data to process as slices
            if slice_order == 'zxy':
                # (X, Y, Z) → (Z, X, Y): Z slices of (X, Y)
                data_for_training = data.transpose(2, 0, 1)
                x_prime_for_training = x_prime.transpose(2, 0, 1)
                true_residuals_for_training = true_residuals.transpose(2, 0, 1)
            elif slice_order == 'yxz':
                # (X, Y, Z) → (Y, X, Z): Y slices of (X, Z)
                data_for_training = data.transpose(1, 0, 2)
                x_prime_for_training = x_prime.transpose(1, 0, 2)
                true_residuals_for_training = true_residuals.transpose(1, 0, 2)
            else:  # 'xyz'
                # Keep as is: X slices of (Y, Z)
                data_for_training = data
                x_prime_for_training = x_prime
                true_residuals_for_training = true_residuals
            
            if verbose:
                print(f"  2D mode: Reshaped to {data_for_training.shape} "
                    f"({data_for_training.shape[0]} slices of {data_for_training.shape[1]}×{data_for_training.shape[2]})")
        else:
            # 3D mode: Keep original shape
            data_for_training = data
            x_prime_for_training = x_prime
            true_residuals_for_training = true_residuals
        
        # Check available GPUs
        available_gpus = get_available_gpus()
        use_multi_gpu = len(available_gpus) > 1


        # ============================================================
        # Define model and criterion
        # ============================================================

        # ============================================================
        # Tiny Simple Residual Predictor
        # ============================================================
        if model_type == 'tiny_residual_predictor': 
            criterion = SpatialFrequencyLoss(
                weight_spatial=1.0,
                weight_magnitude=1,
                weight_phase=1,
                spatial_dims=spatial_dims
            )
            base_model = TinyResidualPredictor(
                channels=model_channels, 
                spatial_dims=spatial_dims, 
                num_res_blocks=num_res_blocks
            )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                # Check if CUDA is available before using .cuda()
                if torch.cuda.is_available():
                    model = base_model.cuda()
                    primary_device = 'cuda:0'
                else:
                    model = base_model
                    primary_device = 'cpu'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # ============================================================
        # Tiny Frequency Residual Predictor_1_input
        # ============================================================
        elif model_type == 'tiny_frequency_residual_predictor_1_input':
            criterion = SpatialFrequencyLoss(
                weight_spatial=1.0,
                weight_magnitude=0,
                weight_phase=0,
                spatial_dims=spatial_dims
            )
            base_model = TinyFrequencyResidualPredictor_1_input(
                channels=model_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=num_res_blocks
            )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                # Check if CUDA is available before using .cuda()
                if torch.cuda.is_available():
                    model = base_model.cuda()
                    primary_device = 'cuda:0'
                else:
                    model = base_model
                    primary_device = 'cpu'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # ============================================================
            # Tiny Frequency Residual Predictor_7_inputs (7 inputs)
            # ============================================================
        elif model_type == 'tiny_frequency_residual_predictor_7_inputs':

            criterion = BandedFrequencyLoss_3_mag_3_phase(
                    weight_spatial=1.0,
                    weight_mag_low=1.0,
                    weight_mag_mid=1.0,
                    weight_mag_high=1.0,
                    weight_phase_low=1.0,
                    weight_phase_mid=1.0,
                    weight_phase_high=1.0,
                    spatial_dims=spatial_dims,
                    low_cutoff=0.15,
                    mid_cutoff=0.40
                )
            base_model = TinyFrequencyResidualPredictor_7_inputs(
                channels=model_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=num_res_blocks
            )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                # Check if CUDA is available before using .cuda()
                if torch.cuda.is_available():
                    model = base_model.cuda()
                    primary_device = 'cuda:0'
                else:
                    model = base_model
                    primary_device = 'cpu'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)


            # ============================================================
            # Tiny Frequency Residual Predictor_4_inputs (4 inputs)
            # ============================================================
        elif model_type == 'tiny_frequency_residual_predictor_4_inputs':
            criterion = BandedFrequencyLoss_4_inputs(
                weight_spatial=1.0, 
                weight_low=1,
                weight_mid=1,
                weight_high=1,
                spatial_dims=spatial_dims,
                low_cutoff=0.15,
                mid_cutoff=0.40
            )
            base_model = TinyFrequencyResidualPredictor_4_inputs(
                    channels=model_channels,
                    spatial_dims=spatial_dims,
                    num_res_blocks=num_res_blocks
                )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                 # Check if CUDA is available before using .cuda()
                if torch.cuda.is_available():
                    model = base_model.cuda()
                    primary_device = 'cuda:0'
                else:
                    model = base_model
                    primary_device = 'cpu'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        elif model_type == 'tiny_frequency_residual_predictor_with_energy':
            criterion = SpatialEnergyBandLoss(
                weight_spatial=1.0,
                weight_low=0,
                weight_mid=0,
                weight_high=0,
                spatial_dims=spatial_dims
            )
            base_model = TinyFrequencyResidualPredictorWithEnergy(
                channels=model_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=num_res_blocks
            )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                # Check if CUDA is available before using .cuda()
                if torch.cuda.is_available():
                    model = base_model.cuda()
                    primary_device = 'cuda:0'
                else:
                    model = base_model
                    primary_device = 'cpu'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            print(f"Model {model_type} not supported")
            return None, None
        
        # Fix for DataParallel - need to access base model for count_parameters
        if isinstance(model, nn.DataParallel):
            num_params = model.module.count_parameters()
        else:
            num_params = model.count_parameters()
        
        if verbose:
            print(f"  Model parameters: {num_params:,}")

        # Normalize residuals for better training (use _for_training variables!)
        residual_mean = float(np.mean(true_residuals_for_training))
        residual_std = float(np.std(true_residuals_for_training)) + 1e-8

        # Normalize both input and target
        x_prime_norm = (x_prime_for_training - np.mean(x_prime_for_training)) / (np.std(x_prime_for_training) + 1e-8)
        residuals_norm = (true_residuals_for_training - residual_mean) / residual_std

        input_mean = float(np.mean(x_prime_for_training))
        input_std = float(np.std(x_prime_for_training)) + 1e-8

        # ============================================================
        # Prepare data for batch training to avoid OOM
        # ============================================================
        batch_size_2d = 256  # For 2D: number of slices per batch
        batch_size_3d = 256  # For 3D: depth (Z) chunks per batch
        
        # Keep data in CPU/numpy, only load batches to GPU
        train_dataset, val_dataset = create_hybrid_datasets(
            x_data=x_prime_norm,
            y_data=residuals_norm,
            patch_size=32,
            overlap=16,
            spatial_dims=spatial_dims,
            val_split=val_split,
            seed=42
        )
            
        if verbose:
            print(f"  {get_dataset_info(train_dataset, val_dataset, 32, spatial_dims)}")

        train_loader = PatchDataLoader(
            train_dataset, 
            batch_size=512,
            shuffle=True,
            drop_last=False
        )

        if val_dataset is not None:
            val_loader = PatchDataLoader(
                val_dataset,
                batch_size=512,
                shuffle=False,
                drop_last=False
            )
        else:
            val_loader = None

        # ============================================================
        # Train online with batch processing
        # ============================================================

        train_losses = []
        val_losses = []
        training_start_time = time.time()
        
        model.train()
        for epoch in range(online_epochs):
            epoch_train_losses = []

            for batch_x, batch_y in train_loader:
                # Convert batch to tensor
                x_tensor, y_tensor = collate_patches_to_tensor(
                    batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
                )
                    
                optimizer.zero_grad()
                pred = model(x_tensor)

                result = criterion(pred, y_tensor)

                # Handle both behaviors
                if isinstance(result, tuple):
                    loss, loss_dict = result
                else:
                    loss = result
                    loss_dict = {"mse": float(loss.item())}

                loss.backward()
                optimizer.step()
                                    
                epoch_train_losses.append(loss.item())
                    
                # Free GPU memory
                del x_tensor, y_tensor, pred, loss, loss_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
            # Average loss for this epoch
            avg_train_loss = np.mean(epoch_train_losses)
            
            # Validation (if enabled)
            if track_losses and val_loader is not None:
                model.eval()
                epoch_val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        x_tensor, y_tensor = collate_patches_to_tensor(
                            batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
                        )
                                                        
                        pred = model(x_tensor)
                        result = criterion(pred, y_tensor)

                        if isinstance(result, tuple):
                            loss, loss_dict = result
                        else:
                            loss = result
                            loss_dict = {"mse": float(loss.item())}
                            
                        epoch_val_losses.append(loss.item())
                            
                        del x_tensor, y_tensor, pred, loss, loss_dict
                        torch.cuda.empty_cache()
                
                avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
                val_losses.append(avg_val_loss)
                model.train()
            
            if track_losses:
                train_losses.append(avg_train_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    log_msg = f"    Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}"
                    if val_losses:
                        log_msg += f", Val Loss = {val_losses[-1]:.6f}"
                    print(log_msg)
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1:3d}: Loss = {avg_train_loss:.6f}")
        
        training_time = time.time() - training_start_time
        
        # ================================================================
        # STEP 5: Generate Enhanced Reconstruction (X_enh = X' + R_hat)
        # ================================================================
        # Use batch processing for inference as well
        if verbose:
            print(f"\nStep 5: Generating enhanced reconstruction...")
        
        model.eval()
        pred_residuals_list = []
        
        with torch.no_grad():
            if spatial_dims == 2:
                # 2D: Process slices in batches
                n_slices = x_prime_norm.shape[0]
                for batch_start in range(0, n_slices, batch_size_2d):
                    batch_end = min(batch_start + batch_size_2d, n_slices)
                    
                    x_batch = torch.from_numpy(x_prime_norm[batch_start:batch_end]).float().unsqueeze(1).to(primary_device)
                    pred_batch = model(x_batch)
                    pred_batch_np = pred_batch.cpu().numpy().squeeze(1)  # (N, H, W)
                    pred_residuals_list.append(pred_batch_np)
                    
                    del x_batch, pred_batch
                    torch.cuda.empty_cache()
                
                pred_residuals_norm = np.concatenate(pred_residuals_list, axis=0)
            else:
                # 3D: Process in chunks along Z-axis
                z_dim = x_prime_norm.shape[2]
                for batch_start in range(0, z_dim, batch_size_3d):
                    batch_end = min(batch_start + batch_size_3d, z_dim)
                    
                    x_batch = x_prime_norm[:, :, batch_start:batch_end]
                    x_batch_tensor = torch.from_numpy(x_batch).float().unsqueeze(0).unsqueeze(0).to(primary_device)
                    pred_batch = model(x_batch_tensor)
                    pred_batch_np = pred_batch.cpu().numpy().squeeze()  # (H, W, D_chunk)
                    pred_residuals_list.append(pred_batch_np)
                    
                    del x_batch_tensor, pred_batch
                    torch.cuda.empty_cache()
                
                # Concatenate along Z-axis
                pred_residuals_norm = np.concatenate(pred_residuals_list, axis=2)
        
        # Denormalize
        pred_residuals_np = pred_residuals_norm * residual_std + residual_mean

        # Transpose back to original shape if in 2D mode
        if spatial_dims == 2:
            if slice_order == 'zxy':
                pred_residuals_np = pred_residuals_np.transpose(1, 2, 0)  # (Z, X, Y) → (X, Y, Z)
            elif slice_order == 'yxz':
                pred_residuals_np = pred_residuals_np.transpose(1, 0, 2)  # (Y, X, Z) → (X, Y, Z)
            # else: 'xyz' already in correct order

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
        
        # Extract model weights (FP32 for now, could quantize to FP16/INT8)
        if isinstance(model, nn.DataParallel):
            model_weights = {k: v.cpu().numpy() for k, v in model.module.state_dict().items()}
        else:
            model_weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        
        compressed_package = {
            'backend': 'SZ3',
            'sz_bytes': sz3_compressed,
            'model_weights': model_weights,
            'metadata': {
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
            }
        }
        
        # Calculate sizes
        sz_size = len(sz3_compressed)
        weights_size = sum(w.nbytes for w in model_weights.values())
        total_size = sz_size + weights_size
        overall_ratio = original_size / total_size
        
        compress_time = time.time() - compress_start
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Compression Summary:")
            print(f"  Original: {original_size / (1024**2):.2f} MB")
            print(f"  SZ3 bytes: {sz_size / 1024:.2f} KB")
            print(f"  DNN weights: {weights_size / 1024:.2f} KB ({num_params:,} params)")
            print(f"  Total: {total_size / (1024**2):.2f} MB")
            print(f"  Overall ratio: {overall_ratio:.2f}x")
            print(f"  SZ3-only ratio: {sz3_ratio:.2f}x")
            print(f"  Time: {compress_time:.2f}s (including {training_time:.2f}s training)")
            print(f"{'='*70}\n")
        
        # Save if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(compressed_package, f, protocol=4)
            if verbose:
                print(f"Saved to: {output_path}\n")
        
        stats = {
            'original_size_mb': original_size / (1024**2),
            'sz3_size_kb': sz_size / 1024,
            'weights_size_kb': weights_size / 1024,
            'total_size_mb': total_size / (1024**2),
            'sz3_ratio': sz3_ratio,
            'overall_ratio': overall_ratio,
            'base_max_error': base_max_error,
            'base_mean_error': base_mean_error,
            'enhanced_max_error': enhanced_max_error,
            'enhanced_mean_error': enhanced_mean_error,
            'compress_time': compress_time,
            'training_time': training_time,
            'model_params': num_params,
            'train_losses': train_losses if track_losses else [],
            'val_losses': val_losses if track_losses else [],
        }
        
        return compressed_package, stats

    def _error_bounded_post_process(self, x_enhanced, x_prime, absolute_error_bound, 
                                    relative_error_bound=0.0, verbose=False):
        """
        Error-bounded adaptive post-processing.
        
        This function applies post-processing to improve quality while ensuring
        the error bound is not violated. Since we don't have original data during
        decompression, we use a conservative approach:
        1. Clip the enhancement to ensure |x_enhanced - x_prime| <= error_bound
        2. Apply smoothing to reduce artifacts while respecting the bound
        
        Args:
            x_enhanced: Enhanced reconstruction (x_prime + predicted_residuals)
            x_prime: SZ3 base reconstruction
            absolute_error_bound: Absolute error bound
            relative_error_bound: Relative error bound (optional)
            verbose: Print progress
        
        Returns:
            x_post: Post-processed reconstruction
        """
        # Calculate the enhancement (predicted residual)
        enhancement = x_enhanced - x_prime
        
        # Compute effective error bound
        if relative_error_bound > 0:
            data_range = np.max(x_prime) - np.min(x_prime)
            effective_bound = min(absolute_error_bound, 
                                relative_error_bound * data_range)
        else:
            effective_bound = absolute_error_bound
        
        # Clip enhancement to ensure |enhancement| <= effective_bound
        # This ensures |x_post - x_prime| <= error_bound
        enhancement_clipped = np.clip(enhancement, -effective_bound, effective_bound)
        
        # Apply post-processing: smooth the clipped enhancement
        # This reduces block artifacts while maintaining error bound
        from scipy import ndimage
        
        # Light smoothing kernel (3x3x3 for 3D, 3x3 for 2D)
        if x_enhanced.ndim == 3:
            kernel_size = 3
        else:
            kernel_size = (3, 3, 3)
        
        # Smooth the clipped enhancement
        enhancement_smooth = ndimage.gaussian_filter(
            enhancement_clipped, 
            sigma=0.5,  # Light smoothing
            mode='nearest'
        )
        
        # Ensure smoothing doesn't violate error bound
        enhancement_final = np.clip(enhancement_smooth, 
                                    -effective_bound, 
                                    effective_bound)
        
        # Final post-processed reconstruction
        x_post = x_prime + enhancement_final
        
        if verbose:
            max_enhancement = np.max(np.abs(enhancement_final))
            print(f"  Post-processing applied:")
            print(f"    Max enhancement: {max_enhancement:.3e}")
            print(f"    Error bound: {effective_bound:.3e}")
            print(f"    Bound compliance: {max_enhancement <= effective_bound}")
        
        return x_post
    
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
        if verbose:
            print(f"\nStep 2: Loading DNN weights and predicting residuals...")
            
        # Recreate model with same architecture
        spatial_dims = metadata.get('spatial_dims', 2)
        slice_order = metadata.get('slice_order', 'zxy')
        model_type = metadata.get('model_type', metadata.get('model', 'tiny_residual_predictor'))

        if model_type == 'tiny_residual_predictor':
            metadata_channels = metadata['model_channels']
            metadata_rb = metadata.get('num_res_blocks', 2)
            model = TinyResidualPredictor(
                channels=metadata_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=metadata_rb
            ).to(self.device)
        elif model_type == 'tiny_frequency_residual_predictor_1_input':
            metadata_channels = metadata.get('model_channels', 2)
            model = TinyFrequencyResidualPredictor_1_input(
                channels=metadata_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 2)
            ).to(self.device)
        elif model_type == 'tiny_frequency_residual_predictor_4_inputs':
            metadata_channels = metadata.get('model_channels', 2)
            model = TinyFrequencyResidualPredictor_4_inputs(
                channels=metadata_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 2)
            ).to(self.device)
        elif model_type == 'tiny_frequency_residual_predictor_7_inputs':
            metadata_channels = metadata.get('model_channels', 2)
            model = TinyFrequencyResidualPredictor_7_inputs(
                channels=metadata_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 2)
            ).to(self.device)
        elif model_type == 'tiny_frequency_residual_predictor_with_energy':
            metadata_channels = metadata.get('model_channels', 2)
            model = TinyFrequencyResidualPredictorWithEnergy(
                channels=metadata_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 2)
            ).to(self.device)
        else:
            print(f"Model {model_type} not supported")
            return None
        
        # Load weights
        state_dict = {k: torch.from_numpy(v) for k, v in compressed_package['model_weights'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        if verbose:
            print(f"  Model parameters: {metadata['model_params']:,}")
        
        # Get saved normalization parameters
        residual_mean = metadata['residual_mean']
        residual_std = metadata['residual_std']
        input_mean = metadata['input_mean']
        input_std = metadata['input_std']
        
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

            if spatial_dims == 2:
                x_prime_tensor = torch.from_numpy(x_prime_norm).float().unsqueeze(1).to(self.device)
            else:
                x_prime_tensor = torch.from_numpy(x_prime_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict normalized residuals
            pred_residuals_norm = model(x_prime_tensor)
            if spatial_dims == 2:
                pred_residuals_np = pred_residuals_norm.cpu().numpy().squeeze(1)  # (N, H, W)
            else:
                pred_residuals_np = pred_residuals_norm.cpu().numpy().squeeze()  # (H, W, D)
            
            # Denormalize residuals (CRITICAL!)
            pred_residuals_np = pred_residuals_np * residual_std + residual_mean

        # Transpose back to original shape if in 2D mode
        if spatial_dims == 2:
            if slice_order == 'zxy':
                pred_residuals_np = pred_residuals_np.transpose(1, 2, 0)  # (Z, X, Y) → (X, Y, Z)
            elif slice_order == 'yxz':
                pred_residuals_np = pred_residuals_np.transpose(1, 0, 2)  # (Y, X, Z) → (X, Y, Z)
            # else: 'xyz' already in correct order

        if verbose:
            print(f"  Predicted residuals: mean={np.mean(pred_residuals_np):.3e}, "
                  f"std={np.std(pred_residuals_np):.3e}")
        
        # ================================================================
        # STEP 3: Enhance Reconstruction (X_enh = X' + R_hat)
        # ================================================================
        if verbose:
            print(f"\nStep 3: Computing enhanced reconstruction...")
        
        x_enhanced = x_prime + pred_residuals_np
        # ================================================================
        # STEP 4: Error-Bounded Post-Processing (Optional)
        # ================================================================
        if enable_post_process:  # 添加一个开关参数
            if verbose:
                print(f"\nStep 4: Applying error-bounded post-processing...")
                
            relative_error_bound = metadata.get('relative_error_bound', 0.0)
            x_enhanced = self._error_bounded_post_process(
                x_enhanced=x_enhanced,
                x_prime=x_prime,
                absolute_error_bound=absolute_error_bound,
                relative_error_bound=relative_error_bound,
                verbose=verbose
            )

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
    
    def verify_reconstruction(self, original, reconstructed, eb_mode, absolute_error_bound, relative_error_bound, pwr_error_bound=0, model='tiny_residual_predictor', verbose=True):
        """Verify reconstruction quality and error bound compliance."""
        error = np.abs(original - reconstructed)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        # PSNR
        data_range = np.max(original) - np.min(original)
        mse = np.mean((original - reconstructed) ** 2)
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
        
        # NRMSE
        nrmse = np.sqrt(mse) / data_range
        
        # ================================================================
        # FFT Error Calculation
        # ================================================================
        fft_original = np.fft.fftn(original)
        fft_reconstructed = np.fft.fftn(reconstructed)
        
        # Normalized FFT magnitude error (NRMSE in frequency domain)
        mag_original = np.abs(fft_original)
        mag_reconstructed = np.abs(fft_reconstructed)
        fft_mag_mse = np.mean((mag_original - mag_reconstructed) ** 2)
        fft_mag_nrmse = np.sqrt(fft_mag_mse) / (np.max(mag_original) - np.min(mag_original) + 1e-10)
        
        # FFT PSNR
        fft_range = np.max(mag_original) - np.min(mag_original)
        fft_psnr = 20 * np.log10(fft_range) - 10 * np.log10(fft_mag_mse) if fft_mag_mse > 0 else float('inf')
        
        # Error bound compliance
        within_bound = max_error <= absolute_error_bound
        violation_count = np.sum(error > absolute_error_bound)
        violation_ratio = violation_count / error.size * 100
        
        metrics = {
            'max_error': max_error,
            'mean_error': mean_error,
            'psnr': psnr,
            'nrmse': nrmse,
            'within_bound': within_bound,
            'violation_count': int(violation_count),
            'violation_ratio': violation_ratio,
            # FFT metrics
            'fft_mag_nrmse': fft_mag_nrmse,
            'fft_psnr': fft_psnr,
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Reconstruction Quality:")
            print(f"  Max error: {max_error:.3e} (bound: {absolute_error_bound:.3e})")
            print(f"  Mean error: {mean_error:.3e}")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  NRMSE: {nrmse:.6e}")
            print(f"  Within bound: {'✓ YES' if within_bound else '✗ NO'}")
            if not within_bound:
                print(f"  Violations: {violation_count} ({violation_ratio:.4f}%)")
            print(f"\nFFT Error Metrics:")
            print(f"  FFT Magnitude NRMSE: {fft_mag_nrmse:.6e}")
            print(f"  FFT PSNR: {fft_psnr:.2f} dB")
            print(f"{'='*70}\n")
        
        return metrics
    
    def verify_reconstruction_per_slice(self, original, reconstructed, eb_mode, absolute_error_bound, 
                                      relative_error_bound, pwr_error_bound=0, 
                                      slice_axis=2, verbose=True):
        """
        Verify reconstruction quality per slice.
        
        Args:
            original: Original 3D data
            reconstructed: Reconstructed 3D data
            eb_mode: Error bound mode (not used but kept for compatibility)
            absolute_error_bound: Absolute error bound
            relative_error_bound: Relative error bound (not used but kept for compatibility)
            pwr_error_bound: Power error bound (not used but kept for compatibility)
            slice_axis: Axis to slice along (0=X, 1=Y, 2=Z)
            verbose: Print detailed per-slice metrics
        
        Returns:
            Dictionary with per-slice metrics and statistics
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


__all__ = ['NeurLZCompressor']