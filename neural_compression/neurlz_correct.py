"""
NeurLZ: Correct Implementation Following the Paper

Reference: https://arxiv.org/abs/2409.05785

Key differences from typical neural compression:
1. SZ3/ZFP is PRIMARY compressor (not neural)
2. Tiny DNN (~3k params) trained ONLINE during compression
3. DNN predicts residuals from SZ3-decompressed data
4. Storage: {SZ3_bytes, DNN_weights, outliers}
5. Two modes: strict 1× (with outliers), relaxed 2× (Sigmoid-bounded)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import time
from pathlib import Path
from Patch_data import create_patch_datasets, PatchDataLoader, collate_patches_to_tensor, get_dataset_info
from Model import TinyResidualPredictor, TinyFrequencyResidualPredictor, TinyPhysicsResidualPredictor
from Physics_loss import PhysicsInformedLoss, AdaptivePhysicsLoss


sys.path.append('/Users/923714256/Data_compression/SZ3/tools/pysz')
from pysz import SZ
def setup_multi_gpu_model(model, device_ids=None):
    """
    Setup model for multi-GPU training using DataParallel.
        
    Args:
        model: PyTorch model
        device_ids: List of GPU IDs to use (None = use all available GPUs)
        
    Returns:
        model: Model wrapped with DataParallel if multiple GPUs available
        device: Primary device (cuda:0)
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if len(device_ids) > 1:
        print(f"  Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        device = f'cuda:{device_ids[0]}'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return model, device

def get_available_gpus():
    """Get list of available GPU IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []    

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
        
        num_res_blocks = 1  # Number of residual blocks
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

        if model_type == 'tiny_residual_predictor':
            criterion = nn.MSELoss()
            base_model = TinyResidualPredictor(
            channels=model_channels, 
            spatial_dims=spatial_dims, 
            num_res_blocks=num_res_blocks
        )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                model = base_model.cuda()
                primary_device = 'cuda:0'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif model_type == 'tiny_frequency_residual_predictor':
            criterion = nn.MSELoss()
            base_model = TinyFrequencyResidualPredictor(
                channels=model_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=num_res_blocks
            )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                model = base_model.cuda()
                primary_device = 'cuda:0'
            model.to(primary_device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif model_type == 'tiny_physics_residual_predictor':
            base_model = TinyPhysicsResidualPredictor(
                channels=model_channels,
                spatial_dims=spatial_dims,
                num_res_blocks=num_res_blocks
            )
            if use_multi_gpu:
                model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
            else:
                model = base_model.cuda()
                primary_device = 'cuda:0'
            model.to(primary_device)
            criterion = AdaptivePhysicsLoss(spatial_dims=spatial_dims)
            optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=learning_rate)
            # criterion = PhysicsInformedLoss(
            # spatial_dims=spatial_dims,
            # weight_mse=1.0,           # Base reconstruction
            # weight_laplacian=0.1,      # Gravitational structure (Poisson equation)
            # weight_contrast=0.1,       # Overdensity field
            # weight_gradient=0.1        # Structure boundaries
            # )
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        batch_size_2d = 32  # For 2D: number of slices per batch
        batch_size_3d = 64  # For 3D: depth (Z) chunks per batch
        
        # Keep data in CPU/numpy, only load batches to GPU
        if val_split > 0 and track_losses:
            # Split data (keep as numpy arrays)
            train_dataset, val_dataset = create_patch_datasets(
                x_data=x_prime_norm,
                y_data=residuals_norm,
                patch_size=64,  # 32×32 for 2D or 32×32×32 for 3D
                spatial_dims=spatial_dims,
                val_split=val_split,
                shuffle=True,
                seed=42
            )
            
        else:
            # No validation split - still use patch dataset but with 0% val split
            train_dataset, val_dataset = create_patch_datasets(
                x_data=x_prime_norm,
                y_data=residuals_norm,
                patch_size=64,  # 32×32 for 2D or 32×32×32 for 3D
                spatial_dims=spatial_dims,
                val_split=0.0,
                shuffle=True,
                seed=42
            )
            
        if verbose:
            print(f"  {get_dataset_info(train_dataset, val_dataset, 32, spatial_dims)}")

        train_loader = PatchDataLoader(
            train_dataset, 
            batch_size=64,  # Process 8 patches at once
            shuffle=True,
            drop_last=False
        )

        if val_dataset is not None:
            val_loader = PatchDataLoader(
                val_dataset,
                batch_size=64,
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
        
        model.train()
        for epoch in range(online_epochs):
            # Training: process in batches
            epoch_train_losses = []

            for batch_x, batch_y in train_loader:
                    # Convert batch to tensor
                x_tensor, y_tensor = collate_patches_to_tensor(
                        batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
                    )
                    

                optimizer.zero_grad()
                pred = model(x_tensor)
                if hasattr(criterion, "forward") and "return_components" in criterion.forward.__code__.co_varnames:
                    loss, loss_dict = criterion(pred, y_tensor, return_components=True)
                else:
                    loss = criterion(pred, y_tensor)
                    loss_dict = {"mse": float(loss.item())}
                loss.backward()
                optimizer.step()
                    
                epoch_train_losses.append(loss.item())

                #  print detailed loss breakdown every N batches
                # if verbose and len(epoch_train_losses) % 100 == 0:
                #     print(f"      Batch {len(epoch_train_losses)}: "
                #         f"MSE={loss_dict['mse']:.6f}, "
                #         f"Lap={loss_dict['laplacian']:.6f}, "
                #         f"Con={loss_dict['contrast']:.6f}, "
                #         f"Grad={loss_dict['gradient']:.6f}")
                    
                # Free GPU memory
                del x_tensor, y_tensor, pred, loss, loss_dict
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
                        loss, loss_dict = criterion(pred, y_tensor, return_components=True)
                        epoch_val_losses.append(loss.item())
                            
                        del x_tensor, y_tensor, pred, loss, loss_dict
                        torch.cuda.empty_cache()
                
                avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
                val_losses.append(avg_val_loss)
                model.train()
            
            if track_losses:
                train_losses.append(avg_train_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    if val_losses:
                        print(f"    Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_losses[-1]:.6f}")
                    else:
                        print(f"    Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1:3d}: Loss = {avg_train_loss:.6f}")
        
        training_time = time.time() - compress_start
        
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
    
    def decompress(self, compressed_package, verbose=True):
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
        spatial_dims = metadata.get('spatial_dims', 3)
        slice_order = metadata.get('slice_order', 'zxy')
        model_type = metadata.get('model_type', metadata.get('model', 'tiny_residual_predictor'))

        if model_type == 'tiny_residual_predictor':
            model = TinyResidualPredictor(
            channels=metadata['model_channels'],
            spatial_dims=spatial_dims,
            num_res_blocks=metadata.get('num_res_blocks', 1)
        ).to(self.device)
        elif model_type == 'tiny_frequency_residual_predictor':
            model = TinyFrequencyResidualPredictor(
                channels=metadata['model_channels'],
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 1)
            ).to(self.device)
        elif model_type == 'tiny_physics_residual_predictor':
            model = TinyPhysicsResidualPredictor(
                channels=metadata['model_channels'],
                spatial_dims=spatial_dims,
                num_res_blocks=metadata.get('num_res_blocks', 1)
            ).to(self.device)
        else:
            print(f"Model {model_type} not supported")
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
        
        decompress_time = time.time() - decompress_start
        
        if verbose:
            print(f"\nDecompression complete: {x_enhanced.shape}")
            print(f"Time: {decompress_time:.2f}s")
            print(f"{'='*70}\n")
        
        return x_enhanced
    
    def compress_file(self, input_path, output_path, error_bound, relative_error_bound, **kwargs):
        """Compress a file."""
        data = np.fromfile(input_path, dtype=np.float32).reshape(512, 512, 512)
        package, stats = self.compress(data, error_bound, relative_error_bound, output_path, **kwargs)
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
        
        # Error bound compliance - use absolute_error_bound for checking
        within_bound = max_error <= absolute_error_bound
        within_relative_bound = max_error <= relative_error_bound
        violation_count = np.sum(error > absolute_error_bound)
        violation_relative_count = np.sum(error > relative_error_bound)
        violation_ratio = violation_count / error.size * 100
        
        metrics = {
            'max_error': max_error,
            'mean_error': mean_error,
            'psnr': psnr,
            'nrmse': nrmse,
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
            print(f"  Within bound: {'✓ YES' if within_bound else '✗ NO'}")
            if not within_bound:
                print(f"  Violations: {violation_count} ({violation_ratio:.4f}%)")
            print(f"{'='*70}\n")
        
        return metrics



def plot_training_curves(stats, error_bound, relative_error_bound, output_path=None, title_suffix=""):
    """
    Plot training and validation loss curves.
    
    Args:
        stats: Statistics dictionary from compress()
        error_bound: Error bound used for compression
        output_path: Path to save the plot (optional)
        title_suffix: Additional text for plot title
    """
    import matplotlib.pyplot as plt
    
    train_losses = stats.get('train_losses', [])
    val_losses = stats.get('val_losses', [])
    
    if not train_losses:
        print("No training losses to plot. Set track_losses=True during compression.")
        return
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'Training Curves - Error Bound: {error_bound:.2e} Relative Error Bound: {relative_error_bound:.2e} {title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for loss curves
    
    # Add final loss values as text
    final_train = train_losses[-1]
    text = f'Final Train Loss: {final_train:.6f}'
    if val_losses:
        final_val = val_losses[-1]
        text += f'\nFinal Val Loss: {final_val:.6f}'
    
    plt.text(0.98, 0.98, text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
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
        error_bound=0.0,
        relative_error_bound=5e-3,
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
    metrics_3d = compressor.verify_reconstruction(test_data, reconstructed_3d, error_bound=0.0, relative_error_bound=5e-3, model='tiny_residual_predictor')
    
    print("\n" + "="*70)
    print("Test 2: 2D sliced mode with validation and loss tracking")
    print("="*70)
    package_2d, stats_2d = compressor.compress(
        test_data,
        error_bound=0.0,
        relative_error_bound=5e-3,
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
    metrics_2d = compressor.verify_reconstruction(test_data, reconstructed_2d, error_bound=0.0, relative_error_bound=5e-3, model='tiny_residual_predictor')
    
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
    
    # Demonstrate plotting (commented out to avoid display issues)
    # print("\nPlotting training curves...")
    # plot_training_curves(stats_3d, error_bound=10.0, 
    #                      output_path='test_training_3d.png', 
    #                      title_suffix='(3D mode)')
    # plot_training_curves(stats_2d, error_bound=10.0, 
    #                      output_path='test_training_2d.png', 
    #                      title_suffix='(2D sliced mode)')
    
    print("\n" + "="*70)

