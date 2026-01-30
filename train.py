"""
Training module for NeurLZ compressor.

This module contains functions for:
- Model creation and initialization
- Single model training
- Dual model training (Scheme B)
- Residual prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import importlib.util

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
    from .utils import setup_multi_gpu_model, get_available_gpus
except ImportError:
    from utils import setup_multi_gpu_model, get_available_gpus

# -----------------------------------------------------------------------------
# IMPORTANT: force importing Patch_data.py from THIS directory.
#
# Reason:
# We also insert '/Users/923714256/Data_compression/neural_compression' into sys.path
# for Model imports, which contains another Patch_data.py without roi_mask support.
# That shadowing causes:
#   TypeError: create_hybrid_datasets() got an unexpected keyword argument 'roi_mask'
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_DATA_PATH = os.path.join(_THIS_DIR, "Patch_data.py")
_spec_pd = importlib.util.spec_from_file_location("patch_data_local", _PATCH_DATA_PATH)
_patch_data_local = importlib.util.module_from_spec(_spec_pd)
_spec_pd.loader.exec_module(_patch_data_local)

create_hybrid_datasets = _patch_data_local.create_hybrid_datasets
PatchDataLoader = _patch_data_local.PatchDataLoader
collate_patches_to_tensor = _patch_data_local.collate_patches_to_tensor
get_dataset_info = _patch_data_local.get_dataset_info

try:
    from .compression_function import compute_gn_groups
except ImportError:
    from compression_function import compute_gn_groups

# Import loss functions
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
BandedFrequencyLoss_4_inputs = BandWeightedSpectralLoss


# =============================================================================
# Model Creation
# =============================================================================

def create_model_and_criterion(model_type, model_channels, spatial_dims, num_res_blocks,
                               available_gpus, use_multi_gpu, primary_device, learning_rate):
    """
    Create model, criterion, and optimizer based on model type.
    
    Returns:
        model, criterion, optimizer, num_params, primary_device
    """
    if model_type == 'tiny_residual_predictor': 
        criterion = SpatialFrequencyLoss(
            weight_spatial=1.0,
            weight_magnitude=0,
            weight_phase=0,
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
            if torch.cuda.is_available():
                model = base_model.cuda()
                primary_device = 'cuda:0'
            else:
                model = base_model
                primary_device = 'cpu'
        model.to(primary_device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    elif model_type == 'tiny_frequency_residual_predictor_1_input':
        criterion = SpatialFrequencyLoss(
            weight_spatial=1.0,
            weight_magnitude=1,
            weight_phase=1,
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
            if torch.cuda.is_available():
                model = base_model.cuda()
                primary_device = 'cuda:0'
            else:
                model = base_model
                primary_device = 'cpu'
        model.to(primary_device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    elif model_type == 'tiny_frequency_residual_predictor_7_inputs':
        try:
            from Model import TinyFrequencyResidualPredictor_7_inputs
        except ImportError:
            print(f"Model {model_type} not available. Please check Model imports.")
            return None, None, None, None, None

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
            if torch.cuda.is_available():
                model = base_model.cuda()
                primary_device = 'cuda:0'
            else:
                model = base_model
                primary_device = 'cpu'
        model.to(primary_device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
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
            if torch.cuda.is_available():
                model = base_model.cuda()
                primary_device = 'cuda:0'
            else:
                model = base_model
                primary_device = 'cpu'
        model.to(primary_device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    elif model_type == 'tiny_frequency_residual_predictor_7_attn_roi':
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
        gn_groups_value = compute_gn_groups(model_channels, preferred=4)
        base_model = TinyFrequencyResidualPredictor7_AttnROI(
            channels=model_channels,
            spatial_dims=spatial_dims,
            num_res_blocks=num_res_blocks,
            low_cutoff=0.15,
            mid_cutoff=0.40,
            use_phase_sincos=True,
            return_roi_mask=False,
            gn_groups=gn_groups_value
        )
        if use_multi_gpu:
            model, primary_device = setup_multi_gpu_model(base_model, device_ids=available_gpus)
        else:
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
        return None, None, None, None, None
    
    # Count parameters
    if isinstance(model, nn.DataParallel):
        num_params = model.module.count_parameters()
    else:
        num_params = model.count_parameters()
    
    return model, criterion, optimizer, num_params, primary_device


# =============================================================================
# Single Model Training
# =============================================================================

# Dual Model Training
# =============================================================================

def train_dual_models(x_prime_norm, residuals_norm, roi_boxes_3d, model_type, model_channels,
                      spatial_dims, num_res_blocks, slice_order, online_epochs, learning_rate,
                      Patch_size, Batch_size, val_split, track_losses, verbose,
                      model_type_bg=None, model_type_roi=None,
                      roi_mask_training=None):
    """
    Train dual models (BG and ROI) for Scheme B.
    
    Args:
        model_type: Default model type (used if model_type_bg or model_type_roi not specified)
        model_type_bg: Model type for BG model (if None, uses model_type)
        model_type_roi: Model type for ROI model (if None, uses model_type)
    
    Returns:
        model_bg, model_roi, train_losses_bg, train_losses_roi, val_losses_bg, val_losses_roi,
        training_time, num_params_bg, num_params_roi, residual_mean, residual_std,
        input_mean, input_std, batch_size_2d, batch_size_3d, primary_device
    """
    # Use separate model types if provided, otherwise use default
    if model_type_bg is None:
        model_type_bg = model_type
    if model_type_roi is None:
        model_type_roi = model_type
    # Check available GPUs
    available_gpus = get_available_gpus()
    use_multi_gpu = len(available_gpus) > 1
    
    # Create ROI mask in training coordinate space (prefer exact mask if provided)
    # NOTE: keep this block strictly space-indented (no tabs) to avoid IndentationError.
    if roi_mask_training is None:
        roi_mask_training = np.zeros(x_prime_norm.shape, dtype=bool)

        # If roi_boxes_3d is empty/None, the mask stays all-False (meaning "no ROI")
        for (x0, x1, y0, y1, z0, z1) in (roi_boxes_3d or []):
            if spatial_dims == 2:
                if slice_order == 'zxy':
                    for z in range(z0, z1):
                        if z < x_prime_norm.shape[0]:
                            roi_mask_training[z, x0:x1, y0:y1] = True
                elif slice_order == 'yxz':
                    for y in range(y0, y1):
                        if y < x_prime_norm.shape[0]:
                            roi_mask_training[y, x0:x1, z0:z1] = True
                else:  # 'xyz'
                    for x in range(x0, x1):
                        if x < x_prime_norm.shape[0]:
                            roi_mask_training[x, y0:y1, z0:z1] = True
            else:
                roi_mask_training[x0:x1, y0:y1, z0:z1] = True

    # Prepare datasets
    batch_size_2d = 256
    batch_size_3d = 256
    
    # Create separate datasets using precise mask:
    # - BG model trains on patches that do NOT intersect ROI mask
    # - ROI model trains on patches that intersect ROI mask
    train_dataset_bg, val_dataset_bg = create_hybrid_datasets(
        x_data=x_prime_norm,
        y_data=residuals_norm,
        patch_size=Patch_size,
        overlap=16,
        spatial_dims=spatial_dims,
        val_split=val_split,
        seed=42,
        roi_mask=roi_mask_training,
        keep="bg",
    )
    train_dataset_roi, val_dataset_roi = create_hybrid_datasets(
        x_data=x_prime_norm,
        y_data=residuals_norm,
        patch_size=Patch_size,
        overlap=16,
        spatial_dims=spatial_dims,
        val_split=val_split,
        seed=42,
        roi_mask=roi_mask_training,
        keep="roi",
    )

    # Fallback: if one side gets no samples, fall back to all-data training for that side
    if len(train_dataset_bg) == 0:
        if verbose:
            print("  WARNING: BG dataset is empty after mask filtering; falling back to all patches for BG.")
        train_dataset_bg, val_dataset_bg = create_hybrid_datasets(
            x_data=x_prime_norm, y_data=residuals_norm,
            patch_size=Patch_size, overlap=16,
            spatial_dims=spatial_dims, val_split=val_split, seed=42
        )
    if len(train_dataset_roi) == 0:
        if verbose:
            print("  WARNING: ROI dataset is empty after mask filtering; falling back to all patches for ROI.")
        train_dataset_roi, val_dataset_roi = create_hybrid_datasets(
            x_data=x_prime_norm, y_data=residuals_norm,
            patch_size=Patch_size, overlap=16,
            spatial_dims=spatial_dims, val_split=val_split, seed=42
        )
    
    # Create BG model
    model_bg, criterion_bg, optimizer_bg, num_params_bg, primary_device = create_model_and_criterion(
        model_type_bg, model_channels, spatial_dims, num_res_blocks,
        available_gpus, use_multi_gpu, None, learning_rate
    )
    
    if model_bg is None:
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    # Create ROI model (can be different architecture)
    model_roi, criterion_roi, optimizer_roi, num_params_roi, _ = create_model_and_criterion(
        model_type_roi, model_channels, spatial_dims, num_res_blocks,
        available_gpus, use_multi_gpu, primary_device, learning_rate
    )
    
    if model_roi is None:
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    if verbose:
        print(f"  BG model: {model_type_bg}, parameters: {num_params_bg:,}")
        print(f"  ROI model: {model_type_roi}, parameters: {num_params_roi:,}")
        if model_type_bg != model_type_roi:
            print(f"  Using different architectures: BG={model_type_bg}, ROI={model_type_roi}")

    # Create data loaders
    train_loader_bg = PatchDataLoader(train_dataset_bg, batch_size=Batch_size, shuffle=True, drop_last=False)
    val_loader_bg = PatchDataLoader(val_dataset_bg, batch_size=Batch_size, shuffle=False, drop_last=False) if val_dataset_bg else None

    train_loader_roi = PatchDataLoader(train_dataset_roi, batch_size=Batch_size, shuffle=True, drop_last=False)
    val_loader_roi = PatchDataLoader(val_dataset_roi, batch_size=Batch_size, shuffle=False, drop_last=False) if val_dataset_roi else None
    
    # Training loop for both models
    train_losses_bg = []
    train_losses_roi = []
    val_losses_bg = []
    val_losses_roi = []
    training_start_time = time.time()
    
    for epoch in range(online_epochs):
        model_bg.train()
        epoch_train_losses_bg = []

        for batch_x, batch_y in train_loader_bg:
            x_tensor, y_tensor = collate_patches_to_tensor(
                batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
            )
                    
            optimizer_bg.zero_grad()
            pred = model_bg(x_tensor)
            
            if isinstance(pred, tuple):
                pred = pred[0]

            result = criterion_bg(pred, y_tensor)

            if isinstance(result, tuple):
                loss, loss_dict = result
            else:
                loss = result
                loss_dict = {"mse": float(loss.item())}

            loss.backward()
            optimizer_bg.step()
                                    
            epoch_train_losses_bg.append(loss.item())
                    
            del x_tensor, y_tensor, pred, loss, loss_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        # Train ROI model
        model_roi.train()
        epoch_train_losses_roi = []

        for batch_x, batch_y in train_loader_roi:
            x_tensor, y_tensor = collate_patches_to_tensor(
                batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
            )
            
            optimizer_roi.zero_grad()
            pred = model_roi(x_tensor)
            
            if isinstance(pred, tuple):
                pred = pred[0]

            result = criterion_roi(pred, y_tensor)

            if isinstance(result, tuple):
                loss, loss_dict = result
            else:
                loss = result
                loss_dict = {"mse": float(loss.item())}

            loss.backward()
            optimizer_roi.step()

            epoch_train_losses_roi.append(loss.item())

            del x_tensor, y_tensor, pred, loss, loss_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # BG Validation
        if track_losses:
            if val_loader_bg:
                model_bg.eval()
                epoch_val_losses_bg = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader_bg:
                        x_tensor, y_tensor = collate_patches_to_tensor(
                            batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
                        )
                        pred = model_bg(x_tensor)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        result = criterion_bg(pred, y_tensor)
                        if isinstance(result, tuple):
                            loss, loss_dict = result
                        else:
                            loss = result
                            loss_dict = {"mse": float(loss.item())}
                        epoch_val_losses_bg.append(loss.item())
                        del x_tensor, y_tensor, pred, loss, loss_dict
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                val_losses_bg.append(np.mean(epoch_val_losses_bg))
        
        # ROI Validation
        if track_losses and val_loader_roi:
            model_roi.eval()
            epoch_val_losses_roi = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader_roi:
                    x_tensor, y_tensor = collate_patches_to_tensor(
                        batch_x, batch_y, device=primary_device, spatial_dims=spatial_dims
                    )
                    pred = model_roi(x_tensor)
                    if isinstance(pred, tuple):
                        pred = pred[0]

                    result = criterion_roi(pred, y_tensor)

                    if isinstance(result, tuple):
                        loss, loss_dict = result
                    else:
                        loss = result
                        loss_dict = {"mse": float(loss.item())}
                    epoch_val_losses_roi.append(loss.item())
                    del x_tensor, y_tensor, pred, loss, loss_dict
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            val_losses_roi.append(np.mean(epoch_val_losses_roi))
            
        train_losses_bg.append(np.mean(epoch_train_losses_bg))
        train_losses_roi.append(np.mean(epoch_train_losses_roi))

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"    Epoch {epoch+1:3d}: BG Loss = {np.mean(epoch_train_losses_bg):.6f}"
            if val_losses_bg:
                msg += f", BG Val = {val_losses_bg[-1]:.6f}"
            msg += f" | ROI Loss = {np.mean(epoch_train_losses_roi):.6f}"
            if val_losses_roi:
                msg += f", ROI Val = {val_losses_roi[-1]:.6f}"
            print(msg)
    
    training_time = time.time() - training_start_time
    
    # Note: normalization stats should be computed from original data, not normalized data
    # These will be passed back but should be computed in compressor.py from true_residuals_for_training
    
    return (model_bg, model_roi, train_losses_bg, train_losses_roi, val_losses_bg, val_losses_roi,
            training_time, num_params_bg, num_params_roi, batch_size_2d, batch_size_3d, primary_device)


# =============================================================================
# Prediction Functions
# =============================================================================


def predict_residuals_dual(model_bg, model_roi, x_prime_norm, roi_boxes_3d,
                         residual_mean, residual_std, spatial_dims, slice_order,
                         batch_size_2d, batch_size_3d, primary_device,
                         roi_mask_pred=None):
    """
    Predict residuals using dual models (BG and ROI).
    
    Returns:
        pred_residuals_np: Denormalized predicted residuals in original coordinate space
    """
    model_bg.eval()
    model_roi.eval()

    # Dual-model path: prefer using provided ROI mask (exact mask mapped to training space).
    # If not provided, fall back to boxes->mask mapping.
    if roi_mask_pred is None:
        roi_mask_pred = np.zeros(x_prime_norm.shape, dtype=bool)
        for (x0, x1, y0, y1, z0, z1) in (roi_boxes_3d or []):
            if spatial_dims == 2:
                if slice_order == 'zxy':
                    for z in range(z0, z1):
                        if z < x_prime_norm.shape[0]:
                            roi_mask_pred[z, x0:x1, y0:y1] = True
                elif slice_order == 'yxz':
                    for y in range(y0, y1):
                        if y < x_prime_norm.shape[0]:
                            roi_mask_pred[y, x0:x1, z0:z1] = True
                else:  # 'xyz'
                    for x in range(x0, x1):
                        if x < x_prime_norm.shape[0]:
                            roi_mask_pred[x, y0:y1, z0:z1] = True
            else:
                roi_mask_pred[x0:x1, y0:y1, z0:z1] = True

    pred_residuals_bg_list = []
    pred_residuals_roi_list = []

    with torch.no_grad():
        if spatial_dims == 2:
            n_slices = x_prime_norm.shape[0]
            for batch_start in range(0, n_slices, batch_size_2d):
                batch_end = min(batch_start + batch_size_2d, n_slices)
                x_batch = torch.from_numpy(x_prime_norm[batch_start:batch_end]).float().unsqueeze(1).to(primary_device)

                # BG prediction 
                pred_bg = model_bg(x_batch)
                if isinstance(pred_bg, tuple):
                    pred_bg = pred_bg[0]
                pred_bg_np = pred_bg.cpu().numpy().squeeze(1)
                pred_residuals_bg_list.append(pred_bg_np)

                # ROI prediction
                pred_roi = model_roi(x_batch)
                if isinstance(pred_roi, tuple):
                    pred_roi = pred_roi[0]
                pred_roi_np = pred_roi.cpu().numpy().squeeze(1)
                pred_residuals_roi_list.append(pred_roi_np)
                
                del x_batch, pred_bg, pred_roi
                torch.cuda.empty_cache()
            
            pred_residuals_bg_norm = np.concatenate(pred_residuals_bg_list, axis=0)
            pred_residuals_roi_norm = np.concatenate(pred_residuals_roi_list, axis=0)
        
        else:  # 3D
            z_dim = x_prime_norm.shape[2]
            for batch_start in range(0, z_dim, batch_size_3d):
                batch_end = min(batch_start + batch_size_3d, z_dim)
                x_batch = x_prime_norm[:, :, batch_start:batch_end]
                x_batch_tensor = torch.from_numpy(x_batch).float().unsqueeze(0).unsqueeze(0).to(primary_device)
                
                # BG prediction
                pred_bg = model_bg(x_batch_tensor)
                if isinstance(pred_bg, tuple):
                    pred_bg = pred_bg[0]
                pred_bg_np = pred_bg.cpu().numpy().squeeze()
                pred_residuals_bg_list.append(pred_bg_np)
                
                # ROI prediction
                pred_roi = model_roi(x_batch_tensor)
                if isinstance(pred_roi, tuple):
                    pred_roi = pred_roi[0]
                pred_roi_np = pred_roi.cpu().numpy().squeeze()
                pred_residuals_roi_list.append(pred_roi_np)
                
                del x_batch_tensor, pred_bg, pred_roi
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            pred_residuals_bg_norm = np.concatenate(pred_residuals_bg_list, axis=2)
            pred_residuals_roi_norm = np.concatenate(pred_residuals_roi_list, axis=2)

    # Denormalize
    pred_residuals_bg_np = pred_residuals_bg_norm * residual_std + residual_mean
    pred_residuals_roi_np = pred_residuals_roi_norm * residual_std + residual_mean

    # Transpose back if 2D
    if spatial_dims == 2:
        if slice_order == 'zxy':
            pred_residuals_bg_np = pred_residuals_bg_np.transpose(1, 2, 0)
            pred_residuals_roi_np = pred_residuals_roi_np.transpose(1, 2, 0)
            roi_mask_pred = roi_mask_pred.transpose(1, 2, 0)
        elif slice_order == 'yxz':
            pred_residuals_bg_np = pred_residuals_bg_np.transpose(1, 0, 2)
            pred_residuals_roi_np = pred_residuals_roi_np.transpose(1, 0, 2)
            roi_mask_pred = roi_mask_pred.transpose(1, 0, 2)
    
    # Combine: R_hat = R_hat_bg * (1 - M_roi) + R_hat_roi * M_roi
    roi_mask_float = roi_mask_pred.astype(np.float32)
    pred_residuals_np = pred_residuals_bg_np * (1 - roi_mask_float) + pred_residuals_roi_np * roi_mask_float
    
    return pred_residuals_np

