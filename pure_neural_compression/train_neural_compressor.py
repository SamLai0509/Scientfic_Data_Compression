"""
Training script for Pure Neural Compressor.

Trains a 3-branch autoencoder on scientific data with:
- Error-bound-aware loss function
- Multi-GPU support
- Data augmentation
- Comprehensive logging
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm

from neural_autoencoder_3d import NeuralAutoencoder3D


class VolumeDataset(Dataset):
    """
    Dataset for loading 3D scientific data volumes.
    """
    
    def __init__(self, data_dir, file_list, shape=(512, 512, 512), 
                 normalize=True, augment=False):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing data files
            file_list: List of file names to load
            shape: Expected data shape
            normalize: Whether to normalize data
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.shape = shape
        self.normalize = normalize
        self.augment = augment
        
        # Load all volumes into memory if possible
        self.volumes = []
        self.stats = []
        
        for filename in tqdm(file_list, desc="Loading volumes"):
            filepath = self.data_dir / filename
            if filepath.exists():
                volume = np.fromfile(filepath, dtype=np.float32).reshape(shape)
                
                # Store statistics
                stats = {
                    'min': float(np.min(volume)),
                    'max': float(np.max(volume)),
                    'mean': float(np.mean(volume)),
                    'std': float(np.std(volume))
                }
                
                self.volumes.append(volume)
                self.stats.append(stats)
            else:
                print(f"Warning: {filepath} not found")
        
        print(f"Loaded {len(self.volumes)} volumes")
    
    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx):
        volume = self.volumes[idx].copy()
        
        # Data augmentation: random flips
        if self.augment:
            for axis in range(3):
                if np.random.random() > 0.5:
                    volume = np.flip(volume, axis=axis).copy()
        
        # Convert to tensor and add channel dimension
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)
        
        return volume_tensor


class PatchDataset(Dataset):
    """
    Dataset that extracts random patches from volumes for training.
    Useful for large volumes that don't fit in GPU memory.
    """
    
    def __init__(self, data_dir, file_list, volume_shape=(512, 512, 512),
                 patch_shape=(128, 128, 128), patches_per_volume=10,
                 normalize=True, augment=True):
        """
        Initialize patch dataset.
        
        Args:
            data_dir: Directory containing data files
            file_list: List of file names
            volume_shape: Shape of full volumes
            patch_shape: Shape of extracted patches
            patches_per_volume: Number of patches to extract per volume per epoch
            normalize: Whether to normalize patches
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.volume_shape = volume_shape
        self.patch_shape = patch_shape
        self.patches_per_volume = patches_per_volume
        self.normalize = normalize
        self.augment = augment
        
        # Load volumes
        self.volumes = []
        for filename in tqdm(file_list, desc="Loading volumes"):
            filepath = self.data_dir / filename
            if filepath.exists():
                volume = np.fromfile(filepath, dtype=np.float32).reshape(volume_shape)
                self.volumes.append(volume)
        
        print(f"Loaded {len(self.volumes)} volumes for patch extraction")
    
    def __len__(self):
        return len(self.volumes) * self.patches_per_volume
    
    def __getitem__(self, idx):
        # Determine which volume and which patch
        volume_idx = idx // self.patches_per_volume
        volume = self.volumes[volume_idx]
        
        # Extract random patch
        d, h, w = self.volume_shape
        pd, ph, pw = self.patch_shape
        
        # Random starting position
        start_d = np.random.randint(0, d - pd + 1)
        start_h = np.random.randint(0, h - ph + 1)
        start_w = np.random.randint(0, w - pw + 1)
        
        patch = volume[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw].copy()
        
        # Augmentation
        if self.augment:
            for axis in range(3):
                if np.random.random() > 0.5:
                    patch = np.flip(patch, axis=axis).copy()
        
        # Convert to tensor
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
        
        return patch_tensor


def error_bound_loss(reconstruction, original, error_bound, lambda_eb=1.0):
    """
    Loss function with error bound penalty.
    
    Args:
        reconstruction: Reconstructed data
        original: Original data
        error_bound: Maximum allowed error
        lambda_eb: Weight for error bound violation penalty
    
    Returns:
        loss: Combined loss
        metrics: Dictionary with loss components
    """
    # MSE loss
    mse = torch.mean((reconstruction - original) ** 2)
    
    # Error bound violation penalty
    error = torch.abs(reconstruction - original)
    violations = torch.relu(error - error_bound)  # Only penalize violations
    eb_penalty = torch.mean(violations ** 2)
    
    # Combined loss
    loss = mse + lambda_eb * eb_penalty
    
    metrics = {
        'mse': mse.item(),
        'eb_penalty': eb_penalty.item(),
        'total_loss': loss.item(),
        'max_error': error.max().item(),
        'mean_error': error.mean().item()
    }
    
    return loss, metrics


def train_epoch(model, dataloader, optimizer, device, error_bound=None, lambda_eb=1.0):
    """
    Train for one epoch.
    
    Args:
        model: NeuralAutoencoder3D model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        error_bound: Error bound (if None, use MSE only)
        lambda_eb: Weight for error bound penalty
    
    Returns:
        epoch_stats: Dictionary with epoch statistics
    """
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    total_eb_penalty = 0.0
    total_max_error = 0.0
    total_mean_error = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        data = batch.to(device)
        
        # Forward pass - NO error bound clipping
        reconstruction, latent = model(data)
        
        # Compute loss (simple MSE for now)
        mse = torch.mean((reconstruction - data) ** 2)
        loss = mse
        
        # Compute error metrics for monitoring
        error = torch.abs(reconstruction - data)
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        # Build metrics dictionary
        metrics = {
            'mse': mse.item(),
            'total_loss': loss.item(),
            'max_error': max_error,
            'mean_error': mean_error
        }
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update statistics
        total_loss += metrics['total_loss']
        total_mse += metrics['mse']
        total_max_error += metrics['max_error']
        total_mean_error += metrics['mean_error']
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.2e}",
            'mse': f"{metrics['mse']:.2e}",
            'max_err': f"{metrics['max_error']:.2e}"
        })
    
    epoch_stats = {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'max_error': total_max_error / n_batches,
        'mean_error': total_mean_error / n_batches,
    }
    
    return epoch_stats


def validate(model, dataloader, device, error_bound=None):
    """
    Validate model.
    
    Args:
        model: NeuralAutoencoder3D model
        dataloader: Validation data loader
        device: Device
        error_bound: Error bound (optional, for monitoring only)
    
    Returns:
        val_stats: Dictionary with validation statistics
    """
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_max_error = 0.0
    total_mean_error = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            data = batch.to(device)
            
            # Forward pass - NO error bound clipping
            reconstruction, latent = model(data)
            
            # Compute metrics on ACTUAL model output
            mse = torch.mean((reconstruction - data) ** 2)
            error = torch.abs(reconstruction - data)
            max_error = error.max().item()
            mean_error = error.mean().item()
            
            total_loss += mse.item()
            total_mse += mse.item()
            total_max_error += max_error
            total_mean_error += mean_error
            n_batches += 1
    
    val_stats = {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'max_error': total_max_error / n_batches,
        'mean_error': total_mean_error / n_batches,
    }
    
    return val_stats


def main():
    parser = argparse.ArgumentParser(description="Train Pure Neural Compressor")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--train_files', type=str, nargs='+', required=True,
                        help='List of training file names')
    parser.add_argument('--val_files', type=str, nargs='+', default=None,
                        help='List of validation file names (optional)')
    parser.add_argument('--data_shape', type=int, nargs=3, default=[512, 512, 512],
                        help='Shape of data volumes')
    
    # Model parameters
    parser.add_argument('--spatial_channels', type=int, default=16,
                        help='Base channels for spatial branch')
    parser.add_argument('--freq_channels', type=int, default=8,
                        help='Base channels for frequency branches')
    parser.add_argument('--latent_dim', type=int, default=2048,
                        help='Latent dimension')
    parser.add_argument('--decoder_channels', type=int, default=64,
                        help='Base channels for decoder')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (typically 1 for 512^3 volumes)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--error_bound', type=float, default=None,
                        help='Error bound for training (optional)')
    parser.add_argument('--lambda_eb', type=float, default=1.0,
                        help='Weight for error bound penalty')
    
    # Patch-based training (for large volumes)
    parser.add_argument('--use_patches', action='store_true',
                        help='Use patch-based training')
    parser.add_argument('--patch_shape', type=int, nargs=3, default=[128, 128, 128],
                        help='Patch shape for patch-based training')
    parser.add_argument('--patches_per_volume', type=int, default=10,
                        help='Number of patches per volume per epoch')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Output directory for models and logs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs if available')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    # Create datasets
    if args.use_patches:
        print(f"Using patch-based training with patches of shape {args.patch_shape}")
        train_dataset = PatchDataset(
            args.data_dir,
            args.train_files,
            volume_shape=tuple(args.data_shape),
            patch_shape=tuple(args.patch_shape),
            patches_per_volume=args.patches_per_volume,
            augment=True
        )
        output_shape = tuple(args.patch_shape)
    else:
        train_dataset = VolumeDataset(
            args.data_dir,
            args.train_files,
            shape=tuple(args.data_shape),
            augment=True
        )
        output_shape = tuple(args.data_shape)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Validation dataset
    val_loader = None
    if args.val_files is not None:
        if args.use_patches:
            val_dataset = PatchDataset(
                args.data_dir,
                args.val_files,
                volume_shape=tuple(args.data_shape),
                patch_shape=tuple(args.patch_shape),
                patches_per_volume=args.patches_per_volume // 2,
                augment=False
            )
        else:
            val_dataset = VolumeDataset(
                args.data_dir,
                args.val_files,
                shape=tuple(args.data_shape),
                augment=False
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    # Create model
    model = NeuralAutoencoder3D(
        spatial_channels=args.spatial_channels,
        freq_channels=args.freq_channels,
        latent_dim=args.latent_dim,
        decoder_channels=args.decoder_channels,
        output_shape=output_shape
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = DataParallel(model)
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_mse': [],
        'val_loss': [],
        'val_mse': [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            error_bound=args.error_bound,
            lambda_eb=args.lambda_eb
        )
        
        print(f"Train Loss: {train_stats['loss']:.6f}, MSE: {train_stats['mse']:.6f}")
        
        history['train_loss'].append(train_stats['loss'])
        history['train_mse'].append(train_stats['mse'])
        
        # Validate
        if val_loader is not None:
            val_stats = validate(model, val_loader, device, args.error_bound)
            print(f"Val Loss: {val_stats['loss']:.6f}, MSE: {val_stats['mse']:.6f}")
            
            history['val_loss'].append(val_stats['loss'])
            history['val_mse'].append(val_stats['mse'])
            
            # Update learning rate
            scheduler.step(val_stats['loss'])
            
            # Save best model
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_stats['loss'],
                    'config': config
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'config': config
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save final model
    checkpoint = {
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    torch.save(checkpoint, output_dir / 'final_model.pth')
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()

