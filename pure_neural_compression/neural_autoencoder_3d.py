"""
Pure Neural Network 3D Autoencoder with 3 branches:
- Spatial Branch: Process 3D volumes with 3D convolutions
- Frequency Branch (Magnitude): Process magnitude of 3D FFT
- Frequency Branch (Phase): Process phase of 3D FFT

Designed for direct data compression (not residuals) with error bound enforcement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock3D(nn.Module):
    """Lightweight Residual Block for 3D convolutions."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Adjust channels if needed
        self.adjust_channels = None
        if in_channels != out_channels or stride != 1:
            self.adjust_channels = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)
        
        out += identity
        out = self.bn2(out)
        out = self.relu(out)
        return out


class SpatialBranch3D(nn.Module):
    """Spatial branch for processing 3D volumes."""
    
    def __init__(self, in_channels=1, base_channels=16):
        super(SpatialBranch3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Residual blocks with downsampling
        self.res_block1 = ResidualBlock3D(base_channels, base_channels * 2, stride=1)
        self.res_block2 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=1)
        
        self.output_channels = base_channels * 4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 512 -> 256
        
        x = self.res_block1(x)
        x = self.pool(x)  # 256 -> 128
        x = self.res_block2(x)
        x = self.pool(x)  # 128 -> 64
        
        return x


class FrequencyBranch3D(nn.Module):
    """Frequency branch for processing magnitude or phase of 3D FFT."""
    
    def __init__(self, in_channels=1, base_channels=8):
        super(FrequencyBranch3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Lighter residual blocks for frequency domain
        self.res_block1 = ResidualBlock3D(base_channels, base_channels * 2, stride=1)
        self.res_block2 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=1)
        
        self.output_channels = base_channels * 4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # Downsample
        
        x = self.res_block1(x)
        x = self.pool(x)
        x = self.res_block2(x)
        x = self.pool(x)
        
        return x


class NeuralEncoder3D(nn.Module):
    """
    Lightweight 3-branch encoder for direct data compression.
    
    Architecture:
    - Spatial Branch: Direct 3D convolutions on normalized data
    - Magnitude Branch: Process magnitude of 3D FFT
    - Phase Branch: Process phase of 3D FFT
    - Fusion Layer: Combine all three branches
    - Compression Head: Reduce to compact latent representation
    
    Target: ~500K-1M parameters for good compression/quality trade-off
    """
    
    def __init__(self, spatial_channels=16, freq_channels=8, latent_dim=2048):
        super(NeuralEncoder3D, self).__init__()
        
        # Three branches
        self.spatial_branch = SpatialBranch3D(in_channels=1, base_channels=spatial_channels)
        self.freq_mag_branch = FrequencyBranch3D(in_channels=1, base_channels=freq_channels)
        self.freq_phase_branch = FrequencyBranch3D(in_channels=1, base_channels=freq_channels)
        
        # Calculate total channels after branches
        total_channels = (self.spatial_branch.output_channels + 
                         self.freq_mag_branch.output_channels + 
                         self.freq_phase_branch.output_channels)
        
        # Fusion layer
        fusion_channels = total_channels // 2
        self.fusion_layer = nn.Sequential(
            nn.Conv3d(total_channels, fusion_channels, kernel_size=1),
            nn.BatchNorm3d(fusion_channels),
            nn.ReLU(),
            nn.Conv3d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(fusion_channels),
            nn.ReLU()
        )
        
        # Adaptive pooling to fixed size before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Compression head - reduce to latent vector
        flattened_size = fusion_channels * 4 * 4 * 4
        self.compression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x, return_stats=False):
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor of shape [batch, 1, D, H, W] (e.g., [B, 1, 512, 512, 512])
            return_stats: If True, return normalization statistics
        
        Returns:
            latent: Compressed latent representation [batch, latent_dim]
            stats: (optional) Dictionary with mean and std for denormalization
        """
        # Store original statistics for denormalization during decoding
        original_mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        original_std = x.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
        
        # Normalize input for better training stability
        x_normalized = (x - original_mean) / original_std
        
        # Spatial processing
        spatial_features = self.spatial_branch(x_normalized)
        
        # Frequency domain processing
        x_fft = torch.fft.rfftn(x_normalized, dim=(-3, -2, -1))
        
        # Convert to magnitude and phase
        freq_mag = torch.abs(x_fft)
        freq_phase = torch.angle(x_fft)
        
        # Normalize frequency components for stable training
        freq_mag = torch.log1p(freq_mag)  # Log scale for magnitude
        
        # Process each frequency component
        mag_features = self.freq_mag_branch(freq_mag)
        phase_features = self.freq_phase_branch(freq_phase)
        
        # Align all features to the same spatial size
        target_size = spatial_features.shape[-3:]
        mag_features = F.interpolate(mag_features, size=target_size, mode='trilinear', align_corners=False)
        phase_features = F.interpolate(phase_features, size=target_size, mode='trilinear', align_corners=False)
        
        # Combine all three branches
        combined_features = torch.cat([spatial_features, mag_features, phase_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        
        # Adaptive pooling to fixed size
        pooled_features = self.adaptive_pool(fused_features)
        
        # Compress to latent vector
        latent = self.compression_head(pooled_features)
        
        if return_stats:
            stats = {
                'mean': original_mean,
                'std': original_std
            }
            return latent, stats
        
        return latent
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralDecoder3D(nn.Module):
    """
    Decoder to reconstruct data from latent representation.
    """
    
    def __init__(self, latent_dim=2048, base_channels=64, output_shape=(512, 512, 512)):
        super(NeuralDecoder3D, self).__init__()
        
        self.output_shape = output_shape
        self.base_channels = base_channels
        
        # Initial upsampling from latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, base_channels * 8 * 8 * 8),
            nn.ReLU()
        )
        
        # Reshape layer
        self.initial_shape = (base_channels, 8, 8, 8)
        
        # Calculate channel progression
        ch1 = max(32, base_channels // 2)
        ch2 = max(16, base_channels // 4)
        ch3 = max(8, base_channels // 8)
        ch4 = 4
        
        # Upsampling layers
        self.decoder = nn.Sequential(
            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(base_channels, ch1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ch1),
            nn.ReLU(),
            
            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(ch1, ch2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ch2),
            nn.ReLU(),
            
            # 32x32x32 -> 64x64x64
            nn.ConvTranspose3d(ch2, ch3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ch3),
            nn.ReLU(),
            
            # 64x64x64 -> 128x128x128
            nn.ConvTranspose3d(ch3, ch4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ch4),
            nn.ReLU(),
            
            # Final conv to single channel
            nn.Conv3d(ch4, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, latent, mean=None, std=None):
        """
        Decode latent to data.
        
        Args:
            latent: Latent tensor [batch, latent_dim]
            mean: Original mean for denormalization
            std: Original std for denormalization
        
        Returns:
            reconstruction: Reconstructed data [batch, 1, D, H, W]
        """
        x = self.fc(latent)
        x = x.view(-1, *self.initial_shape)
        x = self.decoder(x)
        
        # Interpolate to exact output shape
        if x.shape[-3:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode='trilinear', align_corners=False)
        
        # Denormalize if statistics provided
        if mean is not None and std is not None:
            x = x * std + mean
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralAutoencoder3D(nn.Module):
    """
    Complete autoencoder for training and compression.
    Combines encoder and decoder with error bound awareness.
    """
    
    def __init__(self, spatial_channels=16, freq_channels=8, latent_dim=2048,
                 decoder_channels=64, output_shape=(512, 512, 512)):
        super(NeuralAutoencoder3D, self).__init__()
        
        self.encoder = NeuralEncoder3D(spatial_channels, freq_channels, latent_dim)
        self.decoder = NeuralDecoder3D(latent_dim, decoder_channels, output_shape)
        self.output_shape = output_shape
    
    def forward(self, x, error_bound=None):
        """
        Forward pass with optional error bound enforcement.
        
        Args:
            x: Input data [batch, 1, D, H, W]
            error_bound: Optional error bound for clipping
        
        Returns:
            reconstruction: Reconstructed data
            latent: Latent representation
        """
        # Encode with statistics
        latent, stats = self.encoder(x, return_stats=True)
        
        # Decode with denormalization
        reconstruction = self.decoder(latent, stats['mean'], stats['std'])
        
        # Apply error bound clipping if specified
        if error_bound is not None:
            error = reconstruction - x
            error_clipped = torch.clamp(error, -error_bound, error_bound)
            reconstruction = x + error_clipped
        
        return reconstruction, latent
    
    def count_parameters(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def compress(self, data_np):
        """
        Compress numpy array to latent representation.
        
        Args:
            data_np: Numpy array of shape (D, H, W)
        
        Returns:
            latent_np: Numpy array of latent codes
            stats: Dictionary with normalization statistics
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            data_tensor = torch.from_numpy(data_np).float().unsqueeze(0).unsqueeze(0)
            device = next(self.parameters()).device
            data_tensor = data_tensor.to(device)
            
            # Encode
            latent, stats = self.encoder(data_tensor, return_stats=True)
            
            # Convert to numpy
            latent_np = latent.cpu().numpy().squeeze()
            stats_np = {
                'mean': stats['mean'].cpu().numpy().squeeze(),
                'std': stats['std'].cpu().numpy().squeeze()
            }
            
        return latent_np, stats_np
    
    def decompress(self, latent_np, stats):
        """
        Decompress latent codes to data.
        
        Args:
            latent_np: Numpy array of latent codes
            stats: Dictionary with normalization statistics
        
        Returns:
            data_np: Reconstructed numpy array
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            latent_tensor = torch.from_numpy(latent_np).float().unsqueeze(0)
            device = next(self.parameters()).device
            latent_tensor = latent_tensor.to(device)
            
            mean = torch.tensor(stats['mean']).float().to(device)
            std = torch.tensor(stats['std']).float().to(device)
            
            # Ensure mean and std have correct shape
            if mean.dim() == 0:
                mean = mean.view(1, 1, 1, 1, 1)
            if std.dim() == 0:
                std = std.view(1, 1, 1, 1, 1)
            
            # Decode
            reconstruction = self.decoder(latent_tensor, mean, std)
            
            # Convert to numpy
            data_np = reconstruction.cpu().numpy().squeeze()
            
        return data_np


if __name__ == "__main__":
    # Test the model
    print("Testing NeuralAutoencoder3D...")
    
    # Create model
    model = NeuralAutoencoder3D(
        spatial_channels=16,
        freq_channels=8,
        latent_dim=2048,
        decoder_channels=64,
        output_shape=(512, 512, 512)
    )
    
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Encoder parameters: {model.encoder.count_parameters():,}")
    print(f"Decoder parameters: {model.decoder.count_parameters():,}")
    
    # Test with small input
    test_input = torch.randn(1, 1, 64, 64, 64)
    print(f"\nTest input shape: {test_input.shape}")
    
    reconstruction, latent = model(test_input, error_bound=0.01)
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test compression ratio
    input_size = np.prod(test_input.shape) * 4  # float32
    latent_size = latent.numel() * 4
    print(f"\nRaw compression ratio: {input_size / latent_size:.2f}x")
    
    # Test compress/decompress
    test_np = test_input.numpy().squeeze()
    latent_np, stats = model.compress(test_np)
    reconstructed_np = model.decompress(latent_np, stats)
    print(f"\nReconstruction error: {np.abs(test_np - reconstructed_np).max():.6e}")

