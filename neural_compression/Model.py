import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection: out = F(x) + x
    """
    def __init__(self, channels, spatial_dims=3):
        super(ResidualBlock, self).__init__()
        
        if spatial_dims == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3")
        
        self.conv1 = Conv(channels, channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(channels)
        self.conv2 = Conv(channels, channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(channels)
        
    def forward(self, x):
        identity = x

        out = torch.nn.functional.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Residual connection
        out = out + identity
        out = torch.nn.functional.gelu(out)
        
        return out


class TinyResidualPredictor(nn.Module):
    """
    Tiny ResUNet for predicting residuals - supports both 2D and 3D.
    
    Uses residual blocks at each level for better gradient flow.
    ~5k-25k parameters depending on configuration.
    Takes SZ3-decompressed data, predicts residual = original - decompressed.
    """
    
    def __init__(self, channels=4, mode='strict', spatial_dims=3, num_res_blocks=1):
        """
        Args:
            channels: Base channel count (4 → ~15k params for 3D, ~8k for 2D)
            mode: 'strict' (1× bound) or 'relaxed' (2× bound with Sigmoid)
            spatial_dims: 2 for 2D UNet, 3 for 3D UNet
            num_res_blocks: Number of residual blocks per level (1-3)
        """
        super(TinyResidualPredictor, self).__init__()
        self.spatial_dims = spatial_dims
        self.num_res_blocks = num_res_blocks
        
        # Select conv operations based on spatial dimensions
        if spatial_dims == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        # ============================================
        # Encoder Path (Downsampling)
        # ============================================
        
        # Initial convolution: 1 → channels
        self.init_conv = Conv(1, channels, kernel_size=3, padding=1)
        self.init_bn = BatchNorm(channels)
        
        # Encoder Level 1: Residual blocks at resolution 1×
        self.enc1_res_blocks = nn.ModuleList([
            ResidualBlock(channels, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # Downsample with MaxPool: channels → channels*2
        self.down_conv = Conv(channels, channels*2, kernel_size=3, padding=1)
        self.down_bn = BatchNorm(channels*2)
        self.down_pool = MaxPool(kernel_size=2, stride=2)
        
        # Encoder Level 2: Residual blocks at resolution 0.5×
        self.enc2_res_blocks = nn.ModuleList([
            ResidualBlock(channels*2, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Bottleneck
        # ============================================
        self.bottleneck_res_blocks = nn.ModuleList([
            ResidualBlock(channels*2, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Decoder Path (Upsampling)
        # ============================================
        
        # Upsample: channels*2 → channels
        # Using kernel_size=2, stride=2, padding=0 for proper dimension restoration
        self.up1 = ConvTranspose(channels*2, channels, kernel_size=2, stride=2, padding=0)
        self.up1_bn = BatchNorm(channels)
        
        # Fusion after skip connection: channels (decoder) + channels (encoder) → channels
        self.fusion_conv = Conv(channels*2, channels, kernel_size=1)
        self.fusion_bn = BatchNorm(channels)
        
        # Decoder Level 1: Residual blocks at resolution 1×
        self.dec1_res_blocks = nn.ModuleList([
            ResidualBlock(channels, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Output Layer
        # ============================================
        self.out = Conv(channels, 1, kernel_size=3, padding=1)
    
    def forward(self, x_decompressed, error_bound=None):
        """
        Forward pass: predict residuals from SZ3-decompressed data.
        
        Args:
            x_decompressed: SZ3-decompressed data
                - For 3D: [B, 1, H, W, D] or [H, W, D]
                - For 2D: [B, 1, H, W] or [H, W]
            error_bound: Error bound for relaxed mode (optional)
        
        Returns:
            residual: Predicted residuals (same shape as input)
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(x_decompressed, torch.Tensor):
            x_decompressed = torch.from_numpy(x_decompressed).float()
        
        # Add batch and channel dimensions if missing
        if self.spatial_dims == 3:
            if x_decompressed.ndim == 3:
                x_decompressed = x_decompressed.unsqueeze(0).unsqueeze(0)
            elif x_decompressed.ndim == 4:
                x_decompressed = x_decompressed.unsqueeze(1)
        elif self.spatial_dims == 2:
            if x_decompressed.ndim == 2:
                # Single 2D slice: (H, W) -> (1, 1, H, W)
                x_decompressed = x_decompressed.unsqueeze(0).unsqueeze(0)
            elif x_decompressed.ndim == 3:
                # Multiple 2D slices: (N, H, W) -> (N, 1, H, W)
                x_decompressed = x_decompressed.unsqueeze(1)
            # If ndim == 4, it's already correct (batch, channels, H, W) for Conv2d - don't modify!
            # DO NOT add any unsqueeze here!
        
        # Move to the same device as the model
        params = list(self.parameters())
        device = params[0].device if params else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_decompressed = x_decompressed.to(device)
        
        # Normalization
        # x_mean = x_decompressed.mean()
        # x_std = x_decompressed.std() + 1e-8
        # x_norm = (x_decompressed - x_mean) / x_std
        x_norm = x_decompressed
        # ============================================
        # Encoder Path
        # ============================================
        
        # Initial convolution
        x = torch.nn.functional.gelu(self.init_bn(self.init_conv(x_norm)))
        
        # Encoder Level 1 with residual blocks
        enc1 = x
        for res_block in self.enc1_res_blocks:
            enc1 = res_block(enc1)
        
        # Downsample with MaxPool
        x = self.down_pool(enc1)  # First pool to reduce spatial dimensions
        x = torch.nn.functional.gelu(self.down_bn(self.down_conv(x)))  # Then conv
        
        # Encoder Level 2 with residual blocks
        enc2 = x
        for res_block in self.enc2_res_blocks:
            enc2 = res_block(enc2)
        
        # ============================================
        # Bottleneck with residual blocks
        # ============================================
        bottleneck = enc2
        for res_block in self.bottleneck_res_blocks:
            bottleneck = res_block(bottleneck)
        
        # ============================================
        # Decoder Path
        # ============================================
        
        # Upsample
        dec1 = torch.nn.functional.gelu(self.up1_bn(self.up1(bottleneck)))
        
        # Skip connection: concatenate with enc1
        dec1_skip = torch.cat([dec1, enc1], dim=1)
        
        # Fusion convolution
        dec1 = torch.nn.functional.gelu(self.fusion_bn(self.fusion_conv(dec1_skip)))
        
        # Decoder Level 1 with residual blocks
        for res_block in self.dec1_res_blocks:
            dec1 = res_block(dec1)
        
        # ============================================
        # Output
        # ============================================
        residual = self.out(dec1)
        
        return residual
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

class TinyFrequencyResidualPredictor(nn.Module):
    """
    Frequency-aware residual predictor using spatial + FFT features with U-Net structure.
    
    Processes three channels:
    1. Spatial domain (original decompressed data)
    2. Magnitude spectrum (FFT)
    3. Phase spectrum (FFT)
    
    Uses residual blocks and U-Net architecture similar to TinyResidualPredictor.
    Supports both 2D and 3D data.
    ~10-25k parameters depending on configuration.
    """
    
    def __init__(self, channels=4, mode='strict', spatial_dims=3, num_res_blocks=1):
        """
        Args:
            channels: Base channel count (4 → ~10-25k params with 3 input channels)
            mode: 'strict' (1× bound) or 'relaxed' (2× bound with Sigmoid)
            spatial_dims: 2 for 2D UNet, 3 for 3D UNet
            num_res_blocks: Number of residual blocks per level (1-3)
        """
        super(TinyFrequencyResidualPredictor, self).__init__()
        self.spatial_dims = spatial_dims
        self.num_res_blocks = num_res_blocks
        
        # Select conv operations based on spatial dimensions
        if spatial_dims == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        # ============================================
        # Initial Encoders for Each Branch
        # ============================================
        # Spatial encoder
        self.spatial_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.spatial_bn1 = BatchNorm(channels)
        
        # Magnitude encoder (from FFT)
        self.mag_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.mag_bn1 = BatchNorm(channels)
        
        # Phase encoder (from FFT)
        self.phase_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.phase_bn1 = BatchNorm(channels)
        
        # ============================================
        # Fusion: Combine 3 branches → channels
        # ============================================
        self.fusion_conv = Conv(channels*3, channels, kernel_size=1)
        self.fusion_bn = BatchNorm(channels)
        
        # ============================================
        # Encoder Path (Downsampling) with Residual Blocks
        # ============================================
        
        # Encoder Level 1: Residual blocks at resolution 1×
        self.enc1_res_blocks = nn.ModuleList([
            ResidualBlock(channels, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # Downsample with MaxPool: channels → channels*2
        self.down_conv = Conv(channels, channels*2, kernel_size=3, padding=1)
        self.down_bn = BatchNorm(channels*2)
        self.down_pool = MaxPool(kernel_size=2, stride=2)
        
        # Encoder Level 2: Residual blocks at resolution 0.5×
        self.enc2_res_blocks = nn.ModuleList([
            ResidualBlock(channels*2, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Bottleneck with Residual Blocks
        # ============================================
        self.bottleneck_res_blocks = nn.ModuleList([
            ResidualBlock(channels*2, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Decoder Path (Upsampling) with Residual Blocks
        # ============================================
        
        # Upsample: channels*2 → channels
        self.up1 = ConvTranspose(channels*2, channels, kernel_size=2, stride=2, padding=0)
        self.up1_bn = BatchNorm(channels)
        
        # Fusion after skip connection: channels (decoder) + channels (encoder) → channels
        self.dec_fusion_conv = Conv(channels*2, channels, kernel_size=1)
        self.dec_fusion_bn = BatchNorm(channels)
        
        # Decoder Level 1: Residual blocks at resolution 1×
        self.dec1_res_blocks = nn.ModuleList([
            ResidualBlock(channels, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Output Layer
        # ============================================
        self.out = Conv(channels, 1, kernel_size=3, padding=1)
    
    def compute_fft_features(self, x):
        """
        Compute magnitude and phase from FFT (2D or 3D).
        
        Args:
            x: Input tensor 
                - For 3D: [B, 1, H, W, D]
                - For 2D: [B, 1, H, W]
        
        Returns:
            magnitude: FFT magnitude (same shape as input)
            phase: FFT phase (same shape as input)
        """
        if self.spatial_dims == 3:
            # Compute 3D FFT
            fft_result = torch.fft.fftn(x, dim=(-3, -2, -1))
        else:  # 2D
            # Compute 2D FFT
            fft_result = torch.fft.fftn(x, dim=(-2, -1))
        
        # Extract magnitude and phase
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)
        
        # Log-scale magnitude for better dynamic range
        magnitude = torch.log1p(magnitude)
        
        return magnitude, phase
    
    def forward(self, x_decompressed, error_bound=None):
        """
        Forward pass: predict residuals using spatial + frequency features with U-Net structure.
        
        Args:
            x_decompressed: SZ3-decompressed data
                - For 3D: [B, 1, H, W, D] or [H, W, D]
                - For 2D: [B, 1, H, W] or [H, W]
            error_bound: Error bound for relaxed mode (optional)
        
        Returns:
            residual: Predicted residuals (same shape as input)
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(x_decompressed, torch.Tensor):
            x_decompressed = torch.from_numpy(x_decompressed).float()
        
        # Add batch and channel dimensions if missing
        if self.spatial_dims == 3:
            if x_decompressed.ndim == 3:
                x_decompressed = x_decompressed.unsqueeze(0).unsqueeze(0)
            elif x_decompressed.ndim == 4:
                x_decompressed = x_decompressed.unsqueeze(1)
        elif self.spatial_dims == 2:
            if x_decompressed.ndim == 2:
                # Single 2D slice: (H, W) -> (1, 1, H, W)
                x_decompressed = x_decompressed.unsqueeze(0).unsqueeze(0)
            elif x_decompressed.ndim == 3:
                # Multiple 2D slices: (N, H, W) -> (N, 1, H, W)
                x_decompressed = x_decompressed.unsqueeze(1)
            # If ndim == 4, it's already correct (batch, channels, H, W) for Conv2d - don't modify!
            # DO NOT add any unsqueeze here!
        
        # Move to the same device as the model
        params = list(self.parameters())
        device = params[0].device if params else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_decompressed = x_decompressed.to(device)
        
        # Normalize spatial input
        # x_mean = x_decompressed.mean()
        # x_std = x_decompressed.std() + 1e-8
        # x_norm = (x_decompressed - x_mean) / x_std
        x_norm = x_decompressed
        
        # Compute FFT features
        magnitude, phase = self.compute_fft_features(x_norm)
        
        # Normalize magnitude and phase
        mag_norm = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
        phase_norm = phase / (np.pi)  # Normalize to [-1, 1]
        
        # ============================================
        # Process Each Branch Separately
        # ============================================
        spatial_feat = torch.nn.functional.gelu(self.spatial_bn1(self.spatial_conv1(x_norm)))
        mag_feat = torch.nn.functional.gelu(self.mag_bn1(self.mag_conv1(mag_norm)))
        phase_feat = torch.nn.functional.gelu(self.phase_bn1(self.phase_conv1(phase_norm)))
        
        # Fuse all features: 3*channels → channels
        fused = torch.cat([spatial_feat, mag_feat, phase_feat], dim=1)
        fused = torch.nn.functional.gelu(self.fusion_bn(self.fusion_conv(fused)))
        
        # ============================================
        # Encoder Path with Residual Blocks
        # ============================================
        
        # Encoder Level 1 with residual blocks
        enc1 = fused
        for res_block in self.enc1_res_blocks:
            enc1 = res_block(enc1)
        
        # Downsample with MaxPool
        x = self.down_pool(enc1)  # First pool to reduce spatial dimensions
        x = torch.nn.functional.gelu(self.down_bn(self.down_conv(x)))  # Then conv
        
        # Encoder Level 2 with residual blocks
        enc2 = x
        for res_block in self.enc2_res_blocks:
            enc2 = res_block(enc2)
        
        # ============================================
        # Bottleneck with Residual Blocks
        # ============================================
        bottleneck = enc2
        for res_block in self.bottleneck_res_blocks:
            bottleneck = res_block(bottleneck)
        
        # ============================================
        # Decoder Path with Skip Connection
        # ============================================
        
        # Upsample
        dec1 = torch.nn.functional.gelu(self.up1_bn(self.up1(bottleneck)))
        
        # Skip connection: concatenate with enc1
        dec1_skip = torch.cat([dec1, enc1], dim=1)
        
        # Fusion convolution
        dec1 = torch.nn.functional.gelu(self.dec_fusion_bn(self.dec_fusion_conv(dec1_skip)))
        
        # Decoder Level 1 with residual blocks
        for res_block in self.dec1_res_blocks:
            dec1 = res_block(dec1)
        
        # ============================================
        # Output
        # ============================================
        residual = self.out(dec1)
        
        # Denormalize residual
        # residual = residual * x_std
        
        return residual
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

class TinyPhysicsResidualPredictor(nn.Module):
    """
    Physics-aware residual predictor for dark matter density using U-Net structure.
    
    Processes four channels:
    1. Spatial domain (original decompressed data)
    2. Gradient magnitude (|∇ρ|) - structure boundaries
    3. Laplacian (∇²ρ) - gravitational structure (Poisson equation)
    4. Density contrast (δ = (ρ - ρ̄)/ρ̄) - overdensity field
    
    Uses residual blocks and U-Net architecture similar to TinyFrequencyResidualPredictor.
    Supports both 2D and 3D data.
    ~12-30k parameters depending on configuration.
    """
    
    def __init__(self, channels=4, mode='strict', spatial_dims=3, num_res_blocks=1):
        """
        Args:
            channels: Base channel count (4 → ~12-30k params with 4 input channels)
            mode: 'strict' (1× bound) or 'relaxed' (2× bound with Sigmoid)
            spatial_dims: 2 for 2D UNet, 3 for 3D UNet
            num_res_blocks: Number of residual blocks per level (1-3)
        """
        super(TinyPhysicsResidualPredictor, self).__init__()
        self.spatial_dims = spatial_dims
        self.num_res_blocks = num_res_blocks
        self.mode = mode
        
        # Select conv operations based on spatial dimensions
        if spatial_dims == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        # ============================================
        # Initial Encoders for Each Physics Branch
        # ============================================
        # Spatial encoder (original decompressed data)
        self.spatial_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.spatial_bn1 = BatchNorm(channels)
        
        # Gradient encoder (|∇ρ|)
        self.gradient_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.gradient_bn1 = BatchNorm(channels)
        
        # Laplacian encoder (∇²ρ)
        self.laplacian_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.laplacian_bn1 = BatchNorm(channels)
        
        # Density contrast encoder (δ)
        self.contrast_conv1 = Conv(1, channels, kernel_size=3, padding=1)
        self.contrast_bn1 = BatchNorm(channels)
        
        # ============================================
        # Fusion: Combine 4 branches → channels
        # ============================================
        self.fusion_conv = Conv(channels*4, channels, kernel_size=1)
        self.fusion_bn = BatchNorm(channels)
        
        # ============================================
        # Encoder Path (Downsampling) with Residual Blocks
        # ============================================
        
        # Encoder Level 1: Residual blocks at resolution 1×
        self.enc1_res_blocks = nn.ModuleList([
            ResidualBlock(channels, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # Downsample with MaxPool: channels → channels*2
        self.down_conv = Conv(channels, channels*2, kernel_size=3, padding=1)
        self.down_bn = BatchNorm(channels*2)
        self.down_pool = MaxPool(kernel_size=2, stride=2)
        
        # Encoder Level 2: Residual blocks at resolution 0.5×
        self.enc2_res_blocks = nn.ModuleList([
            ResidualBlock(channels*2, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Bottleneck with Residual Blocks
        # ============================================
        self.bottleneck_res_blocks = nn.ModuleList([
            ResidualBlock(channels*2, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Decoder Path (Upsampling) with Residual Blocks
        # ============================================
        
        # Upsample: channels*2 → channels
        self.up1 = ConvTranspose(channels*2, channels, kernel_size=2, stride=2, padding=0)
        self.up1_bn = BatchNorm(channels)
        
        # Fusion after skip connection: channels (decoder) + channels (encoder) → channels
        self.dec_fusion_conv = Conv(channels*2, channels, kernel_size=1)
        self.dec_fusion_bn = BatchNorm(channels)
        
        # Decoder Level 1: Residual blocks at resolution 1×
        self.dec1_res_blocks = nn.ModuleList([
            ResidualBlock(channels, spatial_dims) 
            for _ in range(num_res_blocks)
        ])
        
        # ============================================
        # Output Layer
        # ============================================
        self.out = Conv(channels, 1, kernel_size=3, padding=1)
    
    def compute_physics_features(self, x):
        """
        Compute physics-informed features for dark matter density.
        
        Args:
            x: Input tensor (normalized density field)
                - For 3D: [B, 1, H, W, D]
                - For 2D: [B, 1, H, W]
        
        Returns:
            gradient_mag: Gradient magnitude |∇ρ| (structure boundaries)
            laplacian: Laplacian ∇²ρ (gravitational structure)
            contrast: Density contrast δ = (ρ - ρ̄)/ρ̄ (overdensity field)
        """
        # Remove channel dimension for gradient computation
        x_squeeze = x.squeeze(1)  # [B, H, W, D] or [B, H, W]
        
        if self.spatial_dims == 3:
            # Compute 3D gradients using central differences
            # torch.gradient returns tuple of gradients for each dimension
            grad_z, grad_y, grad_x = torch.gradient(x_squeeze, dim=(-3, -2, -1))
            
            # Gradient magnitude: |∇ρ| = sqrt(∂ρ/∂x² + ∂ρ/∂y² + ∂ρ/∂z²)
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
            
            # Laplacian: ∇²ρ = ∂²ρ/∂x² + ∂²ρ/∂y² + ∂²ρ/∂z²
            grad2_z = torch.gradient(grad_z, dim=-3)[0]
            grad2_y = torch.gradient(grad_y, dim=-2)[0]
            grad2_x = torch.gradient(grad_x, dim=-1)[0]
            laplacian = grad2_x + grad2_y + grad2_z
            
        else:  # 2D
            # Compute 2D gradients
            grad_y, grad_x = torch.gradient(x_squeeze, dim=(-2, -1))
            
            # Gradient magnitude: |∇ρ| = sqrt(∂ρ/∂x² + ∂ρ/∂y²)
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            # Laplacian: ∇²ρ = ∂²ρ/∂x² + ∂²ρ/∂y²
            grad2_y = torch.gradient(grad_y, dim=-2)[0]
            grad2_x = torch.gradient(grad_x, dim=-1)[0]
            laplacian = grad2_x + grad2_y
        
        # Density contrast: δ = (ρ - ρ̄)/ρ̄
        # Compute per-batch statistics to preserve batch dimension
        mean_density = x_squeeze.mean(dim=tuple(range(1, x_squeeze.ndim)), keepdim=True)
        contrast = (x_squeeze - mean_density) / (mean_density.abs() + 1e-8)
        
        # Add channel dimension back: [B, H, W, D] -> [B, 1, H, W, D]
        gradient_mag = gradient_mag.unsqueeze(1)
        laplacian = laplacian.unsqueeze(1)
        contrast = contrast.unsqueeze(1)
        
        return gradient_mag, laplacian, contrast
    
    def forward(self, x_decompressed, error_bound=None):
        """
        Forward pass: predict residuals using physics features with U-Net structure.
        
        Args:
            x_decompressed: SZ3-decompressed data
                - For 3D: [B, 1, H, W, D] or [H, W, D]
                - For 2D: [B, 1, H, W] or [H, W]
            error_bound: Error bound for relaxed mode (optional)
        
        Returns:
            residual: Predicted residuals (same shape as input)
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(x_decompressed, torch.Tensor):
            x_decompressed = torch.from_numpy(x_decompressed).float()
        
        # Add batch and channel dimensions if missing
        if self.spatial_dims == 3:
            if x_decompressed.ndim == 3:
                x_decompressed = x_decompressed.unsqueeze(0).unsqueeze(0)
            elif x_decompressed.ndim == 4:
                x_decompressed = x_decompressed.unsqueeze(1)
        elif self.spatial_dims == 2:
            if x_decompressed.ndim == 2:
                # Single 2D slice: (H, W) -> (1, 1, H, W)
                x_decompressed = x_decompressed.unsqueeze(0).unsqueeze(0)
            elif x_decompressed.ndim == 3:
                # Multiple 2D slices: (N, H, W) -> (N, 1, H, W)
                x_decompressed = x_decompressed.unsqueeze(1)
            # If ndim == 4, it's already correct (batch, channels, H, W) for Conv2d
        
        # Move to the same device as the model
        params = list(self.parameters())
        device = params[0].device if params else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_decompressed = x_decompressed.to(device)
        
        # Normalize spatial input
        # x_mean = x_decompressed.mean()
        # x_std = x_decompressed.std() + 1e-8
        # x_norm = (x_decompressed - x_mean) / x_std
        x_norm = x_decompressed
        
        # Compute physics features
        gradient_mag, laplacian, contrast = self.compute_physics_features(x_norm)
        
        # Normalize physics features
        grad_norm = (gradient_mag - gradient_mag.mean()) / (gradient_mag.std() + 1e-8)
        lap_norm = (laplacian - laplacian.mean()) / (laplacian.std() + 1e-8)
        contrast_norm = (contrast - contrast.mean()) / (contrast.std() + 1e-8)
        
        # ============================================
        # Process Each Branch Separately
        # ============================================
        spatial_feat = torch.nn.functional.gelu(self.spatial_bn1(self.spatial_conv1(x_norm)))
        gradient_feat = torch.nn.functional.gelu(self.gradient_bn1(self.gradient_conv1(grad_norm)))
        laplacian_feat = torch.nn.functional.gelu(self.laplacian_bn1(self.laplacian_conv1(lap_norm)))
        contrast_feat = torch.nn.functional.gelu(self.contrast_bn1(self.contrast_conv1(contrast_norm)))
        
        # Fuse all features: 4*channels → channels
        fused = torch.cat([spatial_feat, gradient_feat, laplacian_feat, contrast_feat], dim=1)
        fused = torch.nn.functional.gelu(self.fusion_bn(self.fusion_conv(fused)))
        
        # ============================================
        # Encoder Path with Residual Blocks
        # ============================================
        
        # Encoder Level 1 with residual blocks
        enc1 = fused
        for res_block in self.enc1_res_blocks:
            enc1 = res_block(enc1)
        
        # Downsample with MaxPool
        x = self.down_pool(enc1)  # First pool to reduce spatial dimensions
        x = torch.nn.functional.gelu(self.down_bn(self.down_conv(x)))  # Then conv
        
        # Encoder Level 2 with residual blocks
        enc2 = x
        for res_block in self.enc2_res_blocks:
            enc2 = res_block(enc2)
        
        # ============================================
        # Bottleneck with Residual Blocks
        # ============================================
        bottleneck = enc2
        for res_block in self.bottleneck_res_blocks:
            bottleneck = res_block(bottleneck)
        
        # ============================================
        # Decoder Path with Skip Connection
        # ============================================
        
        # Upsample
        dec1 = torch.nn.functional.gelu(self.up1_bn(self.up1(bottleneck)))
        
        # Skip connection: concatenate with enc1
        dec1_skip = torch.cat([dec1, enc1], dim=1)
        
        # Fusion convolution
        dec1 = torch.nn.functional.gelu(self.dec_fusion_bn(self.dec_fusion_conv(dec1_skip)))
        
        # Decoder Level 1 with residual blocks
        for res_block in self.dec1_res_blocks:
            dec1 = res_block(dec1)
        
        # ============================================
        # Output
        # ============================================
        residual = self.out(dec1)
        
        # Denormalize residual
        # residual = residual * x_std
        
        
        return residual
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())