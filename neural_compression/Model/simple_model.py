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
    
    def __init__(self, channels=4, spatial_dims=3, num_res_blocks=1):
        """
        Args:
            channels: Base channel count (4 → ~15k params for 3D, ~8k for 2D)
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