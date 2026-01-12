import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class ResidualBlock(nn.Module):
    def __init__(self, channels, spatial_dims=3):
        super().__init__()
        self.channels = channels
        self.spatial_dims = spatial_dims
        
        if spatial_dims == 2:
            Conv = nn.Conv2d; ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d; MaxPool = nn.MaxPool2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d; ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d; MaxPool = nn.MaxPool3d
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        self.conv1 = Conv(channels, channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(channels, channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class TinyFrequencyResidualPredictor_4_inputs(nn.Module):
    """
    STRATEGY 5: Early Fusion Model.
    
    Architecture:
      - Single U-Net (efficient parameter usage like Simple Model).
      - Input: 4 Channels (Original + Low_Spatial + Mid_Spatial + High_Spatial).
      - Goal: Forces the single strong network to use spatial and frequency data.
    """
    
    def __init__(self, channels=4, spatial_dims=2, num_res_blocks=1):
        """
        Args:
            channels: Base filter size for the U-Net (e.g. 4).
            spatial_dims: 2 or 3.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_res_blocks = num_res_blocks
        
        # --------------------------------------------------------
        # 1. Frequency Band Configuration
        # --------------------------------------------------------
        self.low_cutoff = 0.15   # 0% to 15% (Coarse Shapes)
        self.mid_cutoff = 0.40   # 15% to 40% (Textures)
                                 # >40% is High (Edges/Noise)
        
        # --------------------------------------------------------
        # 2. Define Layers (Standard U-Net)
        # --------------------------------------------------------
        if spatial_dims == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            ResBlock = lambda ch: ResidualBlock(ch, spatial_dims=2)
        elif spatial_dims == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            ResBlock = lambda ch: ResidualBlock(ch, spatial_dims=3)
        else:
            raise ValueError(f"spatial_dims must be 2 or 3")
        
        # INPUT CONFIGURATION:
        # Original (1) + Low_Spatial (1) + Mid_Spatial (1) + High_Spatial (1) = 4 Inputs
        # We can keep 'channels' (hidden dim) small (e.g., 4 or 8) 
        # because the inputs are already highly featured.
        in_channels = 4 
        
        # Encoder Path
        self.init_conv = Conv(in_channels, channels, kernel_size=3, padding=1)
        self.init_bn = BatchNorm(channels)
        
        self.enc1_res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        
        self.down_conv = Conv(channels, channels*2, kernel_size=3, padding=1)
        self.down_bn = BatchNorm(channels*2)
        self.down_pool = MaxPool(kernel_size=2, stride=2)
        
        self.enc2_res_blocks = nn.ModuleList([ResBlock(channels*2) for _ in range(num_res_blocks)])
        
        # Bottleneck
        self.bottleneck_res_blocks = nn.ModuleList([ResBlock(channels*2) for _ in range(num_res_blocks)])
        
        # Decoder Path
        self.up1 = ConvTranspose(channels*2, channels, kernel_size=2, stride=2, padding=0)
        self.up1_bn = BatchNorm(channels)
        
        self.fusion_conv = Conv(channels*2, channels, kernel_size=1)
        self.fusion_bn = BatchNorm(channels)
        
        self.dec1_res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Output Layer (Predicts 1 channel residual)
        self.out = Conv(channels, 1, kernel_size=3, padding=1)

    def _create_band_masks(self, shape, device):
        """Generates Low, Mid, High frequency masks."""
        if self.spatial_dims == 2:
            h, w = shape[-2], shape[-1]
            y = torch.linspace(-1, 1, h).to(device)
            x = torch.linspace(-1, 1, w).to(device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            radius = torch.sqrt(grid_x**2 + grid_y**2)
        else:
            h, w, d = shape[-3], shape[-2], shape[-1]
            z = torch.linspace(-1, 1, h).to(device)
            y = torch.linspace(-1, 1, w).to(device)
            x = torch.linspace(-1, 1, d).to(device)
            grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
            radius = torch.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
            
        mask_low = (radius <= self.low_cutoff).float()
        mask_mid = ((radius > self.low_cutoff) & (radius <= self.mid_cutoff)).float()
        mask_high = (radius > self.mid_cutoff).float()
        
        return mask_low, mask_mid, mask_high

    def _compute_spatial_bands(self, x):
        """
        1. FFT
        2. Mask
        3. Inverse FFT -> Return SPATIAL images
        """
        x_squeeze = x.squeeze(1)
        
        # 1. FFT
        if self.spatial_dims == 2:
            fft_result = torch.fft.fft2(x_squeeze, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        else:
            fft_result = torch.fft.fftn(x_squeeze, dim=(-3, -2, -1), norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-3, -2, -1))
        
        # 2. Create Masks
        mask_low, mask_mid, mask_high = self._create_band_masks(x.shape, x.device)
        # Expand masks to match batch [B, H, W]
        mask_low = mask_low.unsqueeze(0).expand_as(fft_result)
        mask_mid = mask_mid.unsqueeze(0).expand_as(fft_result)
        mask_high = mask_high.unsqueeze(0).expand_as(fft_result)
        
        # 3. Apply Masks (in Frequency Domain)
        fft_L = fft_shifted * mask_low
        fft_M = fft_shifted * mask_mid
        fft_H = fft_shifted * mask_high
        
        # 4. Inverse FFT (Back to Spatial Domain)
        # We need to unshift before IFFT
        if self.spatial_dims == 2:
            spatial_L = torch.fft.ifft2(torch.fft.ifftshift(fft_L, dim=(-2, -1)), norm='ortho').real
            spatial_M = torch.fft.ifft2(torch.fft.ifftshift(fft_M, dim=(-2, -1)), norm='ortho').real
            spatial_H = torch.fft.ifft2(torch.fft.ifftshift(fft_H, dim=(-2, -1)), norm='ortho').real
        else:
            spatial_L = torch.fft.fftn(torch.fft.ifftshift(fft_L, dim=(-3, -2, -1)), norm='ortho').real
            spatial_M = torch.fft.fftn(torch.fft.ifftshift(fft_M, dim=(-3, -2, -1)), norm='ortho').real
            spatial_H = torch.fft.fftn(torch.fft.ifftshift(fft_H, dim=(-3, -2, -1)), norm='ortho').real
            
        # Add channel dim back: [B, 1, H, W]
        spatial_L = spatial_L.unsqueeze(1)
        spatial_M = spatial_M.unsqueeze(1)
        spatial_H = spatial_H.unsqueeze(1)
        
        return spatial_L, spatial_M, spatial_H

    def forward(self, x):
        # Ensure input format
        if not isinstance(x, torch.Tensor): x = torch.from_numpy(x).float()
        if self.spatial_dims == 3 and x.ndim == 3: x = x.unsqueeze(0).unsqueeze(0)
        elif self.spatial_dims == 3 and x.ndim == 4: x = x.unsqueeze(1)
        elif self.spatial_dims == 2 and x.ndim == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif self.spatial_dims == 2 and x.ndim == 3: x = x.unsqueeze(1)
        
        # 1. Compute SPATIAL Bands
        spatial_L, spatial_M, spatial_H = self._compute_spatial_bands(x)
        
        # 2. Stack Spatially Aligned Inputs
        # Original + Low + Mid + High
        combined_input = torch.cat([x, spatial_L, spatial_M, spatial_H], dim=1)
        
        # 3. Standard U-Net Forward Pass
        x_enc = torch.nn.functional.gelu(self.init_bn(self.init_conv(combined_input)))
        
        enc1 = x_enc
        for res_block in self.enc1_res_blocks:
            enc1 = res_block(enc1)
        
        x_down = self.down_pool(enc1)
        x_down = torch.nn.functional.gelu(self.down_bn(self.down_conv(x_down)))
        
        enc2 = x_down
        for res_block in self.enc2_res_blocks:
            enc2 = res_block(enc2)
            
        bottleneck = enc2
        for res_block in self.bottleneck_res_blocks:
            bottleneck = res_block(bottleneck)
            
        dec1 = torch.nn.functional.gelu(self.up1_bn(self.up1(bottleneck)))
        dec1_skip = torch.cat([dec1, enc1], dim=1)
        
        dec1 = torch.nn.functional.gelu(self.fusion_bn(self.fusion_conv(dec1_skip)))
        
        for res_block in self.dec1_res_blocks:
            dec1 = res_block(dec1)
            
        residual = self.out(dec1)
        
        return residual
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())