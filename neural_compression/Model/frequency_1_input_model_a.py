import torch

import torch.nn as nn
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

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class TinyFrequencyResidualPredictor_1_input(nn.Module):
    """
    STRATEGY 5: Early Fusion Model.
    
    Architecture:
      - Single U-Net (efficient parameter usage like Simple Model).
      - Input: 1 Channel (Original).
      - Goal: Forces the single strong network to use spatial and frequency data.
    """
    
    def __init__(self, channels=4, spatial_dims=2, num_res_blocks=1):
        """
        # ... existing make_branch code stays the same ...
        # Select conv operations based on spatial dimensions

        """
        super().__init__()
        self.channels = channels
        self.spatial_dims = spatial_dims
        
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
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        
        def make_branch(in_channels, branch_channels):
            """Create encoder-bottleneck-decoder for one branch"""
            return nn.ModuleDict({
                # Encoder
                'init_conv': Conv(in_channels, branch_channels, kernel_size=3, padding=1),
                'init_bn': BatchNorm(branch_channels),
                'enc1': nn.ModuleList([ResBlock(branch_channels) for _ in range(num_res_blocks)]),
                'down_conv': Conv(branch_channels, branch_channels*2, kernel_size=3, padding=1),
                'down_bn': BatchNorm(branch_channels*2),
                'down_pool': MaxPool(kernel_size=2, stride=2),
                'enc2': nn.ModuleList([ResBlock(branch_channels*2) for _ in range(num_res_blocks)]),
                # Bottleneck
                'bottleneck': nn.ModuleList([ResBlock(branch_channels*2) for _ in range(num_res_blocks)]),
                # Decoder
                'up1': ConvTranspose(branch_channels*2, branch_channels, kernel_size=2, stride=2, padding=0),
                'up1_bn': BatchNorm(branch_channels),
                'fusion_conv': Conv(branch_channels*2, branch_channels, kernel_size=1),
                'fusion_bn': BatchNorm(branch_channels),
                'dec1': nn.ModuleList([ResBlock(branch_channels) for _ in range(num_res_blocks)]),
            })
        
        # THREE PARALLEL BRANCHES
        branch_ch = channels
        self.spatial_branch = make_branch(1, branch_ch)
        self.mag_branch = make_branch(1, branch_ch)
        self.phase_branch = make_branch(1, branch_ch)
        
        # LEARNABLE FUSION WEIGHTS
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
        # Final fusion layer
        self.final_fusion = Conv(branch_ch, 1, kernel_size=3, padding=1)
        
        # Print param count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"TinyFrequencyResidualPredictor (GLOBAL FFT) initialized:")
        print(f"  Branches: spatial + magnitude + phase")
        print(f"  Total: {total_params:,} params")

    def _process_branch(self, x, branch):
        """Process input through a single U-Net branch"""
        # ... same as before ...
        x0 = torch.nn.functional.gelu(branch['init_bn'](branch['init_conv'](x)))
        e1 = x0
        for blk in branch['enc1']:
            e1 = blk(e1)
        
        x1 = branch['down_pool'](e1)
        x1 = torch.nn.functional.gelu(branch['down_bn'](branch['down_conv'](x1)))
        
        e2 = x1
        for blk in branch['enc2']:
            e2 = blk(e2)
        
        b = e2
        for blk in branch['bottleneck']:
            b = blk(b)
        
        d1 = torch.nn.functional.gelu(branch['up1_bn'](branch['up1'](b)))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = torch.nn.functional.gelu(branch['fusion_bn'](branch['fusion_conv'](d1)))
        for blk in branch['dec1']:
            d1 = blk(d1)
        
        return d1

    def _compute_global_fft_features(self, x):
        """
        Compute GLOBAL FFT features on the entire data.
        Returns magnitude AND phase with same spatial dimensions as input.
        
        Args:
            x: Input tensor [B, 1, H, W] or [B, 1, H, W, D]
        Returns:
            magnitude: Log magnitude (fftshifted) [B, 1, H, W] or [B, 1, H, W, D]
            phase: Phase (normalized to [-1, 1]) [B, 1, H, W] or [B, 1, H, W, D]
        """
        # Remove channel dim for FFT: [B, 1, H, W] -> [B, H, W]
        x_squeeze = x.squeeze(1)
        
        if self.spatial_dims == 2:
            # Global 2D FFT
            fft_result = torch.fft.fft2(x_squeeze, norm='ortho')
            # Shift zero frequency to center
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        else:  # 3D
            # Global 3D FFT
            fft_result = torch.fft.fftn(x_squeeze, dim=(-3, -2, -1), norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-3, -2, -1))
        
        # Compute magnitude (log-scaled for better dynamic range)
        magnitude = torch.log1p(torch.abs(fft_shifted))
        
        # Compute phase (normalized to [-1, 1])
        phase = torch.angle(fft_shifted) / torch.pi
        
        # Add channel dim back: [B, H, W] -> [B, 1, H, W]
        magnitude = magnitude.unsqueeze(1)
        phase = phase.unsqueeze(1)
        
        return magnitude, phase

    def forward(self, x):
        # Preprocessing
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if self.spatial_dims == 3 and x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif self.spatial_dims == 3 and x.ndim == 4:
            x = x.unsqueeze(1)
        elif self.spatial_dims == 2 and x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif self.spatial_dims == 2 and x.ndim == 3:
            x = x.unsqueeze(1)
        
        # Move to device
        x = x.to(self.spatial_branch['init_conv'].weight.device)
        
        # ============================================
        # Compute GLOBAL FFT magnitude AND phase
        # ============================================
        magnitude, phase = self._compute_global_fft_features(x)
        
        # ============================================
        # Process all THREE branches
        # ============================================
        if x.is_cuda and torch.cuda.is_available():
            stream_spatial = torch.cuda.Stream()
            stream_mag = torch.cuda.Stream()
            stream_phase = torch.cuda.Stream()
            
            with torch.cuda.stream(stream_spatial):
                spatial_out = self._process_branch(x, self.spatial_branch)
            
            with torch.cuda.stream(stream_mag):
                mag_out = self._process_branch(magnitude, self.mag_branch)
            
            with torch.cuda.stream(stream_phase):
                phase_out = self._process_branch(phase, self.phase_branch)
            
            torch.cuda.synchronize()
        else:
            spatial_out = self._process_branch(x, self.spatial_branch)
            mag_out = self._process_branch(magnitude, self.mag_branch)
            phase_out = self._process_branch(phase, self.phase_branch)
        
        # ============================================
        # Learnable fusion of 3 branches
        # ============================================
        weights = torch.softmax(self.fusion_weights, dim=0)
        combined = weights[0] * spatial_out + weights[1] * mag_out + weights[2] * phase_out
        output = self.final_fusion(combined)
        
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_breakdown(self):
        # ... same as before ...
        total = sum(p.numel() for p in self.parameters())
        spatial_params = sum(p.numel() for name, p in self.named_parameters() if 'spatial_' in name)
        mag_params = sum(p.numel() for name, p in self.named_parameters() if 'mag_' in name)
        phase_params = sum(p.numel() for name, p in self.named_parameters() if 'phase_' in name)
        fusion_params = sum(p.numel() for name, p in self.named_parameters() if 'final_' in name or 'fusion_weights' in name)
        
        return {
            'spatial_branch': spatial_params,
            'magnitude_branch': mag_params,
            'phase_branch': phase_params,
            'final_fusion': fusion_params,
            'total': total,
        }