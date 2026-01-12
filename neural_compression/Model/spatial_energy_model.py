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

class TinyFrequencyResidualPredictorWithEnergy(nn.Module):
    """
    TinyFrequencyResidualPredictor 的能量增强版：
    - 分支：spatial + energy(3-band) only, without magnitude and phase
    - 能量来自全局 FFT 的幅值，并按半径划分低/中/高三段
    """
    def __init__(self, channels=4, spatial_dims=3, num_res_blocks=1, local_fft_size=16, energy_from_fft=True):
        super().__init__()
        self.channels = channels
        self.spatial_dims = spatial_dims
        self.num_res_blocks = num_res_blocks
        self.energy_from_fft = energy_from_fft  # True 时自动由 FFT 幅值生成能量图
        # 选择算子
        if spatial_dims == 2:
            Conv = nn.Conv2d; ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d; MaxPool = nn.MaxPool2d
            ResBlock = lambda ch: ResidualBlock(ch, spatial_dims=2)
        elif spatial_dims == 3:
            Conv = nn.Conv3d; ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d; MaxPool = nn.MaxPool3d
            ResBlock = lambda ch: ResidualBlock(ch, spatial_dims=3)
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        def make_branch(in_channels, branch_channels):
            return nn.ModuleDict({
                'init_conv': Conv(in_channels, branch_channels, 3, padding=1),
                'init_bn': BatchNorm(branch_channels),
                'enc1': nn.ModuleList([ResBlock(branch_channels) for _ in range(num_res_blocks)]),
                'down_conv': Conv(branch_channels, branch_channels*2, 3, padding=1),
                'down_bn': BatchNorm(branch_channels*2),
                'down_pool': MaxPool(2, 2),
                'enc2': nn.ModuleList([ResBlock(branch_channels*2) for _ in range(num_res_blocks)]),
                'bottleneck': nn.ModuleList([ResBlock(branch_channels*2) for _ in range(num_res_blocks)]),
                'up1': ConvTranspose(branch_channels*2, branch_channels, 2, stride=2, padding=0),
                'up1_bn': BatchNorm(branch_channels),
                'fusion_conv': Conv(branch_channels*2, branch_channels, 1),
                'fusion_bn': BatchNorm(branch_channels),
                'dec1': nn.ModuleList([ResBlock(branch_channels) for _ in range(num_res_blocks)]),
            })

        branch_ch = channels
        self.spatial_branch = make_branch(1, branch_ch)
        self.energy_branch = make_branch(3, branch_ch)  # 3 个能量子带

        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.final_fusion = Conv(branch_ch, 1, kernel_size=3, padding=1)

        total_params = sum(p.numel() for p in self.parameters())
        print("TinyFrequencyResidualPredictorWithEnergy (GLOBAL FFT) initialized:")
        print("  Branches: spatial + energy(3-band)")
        print(f"  Total: {total_params:,} params")

    def _process_branch(self, x, branch):
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
        x_squeeze = x.squeeze(1)
        if self.spatial_dims == 2:
            fft_result = torch.fft.fft2(x_squeeze, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        else:
            fft_result = torch.fft.fftn(x_squeeze, dim=(-3, -2, -1), norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-3, -2, -1))
        return torch.log1p(torch.abs(fft_shifted)).unsqueeze(1)

    def _compute_energy_bands(self, magnitude):
        # magnitude: [B,1,H,W] or [B,1,H,W,D]
        spatial_dims = magnitude.dim() - 2  # 3 for 2D input, 4 for 3D input
        center = [(s - 1) / 2 for s in magnitude.shape[2:]]
        grids = torch.meshgrid([torch.arange(s, device=magnitude.device) for s in magnitude.shape[2:]], indexing='ij')
        radius = sum((g - c) ** 2 for g, c in zip(grids, center)) ** 0.5
        max_r = radius.max().clamp(min=1e-6)
        r1, r2 = max_r / 3, 2 * max_r / 3
        band_masks = [
            (radius <= r1).float(),
            ((radius > r1) & (radius <= r2)).float(),
            (radius > r2).float(),
        ]
        bands = [magnitude * m.unsqueeze(0).unsqueeze(0) for m in band_masks]  # match [B,1,...]
        energy = torch.cat(bands, dim=1)  # [B,3,...]
        return energy

    def forward(self, x, energy=None):
        # 预处理形状
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

        x = x.to(self.spatial_branch['init_conv'].weight.device)

        magnitude = self._compute_global_fft_features(x)
        if energy is None and self.energy_from_fft:
            energy = self._compute_energy_bands(magnitude)
        elif energy is None:
            energy = magnitude.repeat(1, 3, *([1] * (magnitude.dim() - 2)))  # 兜底

        # 2 branches
        if x.is_cuda and torch.cuda.is_available():
            stream_spatial = torch.cuda.Stream()
            stream_energy = torch.cuda.Stream()
            with torch.cuda.stream(stream_spatial):
                spatial_out = self._process_branch(x, self.spatial_branch)
            with torch.cuda.stream(stream_energy):
                energy_out = self._process_branch(energy, self.energy_branch)
            torch.cuda.synchronize()
        else:
            spatial_out = self._process_branch(x, self.spatial_branch)  
            energy_out = self._process_branch(energy, self.energy_branch)

        weights = torch.softmax(self.fusion_weights, dim=0)
        combined = weights[0] * spatial_out + weights[1] * energy_out
        output = self.final_fusion(combined)
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self):
        total = sum(p.numel() for p in self.parameters())
        spatial_params = sum(p.numel() for n, p in self.named_parameters() if 'spatial_' in n)
        energy_params = sum(p.numel() for n, p in self.named_parameters() if 'energy_' in n)
        fusion_params = sum(p.numel() for n, p in self.named_parameters() if 'final_' in n or 'fusion_weights' in n)
        return {
            'spatial_branch': spatial_params,
            'energy_branch': energy_params,
            'final_fusion': fusion_params,
            'total': total,
        }