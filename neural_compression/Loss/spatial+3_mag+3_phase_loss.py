import torch
import torch.nn as nn
import torch.fft

class BandedFrequencyLoss_3_mag_3_phase(nn.Module):
    """
    与 TinyFrequencyResidualPredictor 模型匹配的频率损失函数。
    使用相同的低/中/高频带划分，分别计算各频带的幅度和相位损失。
    """
    def __init__(self, 
                 weight_spatial=1.0,
                 weight_mag_low=0.1,
                 weight_mag_mid=0.1,
                 weight_mag_high=0.05,
                 weight_phase_low=0.05,
                 weight_phase_mid=0.05,
                 weight_phase_high=0.02,
                 spatial_dims=3,
                 low_cutoff=0.15,
                 mid_cutoff=0.40):
        """
        Args:
            weight_spatial: 空间域 MSE 损失权重
            weight_mag_low/mid/high: 低/中/高频幅度损失权重
            weight_phase_low/mid/high: 低/中/高频相位损失权重
            spatial_dims: 2 或 3
            low_cutoff: 低频截止 (与模型一致，默认 0.15)
            mid_cutoff: 中频截止 (与模型一致，默认 0.40)
        """
        super().__init__()
        self.weight_spatial = weight_spatial
        self.weight_mag_low = weight_mag_low
        self.weight_mag_mid = weight_mag_mid
        self.weight_mag_high = weight_mag_high
        self.weight_phase_low = weight_phase_low
        self.weight_phase_mid = weight_phase_mid
        self.weight_phase_high = weight_phase_high
        self.spatial_dims = spatial_dims
        self.low_cutoff = low_cutoff
        self.mid_cutoff = mid_cutoff
        self.mse = nn.MSELoss()

    def _create_band_masks(self, shape, device):
        """生成与模型一致的低/中/高频掩码"""
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

    def _compute_banded_features(self, x):
        """计算各频带的幅度和相位特征"""
        x_squeeze = x.squeeze(1)
        
        # FFT
        if self.spatial_dims == 2:
            fft_result = torch.fft.fft2(x_squeeze, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        else:
            fft_result = torch.fft.fftn(x_squeeze, dim=(-3, -2, -1), norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-3, -2, -1))
        
        magnitude = torch.log1p(torch.abs(fft_shifted))
        phase = torch.angle(fft_shifted) / torch.pi  # 归一化到 [-1, 1]
        
        # 创建掩码
        mask_low, mask_mid, mask_high = self._create_band_masks(x.shape, x.device)
        mask_low = mask_low.unsqueeze(0).expand_as(magnitude)
        mask_mid = mask_mid.unsqueeze(0).expand_as(magnitude)
        mask_high = mask_high.unsqueeze(0).expand_as(magnitude)
        
        # 应用掩码得到各频带
        mag_bands = {
            'low': magnitude * mask_low,
            'mid': magnitude * mask_mid,
            'high': magnitude * mask_high
        }
        phase_bands = {
            'low': phase * mask_low,
            'mid': phase * mask_mid,
            'high': phase * mask_high
        }
        
        return mag_bands, phase_bands

    def forward(self, pred, target):
        # 1. 空间域 MSE
        spatial_loss = self.mse(pred, target)
        
        # 2. 计算各频带特征
        pred_mag, pred_phase = self._compute_banded_features(pred)
        tgt_mag, tgt_phase = self._compute_banded_features(target)
        
        # 3. 各频带幅度损失
        mag_loss_low = self.mse(pred_mag['low'], tgt_mag['low'])
        mag_loss_mid = self.mse(pred_mag['mid'], tgt_mag['mid'])
        mag_loss_high = self.mse(pred_mag['high'], tgt_mag['high'])
        
        # 4. 各频带相位损失 (使用 sin/cos 处理相位的循环特性)
        phase_loss_low = 0.5 * (
            self.mse(torch.sin(pred_phase['low'] * torch.pi), 
                     torch.sin(tgt_phase['low'] * torch.pi)) +
            self.mse(torch.cos(pred_phase['low'] * torch.pi), 
                     torch.cos(tgt_phase['low'] * torch.pi))
        )
        phase_loss_mid = 0.5 * (
            self.mse(torch.sin(pred_phase['mid'] * torch.pi), 
                     torch.sin(tgt_phase['mid'] * torch.pi)) +
            self.mse(torch.cos(pred_phase['mid'] * torch.pi), 
                     torch.cos(tgt_phase['mid'] * torch.pi))
        )
        phase_loss_high = 0.5 * (
            self.mse(torch.sin(pred_phase['high'] * torch.pi), 
                     torch.sin(tgt_phase['high'] * torch.pi)) +
            self.mse(torch.cos(pred_phase['high'] * torch.pi), 
                     torch.cos(tgt_phase['high'] * torch.pi))
        )
        
        # 5. 加权求和
        total_loss = (
            self.weight_spatial * spatial_loss +
            self.weight_mag_low * mag_loss_low +
            self.weight_mag_mid * mag_loss_mid +
            self.weight_mag_high * mag_loss_high +
            self.weight_phase_low * phase_loss_low +
            self.weight_phase_mid * phase_loss_mid +
            self.weight_phase_high * phase_loss_high
        )
        
        loss_dict = {
            "spatial": float(spatial_loss.item()),
            "mag_low": float(mag_loss_low.item()),
            "mag_mid": float(mag_loss_mid.item()),
            "mag_high": float(mag_loss_high.item()),
            "phase_low": float(phase_loss_low.item()),
            "phase_mid": float(phase_loss_mid.item()),
            "phase_high": float(phase_loss_high.item()),
        }
        
        return total_loss, loss_dict