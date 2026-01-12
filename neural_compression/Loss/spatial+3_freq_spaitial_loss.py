import torch
import torch.nn as nn
import torch.fft

class BandWeightedSpectralLoss(nn.Module):
    """
    The 'Corresponding' Loss for TinySpatialBandPredictor.
    
    It decomposes the Prediction and Target into Low/Mid/High frequency bands
    (matching the model's logic) and calculates separate losses for each.
    
    This allows you to aggressively weight the HIGH frequency band, 
    which is where the residual errors (edges/noise) usually hide.
    """
    def __init__(self, 
                 weight_spatial=1.0, 
                 weight_low=0.1,    # Shape (usually easy)
                 weight_mid=1.0,    # Texture
                 weight_high=5.0,   # Edges/Noise (CRITICAL for residuals)
                 spatial_dims=3,
                 low_cutoff=0.15,
                 mid_cutoff=0.40):
        super().__init__()
        self.weights = {
            'spatial': weight_spatial,
            'low': weight_low,
            'mid': weight_mid,
            'high': weight_high
        }
        self.spatial_dims = spatial_dims
        self.low_cutoff = low_cutoff
        self.mid_cutoff = mid_cutoff
        self.mse = nn.MSELoss()

    def _create_band_masks(self, shape, device):
        """Recreates the exact same masks used inside the Model"""
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

    def forward(self, pred, target):
        # 1. Standard Spatial MSE (The Baseline)
        loss_spatial = self.mse(pred, target)
        
        # 2. Compute FFT of both Pred and Target
        # Note: We calculate loss in Frequency Domain (Parseval's Theorem)
        # because it's faster than IFFT-ing back to Spatial domain.
        if self.spatial_dims == 2:
            pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, norm='ortho'))
            target_fft = torch.fft.fftshift(torch.fft.fft2(target, norm='ortho'))
        else:
            pred_fft = torch.fft.fftshift(torch.fft.fftn(pred, dim=(-3,-2,-1), norm='ortho'))
            target_fft = torch.fft.fftshift(torch.fft.fftn(target, dim=(-3,-2,-1), norm='ortho'))
            
        # Use Magnitude for Loss (Focus on Energy/Strength of features)
        # Log-scale helps balance the dynamic range
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        
        # 3. Apply Masks to separate bands
        mask_low, mask_mid, mask_high = self._create_band_masks(pred.shape, pred.device)
        
        # Expand masks to batch/channel dims
        mask_low = mask_low.unsqueeze(0).unsqueeze(0)
        mask_mid = mask_mid.unsqueeze(0).unsqueeze(0)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)
        
        # 4. Calculate Band-Specific Losses
        # Low Freq Error
        loss_low = self.mse(pred_mag * mask_low, target_mag * mask_low)
        
        # Mid Freq Error
        loss_mid = self.mse(pred_mag * mask_mid, target_mag * mask_mid)
        
        # High Freq Error (The most important for residuals!)
        loss_high = self.mse(pred_mag * mask_high, target_mag * mask_high)
        
        # 5. Combine
        total_loss = (self.weights['spatial'] * loss_spatial +
                      self.weights['low'] * loss_low +
                      self.weights['mid'] * loss_mid +
                      self.weights['high'] * loss_high)
                      
        return total_loss