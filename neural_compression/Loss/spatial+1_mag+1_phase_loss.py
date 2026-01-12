import torch
import torch.nn as nn

class SpatialFrequencyLoss(nn.Module):
    """
    Combines spatial MSE with magnitude/phase losses in the frequency domain.
    """
    def __init__(self, weight_spatial=1.0, weight_magnitude=0.1, weight_phase=0.1, spatial_dims=3):
        super().__init__()
        self.base_weight_spatial = weight_spatial
        self.base_weight_magnitude = weight_magnitude
        self.base_weight_phase = weight_phase
        
        self.weight_spatial = weight_spatial
        self.weight_magnitude = weight_magnitude
        self.weight_phase = weight_phase
        self.spatial_dims = spatial_dims
        self.mse = nn.MSELoss()

    # def update_weights(self, current_epoch, total_epochs):
    #     """
    #     Anneal frequency weights: 
    #     - First 50% epochs: Full frequency weights
    #     - Next 30% epochs: Linear decay to 0
    #     - Last 20% epochs: 0 frequency weight (Pure MSE fine-tuning)
    #     """
    #     progress = current_epoch / total_epochs
        
    #     if progress < 0.5:
    #         factor = 1.0
    #     elif progress < 0.8:
    #         factor = 1.0 - (progress - 0.5) / 0.3
    #     else:
    #         factor = 0.0
            
    #     self.weight_magnitude = self.base_weight_magnitude * factor
    #     self.weight_phase = self.base_weight_phase * factor
        
    #     return {
    #         'mag_weight': self.weight_magnitude, 
    #         'phase_weight': self.weight_phase
    #     }

    def forward(self, pred, target):
        # 1. Spatial MSE
        spatial_loss = self.mse(pred, target)

        # 2. Frequency Domain (Use norm='ortho' to keep scales consistent!)
        if self.spatial_dims == 3:
            fft_pred = torch.fft.fftn(pred, dim=(-3, -2, -1), norm='ortho')
            fft_target = torch.fft.fftn(target, dim=(-3, -2, -1), norm='ortho')
        else:
            fft_pred = torch.fft.fftn(pred, dim=(-2, -1), norm='ortho')
            fft_target = torch.fft.fftn(target, dim=(-2, -1), norm='ortho')

        # Magnitude loss (now on same scale as spatial)
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)
        magnitude_loss = self.mse(mag_pred, mag_target)

        # Phase loss (Robust sin/cos method)
        phase_pred = torch.angle(fft_pred)
        phase_target = torch.angle(fft_target)
        # Use sin/cos to handle circular nature of phase (-pi == pi)
        phase_loss = self.mse(torch.sin(phase_pred), torch.sin(phase_target)) + \
                     self.mse(torch.cos(phase_pred), torch.cos(phase_target))
        phase_loss *= 0.5

        total_loss = (self.weight_spatial * spatial_loss +
                      self.weight_magnitude * magnitude_loss +
                      self.weight_phase * phase_loss)

        loss_dict = {
            "spatial": float(spatial_loss.item()),
            "magnitude": float(magnitude_loss.item()),
            "phase": float(phase_loss.item())
        }
        return total_loss, loss_dict