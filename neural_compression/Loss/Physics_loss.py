import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss for dark matter density field compression.
    
    Combines MSE reconstruction loss with physics-based constraints:
    1. Gravitational structure loss (Laplacian consistency)
    2. Overdensity field loss (density contrast conservation)
    3. Structure boundaries loss (gradient magnitude preservation)
    """
    
    def __init__(self, spatial_dims=3, 
                 weight_mse=1.0,
                 weight_laplacian=0.1,
                 weight_contrast=0.1,
                 weight_gradient=0.1):
        """
        Args:
            spatial_dims: 2 or 3 for spatial dimensions
            weight_mse: Weight for MSE reconstruction loss
            weight_laplacian: Weight for gravitational structure (Laplacian) loss
            weight_contrast: Weight for overdensity field (density contrast) loss
            weight_gradient: Weight for structure boundaries (gradient magnitude) loss
        """
        super(PhysicsInformedLoss, self).__init__()
        self.spatial_dims = spatial_dims
        self.weight_mse = weight_mse
        self.weight_laplacian = weight_laplacian
        self.weight_contrast = weight_contrast
        self.weight_gradient = weight_gradient
        
        self.mse = nn.MSELoss()
    
    def compute_gradient_magnitude(self, x):
        """
        Compute gradient magnitude |∇ρ| - structure boundaries.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, W, D]
        
        Returns:
            gradient_mag: Gradient magnitude [B, C, H, W] or [B, C, H, W, D]
        """
        # Remove channel dimension for gradient computation
        x_squeeze = x.squeeze(1)  # [B, H, W] or [B, H, W, D]
        
        if self.spatial_dims == 3:
            # 3D gradients
            grad_z, grad_y, grad_x = torch.gradient(x_squeeze, dim=(-3, -2, -1))
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        else:
            # 2D gradients
            grad_y, grad_x = torch.gradient(x_squeeze, dim=(-2, -1))
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return gradient_mag.unsqueeze(1)  # Add channel back
    
    def compute_laplacian(self, x):
        """
        Compute Laplacian ∇²ρ - gravitational structure (Poisson equation).
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, W, D]
        
        Returns:
            laplacian: Laplacian [B, C, H, W] or [B, C, H, W, D]
        """
        # Remove channel dimension
        x_squeeze = x.squeeze(1)  # [B, H, W] or [B, H, W, D]
        
        if self.spatial_dims == 3:
            # 3D Laplacian
            grad_z, grad_y, grad_x = torch.gradient(x_squeeze, dim=(-3, -2, -1))
            grad2_z = torch.gradient(grad_z, dim=-3)[0]
            grad2_y = torch.gradient(grad_y, dim=-2)[0]
            grad2_x = torch.gradient(grad_x, dim=-1)[0]
            laplacian = grad2_x + grad2_y + grad2_z
        else:
            # 2D Laplacian
            grad_y, grad_x = torch.gradient(x_squeeze, dim=(-2, -1))
            grad2_y = torch.gradient(grad_y, dim=-2)[0]
            grad2_x = torch.gradient(grad_x, dim=-1)[0]
            laplacian = grad2_x + grad2_y
        
        return laplacian.unsqueeze(1)  # Add channel back
    
    def compute_density_contrast(self, x):
        """
        Compute density contrast δ = (ρ - ρ̄)/ρ̄ - overdensity field.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C, H, W, D]
        
        Returns:
            contrast: Density contrast [B, C, H, W] or [B, C, H, W, D]
        """
        # Remove channel dimension
        x_squeeze = x.squeeze(1)  # [B, H, W] or [B, H, W, D]
        
        # Compute per-batch statistics
        mean_density = x_squeeze.mean(dim=tuple(range(1, x_squeeze.ndim)), keepdim=True)
        contrast = (x_squeeze - mean_density) / (mean_density.abs() + 1e-8)
        
        return contrast.unsqueeze(1)  # Add channel back
    
    def forward(self, pred, target, return_components=False):
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W] or [B, C, H, W, D]
            target: Target tensor (same shape as pred)
            return_components: If True, return dict with individual loss components
        
        Returns:
            total_loss: Combined loss
            (Optional) loss_dict: Dictionary with individual loss components
        """
        # 1. MSE Reconstruction Loss
        loss_mse = self.mse(pred, target)
        
        # 2. Gravitational Structure Loss (Laplacian consistency)
        if self.weight_laplacian > 0:
            laplacian_pred = self.compute_laplacian(pred)
            laplacian_target = self.compute_laplacian(target)
            loss_laplacian = self.mse(laplacian_pred, laplacian_target)
        else:
            loss_laplacian = torch.tensor(0.0, device=pred.device)
        
        # 3. Overdensity Field Loss (density contrast conservation)
        if self.weight_contrast > 0:
            contrast_pred = self.compute_density_contrast(pred)
            contrast_target = self.compute_density_contrast(target)
            loss_contrast = self.mse(contrast_pred, contrast_target)
        else:
            loss_contrast = torch.tensor(0.0, device=pred.device)
        
        # 4. Structure Boundaries Loss (gradient magnitude preservation)
        if self.weight_gradient > 0:
            gradient_pred = self.compute_gradient_magnitude(pred)
            gradient_target = self.compute_gradient_magnitude(target)
            loss_gradient = self.mse(gradient_pred, gradient_target)
        else:
            loss_gradient = torch.tensor(0.0, device=pred.device)
        
        # Total weighted loss
        total_loss = (
            self.weight_mse * loss_mse +
            self.weight_laplacian * loss_laplacian +
            self.weight_contrast * loss_contrast +
            self.weight_gradient * loss_gradient
        )
        
        if return_components:
            loss_dict = {
                'total': total_loss.item(),
                'mse': loss_mse.item(),
                'laplacian': loss_laplacian.item() if isinstance(loss_laplacian, torch.Tensor) else 0.0,
                'contrast': loss_contrast.item() if isinstance(loss_contrast, torch.Tensor) else 0.0,
                'gradient': loss_gradient.item() if isinstance(loss_gradient, torch.Tensor) else 0.0,
            }
            return total_loss, loss_dict
        
        return total_loss


class AdaptivePhysicsLoss(nn.Module):
    """
    Adaptive physics-informed loss with automatic weight balancing.
    
    Uses uncertainty weighting to automatically balance multiple loss terms.
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(self, spatial_dims=3, init_log_vars=None):
        """
        Args:
            spatial_dims: 2 or 3 for spatial dimensions
            init_log_vars: Initial log variance values for uncertainty weighting
                          [log_var_mse, log_var_laplacian, log_var_contrast, log_var_gradient]
        """
        super(AdaptivePhysicsLoss, self).__init__()
        self.spatial_dims = spatial_dims
        
        # Learnable log variances (uncertainty parameters)
        if init_log_vars is None:
            init_log_vars = [0.0, 0.0, 0.0, 0.0]  # Start with equal weighting
        
        self.log_var_mse = nn.Parameter(torch.tensor(init_log_vars[0]))
        self.log_var_laplacian = nn.Parameter(torch.tensor(init_log_vars[1]))
        self.log_var_contrast = nn.Parameter(torch.tensor(init_log_vars[2]))
        self.log_var_gradient = nn.Parameter(torch.tensor(init_log_vars[3]))
        
        self.mse = nn.MSELoss()
    
    def compute_gradient_magnitude(self, x):
        """Same as PhysicsInformedLoss."""
        x_squeeze = x.squeeze(1)
        
        if self.spatial_dims == 3:
            grad_z, grad_y, grad_x = torch.gradient(x_squeeze, dim=(-3, -2, -1))
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        else:
            grad_y, grad_x = torch.gradient(x_squeeze, dim=(-2, -1))
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return gradient_mag.unsqueeze(1)
    
    def compute_laplacian(self, x):
        """Same as PhysicsInformedLoss."""
        x_squeeze = x.squeeze(1)
        
        if self.spatial_dims == 3:
            grad_z, grad_y, grad_x = torch.gradient(x_squeeze, dim=(-3, -2, -1))
            grad2_z = torch.gradient(grad_z, dim=-3)[0]
            grad2_y = torch.gradient(grad_y, dim=-2)[0]
            grad2_x = torch.gradient(grad_x, dim=-1)[0]
            laplacian = grad2_x + grad2_y + grad2_z
        else:
            grad_y, grad_x = torch.gradient(x_squeeze, dim=(-2, -1))
            grad2_y = torch.gradient(grad_y, dim=-2)[0]
            grad2_x = torch.gradient(grad_x, dim=-1)[0]
            laplacian = grad2_x + grad2_y
        
        return laplacian.unsqueeze(1)
    
    def compute_density_contrast(self, x):
        """Same as PhysicsInformedLoss."""
        x_squeeze = x.squeeze(1)
        mean_density = x_squeeze.mean(dim=tuple(range(1, x_squeeze.ndim)), keepdim=True)
        contrast = (x_squeeze - mean_density) / (mean_density.abs() + 1e-8)
        return contrast.unsqueeze(1)
    
    def forward(self, pred, target, return_components=False):
        """
        Compute adaptive physics-informed loss with uncertainty weighting.
        
        Loss = Σ (1/(2*σ²)) * L_i + log(σ)
        where σ² = exp(log_var) and L_i is each loss component
        """
        # Compute individual losses
        loss_mse = self.mse(pred, target)
        loss_laplacian = self.mse(
            self.compute_laplacian(pred),
            self.compute_laplacian(target)
        )
        loss_contrast = self.mse(
            self.compute_density_contrast(pred),
            self.compute_density_contrast(target)
        )
        loss_gradient = self.mse(
            self.compute_gradient_magnitude(pred),
            self.compute_gradient_magnitude(target)
        )
        
        # Uncertainty weighting
        # Weight = 1/(2*exp(log_var))
        # Regularization = log_var / 2
        total_loss = (
            0.5 * torch.exp(-self.log_var_mse) * loss_mse + 
            0.5 * self.log_var_mse +
            0.5 * torch.exp(-self.log_var_laplacian) * loss_laplacian + 
            0.5 * self.log_var_laplacian +
            0.5 * torch.exp(-self.log_var_contrast) * loss_contrast + 
            0.5 * self.log_var_contrast +
            0.5 * torch.exp(-self.log_var_gradient) * loss_gradient + 
            0.5 * self.log_var_gradient
        )
        
        if return_components:
            loss_dict = {
                'total': total_loss.item(),
                'mse': loss_mse.item(),
                'laplacian': loss_laplacian.item(),
                'contrast': loss_contrast.item(),
                'gradient': loss_gradient.item(),
                'weight_mse': torch.exp(-self.log_var_mse).item(),
                'weight_laplacian': torch.exp(-self.log_var_laplacian).item(),
                'weight_contrast': torch.exp(-self.log_var_contrast).item(),
                'weight_gradient': torch.exp(-self.log_var_gradient).item(),
            }
            return total_loss, loss_dict
        
        return total_loss