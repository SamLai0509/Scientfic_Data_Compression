import torch
from torch import nn
import torch.nn.functional as F

def _central_diff_3d(u, axis, dx=1.0):
    # u: [N, 1, D, H, W], central differences with replicate padding
    pad = [0,0, 0,0, 0,0]  # W,H,D (pairs)
    if axis == 2:   # x/W (last)
        pad = [1,1, 0,0, 0,0]
        k = torch.tensor([[-0.5, 0.0, 0.5]], device=u.device, dtype=u.dtype).view(1,1,1,1,3)
    elif axis == 1: # y/H
        pad = [0,0, 1,1, 0,0]
        k = torch.tensor([[-0.5, 0.0, 0.5]], device=u.device, dtype=u.dtype).view(1,1,1,3,1)
    elif axis == 0: # z/D
        pad = [0,0, 0,0, 1,1]
        k = torch.tensor([[-0.5, 0.0, 0.5]], device=u.device, dtype=u.dtype).view(1,1,3,1,1)
    else:
        raise ValueError("axis must be 0/1/2 for (z,y,x).")
    up = F.pad(u, pad=pad, mode='replicate')
    du = F.conv3d(up, k) / dx
    return du

class DivergenceQoI(nn.Module):
    """
    Penalize divergence mismatch between enhanced and original velocity.
    Assumes channels are ordered with ['Vx','Vy','Vz'] available and their indices are provided.
    """
    def __init__(self, vx_idx=None, vy_idx=None, vz_idx=None, spacing=(1.0,1.0,1.0)):
        super().__init__()
        self.name = "divergence"
        self.vx_idx = vx_idx
        self.vy_idx = vy_idx
        self.vz_idx = vz_idx
        self.spacing = spacing

    def forward(self, x_enh, x_true, x_prime, field_names=None, aux=None):
        # Expect 3D tensor [N, C, D, H, W]
        assert x_enh.ndim == 5, "DivergenceQoI expects 3D volumes."
        N, C, D, H, W = x_enh.shape

        # find velocity channel indices
        vx, vy, vz = self.vx_idx, self.vy_idx, self.vz_idx
        if (vx is None or vy is None or vz is None) and field_names is not None:
            mapping = {name: i for i, name in enumerate(field_names)}
            vx = mapping.get("Vx", None)
            vy = mapping.get("Vy", None)
            vz = mapping.get("Vz", None)
        if vx is None or vy is None or vz is None:
            # nothing to do if we cannot find velocity
            return x_enh.new_zeros(()), {}

        dx, dy, dz = self.spacing[2], self.spacing[1], self.spacing[0]  # spacing in x,y,z

        # enhanced divergence
        Vx_enh = x_enh[:, vx:vx+1]
        Vy_enh = x_enh[:, vy:vy+1]
        Vz_enh = x_enh[:, vz:vz+1]
        dvx_dx = _central_diff_3d(Vx_enh, axis=2, dx=dx)
        dvy_dy = _central_diff_3d(Vy_enh, axis=1, dx=dy)
        dvz_dz = _central_diff_3d(Vz_enh, axis=0, dx=dz)
        div_enh = dvx_dx + dvy_dy + dvz_dz

        # true divergence
        Vx_true = x_true[:, vx:vx+1]
        Vy_true = x_true[:, vy:vy+1]
        Vz_true = x_true[:, vz:vz+1]
        dvx_dx_t = _central_diff_3d(Vx_true, axis=2, dx=dx)
        dvy_dy_t = _central_diff_3d(Vy_true, axis=1, dx=dy)
        dvz_dz_t = _central_diff_3d(Vz_true, axis=0, dx=dz)
        div_true = dvx_dx_t + dvy_dy_t + dvz_dz_t

        # L2 loss on divergence mismatch
        loss = torch.mean((div_enh - div_true) ** 2)

        stats = {
            "rmse_div": torch.sqrt(torch.mean((div_enh - div_true) ** 2)).detach(),
            "rmse_div_true": torch.sqrt(torch.mean(div_true ** 2)).detach(),
        }
        return loss, stats