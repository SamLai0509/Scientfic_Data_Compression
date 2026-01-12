import math
import torch
from torch import nn

def _radial_bins(shape, device, dtype, spacing=(1.0, 1.0, 1.0), is_3d=True, K=8):
    """
    Precompute radial bin indices for rfftn grid.
    For 3D: shape = (D, H, W). For 2D: (H, W).
    """
    if is_3d:
        D, H, W = shape
        kz = torch.fft.fftfreq(D, d=spacing[0], device=device, dtype=dtype)      # [-.., ..]
        ky = torch.fft.fftfreq(H, d=spacing[1], device=device, dtype=dtype)
        kx = torch.fft.rfftfreq(W, d=spacing[2], device=device, dtype=dtype)     # [0..]
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
        kr = torch.sqrt(KZ*KZ + KY*KY + KX*KX)
    else:
        H, W = shape
        ky = torch.fft.fftfreq(H, d=spacing[0], device=device, dtype=dtype)
        kx = torch.fft.rfftfreq(W, d=spacing[1], device=device, dtype=dtype)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        kr = torch.sqrt(KY*KY + KX*KX)

    kmin, kmax = float(kr.min()), float(kr.max()) + 1e-12
    edges = torch.linspace(kmin, kmax, K+1, device=device, dtype=dtype)
    # for each frequency cell, get which bin it falls into
    bin_idx = torch.bucketize(kr.flatten(), edges, right=False).clamp(1, K) - 1  # [num_cells], 0..K-1
    return bin_idx.view_as(kr), edges

class SpectralPowerQoI(nn.Module):
    """
    Match shell-averaged power spectrum between enhanced and original data.
    Works for 2D or 3D, uses rfftn for the last axis.
    """
    def __init__(self, K=8, fields=None, spacing=(1.0,1.0,1.0)):
        super().__init__()
        self.name = "spectral"
        self.K = K
        self.fields = fields  # None = all channels
        self.spacing = spacing
        self._cache = {}  # {(shape, device): (bin_idx, edges)}

    def _get_bins(self, spatial_shape, device, dtype, is_3d=True):
        key = (spatial_shape, device, is_3d)
        if key not in self._cache:
            self._cache[key] = _radial_bins(spatial_shape, device, dtype,
                                            spacing=self.spacing, is_3d=is_3d, K=self.K)
        return self._cache[key]

    def forward(self, x_enh, x_true, x_prime, field_names=None, aux=None):
        # x_*: [N, C, D, H, W] or [N, C, H, W]
        is_3d = (x_enh.ndim == 5)
        device, dtype = x_enh.device, x_enh.dtype
        N, C = x_enh.shape[:2]
        spatial = x_enh.shape[2:]

        bin_idx, edges = self._get_bins(spatial, device, dtype, is_3d=is_3d)

        # choose channels
        if self.fields and field_names:
            sel = [field_names.index(f) for f in self.fields if f in field_names]
            if not sel:
                sel = list(range(C))
        else:
            sel = list(range(C))

        # compute per-channel, per-bin power mismatch
        losses = []
        stats = {}
        for c in sel:
            xc = x_enh[:, c]
            yc = x_true[:, c]
            # rfftn last axis
            Xf = torch.fft.rfftn(xc, dim=tuple(range(1, xc.ndim)))
            Yf = torch.fft.rfftn(yc, dim=tuple(range(1, yc.ndim)))
            PX = (Xf.real**2 + Xf.imag**2)
            PY = (Yf.real**2 + Yf.imag**2)
            # average per radial bin
            bi = bin_idx  # [...], broadcast to batch
            PXb = PX.flatten(1)  # [N, M]
            PYb = PY.flatten(1)
            bi_flat = bi.flatten().unsqueeze(0).expand(N, -1)  # [N, M]

            # sums per bin
            K = self.K
            num = torch.zeros((N, K), device=device, dtype=dtype)
            den = torch.zeros((N, K), device=device, dtype=dtype) + 1e-12
            num.scatter_add_(1, bi_flat, PXb)
            den.scatter_add_(1, bi_flat, PYb)
            # relative error of power per bin
            rel = (num / den) - 1.0  # PX/PY - 1
            # L2 over bins, averaged over batch
            loss_c = (rel**2).mean()
            losses.append(loss_c)

            # small stats for logging (mean abs rel per bin)
            stats[f"channel_{c}/mean_abs_rel"] = rel.abs().mean().detach()

        loss = torch.stack(losses).mean() if losses else x_enh.new_zeros(())
        return loss, stats
