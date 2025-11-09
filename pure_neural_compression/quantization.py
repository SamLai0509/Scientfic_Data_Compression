"""
Quantization module for latent codes.

Implements uniform and adaptive quantization strategies for neural compression.
"""

import numpy as np
import torch


class UniformQuantizer:
    """
    Uniform quantization for latent codes.
    
    Maps continuous latent values to discrete bins uniformly.
    """
    
    def __init__(self, n_levels=256, value_range=None):
        """
        Initialize uniform quantizer.
        
        Args:
            n_levels: Number of quantization levels (e.g., 256 for 8-bit)
            value_range: Tuple of (min, max) values. If None, computed from data.
        """
        self.n_levels = n_levels
        self.value_range = value_range
        self.min_val = None
        self.max_val = None
        
    def fit(self, latent):
        """
        Fit quantizer to latent codes (determine value range).
        
        Args:
            latent: Latent codes (numpy array or torch tensor)
        """
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()
        
        if self.value_range is None:
            self.min_val = float(np.min(latent))
            self.max_val = float(np.max(latent))
        else:
            self.min_val, self.max_val = self.value_range
            
        # Add small epsilon to avoid division by zero
        if self.max_val == self.min_val:
            self.max_val = self.min_val + 1e-8
    
    def quantize(self, latent):
        """
        Quantize latent codes to discrete values.
        
        Args:
            latent: Continuous latent codes (numpy array)
        
        Returns:
            quantized: Quantized latent codes (numpy array of uint8/uint16)
        """
        if self.min_val is None or self.max_val is None:
            self.fit(latent)
        
        # Normalize to [0, 1]
        normalized = (latent - self.min_val) / (self.max_val - self.min_val)
        normalized = np.clip(normalized, 0, 1)
        
        # Quantize to [0, n_levels-1]
        quantized = np.round(normalized * (self.n_levels - 1))
        
        # Convert to appropriate integer type
        if self.n_levels <= 256:
            quantized = quantized.astype(np.uint8)
        else:
            quantized = quantized.astype(np.uint16)
        
        return quantized
    
    def dequantize(self, quantized):
        """
        Dequantize discrete codes back to continuous values.
        
        Args:
            quantized: Quantized latent codes (numpy array of integers)
        
        Returns:
            latent: Dequantized continuous latent codes (numpy array of float32)
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Quantizer must be fitted before dequantization")
        
        # Convert to float
        quantized = quantized.astype(np.float32)
        
        # Denormalize from [0, n_levels-1] to [0, 1]
        normalized = quantized / (self.n_levels - 1)
        
        # Scale back to original range
        latent = normalized * (self.max_val - self.min_val) + self.min_val
        
        return latent
    
    def get_params(self):
        """Get quantizer parameters for saving."""
        return {
            'n_levels': self.n_levels,
            'min_val': self.min_val,
            'max_val': self.max_val
        }
    
    def set_params(self, params):
        """Set quantizer parameters from saved values."""
        self.n_levels = params['n_levels']
        self.min_val = params['min_val']
        self.max_val = params['max_val']


class AdaptiveQuantizer:
    """
    Adaptive quantization that adjusts quantization levels based on error bound.
    
    Uses finer quantization for more critical components.
    """
    
    def __init__(self, error_bound, base_levels=256):
        """
        Initialize adaptive quantizer.
        
        Args:
            error_bound: Target error bound for reconstruction
            base_levels: Base number of quantization levels
        """
        self.error_bound = error_bound
        self.base_levels = base_levels
        self.quantizers = []
        self.channel_importance = None
        
    def fit(self, latent, original_data=None):
        """
        Fit quantizer to latent codes.
        
        Optionally use original data to determine importance of each latent dimension.
        
        Args:
            latent: Latent codes [N, D] where D is latent dimension
            original_data: Optional original data for importance estimation
        """
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()
        
        # Calculate importance based on variance of each latent dimension
        if latent.ndim == 1:
            latent = latent.reshape(1, -1)
        
        variance = np.var(latent, axis=0)
        self.channel_importance = variance / (np.sum(variance) + 1e-8)
        
        # Determine quantization levels for each dimension
        # More important dimensions get more levels
        n_dimensions = latent.shape[1]
        self.quantizers = []
        
        for i in range(n_dimensions):
            # Adaptive level allocation
            importance_factor = np.sqrt(self.channel_importance[i])
            n_levels = max(16, int(self.base_levels * importance_factor))
            n_levels = min(n_levels, 65536)  # Cap at 16-bit
            
            quantizer = UniformQuantizer(n_levels=n_levels)
            quantizer.fit(latent[:, i])
            self.quantizers.append(quantizer)
    
    def quantize(self, latent):
        """
        Quantize latent codes adaptively.
        
        Args:
            latent: Continuous latent codes (numpy array)
        
        Returns:
            quantized_list: List of quantized arrays (one per dimension)
        """
        if latent.ndim == 1:
            latent = latent.reshape(1, -1)
        
        quantized_list = []
        for i, quantizer in enumerate(self.quantizers):
            quantized = quantizer.quantize(latent[:, i])
            quantized_list.append(quantized)
        
        return quantized_list
    
    def dequantize(self, quantized_list):
        """
        Dequantize discrete codes.
        
        Args:
            quantized_list: List of quantized arrays
        
        Returns:
            latent: Dequantized continuous latent codes
        """
        dequantized_channels = []
        for i, quantized in enumerate(quantized_list):
            dequantized = self.quantizers[i].dequantize(quantized)
            dequantized_channels.append(dequantized)
        
        latent = np.stack(dequantized_channels, axis=-1)
        return latent
    
    def get_params(self):
        """Get quantizer parameters for saving."""
        return {
            'error_bound': self.error_bound,
            'base_levels': self.base_levels,
            'channel_importance': self.channel_importance.tolist() if self.channel_importance is not None else None,
            'quantizers': [q.get_params() for q in self.quantizers]
        }
    
    def set_params(self, params):
        """Set quantizer parameters from saved values."""
        self.error_bound = params['error_bound']
        self.base_levels = params['base_levels']
        self.channel_importance = np.array(params['channel_importance']) if params['channel_importance'] is not None else None
        
        self.quantizers = []
        for q_params in params['quantizers']:
            quantizer = UniformQuantizer(n_levels=q_params['n_levels'])
            quantizer.set_params(q_params)
            self.quantizers.append(quantizer)


def estimate_quantization_levels(error_bound, latent_dim, data_range=1.0):
    """
    Estimate appropriate number of quantization levels based on error bound.
    
    Args:
        error_bound: Target reconstruction error bound
        latent_dim: Dimensionality of latent space
        data_range: Expected range of data values
    
    Returns:
        n_levels: Recommended number of quantization levels
    """
    # Simple heuristic: smaller error bound requires more levels
    # This is a rough estimate and may need tuning
    
    # Assuming uniform distribution and propagation of quantization error
    # quantization_error ≈ range / (2 * n_levels)
    # For multi-dimensional latent, errors accumulate
    
    accumulated_error_factor = np.sqrt(latent_dim)
    required_precision = error_bound / (accumulated_error_factor * data_range)
    
    # Calculate required levels
    n_levels = int(1.0 / (2 * required_precision))
    
    # Constrain to reasonable range
    n_levels = max(16, min(n_levels, 65536))
    
    # Round to power of 2 for efficiency
    n_levels = 2 ** int(np.log2(n_levels))
    
    return n_levels


if __name__ == "__main__":
    # Test quantization
    print("Testing Quantization...")
    
    # Generate test latent codes
    np.random.seed(42)
    latent = np.random.randn(10, 2048).astype(np.float32)
    
    print(f"Original latent shape: {latent.shape}")
    print(f"Original latent range: [{latent.min():.4f}, {latent.max():.4f}]")
    
    # Test uniform quantization
    print("\n=== Uniform Quantization ===")
    uniform_q = UniformQuantizer(n_levels=256)
    uniform_q.fit(latent)
    
    quantized = uniform_q.quantize(latent)
    print(f"Quantized shape: {quantized.shape}, dtype: {quantized.dtype}")
    print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")
    
    dequantized = uniform_q.dequantize(quantized)
    print(f"Dequantized shape: {dequantized.shape}")
    print(f"Dequantized range: [{dequantized.min():.4f}, {dequantized.max():.4f}]")
    
    error = np.abs(latent - dequantized)
    print(f"Quantization error: mean={error.mean():.6f}, max={error.max():.6f}")
    
    # Compression ratio
    original_size = latent.nbytes
    quantized_size = quantized.nbytes
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    # Test adaptive quantization
    print("\n=== Adaptive Quantization ===")
    adaptive_q = AdaptiveQuantizer(error_bound=1e-3, base_levels=256)
    adaptive_q.fit(latent)
    
    quantized_list = adaptive_q.quantize(latent)
    print(f"Number of quantized channels: {len(quantized_list)}")
    print(f"Quantization levels per channel (first 10): {[q.max()+1 for q in quantized_list[:10]]}")
    
    dequantized_adaptive = adaptive_q.dequantize(quantized_list)
    print(f"Dequantized shape: {dequantized_adaptive.shape}")
    
    error_adaptive = np.abs(latent - dequantized_adaptive)
    print(f"Adaptive quantization error: mean={error_adaptive.mean():.6f}, max={error_adaptive.max():.6f}")
    
    # Test level estimation
    print("\n=== Quantization Level Estimation ===")
    for eb in [1e-2, 1e-4, 1e-6]:
        n_levels = estimate_quantization_levels(eb, latent_dim=2048)
        print(f"Error bound {eb:.1e} -> {n_levels} levels ({int(np.log2(n_levels))} bits)")

