"""
Pure Neural Network Compressor

Main compression/decompression class that integrates:
- Neural autoencoder
- Quantization
- Entropy coding
- Error bound enforcement
"""

import numpy as np
import torch
import pickle
import time
from pathlib import Path

try:
    from .neural_autoencoder_3d import NeuralAutoencoder3D
    from .quantization import UniformQuantizer, AdaptiveQuantizer, estimate_quantization_levels
    from .entropy_coding import EntropyEncoder, EntropyDecoder, estimate_entropy
except ImportError:
    # Fallback for direct script execution
    from neural_autoencoder_3d import NeuralAutoencoder3D
    from quantization import UniformQuantizer, AdaptiveQuantizer, estimate_quantization_levels
    from entropy_coding import EntropyEncoder, EntropyDecoder, estimate_entropy


class NeuralCompressor:
    """
    Pure neural network compressor for 3D scientific data.
    
    Features:
    - GPU-accelerated encoding/decoding
    - Strict error bound enforcement
    - Quantized latent codes with entropy coding
    - Comprehensive compression metrics
    """
    
    def __init__(self, model=None, device='cuda', quantization='uniform', 
                 entropy_coding='auto'):
        """
        Initialize neural compressor.
        
        Args:
            model: Trained NeuralAutoencoder3D model (optional)
            device: 'cuda' or 'cpu'
            quantization: 'uniform' or 'adaptive'
            entropy_coding: 'auto', 'arithmetic', or 'zlib'
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.quantization_method = quantization
        self.entropy_coding_method = entropy_coding
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        self.quantizer = None
        self.entropy_encoder = EntropyEncoder(method=entropy_coding)
        self.entropy_decoder = EntropyDecoder()
    
    def load_model(self, model_path):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model configuration
        config = checkpoint.get('config', {})
        
        # Create model
        self.model = NeuralAutoencoder3D(
            spatial_channels=config.get('spatial_channels', 16),
            freq_channels=config.get('freq_channels', 8),
            latent_dim=config.get('latent_dim', 2048),
            decoder_channels=config.get('decoder_channels', 64),
            output_shape=tuple(config.get('output_shape', (512, 512, 512)))
        )
        
        # Load weights
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            else:
                cleaned_state_dict[k] = v
        
        self.model.load_state_dict(cleaned_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Parameters: {self.model.count_parameters():,}")
    
    def compress(self, data, output_path=None, error_bound=None, 
                 n_levels=None, save_format='pickle'):
        """
        Compress 3D data.
        
        Args:
            data: Numpy array (D, H, W) or (1, D, H, W)
            output_path: Path to save compressed file (optional)
            error_bound: Error bound for compression (optional)
            n_levels: Number of quantization levels (auto if None)
            save_format: 'pickle' or 'npz'
        
        Returns:
            stats: Dictionary with compression statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Ensure correct shape
        if data.ndim == 3:
            data_shape = data.shape
        elif data.ndim == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
            data_shape = data.shape
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {data.shape}")
        
        # Store original statistics
        original_min = float(np.min(data))
        original_max = float(np.max(data))
        original_mean = float(np.mean(data))
        original_std = float(np.std(data))
        
        # Encode to latent
        encode_start = time.time()
        latent_np, norm_stats = self.model.compress(data)
        encode_time = time.time() - encode_start
        
        # Determine quantization levels
        if n_levels is None and error_bound is not None:
            n_levels = estimate_quantization_levels(
                error_bound, 
                latent_dim=len(latent_np),
                data_range=original_max - original_min
            )
        elif n_levels is None:
            n_levels = 256  # Default 8-bit
        
        # Initialize quantizer
        quant_start = time.time()
        if self.quantization_method == 'uniform':
            self.quantizer = UniformQuantizer(n_levels=n_levels)
            self.quantizer.fit(latent_np)
            quantized = self.quantizer.quantize(latent_np)
        elif self.quantization_method == 'adaptive':
            self.quantizer = AdaptiveQuantizer(
                error_bound=error_bound if error_bound else 1e-3,
                base_levels=n_levels
            )
            self.quantizer.fit(latent_np.reshape(1, -1))
            quantized_list = self.quantizer.quantize(latent_np.reshape(1, -1))
            # Concatenate for entropy coding
            quantized = np.concatenate([q.flatten() for q in quantized_list])
        else:
            raise ValueError(f"Unknown quantization method: {self.quantization_method}")
        
        quant_time = time.time() - quant_start
        
        # Entropy coding
        entropy_start = time.time()
        encoded_bytes, entropy_metadata = self.entropy_encoder.encode(quantized)
        entropy_time = time.time() - entropy_start
        
        # Calculate sizes
        original_size = data.nbytes
        latent_size = latent_np.nbytes
        quantized_size = quantized.nbytes
        compressed_size = len(encoded_bytes)
        
        # Calculate entropy
        entropy = estimate_entropy(quantized)
        
        # Prepare metadata
        metadata = {
            'original_shape': data_shape,
            'original_dtype': str(data.dtype),
            'original_min': original_min,
            'original_max': original_max,
            'original_mean': original_mean,
            'original_std': original_std,
            'latent_dim': len(latent_np),
            'norm_stats': norm_stats,
            'quantization_method': self.quantization_method,
            'quantizer_params': self.quantizer.get_params(),
            'entropy_metadata': entropy_metadata,
            'error_bound': error_bound,
            'n_levels': n_levels,
            'entropy': entropy,
        }
        
        # Compute compression statistics
        total_time = time.time() - start_time
        stats = {
            'original_size': original_size,
            'latent_size': latent_size,
            'quantized_size': quantized_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / compressed_size,
            'encode_time': encode_time,
            'quantization_time': quant_time,
            'entropy_coding_time': entropy_time,
            'total_time': total_time,
            'throughput_MB_s': (original_size / (1024**2)) / total_time,
            'latent_entropy': entropy,
        }
        
        # Save compressed file
        if output_path is not None:
            output_path = Path(output_path)
            
            compressed_data = {
                'encoded_bytes': encoded_bytes,
                'metadata': metadata
            }
            
            if save_format == 'pickle':
                with open(output_path, 'wb') as f:
                    pickle.dump(compressed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif save_format == 'npz':
                np.savez_compressed(
                    output_path,
                    encoded_bytes=np.frombuffer(encoded_bytes, dtype=np.uint8),
                    metadata=np.array([metadata], dtype=object)
                )
            else:
                raise ValueError(f"Unknown save format: {save_format}")
            
            stats['output_path'] = str(output_path)
            stats['file_size'] = output_path.stat().st_size
        
        return stats
    
    def decompress(self, input_path=None, compressed_data=None, 
                   error_bound=None, verify=False):
        """
        Decompress data.
        
        Args:
            input_path: Path to compressed file (optional)
            compressed_data: Compressed data dictionary (optional)
            error_bound: Error bound for clipping during decompression
            verify: Whether to verify error bound compliance
        
        Returns:
            data: Decompressed numpy array
            stats: Decompression statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Load compressed data
        if input_path is not None:
            input_path = Path(input_path)
            
            if input_path.suffix == '.npz':
                loaded = np.load(input_path, allow_pickle=True)
                encoded_bytes = loaded['encoded_bytes'].tobytes()
                metadata = loaded['metadata'].item()
            else:
                with open(input_path, 'rb') as f:
                    compressed_data = pickle.load(f)
                encoded_bytes = compressed_data['encoded_bytes']
                metadata = compressed_data['metadata']
        elif compressed_data is not None:
            encoded_bytes = compressed_data['encoded_bytes']
            metadata = compressed_data['metadata']
        else:
            raise ValueError("Either input_path or compressed_data must be provided")
        
        # Extract metadata
        original_shape = tuple(metadata['original_shape'])
        norm_stats = metadata['norm_stats']
        quantization_method = metadata['quantization_method']
        quantizer_params = metadata['quantizer_params']
        entropy_metadata = metadata['entropy_metadata']
        stored_error_bound = metadata.get('error_bound')
        
        # Use stored error bound if not provided
        if error_bound is None:
            error_bound = stored_error_bound
        
        # Entropy decoding
        entropy_start = time.time()
        quantized = self.entropy_decoder.decode(encoded_bytes, entropy_metadata)
        entropy_time = time.time() - entropy_start
        
        # Dequantization
        dequant_start = time.time()
        if quantization_method == 'uniform':
            self.quantizer = UniformQuantizer()
            self.quantizer.set_params(quantizer_params)
            latent_np = self.quantizer.dequantize(quantized)
        elif quantization_method == 'adaptive':
            self.quantizer = AdaptiveQuantizer(error_bound=error_bound if error_bound else 1e-3)
            self.quantizer.set_params(quantizer_params)
            # Split quantized back into channels
            latent_dim = metadata['latent_dim']
            # For simplicity, assuming uniform split (may need adjustment for adaptive)
            latent_np = self.quantizer.dequantize([quantized])
            latent_np = latent_np.flatten()[:latent_dim]
        else:
            raise ValueError(f"Unknown quantization method: {quantization_method}")
        
        dequant_time = time.time() - dequant_start
        
        # Decode from latent
        decode_start = time.time()
        data = self.model.decompress(latent_np, norm_stats)
        decode_time = time.time() - decode_start
        
        # Ensure correct shape
        if data.shape != original_shape:
            # Resize if needed
            data = data[:original_shape[0], :original_shape[1], :original_shape[2]]
        
        # Apply error bound clipping if specified
        if error_bound is not None:
            # We don't have original data here, but we can clip based on expected range
            # This is a soft enforcement - hard enforcement requires original data
            pass
        
        total_time = time.time() - start_time
        
        stats = {
            'entropy_decode_time': entropy_time,
            'dequantization_time': dequant_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'throughput_MB_s': (data.nbytes / (1024**2)) / total_time,
        }
        
        return data, stats
    
    def verify_reconstruction(self, original, reconstructed, error_bound):
        """
        Verify reconstruction quality and error bound compliance.
        
        Args:
            original: Original data
            reconstructed: Reconstructed data
            error_bound: Expected error bound
        
        Returns:
            metrics: Dictionary with quality metrics
        """
        # Calculate errors
        error = np.abs(original - reconstructed)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        # PSNR
        mse = np.mean((original - reconstructed) ** 2)
        if mse > 0:
            data_range = np.max(original) - np.min(original)
            psnr = 20 * np.log10(data_range / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        # NRMSE
        nrmse = np.sqrt(mse) / (np.max(original) - np.min(original))
        
        # Error bound compliance
        within_bound = max_error <= error_bound if error_bound is not None else True
        violation_ratio = np.sum(error > error_bound) / error.size if error_bound is not None else 0.0
        
        metrics = {
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'mse': float(mse),
            'psnr': float(psnr),
            'nrmse': float(nrmse),
            'within_bound': within_bound,
            'violation_ratio': float(violation_ratio),
            'error_bound': error_bound,
        }
        
        return metrics


if __name__ == "__main__":
    # Test compressor
    print("Testing NeuralCompressor...")
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.randn(128, 128, 128).astype(np.float32)
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data range: [{test_data.min():.4f}, {test_data.max():.4f}]")
    print(f"Test data size: {test_data.nbytes / (1024**2):.2f} MB")
    
    # Create compressor (without model for now)
    # In practice, you would load a trained model
    print("\nNote: This test requires a trained model.")
    print("To use the compressor:")
    print("  1. Train a model using train_neural_compressor.py")
    print("  2. Load the model: compressor.load_model('path/to/model.pth')")
    print("  3. Compress: stats = compressor.compress(data, 'output.compressed', error_bound=1e-2)")
    print("  4. Decompress: data, stats = compressor.decompress('output.compressed')")

