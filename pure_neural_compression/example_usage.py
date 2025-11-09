"""
Example usage of Pure Neural Compressor.

This script demonstrates the complete workflow:
1. Load a trained model
2. Compress data
3. Decompress data
4. Verify quality
"""

import numpy as np
import torch
from pathlib import Path
import sys

import neural_compressor
from neural_compressor import NeuralCompressor


def main():
    print("="*60)
    print("Pure Neural Compressor - Example Usage")
    print("="*60)
    
    # Configuration
    MODEL_PATH = "trained_models/best_model.pth"
    DATA_FILE = "data/test_volume.f32"
    OUTPUT_DIR = "compressed_output"
    
    DATA_SHAPE = (512, 512, 512)
    ERROR_BOUND = 1e-2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Data: {DATA_FILE}")
    print(f"  Shape: {DATA_SHAPE}")
    print(f"  Error Bound: {ERROR_BOUND}")
    print(f"  Device: {DEVICE}")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train a model first using train.sh")
        sys.exit(1)
    
    # Check if data exists
    if not Path(DATA_FILE).exists():
        print(f"\nWarning: Data file not found at {DATA_FILE}")
        print("Generating random test data for demonstration...")
        data = np.random.randn(*DATA_SHAPE).astype(np.float32)
    else:
        print(f"\nLoading data from {DATA_FILE}...")
        data = np.fromfile(DATA_FILE, dtype=np.float32).reshape(DATA_SHAPE)
    
    print(f"Data loaded:")
    print(f"  Shape: {data.shape}")
    print(f"  Range: [{data.min():.6e}, {data.max():.6e}]")
    print(f"  Size: {data.nbytes / (1024**2):.2f} MB")
    
    # Initialize compressor
    print(f"\nInitializing compressor...")
    compressor = NeuralCompressor(
        device=DEVICE,
        quantization='uniform',
        entropy_coding='auto'
    )
    
    # Load model
    print(f"Loading model...")
    compressor.load_model(MODEL_PATH)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    compressed_file = Path(OUTPUT_DIR) / "test_volume.compressed"
    
    # Compress
    print(f"\n{'='*60}")
    print("COMPRESSION")
    print("="*60)
    
    compress_stats = compressor.compress(
        data,
        output_path=compressed_file,
        error_bound=ERROR_BOUND,
        n_levels=256  # 8-bit quantization
    )
    
    print(f"\nCompression Results:")
    print(f"  Original size: {compress_stats['original_size'] / (1024**2):.2f} MB")
    print(f"  Compressed size: {compress_stats['compressed_size'] / (1024**2):.2f} MB")
    print(f"  Compression ratio: {compress_stats['compression_ratio']:.2f}x")
    print(f"  Encoding time: {compress_stats['encode_time']:.2f}s")
    print(f"  Quantization time: {compress_stats['quantization_time']:.2f}s")
    print(f"  Entropy coding time: {compress_stats['entropy_coding_time']:.2f}s")
    print(f"  Total time: {compress_stats['total_time']:.2f}s")
    print(f"  Throughput: {compress_stats['throughput_MB_s']:.2f} MB/s")
    
    # Decompress
    print(f"\n{'='*60}")
    print("DECOMPRESSION")
    print("="*60)
    
    reconstructed, decompress_stats = compressor.decompress(
        input_path=compressed_file,
        error_bound=ERROR_BOUND
    )
    
    print(f"\nDecompression Results:")
    print(f"  Entropy decode time: {decompress_stats['entropy_decode_time']:.2f}s")
    print(f"  Dequantization time: {decompress_stats['dequantization_time']:.2f}s")
    print(f"  Decode time: {decompress_stats['decode_time']:.2f}s")
    print(f"  Total time: {decompress_stats['total_time']:.2f}s")
    print(f"  Throughput: {decompress_stats['throughput_MB_s']:.2f} MB/s")
    
    # Verify quality
    print(f"\n{'='*60}")
    print("QUALITY VERIFICATION")
    print("="*60)
    
    metrics = compressor.verify_reconstruction(data, reconstructed, ERROR_BOUND)
    
    print(f"\nQuality Metrics:")
    print(f"  Max error: {metrics['max_error']:.6e}")
    print(f"  Mean error: {metrics['mean_error']:.6e}")
    print(f"  MSE: {metrics['mse']:.6e}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  NRMSE: {metrics['nrmse']:.6f}")
    print(f"  Error bound: {metrics['error_bound']:.6e}")
    print(f"  Within bound: {metrics['within_bound']}")
    print(f"  Violation ratio: {metrics['violation_ratio']:.6%}")
    
    if metrics['within_bound']:
        print(f"\n✓ Compression successful! Error bound satisfied.")
    else:
        print(f"\n⚠ Warning: Error bound violated in {metrics['violation_ratio']:.2%} of points")
    
    # Save reconstructed data (optional)
    save_reconstructed = False
    if save_reconstructed:
        output_file = Path(OUTPUT_DIR) / "test_volume_reconstructed.f32"
        reconstructed.astype(np.float32).tofile(output_file)
        print(f"\nReconstructed data saved to {output_file}")
    
    print(f"\n{'='*60}")
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

