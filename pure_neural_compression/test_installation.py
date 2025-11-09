"""
Test script to verify Pure Neural Compressor installation.

Run this to check if all components are working correctly.
"""

import sys
import numpy as np
import torch

print("="*60)
print("Pure Neural Compressor - Installation Test")
print("="*60)

# Test 1: Import modules
print("\n[1/7] Testing module imports...")
try:
    import neural_autoencoder_3d
    from neural_autoencoder_3d import NeuralAutoencoder3D, NeuralEncoder3D, NeuralDecoder3D
    import quantization
    from quantization import UniformQuantizer, AdaptiveQuantizer
    import entropy_coding
    from entropy_coding import EntropyEncoder, EntropyDecoder
    import neural_compressor
    from neural_compressor import NeuralCompressor
    import utils
    from utils import compute_data_stats, estimate_compression_ratio
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2/7] Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
else:
    print("⚠ CUDA is not available (CPU mode only)")

# Test 3: Create model
print("\n[3/7] Creating neural autoencoder...")
try:
    model = NeuralAutoencoder3D(
        spatial_channels=8,
        freq_channels=4,
        latent_dim=512,
        decoder_channels=32,
        output_shape=(64, 64, 64)
    )
    print(f"✓ Model created successfully")
    print(f"  Parameters: {model.count_parameters():,}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test 4: Test forward pass
print("\n[4/7] Testing forward pass...")
try:
    test_input = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        reconstruction, latent = model(test_input)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Output shape: {reconstruction.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 5: Test quantization
print("\n[5/7] Testing quantization...")
try:
    test_latent = np.random.randn(512).astype(np.float32)
    
    # Uniform quantization
    quantizer = UniformQuantizer(n_levels=256)
    quantizer.fit(test_latent)
    quantized = quantizer.quantize(test_latent)
    dequantized = quantizer.dequantize(quantized)
    
    quant_error = np.abs(test_latent - dequantized).max()
    print(f"✓ Quantization successful")
    print(f"  Quantization levels: 256")
    print(f"  Max quantization error: {quant_error:.6f}")
except Exception as e:
    print(f"✗ Quantization failed: {e}")
    sys.exit(1)

# Test 6: Test entropy coding
print("\n[6/7] Testing entropy coding...")
try:
    test_quantized = np.random.randint(0, 256, size=1000, dtype=np.uint8)
    
    encoder = EntropyEncoder(method='auto')
    encoded, metadata = encoder.encode(test_quantized)
    
    decoder = EntropyDecoder()
    decoded = decoder.decode(encoded, metadata)
    
    if np.array_equal(test_quantized, decoded):
        compression_ratio = test_quantized.nbytes / len(encoded)
        print(f"✓ Entropy coding successful")
        print(f"  Original size: {test_quantized.nbytes} bytes")
        print(f"  Compressed size: {len(encoded)} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    else:
        print(f"✗ Entropy coding decode mismatch")
        sys.exit(1)
except Exception as e:
    print(f"✗ Entropy coding failed: {e}")
    sys.exit(1)

# Test 7: Test compressor (without trained model)
print("\n[7/7] Testing compressor initialization...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compressor = NeuralCompressor(device=device, quantization='uniform')
    print(f"✓ Compressor initialized successfully")
    print(f"  Device: {device}")
    print(f"  Quantization: uniform")
except Exception as e:
    print(f"✗ Compressor initialization failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("Installation Test Summary")
print("="*60)
print("✓ All tests passed!")
print("\nNext steps:")
print("  1. Prepare your training data (3D .f32 files)")
print("  2. Edit train.sh with your data paths")
print("  3. Run: ./train.sh")
print("  4. After training, use evaluate.sh to test compression")
print("\nFor detailed usage, see README.md")
print("="*60)

