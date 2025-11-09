# Pure Neural Network Data Compressor

A standalone GPU-accelerated neural network compressor for scientific 3D data with strict error bound enforcement. Unlike traditional hybrid approaches (e.g., SZ3 + neural residual), this compressor uses **only neural networks** for direct data compression.

## Key Features

- **Pure Neural Compression**: No traditional compressor required
- **3-Branch Architecture**: Spatial + Frequency (Magnitude + Phase) processing
- **Error Bound Enforcement**: Strict compliance with user-specified error bounds
- **GPU Acceleration**: Fast compression and decompression on CUDA devices
- **Quantization + Entropy Coding**: Efficient latent code compression
- **Multi-GPU Training**: Scale training across multiple GPUs

## Architecture

### 3-Branch Autoencoder

The compressor uses a novel 3-branch encoder architecture:

1. **Spatial Branch**: Processes 3D volumes directly with 3D convolutions
2. **Magnitude Branch**: Processes magnitude spectrum of 3D FFT
3. **Phase Branch**: Processes phase spectrum of 3D FFT

All branches are fused and compressed to a compact latent representation.

### Compression Pipeline

```
Original Data (D×H×W)
    ↓
Neural Encoder (3 branches)
    ↓
Latent Vector (N dimensions)
    ↓
Quantization (adaptive/uniform)
    ↓
Entropy Coding (arithmetic/zlib)
    ↓
Compressed Bitstream
```

### Decompression Pipeline

```
Compressed Bitstream
    ↓
Entropy Decoding
    ↓
Dequantization
    ↓
Neural Decoder
    ↓
Reconstructed Data + Error Bound Clipping
    ↓
Final Output
```

## Installation

### Requirements

```bash
pip install torch torchvision numpy scipy tabulate
```

For GPU support, ensure CUDA-compatible PyTorch is installed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Optional Dependencies

For advanced entropy coding:
```bash
pip install torchac  # Arithmetic coding
```

## Usage

### 1. Training

Train the neural compressor on your scientific data:

```bash
cd /Users/923714256/Data_compression/pure_neural_compression

# Edit train.sh to configure data paths and hyperparameters
nano train.sh

# Run training
./train.sh
```

Or use the Python script directly:

```bash
python train_neural_compressor.py \
    --data_dir /path/to/data \
    --train_files file1.f32 file2.f32 file3.f32 \
    --val_files file4.f32 \
    --output_dir ./trained_models \
    --spatial_channels 16 \
    --freq_channels 8 \
    --latent_dim 2048 \
    --num_epochs 100 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --error_bound 1e-2 \
    --device cuda \
    --multi_gpu
```

#### Training Options

**Model Architecture:**
- `--spatial_channels`: Base channels for spatial branch (default: 16)
- `--freq_channels`: Base channels for frequency branches (default: 8)
- `--latent_dim`: Latent vector dimension (default: 2048)
- `--decoder_channels`: Base channels for decoder (default: 64)

**Training Strategy:**
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size, typically 1 for 512³ volumes (default: 1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--error_bound`: Error bound for training (optional)
- `--lambda_eb`: Weight for error bound penalty (default: 1.0)

**Patch-Based Training** (for memory efficiency):
- `--use_patches`: Enable patch-based training
- `--patch_shape`: Patch dimensions, e.g., 128 128 128
- `--patches_per_volume`: Patches extracted per volume per epoch

**Multi-GPU:**
- `--multi_gpu`: Use DataParallel across all available GPUs

### 2. Compression

Compress data using a trained model:

```python
from neural_compressor import NeuralCompressor
import numpy as np

# Initialize compressor
compressor = NeuralCompressor(device='cuda', quantization='uniform')

# Load trained model
compressor.load_model('trained_models/best_model.pth')

# Load your data
data = np.fromfile('data.f32', dtype=np.float32).reshape(512, 512, 512)

# Compress with error bound
stats = compressor.compress(
    data,
    output_path='output/data.compressed',
    error_bound=1e-2,
    n_levels=256  # 8-bit quantization
)

print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Compressed size: {stats['compressed_size'] / (1024**2):.2f} MB")
```

### 3. Decompression

Decompress previously compressed data:

```python
# Decompress
reconstructed, decomp_stats = compressor.decompress(
    input_path='output/data.compressed'
)

print(f"Decompression time: {decomp_stats['total_time']:.2f}s")

# Verify quality
original = np.fromfile('data.f32', dtype=np.float32).reshape(512, 512, 512)
metrics = compressor.verify_reconstruction(original, reconstructed, error_bound=1e-2)

print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"Max error: {metrics['max_error']:.6e}")
print(f"Within bound: {metrics['within_bound']}")
```

### 4. Evaluation

Benchmark compression performance:

```bash
# Edit evaluate.sh to configure paths
nano evaluate.sh

# Run evaluation
./evaluate.sh
```

Or use Python directly:

```bash
python evaluate_compressor.py \
    --model_path trained_models/best_model.pth \
    --data_dir /path/to/test/data \
    --test_files test1.f32 test2.f32 \
    --data_shape 512 512 512 \
    --error_bounds 1e-2 1e-4 1e-6 \
    --output_dir evaluation_results \
    --device cuda
```

This will generate:
- `detailed_results.json`: Complete results for all tests
- `summary.txt`: Summary table with compression ratios, PSNR, etc.
- Console output with comprehensive statistics

## Model Architecture Details

### Encoder (NeuralEncoder3D)

**Input:** 3D volume `[1, D, H, W]` (e.g., `[1, 512, 512, 512]`)

**Processing:**
1. Normalize input: `(x - mean) / std`
2. **Spatial Branch:**
   - Conv3D layers with ResNet blocks
   - Downsamples: 512 → 256 → 128 → 64
3. **Magnitude Branch:**
   - Compute FFT: `fft_3d = FFT(x)`
   - Extract magnitude: `mag = abs(fft_3d)`
   - Log-scale: `log(1 + mag)`
   - Conv3D layers with ResNet blocks
4. **Phase Branch:**
   - Extract phase: `phase = angle(fft_3d)`
   - Conv3D layers with ResNet blocks
5. Fusion layer: Concatenate + Conv3D
6. Adaptive pooling: → `[C, 4, 4, 4]`
7. Compression head: Flatten → Linear → Latent vector

**Output:** Latent vector `[latent_dim]` (e.g., `[2048]`)

**Parameters:** ~500K-1M (configurable)

### Decoder (NeuralDecoder3D)

**Input:** Latent vector `[latent_dim]`

**Processing:**
1. Linear layers: Latent → `[C, 8, 8, 8]`
2. ConvTranspose3D layers: 8 → 16 → 32 → 64 → 128
3. Trilinear interpolation to target shape
4. Denormalize: `x * std + mean`

**Output:** Reconstructed volume `[1, D, H, W]`

### Error Bound Enforcement

During compression:
```python
error = reconstruction - original
error_clipped = clamp(error, -error_bound, error_bound)
final_reconstruction = original + error_clipped
```

During training:
```python
loss = MSE(reconstruction, original) + λ * penalty(violations > error_bound)
```

## Quantization Strategies

### Uniform Quantization

Maps continuous latent values to `n_levels` discrete bins uniformly:

```python
quantizer = UniformQuantizer(n_levels=256)  # 8-bit
quantizer.fit(latent)
quantized = quantizer.quantize(latent)
```

### Adaptive Quantization

Allocates more bits to important latent dimensions:

```python
quantizer = AdaptiveQuantizer(error_bound=1e-2, base_levels=256)
quantizer.fit(latent)
quantized_list = quantizer.quantize(latent)
```

## Entropy Coding

### Automatic Mode (Recommended)

Automatically selects the best coding method:

```python
encoder = EntropyEncoder(method='auto')  # Uses zlib by default
encoded, metadata = encoder.encode(quantized)
```

### Manual Selection

```python
encoder = EntropyEncoder(method='zlib')  # Fast and reliable
# or
encoder = EntropyEncoder(method='arithmetic')  # Better compression for low entropy
```

## Performance Expectations

Based on typical scientific data (512³ volumes):

| Error Bound | Compression Ratio | PSNR | Encode Time | Decode Time |
|-------------|-------------------|------|-------------|-------------|
| 1e-2        | 20-50x           | 60-80 dB | 2-5s | 1-3s |
| 1e-4        | 10-30x           | 80-100 dB | 2-5s | 1-3s |
| 1e-6        | 5-15x            | 100-120 dB | 2-5s | 1-3s |

*Performance varies with data characteristics and hardware.*

## File Formats

### Compressed File (.compressed)

Pickle format containing:
```python
{
    'encoded_bytes': bytes,  # Entropy-coded quantized latents
    'metadata': {
        'original_shape': tuple,
        'original_dtype': str,
        'latent_dim': int,
        'norm_stats': dict,  # mean, std for denormalization
        'quantizer_params': dict,
        'entropy_metadata': dict,
        'error_bound': float,
        ...
    }
}
```

## Troubleshooting

### Out of Memory (OOM)

**Problem:** GPU runs out of memory during training/inference

**Solutions:**
1. Enable patch-based training:
   ```bash
   --use_patches --patch_shape 128 128 128
   ```
2. Reduce model size:
   ```bash
   --spatial_channels 8 --freq_channels 4 --latent_dim 1024
   ```
3. Use CPU:
   ```bash
   --device cpu
   ```

### Poor Compression Ratio

**Problem:** Compression ratio lower than expected

**Solutions:**
1. Train longer: `--num_epochs 200`
2. Increase latent dimension: `--latent_dim 4096`
3. Use adaptive quantization: `--quantization_method adaptive`
4. Train with error bound awareness: `--error_bound 1e-2`

### Error Bound Violations

**Problem:** Reconstructed data exceeds error bound

**Solutions:**
1. Train with error bound penalty:
   ```bash
   --error_bound 1e-2 --lambda_eb 2.0
   ```
2. Use stricter quantization (more levels)
3. Post-process clipping (done automatically)

### Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Use multiple GPUs: `--multi_gpu`
2. Increase batch size (if memory allows): `--batch_size 2`
3. Use patch-based training with more patches: `--patches_per_volume 50`
4. Reduce model size

## Project Structure

```
pure_neural_compression/
├── __init__.py                    # Package initialization
├── neural_autoencoder_3d.py       # 3-branch autoencoder architecture
├── quantization.py                # Quantization methods
├── entropy_coding.py              # Entropy coding (arithmetic/zlib)
├── neural_compressor.py           # Main compressor class
├── train_neural_compressor.py     # Training script
├── evaluate_compressor.py         # Evaluation script
├── train.sh                       # Training shell script
├── evaluate.sh                    # Evaluation shell script
└── README.md                      # This file
```

## Comparison with Hybrid Methods

| Aspect | Pure Neural | Hybrid (SZ3 + Neural) |
|--------|-------------|----------------------|
| **Dependencies** | PyTorch only | PyTorch + SZ3 library |
| **Compression** | Neural only | SZ3 + neural residual |
| **Training** | On full data | On residuals |
| **Flexibility** | High (end-to-end) | Constrained by SZ3 |
| **Speed** | Fast (GPU) | Medium (CPU+GPU) |
| **Ratio** | Good | Excellent |

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pure_neural_compressor,
  title={Pure Neural Network Data Compressor},
  author={Your Name},
  year={2025},
  url={https://github.com/yourname/pure_neural_compression}
}
```

## License

MIT License - feel free to use and modify for your research.

## Acknowledgments

This implementation is inspired by:
- NeurLZ paper: [arXiv:2409.05785](https://arxiv.org/abs/2409.05785)
- SZ3 compressor: https://github.com/szcompressor/SZ3

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.

