# Pure Neural Compression - Quick Start Guide

This guide will help you get started with training and using the pure neural compressor.

## Step 1: Verify Installation

```bash
cd /Users/923714256/Data_compression/pure_neural_compression
python test_installation.py
```

You should see all tests pass. If CUDA is available, it will be automatically detected and used.

## Step 2: Prepare Your Data

The compressor expects 3D scientific data in binary format (.f32 files).

Your data should be organized like:
```
/path/to/data/
├── field1.f32  # 512×512×512 float32 array
├── field2.f32
├── field3.f32
└── ...
```

**Data Format:** Raw binary files containing float32 values in C-order (depth, height, width).

## Step 3: Configure Training

Edit `train.sh` to set your data paths and hyperparameters:

```bash
nano train.sh
```

Key settings to modify:
- `DATA_DIR`: Path to your training data
- `OUTPUT_DIR`: Where to save trained models
- `TRAIN_FILES`: List of files for training
- `VAL_FILES`: List of files for validation (optional)
- `ERROR_BOUND`: Target error bound (e.g., 1e-2)

## Step 4: Train the Model

### Option A: Using the Shell Script (Recommended)

```bash
./train.sh
```

### Option B: Using Python Directly

```bash
python train_neural_compressor.py \
    --data_dir /path/to/SDRBENCH-data \
    --train_files baryon_density.f32 dark_matter_density.f32 \
    --val_files temperature.f32 \
    --output_dir ./trained_models \
    --num_epochs 100 \
    --error_bound 1e-2 \
    --device cuda \
    --multi_gpu
```

### Training Tips

**For small datasets (<5 volumes):**
- Use full volume training (don't use `--use_patches`)
- Increase epochs: `--num_epochs 200`
- May overfit, so monitor validation loss

**For large datasets (>10 volumes):**
- Use patch-based training: `--use_patches --patch_shape 128 128 128`
- More patches per volume: `--patches_per_volume 50`
- Standard epochs: `--num_epochs 100`

**For memory issues:**
- Enable patch mode
- Reduce model size: `--spatial_channels 8 --freq_channels 4 --latent_dim 1024`
- Reduce batch size to 1 (default)

**Training time estimates (single GPU):**
- Small model (512 latent): ~2-4 hours for 100 epochs on 5 volumes
- Medium model (2048 latent): ~4-8 hours for 100 epochs on 5 volumes
- Large model (4096 latent): ~8-16 hours for 100 epochs on 5 volumes

## Step 5: Monitor Training

Training progress is saved to:
- `trained_models/config.json`: Training configuration
- `trained_models/history.json`: Loss curves
- `trained_models/best_model.pth`: Best model (lowest validation loss)
- `trained_models/final_model.pth`: Final model after all epochs
- `trained_models/checkpoint_epoch_N.pth`: Periodic checkpoints

**Check validation loss:**
```bash
python -c "import json; print(json.load(open('trained_models/history.json'))['val_loss'][-1])"
```

## Step 6: Compress Data

### Option A: Using Python Script

```python
from neural_compressor import NeuralCompressor
import numpy as np

# Initialize and load model
compressor = NeuralCompressor(device='cuda')
compressor.load_model('trained_models/best_model.pth')

# Load data
data = np.fromfile('data/test_volume.f32', dtype=np.float32).reshape(512, 512, 512)

# Compress
stats = compressor.compress(
    data,
    output_path='compressed/test.compressed',
    error_bound=1e-2,
    n_levels=256
)

print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
```

### Option B: Using Example Script

```bash
python example_usage.py
```

(Edit the script to set your model path and data file)

## Step 7: Evaluate Performance

Edit `evaluate.sh` and run:

```bash
./evaluate.sh
```

This will:
- Test compression on held-out data
- Measure compression ratios
- Verify error bound compliance
- Benchmark speed
- Generate summary tables

Results saved to: `evaluation_results/`

## Common Use Cases

### Case 1: Maximum Compression (relaxed error bound)

```python
compressor.compress(data, 'output.compressed', error_bound=1e-2, n_levels=128)
# Expect: 30-50x compression, ~60-80 dB PSNR
```

### Case 2: Balanced Compression

```python
compressor.compress(data, 'output.compressed', error_bound=1e-4, n_levels=256)
# Expect: 15-30x compression, ~80-100 dB PSNR
```

### Case 3: High Precision (tight error bound)

```python
compressor.compress(data, 'output.compressed', error_bound=1e-6, n_levels=512)
# Expect: 5-15x compression, ~100-120 dB PSNR
```

## Troubleshooting

### "CUDA out of memory"
```bash
# Solution 1: Use patch-based training
./train.sh  # (enable USE_PATCHES=true in script)

# Solution 2: Reduce model size
python train_neural_compressor.py --spatial_channels 8 --latent_dim 1024 ...

# Solution 3: Use CPU
python train_neural_compressor.py --device cpu ...
```

### "Error bound violated"
```bash
# Solution: Train with error bound awareness
python train_neural_compressor.py --error_bound 1e-2 --lambda_eb 2.0 ...
```

### "Poor compression ratio"
```bash
# Solution 1: Train longer
python train_neural_compressor.py --num_epochs 200 ...

# Solution 2: Increase latent dimension
python train_neural_compressor.py --latent_dim 4096 ...

# Solution 3: Use adaptive quantization
# (Set in neural_compressor: quantization='adaptive')
```

### "Training is slow"
```bash
# Solution 1: Use multiple GPUs
python train_neural_compressor.py --multi_gpu ...

# Solution 2: Use patches with more workers
python train_neural_compressor.py --use_patches --patches_per_volume 50 ...

# Solution 3: Reduce model size
python train_neural_compressor.py --spatial_channels 8 --freq_channels 4 ...
```

## Performance Expectations

Based on typical scientific data (512³ volumes):

| Model Size | Training Time | Compression Ratio | Quality (PSNR) |
|------------|---------------|-------------------|----------------|
| Small      | 2-4 hours     | 10-25x           | 50-70 dB       |
| Medium     | 4-8 hours     | 20-40x           | 60-80 dB       |
| Large      | 8-16 hours    | 30-60x           | 70-90 dB       |

*Assumes single GPU (V100/A100), error bound 1e-2, 100 epochs*

## Next Steps

1. **Experiment with hyperparameters**: Try different latent dimensions, error bounds
2. **Fine-tune for your data**: Adjust model architecture for your specific dataset
3. **Benchmark against baselines**: Compare with SZ3, ZFP, or other compressors
4. **Deploy**: Integrate into your scientific workflow

## Getting Help

- Read the full README.md for detailed documentation
- Check examples in example_usage.py
- Review training outputs in trained_models/
- Examine evaluation results in evaluation_results/

## Citation

If you use this compressor in your research, please cite appropriately.

---

Happy compressing! 🚀

