#!/bin/bash
#SBATCH --job-name=pure_neural_compression        # Job name
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=8:00:00                   # Time limit hrs:min:sec
#SBATCH --output=pure_neural_compression_%j.out   # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:4
# Script to train the residual encoder for NeurLZ-SZ3 compression

# Training script for Pure Neural Compressor
# Multi-GPU training for SDRBENCH data
srun --partition=gpucluster nvidia-smi


# Configuration
DATA_DIR="/Users/923714256/Data_compression/SDRBENCH-EXASKY-NYX-512x512x512"
OUTPUT_DIR="/Users/923714256/Data_compression/pure_neural_compression/models/"

# Training files (adjust based on your dataset)
TRAIN_FILES=(
    "velocity_x.f32"
    "velocity_y.f32"
)

# Validation files (optional)
VAL_FILES=(
    "velocity_z.f32"
)

# Model hyperparameters
SPATIAL_CHANNELS=16
FREQ_CHANNELS=16
LATENT_DIM=512
DECODER_CHANNELS=32

# Training hyperparameters
NUM_EPOCHS=100
BATCH_SIZE=4  # Use 1 for full 512^3 volumes
LEARNING_RATE=1e-2
ERROR_BOUND=2 # Optional: set to train with error bound awareness
LAMBDA_EB=1.0

# Patch-based training (for memory efficiency)
USE_PATCHES=True
PATCH_SHAPE="128 128 128"
PATCHES_PER_VOLUME=20

# Device configuration
DEVICE="cuda"
MULTI_GPU=true

# Save configuration
SAVE_INTERVAL=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Build training command
CMD="srun --partition=gpucluster python train_neural_compressor.py \
    --data_dir $DATA_DIR \
    --train_files ${TRAIN_FILES[@]} \
    --output_dir $OUTPUT_DIR \
    --spatial_channels $SPATIAL_CHANNELS \
    --freq_channels $FREQ_CHANNELS \
    --latent_dim $LATENT_DIM \
    --decoder_channels $DECODER_CHANNELS \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_interval $SAVE_INTERVAL \
    --device $DEVICE"

# Add validation files if provided
if [ ${#VAL_FILES[@]} -gt 0 ]; then
    CMD="$CMD --val_files ${VAL_FILES[@]}"
fi

# Add error bound if specified
if [ ! -z "$ERROR_BOUND" ]; then
    CMD="$CMD --error_bound $ERROR_BOUND --lambda_eb $LAMBDA_EB"
fi

# Add patch-based training if enabled
if [ "$USE_PATCHES" = true ]; then
    CMD="$CMD --use_patches --patch_shape $PATCH_SHAPE --patches_per_volume $PATCHES_PER_VOLUME"
fi

# Add multi-GPU flag
if [ "$MULTI_GPU" = true ]; then
    CMD="$CMD --multi_gpu"
fi

# Print command
echo "Training command:"
echo "$CMD"
echo ""

# Run training
eval $CMD

