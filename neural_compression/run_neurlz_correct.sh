#!/bin/bash
#SBATCH --job-name=SZ3+NeurLZ        # Job name
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=8:00:00                   # Time limit hrs:min:sec
#SBATCH --output=5e-4_physics3d_xyz_physics_adaptive_loss%j.out   # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task

srun nvidia-smi

echo "========================================="
echo "NeurLZ Evaluation (Correct Implementation)"
echo "========================================="
echo "Following actual NeurLZ paper approach:"
echo "  1. SZ3 is PRIMARY compressor"
echo "  2. Tiny DNN (~3k params) trained ONLINE"
echo "  3. DNN predicts residuals from SZ3-decompressed"
echo "  4. Storage: SZ3_bytes + DNN_weights + outliers"
echo "========================================="

cd /Users/923714256/Data_compression/neural_compression

# ============================================================
# Multi-GPU Configuration
# ============================================================
# Set to "multi" to use all GPUs, or "single" to use one GPU
GPU_MODE="multi"  # Options: "multi" or "single"

echo "Scanning GPUs for available memory..."

# Get GPU memory info: index, total memory (MiB), used memory (MiB)
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)

if [ "$GPU_MODE" == "multi" ]; then
    # Multi-GPU mode: Use all available GPUs
    AVAILABLE_GPUS=()
    MAX_FREE_MEM=0
    BEST_GPU=0
    
    while IFS=',' read -r gpu_index total_mem used_mem; do
        # Remove whitespace
        gpu_index=$(echo $gpu_index | xargs)
        total_mem=$(echo $total_mem | xargs)
        used_mem=$(echo $used_mem | xargs)
        
        # Calculate free memory
        free_mem=$((total_mem - used_mem))
        
        echo "  GPU $gpu_index: ${free_mem} MiB free (${used_mem}/${total_mem} MiB used)"
        
        # Collect all available GPUs
        AVAILABLE_GPUS+=($gpu_index)
        
        # Track best GPU for reference
        if [ $free_mem -gt $MAX_FREE_MEM ]; then
            MAX_FREE_MEM=$free_mem
            BEST_GPU=$gpu_index
        fi
    done <<< "$GPU_INFO"
    
    # Set CUDA_VISIBLE_DEVICES to all GPUs (comma-separated) 
    CUDA_DEVICES=$(IFS=','; echo "${AVAILABLE_GPUS[*]}")
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
    echo "Multi-GPU mode: Using GPUs ${CUDA_DEVICES}"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
else
    # Single GPU mode: Select GPU with most free memory
    BEST_GPU=0
    MAX_FREE_MEM=0
    
    while IFS=',' read -r gpu_index total_mem used_mem; do
        # Remove whitespace
        gpu_index=$(echo $gpu_index | xargs)
        total_mem=$(echo $total_mem | xargs)
        used_mem=$(echo $used_mem | xargs)
        
        # Calculate free memory
        free_mem=$((total_mem - used_mem))
        
        echo "  GPU $gpu_index: ${free_mem} MiB free (${used_mem}/${total_mem} MiB used)"
        
        # Check if this GPU has more free memory
        if [ $free_mem -gt $MAX_FREE_MEM ]; then
            MAX_FREE_MEM=$free_mem
            BEST_GPU=$gpu_index
        fi
    done <<< "$GPU_INFO"
    
    echo "Single-GPU mode: Selected GPU $BEST_GPU with ${MAX_FREE_MEM} MiB free memory"
    export CUDA_VISIBLE_DEVICES=$BEST_GPU
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo ""

# Configuration
DATA_DIR="/Users/923714256/Data_compression/SDRBENCH-EXASKY-NYX-512x512x512"
OUTPUT_DIR="/Users/923714256/Data_compression/neural_compression/evaluation_results_correct"
SZ_LIB="/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so"

# Test files
TEST_FILES="dark_matter_density.f32" 

# Error bounds to test
ERROR_BOUNDS="1e-6"
ERROR_RELATIVE_BOUNDS="5e-4"
ERROR_PWR_BOUNDS="0"
# NeurLZ modes
EB_MODES="1"
MODEL="tiny_physics_residual_predictor"

# Online training parameters
ONLINE_EPOCHS=200
LEARNING_RATE=1e-3
MODEL_CHANNELS=4  # 4 channels → ~3k params (as in paper)
NUM_RES_BLOCKS=2
# Device
DEVICE="cuda"
# REMOVED: export CUDA_VISIBLE_DEVICES=2  (line 81 - this was overriding the selection)
SPATIAL_DIMS=3
SLICE_ORDER="zxy"
VAL_SPLIT=0.1
TRACK_LOSSES=True

mkdir -p $OUTPUT_DIR

echo ""
echo "Running NeurLZ evaluation..."
echo "  Online epochs: $ONLINE_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Model channels: $MODEL_CHANNELS"
echo "  Error bounds: $ERROR_BOUNDS"
echo "  Relative error bounds: $ERROR_RELATIVE_BOUNDS"
echo "  Power error bounds: $ERROR_PWR_BOUNDS"
echo "  Error bound modes: $EB_MODES"
echo "  Model: $MODEL"
echo "  Number of residual blocks: $NUM_RES_BLOCKS"
echo "  Spatial dimensions: $SPATIAL_DIMS"
echo "  Slice order: $SLICE_ORDER"
echo "  Validation split: $VAL_SPLIT"
echo "  Track losses: $TRACK_LOSSES"
echo "  GPU mode: $GPU_MODE"
echo ""

srun --partition=gpucluster python evaluate_neurlz_correct.py \
    --data_dir "$DATA_DIR" \
    --sz_lib "$SZ_LIB" \
    --test_files $TEST_FILES \
    --absolute_error_bounds $ERROR_BOUNDS \
    --relative_error_bounds $ERROR_RELATIVE_BOUNDS \
    --pwr_error_bounds $ERROR_PWR_BOUNDS \
    --eb_modes $EB_MODES \
    --output_dir "$OUTPUT_DIR" \
    --device $DEVICE \
    --online_epochs $ONLINE_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --model_channels $MODEL_CHANNELS \
    --model $MODEL \
    --num_res_blocks $NUM_RES_BLOCKS \
    --spatial_dims $SPATIAL_DIMS \
    --slice_order $SLICE_ORDER \
    --val_split $VAL_SPLIT \
    --track_losses $TRACK_LOSSES

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="