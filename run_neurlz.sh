#!/bin/bash
#SBATCH --job-name=SZ3+NeurLZ
#SBATCH --partition=gpucluster
#SBATCH --time=8:00:00     
#SBATCH --output=dual_model_ROI_0.15_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   
#SBATCH --nodes=1                   


nvidia-smi

echo "========================================="
echo "NeurLZ Evaluation (Correct Implementation)"
echo "========================================="
echo "Following actual NeurLZ paper approach:"
echo "  1. SZ3 is PRIMARY compressor"
echo "  2. Tiny DNN (~3k params) trained ONLINE"
echo "  3. DNN predicts residuals from SZ3-decompressed"
echo "  4. Storage: SZ3_bytes + DNN_weights + outliers"
echo "========================================="

cd /Users/923714256/Data_compression/SZ3_+_2NN
source /Users/923714256/miniconda3/bin/activate
# conda activate grandlib

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
    CUDA_DEVICES=0,1,2,3 #$(IFS=','; echo "${AVAILABLE_GPUS[*]}")
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
OUTPUT_DIR="./relative_error_bound_4e-4/dual_model_ROI_0.15"
SZ_LIB="/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so"

# Test files
TEST_FILES="dark_matter_density.f32" 

# Error bounds to test
ERROR_BOUNDS="1e-6"
ERROR_RELATIVE_BOUNDS="4e-4"
ERROR_PWR_BOUNDS="0"
# NeurLZ modes
EB_MODES="1"
# Model
MODEL="tiny_frequency_residual_predictor_7_attn_roi"
PATCH_SIZE=256
BATCH_SIZE=512
# Online training parameters
ONLINE_EPOCHS=50
LEARNING_RATE=1e-3
MODEL_CHANNELS="4"  # Multiple channels to test max performance
NUM_RES_BLOCKS=2
# Device
DEVICE="cuda"
ENABLE_POST_PROCESS=false
SPATIAL_DIMS=2
SLICE_ORDER="zxy"
VAL_SPLIT=0.1
# TRACK_LOSSES is a boolean flag, no value needed (--track_losses enables it)
TRACK_LOSSES=true  # set to "true" or "false"
PRINT_LEVEL="verbose"  # summary|verbose

# Save components (SZ3 bytes, model weights, metadata) separately
SAVE_COMPONENTS=true  # set to "true" or "false"
COMPONENTS_DIR="${OUTPUT_DIR}/compressed_components"

# ============================================================
# Dual Model Configuration (Scheme B)
# ============================================================
USE_DUAL_MODELS=true
MODEL_BG="tiny_frequency_residual_predictor_7_attn_roi"
MODEL_ROI="tiny_frequency_residual_predictor_7_attn_roi"

# ============================================================
# ROI Configuration
# ============================================================
AUTO_SELECT_ROI=true
ROI_PERCENTAGE=0.15  # 5% as ROI, 95% as Background
# ROI=""  # Manual ROI definition (uncomment if needed)
# Example: ROI="100,164,200,264,300,364"  # 3D box format

NUM_RUNS=5
mkdir -p $OUTPUT_DIR
mkdir -p $COMPONENTS_DIR

echo ""
echo "Running NeurLZ evaluation..."
echo "  Online epochs: $ONLINE_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Model channels: $MODEL_CHANNELS (will test each)"
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
echo "  Number of runs: $NUM_RUNS"
echo "  Enable post-process: $ENABLE_POST_PROCESS"
echo "  Patch size: $PATCH_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Save components: $SAVE_COMPONENTS"
echo "  Components directory: $COMPONENTS_DIR"
if [ "$USE_DUAL_MODELS" = true ]; then
    echo "  Dual Models: Enabled"
    echo "    BG Model: $MODEL_BG"
    echo "    ROI Model: $MODEL_ROI"
    if [ "$AUTO_SELECT_ROI" = true ]; then
        ROI_PERCENTAGE_PCT=$(awk "BEGIN {printf \"%.1f\", $ROI_PERCENTAGE * 100}")
        echo "    ROI Selection: Auto (${ROI_PERCENTAGE_PCT}% of dataset)"
    else
        echo "    ROI Selection: Manual"
    fi
else
    echo "  Dual Models: Disabled (Single Model)"
fi
echo ""

# ============================================================
# Safety checks (dual-only pipeline)
# ============================================================
if [ "$USE_DUAL_MODELS" = true ]; then
    if [ "$AUTO_SELECT_ROI" != true ] && [ -z "$ROI" ]; then
        echo "[FATAL] Dual-only pipeline requires ROI. Set AUTO_SELECT_ROI=true or provide ROI=\"x0,x1,y0,y1,z0,z1\"."
        exit 1
    fi
fi

# Build track_losses flag (store_true args don't take values)
TRACK_LOSSES_FLAG=""
if [ "$TRACK_LOSSES" = "true" ]; then
    TRACK_LOSSES_FLAG="--track_losses"
fi

# Build enable_post_process flag (store_true args don't take values)
ENABLE_POST_PROCESS_FLAG=""
if [ "$ENABLE_POST_PROCESS" = "true" ]; then
    ENABLE_POST_PROCESS_FLAG="--enable_post_process"
fi

# Build save_components flag (store_true args don't take values)
SAVE_COMPONENTS_FLAG=""
if [ "$SAVE_COMPONENTS" = "true" ]; then
    SAVE_COMPONENTS_FLAG="--save_components"
fi

# Build dual model flags
DUAL_MODELS_FLAG=""
if [ "$USE_DUAL_MODELS" = true ]; then
    DUAL_MODELS_FLAG="--use_dual_models"
fi

# Build ROI flags
ROI_FLAGS=""
if [ "$AUTO_SELECT_ROI" = true ]; then
    ROI_FLAGS="--auto_select_roi --roi_percentage $ROI_PERCENTAGE"
elif [ -n "$ROI" ]; then
    ROI_FLAGS="--roi \"$ROI\""
fi

python evaluate_neurlz_correct.py \
    --data_dir "$DATA_DIR" \
    --sz_lib "$SZ_LIB" \
    --test_files $TEST_FILES \
    --absolute_error_bounds $ERROR_BOUNDS \
    --relative_error_bounds $ERROR_RELATIVE_BOUNDS \
    --pwr_error_bounds $ERROR_PWR_BOUNDS \
    --eb_modes $EB_MODES \
    --output_dir "$OUTPUT_DIR" \
    --device $DEVICE \
    --print_level $PRINT_LEVEL \
    --online_epochs $ONLINE_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --model_channels $MODEL_CHANNELS \
    --model $MODEL \
    --num_res_blocks $NUM_RES_BLOCKS \
    --spatial_dims $SPATIAL_DIMS \
    --slice_order $SLICE_ORDER \
    --val_split $VAL_SPLIT \
    --num_runs $NUM_RUNS \
    $TRACK_LOSSES_FLAG \
    $ENABLE_POST_PROCESS_FLAG \
    --Patch_size $PATCH_SIZE \
    --Batch_size $BATCH_SIZE \
    $SAVE_COMPONENTS_FLAG \
    --components_dir "$COMPONENTS_DIR" \
    $DUAL_MODELS_FLAG \
    --model_bg "$MODEL_BG" \
    --model_roi "$MODEL_ROI" \
    $ROI_FLAGS