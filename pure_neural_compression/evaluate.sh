#!/bin/bash

# Evaluation script for Pure Neural Compressor

# Configuration
MODEL_PATH="/Users/923714256/Data_compression/pure_neural_compression/models/best_model.pth"
DATA_DIR="/Users/923714256/Data_compression/SDRBENCH-EXASKY-NYX-512x512x512"
OUTPUT_DIR="/Users/923714256/Data_compression/pure_neural_compression/evaluation_results"

# Test files
TEST_FILES=(
    "velocity_x.f32"
    "velocity_y.f32"
    "velocity_z.f32"
)

# Data shape
DATA_SHAPE="512 512 512"

# Error bounds to test
ERROR_BOUNDS=(2)

# Compression parameters
QUANTIZATION_METHOD="adaptive"  # or "adaptive"
ENTROPY_CODING="auto"  # or "arithmetic" or "zlib"
QUANTIZATION_LEVELS=""  # Leave empty for auto

# Device
DEVICE="cuda"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build evaluation command
CMD="python evaluate_compressor.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --test_files ${TEST_FILES[@]} \
    --data_shape $DATA_SHAPE \
    --error_bounds ${ERROR_BOUNDS[@]} \
    --quantization_method $QUANTIZATION_METHOD \
    --entropy_coding $ENTROPY_CODING \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE"

# Add quantization levels if specified
if [ ! -z "$QUANTIZATION_LEVELS" ]; then
    CMD="$CMD --quantization_levels $QUANTIZATION_LEVELS"
fi

# Print command
echo "Evaluation command:"
echo "$CMD"
echo ""

# Run evaluation
eval $CMD

