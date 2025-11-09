#!/bin/bash
#SBATCH --job-name=SZ3+NeurLZ        # Job name
#SBATCH --partition=gpucluster            # Partition name
#SBATCH --time=8:00:00                   # Time limit hrs:min:sec
#SBATCH --output=SZ3+NeurLZ_%j.out   # Standard output and error log
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
# Script to evaluate NeurLZ-SZ3 compression vs baseline

# Evaluation configuration
DATA_DIR="/Users/923714256/Data_compression/SDRBENCH-EXASKY-NYX-512x512x512"
MODEL_PATH="/Users/923714256/Data_compression/neural_compression/models_fullvolume/best_model_fullvolume.pth"
OUTPUT_DIR="/Users/923714256/Data_compression/neural_compression/evaluation_results"
SZ_LIB="/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so"

# Test configuration
ERROR_BOUNDS="300"
TEST_FILES="velocity_x.f32"
DEVICE="cpu"  # Use "cpu" if GPU not available

echo "========================================="
echo "NeurLZ-SZ3 Evaluation"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Error bounds: $ERROR_BOUNDS"
echo "Test files: $TEST_FILES"
echo "Device: $DEVICE"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please train the model first using run_training.sh"
    exit 1
fi

# Run evaluation
cd /Users/923714256/Data_compression/neural_compression

PYTHONPATH=/Users/923714256/Data_compression/neural_compression

srun --partition=gpucluster python evaluate_neurlz.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sz_lib "$SZ_LIB" \
    --error_bounds $ERROR_BOUNDS \
    --test_files $TEST_FILES \
    --device "$DEVICE"

echo ""
echo "========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="

