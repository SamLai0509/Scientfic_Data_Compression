#!/bin/bash
# Script to check which GPUs are available for training

echo "============================================"
echo "GPU Availability Check"
echo "============================================"
echo ""

# Run nvidia-smi to show GPU status
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | \
awk -F', ' '
BEGIN {
    print "GPU | Name                    | Used    | Total   | Free    | Util | Status"
    print "----+--------------------------+---------+---------+---------+------+----------"
}
{
    gpu_id = $1
    name = $2
    used = $3
    total = $4
    free = $5
    util = $6
    
    # Determine status based on free memory
    if (free > 70000) {
        status = "FREE ✓"
    } else if (free > 10000) {
        status = "PARTIAL"
    } else {
        status = "FULL ✗"
    }
    
    printf "%3d | %-24s | %6d MB | %6d MB | %6d MB | %3d%% | %s\n", 
           gpu_id, name, used, total, free, util, status
}
'

echo ""
echo "============================================"
echo "Recommendations:"
echo "============================================"

# Get free GPUs (>70GB free for A100 80GB)
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
            awk -F', ' '$2 > 70000 {printf "%s,", $1}' | sed 's/,$//')

if [ -z "$FREE_GPUS" ]; then
    echo "⚠️  No completely free GPUs available!"
    echo ""
    echo "Options:"
    echo "1. Wait for other jobs to complete"
    echo "2. Kill your other processes (check with: ps aux | grep python)"
    echo "3. Use partially free GPUs with reduced batch size"
    
    # Show partially free GPUs (>10GB)
    PARTIAL_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                   awk -F', ' '$2 > 10000 && $2 <= 70000 {printf "%s,", $1}' | sed 's/,$//')
    
    if [ ! -z "$PARTIAL_GPUS" ]; then
        echo ""
        echo "Partially free GPUs: $PARTIAL_GPUS"
        echo "You could try: GPU_IDS=\"$PARTIAL_GPUS\" with BATCH_SIZE=1"
    fi
else
    echo "✓ Free GPUs available: $FREE_GPUS"
    echo ""
    echo "Recommended configuration for run_training.sh:"
    echo "  GPU_IDS=\"$FREE_GPUS\""
    
    NUM_FREE=$(echo $FREE_GPUS | tr ',' '\n' | wc -l)
    echo "  NUM_GPUS=$NUM_FREE"
    
    if [ $NUM_FREE -ge 4 ]; then
        echo "  BATCH_SIZE=2  # You have plenty of GPUs!"
    elif [ $NUM_FREE -ge 2 ]; then
        echo "  BATCH_SIZE=3  # Good balance for $NUM_FREE GPUs"
    else
        echo "  BATCH_SIZE=4  # Compensate for single GPU"
    fi
fi

echo ""
echo "To update your training script, edit:"
echo "  nano run_training.sh"
echo ""
echo "Current run_training.sh configuration:"
grep "^GPU_IDS=" /Users/923714256/Data_compression/run_training.sh
grep "^NUM_GPUS=" /Users/923714256/Data_compression/run_training.sh
grep "^BATCH_SIZE=" /Users/923714256/Data_compression/run_training.sh
echo "============================================"

