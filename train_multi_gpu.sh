#!/bin/bash
# Training script for Multi-GPU
# Usage: bash train_multi_gpu.sh [num_gpus] [batch_size] [grad_accum]
# Example: bash train_multi_gpu.sh 2 4 2
# Example: bash train_multi_gpu.sh 4 2 4

NUM_GPUS=${1:-2}        # Default to 2 GPUs if not specified
BATCH_SIZE=${2:-2}      # Default to 2 batch size per GPU if not specified
GRAD_ACCUM=${3:-4}      # Default to 4 gradient accumulation if not specified

echo "🚀 Starting Multi-GPU Training with $NUM_GPUS GPUs..."
echo "======================================================="

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

echo "Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo ""

# Validate inputs
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "❌ Error: Number of GPUs must be at least 1"
    exit 1
fi

if [ "$BATCH_SIZE" -lt 1 ]; then
    echo "❌ Error: Batch size must be at least 1"
    exit 1
fi

if [ "$GRAD_ACCUM" -lt 1 ]; then
    echo "❌ Error: Gradient accumulation must be at least 1"
    exit 1
fi

torchrun --nproc_per_node=$NUM_GPUS train_qwen3_vl_ddp.py \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs 3 \
    --learning_rate 5e-5 \
    --eval_steps 20 \
    --save_model \
    --model_save_path outputs/multi_gpu_model \
    --output_dir outputs/multi_gpu_${NUM_GPUS}gpus_bs${EFFECTIVE_BATCH_SIZE} \
    --report_to none

echo "======================================================="
echo "✅ Training completed!"
echo "📁 Output directory: outputs/multi_gpu_${NUM_GPUS}gpus_bs${EFFECTIVE_BATCH_SIZE}"