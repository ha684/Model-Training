#!/bin/bash
# Training script for Multi-GPU (2 GPUs)
# Usage: bash train_multi_gpu.sh [num_gpus]
# Example: bash train_multi_gpu.sh 2

NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified

echo "🚀 Starting Multi-GPU Training with $NUM_GPUS GPUs..."
echo "======================================================="

# Adjust batch size based on number of GPUs to maintain effective batch size
if [ "$NUM_GPUS" -eq 2 ]; then
    BATCH_SIZE=2
    GRAD_ACCUM=4
elif [ "$NUM_GPUS" -eq 4 ]; then
    BATCH_SIZE=1
    GRAD_ACCUM=8
else
    BATCH_SIZE=2
    GRAD_ACCUM=4
fi

echo "Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo ""

torchrun --nproc_per_node=$NUM_GPUS train_qwen3_vl_ddp.py \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs 3 \
    --learning_rate 5e-5 \
    --eval_steps 20 \
    --save_model \
    --model_save_path outputs/multi_gpu_model \
    --output_dir outputs/multi_gpu_${NUM_GPUS}gpus \
    --report_to none

echo "======================================================="
echo "✅ Training completed!"

