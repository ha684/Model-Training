#!/bin/bash
# Training script for Multi-GPU
# Usage: bash train_multi_gpu.sh [num_gpus] [batch_size] [grad_accum]
# Example: bash train_multi_gpu.sh 2 4 1 1

NUM_GPUS=${1:-2}        # Default to 2 GPUs
BATCH_SIZE=${2:-2}      # Default to 2 batch size per GPU
GRAD_ACCUM=${3:-4}      # Default to 4 gradient accumulation
EPOCHS=${4:-3}     
HF_TOKEN=${5:-"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"}
DATASET_NAME=${6:-"ha684/senko1"}
MODEL_NAME=${7:-"Qwen/Qwen3-VL-2B-Instruct"}

echo "🚀 Starting Multi-GPU Training with $NUM_GPUS GPUs..."
echo "======================================================="

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

echo "Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo ""

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "❌ Error: Number of GPUs must be at least 1"
    exit 1
fi

torchrun --nproc_per_node=$NUM_GPUS train_qwen3_vl_ddp.py \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --learning_rate 5e-5 \
    --eval_steps 20 \
    --save_model \
    --model_save_path outputs/multi_gpu_model \
    --output_dir outputs/multi_gpu_${NUM_GPUS}gpus_bs${EFFECTIVE_BATCH_SIZE} \
    --report_to none \
    --hf_token $HF_TOKEN \
    --dataset_name $DATASET_NAME \
    --model_name $MODEL_NAME \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --eval_batch_size 4 \
    --logging_steps 1 \
    --max_seq_length 8192

if [ $? -eq 0 ]; then
    echo "======================================================="
    echo "✅ Training completed!"
    echo "📁 Output directory: outputs/multi_gpu_${NUM_GPUS}gpus_bs${EFFECTIVE_BATCH_SIZE}"
else
    echo "❌ Training failed!"
fi