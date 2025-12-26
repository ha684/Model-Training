#!/bin/bash
# Training script for Single GPU
# Usage: bash train_single_gpu.sh

echo "🚀 Starting Single GPU Training..."
echo "=================================="

python train_qwen3_vl_ddp.py \
    --batch_size 2 \
    --grad_accum 4 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --eval_steps 10 \
    --save_model \
    --model_save_path lora_model \
    --output_dir outputs/single_gpu \
    --report_to none

echo "=================================="
echo "✅ Training completed!"

