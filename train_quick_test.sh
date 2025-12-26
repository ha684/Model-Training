#!/bin/bash
# Quick test training script (fast iteration for debugging)
# Usage: bash train_quick_test.sh

echo "🧪 Starting Quick Test Training..."
echo "==================================="
echo "This is a quick test run for debugging"
echo "Using minimal settings for fast iteration"
echo ""

python train_qwen3_vl_ddp.py \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --eval_steps 5 \
    --logging_steps 1 \
    --output_dir outputs/quick_test \
    --report_to none

echo "==================================="
echo "✅ Quick test completed!"

