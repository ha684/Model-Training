#!/bin/bash
# Production training script with optimal settings
# Usage: bash train_production.sh [num_gpus]

NUM_GPUS=${1:-2}

echo "🎯 Starting Production Training..."
echo "=================================="
echo "Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Mode: Production"
echo "  - Epochs: 5"
echo "  - With TensorBoard logging"
echo ""

# Production settings
BATCH_SIZE=2
GRAD_ACCUM=4
EPOCHS=5
LR=1e-4
EVAL_STEPS=50

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    torchrun --nproc_per_node=$NUM_GPUS train_qwen3_vl_ddp.py \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --epochs $EPOCHS \
        --learning_rate $LR \
        --warmup_steps 50 \
        --eval_steps $EVAL_STEPS \
        --save_model \
        --model_save_path outputs/production_model \
        --output_dir outputs/production_${NUM_GPUS}gpus \
        --report_to tensorboard
else
    # Single GPU training
    python train_qwen3_vl_ddp.py \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --epochs $EPOCHS \
        --learning_rate $LR \
        --warmup_steps 50 \
        --eval_steps $EVAL_STEPS \
        --save_model \
        --model_save_path outputs/production_model \
        --output_dir outputs/production_1gpu \
        --report_to tensorboard
fi

echo ""
echo "=================================="
echo "✅ Production training completed!"
echo "📊 View results: tensorboard --logdir outputs/"

