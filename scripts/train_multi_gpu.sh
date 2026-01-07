#!/bin/bash
# Advanced Training script for Multi-GPU with full hyperparameter configuration
# Usage: bash train_multi_gpu.sh [options]
# Example: bash train_multi_gpu.sh --num_gpus 2 --learning_rate 5e-5 --lora_r 64

# Default values
NUM_GPUS=2
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_STEPS=100
EVAL_STEPS=20
LOGGING_STEPS=1
LEARNING_RATE=5e-5
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
WEIGHT_DECAY=0.01
WARMUP_STEPS=50
MAX_SEQ_LENGTH=8192
EVAL_BATCH_SIZE=4
HF_TOKEN="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"
DATASET_NAME="ha684/senko1"
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="outputs/multi_gpu_training"
MODEL_SAVE_PATH="outputs/saved_model"
REPORT_TO="none"
MLFLOW_URI=""
MLFLOW_EXPERIMENT=""
RUN_NAME=""
SAVE_MODEL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --eval_steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora_dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --eval_batch_size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_save_path)
            MODEL_SAVE_PATH="$2"
            shift 2
            ;;
        --report_to)
            REPORT_TO="$2"
            shift 2
            ;;
        --mlflow_uri)
            MLFLOW_URI="$2"
            shift 2
            ;;
        --mlflow_experiment_name)
            MLFLOW_EXPERIMENT="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --save_model)
            SAVE_MODEL=true
            shift
            ;;
        --help)
            echo "Usage: bash train_multi_gpu.sh [options]"
            echo ""
            echo "Options:"
            echo "  --num_gpus              Number of GPUs (default: 2)"
            echo "  --batch_size            Batch size per GPU (default: 2)"
            echo "  --grad_accum            Gradient accumulation steps (default: 4)"
            echo "  --max_steps             Maximum training steps (default: 100)"
            echo "  --eval_steps            Evaluation interval (default: 20)"
            echo "  --logging_steps         Logging interval (default: 1)"
            echo "  --learning_rate         Learning rate (default: 5e-5)"
            echo "  --lora_r                LoRA rank (default: 64)"
            echo "  --lora_alpha            LoRA alpha (default: 128)"
            echo "  --lora_dropout          LoRA dropout (default: 0.0)"
            echo "  --weight_decay          Weight decay (default: 0.01)"
            echo "  --warmup_steps          Warmup steps (default: 50)"
            echo "  --max_seq_length        Max sequence length (default: 8192)"
            echo "  --eval_batch_size       Evaluation batch size (default: 4)"
            echo "  --hf_token              HuggingFace token"
            echo "  --dataset_name          Dataset name (default: ha684/senko1)"
            echo "  --model_name            Model name (default: Qwen/Qwen3-VL-2B-Instruct)"
            echo "  --output_dir            Output directory"
            echo "  --model_save_path       Model save path"
            echo "  --report_to             Reporting backend (none/mlflow)"
            echo "  --mlflow_uri            MLflow tracking URI"
            echo "  --mlflow_experiment_name MLflow experiment name"
            echo "  --run_name              Run name for MLflow"
            echo "  --save_model            Save model after training"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validation
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "❌ Error: Number of GPUs must be at least 1"
    exit 1
fi

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="train_${NUM_GPUS}gpu_bs${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}_r${LORA_R}_${TIMESTAMP}"
fi

echo "🚀 Starting Multi-GPU Training"
echo "======================================================="
echo "Configuration:"
echo "  Model & Data:"
echo "    - Model: $MODEL_NAME"
echo "    - Dataset: $DATASET_NAME"
echo "    - Max seq length: $MAX_SEQ_LENGTH"
echo ""
echo "  Hardware:"
echo "    - GPUs: $NUM_GPUS"
echo "    - Batch size per GPU: $BATCH_SIZE"
echo "    - Gradient accumulation: $GRAD_ACCUM"
echo "    - Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo ""
echo "  Training:"
echo "    - Learning rate: $LEARNING_RATE"
echo "    - Max steps: $MAX_STEPS"
echo "    - Warmup steps: $WARMUP_STEPS"
echo "    - Weight decay: $WEIGHT_DECAY"
echo ""
echo "  LoRA:"
echo "    - Rank: $LORA_R"
echo "    - Alpha: $LORA_ALPHA"
echo "    - Dropout: $LORA_DROPOUT"
echo ""
echo "  Logging:"
echo "    - Eval steps: $EVAL_STEPS"
echo "    - Logging steps: $LOGGING_STEPS"
echo "    - Report to: $REPORT_TO"
echo "    - Run name: $RUN_NAME"
echo ""
echo "  Output:"
echo "    - Output dir: $OUTPUT_DIR"
if [ "$SAVE_MODEL" = true ]; then
    echo "    - Model save path: $MODEL_SAVE_PATH"
fi
echo "======================================================="
echo ""

CMD="torchrun --nproc_per_node=$NUM_GPUS train_qwen3_vl_ddp.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --report_to $REPORT_TO"

# Add optional parameters
if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

if [ "$SAVE_MODEL" = true ]; then
    CMD="$CMD --save_model --model_save_path $MODEL_SAVE_PATH"
fi

if [ -n "$MLFLOW_URI" ]; then
    CMD="$CMD --mlflow_uri $MLFLOW_URI"
fi

if [ -n "$MLFLOW_EXPERIMENT" ]; then
    CMD="$CMD --mlflow_experiment_name $MLFLOW_EXPERIMENT"
fi

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --mlflow_run_name $RUN_NAME"
fi

# Execute training
echo "🏃 Executing command..."
echo ""
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================="
    echo "✅ Training completed successfully!"
    echo "📁 Output directory: $OUTPUT_DIR"
    if [ "$SAVE_MODEL" = true ]; then
        echo "💾 Model saved to: $MODEL_SAVE_PATH"
    fi
    echo "======================================================="
else
    echo ""
    echo "======================================================="
    echo "❌ Training failed!"
    echo "======================================================="
    exit 1
fi