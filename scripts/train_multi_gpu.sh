#!/bin/bash
# Advanced Training script for Multi-GPU with full hyperparameter configuration
# Usage: bash train_multi_gpu.sh [options]
# Example: bash train_multi_gpu.sh --num_gpus 2 --learning_rate 5e-5 --lora_r 64

# Default values (matching parse_args())
NUM_GPUS=2
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
MAX_SEQ_LENGTH=8192
DATASET_NAME="ha684/senko1"
HF_TOKEN=""
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
BATCH_SIZE=2
EVAL_BATCH_SIZE=2
GRAD_ACCUM=4
EPOCHS=1
MAX_STEPS=-1
LEARNING_RATE=1e-4
WARMUP_STEPS=5
WEIGHT_DECAY=0.001
EVAL_STEPS=10
LOGGING_STEPS=1
OUTPUT_DIR="outputs/qwen3_vl_lora"
SAVE_MODEL=false
MODEL_SAVE_PATH="lora_model"
REPORT_TO="none"
MLFLOW_URI="http://localhost:5000"
MLFLOW_EXPERIMENT="qwen3_vl_finetune"
MLFLOW_RUN_NAME="qwen3_vl_finetune"
SEED=3407
PUSH_TO_HUB=false
HUB_MODEL_ID=""
METRIC_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="$2"
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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_batch_size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save_model)
            SAVE_MODEL=true
            shift
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
        --mlflow_run_name)
            MLFLOW_RUN_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --hub_model_id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        --metric_file)
            METRIC_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash train_multi_gpu.sh [options]"
            echo ""
            echo "Model & Data:"
            echo "  --model_name              Model name or path (default: Qwen/Qwen3-VL-2B-Instruct)"
            echo "  --max_seq_length          Maximum sequence length (default: 8192)"
            echo "  --dataset_name            HuggingFace dataset name (default: ha684/senko1)"
            echo "  --hf_token                HuggingFace API token"
            echo ""
            echo "LoRA Configuration:"
            echo "  --lora_r                  LoRA rank (default: 64)"
            echo "  --lora_alpha              LoRA alpha parameter (default: 128)"
            echo "  --lora_dropout            LoRA dropout (default: 0.0)"
            echo ""
            echo "Training Configuration:"
            echo "  --num_gpus                Number of GPUs (default: 2)"
            echo "  --batch_size              Per device train batch size (default: 2)"
            echo "  --eval_batch_size         Per device eval batch size (default: 2)"
            echo "  --grad_accum              Gradient accumulation steps (default: 4)"
            echo "  --epochs                  Number of training epochs (default: 1)"
            echo "  --max_steps               Maximum training steps, overrides epochs if > 0 (default: -1)"
            echo "  --learning_rate           Learning rate (default: 1e-4)"
            echo "  --warmup_steps            Warmup steps (default: 5)"
            echo "  --weight_decay            Weight decay (default: 0.001)"
            echo "  --seed                    Random seed (default: 3407)"
            echo ""
            echo "Logging & Evaluation:"
            echo "  --eval_steps              Evaluation interval (default: 10)"
            echo "  --logging_steps           Logging interval (default: 1)"
            echo "  --report_to               Reporting tool: none/tensorboard/wandb/mlflow (default: none)"
            echo "  --mlflow_uri              MLflow tracking URI (default: http://localhost:5000)"
            echo "  --mlflow_experiment_name  MLflow experiment name (default: qwen3_vl_finetune)"
            echo "  --mlflow_run_name         MLflow run name (default: qwen3_vl_finetune)"
            echo "  --metric_file             File path to save the final metric"
            echo ""
            echo "Output & Saving:"
            echo "  --output_dir              Output directory for checkpoints (default: outputs/qwen3_vl_lora)"
            echo "  --save_model              Save the model after training"
            echo "  --model_save_path         Path to save the fine-tuned model (default: lora_model)"
            echo "  --push_to_hub             Push model to Hugging Face Hub"
            echo "  --hub_model_id            Repository ID on HF (e.g., username/qwen-finetuned)"
            echo ""
            echo "  --help                    Show this help message"
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

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

# Generate run name if using default
if [ "$MLFLOW_RUN_NAME" = "qwen3_vl_finetune" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MLFLOW_RUN_NAME="train_${NUM_GPUS}gpu_bs${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}_${TIMESTAMP}"
fi

# Print configuration
echo "🚀 Starting Multi-GPU Training"
echo "======================================================="
echo "Model & Data:"
echo "  - Model: $MODEL_NAME"
echo "  - Dataset: $DATASET_NAME"
echo "  - Max seq length: $MAX_SEQ_LENGTH"
echo "  - Seed: $SEED"
echo ""
echo "Hardware:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Eval batch size per GPU: $EVAL_BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo ""
echo "Training:"
echo "  - Learning rate: $LEARNING_RATE"
if [ "$MAX_STEPS" -gt 0 ]; then
    echo "  - Max steps: $MAX_STEPS (overriding epochs)"
else
    echo "  - Epochs: $EPOCHS"
fi
echo "  - Warmup steps: $WARMUP_STEPS"
echo "  - Weight decay: $WEIGHT_DECAY"
echo ""
echo "LoRA:"
echo "  - Rank: $LORA_R"
echo "  - Alpha: $LORA_ALPHA"
echo "  - Dropout: $LORA_DROPOUT"
echo ""
echo "Logging:"
echo "  - Eval steps: $EVAL_STEPS"
echo "  - Logging steps: $LOGGING_STEPS"
echo "  - Report to: $REPORT_TO"
if [ "$REPORT_TO" = "mlflow" ]; then
    echo "  - MLflow URI: $MLFLOW_URI"
    echo "  - Experiment: $MLFLOW_EXPERIMENT"
    echo "  - Run name: $MLFLOW_RUN_NAME"
fi
echo ""
echo "Output:"
echo "  - Output dir: $OUTPUT_DIR"
if [ "$SAVE_MODEL" = true ]; then
    echo "  - Model save path: $MODEL_SAVE_PATH"
fi
if [ "$PUSH_TO_HUB" = true ]; then
    echo "  - Push to Hub: Yes ($HUB_MODEL_ID)"
fi
if [ -n "$METRIC_FILE" ]; then
    echo "  - Metric file: $METRIC_FILE"
fi
echo "======================================================="
echo ""

# Build command
CMD="torchrun --nproc_per_node=$NUM_GPUS qwen3-vl/train_qwen3_vl_ddp.py \
    --model_name $MODEL_NAME \
    --max_seq_length $MAX_SEQ_LENGTH \
    --dataset_name $DATASET_NAME \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --output_dir $OUTPUT_DIR \
    --model_save_path $MODEL_SAVE_PATH \
    --report_to $REPORT_TO \
    --mlflow_uri $MLFLOW_URI \
    --mlflow_experiment_name $MLFLOW_EXPERIMENT \
    --mlflow_run_name $MLFLOW_RUN_NAME \
    --seed $SEED"

# Add optional parameters
if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

if [ "$SAVE_MODEL" = true ]; then
    CMD="$CMD --save_model"
fi

if [ "$PUSH_TO_HUB" = true ]; then
    CMD="$CMD --push_to_hub"
    if [ -n "$HUB_MODEL_ID" ]; then
        CMD="$CMD --hub_model_id $HUB_MODEL_ID"
    fi
fi

if [ -n "$METRIC_FILE" ]; then
    CMD="$CMD --metric_file $METRIC_FILE"
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
    if [ "$PUSH_TO_HUB" = true ]; then
        echo "🚀 Model pushed to Hub: $HUB_MODEL_ID"
    fi
    if [ -n "$METRIC_FILE" ] && [ -f "$METRIC_FILE" ]; then
        echo "📊 Metrics saved to: $METRIC_FILE"
    fi
    echo "======================================================="
else
    echo ""
    echo "======================================================="
    echo "❌ Training failed!"
    echo "======================================================="
    exit 1
fi