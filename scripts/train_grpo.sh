#!/bin/bash
# GRPO Training script for Multi-GPU with full hyperparameter configuration
# Usage: bash train_grpo.sh [options]
# Example: bash train_grpo.sh --num_gpus 2 --learning_rate 5e-6 --use_gspo

# Default values (matching parse_args() in train_qwen3_vl_grpo.py)
NUM_GPUS=2
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
MAX_SEQ_LENGTH=8192
DATASET_NAME="ha684/senko1"
HF_TOKEN=""
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# GRPO-specific defaults
BATCH_SIZE=1
GRAD_ACCUM=1
EPOCHS=0.5
MAX_STEPS=-1
LEARNING_RATE=5e-6
WARMUP_STEPS=5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.001
LOGGING_STEPS=1

# GRPO-specific hyperparameters
NUM_GENERATIONS=2
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=1024
MAX_GRAD_NORM=0.1
SAVE_STEPS=60
LOG_COMPLETIONS=false
ADAM_BETA1=0.9
ADAM_BETA2=0.99
USE_GSPO=false

# Output & Logging
OUTPUT_DIR="outputs/qwen3_vl_grpo"
SAVE_MODEL=false
MODEL_SAVE_PATH="lora_model"
REPORT_TO="none"
MLFLOW_URI="http://localhost:5000"
MLFLOW_EXPERIMENT="qwen3_vl_grpo"
MLFLOW_RUN_NAME="qwen3_vl_grpo"
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
        --warmup_ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --num_generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --max_prompt_length)
            MAX_PROMPT_LENGTH="$2"
            shift 2
            ;;
        --max_completion_length)
            MAX_COMPLETION_LENGTH="$2"
            shift 2
            ;;
        --max_grad_norm)
            MAX_GRAD_NORM="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --log_completions)
            LOG_COMPLETIONS=true
            shift
            ;;
        --adam_beta1)
            ADAM_BETA1="$2"
            shift 2
            ;;
        --adam_beta2)
            ADAM_BETA2="$2"
            shift 2
            ;;
        --use_gspo)
            USE_GSPO=true
            shift
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
            echo "Usage: bash train_grpo.sh [options]"
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
            echo "  --batch_size              Per device train batch size (default: 1)"
            echo "  --grad_accum              Gradient accumulation steps (default: 1)"
            echo "  --epochs                  Number of training epochs (default: 0.5)"
            echo "  --max_steps               Maximum training steps, overrides epochs if > 0 (default: -1)"
            echo "  --learning_rate           Learning rate (default: 5e-6)"
            echo "  --warmup_steps            Warmup steps (default: 5)"
            echo "  --warmup_ratio            Warmup ratio (default: 0.1)"
            echo "  --weight_decay            Weight decay (default: 0.001)"
            echo "  --seed                    Random seed (default: 3407)"
            echo ""
            echo "GRPO-Specific Configuration:"
            echo "  --num_generations         Number of completions per prompt (default: 2)"
            echo "  --max_prompt_length       Maximum prompt length (default: 1024)"
            echo "  --max_completion_length   Maximum completion length (default: 1024)"
            echo "  --max_grad_norm           Max gradient norm for clipping (default: 0.1)"
            echo "  --save_steps              Save checkpoint every N steps (default: 60)"
            echo "  --log_completions         Log full completions to wandb/mlflow"
            echo "  --adam_beta1              Adam beta1 parameter (default: 0.9)"
            echo "  --adam_beta2              Adam beta2 parameter (default: 0.99)"
            echo "  --use_gspo                Enable GSPO (Generalized Sequential Policy Optimization)"
            echo ""
            echo "Logging & Evaluation:"
            echo "  --logging_steps           Logging interval (default: 1)"
            echo "  --report_to               Reporting tool: none/tensorboard/wandb/mlflow (default: none)"
            echo "  --mlflow_uri              MLflow tracking URI (default: http://localhost:5000)"
            echo "  --mlflow_experiment_name  MLflow experiment name (default: qwen3_vl_grpo)"
            echo "  --mlflow_run_name         MLflow run name (default: qwen3_vl_grpo)"
            echo "  --metric_file             File path to save the final metric"
            echo ""
            echo "Output & Saving:"
            echo "  --output_dir              Output directory for checkpoints (default: outputs/qwen3_vl_grpo)"
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
if [ "$MLFLOW_RUN_NAME" = "qwen3_vl_grpo" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    GSPO_SUFFIX=""
    if [ "$USE_GSPO" = true ]; then
        GSPO_SUFFIX="_gspo"
    fi
    MLFLOW_RUN_NAME="grpo_${NUM_GPUS}gpu_bs${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}${GSPO_SUFFIX}_${TIMESTAMP}"
fi

# Print configuration
echo "🚀 Starting GRPO Multi-GPU Training"
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
echo "  - Warmup ratio: $WARMUP_RATIO"
echo "  - Weight decay: $WEIGHT_DECAY"
echo "  - Adam beta1: $ADAM_BETA1"
echo "  - Adam beta2: $ADAM_BETA2"
echo "  - Max grad norm: $MAX_GRAD_NORM"
echo ""
echo "GRPO Configuration:"
echo "  - Num generations: $NUM_GENERATIONS"
echo "  - Max prompt length: $MAX_PROMPT_LENGTH"
echo "  - Max completion length: $MAX_COMPLETION_LENGTH"
echo "  - Save steps: $SAVE_STEPS"
if [ "$USE_GSPO" = true ]; then
    echo "  - GSPO: ENABLED ✅"
else
    echo "  - GSPO: DISABLED"
fi
if [ "$LOG_COMPLETIONS" = true ]; then
    echo "  - Log completions: YES"
fi
echo ""
echo "LoRA:"
echo "  - Rank: $LORA_R"
echo "  - Alpha: $LORA_ALPHA"
echo "  - Dropout: $LORA_DROPOUT"
echo ""
echo "Logging:"
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
CMD="torchrun --nproc_per_node=$NUM_GPUS qwen3_vl/train_qwen3_vl_grpo.py \
    --model_name $MODEL_NAME \
    --max_seq_length $MAX_SEQ_LENGTH \
    --dataset_name $DATASET_NAME \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --logging_steps $LOGGING_STEPS \
    --num_generations $NUM_GENERATIONS \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_grad_norm $MAX_GRAD_NORM \
    --save_steps $SAVE_STEPS \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
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

if [ "$USE_GSPO" = true ]; then
    CMD="$CMD --use_gspo"
fi

if [ "$LOG_COMPLETIONS" = true ]; then
    CMD="$CMD --log_completions"
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
    echo "✅ GRPO Training completed successfully!"
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
    echo "❌ GRPO Training failed!"
    echo "======================================================="
    exit 1
fi
