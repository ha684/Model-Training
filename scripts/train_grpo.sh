#!/bin/bash
# GRPO Training script for Multi-GPU
# Usage: bash train_grpo.sh [options]

# --- 1. DEFAULT VALUES ---
NUM_GPUS=2
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
MAX_SEQ_LENGTH=8192
DATASET_NAME="ha684/senko1"
HF_TOKEN=""
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=1
MAX_STEPS=-1
LEARNING_RATE=5e-6
WARMUP_STEPS=5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.001
LOGGING_STEPS=1

NUM_GENERATIONS=2
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=1024
MAX_GRAD_NORM=0.1
SAVE_STEPS=60
LOG_COMPLETIONS=false
ADAM_BETA1=0.9
ADAM_BETA2=0.99
USE_GSPO=false

# New Eval params
EVAL_STEPS=30
EVAL_BATCH_SIZE=2

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
DATASET_CACHE_DIR=""
# --- 2. PARSE ARGUMENTS ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --max_seq_length) MAX_SEQ_LENGTH="$2"; shift 2 ;;
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --hf_token) HF_TOKEN="$2"; shift 2 ;;
        --lora_r) LORA_R="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum) GRAD_ACCUM="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --warmup_steps) WARMUP_STEPS="$2"; shift 2 ;;
        --warmup_ratio) WARMUP_RATIO="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --logging_steps) LOGGING_STEPS="$2"; shift 2 ;;
        --num_generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --max_completion_length) MAX_COMPLETION_LENGTH="$2"; shift 2 ;;
        --max_grad_norm) MAX_GRAD_NORM="$2"; shift 2 ;;
        --save_steps) SAVE_STEPS="$2"; shift 2 ;;
        --log_completions) LOG_COMPLETIONS=true; shift ;;
        --adam_beta1) ADAM_BETA1="$2"; shift 2 ;;
        --adam_beta2) ADAM_BETA2="$2"; shift 2 ;;
        --use_gspo) USE_GSPO=true; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --save_model) SAVE_MODEL=true; shift ;;
        --model_save_path) MODEL_SAVE_PATH="$2"; shift 2 ;;
        --report_to) REPORT_TO="$2"; shift 2 ;;
        --mlflow_uri) MLFLOW_URI="$2"; shift 2 ;;
        --mlflow_experiment_name) MLFLOW_EXPERIMENT="$2"; shift 2 ;;
        --mlflow_run_name) MLFLOW_RUN_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --push_to_hub) PUSH_TO_HUB=true; shift ;;
        --hub_model_id) HUB_MODEL_ID="$2"; shift 2 ;;
        --metric_file) METRIC_FILE="$2"; shift 2 ;;
        --dataset_cache_dir) DATASET_CACHE_DIR="$2"; shift 2 ;;
        # --- ĐÃ THÊM PHẦN XỬ LÝ EVAL ---
        --eval_steps) EVAL_STEPS="$2"; shift 2 ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        
        --help)
            echo "Usage: bash train_grpo.sh [options]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- 3. BUILD COMMAND ---
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
    --seed $SEED \
    --eval_steps $EVAL_STEPS \
    --eval_batch_size $EVAL_BATCH_SIZE"
    
if [ -n "$DATASET_CACHE_DIR" ]; then
    CMD="$CMD --dataset_cache_dir $DATASET_CACHE_DIR"
fi
# Add optional flags
if [ -n "$HF_TOKEN" ]; then CMD="$CMD --hf_token $HF_TOKEN"; fi
if [ "$SAVE_MODEL" = true ]; then CMD="$CMD --save_model"; fi
if [ "$PUSH_TO_HUB" = true ]; then 
    CMD="$CMD --push_to_hub"
    if [ -n "$HUB_MODEL_ID" ]; then CMD="$CMD --hub_model_id $HUB_MODEL_ID"; fi
fi
if [ "$USE_GSPO" = true ]; then CMD="$CMD --use_gspo"; fi
if [ "$LOG_COMPLETIONS" = true ]; then CMD="$CMD --log_completions"; fi
if [ -n "$METRIC_FILE" ]; then CMD="$CMD --metric_file $METRIC_FILE"; fi

# Execute
echo "🚀 Executing..."
echo "$CMD"
echo ""
eval $CMD