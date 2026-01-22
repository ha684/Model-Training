#!/bin/bash
set -euo pipefail

# Optuna HPO Script (with MLflow + pruning-ready metric JSONL)
#
# Usage:
#   bash run_optuna_hpo.sh [num_trials] [mlflow_uri] [num_gpus] [dataset_name] [hf_token]
#
# Example:
#   bash run_optuna_hpo.sh 15 http://localhost:5000 2 ha684/senko1

NUM_TRIALS=${1:-10}
MLFLOW_URI=${2:-"http://localhost:5000"}
NUM_GPUS=${3:-2}
DATASET_NAME=${4:-"ha684/senko1"}
HF_TOKEN=${5:-""}

HPO_FILE=${HPO_FILE:-"hpo_runner.py"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"qwen3-vl/train_qwen3_vl_ddp.py"}
STUDY_NAME=${STUDY_NAME:-"qwen3_vl_optimization"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_vl_optuna"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/hpo"}

MAX_STEPS=${MAX_STEPS:-30}
EVAL_STEPS=${EVAL_STEPS:-10}
LOGGING_STEPS=${LOGGING_STEPS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
WARMUP_STEPS=${WARMUP_STEPS:-5}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          🧪 Optuna Hyperparameter Optimization (HPO)          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  🔢 Trials:            $NUM_TRIALS"
echo "  🌐 MLflow URI:        $MLFLOW_URI"
echo "  🧪 MLflow Experiment: $EXPERIMENT_NAME"
echo "  🧠 Study name:        $STUDY_NAME"
echo "  🖥️  GPUs:             $NUM_GPUS"
echo "  📦 Dataset:           $DATASET_NAME"
echo "  📁 Output root:       $OUTPUT_ROOT"
echo "  ⏱️  max_steps:         $MAX_STEPS"
echo "  🧪 eval_steps:        $EVAL_STEPS"
echo ""

# Basic checks
if ! command -v python &> /dev/null; then
  echo "❌ Error: python not found."
  exit 1
fi

if [ ! -f "$HPO_FILE" ]; then
  echo "❌ Error: $HPO_FILE not found in current directory."
  exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "❌ Error: training script not found: $TRAIN_SCRIPT"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "🚀 Starting Optuna HPO..."
echo ""

COMMON_ARGS=(
  --num_trials "$NUM_TRIALS"
  --mlflow_uri "$MLFLOW_URI"
  --mlflow_experiment "$EXPERIMENT_NAME"
  --study_name "$STUDY_NAME"
  --num_gpus "$NUM_GPUS"
  --dataset_name "$DATASET_NAME"
  --train_script "$TRAIN_SCRIPT"
  --output_root "$OUTPUT_ROOT"
  --max_steps "$MAX_STEPS"
  --eval_steps "$EVAL_STEPS"
  --logging_steps "$LOGGING_STEPS"
  --batch_size "$BATCH_SIZE"
  --warmup_steps "$WARMUP_STEPS"
)

if [ -z "$HF_TOKEN" ]; then
  python "$HPO_FILE" "${COMMON_ARGS[@]}"
else
  python "$HPO_FILE" "${COMMON_ARGS[@]}" --hf_token "$HF_TOKEN"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ HPO completed!"
echo ""
echo "Next steps:"
echo "  1) Open MLflow UI: $MLFLOW_URI"
echo "  2) Experiment:     $EXPERIMENT_NAME"
echo "  3) Sort by metric: final_eval_loss (or eval_loss)"
echo "  4) Check tags:     optuna.state, optuna.trial_number"
echo "  5) Take best params from the best run and do full training"
echo "════════════════════════════════════════════════════════════════"
