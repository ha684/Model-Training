#!/bin/bash
# Random Search HPO Script
# This script runs hyperparameter optimization using random search
# Each trial runs for a limited number of steps (not full epochs) to quickly explore the search space
#
# Usage: bash run_random_search.sh [num_trials] [ngrok_url]
# Example: bash run_random_search.sh 15 https://your-ngrok-url.ngrok.io

NUM_TRIALS=${1:-10}
NGROK_URL=${2:-"http://localhost:5000"}
NUM_GPUS=${3:-2}
DATASET_NAME=${4:-"ha684/senko1"}
HF_TOKEN=${5:-""}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🔍 Random Search Hyperparameter Optimization          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  🎲 Number of trials: $NUM_TRIALS"
echo "  🌐 MLflow URI: $NGROK_URL"
echo "  🖥️  Number of GPUs: $NUM_GPUS"
echo "  📦 Dataset: $DATASET_NAME"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python first."
    exit 1
fi

# Check if hpo_runner.py exists
if [ ! -f "hpo_runner.py" ]; then
    echo "❌ Error: hpo_runner.py not found in current directory"
    exit 1
fi

# Check if train_qwen3_vl_ddp.py exists
if [ ! -f "train_qwen3_vl_ddp.py" ]; then
    echo "❌ Error: train_qwen3_vl_ddp.py not found in current directory"
    exit 1
fi

# Create output directory for HPO results
mkdir -p outputs/hpo

# Run the HPO
echo "🚀 Starting random search..."
echo ""

if [ -z "$HF_TOKEN" ]; then
    python hpo_runner.py \
        --num_trials $NUM_TRIALS \
        --ngrok_url "$NGROK_URL" \
        --num_gpus $NUM_GPUS \
        --dataset_name "$DATASET_NAME"
else
    python hpo_runner.py \
        --num_trials $NUM_TRIALS \
        --ngrok_url "$NGROK_URL" \
        --num_gpus $NUM_GPUS \
        --dataset_name "$DATASET_NAME" \
        --hf_token "$HF_TOKEN"
fi

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Random search completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Open MLflow UI: $NGROK_URL"
    echo "  2. Navigate to experiment: 'qwen3_vl_random_search'"
    echo "  3. Sort runs by 'metrics.eval_loss' (ascending)"
    echo "  4. Select the best performing model"
    echo "  5. Use the hyperparameters from that run for full training"
    echo ""
    echo "📊 Tip: Look for runs with lowest eval_loss and stable training"
else
    echo "❌ Random search failed or was interrupted"
    echo "   Check the logs above for errors"
fi

echo "════════════════════════════════════════════════════════════════"
