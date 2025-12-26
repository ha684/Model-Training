#!/bin/bash
# Setup script for Qwen3-VL training environment
# Usage: bash setup_environment.sh

echo "🔧 Setting up Qwen3-VL training environment..."
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python version: $(python --version)"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA available"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠️  CUDA not detected. GPU training may not work."
fi

echo ""
echo "📦 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "📦 Installing core dependencies..."
pip install transformers datasets accelerate peft huggingface_hub trl bitsandbytes

echo ""
echo "📦 Installing Unsloth..."
pip install unsloth

echo ""
echo "📦 Installing OpenSloth for multi-GPU optimization..."
pip install git+https://github.com/anhvth/opensloth.git

echo ""
echo "📦 Installing additional utilities..."
pip install pillow tqdm psutil tensorboard

echo ""
echo "================================================"
echo "✅ Environment setup completed!"
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset on HuggingFace Hub"
echo "  2. Configure fields in config/fields_config.py (if needed)"
echo "  3. Run training:"
echo "     - Single GPU: python train_qwen3_vl_ddp.py"
echo "     - Multi-GPU:  torchrun --nproc_per_node=2 train_qwen3_vl_ddp.py"
echo ""
echo "📖 See TRAINING_GUIDE.md for detailed instructions"
echo "================================================"

