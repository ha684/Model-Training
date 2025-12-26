# 🚀 Quick Start Guide

## ⚡ Fastest Way to Get Started

### 1️⃣ Setup (One-time)

**Windows (Git Bash or WSL):**
```bash
bash setup_environment.sh
```

**Or manually:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/anhvth/opensloth.git
```

### 2️⃣ Quick Test (Verify Setup)

```bash
bash train_quick_test.sh
```

Hoặc:
```bash
python train_qwen3_vl_ddp.py --epochs 1 --batch_size 1 --grad_accum 2
```

### 3️⃣ Real Training

**Single GPU:**
```bash
bash train_single_gpu.sh
```

**2 GPUs:**
```bash
bash train_multi_gpu.sh 2
```

**4 GPUs:**
```bash
bash train_multi_gpu.sh 4
```

### 4️⃣ Inference

```bash
# Test trên dataset
python inference_qwen3_vl.py --model_path lora_model --test_dataset

# Test trên 1 ảnh
python inference_qwen3_vl.py --model_path lora_model --image_path your_image.jpg
```

## 📝 Important Notes

### Windows Users

Nếu không có `bash`, dùng các commands trực tiếp:

**Single GPU:**
```bash
python train_qwen3_vl_ddp.py --batch_size 2 --grad_accum 4 --save_model
```

**Multi-GPU (2 GPUs):**
```bash
torchrun --nproc_per_node=2 train_qwen3_vl_ddp.py --batch_size 2 --grad_accum 4 --save_model
```

### Dataset Format

Dataset cần có trên HuggingFace Hub với format:

```python
{
    "image": PIL.Image,    # Ảnh document
    "text": str           # JSON string kết quả extraction
}
```

Splits: `train` và `validation`

### Custom Dataset

```bash
python train_qwen3_vl_ddp.py \
    --dataset_name your_username/your_dataset \
    --hf_token your_token_here
```

## 🎯 What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_quick_test.sh` | Fast test (1 epoch, small batch) | Verify setup works |
| `train_single_gpu.sh` | Standard single GPU training | 1 GPU only |
| `train_multi_gpu.sh N` | Multi-GPU training | Have 2+ GPUs |
| `train_production.sh N` | Production with logging | Final training |

## 📊 Expected Results

### Training Time (1 epoch, ~1000 samples)

- 1 GPU: ~30-40 minutes
- 2 GPUs: ~15-20 minutes  
- 4 GPUs: ~8-10 minutes

### VRAM Usage

- ~10-12 GB per GPU (with batch_size=2)
- If OOM: reduce `--batch_size` to 1

## 🐛 Common Issues

### Issue 1: Out of Memory

**Solution:**
```bash
python train_qwen3_vl_ddp.py --batch_size 1 --grad_accum 8
```

### Issue 2: Dataset không tìm thấy

**Solution:**
- Check dataset name đúng chưa
- Private dataset cần `--hf_token`

### Issue 3: OpenSloth không install được

**Solution:**
```bash
# Không có OpenSloth vẫn chạy được, chỉ chậm hơn một chút
python train_qwen3_vl_ddp.py --no_patches
```

## 💡 Pro Tips

1. **Monitor GPU**: `nvidia-smi -l 1` trong terminal khác
2. **TensorBoard**: `tensorboard --logdir outputs/` để xem training progress
3. **Start Small**: Test với `train_quick_test.sh` trước
4. **Save Often**: Luôn dùng `--save_model` để lưu kết quả

## 📚 More Details

- **Full Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **README**: [README_QWEN3VL_TRAINING.md](README_QWEN3VL_TRAINING.md)
- **Code**: `train_qwen3_vl_ddp.py` - main training script

---

**Need Help?** Check TRAINING_GUIDE.md for detailed examples! 🦥⚡

