#!/bin/bash
# Setup script for Qwen3-VL training environment

echo "⏳ Đang thiết lập môi trường (vui lòng đợi trong giây lát)..."

{
    python -m pip install --upgrade pip -q
    pip install unsloth -q
    
    # Cài đặt các tiện ích bổ sung
    pip install pillow tqdm psutil tensorboard mlflow -q
} > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Thiết lập hoàn tất!"
    echo "🚀 Bạn có thể bắt đầu training ngay bây giờ."
else
    echo "❌ Có lỗi xảy ra trong quá trình cài đặt. Vui lòng kiểm tra lại!"
fi