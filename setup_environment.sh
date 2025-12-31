#!/bin/bash
# Setup script for Qwen3-VL training environment (Silent Version for Kaggle)

echo "⏳ Đang thiết lập môi trường (vui lòng đợi trong giây lát)..."

# Chặn mọi output thông thường, chỉ hiện lỗi nếu có
{
    # Update pip trước
    python -m pip install --upgrade pip -q

    # Cài đặt unsloth và các thư viện liên quan
    # Lưu ý: Trên Kaggle thường cần cài đặt unsloth theo đúng bản cho GPU
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