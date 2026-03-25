#!/bin/bash
# Script to train CRNN + MobileNetV3 trên môi trường Kaggle Notebooks

echo "1. Cài đặt các thư viện phụ thuộc..."
pip install -r requirements.txt

echo "2. Chuẩn bị thư mục lưu mô hình..."
mkdir -p expr

# Nhận đường dẫn Dataset từ tham số truyền vào Kaggle Cell (nếu có).
# Nếu không truyền, dùng thư mục mặc định bên dưới:
TRAIN_ROOT="${1:-/kaggle/input/your-dataset-name/train}"
VAL_ROOT="${2:-/kaggle/input/your-dataset-name/val}"

echo "3. Bắt đầu quá trình huấn luyện mô hình..."

python train.py \
    --trainRoot "$TRAIN_ROOT" \
    --valRoot "$VAL_ROOT" \
    --batchSize 64 \
    --imgH 32 \
    --imgW 256 \
    --nepoch 25 \
    --adadelta \
    --cuda True \
    --dict "dict.txt" \
    --expr_dir "expr" \
    --workers 8

echo "Quá trình huấn luyện đã kết thúc. Bạn có thể kiểm tra mục Output để tải về biến thể của file models (.pth)!"
