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
# Giải thích các cờ (flags):
# --adadelta: Khuyến nghị dùng Adadelta cho mô hình CRNN để tối ưu hoá tốc độ giảm loss
# --keep_ratio: Giữ nguyên tỉ lệ hình dáng ảnh nếu kích thước tập dữ liệu không đều
# --cuda: Kích hoạt chạy bằng GPU của Kaggle
python train.py \
    --trainRoot "$TRAIN_ROOT" \
    --valRoot "$VAL_ROOT" \
    --batchSize 64 \
    --imgH 32 \
    --imgW 256 \
    --nepoch 25 \
    --adadelta \
    --keep_ratio \
    --cuda \
    --dict "dict.txt" \
    --expr_dir "expr" \
    --workers 8

echo "Quá trình huấn luyện đã kết thúc. Bạn có thể kiểm tra mục Output để tải về biến thể của file models (.pth)!"
