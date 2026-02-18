#!/bin/bash

# =============================================================================
# Script Huấn Luyện (Training) cho bộ dữ liệu DAiSEE
# =============================================================================

# --- Cấu hình Đường dẫn ---
# Giả định script được chạy từ thư mục gốc của dự án
DATASET="DAiSEE"
ROOT_DIR="." 

# Đường dẫn đến các file nhãn (Label CSV)
TRAIN_LABEL="DAiSEE/Labels/TrainLabels.csv"
VAL_LABEL="DAiSEE/Labels/ValidationLabels.csv"
TEST_LABEL="DAiSEE/Labels/TestLabels.csv"

# --- Tham số Huấn luyện ---
BATCH_SIZE=8           # Giảm xuống 4 nếu gặp lỗi Out of Memory (OOM)
EPOCHS=20              # Số vòng lặp huấn luyện
LR=2e-5                # Tốc độ học (Learning Rate)
NUM_SEGMENTS=16        # Số lượng frame lấy mẫu từ mỗi video (quan trọng cho mô hình video)
WORKERS=4              # Số luồng xử lý dữ liệu

# --- Tên Experiment ---
EXPER_NAME="DAiSEE_Training"

# --- Lệnh Chạy ---
echo "Bắt đầu huấn luyện trên dataset: $DATASET"
echo "Sử dụng thiết bị: mps (Apple Silicon)"

python3 main.py \
    --mode train \
    --dataset ${DATASET} \
    --root-dir ${ROOT_DIR} \
    --train-annotation ${TRAIN_LABEL} \
    --val-annotation ${VAL_LABEL} \
    --test-annotation ${TEST_LABEL} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num-segments ${NUM_SEGMENTS} \
    --workers ${WORKERS} \
    --exper-name ${EXPER_NAME} \
    --gpu mps \
    --print-freq 10
