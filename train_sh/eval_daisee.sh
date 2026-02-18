#!/bin/bash

# =============================================================================
# Script Đánh Giá (Evaluation) cho bộ dữ liệu DAiSEE
# =============================================================================

# --- Cấu hình Đường dẫn ---
DATASET="DAiSEE"
ROOT_DIR="."
TEST_LABEL="DAiSEE/Labels/TestLabels.csv"

# --- Tham số ---
BATCH_SIZE=8
NUM_SEGMENTS=16
WORKERS=4

# --- Đường dẫn Model ---
# LƯU Ý: Bạn cần thay đổi đường dẫn này sau khi huấn luyện xong để trỏ đến file model_best.pth thực tế
# Ví dụ: outputs/DAiSEE_Training-[NGAY]-[GIO]/model_best.pth
CHECKPOINT="outputs/DAiSEE_Training-EXAMPLE/model_best.pth"

# --- Lệnh Chạy ---
echo "Bắt đầu đánh giá trên dataset: $DATASET"

if [ ! -f "$CHECKPOINT" ]; then
    echo "LỖI: Không tìm thấy file checkpoint tại: $CHECKPOINT"
    echo "Vui lòng mở file train_sh/eval_daisee.sh và cập nhật biến CHECKPOINT."
    exit 1
fi

python3 main.py \
    --mode eval \
    --dataset ${DATASET} \
    --root-dir ${ROOT_DIR} \
    --test-annotation ${TEST_LABEL} \
    --batch-size ${BATCH_SIZE} \
    --num-segments ${NUM_SEGMENTS} \
    --workers ${WORKERS} \
    --eval-checkpoint ${CHECKPOINT} \
    --gpu mps
