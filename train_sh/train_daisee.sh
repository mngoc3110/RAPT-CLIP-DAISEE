#!/bin/bash

# =============================================================================
# Script Huấn Luyện (Training) cho bộ dữ liệu DAiSEE (Video)
# =============================================================================

# --- Cấu hình Đường dẫn ---
# Điều chỉnh lại đường dẫn dataset phù hợp với môi trường Colab/Server của bạn
DATASET_ROOT="/content/RAPT-CLIP-DAISEE/DAiSEE" 
# Hoặc nếu dataset nằm ngay trong project: DATASET_ROOT="./DAiSEE"

ANN_DIR="./DAiSEE/Labels" # Thư mục chứa các file CSV nhãn

echo "Starting DAiSEE Training..."
echo "Dataset Root: $DATASET_ROOT"
echo "Annotations: $ANN_DIR"

# --- Chạy Huấn Luyện ---
python3 main.py \
  --mode train \
  --exper-name DAiSEE_Video_Train \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --temporal-layers 2 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANN_DIR/TrainLabels.csv" \
  --val-annotation "$ANN_DIR/ValidationLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --label-smoothing 0.05 \
  --use-amp \
  --grad-clip 1.0

echo "Training Finished!"
