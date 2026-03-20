#!/bin/bash

# =============================================================================
# Script Huấn Luyện DAiSEE v7 — Fix Critical Bugs + Best Practices
# Fixes:
#   1. LDAM loss (s=2.0) thay CE cho imbalanced data
#   2. BỎ WeightedSampler → tránh val distribution mismatch  
#   3. Tắt MI/DC loss → tránh shape mismatch với prompt ensemble
#   4. Thêm EMA → stabilize training metrics
#   5. early-stop = 8 (đủ thời gian cho cosine schedule)
#   6. Same-Crop (50%/50%) trong dataloader
# =============================================================================

DATASET_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${DATASET_ROOT}/Labels"

if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/content/DAiSEE_local"
    ANN_DIR="${DATASET_ROOT}/Labels"
fi
if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/content/drive/MyDrive/RAPT-CLIP-DAISEE/DAiSEE"
    ANN_DIR="${DATASET_ROOT}/Labels"
fi

echo "Starting DAiSEE v7 (Bug-Fixed) Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3class_v7_fixed \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 2e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 3 \
  --temporal-layers 2 \
  --num-segments 8 \
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
  --temperature 0.07 \
  --loss-type ldam \
  --ldam-s 2.0 \
  --ldam-max-m 0.5 \
  --label-smoothing 0.1 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.999 \
  --grad-clip 1.0 \
  --early-stop 8

echo "Training Finished!"
