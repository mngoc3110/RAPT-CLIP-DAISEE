#!/bin/bash

# =============================================================================
# Script Huấn Luyện DAiSEE v10 — Undersampling Majority Classes
# Strategy:
#   1. Undersample High(2617→750) và VeryHigh(2494→750) xuống ngang Low(247)
#      → Tỉ lệ mới: [247, 750, 750] - cân bằng hơn nhiều
#   2. Focal Loss (gamma=2.0) + auto class weights (double coverage)
#   3. EMA reinit từ trained model tại ema_start_epoch=5
#   4. BỎ WeightedSampler (không cần thiết khi đã undersample)
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

echo "Starting DAiSEE v10 (Undersampling) Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3class_v10_undersample \
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
  --loss-type focal \
  --focal-gamma 2.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mixup-alpha 0.0 \
  --max-samples-per-class 750 \
  --use-amp \
  --use-ema \
  --ema-decay 0.99 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 10

echo "Training Finished!"
