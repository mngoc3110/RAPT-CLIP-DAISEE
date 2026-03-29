#!/bin/bash

# =============================================================================
# DAiSEE v11 — Original RAPT-CLIP Architecture with LDAM Loss
# Fixes from v10:
#   1. LDAM loss (original RAPT-CLIP) instead of Focal loss
#   2. MI & DC auxiliary losses enabled (lambda=0.1)
#   3. Body stream uses wide crop (0.9) vs face tight crop (0.5)
#   4. Undersampling kept at 750 for majority classes
#   5. No mixup (keep original)
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

echo "Starting DAiSEE v11 (LDAM + MI/DC) Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3class_v11_ldam \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
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
  --ldam-max-m 0.5 \
  --ldam-s 30.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --mixup-alpha 0.0 \
  --max-samples-per-class 750 \
  --use-amp \
  --use-ema \
  --ema-decay 0.999 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 10

echo "Training Finished!"
