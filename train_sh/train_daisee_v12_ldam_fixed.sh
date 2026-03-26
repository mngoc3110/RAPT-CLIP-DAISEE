#!/bin/bash

# =============================================================================
# DAiSEE v12 — Fix double-scaling & EMA freeze from v11
# Changes from v11:
#   1. ldam-s=1.0 (was 30.0) — temperature 0.07 already scales logits 14x,
#      s=30 on top created 428x total → loss ~14 and frozen validation
#   2. ema-decay=0.995 (was 0.999) — 0.999 only mixed 0.1% new weights/epoch
#      → validation confusion matrix frozen from epoch 5-12
#   3. ema-start-epoch=3 (was 5) — start EMA earlier after warmup ends
#   4. mi-warmup=3, dc-warmup=3 (was 5) — let auxiliary losses help sooner
#   5. max-samples-per-class=0 — LDAM handles imbalance, no undersampling needed
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

echo "Starting DAiSEE v12 (LDAM fixed scaling) Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3class_v12_ldam_fixed \
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
  --ldam-s 1.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 3 \
  --mi-ramp 10 \
  --dc-warmup 3 \
  --dc-ramp 10 \
  --mixup-alpha 0.0 \
  --max-samples-per-class 0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.995 \
  --ema-start-epoch 3 \
  --grad-clip 1.0 \
  --early-stop 10

echo "Training Finished!"
