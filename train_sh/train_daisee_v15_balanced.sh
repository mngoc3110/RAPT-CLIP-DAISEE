#!/bin/bash

# =============================================================================
# DAiSEE v15 — Balanced between v13 (overfit) and v14 (underfit)
#
# v13: lr-image-encoder=1e-6 → Train 70% vs Valid 40% (overfit 30%)
# v14: lr-image-encoder=0   → Train 45% vs Valid 40% (underfit)
# v15: lr-image-encoder=5e-7 → sweet spot, minimal CLIP adaptation
#
# Other balance changes:
#   - mixup=0.1 (v14 had 0.2, too aggressive; v13 had 0)
#   - label-smoothing=0.05 (v14 had 0.1)
#   - num-segments=8 (v14's 16 too much for small dataset, slower training)
#   - undersample 750 + sampler (proven to balance classes)
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

echo "Starting DAiSEE v15 (Balanced Fine-tuning) Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3class_v15_balanced \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 5e-7 \
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
  --mixup-alpha 0.1 \
  --max-samples-per-class 750 \
  --use-weighted-sampler \
  --use-amp \
  --use-ema \
  --ema-decay 0.995 \
  --ema-start-epoch 3 \
  --grad-clip 1.0 \
  --early-stop 10

echo "Training Finished!"
