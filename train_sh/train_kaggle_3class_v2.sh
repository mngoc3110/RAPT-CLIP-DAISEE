#!/bin/bash

# =============================================================================
# DAiSEE Kaggle Training — 3-Class (Fix Mode Collapse)
# GPU: T4/P100 trên Kaggle
#
# Fixes:
#   1. lr-image-encoder = 1e-6 (unfreeze nhẹ CLIP encoder)
#   2. BỎ WeightedSampler → Focal Loss xử lý imbalance
#   3. temperature = 0.1 (tránh double-scaling)
#   4. temporal-layers = 2, num-segments = 12
#   5. merge_3class=True (Very Low + Low → Low, 247 mẫu thay vì 34)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE Kaggle 3-Class Training (Fix v2)"
echo "  Root: $ROOT"
echo "============================================"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3Class_Kaggle_v2 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 25 \
  --batch-size 16 \
  --workers 4 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 2e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 3 \
  --temporal-layers 2 \
  --num-segments 12 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$ROOT" \
  --train-annotation "$ANN_DIR/TrainLabels.csv" \
  --val-annotation "$ANN_DIR/TestLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.1 \
  --loss-type focal \
  --focal-gamma 2.0 \
  --lambda_mi 0.05 \
  --lambda_dc 0.05 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --max-samples-per-class 0 \
  --mixup-alpha 0.2 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 8 \
  --no-tta

echo "Training Finished!"
