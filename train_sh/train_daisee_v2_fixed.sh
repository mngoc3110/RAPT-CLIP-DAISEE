#!/bin/bash

# =============================================================================
# DAiSEE Optimized Training — Version 2 (Fix accuracy issues)
#
# Các fix so với version cũ:
#   1. BỎ --use-weighted-sampler (gây overfit class 0 chỉ 34 mẫu)
#   2. --temperature 0.1 thay 0.07 (tránh double-scaling trap)
#   3. --temporal-layers 2 (deeper temporal modeling)
#   4. --num-segments 12 (nhiều frame hơn)
#   5. --scheduler cosine + warmup (ổn định hơn MultiStep)
#   6. --mixup-alpha 0.2 (regularization)
#   7. Focal gamma 2.0 (không cần sampler)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "Starting DAiSEE Optimized Training v2..."
echo "Root: $ROOT"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_Optimized_v2 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 3e-5 \
  --lr-image-encoder 5e-7 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 5 \
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
  --mi-warmup 8 \
  --mi-ramp 12 \
  --dc-warmup 8 \
  --dc-ramp 12 \
  --max-samples-per-class 0 \
  --mixup-alpha 0.2 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 8 \
  --grad-clip 1.0 \
  --early-stop 10 \
  --no-tta

echo "Training Finished!"
