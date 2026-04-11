#!/bin/bash

# =============================================================================
# DAiSEE Kaggle Training — 3-Class (Anti Mode-Collapse v3)
# GPU: T4/P100 trên Kaggle
#
# Fixes v3 (anti mode-collapse):
#   1. --use-weighted-sampler (sqrt-freq: class 0=x10, class 1=x1, class 2=x1)
#   2. focal-gamma 3.0 (higher = more penalty on easy majority class)
#   3. drw-start-epoch 3 (kích hoạt class weight SỚM)
#   4. tắt MI/DC loss (nhiễu khi model chưa học được gì)
#   5. label-smoothing 0 (không làm mềm target khi đang collapse)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE Kaggle 3-Class Training (Anti-Collapse v3)"
echo "  Root: $ROOT"
echo "============================================"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3Class_Kaggle_v2 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 25 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 2e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 3 \
  --temporal-layers 2 \
  --num-segments 8 \
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
  --focal-gamma 3.0 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mi-warmup 0 \
  --mi-ramp 0 \
  --dc-warmup 0 \
  --dc-ramp 0 \
  --max-samples-per-class 0 \
  --label-smoothing 0.0 \
  --use-weighted-sampler \
  --drw-start-epoch 3 \
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 8 \
  --no-tta

echo "Training Finished!"
