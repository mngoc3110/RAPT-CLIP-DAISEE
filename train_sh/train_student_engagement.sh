#!/bin/bash

# =============================================================================
# Student Engagement Dataset Training Script
# Dataset: https://www.kaggle.com/datasets/joyee19/studentengagement
# Binary: Engaged (0) vs Not Engaged (1)
# Static images → replicated to 8 frames for temporal module
# =============================================================================

DATASET_ROOT="/kaggle/input/datasets/joyee19/studentengagement/Student-engagement"

if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/kaggle/input/studentengagement/Student-engagement"
fi
if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/content/Student-engagement"
fi

echo "Starting Student Engagement Training..."
echo "Dataset Root: $DATASET_ROOT"

python3 main.py \
  --mode train \
  --exper-name StudentEng_v1 \
  --dataset StudentEngagement \
  --gpu 0 \
  --epochs 20 \
  --batch-size 16 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 2 \
  --temporal-layers 2 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 20 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "" \
  --val-annotation "" \
  --test-annotation "" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.07 \
  --loss-type ce \
  --label-smoothing 0.1 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 2 \
  --mi-ramp 5 \
  --dc-warmup 2 \
  --dc-ramp 5 \
  --mixup-alpha 0.0 \
  --use-amp \
  --grad-clip 1.0

echo "Training Finished!"
