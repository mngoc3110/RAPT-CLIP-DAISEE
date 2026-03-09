#!/bin/bash

# =============================================================================
# DAiSEE 4-Class Discrete Training Script
# Classes: Boredom(0), Engagement(1), Confusion(2), Frustration(3)
# Uses dominant affective state as class label
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels"

if [ ! -d "$ROOT" ]; then
    ROOT="/content/DAiSEE_data"
    ANN_DIR="$ROOT/Labels"
fi

echo "Starting DAiSEE 4-Discrete Training..."
echo "Root: $ROOT"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_4Discrete_v1 \
  --dataset DAiSEE4Discrete \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
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
  --root-dir "$ROOT" \
  --train-annotation "$ANN_DIR/TrainLabels.csv" \
  --val-annotation "$ANN_DIR/ValidationLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.07 \
  --loss-type ce \
  --label-smoothing 0.1 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mi-warmup 3 \
  --mi-ramp 7 \
  --dc-warmup 3 \
  --dc-ramp 7 \
  --mixup-alpha 0.0 \
  --use-amp \
  --grad-clip 1.0 \
  --early-stop 5 \
  --use-weighted-sampler

echo "Training Finished!"
