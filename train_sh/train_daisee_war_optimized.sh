#!/bin/bash

# =============================================================================
# DAiSEE 4-Level Engagement Training - MAXIMIZING WAR (Weighted Average Recall)
# Classes: Very Low (0), Low (1), High (2), Very High (3)
# 
# Chiến lược cho WAR > 60:
#   - BỎ WeightedRandomSampler để mô hình học theo phân bố tự nhiên (thiên về High/Very High).
#   - Dùng Cross Entropy (ce) kết hợp Label Distribution Learning (LDL) thay vì LDAM.
#   - Giữ nguyên EMA và LR cosin để đảm bảo tính ổn định.
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

if [ ! -d "$ROOT" ]; then
    ROOT="/content/DAiSEE_data"
    ANN_DIR="$ROOT/Labels"
fi

echo "Starting DAiSEE 4-Level Engagement Training (WAR Optimized)..."
echo "Root: $ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: $ANN_DIR/TrainLabels.csv not found"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_4Level_WAR_Optimized \
  --dataset DAiSEE4Level \
  --gpu 0 \
  --epochs 25 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 20 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
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
  --label-smoothing 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --mixup-alpha 0.0 \
  --use-ldl \
  --ldl-temperature 2.0 \
  --ldl-warmup 5 \
  --use-amp \
  --use-ema \
  --ema-decay 0.99 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 8 \
  --no-tta

echo "Training Finished!"
