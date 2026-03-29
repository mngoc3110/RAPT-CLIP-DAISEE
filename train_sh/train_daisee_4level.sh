#!/bin/bash

# =============================================================================
# DAiSEE 4-Level Engagement Training
# Classes: Very Low (0), Low (1), High (2), Very High (3)
# 
# Đây là bài toán phân loại mức độ Engagement nguyên bản của DAiSEE.
# Do dữ liệu mất cân bằng trầm trọng (Very Low cực hiếm), dùng:
#   - LDAM Loss + WeightedRandomSampler
#   - Khớp config tốt nhất của RAPT-CLIP (MI/DC=0.1)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

if [ ! -d "$ROOT" ]; then
    ROOT="/content/DAiSEE_data"
    ANN_DIR="$ROOT/Labels"
fi

echo "Starting DAiSEE 4-Level Engagement Training..."
echo "Root: $ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: $ANN_DIR/TrainLabels.csv not found"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_4Level_Engagement \
  --dataset DAiSEE4Level \
  --gpu 0 \
  --epochs 25 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
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
  --loss-type ldam \
  --ldam-s 2.0 \
  --ldam-max-m 0.5 \
  --label-smoothing 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --use-weighted-sampler \
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.99 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 8 \
  --no-tta

echo "Training Finished!"
