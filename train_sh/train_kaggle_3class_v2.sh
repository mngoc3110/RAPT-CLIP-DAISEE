#!/bin/bash

# =============================================================================
# DAiSEE Kaggle Training — SOTA-Aligned (v5)
#
# DAiSEE Kaggle Training — Gaze + CLIP Fusion (v6)
# Root cause fix:
#   - Trước: lr=5e-5 overfit train students, val collapse toàn bộ sang VH
#   - Gaze features (300,3) là student-independent → discriminative hơn raw CLIP
#   - lr-image-encoder 5e-6 (middle ground: học mực độ vừa phải, không overfit)
#   - Gaze MLP stream đã có sẵn trong kiến trúc, chỉ cần enable
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE Kaggle — Gaze+CLIP Fusion v7"
echo "  Root: $ROOT"
echo "============================================"

# Extract Gaze Features if not already extracted
if [ ! -d "/kaggle/working/Gaze_Features" ]; then
    echo "Extracting Gaze_Features.zip..."
    unzip -q Gaze_Features.zip -d /kaggle/working/
    echo "Gaze_Features extraction complete!"
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_GazeFusion_v7 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 2 \
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
  --focal-gamma 2.0 \
  --drw-start-epoch 0 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mi-warmup 0 \
  --mi-ramp 0 \
  --dc-warmup 0 \
  --dc-ramp 0 \
  --max-samples-per-class 0 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 3 \
  --grad-clip 1.0 \
  --early-stop 10 \
  --no-tta

echo "Training Finished!"
