#!/bin/bash

# =============================================================================
# DAiSEE Local Training — 3-Class (Merge Very Low + Low → Low)
#
# Dựa trên kết quả chẩn đoán:
#   - Class 0 (Very Low): chỉ 34 mẫu train / 4 mẫu test → MERGE
#   - Imbalance ratio 77:1 → BỎ WeightedSampler
#   - Face detection 100% OK → giữ dataloader hiện tại
# =============================================================================

ROOT="/Users/macbook/Downloads/RAPT-CLIP-DAISEE/DAiSEE"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  Fix: unfreeze image encoder + higher LR"
echo "  DAiSEE Local 3-Class Training"
echo "  Dataset: $ROOT"
echo "============================================"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_3Class_Local \
  --dataset DAiSEE \
  --gpu mps \
  --epochs 20 \
  --batch-size 4 \
  --workers 0 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 2e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 1 \
  --temporal-layers 2 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 20 \
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
  --mi-ramp 8 \
  --dc-warmup 5 \
  --dc-ramp 8 \
  --max-samples-per-class 0 \
  --mixup-alpha 0.2 \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 8 \
  --no-tta

echo "============================================"
echo "  Training Finished!"
echo "============================================"
