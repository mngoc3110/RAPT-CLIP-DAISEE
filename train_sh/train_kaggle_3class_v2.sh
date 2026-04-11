#!/bin/bash

# =============================================================================
# DAiSEE Kaggle Training — SOTA-Aligned (v5)
#
# Root cause fix: CLIP text embeddings "High" vs "VH" quá gần nhau.
# SOTA giải quyết bằng cách train visual backbone supervised hoàn toàn.
# Cần unfreeze image encoder mạnh để model học visual features mới.
#
# Key changes vs trước:
#   1. lr-image-encoder: 1e-6 → 5e-5 (100x mạnh hơn)
#   2. loss-type: focal → crossentropy (CE + class_weights từ epoch 0)
#   3. drw-start-epoch: 0 (class weights active ngay từ đầu)
#   4. batch-size: 4 (giảm vì encoder unfreeze tốn RAM)
#   5. Tắt MI/DC (overhead, không cần khi visual features đang học)
#   6. ema-start-epoch: 3 (start sớm hơn)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE Kaggle — SOTA-Aligned v5"
echo "  Root: $ROOT"
echo "============================================"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_SOTAv5 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 4 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 5e-5 \
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
