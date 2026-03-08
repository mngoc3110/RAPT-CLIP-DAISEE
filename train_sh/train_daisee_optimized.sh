#!/bin/bash

# =============================================================================
# Script Huấn Luyện DAiSEE v4 - Balanced
# Key changes:
#   - Face = center 50% crop, Body = full frame (CLIP understands natural images)
#   - lr-image-encoder = 5e-7 (slight fine-tune, not frozen)
#   - label-smoothing 0.1 (keep, reduces overconfidence)
#   - mixup 0.0 (removed — was hurting convergence)
#   - temperature 0.07 (back to original, sharper separation)
#   - lr 2e-5, lr-prompt-learner 2e-4 (back to v2 levels)
# =============================================================================

DATASET_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${DATASET_ROOT}/Labels"

if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/content/DAiSEE_local"
    ANN_DIR="${DATASET_ROOT}/Labels"
fi
if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/content/drive/MyDrive/RAPT-CLIP-DAISEE/DAiSEE"
    ANN_DIR="${DATASET_ROOT}/Labels"
fi

echo "Starting DAiSEE v4 Balanced Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "LỖI: Không tìm thấy file nhãn tại $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_v4_Balanced \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 16 \
  --workers 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 5e-7 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --scheduler cosine \
  --warmup-epochs 2 \
  --temporal-layers 2 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
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
  --ldam-max-m 0.5 \
  --ldam-s 1.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 3 \
  --mi-ramp 7 \
  --dc-warmup 3 \
  --dc-ramp 7 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.0 \
  --use-weighted-sampler \
  --use-amp \
  --grad-clip 1.0

echo "Training Finished!"
