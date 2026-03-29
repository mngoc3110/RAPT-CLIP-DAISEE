#!/bin/bash

# =============================================================================
# DAiSEE 4-Class Discrete Training v2
# Classes: Boredom(0) / Engagement(1) / Confusion(2) / Frustration(3)
# Strategy:
#   - Dùng dominant affective state làm label cho mỗi video clip
#   - Merge Train + Validation → train (thêm data, ~6787 samples)
#   - Dùng Test làm validation (validation ko update weights)
#   - Focal Loss + WeightedSampler (imbalanced 4 class)
#   - MI+DC loss bật lại (shape mismatch đã fix)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

if [ ! -d "$ROOT" ]; then
    ROOT="/content/DAiSEE_data"
    ANN_DIR="$ROOT/Labels"
fi

echo "Starting DAiSEE 4-Discrete v2 Training..."
echo "Root: $ROOT"
echo "Train = TrainLabels + ValidationLabels merged"
echo "Val   = TestLabels"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: $ANN_DIR/TrainLabels.csv not found"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_4Discrete_v2 \
  --dataset DAiSEE4Discrete \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 2e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
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
  --extra-train-annotations "$ANN_DIR/ValidationLabels.csv" \
  --val-annotation "$ANN_DIR/TestLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.07 \
  --loss-type focal \
  --focal-gamma 2.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.05 \
  --lambda_dc 0.05 \
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
  --early-stop 10

echo "Training Finished!"
