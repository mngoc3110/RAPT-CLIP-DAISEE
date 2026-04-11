#!/bin/bash

# =============================================================================
# DAiSEE 3-Class Engagement — LINEAR CLASSIFIER HEAD + Gaze Fusion (v11)
#
# ROOT CAUSE of v10 mode collapse:
#   video_features were L2-normalized to unit sphere BEFORE classifier head
#   → all inputs had magnitude 1.0 → gradient diversity killed → logits ~0.07
#
# Fix v11:
#   1. Classifier head receives RAW (un-normalized) features from project_fc
#   2. LayerNorm inside head normalizes properly for gradient flow
#   3. Temperature = 1.0 (no extra scaling needed for linear head)
#   4. Focal Loss + WeightedRandomSampler to fight class imbalance
#   5. Higher LR (1e-3) for classifier head to learn faster
#   6. DRW Phase 2 at epoch 5 for re-weighting
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE — Classifier Head + Gaze (v11)"
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
  --exper-name DAiSEE_ClassifierHead_v11 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 1e-3 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 3 \
  --temporal-layers 1 \
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
  --temperature 0.5 \
  --use-classifier-head \
  --loss-type focal \
  --focal-gamma 2.0 \
  --use-weighted-sampler \
  --drw-start-epoch 5 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mi-warmup 0 \
  --mi-ramp 0 \
  --dc-warmup 0 \
  --dc-ramp 0 \
  --max-samples-per-class 0 \
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 10 \
  --no-tta

echo "Training Finished!"
