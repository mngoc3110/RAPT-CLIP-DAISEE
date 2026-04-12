#!/bin/bash

# =============================================================================
# DAiSEE 3-Class Engagement — COSINE CLASSIFIER + Gaze Fusion (v14)
#
# ROOT CAUSE of v13 mode collapse in Epoch 2:
#   LR 1e-3 was overshooting + Focal Gamma 1.0 was too weak to suppress
#   the massive majority class (Class 1) once warmup ended.
#
# DEFINITIVE FIX v14:
#   1. Focal Gamma 2.0 (stronger suppression of easy samples)
#   2. Lower Classifier LR (5e-4) for stable convergence
#   3. DRW Weight Cap 15.0 (for real rebalancing: 247 vs 2615 samples)
#   4. Tau init 12.0 (softer confidence early on)
#   5. Weight Decay 0.02 (stronger regularization)
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE — Cosine Classifier + Gaze (v14)"
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
  --exper-name DAiSEE_CosineClassifier_v14 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-4 \
  --lr-image-encoder 2e-5 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.02 \
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
  --temperature 1.0 \
  --use-classifier-head \
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
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 10 \
  --no-tta

echo "Training Finished!"
