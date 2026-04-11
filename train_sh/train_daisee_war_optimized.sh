#!/bin/bash

# =============================================================================
# DAiSEE 3-Class Engagement — LINEAR CLASSIFIER HEAD + Gaze Fusion (v10)
#
# Root cause: CLIP text embeddings for "High" vs "Very High" are identical
# in cosine space → model cannot create decision boundary via similarity.
#
# Fix: Replace CLIP text-image similarity with a LEARNED linear classifier.
# Visual backbone still uses CLIP pre-trained weights (frozen/low lr).
# Classification is done by: classifier_head(video_features) → logits
#
# Components:
#   1. Classifier Head: Linear(512→256→3) — learns proper boundaries
#   2. OrdinalCE Loss: Soft labels respect label ordering
#   3. Gaze Fusion: alpha_gaze * gaze_mlp(gaze_avg) added to video_features
#   4. Image encoder lr=5e-6: Gentle fine-tune, no identity overfit
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE — Classifier Head + Gaze (v10)"
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
  --exper-name DAiSEE_ClassifierHead_v10 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-4 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 2 \
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
  --temperature 0.1 \
  --use-classifier-head \
  --loss-type ordinal_ce \
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
  --ema-start-epoch 3 \
  --grad-clip 1.0 \
  --early-stop 10 \
  --no-tta

echo "Training Finished!"
