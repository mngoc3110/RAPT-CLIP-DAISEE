#!/bin/bash

# =============================================================================
# DAiSEE 3-Class Engagement — COSINE CLASSIFIER + Gaze Fusion (v13)
#
# ROOT CAUSE of v10-v12 validation mode collapse:
#   Linear classifier with L2-normalized or near-identical features always
#   converges to predicting majority class because output depends on magnitude
#   (which is uniform after normalization), not angular differences.
#
# DEFINITIVE FIX — CosineClassifier (τ-normalized):
#   - L2-normalizes BOTH features AND class prototypes to unit sphere
#   - Output = tau * cosine_similarity(features, prototypes)
#   - Random prototypes → DIVERSE initial predictions guaranteed
#   - Learnable tau (init=16) → model controls confidence
#   - PROVEN: Kang et al. "Decoupling Representation and Classifier" (ICLR 2020)
#
# Additional:
#   - Focal Loss gamma=1.0 + DRW class weights (cap 5.0) from epoch 0
#   - No WeightedRandomSampler (avoid train/val distribution mismatch)
#   - Image encoder lr=2e-5 for feature adaptation
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE — Cosine Classifier + Gaze (v13)"
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
  --exper-name DAiSEE_CosineClassifier_v13 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 1e-3 \
  --lr-image-encoder 2e-5 \
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
  --val-annotation "$ANN_DIR/ValidationLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 1.0 \
  --use-classifier-head \
  --loss-type focal \
  --focal-gamma 1.0 \
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
