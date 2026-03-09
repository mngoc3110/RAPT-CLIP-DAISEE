#!/bin/bash

# =============================================================================
# Student Engagement 6-Class Training Script
# Classes: confused(0), engaged(1), frustrated(2), looking_away(3), bored(4), drowsy(5)
# =============================================================================

DATASET_ROOT=""

# Search common Kaggle paths
for CANDIDATE in \
    "/kaggle/input/studentengagement" \
    "/kaggle/input/studentengagement/Student-engagement" \
    "/kaggle/input/datasets/joyee19/studentengagement" \
    "/kaggle/input/datasets/joyee19/studentengagement/Student-engagement" \
    "/kaggle/input/datasets/joyee19/studentengagement/Student-engagement-dataset" \
    "/content/Student-engagement" \
    "/content/studentengagement"; do
    if [ -d "$CANDIDATE" ]; then
        DATASET_ROOT="$CANDIDATE"
        break
    fi
done

if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: Cannot find dataset."
    ls -laR /kaggle/input/ 2>/dev/null | head -30
    exit 1
fi

# Auto-detect subfolder with class directories
if [ ! -d "$DATASET_ROOT/Engaged" ] && [ ! -d "$DATASET_ROOT/Not Engaged" ]; then
    SUBFOLDER=$(find "$DATASET_ROOT" -maxdepth 2 -type d -name "confused" 2>/dev/null | head -1)
    if [ -n "$SUBFOLDER" ]; then
        DATASET_ROOT=$(dirname $(dirname "$SUBFOLDER"))
        echo "Auto-detected root: $DATASET_ROOT"
    fi
fi

echo "Starting Student Engagement 6-Class Training..."
echo "Dataset Root: $DATASET_ROOT"
echo "Contents:"
ls -la "$DATASET_ROOT"

python3 main.py \
  --mode train \
  --exper-name StudentEng_6class_v1 \
  --dataset StudentEngagement6 \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 2 \
  --temporal-layers 2 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 20 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "" \
  --val-annotation "" \
  --test-annotation "" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.07 \
  --loss-type ce \
  --label-smoothing 0.1 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mi-warmup 2 \
  --mi-ramp 5 \
  --dc-warmup 2 \
  --dc-ramp 5 \
  --mixup-alpha 0.0 \
  --use-amp \
  --grad-clip 1.0 \
  --early-stop 5

echo "6-Class Training Finished!"
