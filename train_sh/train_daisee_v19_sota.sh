#!/bin/bash

# =============================================================================
# DAiSEE v19 — SOTA Data Processing + RAPT-CLIP
#
# Applies state-of-the-art data processing techniques from top DAiSEE papers:
#   1. Face Detection (OpenCV Haar Cascade) — adaptive face-aware cropping
#   2. Multi-scale Crop — random from [0.4, 0.5, 0.6] ratios
#   3. Strong Augmentation — GaussianBlur, RandomGrayscale, RandomErasing
#   4. Temporal Frame Dropout (15%) — forces temporal robustness
#   5. Class-balanced training — LDAM + WeightedSampler + undersample cap
#
# Based on techniques from:
#   - ResNet+TCN (63.9%), EfficientNetB7+LSTM (67.48%)
#   - CNN+SVD (77.97%), CNN+OpenFace+SMOTE (93.6% binary)
#
# Key differences from v17/v18:
#   + --use-face-detection        (new)
#   + --temporal-dropout 0.15     (new)
#   + --augment-strength strong   (new)
#   + --num-segments 16           (increased from 8 for more temporal info)
#   + --epochs 40                 (more epochs for stronger augmentation)
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

echo "Starting DAiSEE v19 (SOTA Data Processing) Training..."
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_4level_v19_sota \
  --dataset DAiSEE4Level \
  --gpu 0 \
  --epochs 40 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --scheduler cosine \
  --warmup-epochs 5 \
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
  --label-smoothing 0.05 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 3 \
  --mi-ramp 10 \
  --dc-warmup 3 \
  --dc-ramp 10 \
  --mixup-alpha 0.1 \
  --max-samples-per-class 1500 \
  --use-weighted-sampler \
  --use-amp \
  --use-ema \
  --ema-decay 0.995 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 12 \
  --use-face-detection \
  --temporal-dropout 0.15 \
  --augment-strength strong

echo "Training Finished!"
