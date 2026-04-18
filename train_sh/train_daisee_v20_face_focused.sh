#!/bin/bash

# =============================================================================
# DAiSEE v20 — Face-Focused 4-Level Engagement Recognition
#
# Key innovations:
#   1. Face-Only Mode (--face-only-mode)
#      - Learnable Face Attention Gate (init 70/30 face/body)
#      - Both streams receive face crops at different scales:
#        Face branch: tight crop (0.4-0.6), Body branch: wider crop (0.65-0.85)
#      - Face Adapter with doubled capacity (reduction=2)
#      - Body Adapter added for richer secondary features
#
#   2. Face-Focused Prompt Ensemble
#      - 7 prompts per class × 4 classes = 28 prompts
#      - All prompts describe facial features (eyes, brows, mouth, head pose)
#      - Based on Facial Action Units (AUs) for CLIP discriminability
#
#   3. Extreme Imbalance Handling (DAiSEE 4-level is ~200:800:4000:1000)
#      - LDAM Loss with margin-based classification
#      - WeightedRandomSampler for balanced batch sampling
#      - Undersample cap at 1200 per class (prevents High class domination)
#      - Strong augmentation to increase effective minority samples
#      - Label smoothing 0.05 to prevent overconfident predictions
#
#   4. SOTA Data Processing
#      - Face Detection (OpenCV Haar Cascade)
#      - Multi-scale crop with jitter
#      - Strong augmentation (GaussianBlur + Grayscale + RandomErasing)
#      - Temporal Frame Dropout (15%)
#
# Expected performance improvement over v19:
#   - Better separation between adjacent levels (VeryLow↔Low, High↔VeryHigh)
#   - Improved minority class recall (VeryLow, VeryHigh)
#   - Face gate learns optimal face vs context weighting
# =============================================================================

# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

echo "======================================================"
echo "  DAiSEE v20 — Face-Focused 4-Level Engagement"
echo "  Architecture: RAPT-CLIP + Face Attention Gate (70/30)"
echo "  Classes: Very Low | Low | High | Very High"
echo "======================================================"
echo "Dataset Root: $DATASET_ROOT"

if [ ! -f "$ANN_DIR/TrainLabels.csv" ]; then
    echo "ERROR: Label file not found at $ANN_DIR/TrainLabels.csv"
    exit 1
fi

python3 main.py \
  --mode train \
  --exper-name DAiSEE_4level_v20_face_focused \
  --dataset DAiSEE4Level \
  --gpu 0 \
  --epochs 50 \
  --batch-size 4 \
  --workers 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 5e-7 \
  --lr-prompt-learner 1e-4 \
  --lr-adapter 5e-5 \
  --weight-decay 0.05 \
  --scheduler cosine \
  --warmup-epochs 8 \
  --temporal-layers 2 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANN_DIR/TrainLabels.csv" \
  --val-annotation "$ANN_DIR/TestLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.05 \
  --loss-type ldam \
  --ldam-max-m 0.3 \
  --ldam-s 1.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --mixup-alpha 0.1 \
  --max-samples-per-class 1200 \
  --use-weighted-sampler \
  --use-amp \
  --use-ema \
  --ema-decay 0.995 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 15 \
  --face-only-mode \
  --use-face-detection \
  --temporal-dropout 0.15 \
  --augment-strength strong

echo "Training Finished!"
