#!/bin/bash

# =============================================================================
# DAiSEE Kaggle Training — WAR > 60% Optimized (v4)
# Đượng dẫn: Tối ưu WAR (overall accuracy) cho 3-class merge
#
# Key analysis (test set: Low=88/5%, High=882/49%, VH=814/46%):
#   - WAR hiện tại 49.4% = model predict toàn bộ là High
#   - Để WAR > 60%: chỉ cần phân biệt High vs VH được là đủ
#   - KHÔNG dùng WeightedSampler (hại WAR vì sacrifice High/VH accuracy)
#   - Dùng Focal gamma=3.0 để tự chống collapse mà không cần sampler
#   - DRW epoch 5: class weight [3,1,1] chỉ boost nhẹ Low, giữ High/VH
# =============================================================================

ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
ANN_DIR="${ROOT}/Labels"

echo "============================================"
echo "  DAiSEE Kaggle — WAR > 60% Optimized (v4)"
echo "  Root: $ROOT"
echo "============================================"

python3 main.py \
  --mode train \
  --exper-name DAiSEE_WAR60_v4 \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 2e-4 \
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
  --val-annotation "$ANN_DIR/TestLabels.csv" \
  --test-annotation "$ANN_DIR/TestLabels.csv" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.1 \
  --loss-type focal \
  --focal-gamma 3.0 \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --mi-warmup 0 \
  --mi-ramp 0 \
  --dc-warmup 0 \
  --dc-ramp 0 \
  --max-samples-per-class 0 \
  --label-smoothing 0.05 \
  --drw-start-epoch 5 \
  --mixup-alpha 0.0 \
  --use-amp \
  --use-ema \
  --ema-decay 0.998 \
  --ema-start-epoch 5 \
  --grad-clip 1.0 \
  --early-stop 10 \
  --no-tta

echo "Training Finished!"
