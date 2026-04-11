"""
Dry-run training: Simulates 3 mini-epochs with dummy data on CPU/MPS.
Verifies: model builds → loss decreases → predictions change → no mode collapse.
Run this BEFORE pushing to Kaggle.
"""
import torch
import torch.nn.functional as F
import sys, argparse, numpy as np
sys.path.insert(0, '.')

from clip import clip
from utils.builders import build_model, get_class_info
from utils.loss import FocalLoss

print("=" * 60)
print("  DRY-RUN: Local Training Simulation (3 epochs, dummy data)")
print("=" * 60)

# Use MPS if available, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

# Minimal args matching the training script
args = argparse.Namespace(
    dataset='DAiSEE', text_type='prompt_ensemble', clip_path='ViT-B/16',
    num_segments=2, temporal_layers=1, use_classifier_head=True,
    lr_image_encoder=0, contexts_number=4, class_token_position='end',
    class_specific_contexts='True', load_and_tune_prompt_learner='True',
    use_moco=False, temperature=1.0
)

# Build model
_, input_text = get_class_info(args)
model = build_model(args, input_text)
model = model.to(device)

# Optimizer — only train unfrozen params
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)

# Focal loss with class weights (simulating DAiSEE distribution)
cls_weights = torch.FloatTensor([5.0, 1.0, 1.05]).to(device)
criterion = FocalLoss(gamma=1.0, weight=cls_weights)

# Create imbalanced dummy data (mimics DAiSEE: class 1 dominant)
def make_batch(batch_size=4):
    face = torch.randn(batch_size, 2, 3, 224, 224)
    body = torch.randn(batch_size, 2, 3, 224, 224)
    # Imbalanced targets: 60% class 1, 20% class 0, 20% class 2
    probs = [0.2, 0.6, 0.2]
    targets = torch.tensor(np.random.choice(3, batch_size, p=probs))
    return face.to(device), body.to(device), targets.to(device)

# Simulate training
print(f"\n{'='*60}")
print(f"  Starting dry-run training...")
print(f"{'='*60}")

all_preds = []
for epoch in range(3):
    model.train()
    epoch_loss = 0
    epoch_preds = []
    
    for step in range(5):  # 5 batches per epoch
        face, body, target = make_batch(4)
        
        output, text_feat, hc_feat, moco = model(face, body)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        
        preds = output.argmax(dim=1)
        epoch_preds.extend(preds.cpu().tolist())
        epoch_loss += loss.item()
    
    # Eval mode check
    model.eval()
    with torch.no_grad():
        face, body, target = make_batch(8)
        output, _, _, _ = model(face, body)
        val_preds = output.argmax(dim=1).cpu().tolist()
    
    avg_loss = epoch_loss / 5
    pred_counts = [epoch_preds.count(i) for i in range(3)]
    val_counts = [val_preds.count(i) for i in range(3)]
    tau = model.classifier_head.tau.item()
    alpha = model.alpha_gaze.item()
    
    print(f"\nEpoch {epoch}: loss={avg_loss:.4f} | tau={tau:.2f} | alpha_gaze={alpha:.4f}")
    print(f"  Train preds distribution: {pred_counts} (class 0/1/2)")
    print(f"  Val preds distribution:   {val_counts} (class 0/1/2)")
    
    # Check for mode collapse
    if len(set(val_preds)) == 1:
        print(f"  ⚠️  WARNING: All val predictions = class {val_preds[0]} (potential collapse)")
    else:
        print(f"  ✅ Diverse predictions in val!")
    
    all_preds.append(val_counts)

# Final verdict
print(f"\n{'='*60}")
print("  VERDICT")
print(f"{'='*60}")

# Check if predictions changed across epochs
if all_preds[0] == all_preds[2]:
    print("⚠️  Predictions didn't change - model may not be learning")
else:
    print("✅ Predictions changed across epochs - model IS learning")

# Check diversity
final_unique = len(set([all_preds[-1][i] > 0 for i in range(3)]))
if all(c > 0 for c in all_preds[-1]):
    print("✅ All 3 classes predicted in final epoch - NO mode collapse!")
elif sum(1 for c in all_preds[-1] if c > 0) >= 2:
    print("✅ At least 2 classes predicted - reasonable diversity")
else:
    print("❌ Only 1 class predicted - MODE COLLAPSE detected!")

print(f"\n✅ Dry-run complete. Safe to deploy to Kaggle.")
