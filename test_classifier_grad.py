"""Quick test: verify classifier_head is trainable and gradients flow."""
import torch
import sys
sys.path.insert(0, '.')

from clip import clip
from models.Generate_Model import GenerateModel, CosineClassifier
from utils.builders import build_model, get_class_info
import argparse

print("=" * 60)
print("TEST: Classifier Head Gradient Flow")
print("=" * 60)

# Minimal args
args = argparse.Namespace(
    dataset='DAiSEE', text_type='prompt_ensemble', clip_path='ViT-B/16',
    num_segments=2, temporal_layers=1, use_classifier_head=True,
    lr_image_encoder=1e-5, contexts_number=4, class_token_position='end',
    class_specific_contexts='True', load_and_tune_prompt_learner='True',
    use_moco=False, temperature=1.0
)

# Build model using the SAME function as training
_, input_text = get_class_info(args)
model = build_model(args, input_text)

# TEST 1: Check requires_grad
print("\n--- TEST 1: Parameter requires_grad ---")
frozen_critical = []
trainable_critical = []
for name, param in model.named_parameters():
    if any(k in name for k in ['classifier_head', 'gaze_mlp', 'alpha_gaze']):
        if param.requires_grad:
            trainable_critical.append(name)
        else:
            frozen_critical.append(name)

if frozen_critical:
    print(f"❌ FAIL: These critical params are FROZEN:")
    for n in frozen_critical:
        print(f"   - {n}")
else:
    print(f"✅ PASS: All critical params are trainable ({len(trainable_critical)} params)")
    for n in trainable_critical:
        print(f"   - {n}")

# TEST 2: Forward + backward pass
print("\n--- TEST 2: Forward + Backward ---")
model.train()
batch_size = 2
dummy_face = torch.randn(batch_size, 2, 3, 224, 224)  # (B, T, C, H, W)
dummy_body = torch.randn(batch_size, 2, 3, 224, 224)
dummy_target = torch.tensor([0, 2])

output, text_feat, hc_feat, moco = model(dummy_face, dummy_body)
print(f"Output shape: {output.shape}")
print(f"Output logits: {output.detach().numpy()}")
print(f"Predictions: {output.argmax(dim=1).tolist()}")

loss = torch.nn.functional.cross_entropy(output, dummy_target)
loss.backward()

# TEST 3: Check gradients are non-zero
print("\n--- TEST 3: Gradient Check ---")
grad_ok = True
for name, param in model.named_parameters():
    if 'classifier_head' in name and param.requires_grad:
        if param.grad is None:
            print(f"❌ FAIL: {name} has NO gradient")
            grad_ok = False
        elif param.grad.abs().sum() == 0:
            print(f"❌ FAIL: {name} gradient is all zeros")
            grad_ok = False
        else:
            print(f"✅ {name}: grad norm = {param.grad.norm().item():.6f}")

if grad_ok:
    print("\n🎉 ALL TESTS PASSED - classifier_head IS being trained!")
else:
    print("\n💀 TESTS FAILED - classifier_head is NOT learning!")

# TEST 4: Check tau
if hasattr(model.classifier_head, 'tau'):
    print(f"\n--- TEST 4: CosineClassifier tau = {model.classifier_head.tau.item():.2f} ---")
    if model.classifier_head.tau.grad is not None:
        print(f"✅ tau grad = {model.classifier_head.tau.grad.item():.6f}")
    else:
        print(f"❌ tau has no gradient")
