# utils/ema.py
import torch
import copy

class ModelEMA:
    """Exponential Moving Average of model weights.
    
    Keeps a running average of model parameters for more stable predictions.
    Usage:
        ema = ModelEMA(model, decay=0.999)
        # After each training step:
        ema.update(model)
        # For evaluation:
        ema.apply(model)  # temporarily load EMA weights
        # ... evaluate ...
        ema.restore(model)  # restore original weights
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Deep copy of the model's state dict
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters after each training step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model):
        """Apply EMA weights to model (save originals for restore)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
