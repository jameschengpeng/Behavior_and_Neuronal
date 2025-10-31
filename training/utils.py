"""Training utility functions."""

import torch
import torch.nn as nn


@torch.no_grad()
def compute_pos_weight_and_prior(loader, num_labels, device, cap=50.0, eps=1e-8):
    """
    Compute pos_weight and prior over the dataset, masking out invalid timesteps.
    
    Args:
        loader: DataLoader yielding (xb, yb, lengths, ...)
        num_labels: Number of behavior labels
        device: torch device
        cap: Maximum value for pos_weight (prevents explosion)
        eps: Small constant for numerical stability
        
    Returns:
        pos_weight: (C,) tensor for BCE loss weighting
        prior: (C,) tensor of class priors
    """
    pos_sum = torch.zeros(num_labels, dtype=torch.float64, device=device)
    valid_frames = torch.zeros((), dtype=torch.float64, device=device)

    for xb, yb, lengths, *_ in loader:
        yb = yb.to(device).float()             # (B, T_max, C)
        lengths = lengths.to(device).long()
        B, T_max, C = yb.shape
        # Build valid-time mask
        mask = (torch.arange(T_max, device=device).view(1, T_max, 1)
                < lengths.view(B, 1, 1)).to(yb.dtype)           # (B, T, 1)

        pos_sum += (yb * mask).sum(dim=(0,1))                   # (C,)
        valid_frames += mask.sum()                               # scalar (#valid timesteps)

    prior = (pos_sum / (valid_frames + eps)).clamp(1e-6, 1-1e-6)  # (C,)
    neg_sum = valid_frames - pos_sum
    pos_weight = (neg_sum / (pos_sum + eps)).clamp(max=cap).to(torch.float32)
    return pos_weight.to(torch.float32), prior.to(torch.float32)


def init_classifier_bias_from_prior(model, prior, eps=1e-6):
    """
    Initialize classifier bias based on class priors.
    
    Sets bias = log(prior / (1 - prior)) so that initially the model
    outputs the prior probabilities, avoiding early saturation at 0 or 1.
    
    Args:
        model: Model with a .classifier.bias parameter
        prior: (C,) tensor of class priors in [0, 1]
        eps: Small constant to avoid log(0)
    """
    # Clamp away from 0/1 before logit
    prior = prior.clamp(min=eps, max=1 - eps)
    bias = torch.log(prior) - torch.log(1 - prior)  # safe logit
    with torch.no_grad():
        model.classifier.bias.copy_(bias)
