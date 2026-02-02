"""Utility functions for temporal MAE."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def generate_patch_mask(
    lengths: torch.Tensor,
    patch_len: int,
    mask_ratio: float,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate patch-level masks for sequences with variable lengths.

    Args:
        lengths: (B,) actual sequence lengths
        patch_len: length of each temporal patch
        mask_ratio: fraction of patches to mask
        device: torch device

    Returns:
        mask: (B, T_max) boolean, True for masked timesteps
        valid_mask: (B, T_max) boolean, True for valid (non-padded) timesteps
    """
    if device is None:
        device = lengths.device

    lengths = lengths.to(device)
    batch_size = int(lengths.shape[0])
    t_max = int(lengths.max().item())
    patch_len = max(1, int(patch_len))
    mask_ratio = max(0.0, min(1.0, float(mask_ratio)))

    valid_mask = torch.arange(t_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = torch.zeros((batch_size, t_max), dtype=torch.bool, device=device)

    for b in range(batch_size):
        t_len = int(lengths[b].item())
        num_patches = max(1, math.ceil(t_len / patch_len))
        num_mask = int(round(num_patches * mask_ratio))
        if num_mask <= 0:
            continue

        patch_indices = torch.randperm(num_patches, device=device)[:num_mask]
        for p in patch_indices.tolist():
            start = p * patch_len
            end = min(t_len, start + patch_len)
            mask[b, start:end] = True

    return mask, valid_mask


def masked_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute MSE over masked positions with optional lower weight for S channels.

    recon/target: (B, T, D)
    mask: (B, T) boolean, True indicates masked
    """
    if recon.shape != target.shape:
        raise ValueError("recon and target must have the same shape")

    B, T, D = target.shape

    mask_bt = mask.unsqueeze(-1).expand(-1, -1, D)  # (B, T, D)
    if valid_mask is not None:
        valid_bt = valid_mask.unsqueeze(-1).expand(-1, -1, D)
        mask_bt = mask_bt & valid_bt
    diff = (recon - target) ** 2

    masked = diff * mask_bt
    denom = mask_bt.sum().clamp_min(1.0)
    return masked.sum() / denom
