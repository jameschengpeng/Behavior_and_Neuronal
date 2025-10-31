"""Custom loss functions for behavior prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def asymmetric_loss_with_mask(
    logits,
    targets,
    mask,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float | None = 0.05,
    eps: float = 1e-8,
    onset_tau: float | None = 5.0,
    normalize_onset: bool = True
):
    """
    Multi-label Asymmetric Loss with padding mask + onset-weighted positives.
    
    Heavily down-weights easy negatives (gamma_neg), lightly (or not) focuses 
    positives (gamma_pos), clips negatives for stability, and averages over 
    valid timesteps and all classes.

    Args:
        logits: (B, T, C) raw scores
        targets: (B, T, C) binary {0,1}
        mask: (B, T, 1) float (1=valid, 0=pad)
        gamma_pos: Focusing parameter for positives (0 = no focusing)
        gamma_neg: Focusing parameter for negatives (4 = strong down-weighting)
        clip: Clipping value for negative probabilities (0.05 typical)
        eps: Small constant for numerical stability
        onset_tau: If not None, apply exp(-d/onset_tau) within each positive segment,
                   where d = frames since the segment's onset (d=0 at onset).
                   Units are *frames* (after any downsampling).
        normalize_onset: If True, normalize positive-frame weights so mean is ~1
                         (prevents overall loss scale from shrinking with long segments).

    Returns:
        Scalar loss value
    """
    B, T, C = logits.shape
    device = logits.device
    dtype  = logits.dtype

    # Sigmoid and ASL parts
    x_sigmoid = torch.sigmoid(logits)
    xs_pos = x_sigmoid
    xs_neg = 1.0 - x_sigmoid

    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)

    log_pos = torch.log(xs_pos.clamp_min(eps))
    log_neg = torch.log(xs_neg.clamp_min(eps))

    loss_pos = targets * (1.0 - xs_pos) ** gamma_pos * (-log_pos)     # (B,T,C)
    loss_neg = (1.0 - targets) * (xs_pos) ** gamma_neg * (-log_neg)   # (B,T,C)

    # ----------------------- Onset-weighted positives -----------------------
    # Only applied where targets==1 and mask==1.
    if onset_tau is not None and onset_tau > 0:
        # Boolean masks
        mask_btc = (mask > 0).expand(-1, -1, C)                # (B,T,C) bool
        targets_bin = (targets > 0.5) & mask_btc               # (B,T,C) bool

        # Find onsets: current positive & previous frame not positive
        prev = torch.zeros_like(targets_bin)
        prev[:, 1:, :] = targets_bin[:, :-1, :]
        onset = targets_bin & (~prev)                           # (B,T,C) bool

        # For each (B,C), carry forward the most recent onset time index via cummax
        t_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, C)  # (B,T,C)
        neg_big = torch.full_like(t_idx, fill_value=-10_000, dtype=t_idx.dtype)
        onset_tidx = torch.where(onset, t_idx, neg_big)         # time-of-onset or -inf
        onset_cummax, _ = torch.cummax(onset_tidx, dim=1)       # last onset index so far

        # Frames since onset (>=0 inside positive segments; garbage elsewhere)
        d_since = (t_idx - onset_cummax).clamp_min(0)

        # Exponential weights w = exp(-d / tau) inside positives; 1 elsewhere
        w_onset = torch.exp(-d_since.to(dtype) / float(onset_tau))
        # Keep weights only where truly positive; elsewhere use 1.0 (no effect)
        w_onset = torch.where(targets_bin, w_onset, torch.ones_like(w_onset, dtype=dtype))

        if normalize_onset:
            # Normalize mean weight over positive frames to ~1 for stable loss scale
            pos_count = targets_bin.float().sum().clamp_min(1.0)
            mean_w = (w_onset * targets_bin.float()).sum() / pos_count
            w_onset = w_onset / mean_w.clamp_min(1e-6)

        # Apply to positive part only
        loss_pos = loss_pos * w_onset

    # ----------------------- Mask & reduction -----------------------
    loss = (loss_pos + loss_neg) * mask                          # (B,T,C)
    denom = (mask.sum() * C).clamp_min(1.0)
    return loss.sum() / denom


def embedding_aux_loss(model, mode="type", pairwise=True):
    """
    Auxiliary loss to encourage diversity in stimulus embeddings.
    
    Args:
        model: Model with stim_type_embedding or stim_loc_embedding
        mode: "type" or "loc" - which embedding to apply loss to
        pairwise: If True, sum over all unique pairs; if False, only adjacent pairs
        
    Returns:
        Scalar loss value (1 - cosine similarity averaged over pairs)
    """
    if mode == "type":
        emb = model.stim_type_embedding.weight  # (stim_type_dim, stim_embed_dim)
    elif mode == "loc":
        emb = model.stim_loc_embedding.weight   # (stim_loc_dim, stim_embed_dim)
    else:
        raise ValueError("mode must be 'type' or 'loc'")

    n = emb.shape[0]
    loss = 0.0
    count = 0
    if pairwise:
        for i in range(n):
            for j in range(i+1, n):
                loss += 1 - torch.cosine_similarity(emb[i], emb[j], dim=0)
                count += 1
    else:
        for i in range(n-1):
            loss += 1 - torch.cosine_similarity(emb[i], emb[i+1], dim=0)
            count += 1
    return loss / count if count > 0 else loss
