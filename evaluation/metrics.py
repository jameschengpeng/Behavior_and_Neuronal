"""Metrics computation and threshold tuning."""

import torch
import torch.nn as nn


@torch.inference_mode()
def tune_per_class_thresholds(val_loader, model, num_labels, grid=None):
    """
    Tune per-class decision thresholds on validation set.
    
    For each class, searches over a grid of thresholds to find the one
    that maximizes F1 score on the validation set.
    
    Args:
        val_loader: Validation DataLoader
        model: Trained model
        num_labels: Number of behavior classes
        grid: Threshold grid to search (defaults to linspace(0.1, 0.9, 17))
        
    Returns:
        best_th: (C,) tensor of per-class optimal thresholds
    """
    device = next(model.parameters()).device
    sigmoid = nn.Sigmoid()
    if grid is None:
        grid = torch.linspace(0.1, 0.9, 17, device=device)  # 0.10..0.90

    num_th = grid.numel()
    tp = torch.zeros(num_th, num_labels, dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)

    for batch in val_loader:
        if len(batch) == 8:
            xb, yb, lengths, st, sl, so, sf, sent_text = batch
        else:
            xb, yb, lengths, st, sl, so, sf = batch
            sent_text = None

        xb = xb.to(device)
        yb = yb.to(device).float()
        lengths = lengths.to(device).long()
        st, sl, so, sf = st.to(device), sl.to(device), so.to(device), sf.to(device)

        logits = model(xb, st, sl, so, sf, lengths, sent_text=sent_text)
        probs  = sigmoid(logits)

        B, T, C = yb.shape
        time_idx = torch.arange(T, device=device).view(1, T, 1)
        mask = (time_idx < lengths.view(B,1,1))  # (B,T,1) bool
        y_true = (yb > 0.5) & mask

        probs_th = probs.unsqueeze(0).expand(num_th, -1, -1, -1)  # (num_th,B,T,C)
        th_mat = grid.view(num_th, 1, 1, 1)
        preds = (probs_th >= th_mat) & mask.unsqueeze(0)

        tp += (preds & y_true.unsqueeze(0)).sum(dim=(1,2)).to(tp.dtype)
        fp += (preds & (~y_true.unsqueeze(0))).sum(dim=(1,2)).to(fp.dtype)
        fn += ((~preds) & y_true.unsqueeze(0)).sum(dim=(1,2)).to(fn.dtype)

    eps = 1e-8
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)

    best_idx = torch.argmax(f1, dim=0)     # (C,)
    best_th  = grid[best_idx]              # (C,)
    return best_th.detach().float().cpu()
