"""Model evaluation and testing functions."""

import torch
import torch.nn as nn
from models import BehaviorPredictor
from .postprocessing import hysteresis_decode
from .metrics import tune_per_class_thresholds as _tune_per_class_thresholds


@torch.inference_mode()
def test_model(
    test_loader,
    model_path,
    num_labels,
    threshold: float = 0.5,
    use_hysteresis: bool = False,
    t_low: float = 0.3,
    t_high: float = 0.7,
    min_len: int = 4,
    val_loader=None,
    tune_per_class: bool = False,
    threshold_grid=None,
    presence_mode: str = "probs",
    presence_threshold: float | None = None,
    use_text_encoder: bool = False,
    text_mode: str = "precomputed",
    text_in_dim: int | None = None,
    text_model_name: str | None = None,
    use_stim_window: bool = False
):
    """
    Evaluate frame-level metrics AND timing-agnostic presence metrics.
    
    Framewise decoding:
      - If per_class_thresholds are tuned: use those
      - Else if use_hysteresis: use hysteresis_decode
      - Else use global `threshold`

    Presence (timing-agnostic):
      - presence_mode="decoded": presence from decoded framewise preds
      - presence_mode="probs": presence from probs via max over time >= presence_threshold
      
    Args:
        test_loader: Test DataLoader
        model_path: Path to saved model weights
        num_labels: Number of behavior classes
        threshold: Global threshold for framewise decoding
        use_hysteresis: Whether to use hysteresis decoding
        t_low: Lower threshold for hysteresis
        t_high: Upper threshold for hysteresis
        min_len: Minimum segment length for hysteresis
        val_loader: Optional validation loader for threshold tuning
        tune_per_class: Whether to tune per-class thresholds
        threshold_grid: Grid of thresholds to search
        presence_mode: "probs" or "decoded" for presence evaluation
        presence_threshold: Threshold for presence detection (probs mode)
        use_text_encoder: Whether model uses text encoder
        text_mode: Text encoder mode
        text_in_dim: Text input dimension
        text_model_name: Sentence transformer model name
        use_stim_window: Whether to use stimulus window conditioning
        
    Returns:
        per_class_thresholds: Tuned thresholds if tune_per_class=True, else None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BehaviorPredictor(
        num_labels=num_labels,
        use_text_encoder=use_text_encoder,
        text_mode=text_mode,
        text_in_dim=text_in_dim,
        text_model_name=text_model_name,
        use_stim_window=use_stim_window
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sigmoid = nn.Sigmoid()
    per_class_thresholds = None

    # ---- Optional: tune per-class thresholds on validation set ----
    if tune_per_class:
        if val_loader is None:
            print("[WARN] tune_per_class=True but val_loader is None. Skipping tuning.")
        else:
            grid = threshold_grid
            if grid is not None and hasattr(grid, "device") and grid.device != device:
                grid = grid.to(device)
            per_class_thresholds = _tune_per_class_thresholds(val_loader, model, num_labels, grid)
            try:
                print("Per-class thresholds:", per_class_thresholds.cpu().numpy().round(3))
            except Exception:
                print("Per-class thresholds (tensor):", per_class_thresholds)

    # ---- Accumulators ----
    # Frame-level
    total_correct, total_count = 0.0, 0.0
    total_tp = total_fp = total_fn = 0

    # Presence (micro over (clip, class) pairs)
    presence_tp = presence_fp = presence_tn = presence_fn = 0
    # Per-class presence tallies
    pres_tp_c = torch.zeros(num_labels, dtype=torch.long)
    pres_fn_c = torch.zeros(num_labels, dtype=torch.long)
    pres_sup_c = torch.zeros(num_labels, dtype=torch.long)

    # Exact-match & set IoU
    exact_match_count = 0
    sample_count = 0
    ethogram_iou_sum = 0.0
    ethogram_iou_count = 0

    # Decide effective presence threshold
    if presence_mode == "probs" and presence_threshold is None:
        presence_threshold_eff = threshold
    else:
        presence_threshold_eff = presence_threshold

    for step, batch in enumerate(test_loader):
        if len(batch) == 8:
            xb, yb, lengths, stim_type, stim_loc, stim_onset, stim_offset, sent_text = batch
        else:
            xb, yb, lengths, stim_type, stim_loc, stim_onset, stim_offset = batch
            sent_text = None

        xb = xb.to(device)
        yb = yb.to(device).float()
        lengths = lengths.to(device).long()
        stim_type = stim_type.to(device)
        stim_loc = stim_loc.to(device)
        stim_onset = stim_onset.to(device)
        stim_offset = stim_offset.to(device)

        # Forward
        logits = model(xb, stim_type, stim_loc, stim_onset, stim_offset, lengths, sent_text=sent_text)
        probs  = sigmoid(logits)

        # ----- Framewise decode -----
        if per_class_thresholds is not None:
            th = per_class_thresholds.to(device).view(1, 1, -1)
            preds = (probs >= th).float()
        elif use_hysteresis:
            preds = hysteresis_decode(probs, t_low=t_low, t_high=t_high, min_len=min_len)
        else:
            preds = (probs >= threshold).float()

        # ----- Mask valid timesteps -----
        B, T_max, C = yb.shape
        time_idx = torch.arange(T_max, device=device).view(1, T_max, 1)
        mask_t = (time_idx < lengths.view(B, 1, 1))
        mask_f = mask_t.float()

        # ===== Frame-level metrics =====
        total_correct += ((preds == yb).float() * mask_f).sum().item()
        total_count   += (mask_f.sum().item() * C)

        yb_bool    = (yb > 0.5)
        preds_bool = preds.bool()

        tp = (preds_bool & yb_bool & mask_t).sum().item()
        fp = (preds_bool & (~yb_bool) & mask_t).sum().item()
        fn = ((~preds_bool) & yb_bool & mask_t).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # ===== Presence (timing-agnostic) =====
        gt_presence = (yb_bool & mask_t).any(dim=1)

        if presence_mode == "decoded":
            pred_presence = (preds_bool & mask_t).any(dim=1)
            presence_descr = f"decoded ({'hyst' if use_hysteresis else f'th={threshold:.2f}'})"
        else:
            neg_inf = torch.finfo(probs.dtype).min
            masked_probs = probs.masked_fill(~mask_t, neg_inf)
            max_over_t   = masked_probs.amax(dim=1)
            if presence_threshold_eff is None:
                raise ValueError("presence_threshold is None while presence_mode='probs'")
            pred_presence = (max_over_t >= presence_threshold_eff)
            presence_descr = f"probs≥{presence_threshold_eff:.2f}"

        # Micro presence confusion
        presence_tp += (pred_presence &  gt_presence).sum().item()
        presence_tn += ((~pred_presence) & (~gt_presence)).sum().item()
        presence_fp += (pred_presence & (~gt_presence)).sum().item()
        presence_fn += ((~pred_presence) &  gt_presence).sum().item()

        # Per-class tallies
        pres_tp_c += (pred_presence & gt_presence).sum(dim=0).cpu()
        pres_fn_c += ((~pred_presence) & gt_presence).sum(dim=0).cpu()
        pres_sup_c += gt_presence.sum(dim=0).cpu()

        # Exact-match
        exact_match = (pred_presence == gt_presence).all(dim=1)
        exact_match_count += int(exact_match.sum().item())
        sample_count       += int(gt_presence.shape[0])

        # Set IoU
        inter = (pred_presence & gt_presence).sum(dim=1).float()
        union = (pred_presence | gt_presence).sum(dim=1).float()
        valid = union > 0
        if valid.any():
            ethogram_iou_sum   += (inter[valid] / union[valid]).sum().item()
            ethogram_iou_count += int(valid.sum().item())

    # ===== Final frame-level metrics =====
    acc  = total_correct / max(total_count, 1.0)
    prec = total_tp / max(total_tp + total_fp, 1e-8)
    rec  = total_tp / max(total_tp + total_fn, 1e-8)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"\nMasked per-frame, per-label accuracy: {acc:.4f}")
    print(f"Micro Precision: {prec:.4f} | Micro Recall: {rec:.4f} | Micro F1: {f1:.4f}")

    # ===== Presence metrics =====
    total_pairs = presence_tp + presence_tn + presence_fp + presence_fn
    if total_pairs > 0:
        pres_acc = (presence_tp + presence_tn) / total_pairs
        pres_pr  = presence_tp / max(presence_tp + presence_fp, 1e-8)
        pres_rc  = presence_tp / max(presence_tp + presence_fn, 1e-8)
        pres_f1  = 2 * pres_pr * pres_rc / max(pres_pr + pres_rc, 1e-8)
    else:
        pres_acc = pres_pr = pres_rc = pres_f1 = 0.0

    exact_match_acc = (exact_match_count / sample_count) if sample_count > 0 else 0.0
    mean_ethogram_iou = (ethogram_iou_sum / ethogram_iou_count) if ethogram_iou_count > 0 else 0.0

    print(f"[QUAL] Presence({presence_descr}) — "
          f"Acc: {pres_acc:.4f} | Prec: {pres_pr:.4f} | Rec: {pres_rc:.4f} | F1: {pres_f1:.4f}")
    print(f"[QUAL] Ethogram exact-match accuracy: {exact_match_acc:.4f} (over {sample_count} clips)")
    print(f"[QUAL] Ethogram set IoU (avg over clips with non-empty union): {mean_ethogram_iou:.4f} (n={ethogram_iou_count})")

    # Per-class presence recall/support
    pres_rec_c = torch.where(
        pres_sup_c > 0,
        pres_tp_c.float() / pres_sup_c.float(),
        torch.zeros_like(pres_sup_c, dtype=torch.float)
    )
    print("[QUAL] Per-class presence recall:", pres_rec_c.numpy().round(3).tolist())
    print("[QUAL] Per-class presence support (GT #clips):", pres_sup_c.tolist())

    return per_class_thresholds
