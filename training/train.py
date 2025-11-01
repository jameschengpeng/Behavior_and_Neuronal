"""Training functions for behavior prediction models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BehaviorPredictor
from .losses import asymmetric_loss_with_mask
from .utils import compute_pos_weight_and_prior, init_classifier_bias_from_prior


def train_model(train_loader,
                num_labels,
                num_epochs: int = 10,
                lr: float = 3e-4,
                weight_decay: float = 1e-4,
                save_path: str = "model.pth",
                use_asl: bool = True,
                asl_gamma_neg: float = 4.0,
                asl_gamma_pos: float = 0.0,
                asl_clip: float = 0.05,
                use_bias_prior_init: bool = True,
                cap_pos_weight: float = 10.0,
                stats_loader = None,
                use_text_encoder: bool = False,
                text_mode: str = "precomputed",
                text_in_dim: int | None = None,
                text_model_name: str | None = None,
                use_stim_window: bool = False):
    """
    Train BehaviorPredictor with masked loss and optional Asymmetric Loss.
    
    Args:
        train_loader: Training DataLoader
        num_labels: Number of behavior classes
        num_epochs: Number of training epochs
        lr: Learning rate for head (backbone uses lr * 0.1)
        weight_decay: L2 regularization weight
        save_path: Path to save trained model
        use_asl: Whether to use Asymmetric Loss (vs BCE)
        asl_gamma_neg: ASL negative focusing parameter
        asl_gamma_pos: ASL positive focusing parameter
        asl_clip: ASL negative clipping value
        use_bias_prior_init: Initialize classifier bias with class priors
        cap_pos_weight: Maximum pos_weight value
        stats_loader: Optional separate loader for computing statistics
        use_text_encoder: Whether model uses text encoder
        text_mode: Text encoder mode ("precomputed" or "sentence_transformer")
        text_in_dim: Text input dimension (for precomputed mode)
        text_model_name: Sentence transformer model name
        use_stim_window: Whether to use stimulus window conditioning
        
    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorPredictor(
        num_labels=num_labels,
        use_text_encoder=use_text_encoder,
        text_mode=text_mode,
        text_in_dim=text_in_dim,
        text_model_name=text_model_name,
        use_stim_window=use_stim_window
    ).to(device)

    # --- Compute class stats from train data (masked to valid timesteps) ---
    which = stats_loader if stats_loader is not None else train_loader
    pos_weight, prior = compute_pos_weight_and_prior(which, num_labels, device, cap=cap_pos_weight)
    print("[STATS] prior (unbiased):", prior.detach().cpu().numpy().round(6))
    print("[STATS] pos_weight (unbiased):", pos_weight.detach().cpu().numpy().round(3))

    # --- Initialize classifier bias with logit(prior) to avoid saturated zeros/ones ---
    if use_bias_prior_init and prior is not None:
        init_classifier_bias_from_prior(model, prior.clamp(1e-6, 1 - 1e-6).to(device))

    # BCE fallback if ASL is off
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction='none')

    # --- Build parameter groups: low LR for backbone, higher for head ---
    def is_backbone_name(n: str) -> bool:
        n = n.lower()
        return any(k in n for k in ["efficientnet", "effnet", "backbone", "cnn", "feature_extractor"])

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if is_backbone_name(n) else head_params).append(p)

    lr_head = lr           # e.g., 1e-4 or 3e-4
    lr_backbone = lr * 0.1 # e.g., 1e-5 if lr_head=1e-4

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ],
        weight_decay=weight_decay
    )

    # Optional: freeze backbone for first few epochs
    warmup_freeze_epochs = 2
    def set_backbone_requires_grad(flag: bool):
        for n, p in model.named_parameters():
            if is_backbone_name(n):
                p.requires_grad = flag

    clip_norm = 5.0

    for epoch in range(num_epochs):
        # Freeze backbone initially, unfreeze after warmup_freeze_epochs
        set_backbone_requires_grad(epoch >= warmup_freeze_epochs)
        
        # Keep BatchNorm in eval mode for backbone
        for name, m in model.named_modules():
            if any(k in name.lower() for k in ["efficientnet", "effnet", "backbone", "cnn", "feature_extractor"]):
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        
        model.train()
        running = 0.0
        denom = 0.0

        for batch in train_loader:
            # Backward compatible unpack: 7-tuple (legacy) or 8-tuple (with sentences)
            if len(batch) == 8:
                xb, yb, lengths, stim_type, stim_loc, stim_onset, stim_offset, sent_text = batch
            else:
                xb, yb, lengths, stim_type, stim_loc, stim_onset, stim_offset = batch
                sent_text = None

            xb = xb.to(device)                         # (B, C, T_max, H, W)
            yb = yb.to(device).float()                 # (B, T_max, C)
            lengths = lengths.to(device).long()
            stim_type = stim_type.to(device)
            stim_loc = stim_loc.to(device)
            stim_onset = stim_onset.to(device)
            stim_offset = stim_offset.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Pass sentences when available; model handles None
            logits = model(
                xb, stim_type, stim_loc, stim_onset, stim_offset, lengths,
                sent_emb=None, sent_text=sent_text
            )  # (B, T_max, C)

            # --- Mask padded timesteps ---
            B, T, C = logits.shape
            t_idx = torch.arange(T, device=lengths.device).unsqueeze(0)          # (1, T)
            valid_mask_t = (t_idx < lengths.unsqueeze(1))                        # (B, T) bool
            mask_bt1 = valid_mask_t.unsqueeze(-1).float()                        # (B, T, 1)
            mask_btc = mask_bt1.expand(-1, -1, C)                                # (B, T, C)

            # --- Framewise loss ---
            if use_asl:
                frame_loss = asymmetric_loss_with_mask(
                    logits, yb, mask_bt1,
                    gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip,
                    onset_tau=6
                )
            else:
                loss_raw = criterion_bce(logits, yb)                              # (B, T, C)
                valid_elems = mask_btc.sum().clamp_min(1.0)
                frame_loss = (loss_raw * mask_btc).sum() / valid_elems

            # ---------------- Presence auxiliary loss (timing-agnostic) ----------------
            # Encourage "at least one frame" to fire when a class is present in the clip.
            p = torch.sigmoid(logits)
            # GT presence per (clip, class), respecting mask
            y_presence = (((yb > 0.5).float() * mask_btc).sum(dim=1) > 0).float()  # (B, C)

            # Presence probability via 1 - ∏_t (1 - p_{t}) over VALID frames (stable with logs)
            eps = 1e-6
            p_clamped = p.clamp(min=eps, max=1.0 - eps)
            log_no = torch.log1p(-p_clamped) * mask_btc                           # (B, T, C)
            log_no_sum = log_no.sum(dim=1)                                        # (B, C)
            presence_prob = 1.0 - torch.exp(log_no_sum)                           # (B, C)

            presence_loss = F.binary_cross_entropy(presence_prob, y_presence)

            # ---------------- Temporal smoothness (optional) ----------------
            if T > 1:
                dp = (p[:, 1:, :] - p[:, :-1, :]).abs() * mask_btc[:, 1:, :]      # (B, T-1, C)
                smooth_denom = mask_btc[:, 1:, :].sum().clamp_min(1.0)
                smooth_loss = dp.sum() / smooth_denom
            else:
                smooth_loss = torch.zeros((), device=device, dtype=frame_loss.dtype)

            # ---------------- Combine losses ----------------
            lambda_presence = 0.10
            lambda_smooth   = 0.01
            warm = min(1.0, (epoch + 1) / 3.0)   # linear warmup over first 3 epochs

            loss = frame_loss + lambda_presence * warm * presence_loss + lambda_smooth * smooth_loss

            # ---------------- Safety + step ----------------
            if not torch.isfinite(loss):
                print("[WARN] non-finite loss; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            running += loss.item() * B
            denom   += B

        print(f"[Epoch {epoch+1}] Train Loss: {running / max(denom,1):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    return model
