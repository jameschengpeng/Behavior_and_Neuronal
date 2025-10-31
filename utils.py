import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from NN_model import BehaviorPredictor
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import math
import random

class H5VideoDataset(Dataset):
    def __init__(self, h5_path, merge_groups=None, drop_indices=None):
        """
        Args:
            h5_path: path to grouped HDF5
            merge_groups: list[list[int]] – groups of ethogram indices to merge via OR
            drop_indices: list[int] – ethogram indices to drop
        """
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = None

        self.merge_groups = merge_groups or []
        self.drop_indices = set(drop_indices or [])

        # Preload group names
        with h5py.File(h5_path, 'r') as f:
            self.video_keys = sorted(list(f.keys()))  # ['video_01', 'video_02', ...]
        self.lengths = []
        with h5py.File(h5_path, 'r') as f:
            for k in self.video_keys:
                # y is (num_labels, T) in your MATLAB save; T is dim=1
                T = f[k]['y'].shape[1]
                self.lengths.append(int(T))

        # Internal plan built lazily once we see num_labels
        self._plan_built = False
        self._groups = None        # normalized groups (list[list[int]])
        self._singles = None       # remaining indices kept as single labels


    def __len__(self):
        return len(self.video_keys)

    def _build_plan(self, num_labels: int):
        """
        Validate merge/drop indices and construct a plan:
          - self._groups: sanitized, non-overlapping groups with valid indices
          - self._singles: indices kept as-is (not merged, not dropped)
        """
        # Validate indices in range
        def _check_idx_list(name, idxs):
            for i in idxs:
                if i < 0 or i >= num_labels:
                    raise IndexError(f"{name} index {i} out of range [0, {num_labels-1}]")

        # Flatten and validate merge groups
        all_in_groups = set()
        norm_groups = []
        for g in self.merge_groups:
            if not g:
                continue
            _check_idx_list("merge_groups", g)
            g_set = set(g)
            # Disallow overlap across groups
            if any(idx in all_in_groups for idx in g_set):
                raise ValueError(f"merge_groups overlap detected in group {g}")
            all_in_groups.update(g_set)
            norm_groups.append(sorted(g))

        # Validate drops
        _check_idx_list("drop_indices", self.drop_indices)

        # Remove any dropped indices from groups; if a whole group gets dropped, skip it
        cleaned_groups = []
        for g in norm_groups:
            g_kept = [i for i in g if i not in self.drop_indices]
            if len(g_kept) >= 2:
                cleaned_groups.append(g_kept)
            elif len(g_kept) == 1:
                # If only one survives drop, treat it as a singleton (no real merge)
                pass  # it will fall into singles below
            # if none survive, the group disappears

        # Singles = indices not in any (remaining) group and not dropped
        grouped_members = set([i for g in cleaned_groups for i in g])
        singles = [i for i in range(num_labels) if (i not in grouped_members) and (i not in self.drop_indices)]

        self._groups = cleaned_groups
        self._singles = sorted(singles)
        self._plan_built = True

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        group = self.h5_file[self.video_keys[idx]]

        # X saved from MATLAB as (T, H, W, 3) → convert to (C, T, H, W)
        X = group['X'][:]  # (T, H, W, 3)
        y = group['y'][:]  # saved as (T, num_labels)

        # Handle both old/new layouts safely
        if X.ndim == 4:
            if X.shape[-1] == 3:        # (T, H, W, C)
                X = np.transpose(X, (3, 0, 1, 2))  # -> (C, T, H, W)
            elif X.shape[0] == 3:       # legacy (C, H, W, T)
                X = np.transpose(X, (0, 3, 2, 1))  # -> (C, T, H, W)
            else:
                raise ValueError(f"Unexpected X shape {X.shape}: expected (T,H,W,3) or (3,H,W,T)")
        else:
            raise ValueError(f"Unexpected X ndim {X.ndim}: expected 4D array")

        # y: ensure (T, num_labels)
        if y.ndim != 2:
            raise ValueError(f"Unexpected y shape {y.shape}: expected 2D (T, num_labels)")
        if y.shape[0] < y.shape[1]:  # legacy (num_labels, T)
            y = y.T  # -> (T, num_labels)

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)       # (C, T, H, W)
        y = torch.tensor(y, dtype=torch.float32)       # (T, num_labels)

        # X is (C, T, H, W)
        X = X.float()
        if X.max() > 1.0:
            X = X / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=X.dtype, device=X.device).view(3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=X.dtype, device=X.device).view(3, 1, 1, 1)
        X = (X - mean) / std

        stim_type   = group['stim_type'][()]
        stim_loc    = group['stim_location'][()]
        stim_onset  = group['stim_onset'][()]
        stim_offset = group['stim_offset'][()]

        # Read optional sentence
        stim_sentence = ""
        if 'stim_sentence' in group:
            raw = group['stim_sentence'][()]
            if isinstance(raw, bytes):
                stim_sentence = raw.decode('utf-8', errors='ignore')
            else:
                try:
                    # Handles numpy scalar string types
                    stim_sentence = raw.astype(str).item()
                except Exception:
                    stim_sentence = str(raw)

        # Sanitize/cast
        stim_type = int(stim_type)
        # Force stim_loc to binary: anything non-zero -> 1
        stim_loc = 1 if int(stim_loc) != 0 else 0
        stim_onset = int(stim_onset)
        stim_offset = int(stim_offset)

        # Build plan once we know num_labels
        num_labels = y.shape[1]
        if not self._plan_built:
            self._build_plan(num_labels)

        # Ensure binary
        y_bin = (y > 0.5).to(torch.float32)  # (T, num_labels)

        # Merge groups via logical OR across columns
        merged_cols = []
        for g in self._groups:
            # y_bin[:, g] shape: (T, |g|) → OR across dim=1
            merged = (y_bin[:, g] > 0.5).any(dim=1).float().unsqueeze(1)  # (T,1)
            merged_cols.append(merged)

        # Keep singles
        single_cols = [y_bin[:, i:i+1] for i in self._singles]

        # New y: [merged groups in the order provided] + [remaining singles in ascending index]
        if merged_cols or single_cols:
            y_new = torch.cat(merged_cols + single_cols, dim=1)  # (T, new_num_labels)
        else:
            # Edge case: everything was dropped. Return an empty label matrix.
            y_new = torch.zeros((y_bin.shape[0], 0), dtype=torch.float32)

        return X, y_new, stim_type, stim_loc, stim_onset, stim_offset, stim_sentence


# Build a bucketed batch sampler that works with a torch.utils.data.Subset
def make_bucket_batch_sampler_for_subset(subset, base_dataset_lengths, batch_size, bucket_size=50, shuffle=True, seed=42):
    """
    subset: torch.utils.data.Subset
    base_dataset_lengths: list[int] aligned with the base dataset (dataset.lengths)
    Returns a batch sampler that yields indices in range [0..len(subset)-1]
    """

    # subset maps local idx -> global idx via subset.indices
    global_idxs = list(subset.indices)
    local_lengths = [base_dataset_lengths[g] for g in global_idxs]  # lengths aligned to subset order

    class _SubsetBucketBatchSampler(Sampler):
        def __init__(self):
            self.batch_size = batch_size
            self.bucket_size = bucket_size
            self.shuffle = shuffle
            self.seed = seed
            # work in local index space [0..len(subset)-1], sorted by length
            self.local_indices = list(range(len(global_idxs)))
            self.local_indices.sort(key=lambda i: local_lengths[i])

        def __iter__(self):
            rng = random.Random(self.seed)
            # 1) contiguous buckets by length
            buckets = [
                self.local_indices[i:i + self.bucket_size]
                for i in range(0, len(self.local_indices), self.bucket_size)
            ]
            # 2) shuffle bucket order (not the items inside each bucket)
            if self.shuffle:
                rng.shuffle(buckets)
            # 3) yield fixed-size batches from each bucket
            for bucket in buckets:
                for bstart in range(0, len(bucket), self.batch_size):
                    batch = bucket[bstart:bstart + self.batch_size]
                    if batch:
                        yield batch

        def __len__(self):
            import math
            return math.ceil(len(self.local_indices) / self.batch_size)

    return _SubsetBucketBatchSampler()



@torch.no_grad() # don't track gradients for anything happens inside this function
def compute_pos_weight_and_prior(loader, num_labels, device, cap=50.0, eps=1e-8):
    # Compute pos_weight and prior over the dataset, masking out invalid timesteps.
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
    # Clamp away from 0/1 before logit
    prior = prior.clamp(min=eps, max=1 - eps)
    bias = torch.log(prior) - torch.log(1 - prior)  # safe logit
    with torch.no_grad():
        model.classifier.bias.copy_(bias)


def train_model(train_loader,
                num_labels,
                num_epochs: int = 10,
                lr: float = 3e-4,                 # slightly lower LR helps precision
                weight_decay: float = 1e-4,       # mild regularization
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
    Trains BehaviorPredictor with masked loss and optional ASL.
    Will compute pos_weight & prior internally (masked), cap pos_weight to avoid explosions,
    and initialize classifier bias from prior if requested.
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

    # --- build parameter groups: low LR for backbone, higher for head ---
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


@torch.inference_mode()
def _tune_per_class_thresholds(val_loader, model, num_labels, grid=None):
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


@torch.inference_mode()
def test_model(
    test_loader,
    model_path,
    num_labels,
    threshold: float = 0.5,          # framewise threshold (when not using hysteresis or per-class)
    use_hysteresis: bool = False,
    t_low: float = 0.3,
    t_high: float = 0.7,
    min_len: int = 4,
    val_loader=None,                  # optional: provide to enable per-class tuning
    tune_per_class: bool = False,     # set True to tune per-class thresholds on val_loader
    threshold_grid=None,              # optional grid for tuning (torch.linspace(...) etc.)
    # Presence (timing-agnostic) evaluation controls:
    presence_mode: str = "probs",     # "probs" (use probs with presence_threshold) or "decoded" (use framewise preds)
    presence_threshold: float | None = None,  # if None and mode=="probs", will default to `threshold`
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
      - presence_mode="decoded": presence from decoded framewise preds (same thresholds/hysteresis as framewise)
      - presence_mode="probs":   presence from probs via max over time >= presence_threshold
                                 (if presence_threshold is None, it ties to frame `threshold`)
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
    # Per-class presence tallies (for per-class presence recall/support)
    pres_tp_c = torch.zeros(num_labels, dtype=torch.long)
    pres_fn_c = torch.zeros(num_labels, dtype=torch.long)
    pres_sup_c = torch.zeros(num_labels, dtype=torch.long)

    # Exact-match across presence vectors & set IoU
    exact_match_count = 0
    sample_count = 0
    ethogram_iou_sum = 0.0
    ethogram_iou_count = 0

    # Decide effective presence threshold if using probs
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

        # Forward (pass sentences if present)
        logits = model(xb, stim_type, stim_loc, stim_onset, stim_offset, lengths, sent_text=sent_text)  # (B,T,C)
        probs  = sigmoid(logits)                                            # (B,T,C)

        # ----- Framewise decode (for frame metrics) -----
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
        mask_t = (time_idx < lengths.view(B, 1, 1))     # (B,T,1) bool
        mask_f = mask_t.float()

        # ===== Frame-level metrics =====
        total_correct += ((preds == yb).float() * mask_f).sum().item()
        total_count   += (mask_f.sum().item() * C)

        yb_bool    = (yb > 0.5)
        preds_bool = preds.bool()  # already binary

        tp = (preds_bool & yb_bool & mask_t).sum().item()
        fp = (preds_bool & (~yb_bool) & mask_t).sum().item()
        fn = ((~preds_bool) & yb_bool & mask_t).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # ===== Presence (timing-agnostic) =====
        gt_presence = (yb_bool & mask_t).any(dim=1)  # (B,C) bool

        if presence_mode == "decoded":
            # presence from decoded predictions (same thresholds/hysteresis)
            pred_presence = (preds_bool & mask_t).any(dim=1)  # (B,C) bool
            presence_descr = f"decoded ({'hyst' if use_hysteresis else f'th={threshold:.2f}'})"
        else:
            # presence from probabilities via max over time
            neg_inf = torch.finfo(probs.dtype).min
            masked_probs = probs.masked_fill(~mask_t, neg_inf)      # (B,T,C)
            max_over_t   = masked_probs.amax(dim=1)                 # (B,C)
            if presence_threshold_eff is None:
                raise ValueError("presence_threshold is None while presence_mode='probs'")
            pred_presence = (max_over_t >= presence_threshold_eff)  # (B,C) bool
            presence_descr = f"probs≥{presence_threshold_eff:.2f}"

        # Micro presence confusion
        presence_tp += (pred_presence &  gt_presence).sum().item()
        presence_tn += ((~pred_presence) & (~gt_presence)).sum().item()
        presence_fp += (pred_presence & (~gt_presence)).sum().item()
        presence_fn += ((~pred_presence) &  gt_presence).sum().item()

        # Per-class tallies for presence recall/support
        pres_tp_c += (pred_presence & gt_presence).sum(dim=0).cpu()
        pres_fn_c += ((~pred_presence) & gt_presence).sum(dim=0).cpu()
        pres_sup_c += gt_presence.sum(dim=0).cpu()

        # Exact-match of presence vectors
        exact_match = (pred_presence == gt_presence).all(dim=1)     # (B,)
        exact_match_count += int(exact_match.sum().item())
        sample_count       += int(gt_presence.shape[0])

        # Set IoU per clip (only if union non-empty)
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

    # Per-class presence recall/support table
    pres_rec_c = torch.where(
        pres_sup_c > 0,
        pres_tp_c.float() / pres_sup_c.float(),
        torch.zeros_like(pres_sup_c, dtype=torch.float)
    )
    print("[QUAL] Per-class presence recall:", pres_rec_c.numpy().round(3).tolist())
    print("[QUAL] Per-class presence support (GT #clips):", pres_sup_c.tolist())

    return per_class_thresholds


def pad_collate_fn(
    batch,
    training: bool = False,
    max_T: int | None = None,
    positive_bias: bool = True,
    pos_margin: int = 32,         # keep some room after a positive
    tail_bias_prob: float = 0.7,  # when no positives, bias crop toward the tail
):
    """
    batch: list of tuples
      - legacy: (X, y, stim_type, stim_loc, stim_onset, stim_offset)
      - new:    (X, y, stim_type, stim_loc, stim_onset, stim_offset, stim_sentence)
        X: (C, T, H, W)   y: (T, C_labels)

    Returns (legacy):
      x_padded: (B, C, T_max, H, W)
      y_padded: (B, T_max, C_labels)
      lengths:  (B,)
      stim_type, stim_loc, stim_onset, stim_offset: (B,)

    If batch includes sentences, returns the above plus:
      sent_text: list[str] length B
    """
    # Unpack with backward compatibility
    first_len = len(batch[0])
    has_sentence = (first_len == 7)

    if has_sentence:
        xs, ys, stypes, slocs, onsets, offsets, sents = zip(*batch)
    else:
        xs, ys, stypes, slocs, onsets, offsets = zip(*batch)
        sents = None

    # ---- positives-aware cropping (TRAIN ONLY) ----
    if training and (max_T is not None):
        new_xs, new_ys, new_st, new_sl, new_on, new_off = [], [], [], [], [], []
        new_sents = [] if has_sentence else None
        for i, (x, y, st, sl, on, off) in enumerate(zip(xs, ys, stypes, slocs, onsets, offsets)):
            T = x.shape[1]
            if T > max_T:
                # frames containing any positive label
                pos_frames = (y.sum(dim=1) > 0).nonzero(as_tuple=True)[0]  # 1D idx
                if positive_bias and pos_frames.numel() > 0:
                    # aim the window so it includes the last positive, leaving room after it
                    last_pos = int(pos_frames.max().item())
                    start_low  = max(0, last_pos - max_T + pos_margin)
                    start_high = min(last_pos, T - max_T)
                    if start_low > start_high:
                        # fallback: clip near the end while ensuring window fits
                        start_low  = max(0, min(T - max_T, last_pos - max_T // 2))
                        start_high = min(T - max_T, max(0, last_pos))
                    start = random.randint(start_low, start_high)
                else:
                    # no positives in this sample → bias toward tail sometimes
                    if random.random() < tail_bias_prob:
                        start_low  = max(0, T - max_T - pos_margin)
                        start_high = T - max_T
                        start = random.randint(max(0, start_low), max(0, start_high))
                    else:
                        start = random.randint(0, T - max_T)
                end = start + max_T

                # crop
                x = x[:, start:end]       # (C, max_T, H, W)
                y = y[start:end]          # (max_T, C_labels)

                # shift/clamp stim indices into cropped window
                on_new = int(max(0, min(on - start, max_T - 1)))
                off_new = int(max(0, min(off - start, max_T - 1)))
                if off_new < on_new:
                    off_new = on_new

                new_xs.append(x); new_ys.append(y)
                new_st.append(int(st)); new_sl.append(int(sl))
                new_on.append(on_new);  new_off.append(off_new)
                if has_sentence:
                    new_sents.append(sents[i])
            else:
                new_xs.append(x); new_ys.append(y)
                new_st.append(int(st)); new_sl.append(int(sl))
                new_on.append(int(on)); new_off.append(int(off))
                if has_sentence:
                    new_sents.append(sents[i])

        xs, ys, stypes, slocs, onsets, offsets = new_xs, new_ys, new_st, new_sl, new_on, new_off
        if has_sentence:
            sents = new_sents

    # ---- pad within batch ----
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)  # each sample T

    # videos: (C,T,H,W) -> (T,C,H,W) for pad_sequence, then back
    x_seqs = [x.permute(1, 0, 2, 3) for x in xs]                            # (T,C,H,W)
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0.0)    # (B,T_max,C,H,W)
    x_padded = x_padded.permute(0, 2, 1, 3, 4)                              # (B,C,T_max,H,W)

    # labels: already (T,C_labels)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)        # (B,T_max,C_labels)

    stim_types  = torch.tensor(stypes,  dtype=torch.long)
    stim_locs   = torch.tensor(slocs,   dtype=torch.long)
    stim_onsets = torch.tensor(onsets,  dtype=torch.long)
    stim_offsets= torch.tensor(offsets, dtype=torch.long)

    # Return with backward compatibility
    if has_sentence:
        return x_padded, y_padded, lengths, stim_types, stim_locs, stim_onsets, stim_offsets, list(sents)
    else:
        return x_padded, y_padded, lengths, stim_types, stim_locs, stim_onsets, stim_offsets



def embedding_aux_loss(model, mode="type", pairwise=True):
    """
    mode: "type" or "loc"
    pairwise: if True, sum over all unique pairs; if False, only adjacent pairs
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


# Masked, asymmetric focal-style BCE for multi-label sequence data: heavily down-weights easy negatives (gamma_neg), 
# lightly (or not) focuses positives (gamma_pos), 
# clips negatives for stability, and averages over valid timesteps and all classes.
def asymmetric_loss_with_mask(
    logits,
    targets,
    mask,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float | None = 0.05,
    eps: float = 1e-8,
    # NEW: onset-weighting controls (in FRAMES)
    onset_tau: float | None = 5.0,          # e-fold decay in frames; set None to disable weighting
    normalize_onset: bool = True            # keep avg positive weight ~ 1.0 for stability
):
    """
    Multi-label Asymmetric Loss with padding mask + onset-weighted positives.

    Args
    ----
    logits:   (B, T, C) raw scores
    targets:  (B, T, C) binary {0,1}
    mask:     (B, T, 1) float (1=valid, 0=pad)
    gamma_pos, gamma_neg, clip, eps: same as before
    onset_tau: if not None, apply exp(-d/onset_tau) within each positive segment,
               where d = frames since the segment's onset (d=0 at onset).
               Units are *frames* (after any downsampling).
    normalize_onset: if True, normalize positive-frame weights so that their mean is ~1
                     (prevents overall loss scale from shrinking when long segments appear).

    Returns
    -------
    scalar loss
    """
    B, T, C = logits.shape
    device = logits.device
    dtype  = logits.dtype

    # Sigmoid and ASL parts (unchanged)
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

# Temporal smoothing to output labels
def hysteresis_decode(probs, t_low=0.3, t_high=0.7, min_len=4):
    """
    Per-class hysteresis decoding.
    probs: (B, T, C) torch float in [0,1]
    Returns binary (B, T, C) torch float {0,1}
    """
    B, T, C = probs.shape
    out = torch.zeros_like(probs, dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            on = False; start = 0
            for t in range(T):
                p = probs[b, t, c].item()
                if not on and p >= t_high:
                    on = True; start = t
                elif on and p <= t_low:
                    if t - start >= min_len:
                        out[b, start:t, c] = 1.0
                    on = False
            if on and T - start >= min_len:
                out[b, start:T, c] = 1.0
    return out

# utils.py
def build_weighted_sampler(dataset):
    # crude but effective: weight by total #positives in each sample
    import numpy as np
    weights = []
    for i in range(len(dataset)):
        # WARNING: __getitem__ will read H5 for y; okay for 30–50 videos
        _, y, *_ = dataset[i]               # (T, C)
        w = float(y.sum().item())
        weights.append(max(w, 1.0))         # at least 1 to avoid zero-prob
    weights = torch.tensor(weights, dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return sampler





def plot_ethogram_pair(
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    ethogram_labels=None,
    fps: int = 10,
    cmap: str = "viridis",
    titles=("Ground truth", "Prediction"),
    xlabel: str = "Time (in seconds)",
    ylabel: str = "Ethogram label",
    figsize=(10, 6),
    show: bool = True,
    block: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0,
    # colorbar controls
    cbar_location: str = "right",   # "right" | "left" | "bottom" | "top"
    cbar_pad: float = 0.02,
    cbar_fraction: float = 0.05,
    cbar_shrink: float = 0.95,
):
    """
    Plot two ethogram heatmaps stacked vertically (GT on top, Pred below).

    Colorbar is placed outside the panels (default: right). Use cbar_location
    to move it, e.g., cbar_location='bottom' for a horizontal bar.
    """
    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError("Inputs must be torch.Tensor")
        a = t.detach().cpu()
        if a.ndim == 3:
            if a.shape[0] != 1:
                raise ValueError(f"If 3D, first dim must be 1 (got {tuple(a.shape)})")
            a = a[0]
        if a.ndim != 2:
            raise ValueError("Each input must be 2D (T, K) or 3D with first dim = 1")
        return a  # (T, K)

    gt = _to_2d(gt_tensor)
    pr = _to_2d(pred_tensor)
    if gt.shape != pr.shape:
        raise ValueError(f"GT and Prediction must have same shape; got {gt.shape} vs {pr.shape}")

    T, K = gt.shape
    if ethogram_labels is not None:
        if len(ethogram_labels) != K:
            raise ValueError("Length of ethogram_labels must match number of columns (K)")
    else:
        ethogram_labels = [str(i) for i in range(K)]

    gt_plot = gt.T  # (K, T)
    pr_plot = pr.T  # (K, T)

    # Use constrained layout so colorbar sits outside without overlap
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize, constrained_layout=True)

    # --- Top (GT) ---
    im_top = axes[0].imshow(gt_plot, cmap=cmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
    axes[0].set_yticks(np.arange(K))
    axes[0].set_yticklabels(ethogram_labels)
    axes[0].set_ylabel(ylabel)
    if titles and len(titles) > 0 and titles[0]:
        axes[0].set_title(titles[0])
    for y in range(1, K):
        axes[0].axhline(y - 0.5, color="white", linewidth=0.8)

    # --- Bottom (Prediction) ---
    im_bot = axes[1].imshow(pr_plot, cmap=cmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
    
    axes[1].set_yticks(np.arange(K))
    axes[1].set_yticklabels(ethogram_labels)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    if titles and len(titles) > 1 and titles[1]:
        axes[1].set_title(titles[1])
    for y in range(1, K):
        axes[1].axhline(y - 0.5, color="white", linewidth=0.8)

    # x-axis ticks in seconds
    axes[1].set_xticks(np.linspace(0, T - 1, num=6))
    axes[1].set_xticklabels([f"{t / fps:.1f}" for t in np.linspace(0, T - 1, num=6)])

    # One colorbar outside the stack
    # (Matplotlib >=3.3 supports 'location'; if older, it will ignore and still work on the right.)
    cbar = fig.colorbar(
        im_bot, ax=axes, location=cbar_location,
        pad=cbar_pad, fraction=cbar_fraction, shrink=cbar_shrink
    )
    cbar.set_label("Value")

    if show:
        plt.show(block=block)

    return fig, axes
