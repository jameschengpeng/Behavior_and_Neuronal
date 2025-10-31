"""Batch collation and sampling utilities for video data."""

import random
import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence


def make_bucket_batch_sampler_for_subset(subset, base_dataset_lengths, batch_size, bucket_size=50, shuffle=True, seed=42):
    """
    Build a bucketed batch sampler that works with a torch.utils.data.Subset.
    
    Args:
        subset: torch.utils.data.Subset
        base_dataset_lengths: list[int] aligned with the base dataset (dataset.lengths)
        batch_size: Size of each batch
        bucket_size: Size of buckets for grouping similar-length sequences
        shuffle: Whether to shuffle bucket order
        seed: Random seed for reproducibility
        
    Returns:
        A batch sampler that yields indices in range [0..len(subset)-1]
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


def build_weighted_sampler(dataset):
    """
    Build a weighted sampler that weights samples by number of positive labels.
    Useful for handling class imbalance.
    
    Args:
        dataset: Dataset with __getitem__ returning (X, y, ...)
        
    Returns:
        WeightedRandomSampler
        
    Warning:
        This will iterate through the entire dataset once to compute weights,
        which may be slow for large datasets.
    """
    import numpy as np
    weights = []
    for i in range(len(dataset)):
        # WARNING: __getitem__ will read H5 for y; okay for 30–50 videos
        _, y, *_ = dataset[i]               # (T, C)
        w = float(y.sum().item())
        weights.append(max(w, 1.0))         # at least 1 to avoid zero-prob
    weights = torch.tensor(weights, dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True
    )
    return sampler


def pad_collate_fn(
    batch,
    training: bool = False,
    max_T: int | None = None,
    positive_bias: bool = True,
    pos_margin: int = 32,         # keep some room after a positive
    tail_bias_prob: float = 0.7,  # when no positives, bias crop toward the tail
):
    """
    Collate function with padding and optional positive-aware temporal cropping.
    
    Args:
        batch: List of tuples from dataset
          - legacy: (X, y, stim_type, stim_loc, stim_onset, stim_offset)
          - new:    (X, y, stim_type, stim_loc, stim_onset, stim_offset, stim_sentence)
            where X: (C, T, H, W), y: (T, C_labels)
        training: If True and max_T is set, apply temporal cropping
        max_T: Maximum temporal length (crop longer sequences during training)
        positive_bias: If True, bias cropping to include positive frames
        pos_margin: Frames to keep after last positive frame
        tail_bias_prob: Probability of biasing toward tail when no positives
        
    Returns:
        x_padded: (B, C, T_max, H, W)
        y_padded: (B, T_max, C_labels)
        lengths:  (B,)
        stim_type, stim_loc, stim_onset, stim_offset: (B,)
        sent_text: list[str] length B (if sentences present in batch)
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
