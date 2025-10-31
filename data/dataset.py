"""Dataset classes for loading calcium imaging video data."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
