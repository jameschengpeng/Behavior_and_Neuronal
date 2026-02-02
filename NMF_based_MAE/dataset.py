"""Dataset utilities for NMF-based temporal MAE."""

from __future__ import annotations

from typing import Iterable, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5NMFTemporalDataset(Dataset):
    """
    Dataset that loads full temporal sequences from an H5 file produced by NMF_result_to_h5.m.

            Each item returns:
                - c: Tensor of shape (T, k)
                - s: Tensor of shape (T, s) or None
                - meta: dict with group name
    """

    def __init__(
        self,
        h5_path: str,
        patch_length: int = 20,
    include_s: bool = True,
        video_groups: Iterable[str] | None = None,
    ) -> None:
        self.h5_path = h5_path
        self.patch_length = int(patch_length)
        self.include_s = include_s

        with h5py.File(self.h5_path, "r") as f:
            if video_groups is None:
                self.video_groups = sorted([
                    k for k in f.keys()
                    if k.startswith("video_") and isinstance(f[k], h5py.Group)
                ])
            else:
                self.video_groups = list(video_groups)

            if not self.video_groups:
                raise ValueError("No video groups found in H5 file.")

            sample_group = self.video_groups[0]
            c_shape = f[f"{sample_group}/C"].shape  # (k, T)
            self.k = int(c_shape[0])
            self.s = 0
            if self.include_s and f.get(f"{sample_group}/S") is not None:
                self.s = int(f[f"{sample_group}/S"].shape[0])

            self._lengths = {}
            for grp in self.video_groups:
                c_shape = f[f"{grp}/C"].shape
                self._lengths[grp] = int(c_shape[1])

        if not self.video_groups:
            raise ValueError("No video groups available in dataset.")

    @property
    def feature_dim(self) -> int:
        return self.k

    @property
    def max_seq_len(self) -> int:
        return max(self._lengths.values()) if self._lengths else 0

    def __len__(self) -> int:
        return len(self.video_groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | None, dict]:
        group = self.video_groups[idx]
        with h5py.File(self.h5_path, "r") as f:
            c = f[f"{group}/C"][:]  # (k, T)
            c = np.asarray(c, dtype=np.float32).T  # (T, k)

            if self.include_s and f.get(f"{group}/S") is not None:
                s = f[f"{group}/S"][:]
                s = np.asarray(s, dtype=np.float32).T  # (T, s)
            else:
                s = None

        meta = {
            "group": group,
            "length": c.shape[0],
        }
        c_tensor = torch.from_numpy(c)
        s_tensor = torch.from_numpy(s) if s is not None else None
        return c_tensor, s_tensor, meta
