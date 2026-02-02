"""Smoke test for temporal MAE components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from .dataset import H5NMFTemporalDataset
from .model import MaskedAutoencoder1D
from .utils import generate_patch_mask, masked_recon_loss


def _build_dummy_h5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("/A", data=np.random.rand(12, 4).astype(np.float32))
        grp = f.create_group("/video_01")
        grp.create_dataset("C", data=np.random.rand(4, 60).astype(np.float32))
        grp.create_dataset("S", data=np.random.randint(0, 2, size=(2, 60)).astype(np.float32))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / "dummy.h5"
        _build_dummy_h5(h5_path)

        dataset = H5NMFTemporalDataset(
            h5_path=str(h5_path),
            patch_length=20,
            include_s=True,
        )
        c, s, _ = dataset[0]
        c = c.unsqueeze(0)  # (1, T, k)
        s = s.unsqueeze(0)  # (1, T, s)
        lengths = torch.tensor([c.shape[1]], dtype=torch.long)

        model = MaskedAutoencoder1D(
            input_dim=dataset.feature_dim,
            s_dim=dataset.s,
            max_len=c.shape[1],
            patch_len=20,
        )
        mask, valid_mask = generate_patch_mask(lengths, patch_len=20, mask_ratio=0.4)
        recon = model(c, mask, s=s)
        loss = masked_recon_loss(
            recon,
            c,
            mask,
            valid_mask=valid_mask,
        )

        assert torch.isfinite(loss).item(), "Loss is not finite"
        print("Smoke test passed. Loss:", float(loss))


if __name__ == "__main__":
    main()
