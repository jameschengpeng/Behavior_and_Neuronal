"""Train a temporal masked autoencoder on NMF C(+S) time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .dataset import H5NMFTemporalDataset
from .model import MaskedAutoencoder1D
from .utils import generate_patch_mask, masked_recon_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal MAE on NMF C(+S).")
    parser.add_argument("--h5-path", type=str, required=True, help="Path to NMF_preprocessed_videos.h5")
    parser.add_argument("--patch-length", type=int, default=20, help="Temporal patch length (frames)")
    parser.add_argument("--include-s", action="store_true", help="Use S as conditioning (not reconstructed)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--mask-span", type=int, default=4, help="Contiguous span length; <=1 for random")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="mae_ckpt.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = H5NMFTemporalDataset(
        h5_path=args.h5_path,
        patch_length=args.patch_length,
        include_s=args.include_s,
    )
    def collate_fn(batch):
        cs, ss, metas = zip(*batch)
        lengths = torch.tensor([c.shape[0] for c in cs], dtype=torch.long)
        c_pad = pad_sequence(cs, batch_first=True)  # (B, T_max, k)

        if any(s is None for s in ss):
            s_pad = None
        else:
            s_pad = pad_sequence(ss, batch_first=True)  # (B, T_max, s)
        return c_pad, s_pad, lengths, metas

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = MaskedAutoencoder1D(
        input_dim=dataset.feature_dim,
        s_dim=dataset.s if args.include_s else 0,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        max_len=max(dataset.max_seq_len, args.patch_length),
        patch_len=args.patch_length,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        denom = 0

        for x, s, lengths, _ in loader:
            x = x.to(device)  # (B, T_max, k)
            if s is not None:
                s = s.to(device)  # (B, T_max, s)
            lengths = lengths.to(device)
            B, T, _ = x.shape

            mask, valid_mask = generate_patch_mask(
                lengths=lengths,
                patch_len=args.patch_length,
                mask_ratio=args.mask_ratio,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            recon = model(x, mask, s=s)
            loss = masked_recon_loss(
                recon,
                x,
                mask,
                valid_mask=valid_mask,
            )

            loss.backward()
            optimizer.step()

            running += loss.item() * B
            denom += B

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {running / max(denom,1):.6f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "feature_dim": dataset.feature_dim,
            "k": dataset.k,
            "s": dataset.s if args.include_s else 0,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
