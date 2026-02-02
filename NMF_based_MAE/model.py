"""Temporal masked autoencoder model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedAutoencoder1D(nn.Module):
    """Simple temporal MAE with a Transformer encoder and linear decoder."""

    def __init__(
        self,
        input_dim: int,
        s_dim: int = 0,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        patch_len: int = 20,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.s_dim = int(s_dim)
        self.patch_len = max(1, int(patch_len))
        max_patches = (max_len + self.patch_len - 1) // self.patch_len
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.s_proj = None
        if self.s_dim > 0:
            self.s_proj = nn.Linear(self.s_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, input_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        if self.s_proj is not None:
            nn.init.xavier_uniform_(self.s_proj.weight)
            nn.init.zeros_(self.s_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) boolean, True indicates masked timestep
        Returns:
            recon: (B, T, D)
        """
        B, T, _ = x.shape
        max_len = self.patch_pos_embed.shape[1] * self.patch_len
        if T > max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {max_len}")

        h = self.input_proj(x)
        if mask is not None:
            mask_tok = self.mask_token.expand(B, T, -1)
            h = torch.where(mask.unsqueeze(-1), mask_tok, h)

        if self.s_proj is not None and s is not None:
            h = h + self.s_proj(s)

        patch_ids = torch.arange(T, device=x.device) // self.patch_len
        patch_emb = self.patch_pos_embed[:, patch_ids, :]
        h = h + patch_emb
        h = self.encoder(h)
        recon = self.decoder(h)
        return recon
