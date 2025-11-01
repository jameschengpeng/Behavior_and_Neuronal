"""Text encoder for stimulus metadata."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """
    Text encoder with two modes:
      - mode='precomputed': project incoming (B, text_in_dim) tensor
      - mode='sentence_transformer': encode list[str] with a frozen ST model, then project
    """
    def __init__(self,
                 out_dim: int,
                 mode: str = "precomputed",          # 'precomputed' | 'sentence_transformer'
                 text_in_dim: int | None = None,     # needed for 'precomputed'
                 st_model_name: str | None = None,   # needed for 'sentence_transformer'
                 l2_normalize: bool = True):
        super().__init__()
        self.mode = mode.lower()
        self.l2_normalize = l2_normalize

        if self.mode == "precomputed":
            assert text_in_dim is not None, "text_in_dim must be set for precomputed mode"
            self.proj = nn.Linear(text_in_dim, out_dim)

        elif self.mode == "sentence_transformer":
            assert st_model_name is not None, "st_model_name is required for sentence_transformer mode"
            self.st = SentenceTransformer(st_model_name)
            for p in self.st.parameters():
                p.requires_grad = False
            # probe to get dimensionality
            probe = self.st.encode(["probe"], convert_to_numpy=True)
            text_in_dim = int(probe.shape[-1])
            self.proj = nn.Linear(text_in_dim, out_dim)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self,
                sent_emb: torch.Tensor | None = None,   # used in 'precomputed'
                sent_text: list[str] | None = None,     # used in 'sentence_transformer'
                device: torch.device | None = None) -> torch.Tensor:
        if self.mode == "precomputed":
            assert sent_emb is not None, "Provide sent_emb for precomputed mode"
            x = sent_emb
        else:
            assert sent_text is not None, "Provide sent_text (list[str]) for sentence_transformer mode"
            with torch.no_grad():
                enc = self.st.encode(sent_text, convert_to_numpy=True, normalize_embeddings=self.l2_normalize)
            x = torch.from_numpy(enc).to(device if device is not None else torch.device("cpu")).float()

        if self.l2_normalize:
            x = nn.functional.normalize(x, p=2, dim=-1)
        return self.proj(x)
