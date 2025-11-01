"""Behavior prediction model using EfficientNet + RNN."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .encoders import TextEncoder


class BehaviorPredictor(nn.Module):
    """
    EfficientNet-B0 per-frame encoder + temporal GRU/LSTM head.
    Mimics Ajioka et al. (2024): pseudo-RGB frames [t-1, t, t+1] → EfficientNet features → RNN.
    """
    def __init__(self, num_labels, rnn_type="lstm", rnn_hidden_size=128, rnn_layers=2,
                 rnn_dropout=0.2, finetune_cnn=True,
                 stim_type_dim=4, stim_loc_dim=2, stim_embed_dim=8,
                 use_text_encoder=False, text_mode="precomputed",
                 text_in_dim=None, text_model_name=None,
                 global_conditioning=True, use_stim_window=False):
        super().__init__()

        # -------- EfficientNet-B0 trunk --------
        weights = EfficientNet_B0_Weights.DEFAULT
        effb0 = efficientnet_b0(weights=weights)
        self.cnn = nn.Sequential(effb0.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cnn_feat_dim = 1280
        if not finetune_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # -------- Stimulus conditioning --------
        self.use_text_encoder = use_text_encoder
        self.text_mode = text_mode
        self.global_conditioning = global_conditioning
        self.use_stim_window = use_stim_window
        self.stim_fusion_dim = 2 * stim_embed_dim
        if use_text_encoder:
            self.text_encoder = TextEncoder(
                out_dim=self.stim_fusion_dim,
                mode=text_mode,
                text_in_dim=text_in_dim,
                st_model_name=text_model_name,
                l2_normalize=True
            )
        else:
            self.stim_type_embedding = nn.Embedding(stim_type_dim, stim_embed_dim)
            self.stim_loc_embedding  = nn.Embedding(stim_loc_dim,  stim_embed_dim)

        # -------- Temporal RNN head --------
        rnn_in = self.cnn_feat_dim + self.stim_fusion_dim
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(rnn_in, rnn_hidden_size, rnn_layers,
                               batch_first=True, dropout=rnn_dropout, bidirectional=True)
        else:
            self.rnn = nn.GRU(rnn_in, rnn_hidden_size, rnn_layers,
                              batch_first=True, dropout=rnn_dropout, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_hidden_size * 2)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        mean = self.imagenet_mean.to(x.device).type_as(x).view(1, 3, 1, 1, 1)
        std  = self.imagenet_std.to(x.device).type_as(x).view(1, 3, 1, 1, 1)
        return (x - mean) / std

    def forward(self, x, stim_type, stim_loc, stim_onsets, stim_offsets, lengths,
                sent_emb=None, sent_text=None):
        B, C, T, H, W = x.shape
        device = x.device
        x = self._normalize(x)

        # ----- EfficientNet features per frame -----
        frames = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        feats = self.cnn(frames).view(B, T, self.cnn_feat_dim)   # (B, T, 1280)

        # ----- Stimulus embedding -----
        if self.use_text_encoder:
            stim_vec = self.text_encoder(sent_emb=sent_emb if self.text_mode=="precomputed" else None,
                                         sent_text=sent_text if self.text_mode=="sentence_transformer" else None,
                                         device=device)
        else:
            t_emb = self.stim_type_embedding(stim_type)
            l_emb = self.stim_loc_embedding(stim_loc)
            stim_vec = torch.cat([t_emb, l_emb], dim=-1)
        F = stim_vec.size(-1)
        stim_mat = torch.zeros(B, T, F, device=device)
        for i in range(B):
            L = int(lengths[i])
            if self.use_stim_window:
                on, off = int(stim_onsets[i]), int(stim_offsets[i])
                stim_mat[i, max(0,on):min(L,off+1)] = stim_vec[i]
            elif self.global_conditioning:
                stim_mat[i,:L] = stim_vec[i]

        fused = torch.cat([feats, stim_mat], dim=-1)             # (B, T, 1280+F)
        packed = pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        normed = self.layer_norm(rnn_out)
        logits = self.classifier(normed)                         # (B, T, num_labels)
        return logits
