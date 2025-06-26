# src/eeg2clap/models/atm.py
"""
Adaptive Thinking Mapper (ATM)
--------------------------------
A lightweight EEG-to-CLIP encoder:
1. Patchify raw EEG (B × C × T) into N fixed-length chunks
2. Linear projection → d_model
3. Add learnable positional embeddings
4. Transformer encoder (L layers)
5. CLS token → MLP projector → 512-d CLIP space

The file now contains **only** model code. All training utilities, dataset
handling, logging, and CLI entry-points have been moved to dedicated training
scripts (e.g. `train_atm_brennan.py`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: F401 (kept for potential future use)

from losses.multiview_infonce import ClipLoss

__all__ = ["EEGPatchEmbed", "ATM"]


class EEGPatchEmbed(nn.Module):
    """Convert continuous EEG into a sequence of patch embeddings."""

    def __init__(self, patch_len: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(in_chans * patch_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B × C × T
        B, C, T = x.shape
        if T % self.patch_len != 0:
            raise ValueError("T must be divisible by patch_len (got T=%d, patch_len=%d)" % (T, self.patch_len))
        # Split into non-overlapping patches along the time axis
        x = x.unfold(2, self.patch_len, self.patch_len)  # B, C, N, patch
        x = x.contiguous().view(B, C * self.patch_len, -1)  # B, C*patch, N
        x = x.transpose(1, 2)  # B, N, C*patch
        return self.proj(x)     # B, N, D


class ATM(nn.Module):
    """Transformer encoder that maps EEG patches to CLIP-aligned embeddings."""

    def __init__(
        self,
        patch_len: int = 10,
        in_chans: int = 64,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        proj_dim: int = 768,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.patch_embed = EEGPatchEmbed(patch_len, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))  # +1 for CLS

        enc_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, depth)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim),
        )

        # CLIP-style logit scale (used by separate training scripts)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

        self._init_weights()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.projector[0].weight)
        nn.init.zeros_(self.projector[0].bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B × C × T
        tok = self.patch_embed(x)  # B × N × D
        cls = self.cls_token.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)  # prepend CLS
        pos = self.pos_embed[:, : tok.size(1), :]
        h = self.encoder(tok + pos)
        return self.projector(h[:, 0])  # CLS token output      