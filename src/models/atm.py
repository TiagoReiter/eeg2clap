# src/eeg2clap/models/atm.py
"""
Adaptive Thinking Mapper (ATM)
------------------------------------------------
An EEG-to-CLIP encoder inspired by Li et al. 2024 :contentReference[oaicite:0]{index=0}.
– Patchify raw EEG (shape B × C × T) into N fixed-length chunks
– Linear( C·patch_len → d_model )
– Add spatial-temporal position encodings
– TransformerEncoder (L layers, multi-head self-attn)
– CLS token → MLP projector → 512-d CLIP space
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn

class EEGPatchEmbed(nn.Module):
    def __init__(self, patch_len: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(in_chans * patch_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, T
        B, C, T = x.shape
        assert T % self.patch_len == 0, "T must be divisible by patch_len"
        x = x.unfold(2, self.patch_len, self.patch_len)  # B, C, N, patch
        x = x.contiguous().view(B, C * self.patch_len, -1)  # B, C*patch, N
        x = x.transpose(1, 2)                              # B, N, C*patch
        return self.proj(x)                                # B, N, D

class ATM(nn.Module):
    def __init__(
        self,
        patch_len: int = 10,
        in_chans: int = 64,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        proj_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.patch_embed = EEGPatchEmbed(patch_len, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))  # +1 for CLS
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads,
                                               dim_feedforward=4*embed_dim,
                                               dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights properly"""
        # Initialize CLS token with small random values
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # Initialize position embeddings with small random values  
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        # Initialize projection layer
        torch.nn.init.xavier_uniform_(self.projector[0].weight)
        torch.nn.init.zeros_(self.projector[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float32  B × C × T   (eeg_words from your .npz)
        returns:    B × 512     512-d CLIP-aligned embedding
        """
        tok = self.patch_embed(x)                     # B × N × D
        cls = self.cls_token.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)            # prepend CLS
        pos = self.pos_embed[:, :tok.size(1), :]
        h = self.encoder(tok + pos)
        return self.projector(h[:, 0])      