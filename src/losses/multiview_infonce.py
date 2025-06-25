"""Contrastive InfoNCE losses used by the ATM training pipeline.

ClipLoss – symmetric cross-entropy on a pair of modalities (identical to
OpenAI CLIP implementation).
MultiViewInfoNCELoss – extends to *K* modalities by averaging the pairwise
ClipLoss over all combinations.
"""
from __future__ import annotations

from itertools import combinations
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ClipLoss", "MultiViewInfoNCELoss"]


class ClipLoss(nn.Module):
    """Standard CLIP contrastive loss for a pair of embedding batches."""

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute symmetric InfoNCE between batches *x* and *y*.

        Args
        ----
        x, y:  float tensors with shape (B, D). They should already be L2-normalised.
        logit_scale:  a learnable scalar (usually from the model) in log space.
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Embeddings must be 2-D (B×D)")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        logits = torch.matmul(x, y.t()) * logit_scale.exp()
        targets = torch.arange(x.size(0), device=x.device)
        loss_xy = self.cross_entropy(logits, targets)
        loss_yx = self.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_xy + loss_yx)


class MultiViewInfoNCELoss(nn.Module):
    """Average ClipLoss over all pairwise combinations of *K* modalities."""

    def __init__(self):
        super().__init__()
        self.clip_loss = ClipLoss()

    def forward(self, embeds: List[torch.Tensor], logit_scale: torch.Tensor) -> torch.Tensor:
        if len(embeds) < 2:
            raise ValueError("Need at least two modalities for MultiView loss")
        losses = []
        for i, j in combinations(range(len(embeds)), 2):
            losses.append(self.clip_loss(embeds[i], embeds[j], logit_scale))
        return torch.stack(losses).mean() 