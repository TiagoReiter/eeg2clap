"""Utility wrapper around Weights & Biases for easy experiment tracking.

The class abstracts common actions such as logging scalars/images, saving
checkpoints, and re-loading models so that the rest of the codebase can stay
framework-agnostic.  All paths are derived from the keys provided in the
``config`` dictionary.
"""

from __future__ import annotations

import os
from pathlib import Path
import torch
import wandb


class WandbLogger:
    """Thin convenience wrapper around *wandb* API."""

    def __init__(self, config: dict):
        """Start a WandB run and store the configuration locally."""
        self.config = config
        self.step: int | None = None

        wandb.init(
            project=config["project"],
            name=config.get("name"),
            config=config,
            entity=config.get("entity"),
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def log(self, data: dict, step: int | None = None):
        """Log a dictionary of scalars to WandB.

        Args:
            data: Dictionary *key → value* where value is a scalar or list.
            step: Optional global step. If omitted, WandB will auto-increment.
        """
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)
            self.step = step

    def log_image(self, figs: dict):
        """Log image or matplotlib figure dictionaries.

        Args:
            figs: Dictionary understood by ``wandb.Image`` helpers.
        """
        if self.step is None:
            wandb.log(figs)
        else:
            wandb.log(figs, step=self.step)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    @staticmethod
    def watch_model(model: torch.nn.Module, log: str = "gradients", **kwargs):
        """Tell WandB to watch the model (gradients/parameters)."""
        wandb.watch(model, log=log, **kwargs)

    # Backwards-compat alias
    watch = watch_model

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _ckpt_dir(self) -> Path:
        base = Path(self.config["path_data"]) / self.config["path_ckpt"]
        base.mkdir(parents=True, exist_ok=True)
        return base

    def save(self, net: torch.nn.Module, file_name: str | None = None):
        """Save model weights to the checkpoint directory."""
        if file_name is None:
            file_name = self.config.get("file_ckpt", "checkpoint.pt")
        path = self._ckpt_dir() / file_name
        torch.save(net.state_dict(), path)
        print(f"Checkpoint saved → {path}")

    def load(self, net: torch.nn.Module, file_name: str | None = None):
        """Load model weights from the checkpoint directory."""
        if file_name is None:
            file_name = self.config.get("file_ckpt", "checkpoint.pt")
        path = self._ckpt_dir() / file_name
        net.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded checkpoint ← {path}")

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------
    @staticmethod
    def finish():
        """End the current WandB run (quietly)."""
        wandb.finish(quiet=True)
        