#!/usr/bin/env python
"""Train the ATM encoder on the Brennan «Alice in Wonderland» dataset.

The script loads word-level EEG epochs, CLIP text embeddings, and Wav2Vec2 audio
embeddings via `BrennanAliceDataset`, feeds the EEG into an ATM encoder, and
optimises a three-view InfoNCE objective (EEG ↔ text ↔ audio).

Important implementation details
--------------------------------
1.  **Input shape sanity-check** – we automatically infer `(C, T)` from the
    dataset and make sure the chosen `patch_len` divides *T* (else we exit with
    a helpful error message).
2.  **Dataset splits** – with `--use_split` the script reads *_train.npz* and
    *_val.npz* files created by `utils/split_brennan_npz.py`; otherwise it
    uses the full *_words.npz*.
3.  **Loss** – `MultiViewInfoNCELoss` averages CLIP-style InfoNCE over the
    three modality pairs (EEG–Text, EEG–Audio, Text–Audio).
4.  **Validation** – optional per-epoch validation pass logged as `val_loss`.
5.  **Logging** – metrics are sent to Weights & Biases via `WandbLogger`.

Run:
    python train_atm_brennan.py \
        --npz_dir "D:/Universität/Master/Masterarbeit/python/data/preprocessed/Brennan" \
        --audio_dir "D:/Universität/Master/Masterarbeit/python/data/audio" \
        --subjects S01 S03 S04 \
        --epochs 10 --batch_size 128
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from util import WandbLogger
from eegdataset import BrennanAliceDataset
from models.atm import ATM
from losses.multiview_infonce import MultiViewInfoNCELoss

# -----------------------------------------------------------------------------
# WandB credentials – set once at import time so any subsequent WandbLogger
# initialisation picks them up.
# -----------------------------------------------------------------------------
import os
os.environ["WANDB_API_KEY"] = "fc2d1bc24195ed0dc256cf0d1a94a44630eff1e7"
os.environ["WANDB_MODE"] = "online"

# -----------------------------------------------------------------------------
# Helper – build ATM with a patch length that divides T
# -----------------------------------------------------------------------------

def choose_patch_len(T: int, candidate: int | None = None) -> int:
    """Return a patch length that exactly divides *T*.

    If *candidate* is given, assert divisibility. Otherwise choose the greatest
    common divisor between *T* and 25, 20, 10, 5 (in that order)."""
    if candidate is not None:
        if T % candidate != 0:
            raise ValueError(f"Time-dimension {T} is not divisible by patch_len={candidate}.")
        return candidate

    for p in (25, 20, 10, 5, 1):
        if T % p == 0:
            return p
    raise RuntimeError("Could not find suitable patch_len (this should never happen).")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Train ATM on Brennan EEG data")
    # Data
    p.add_argument("--npz_dir", type=Path, default="D:/Universität/Master/Masterarbeit/python/data/preprocessed/Brennan")
    p.add_argument("--audio_dir", type=Path, default="D:/Universität/Master/Masterarbeit/python/data/audio")
    p.add_argument("--subjects", nargs="+", default=["S01", "S03", "S04"], help="Subject IDs like S01 S03 …")
    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patch_len", type=int, default=None, help="Force patch length (must divide T)")
    p.add_argument("--use_split", default=True, action="store_true", help="Use _train/_val/_test split files")
    # WandB
    p.add_argument("--wandb_project", default="Masterthesis")
    p.add_argument("--wandb_entity", default="t-reiter-technical-university-of-munich")
    p.add_argument("--wandb_name", default="ATM_run")
    # Debug
    p.add_argument("--check_dims", action="store_true", help="Check embedding dimensions and exit")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    suffix = "_train.npz" if args.use_split else "_words.npz"
    val_suffix = "_val.npz" if args.use_split else "_words.npz"

    print(f"Loading data from: {args.npz_dir}")
    print(f"Using subjects: {args.subjects}")
    print(f"Loading audio from: {args.audio_dir}")
    
    train_ds = BrennanAliceDataset(
        npz_dir=str(args.npz_dir),
        audio_dir=str(args.audio_dir),
        subjects=args.subjects,
        device=device,
        npz_suffix=suffix,
    )

    if args.use_split:
        val_ds = BrennanAliceDataset(
            npz_dir=str(args.npz_dir),
            audio_dir=str(args.audio_dir),
            subjects=args.subjects,
            device=device,
            npz_suffix=val_suffix,
        )
    else:
        val_ds = None

    # Infer C, T
    C, T = train_ds.eeg_data.shape[1:]
    print(f"Detected EEG shape per sample: channels={C}, timepoints={T}")

    # Choose/validate patch_len
    patch_len = choose_patch_len(T, args.patch_len)
    print(f"Using patch_len={patch_len} (T/patch_len = {T//patch_len} patches)")

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None
    
    # ------------------------------------------------------------------
    # Check embedding dimensions if requested
    # ------------------------------------------------------------------
    if args.check_dims:
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        
        # Get text and audio embeddings
        txt_emb = sample_batch["text_features"]
        aud_emb = sample_batch["audio_features"]
        
        # Create ATM model
        model = ATM(
            patch_len=patch_len,
            in_chans=C,
            embed_dim=512,
            depth=6,
            num_heads=8,
            proj_dim=512,  # Default projection dimension
        ).to(device)
        
        # Get EEG embedding
        eeg = sample_batch["eeg"].to(device).float()
        eeg_emb = model(eeg)
        
        # Print dimensions
        print("\n----- EMBEDDING DIMENSIONS CHECK -----")
        print(f"EEG embedding shape: {eeg_emb.shape}")
        print(f"Text embedding shape: {txt_emb.shape}")
        print(f"Audio embedding shape: {aud_emb.shape}")
        print(f"EEG embedding dim: {eeg_emb.shape[1]}")
        print(f"Text embedding dim: {txt_emb.shape[1]}")
        print(f"Audio embedding dim: {aud_emb.shape[1]}")
        print("----- END DIMENSIONS CHECK -----\n")
        
        if eeg_emb.shape[1] != txt_emb.shape[1] or eeg_emb.shape[1] != aud_emb.shape[1]:
            print("WARNING: Embedding dimensions don't match!")
            print(f"To fix: Set proj_dim={txt_emb.shape[1]} in the ATM model")
            return
        else:
            print("All embedding dimensions match. Good to go!")
            return

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    model = ATM(
        patch_len=patch_len,
        in_chans=C,
        embed_dim=512,
        depth=6,
        num_heads=8,
        proj_dim=768,
    ).to(device)

    optimiser = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = MultiViewInfoNCELoss()
    
    # Check embedding dimensions
    with torch.no_grad():
        # Get a small batch of data to check dimensions
        sample_batch = next(iter(train_loader))
        sample_eeg = sample_batch["eeg"].to(device).float()
        sample_txt = sample_batch["text_features"].to(device).float()
        sample_aud = sample_batch["audio_features"].to(device).float()
        
        # Forward pass to get EEG embeddings
        sample_eeg_emb = model(sample_eeg)
        
        # Print dimensions for debugging
        print(f"EEG embedding dimension: {sample_eeg_emb.shape}")
        print(f"Text embedding dimension: {sample_txt.shape}")
        print(f"Audio embedding dimension: {sample_aud.shape}")
        
        # Verify dimensions match
        if sample_eeg_emb.shape[1] != sample_txt.shape[1] or sample_eeg_emb.shape[1] != sample_aud.shape[1]:
            print("WARNING: Embedding dimensions don't match!")
            print(f"  EEG: {sample_eeg_emb.shape[1]}")
            print(f"  Text: {sample_txt.shape[1]}")
            print(f"  Audio: {sample_aud.shape[1]}")
            
            # Adjust dimensions if needed
            if sample_txt.shape[1] == sample_aud.shape[1]:
                print(f"Adjusting ATM proj_dim to {sample_txt.shape[1]}")
                # Recreate model with matching dimension
                model = ATM(
                    patch_len=patch_len,
                    in_chans=C,
                    embed_dim=512,
                    depth=6,
                    num_heads=8,
                    proj_dim=sample_txt.shape[1],
                ).to(device)
                optimiser = optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # WandB logger
    # ------------------------------------------------------------------
    import json, pathlib
    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "wandb_config.json"
    cfg = json.loads(cfg_path.read_text())
    logger = WandbLogger(cfg)
    logger.watch_model(model, log="all")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            eeg = batch["eeg"].to(device).float()        # B × C × T
            txt = batch["text_features"].to(device).float()   # B × D
            aud = batch["audio_features"].to(device).float()  # B × D

            optimiser.zero_grad()

            eeg_emb = model(eeg)                          # B × D
            logit_scale = model.logit_scale.exp()

            # Normalise (should already be, but just in case)
            eeg_emb = F.normalize(eeg_emb, dim=-1)
            txt = F.normalize(txt, dim=-1)
            aud = F.normalize(aud, dim=-1)
            
            # Debug: Print shapes to diagnose the issue
            print(f"Shapes: EEG={eeg_emb.shape}, Text={txt.shape}, Audio={aud.shape}")

            loss = loss_fn([eeg_emb, txt, aud], model.logit_scale)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")

        log_dict = {"train_loss": avg_loss, "epoch": epoch}

        if val_loader is not None:
            model.eval()
            val_loss_acc = 0.0
            with torch.no_grad():
                for vb in val_loader:
                    eeg = vb["eeg"].to(device).float()
                    txt = vb["text_features"].to(device).float()
                    aud = vb["audio_features"].to(device).float()
                    eeg_emb = model(eeg)
                    loss_val = loss_fn([F.normalize(eeg_emb,dim=-1), F.normalize(txt,dim=-1), F.normalize(aud,dim=-1)], model.logit_scale)
                    val_loss_acc += loss_val.item()
            val_loss_acc /= len(val_loader)
            print(f"  validation loss={val_loss_acc:.4f}")
            log_dict["val_loss"] = val_loss_acc

        logger.log(log_dict)

    logger.finish()


if __name__ == "__main__":
    main() 