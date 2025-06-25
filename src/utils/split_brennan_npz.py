#!/usr/bin/env python
"""Split Brennan subject .npz files into train / val / test sets.

Usage
-----
    python split_brennan_npz.py \
        --root_dir "D:/…/data/preprocessed/Brennan" \
        --seed 42

For every subject folder ``S01 … S48`` inside *root_dir* the script expects
exactly one file named ``Sxx_preprocessed_125hz.npz``.  It is split along the
*word* axis into 80 % train, 10 % val, 10 % test, keeping all modalities
aligned.  Resulting files are written next to the original one:

    S01_train.npz
    S01_val.npz
    S01_test.npz
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _slice_npz(src: Path, indices: np.ndarray, dst: Path):
    """Save a sub-selection of the npz (rows along first axis) to *dst*."""
    data = np.load(src, allow_pickle=True)

    sel = indices
    out_kwargs = {}
    for key in data.files:
        arr = data[key]
        if key == "word_info":
            info = arr.item()
            sliced_info = {k: np.asarray(v)[sel].tolist() for k, v in info.items()}
            out_kwargs[key] = sliced_info
        else:
            # Some entries (sfreq, ch_names, etc.) are scalars or 1-D arrays
            # independent of the word dimension.  Only slice if the first
            # dimension matches n_words; otherwise copy as is.
            if arr.ndim >= 1 and arr.shape[0] == len(sel):
                out_kwargs[key] = arr[sel]
            else:
                out_kwargs[key] = arr

    np.savez_compressed(dst, **out_kwargs)


def split_subject_file(file_path: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    with np.load(file_path, allow_pickle=True) as data:
        n_words = data["eeg_words"].shape[0]

    idx = np.arange(n_words)
    rng.shuffle(idx)

    n_train = int(0.8 * n_words)
    n_val = int(0.1 * n_words)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    for split, indices in zip(("train", "val", "test"), (train_idx, val_idx, test_idx)):
        dst = file_path.with_name(file_path.stem.replace("preprocessed_125hz", split) + ".npz")
        _slice_npz(file_path, indices, dst)
        print(f"  → wrote {dst.name}  ({len(indices)} words)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=Path, required=True, help="Root folder with S01 … Sxx subfolders")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for subj_dir in sorted(args.root_dir.glob("S*")):
        if not subj_dir.is_dir():
            continue
        npz_files = list(subj_dir.glob("*_preprocessed_125hz.npz"))
        if not npz_files:
            print(f"[skip] {subj_dir.name}: no *_preprocessed_125hz.npz found")
            continue
        if len(npz_files) > 1:
            print(f"[warn] {subj_dir.name}: multiple candidates → using {npz_files[0].name}")
        src = npz_files[0]
        print(f"Splitting {src.relative_to(args.root_dir.parent)} …")
        split_subject_file(src, args.seed)


if __name__ == "__main__":
    main() 