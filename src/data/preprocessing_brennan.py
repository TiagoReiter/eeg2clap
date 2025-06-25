"""Preprocess raw Brennan EEG (BrainVision) files and extract word-locked epochs.

This script takes the *raw* EEG recordings released by Brennan & Hale for the
paper «Hierarchical structure guides rapid linguistic predictions during
naturalistic listening» and turns them into NumPy arrays that contain one EEG
epoch per *word* in the story.  The output closely mirrors what
``data/brennan.py`` produces from the already pre-processed MATLAB files – the
only difference is that we start from the *raw* ``.vhdr/.eeg/.vmrk`` files
instead of the authors' time-locked MATLAB structures.

The pipeline performs the following steps for each participant:
    1.  Load the BrainVision recording using *MNE-Python*.
    2.  Apply basic EEG preprocessing:
        – band-pass filter (default 0.1–40 Hz)
        – average reference
        – optional down-sampling (default 125 Hz)
    3.  Parse the event annotations to determine the onsets of the 12 audio
        segments.
    4.  Read the word-level metadata from «AliceChapterOne-EEG.csv» and convert
        the relative onset (within segment) to an *absolute* time inside the
        recording.
    5.  Cut stimulus-locked epochs (default −0.2 s … +0.8 s) around each word
        onset and export everything (EEG, word table, metadata, time axis …)
        as compressed ``.npz``.

Run ``python preprocessing_brennan.py --help`` for command-line options.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

###############################################################################
# Helper functions
###############################################################################

def _load_raw(
    vhdr_path: Path,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    resample_sfreq: int | None = 125,
) -> mne.io.Raw:
    """Load a BrainVision file and run basic preprocessing."""
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")

    # Filtering -------------------------------------------------------------
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    # Average reference (common choice for EEG) 
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # Down-sample if requested 
    if resample_sfreq is not None and resample_sfreq != raw.info["sfreq"]:
        raw.resample(resample_sfreq, verbose="ERROR")

    # Apply montage if the lab's SFP file is present
    sfp_path = vhdr_path.parent / "easycapM10-acti61_elec.sfp"
    if sfp_path.exists():
        montage = mne.channels.read_custom_montage(str(sfp_path))
        raw.set_montage(montage, on_missing="ignore")
    else:
        # Fall back to a built-in template (channel names match EasyCap M10)
        montage = mne.channels.make_standard_montage("easycap-M10")
        raw.set_montage(montage, on_missing="ignore")

    return raw


def _extract_segment_onsets(raw: mne.io.BaseRaw) -> Dict[int, float]:
    """Return a mapping *segment_id → onset (sec)* extracted from annotations.

    The BrainVision marker file contains triggers that mark the start of the 12
    audio segments.  In the shared data they are encoded as either plain digits
    (``"1" … "12"``) or as ``"S  1"`` / ``"Stimulus/S  1"``.  We try to be
    generous and parse any integer we can find in the annotation description.
    """
    seg_onsets: Dict[int, float] = {}
    pattern = re.compile(r"(\d+)")

    for desc, onset in zip(raw.annotations.description, raw.annotations.onset):
        match = pattern.match(desc.strip())
        if match is None:
            # Try to find digits after last slash, e.g. "Stimulus/S  1"
            match = pattern.search(desc.split("/")[-1].strip())
        if match is None:
            continue
        seg_id = int(match.group(1))
        if 1 <= seg_id <= 12:
            # Keep the *first* occurrence as the segment onset
            seg_onsets.setdefault(seg_id, onset)

    if len(seg_onsets) < 12:
        missing = set(range(1, 13)) - set(seg_onsets)
        raise RuntimeError(
            f"Could not find triggers for segments: {sorted(missing)} in file {raw.filenames[0]}"
        )
    return seg_onsets


def _create_word_events(
    word_csv: Path, seg_onsets: Dict[int, float]
) -> pd.DataFrame:
    """Read word-level CSV and add an absolute onset column."""
    words = pd.read_csv(word_csv)

    if not {"Segment", "onset", "offset"}.issubset(words.columns):
        raise ValueError("CSV must contain columns: Segment, onset, offset")

    # Convert relative onset (within segment) -> absolute onset in recording
    abs_onsets = [seg_onsets[int(seg)] + rel for seg, rel in zip(words.Segment, words.onset)]
    words["start"] = abs_onsets
    words["duration"] = words.offset - words.onset
    words["kind"] = "word"

    # Align column names with brennan.py export helper ----------------------
    words = words.rename(
        columns={
            "Word": "word",
            "Position": "word_id",
            "Sentence": "sequence_id",
        }
    )
    return words


def _epochs_from_words(
    raw: mne.io.BaseRaw,
    words: pd.DataFrame,
    tmin: float,
    tmax: float,
    baseline: Tuple[float, float] | None = (None, 0.0),
) -> mne.Epochs:
    """Cut word-locked MNE epochs."""
    sfreq = raw.info["sfreq"]
    sample_onsets = (words.start.values * sfreq).round().astype(int)

    events = np.column_stack([sample_onsets, np.zeros(len(words), int), words.Order.values])
    event_id = {f"word_{o}": int(o) for o in words.Order}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        detrend=1,
        reject_by_annotation=False,
        verbose=False,
    )
    return epochs


def _export_npz(
    epochs: mne.Epochs,
    words: pd.DataFrame,
    out_path: Path,
):
    """Save data + metadata in a compressed NPZ container."""
    if len(epochs) == 0:
        raise RuntimeError("No epochs to export – something went wrong.")

    np.savez_compressed(
        out_path,
        eeg_words=epochs.get_data().astype(np.float32),
        word_info=words.to_dict("list"),
        events=words.Order.values.astype(int),
        sfreq=epochs.info["sfreq"],
        ch_names=epochs.ch_names,
        time_axis=epochs.times.astype(np.float32),
    )

###############################################################################
# Command-line interface
###############################################################################

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="preprocessing_brennan",
        description="Preprocess raw Brennan EEG data and export word-locked epochs (NPZ)",
    )

    p.add_argument("--input_dir", type=Path, required=True, help="Folder with BrainVision files (S*.vhdr)")
    p.add_argument("--word_csv", type=Path, required=True, help="AliceChapterOne-EEG.csv")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to write the *.npz files")

    p.add_argument("--sfreq", type=int, default=125, help="Target sampling rate after resampling (Hz)")
    p.add_argument("--l_freq", type=float, default=0.1, help="High-pass cutoff for band-pass filter (Hz)")
    p.add_argument("--h_freq", type=float, default=40.0, help="Low-pass cutoff for band-pass filter (Hz)")
    p.add_argument("--tmin", type=float, default=-0.2, help="Epoch start relative to word onset (sec)")
    p.add_argument("--tmax", type=float, default=0.8, help="Epoch end relative to word onset (sec)")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):  # noqa: D401 – simple wrapper
    args = _parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vhdr_files = sorted(args.input_dir.glob("S*.vhdr"))
    if not vhdr_files:
        raise SystemExit(f"No BrainVision *.vhdr files found in {args.input_dir}")

    print(f"Found {len(vhdr_files)} subjects → processing …")

    for vhdr in tqdm(vhdr_files):
        subj = vhdr.stem  # e.g. "S01"
        out_file = args.output_dir / f"{subj}_words.npz"

        try:
            raw = _load_raw(vhdr, l_freq=args.l_freq, h_freq=args.h_freq, resample_sfreq=args.sfreq)
            seg_onsets = _extract_segment_onsets(raw)
            words = _create_word_events(args.word_csv, seg_onsets)
            epochs = _epochs_from_words(raw, words, tmin=args.tmin, tmax=args.tmax)
            _export_npz(epochs, words, out_file)
            tqdm.write(f"✅ {subj}: {len(epochs)} epochs → {out_file.name}")
        except Exception as exc:
            tqdm.write(f"⚠️  {subj}: {exc}")

    print("All done!")


if __name__ == "__main__":
    main()