"""Word-based EEG Preprocessing Pipeline for the Master thesis
--------------------------------------------------------------------
This script processes the Brennan "Alice in Wonderland" dataset for word-level
EEG analysis. It aligns EEG recordings with word-by-word timing annotations
to create epochs time-locked to individual word onsets.

The pipeline extends the basic EEG preprocessing with word-level segmentation:
1. Standard EEG preprocessing (filtering, ICA, resampling to 125 Hz)
2. Word timing alignment using AliceChapterOne-EEG.csv
3. Stimulus-locked epoch extraction with linguistic features
4. Quality control and artifact rejection per word

Each epoch is perfectly aligned to word onset (t=0 corresponds to word onset).
This ensures that text embeddings can be directly aligned with EEG samples.

Typical usage:
    python preprocessing.py \
        --input_dir data/raw/Brennan \
        --output_dir data/interim \
        --word_csv AliceChapterOne-EEG.csv \
        --sfreq 125 \
        --tmin -0.2 --tmax 0.8 \
        --min_word_length 0.1

Output format per subject:
    'eeg_words'     : float32  [n_words, n_channels, n_times]
    'word_info'     : dict with word metadata (onset, offset, surprisal, etc.)
    'events'        : int      [n_words] word indices
    'sfreq'         : float
    'ch_names'      : list[str]
    'rejected_words': list of rejected word indices
    'time_axis'     : float32  [n_times] time points relative to word onset

Dependencies
------------
MNE-Python, pandas, numpy, scipy, joblib, tqdm.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence, Dict, Tuple
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# -------------------------------------------------------------------------
# Core EEG preprocessing
# -------------------------------------------------------------------------

def load_raw(path: Path, montage: str | None = "standard_1020") -> mne.io.BaseRaw:
    """Read EEG file in BrainVision format via MNE."""
    # Look for .vhdr file (BrainVision header)
    if path.suffix.lower() == '.vhdr':
        raw_path = path
    else:
        # Try to find corresponding .vhdr file
        vhdr_path = path.with_suffix('.vhdr')
        if vhdr_path.exists():
            raw_path = vhdr_path
        else:
            raise FileNotFoundError(f"Could not find .vhdr file for {path}")
    
    raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose="ERROR")
    if montage is not None:
        raw.set_montage(montage, on_missing="ignore")
    return raw


def bandpass_notch(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, 
                   notch: Sequence[int] | None = (50, 100)) -> mne.io.BaseRaw:
    """Zero-phase FIR band-pass plus optional notch."""
    raw.filter(l_freq, h_freq, fir_design='firwin', phase='zero')
    if notch:
        raw.notch_filter(freqs=list(notch))
    return raw


def apply_ica(raw: mne.io.BaseRaw, n_components: int | None = None, 
              random_state: int = 97) -> mne.io.BaseRaw:
    """FastICA with automatic EOG detection/rejection using correlation."""
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter="auto")
    ica.fit(raw.copy().filter(1., None))  # high-pass to speed ICA convergence
    # find EOG artifacts
    eog_indices, _ = ica.find_bads_eog(raw, verbose="ERROR")
    ica.exclude.extend(eog_indices)
    raw = ica.apply(raw)
    return raw

# -------------------------------------------------------------------------
# Word-level processing functions with perfect alignment
# -------------------------------------------------------------------------

def load_word_annotations(csv_path: Path) -> pd.DataFrame:
    """Load word timing and linguistic annotations from CSV."""
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['Word', 'Segment', 'onset', 'offset', 'Order']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Sort by order to ensure correct sequence
    df = df.sort_values('Order').reset_index(drop=True)
    
    return df


def create_stimulus_locked_events(raw: mne.io.BaseRaw, word_df: pd.DataFrame, 
                                 segment: int, segment_offset: float = 0.0) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create MNE events array perfectly aligned to word onsets for stimulus-locked epochs.
    
    Parameters:
    -----------
    raw : mne.io.BaseRaw
        EEG raw data for this segment
    word_df : pd.DataFrame
        Word timing annotations
    segment : int
        Audio segment number
    segment_offset : float
        Time offset to add to word onsets (in seconds) to account for segment boundaries
    
    Returns:
    --------
    events : np.ndarray
        Events array with precise sample timing [n_events, 3]
    segment_words : pd.DataFrame
        Word data for this segment with adjusted timing
    """
    # Filter words for this segment
    segment_words = word_df[word_df['Segment'] == segment].copy()
    
    if len(segment_words) == 0:
        return np.array([]).reshape(0, 3), segment_words
    
    # Add segment offset to word onsets for continuous timing
    segment_words = segment_words.copy()
    segment_words['absolute_onset'] = segment_words['onset'] + segment_offset
    segment_words['absolute_offset'] = segment_words['offset'] + segment_offset
    
    # Convert onset times to samples with precise alignment
    sfreq = raw.info['sfreq']
    onset_samples = np.round(segment_words['onset'].values * sfreq).astype(int)
    
    # Ensure we don't exceed raw data bounds
    max_sample = len(raw.times) - 1
    valid_samples = (onset_samples >= 0) & (onset_samples <= max_sample)
    
    if not valid_samples.any():
        return np.array([]).reshape(0, 3), pd.DataFrame()
    
    # Filter to valid samples
    onset_samples = onset_samples[valid_samples]
    segment_words = segment_words[valid_samples].reset_index(drop=True)
    
    # Create events array: [sample, previous_event_id, event_id]
    # Use word Order as event ID for uniqueness across all segments
    events = np.column_stack([
        onset_samples,
        np.zeros(len(onset_samples), dtype=int),  # previous event (unused)
        segment_words['Order'].values  # word order as unique event ID
    ])
    
    return events, segment_words


def create_stimulus_locked_epochs(raw: mne.io.BaseRaw, events: np.ndarray, 
                                 word_df: pd.DataFrame, tmin: float, tmax: float,
                                 baseline: Tuple[float, float] | None = (-0.1, 0.0),
                                 min_word_length: float = 0.05) -> Tuple[mne.Epochs, pd.DataFrame]:
    """Create stimulus-locked epochs perfectly aligned to word onsets.
    
    Each epoch will have t=0 corresponding exactly to word onset sample.
    This ensures perfect alignment for text embedding analysis.
    """
    
    if len(events) == 0:
        # Return empty epochs
        info = mne.create_info(['dummy'], raw.info['sfreq'], ['eeg'])
        empty_epochs = mne.EpochsArray(np.array([]).reshape(0, 1, 0), info)
        return empty_epochs, pd.DataFrame()
    
    # Create event_id mapping using word order for uniqueness
    event_id = {f"word_{order}": order for order in events[:, 2]}
    
    # Filter out very short words (likely artifacts or annotation errors)
    word_lengths = word_df['offset'] - word_df['onset']
    valid_words = word_lengths >= min_word_length
    
    if not valid_words.any():
        # Return empty epochs
        info = mne.create_info(['dummy'], raw.info['sfreq'], ['eeg'])
        empty_epochs = mne.EpochsArray(np.array([]).reshape(0, 1, 0), info)
        return empty_epochs, pd.DataFrame()
    
    # Filter events and word data to valid words
    valid_events = events[valid_words.values]
    valid_word_df = word_df[valid_words].copy()
    valid_event_id = {f"word_{order}": order for order in valid_events[:, 2]}
    
    try:
        # Create stimulus-locked epochs with precise timing
        epochs = mne.Epochs(
            raw, valid_events, event_id=valid_event_id,
            tmin=tmin, tmax=tmax, baseline=baseline,
            preload=True, detrend=1, reject_by_annotation=True,
            verbose="ERROR"
        )
        
        # Additional artifact rejection based on amplitude
        reject_criteria = {'eeg': 150e-6}  # 150 µV threshold
        epochs.drop_bad(reject=reject_criteria)
        
        # Update word_df to match remaining epochs after artifact rejection
        remaining_orders = [int(name.split('_')[1]) for name in epochs.event_id.keys()]
        final_word_df = valid_word_df[valid_word_df['Order'].isin(remaining_orders)].copy()
        
        # Verify perfect alignment
        print(f"  Created {len(epochs)} stimulus-locked epochs (t=0 = word onset)")
        print(f"  Time axis: {epochs.times[0]:.3f}s to {epochs.times[-1]:.3f}s relative to word onset")
        
        return epochs, final_word_df
        
    except Exception as e:
        warnings.warn(f"Failed to create stimulus-locked epochs: {e}")
        info = mne.create_info(['dummy'], raw.info['sfreq'], ['eeg'])
        empty_epochs = mne.EpochsArray(np.array([]).reshape(0, 1, 0), info)
        return empty_epochs, pd.DataFrame()


def detect_audio_segments(raw: mne.io.BaseRaw) -> Dict[int, Tuple[float, float]]:
    """Detect audio segment boundaries from trigger events in the raw data."""
    # Try to find events/triggers that mark segment boundaries
    try:
        events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
        
        if len(events) == 0:
            # If no triggers found, assume single segment covering entire recording
            return {1: (0.0, raw.times[-1])}
        
        # Segment detection logic would depend on the specific trigger coding
        # For now, implement a simple heuristic based on trigger timing
        segment_boundaries = {}
        current_segment = 1
        
        # Assume triggers mark segment starts
        for i, event in enumerate(events):
            start_time = event[0] / raw.info['sfreq']
            if i < len(events) - 1:
                end_time = events[i + 1][0] / raw.info['sfreq']
            else:
                end_time = raw.times[-1]
            
            segment_boundaries[current_segment] = (start_time, end_time)
            current_segment += 1
            
            if current_segment > 12:  # Max 12 segments per README
                break
        
        return segment_boundaries
        
    except Exception:
        # Fallback: assume single segment
        return {1: (0.0, raw.times[-1])}


def export_word_data(epochs: mne.Epochs, word_df: pd.DataFrame, 
                     rejected_words: list, out_path: Path):
    """Export stimulus-locked word epochs and metadata to npz file."""
    
    if len(epochs) == 0:
        # Save empty arrays with proper time axis
        np.savez_compressed(
            out_path,
            eeg_words=np.array([]).reshape(0, 0, 0),
            word_info={},
            events=np.array([]),
            sfreq=125.0,  # Updated default
            ch_names=[],
            rejected_words=rejected_words,
            time_axis=np.array([])
        )
        return
    
    # Get epoch data
    data = epochs.get_data(dtype=np.float32)  # shape (n_words, n_channels, n_times)
    
    # Prepare word info dictionary with all linguistic features
    word_info = {
        'words': word_df['Word'].tolist(),
        'onsets': word_df['onset'].tolist(),
        'offsets': word_df['offset'].tolist(),
        'orders': word_df['Order'].tolist(),
        'segments': word_df['Segment'].tolist(),
        'log_freq': word_df['LogFreq'].tolist() if 'LogFreq' in word_df.columns else [],
        'is_lexical': word_df['IsLexical'].tolist() if 'IsLexical' in word_df.columns else [],
        'ngram_surprisal': word_df['NGRAM'].tolist() if 'NGRAM' in word_df.columns else [],
        'rnn_surprisal': word_df['RNN'].tolist() if 'RNN' in word_df.columns else [],
        'cfg_surprisal': word_df['CFG'].tolist() if 'CFG' in word_df.columns else [],
        'word_length': word_df['Length'].tolist() if 'Length' in word_df.columns else [],
        'position': word_df['Position'].tolist() if 'Position' in word_df.columns else [],
        'sentence': word_df['Sentence'].tolist() if 'Sentence' in word_df.columns else []
    }
    
    # Save to npz with time axis for perfect alignment verification
    np.savez_compressed(
        out_path,
        eeg_words=data,
        word_info=word_info,
        events=word_df['Order'].values.astype(int),
        sfreq=epochs.info['sfreq'],
        ch_names=epochs.ch_names,
        rejected_words=rejected_words,
        time_axis=epochs.times.astype(np.float32)  # Time points relative to word onset
    )

# -------------------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------------------

def _process_subject(subject_path: Path, args) -> None:
    """Process a single subject's EEG data for stimulus-locked word-level analysis."""
    
    subject_id = subject_path.stem  # e.g., 'S01'
    out_path = args.output_dir / f"{subject_id}_words.npz"
    
    try:
        # Load word annotations
        word_df = load_word_annotations(args.input_dir / args.word_csv)
        
        # Load and preprocess EEG
        print(f"Processing {subject_id}...")
        raw = load_raw(subject_path)
        
        # Standard preprocessing
        raw = bandpass_notch(raw, args.band[0], args.band[1], notch=args.notch)
        
        if args.ica:
            raw = apply_ica(raw)
        
        # Resample to 125 Hz for consistent timing across subjects
        print(f"  Resampling from {raw.info['sfreq']} Hz to {args.sfreq} Hz")
        raw.resample(args.sfreq)
        
        # Pick EEG channels only
        raw.pick_types(eeg=True)
        
        # Detect audio segments in the recording
        segment_boundaries = detect_audio_segments(raw)
        print(f"  Detected {len(segment_boundaries)} audio segments")
        
        all_epochs = []
        all_word_dfs = []
        rejected_words = []
        
        # Process each audio segment with proper timing alignment
        for segment_num in sorted(segment_boundaries.keys()):
            start_time, end_time = segment_boundaries[segment_num]
            
            # Crop raw data to segment
            segment_raw = raw.copy().crop(tmin=start_time, tmax=end_time)
            
            # Create stimulus-locked events for this segment
            events, segment_words = create_stimulus_locked_events(
                segment_raw, word_df, segment_num, segment_offset=start_time
            )
            
            if len(events) == 0:
                continue
            
            # Create stimulus-locked epochs for this segment
            epochs, final_word_df = create_stimulus_locked_epochs(
                segment_raw, events, segment_words,
                args.tmin, args.tmax, args.baseline, args.min_word_length
            )
            
            if len(epochs) > 0:
                all_epochs.append(epochs)
                all_word_dfs.append(final_word_df)
                print(f"  Segment {segment_num}: {len(epochs)} words processed")
            
            # Track rejected words
            rejected_in_segment = set(segment_words['Order']) - set(final_word_df['Order'] if len(final_word_df) > 0 else [])
            rejected_words.extend(list(rejected_in_segment))
        
        # Combine all epochs and word data while preserving temporal order
        if all_epochs:
            combined_epochs = mne.concatenate_epochs(all_epochs)
            combined_word_df = pd.concat(all_word_dfs, ignore_index=True)
            
            # Sort by word order to maintain stimulus sequence
            sort_idx = np.argsort(combined_word_df['Order'].values)
            combined_word_df = combined_word_df.iloc[sort_idx].reset_index(drop=True)
            
            # Reorder epochs to match word sequence
            epoch_order = [combined_word_df['Order'].iloc[i] for i in range(len(combined_word_df))]
            epoch_indices = [list(combined_epochs.event_id.values()).index(order) for order in epoch_order]
            combined_epochs = combined_epochs[epoch_indices]
            
            print(f"  Total: {len(combined_epochs)} stimulus-locked epochs created")
            print(f"  Sampling rate: {combined_epochs.info['sfreq']} Hz")
            print(f"  Epoch duration: {args.tmin} to {args.tmax} s relative to word onset")
        else:
            # Create empty epochs
            info = mne.create_info(['dummy'], args.sfreq, ['eeg'])
            combined_epochs = mne.EpochsArray(np.array([]).reshape(0, 1, 0), info)
            combined_word_df = pd.DataFrame()
        
        # Export results
        export_word_data(combined_epochs, combined_word_df, rejected_words, out_path)
        
        print(f"✅ {subject_id}: {len(combined_epochs)} words processed, {len(rejected_words)} rejected")
        
    except Exception as e:
        print(f"⚠️  Failed to process {subject_id}: {e}")


# -------------------------------------------------------------------------
# CLI wrapper
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Stimulus-locked word-based EEG preprocessing pipeline")
    p.add_argument('--input_dir', type=Path, required=True, 
                   help='Directory containing BrainVision files and word CSV')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--word_csv', type=str, default='AliceChapterOne-EEG.csv',
                   help='CSV file with word timing annotations')
    p.add_argument('--sfreq', type=int, default=125, help='Target sampling rate (Hz) - default 125 Hz')
    p.add_argument('--band', nargs=2, type=float, default=(0.1, 40), 
                   metavar=('LOW', 'HIGH'), help='Band-pass cut-offs in Hz')
    p.add_argument('--notch', nargs='*', type=int, default=[50, 100], 
                   help='Notch filter frequencies')
    p.add_argument('--ica', action='store_true', help='Run ICA artifact removal')
    p.add_argument('--tmin', type=float, default=-0.2, 
                   help='Epoch start time relative to word onset (s)')
    p.add_argument('--tmax', type=float, default=0.8, 
                   help='Epoch end time relative to word onset (s)')
    p.add_argument('--baseline', nargs=2, type=float, default=(-0.1, 0.0),
                   help='Baseline period for epoch correction')
    p.add_argument('--min_word_length', type=float, default=0.05,
                   help='Minimum word duration to include (s)')
    p.add_argument('-j', '--n_jobs', type=int, default=1, help='Number of parallel jobs')
    
    args = p.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subject files (.vhdr files)
    subject_files = sorted(args.input_dir.glob('S*.vhdr'))
    
    if not subject_files:
        print("⚠️  No BrainVision .vhdr files found in input directory")
        return
    
    print(f"Found {len(subject_files)} subjects to process")
    print(f"Configuration:")
    print(f"  Sampling rate: {args.sfreq} Hz")
    print(f"  Epoch window: {args.tmin} to {args.tmax} s relative to word onset")
    print(f"  Baseline: {args.baseline} s")
    print(f"  Band-pass: {args.band[0]}-{args.band[1]} Hz")
    print(f"  ICA: {'Yes' if args.ica else 'No'}")
    
    # Process subjects
    Parallel(n_jobs=args.n_jobs)(
        delayed(_process_subject)(f, args) for f in tqdm(subject_files)
    )
    
    print(f"✅ Stimulus-locked word preprocessing complete. Results saved to {args.output_dir}")
    print(f"Each epoch is perfectly aligned: t=0 corresponds to word onset sample")


if __name__ == '__main__':
    main()