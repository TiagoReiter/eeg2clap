import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import open_clip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm

class BrennanAliceDataset(Dataset):
    """
    A PyTorch Dataset for the Brennan & Hale 'Alice in Wonderland' EEG dataset.

    This class loads preprocessed EEG epochs from .npz files, extracts the
    corresponding word-level text and audio, and pre-computes feature
    embeddings for all three modalities.
    """
    def __init__(
        self,
        npz_dir: str,
        audio_dir: str,
        subjects: list[str],
        clip_model_name: str = 'ViT-B-32',
        clip_pretrained: str = 'laion2b_s32b_b79k',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            npz_dir (str): Path to the directory containing the processed .npz files.
            audio_dir (str): Path to the directory with the 12 .wav audio files.
            subjects (list[str]): A list of subject identifiers (e.g., ['S01', 'S03'])
                to include in the dataset.
            clip_model_name (str): The CLIP model architecture to use.
            clip_pretrained (str): The pretrained CLIP weights to use.
            device (str): The device to run model inference on ('cpu' or 'cuda').
        """
        self.npz_dir = Path(npz_dir)
        self.audio_dir = Path(audio_dir)
        self.subjects = subjects
        self.device = device

        # 1. Load and concatenate data from all subjects
        self.eeg_data, self.metadata = self._load_all_subject_data()

        # 2. Load all 12 audio segments
        self.audio_waveforms, self.audio_sr = self._load_audio()

        # 3. Initialize encoders
        self.text_encoder, self.text_tokenizer = self._init_text_encoder(clip_model_name, clip_pretrained)
        self.audio_processor, self.audio_encoder = self._init_audio_encoder()

        # 4. Pre-compute all features
        print("Pre-computing text features for all words...")
        self.text_features = self._encode_text()

        print("Pre-computing audio features for all words...")
        self.audio_features = self._encode_audio()

        print("Dataset initialized and all features computed.")

    def _load_all_subject_data(self):
        """Loads and combines .npz data for all specified subjects."""
        all_eeg = []
        all_meta_dfs = []

        print(f"Loading data for subjects: {self.subjects}")
        for subj in tqdm(self.subjects):
            npz_path = self.npz_dir / f"{subj}_words.npz"
            if not npz_path.exists():
                print(f"Warning: File not found for subject {subj}, skipping.")
                continue

            data = np.load(npz_path, allow_pickle=True)
            
            # Append EEG data
            all_eeg.append(torch.from_numpy(data['eeg_words']))

            # Create and append metadata DataFrame
            word_info = data['word_info'].item()
            meta_df = pd.DataFrame(word_info)

            # ------------------------------------------------------------------
            # Standardise column names so the rest of the class can assume a
            # fixed schema regardless of how `preprocessing_brennan.py` named
            # the keys.  The NPZ export currently uses plural keys ("words",
            # "onsets", …).  We rename them to the singular form expected by
            # the downstream code ("word", "onset", …).
            # ------------------------------------------------------------------
            rename_map = {
                "words": "word",
                "onsets": "onset",
                "offsets": "offset",
                "segments": "Segment",
                "orders": "Order",
            }
            meta_df = meta_df.rename(columns={k: v for k, v in rename_map.items() if k in meta_df.columns})

            # Ensure correct dtypes for critical columns
            if "Segment" in meta_df.columns:
                meta_df["Segment"] = meta_df["Segment"].astype(int)
            if "onset" in meta_df.columns:
                meta_df["onset"] = meta_df["onset"].astype(float)
            if "offset" in meta_df.columns:
                meta_df["offset"] = meta_df["offset"].astype(float)

            meta_df['subject'] = subj
            all_meta_dfs.append(meta_df)

        if not all_eeg:
            raise ValueError("No data loaded. Check subjects and npz_dir.")

        # Concatenate all data
        eeg_tensor = torch.cat(all_eeg, dim=0).float()
        metadata_df = pd.concat(all_meta_dfs, ignore_index=True)

        print(f"Loaded {eeg_tensor.shape[0]} total words from {len(self.subjects)} subjects.")
        
        return eeg_tensor, metadata_df

    def _load_audio(self):
        """Loads the 12 raw .wav files into a dictionary."""
        waveforms = {}
        sampling_rate = None
        
        audio_files = sorted(self.audio_dir.glob('*.wav'))
        if len(audio_files) != 12:
            raise ValueError(f"Expected 12 .wav files in {self.audio_dir}, but found {len(audio_files)}.")

        print("Loading audio files...")
        for i, wav_path in enumerate(tqdm(audio_files)):
            # librosa loading resamples by default, which is what we want
            wav, sr = librosa.load(wav_path, sr=16000) # Wav2Vec2 requires 16kHz
            waveforms[i + 1] = wav # Segments are 1-indexed
            if sampling_rate is None:
                sampling_rate = sr
            elif sampling_rate != sr:
                raise ValueError("All audio files must have the same sampling rate.")
        
        return waveforms, sampling_rate

    def _init_text_encoder(self, model_name, pretrained):
        """Initializes and returns the CLIP text model and tokenizer."""
        print(f"Initializing CLIP model: {model_name} ({pretrained})")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, tokenizer

    def _init_audio_encoder(self):
        """Initializes and returns the Wav2Vec2 model and processor."""
        print("Initializing Wav2Vec2 model...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        return processor, model

    def _encode_text(self):
        """Encodes all words in the metadata using the text encoder."""
        # Note: In a real scenario, you might batch this for efficiency
        with torch.no_grad():
            word_list = self.metadata['word'].tolist()
            text_tokens = self.text_tokenizer(word_list).to(self.device)
            text_features = self.text_encoder.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu()

    def _encode_audio(self):
        """Extracts audio for each word and encodes it with Wav2Vec2."""
        all_audio_features = []
        
        # Process in batches to manage memory
        batch_size = 64 
        num_batches = (len(self.metadata) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches)):
            batch_df = self.metadata.iloc[i*batch_size : (i+1)*batch_size]
            
            audio_segments = []
            for _, row in batch_df.iterrows():
                segment_id = row['Segment']
                onset_sec = row['onset']
                offset_sec = row['offset']
                
                # Get audio slice
                full_waveform = self.audio_waveforms[segment_id]
                start_sample = int(onset_sec * self.audio_sr)
                end_sample = int(offset_sec * self.audio_sr)
                word_audio = full_waveform[start_sample:end_sample]
                audio_segments.append(word_audio)
            
            # Process batch with the audio processor
            inputs = self.audio_processor(
                audio_segments, 
                sampling_rate=self.audio_sr, 
                return_tensors="pt", 
                padding=True
            )

            with torch.no_grad():
                # Move inputs to the correct device
                input_values = inputs.input_values.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                
                # Get hidden states and mean-pool
                hidden_states = self.audio_encoder(input_values, attention_mask=attention_mask).last_hidden_state
                # Mean pool across the time dimension
                pooled_features = hidden_states.mean(dim=1)

            all_audio_features.append(pooled_features.cpu())
        
        return torch.cat(all_audio_features, dim=0)

    def __len__(self):
        """Returns the total number of words in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns the data for a single word at the given index.
        """
        eeg = self.eeg_data[idx]
        text_feat = self.text_features[idx]
        audio_feat = self.audio_features[idx]
        
        # Also return original word for context
        word = self.metadata.iloc[idx]['word']
        
        return {
            "eeg": eeg,
            "text_features": text_feat,
            "audio_features": audio_feat,
            "word": word
        } 