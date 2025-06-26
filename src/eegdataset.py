import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from models.text_encoder import get_text_encoder, encode_texts
from models.audio_encoder import get_audio_encoder, encode_audios

class BrennanAliceDataset(Dataset):
    def __init__(
        self,
        npz_dir: str,
        audio_dir: str,
        subjects: list[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        npz_suffix: str = "_words.npz"
    ):
        self.npz_dir = Path(npz_dir)
        self.audio_dir = Path(audio_dir)
        self.subjects = subjects
        self.device = device
        self.suffix = npz_suffix

        self.eeg_data, self.metadata = self._load_all_subject_data()

        self.audio_waveforms, self.audio_sr = self._load_audio()

        self.text_encoder, self.text_tokenizer = get_text_encoder(device=self.device)
        self.audio_processor, self.audio_encoder, self.audio_projection = get_audio_encoder(device=self.device)

        print("Pre-computing text features for all words...")
        self.text_features = self._encode_text()

        print("Pre-computing audio features for all words...")
        self.audio_features = self._encode_audio()

        print("Dataset initialized and all features computed.")

    def _load_all_subject_data(self):
        all_eeg = []
        all_meta_dfs = []
        print(f"Loading data for subjects: {self.subjects}")
        for subj in tqdm(self.subjects):
            npz_path = self.npz_dir / subj / f"{subj}{self.suffix}"
            if not npz_path.exists():
                print(f"Warning: File not found for subject {subj}, skipping.")
                continue
            data = np.load(npz_path, allow_pickle=True)
            all_eeg.append(torch.from_numpy(data['eeg_words']))
            word_info = data['word_info'].item()
            meta_df = pd.DataFrame(word_info)
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
        eeg_tensor = torch.cat(all_eeg, dim=0).float()
        metadata_df = pd.concat(all_meta_dfs, ignore_index=True)
        print(f"Loaded {eeg_tensor.shape[0]} total words from {len(self.subjects)} subjects.")
        return eeg_tensor, metadata_df

    def _load_audio(self):
        waveforms = {}
        sampling_rate = None
        audio_files = sorted(self.audio_dir.glob('*.wav'))
        if len(audio_files) != 12:
            raise ValueError(f"Expected 12 .wav files in {self.audio_dir}, but found {len(audio_files)}.")
        print("Loading audio files...")
        for i, wav_path in enumerate(tqdm(audio_files)):
            import librosa
            wav, sr = librosa.load(wav_path, sr=16000)
            waveforms[i + 1] = wav
            if sampling_rate is None:
                sampling_rate = sr
            elif sampling_rate != sr:
                raise ValueError("All audio files must have the same sampling rate.")
        return waveforms, sampling_rate

    def _encode_text(self):
        word_list = self.metadata['word'].tolist()
        return encode_texts(word_list, self.text_encoder, self.text_tokenizer, device=self.device)

    def _encode_audio(self):
        all_audio_features = []
        batch_size = 64
        num_batches = (len(self.metadata) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches)):
            batch_df = self.metadata.iloc[i*batch_size : (i+1)*batch_size]
            audio_segments = []
            for _, row in batch_df.iterrows():
                segment_id = row['Segment']
                onset_sec = row['onset']
                offset_sec = row['offset']
                full_waveform = self.audio_waveforms[segment_id]
                start_sample = int(onset_sec * self.audio_sr)
                end_sample = int(offset_sec * self.audio_sr)
                word_audio = full_waveform[start_sample:end_sample]
                audio_segments.append(word_audio)
            features = encode_audios(
                audio_segments, 
                self.audio_processor, 
                self.audio_encoder, 
                device=self.device, 
                sampling_rate=self.audio_sr,
                projection=self.audio_projection
            )
            all_audio_features.append(features)
        return torch.cat(all_audio_features, dim=0)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        text_feat = self.text_features[idx]
        audio_feat = self.audio_features[idx]
        word = self.metadata.iloc[idx]['word']
        return {
            "eeg": eeg,
            "text_features": text_feat,
            "audio_features": audio_feat,
            "word": word
        } 