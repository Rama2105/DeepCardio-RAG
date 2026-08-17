"""
CirCor DigiScope Heart Sound (Phonocardiogram) Dataset Loader
==============================================================
Loads the PhysioNet CirCor dataset for heart murmur detection.

Dataset: https://physionet.org/content/circor-heart-sound/1.0.1/
Contains 5272 heart sound recordings from 1568 subjects at up to 4
auscultation locations (AV, PV, TV, MV, Phc).

File types:
    .wav  — Heart sound audio (4000 Hz sample rate)
    .hea  — WFDB header files
    .tsv  — Heart sound segmentation (S1, systole, S2, diastole)
    .txt  — Subject demographics + murmur labels

Structure expected:
    circor-heart-sound/
    └── training_data/
        ├── training_data.csv       # Combined demographics CSV
        ├── 50001_AV.wav            # Audio files
        ├── 50001_AV.hea
        ├── 50001_AV.tsv            # Segmentation
        ├── 50001.txt               # Subject info
        └── ...
"""

import os
import csv
import wave
import struct
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "circor-heart-sound"
)
TRAINING_SUBDIR = "training_data"
SAMPLE_RATE = 4000        # Hz
SEGMENT_DURATION = 5.0    # seconds for fixed-length input
N_MELS = 64              # Mel spectrogram bands
HOP_LENGTH = 128          # Spectrogram hop
MURMUR_CLASSES = ["Absent", "Present", "Unknown"]
AUSCULTATION_LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------
class HeartSoundDataset:
    """
    Loads CirCor heart sound recordings with demographics and murmur labels.
    """

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.training_dir = os.path.join(dataset_dir, TRAINING_SUBDIR)
        self.subjects: Dict[str, Dict] = {}   # subject_id → demographics
        self.recordings: List[Dict] = []       # all wav recording entries
        self._load_metadata()

    def _load_metadata(self):
        """Parse training_data.csv for subject demographics and murmur labels."""
        csv_path = os.path.join(self.training_dir, "training_data.csv")
        if not os.path.isfile(csv_path):
            print(f"[HeartSound] training_data.csv not found at {csv_path}. Demo mode.")
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                sid = row.get("Patient ID", "")
                self.subjects[sid] = {
                    "patient_id": sid,
                    "age": row.get("Age", ""),
                    "sex": row.get("Sex", ""),
                    "height": _safe_float(row.get("Height")),
                    "weight": _safe_float(row.get("Weight")),
                    "pregnancy": row.get("Pregnancy status", "False"),
                    "murmur": row.get("Murmur", "Unknown"),
                    "outcome": row.get("Outcome", ""),
                    "locations": row.get("Recording locations:", "").split("+"),
                }

        # Index wav files.
        #
        # Subject id is the part BEFORE the first underscore. Splitting on the
        # LAST underscore (rsplit) breaks the repeat-recording names CirCor uses
        # — '50204_MV_1.wav' would yield patient_id '50204_MV', which (a) misses
        # the metadata row so the murmur label silently falls back to 'Unknown',
        # and (b) makes the subject-level split treat '50204_MV' and '50204' as
        # two different people, letting one patient appear in both train and
        # test. That is patient leakage, so parse with partition().
        skipped_no_metadata = []
        if os.path.isdir(self.training_dir):
            for fname in sorted(os.listdir(self.training_dir)):
                if not fname.endswith(".wav"):
                    continue
                base = fname[:-4]
                sid, _, loc = base.partition("_")
                if not sid:
                    continue
                if sid not in self.subjects:
                    # No metadata row: label is genuinely unknown to us. Record it
                    # rather than defaulting to the 'Unknown' murmur CLASS, which
                    # is a real annotation meaning "auscultation inconclusive".
                    skipped_no_metadata.append(fname)
                    continue
                subj = self.subjects[sid]
                self.recordings.append({
                    "filename": fname,
                    "filepath": os.path.join(self.training_dir, fname),
                    "patient_id": sid,
                    "location": loc,
                    "murmur": subj.get("murmur", "Unknown"),
                    "age": subj.get("age", ""),
                    "sex": subj.get("sex", ""),
                })

        n_pat = len({r["patient_id"] for r in self.recordings})
        print(f"[HeartSound] Loaded {len(self.subjects)} subjects, "
              f"{len(self.recordings)} recordings ({n_pat} with audio).")
        if skipped_no_metadata:
            print(f"[HeartSound] Skipped {len(skipped_no_metadata)} recordings with no "
                  f"metadata row (e.g. {skipped_no_metadata[0]}) — excluded rather than "
                  f"labelled 'Unknown'.")

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------
    def load_audio(self, index: int) -> Tuple[np.ndarray, Dict]:
        """Load raw audio waveform and metadata for a recording."""
        rec = self.recordings[index]
        signal = self._read_wav(rec["filepath"])
        return signal, rec

    def load_audio_tensor(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """Load audio as a fixed-length tensor for model input."""
        signal, rec = self.load_audio(index)
        # Pad or truncate to SEGMENT_DURATION
        target_len = int(SEGMENT_DURATION * SAMPLE_RATE)
        if len(signal) >= target_len:
            signal = signal[:target_len]
        else:
            signal = np.pad(signal, (0, target_len - len(signal)))
        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, T)
        return tensor, rec

    @staticmethod
    def _read_wav(filepath: str) -> np.ndarray:
        """Read a .wav file into a numpy float32 array."""
        with wave.open(filepath, "rb") as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        # Decode from the bytes that are ACTUALLY present, never from the frame
        # count the header claims. Some CirCor .wav headers overstate their
        # length; sizing the unpack from getnframes() then raises
        # "struct.error: unpack requires a buffer of N bytes" and kills the
        # DataLoader worker mid-epoch. np.frombuffer is also far faster than
        # struct.unpack across 3,163 recordings.
        frame_bytes = max(1, sample_width * n_channels)
        usable = len(raw) - (len(raw) % frame_bytes)
        declared = n_frames * frame_bytes
        if usable < declared * 0.99:
            print(f"[HeartSound] {os.path.basename(filepath)}: header declares "
                  f"{declared} bytes, file holds {usable} — decoding what is present.")
        raw = raw[:usable]

        if sample_width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

        if n_channels > 1:
            samples = samples[::n_channels]  # take first channel
        return samples

    def load_segmentation(self, index: int) -> List[Dict]:
        """Load heart sound segmentation (.tsv) for a recording."""
        rec = self.recordings[index]
        tsv_path = rec["filepath"].replace(".wav", ".tsv")
        segments = []
        if os.path.isfile(tsv_path):
            with open(tsv_path, newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 3:
                        seg_map = {0: "Unannotated", 1: "S1", 2: "Systole",
                                   3: "S2", 4: "Diastole"}
                        segments.append({
                            "start": float(row[0]),
                            "end": float(row[1]),
                            "label": seg_map.get(int(row[2]), "Unknown"),
                        })
        return segments

    # ------------------------------------------------------------------
    # Mel spectrogram (simple numpy implementation, no librosa needed)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_mel_spectrogram(signal: np.ndarray, sr: int = SAMPLE_RATE,
                                 n_mels: int = N_MELS,
                                 hop_length: int = HOP_LENGTH) -> np.ndarray:
        """Compute a simple log-mel spectrogram using numpy FFT."""
        n_fft = 512
        # Pad signal
        pad_len = n_fft - (len(signal) % hop_length)
        signal = np.pad(signal, (0, pad_len))

        # STFT
        n_frames = 1 + (len(signal) - n_fft) // hop_length
        stft = np.zeros((n_fft // 2 + 1, n_frames))
        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * hop_length
            frame = signal[start:start + n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame))
            stft[:, i] = spectrum

        # Mel filterbank
        fmin, fmax = 0, sr / 2
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(n_mels):
            for k in range(bin_points[m], bin_points[m + 1]):
                filterbank[m, k] = (k - bin_points[m]) / max(1, bin_points[m + 1] - bin_points[m])
            for k in range(bin_points[m + 1], bin_points[m + 2]):
                filterbank[m, k] = (bin_points[m + 2] - k) / max(1, bin_points[m + 2] - bin_points[m + 1])

        mel_spec = filterbank @ (stft ** 2)
        log_mel = np.log(mel_spec + 1e-9)
        return log_mel

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def get_eda_summary(self) -> Dict[str, Any]:
        if not self.recordings:
            return self._demo_eda()

        murmur_dist = {}
        location_dist = {}
        sex_dist = {}
        age_dist = {}

        for rec in self.recordings:
            m = rec.get("murmur", "Unknown")
            murmur_dist[m] = murmur_dist.get(m, 0) + 1
            loc = rec.get("location", "")
            location_dist[loc] = location_dist.get(loc, 0) + 1

        for subj in self.subjects.values():
            sex = subj.get("sex", "Unknown")
            sex_dist[sex] = sex_dist.get(sex, 0) + 1
            age = subj.get("age", "")
            age_dist[age] = age_dist.get(age, 0) + 1

        return {
            "dataset": "CirCor DigiScope Phonocardiogram (PhysioNet)",
            "total_subjects": len(self.subjects),
            "total_recordings": len(self.recordings),
            "sample_rate": f"{SAMPLE_RATE} Hz",
            "murmur_distribution": murmur_dist,
            "location_distribution": location_dist,
            "sex_distribution": sex_dist,
            "age_group_distribution": age_dist,
            "auscultation_locations": AUSCULTATION_LOCATIONS,
        }

    @staticmethod
    def _demo_eda() -> Dict[str, Any]:
        return {
            "dataset": "CirCor DigiScope Phonocardiogram (PhysioNet) — DEMO MODE",
            "total_subjects": 1568,
            "total_recordings": 5272,
            "sample_rate": f"{SAMPLE_RATE} Hz",
            "murmur_distribution": {"Absent": 2575, "Present": 1727, "Unknown": 970},
            "location_distribution": {"AV": 1568, "PV": 1399, "TV": 1214, "MV": 1044, "Phc": 47},
            "sex_distribution": {"Male": 908, "Female": 660},
            "age_group_distribution": {"Neonate": 227, "Infant": 326, "Child": 741, "Adolescent": 210, "Young Adult": 64},
            "auscultation_locations": AUSCULTATION_LOCATIONS,
            "note": "Reference stats. Download from PhysioNet for real analysis.",
        }

    def __len__(self):
        return len(self.recordings)


# ---------------------------------------------------------------------------
# Heart Sound Classifier (1D-CNN on mel spectrograms)
# ---------------------------------------------------------------------------
class HeartSoundClassifier(nn.Module):
    """
    1D-CNN operating on mel-spectrogram features for murmur detection.
    Input: (B, 1, n_mels, T) log-mel spectrogram
    Output: 3-class (Absent, Present, Unknown)
    """

    def __init__(self, n_mels: int = N_MELS, num_classes: int = 3, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (3, 7), padding=(1, 3)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(16, 32, (3, 5), padding=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim), nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x).flatten(1)
        emb = self.embedding(feat)
        logits = self.classifier(emb)
        return {"logits": logits, "embeddings": emb}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[Dict]:
        self.eval()
        out = self.forward(x)
        probs = torch.softmax(out["logits"], dim=1)
        pred_idx = probs.argmax(dim=1)
        results = []
        for i in range(x.size(0)):
            idx = pred_idx[i].item()
            results.append({
                "murmur_class": MURMUR_CLASSES[idx],
                "confidence": round(probs[i, idx].item(), 4),
                "probabilities": {
                    MURMUR_CLASSES[j]: round(probs[i, j].item(), 4)
                    for j in range(len(MURMUR_CLASSES))
                },
            })
        return results


# ---------------------------------------------------------------------------
# Clinical guidelines for heart sound findings
# ---------------------------------------------------------------------------
HEART_SOUND_GUIDELINES = {
    "Present": [
        "Murmur detected. Refer for echocardiography to evaluate valvular pathology.",
        "Guideline (AHA/ACC): All newly detected murmurs warrant transthoracic echocardiogram.",
        "Assess murmur characteristics: timing, location, radiation, intensity (Levine grade).",
    ],
    "Absent": [
        "No murmur detected. Normal heart sounds.",
        "S1 and S2 present with normal splitting. No additional heart sounds.",
    ],
    "Unknown": [
        "Heart sound quality insufficient for definitive classification.",
        "Recommend repeat auscultation in quiet environment or digital stethoscope recording.",
    ],
}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_hs_dataset: Optional[HeartSoundDataset] = None
_hs_classifier: Optional[HeartSoundClassifier] = None

def get_heart_sound_dataset() -> HeartSoundDataset:
    global _hs_dataset
    if _hs_dataset is None:
        _hs_dataset = HeartSoundDataset()
    return _hs_dataset

def get_heart_sound_classifier() -> HeartSoundClassifier:
    global _hs_classifier
    if _hs_classifier is None:
        _hs_classifier = HeartSoundClassifier()
        _hs_classifier.eval()
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "data", "heart_sound_classifier.pt")
        if os.path.isfile(weights_path):
            _hs_classifier.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print("[HeartSound] Loaded pretrained classifier.")
        else:
            print("[HeartSound] No pretrained weights. Using demo mode.")
    return _hs_classifier


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
