"""
Cardiac Arrhythmia Video Dataset Loader
========================================
Extends the EchoNet-Dynamic loader with arrhythmia classification labels.

Compatible datasets:
  1. EchoNet-Dynamic (Stanford AIMI) — echocardiogram videos with EF labels
     https://echonet.github.io/dynamic/
     → Arrhythmia label: derived from EF thresholds or optional Arrhythmia column

  2. Any custom cardiac video dataset with:
       Videos/         — folder of .avi / .mp4 clips
       FileList.csv    — columns: FileName, EF (optional), Arrhythmia (0/1), Split

Arrhythmia classes produced:
    0 — Normal Sinus Rhythm
    1 — Arrhythmia (atrial fibrillation, ventricular tachycardia, flutter, etc.)

Author: DeepCardio-RAG project
"""

import os
import csv
import struct
import threading
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "EchoNet-Dynamic"
)

NUM_FRAMES  = 32    # Uniform temporal sampling
FRAME_SIZE  = 112   # Spatial resize target (112 × 112)
MEAN        = 0.131  # EchoNet grayscale mean
STD         = 0.289  # EchoNet grayscale std

# Arrhythmia-related EF thresholds (used as proxy when no explicit label exists)
EF_ARRHYTHMIA_LOW  = 35.0   # EF < 35 strongly associated with arrhythmia
EF_ARRHYTHMIA_HIGH = 75.0   # EF > 75 associated with hypertrophic conditions

ARRHYTHMIA_CLASSES = {
    0: "Normal Sinus Rhythm",
    1: "Arrhythmia",
}

ARRHYTHMIA_GUIDELINES = {
    "Normal Sinus Rhythm": (
        "Heart rhythm is regular with a rate of 60–100 bpm. "
        "No immediate intervention required. Continue routine monitoring."
    ),
    "Arrhythmia": (
        "Irregular cardiac rhythm detected. Recommend 12-lead ECG, Holter monitoring, "
        "and cardiology consultation. Rule out atrial fibrillation, ventricular tachycardia, "
        "and flutter. Assess stroke risk and consider anticoagulation if AF confirmed."
    ),
}


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------
def _try_import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python-headless is required. "
            "Install: pip install opencv-python-headless"
        )


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": round(float(arr.mean()), 2),
        "std":  round(float(arr.std()),  2),
        "min":  round(float(arr.min()),  2),
        "max":  round(float(arr.max()),  2),
        "count": len(values),
    }


# ---------------------------------------------------------------------------
# Arrhythmia Video Dataset
# ---------------------------------------------------------------------------
class CardiacArrhythmiaVideoDataset:
    """
    Loads cardiac echocardiogram videos with arrhythmia labels.

    Each sample:
        video  — torch.Tensor  (1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
        labels — dict with keys:
                   filename, arrhythmia (0/1), arrhythmia_label (str),
                   ef (float | None), esv (float | None), edv (float | None),
                   split (str), video_path (str)
    """

    def __init__(
        self,
        dataset_dir:  str = DEFAULT_DATASET_DIR,
        split:        Optional[str] = None,    # "TRAIN" | "VAL" | "TEST" | None
        num_frames:   int = NUM_FRAMES,
        frame_size:   int = FRAME_SIZE,
        ef_proxy:     bool = True,             # derive arrhythmia label from EF if column absent
    ):
        self.dataset_dir = dataset_dir
        self.videos_dir  = os.path.join(dataset_dir, "Videos")
        self.num_frames  = num_frames
        self.frame_size  = frame_size
        self.split       = split
        self.ef_proxy    = ef_proxy

        self.samples: List[Dict[str, Any]] = []
        self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
    def _load_metadata(self):
        filelist_path = os.path.join(self.dataset_dir, "FileList.csv")

        if not os.path.isfile(filelist_path):
            print(
                f"[ArrhythmiaVideo] FileList.csv not found at {filelist_path}. "
                "Running in DEMO mode (synthetic data)."
            )
            self._load_demo_samples()
            return

        with open(filelist_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                fname  = row.get("FileName", "")
                ef     = _safe_float(row.get("EF"))
                esv    = _safe_float(row.get("ESV"))
                edv    = _safe_float(row.get("EDV"))
                sp     = row.get("Split", "TRAIN").upper()
                fps    = _safe_float(row.get("FPS"))
                nframes= _safe_int(row.get("NumberOfFrames"))

                # Arrhythmia label: explicit column takes priority
                if "Arrhythmia" in row:
                    arrh = int(_safe_int(row["Arrhythmia"]) or 0)
                elif self.ef_proxy and ef is not None:
                    arrh = 1 if (ef < EF_ARRHYTHMIA_LOW or ef > EF_ARRHYTHMIA_HIGH) else 0
                else:
                    arrh = 0

                if self.split and sp != self.split.upper():
                    continue

                # Support .avi with or without extension in CSV
                for ext in ["", ".avi", ".mp4"]:
                    vpath = os.path.join(self.videos_dir, fname + ext)
                    if os.path.isfile(vpath):
                        break
                else:
                    continue   # skip samples with no video file

                self.samples.append({
                    "filename":        fname,
                    "video_path":      vpath,
                    "ef":              ef,
                    "esv":             esv,
                    "edv":             edv,
                    "fps":             fps,
                    "num_raw_frames":  nframes,
                    "split":           sp,
                    "arrhythmia":      arrh,
                    "arrhythmia_label": ARRHYTHMIA_CLASSES[arrh],
                })

        print(
            f"[ArrhythmiaVideo] Loaded {len(self.samples)} samples"
            f"{f' (split={self.split})' if self.split else ''}."
        )
        if self.samples:
            n_arrh = sum(s["arrhythmia"] for s in self.samples)
            print(f"[ArrhythmiaVideo] Arrhythmia: {n_arrh} | Normal: {len(self.samples)-n_arrh}")

    def _load_demo_samples(self):
        """Generate synthetic samples for testing when no dataset is present."""
        rng = np.random.default_rng(42)
        for i in range(50):
            ef = float(rng.normal(55, 12))
            arrh = 1 if (ef < EF_ARRHYTHMIA_LOW or ef > EF_ARRHYTHMIA_HIGH) else 0
            self.samples.append({
                "filename":        f"DEMO_{i:04d}",
                "video_path":      None,
                "ef":              round(ef, 1),
                "esv":             round(float(rng.normal(50, 20)), 1),
                "edv":             round(float(rng.normal(120, 30)), 1),
                "fps":             30.0,
                "num_raw_frames":  int(rng.integers(40, 120)),
                "split":           rng.choice(["TRAIN", "VAL", "TEST"]),
                "arrhythmia":      arrh,
                "arrhythmia_label": ARRHYTHMIA_CLASSES[arrh],
                "_demo": True,
            })
        n_arrh = sum(s["arrhythmia"] for s in self.samples)
        print(f"[ArrhythmiaVideo] DEMO mode: {len(self.samples)} synthetic samples "
              f"({n_arrh} arrhythmia, {len(self.samples)-n_arrh} normal)")

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def load_video_tensor(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """Return (video_tensor, labels) for sample at *index*."""
        sample = self.samples[index]
        if sample.get("_demo") or sample["video_path"] is None:
            tensor = self._synthetic_tensor(sample)
        else:
            frames = self._read_video(sample["video_path"])
            tensor = self._preprocess(frames)
        return tensor, sample

    def load_video_from_path(self, path: str) -> torch.Tensor:
        """Load and preprocess any .avi / .mp4 file into a model-ready tensor."""
        frames = self._read_video(path)
        return self._preprocess(frames)

    def _read_video(self, path: str) -> List[np.ndarray]:
        cv2 = _try_import_cv2()
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.frame_size, self.frame_size))
            frames.append(resized.astype(np.float32) / 255.0)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames could be read from: {path}")
        return frames

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Uniformly sample `num_frames` and normalise."""
        total = len(frames)
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        sampled = np.stack([frames[i] for i in indices], axis=0)   # (T, H, W)
        sampled = (sampled - MEAN) / STD
        return torch.from_numpy(sampled).unsqueeze(0)               # (1, T, H, W)

    def _synthetic_tensor(self, sample: Dict) -> torch.Tensor:
        """Generate a plausible synthetic video tensor for demo mode."""
        rng = np.random.default_rng(hash(sample["filename"]) % (2**31))
        t   = np.linspace(0, 4 * np.pi, self.num_frames)
        base = (0.5 + 0.3 * np.sin(t)).reshape(-1, 1, 1)
        noise = rng.normal(0, 0.05, (self.num_frames, self.frame_size, self.frame_size))
        if sample["arrhythmia"] == 1:
            noise += rng.normal(0, 0.08, noise.shape)
        frames = (base + noise).clip(0, 1).astype(np.float32)
        frames = (frames - MEAN) / STD
        return torch.from_numpy(frames).unsqueeze(0)

    # ------------------------------------------------------------------
    # Dataset-level EDA
    # ------------------------------------------------------------------
    def get_eda_summary(self) -> Dict[str, Any]:
        """Return statistics for the dashboard or Colab display."""
        if not self.samples:
            return {"error": "No samples loaded."}

        efs  = [s["ef"]  for s in self.samples if s["ef"]  is not None]
        esvs = [s["esv"] for s in self.samples if s["esv"] is not None]
        edvs = [s["edv"] for s in self.samples if s["edv"] is not None]

        arrh_samples  = [s for s in self.samples if s["arrhythmia"] == 1]
        norm_samples  = [s for s in self.samples if s["arrhythmia"] == 0]

        splits = {}
        for s in self.samples:
            sp = s.get("split", "UNKNOWN")
            splits[sp] = splits.get(sp, 0) + 1

        return {
            "dataset":          "Cardiac Arrhythmia Video Dataset",
            "source":           "EchoNet-Dynamic (Stanford AIMI) + arrhythmia labels",
            "total_videos":     len(self.samples),
            "arrhythmia_count": len(arrh_samples),
            "normal_count":     len(norm_samples),
            "arrhythmia_pct":   round(len(arrh_samples) / max(len(self.samples), 1) * 100, 1),
            "splits":           splits,
            "ef_stats_all":     _describe(efs),
            "ef_stats_arrh":    _describe([s["ef"] for s in arrh_samples if s["ef"] is not None]),
            "ef_stats_normal":  _describe([s["ef"] for s in norm_samples if s["ef"] is not None]),
            "esv_stats":        _describe(esvs),
            "edv_stats":        _describe(edvs),
            "video_format":     "AVI (apical-4-chamber echocardiogram)",
            "frame_sampling":   f"{self.num_frames} frames @ {self.frame_size}×{self.frame_size}",
            "label_method":     (
                "Explicit 'Arrhythmia' column" if any("Arrhythmia" in s for s in self.samples)
                else f"EF proxy (EF < {EF_ARRHYTHMIA_LOW} or > {EF_ARRHYTHMIA_HIGH})"
            ),
            "guidelines":       ARRHYTHMIA_GUIDELINES,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"CardiacArrhythmiaVideoDataset("
            f"n={len(self.samples)}, "
            f"split={self.split}, "
            f"frames={self.num_frames}, "
            f"size={self.frame_size})"
        )


# ---------------------------------------------------------------------------
# 3D-CNN Arrhythmia Classifier
# ---------------------------------------------------------------------------
class ArrhythmiaVideoCNN(nn.Module):
    """
    Lightweight 3D-CNN for binary arrhythmia classification from video.

    Input:  (B, 1, T, H, W)  — grayscale video tensor
    Output: (B, 2)            — logits for [Normal, Arrhythmia]
    """

    def __init__(self, num_frames: int = NUM_FRAMES, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),   # T/2, H/2, W/2

            # Block 2
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),   # T/4, H/4, W/4

            # Block 3
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),       # fixed output
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run inference and return structured result dict."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
            pred   = int(probs.argmax(dim=-1).item())
            conf   = float(probs[0, pred].item())
        return {
            "prediction":        pred,
            "label":             ARRHYTHMIA_CLASSES[pred],
            "confidence":        round(conf, 4),
            "probabilities":     {ARRHYTHMIA_CLASSES[i]: round(float(probs[0, i].item()), 4)
                                  for i in range(len(ARRHYTHMIA_CLASSES))},
            "guideline":         ARRHYTHMIA_GUIDELINES.get(ARRHYTHMIA_CLASSES[pred], ""),
        }


# ---------------------------------------------------------------------------
# Thread-safe singleton accessors
# ---------------------------------------------------------------------------
_dataset_instance: Optional[CardiacArrhythmiaVideoDataset] = None
_dataset_lock = threading.Lock()

_model_instance: Optional[ArrhythmiaVideoCNN] = None
_model_lock = threading.Lock()


def get_arrhythmia_video_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    split: Optional[str] = None,
) -> CardiacArrhythmiaVideoDataset:
    global _dataset_instance
    with _dataset_lock:
        if _dataset_instance is None:
            _dataset_instance = CardiacArrhythmiaVideoDataset(
                dataset_dir=dataset_dir, split=split
            )
    return _dataset_instance


def get_arrhythmia_video_classifier() -> ArrhythmiaVideoCNN:
    global _model_instance
    with _model_lock:
        if _model_instance is None:
            _model_instance = ArrhythmiaVideoCNN()
            _model_instance.eval()
            print("[ArrhythmiaVideo] 3D-CNN classifier initialised (random weights — "
                  "load trained weights for real inference).")
    return _model_instance
