"""
EchoNet-Dynamic Dataset Loader
==============================
Loads echocardiogram videos (AVI) and metadata from the Stanford AIMI
EchoNet-Dynamic dataset for cardiac function assessment.

Dataset structure expected:
    EchoNet-Dynamic/
    ├── Videos/               # 10,030 .avi echocardiogram clips
    ├── FileList.csv          # metadata: FileName, EF, ESV, EDV, Split
    └── VolumeTracings.csv    # LV tracings per frame
"""

import os
import csv
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "EchoNet-Dynamic")
NUM_FRAMES = 32          # Uniform temporal sampling depth
FRAME_SIZE = 112          # Spatial resize target (112x112)
MEAN = 0.131              # EchoNet grayscale mean (approximate)
STD  = 0.289              # EchoNet grayscale std  (approximate)

# EF clinical thresholds
EF_REDUCED  = 40.0   # HFrEF  (Heart Failure with Reduced EF)
EF_MIDRANGE = 50.0   # HFmrEF (Mid-Range)
# >= 50 is HFpEF / Normal


def _try_import_cv2():
    """Lazy import so the rest of the project doesn't hard-depend on opencv."""
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required to load echocardiogram videos. "
            "Install it with:  pip install opencv-python-headless"
        )


# ---------------------------------------------------------------------------
# EchoNet Dataset Class
# ---------------------------------------------------------------------------
class EchoNetDataset:
    """
    Loads EchoNet-Dynamic echocardiogram videos and their clinical labels.

    Each sample yields:
        video  – torch.Tensor of shape (1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
        labels – dict with keys: ef, esv, edv, split, filename
    """

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR,
                 split: Optional[str] = None,
                 num_frames: int = NUM_FRAMES,
                 frame_size: int = FRAME_SIZE):
        self.dataset_dir = dataset_dir
        self.videos_dir = os.path.join(dataset_dir, "Videos")
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split  # "TRAIN", "VAL", "TEST", or None for all

        self.samples: List[Dict[str, Any]] = []
        self._load_filelist()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _load_filelist(self):
        """Parse FileList.csv into self.samples."""
        filelist_path = os.path.join(self.dataset_dir, "FileList.csv")
        if not os.path.isfile(filelist_path):
            print(f"[EchoNet] FileList.csv not found at {filelist_path}. "
                  "Dataset not downloaded yet — loader will operate in demo mode.")
            return

        with open(filelist_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalise column names (strip whitespace)
                row = {k.strip(): v.strip() for k, v in row.items()}
                sample = {
                    "filename": row.get("FileName", ""),
                    "ef":  _safe_float(row.get("EF")),
                    "esv": _safe_float(row.get("ESV")),
                    "edv": _safe_float(row.get("EDV")),
                    "split": row.get("Split", "TRAIN"),
                }
                if self.split and sample["split"].upper() != self.split.upper():
                    continue
                video_path = os.path.join(self.videos_dir, sample["filename"] + ".avi")
                if os.path.isfile(video_path):
                    sample["video_path"] = video_path
                    self.samples.append(sample)

        print(f"[EchoNet] Loaded {len(self.samples)} samples"
              f"{f' (split={self.split})' if self.split else ''}.")

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def load_video_tensor(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns (video_tensor, labels) for sample at *index*.

        video_tensor shape: (1, num_frames, frame_size, frame_size)
        """
        sample = self.samples[index]
        frames = self._read_avi(sample["video_path"])
        tensor = self._preprocess(frames)
        return tensor, sample

    def _read_avi(self, path: str) -> List[np.ndarray]:
        cv2 = _try_import_cv2()
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.frame_size, self.frame_size))
            frames.append(resized.astype(np.float32) / 255.0)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from {path}")
        return frames

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Uniformly sample `num_frames` and normalise."""
        total = len(frames)
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            # Repeat last frame if video is shorter than target
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        sampled = np.stack([frames[i] for i in indices], axis=0)  # (T, H, W)
        # Normalise
        sampled = (sampled - MEAN) / STD
        tensor = torch.from_numpy(sampled).unsqueeze(0)  # (1, T, H, W)
        return tensor

    # ------------------------------------------------------------------
    # Summary / EDA helpers
    # ------------------------------------------------------------------
    def get_eda_summary(self) -> Dict[str, Any]:
        """Return dataset-level statistics for the dashboard."""
        if not self.samples:
            return self._demo_eda()

        efs  = [s["ef"]  for s in self.samples if s["ef"]  is not None]
        esvs = [s["esv"] for s in self.samples if s["esv"] is not None]
        edvs = [s["edv"] for s in self.samples if s["edv"] is not None]

        splits = {}
        for s in self.samples:
            sp = s["split"]
            splits[sp] = splits.get(sp, 0) + 1

        ef_cats = {"HFrEF (EF<40)": 0, "HFmrEF (40-50)": 0, "Normal (EF>=50)": 0}
        for ef in efs:
            if ef < EF_REDUCED:
                ef_cats["HFrEF (EF<40)"] += 1
            elif ef < EF_MIDRANGE:
                ef_cats["HFmrEF (40-50)"] += 1
            else:
                ef_cats["Normal (EF>=50)"] += 1

        return {
            "dataset": "EchoNet-Dynamic (Stanford AIMI)",
            "total_videos": len(self.samples),
            "splits": splits,
            "ef_stats": _describe(efs),
            "esv_stats": _describe(esvs),
            "edv_stats": _describe(edvs),
            "ef_categories": ef_cats,
            "video_format": "AVI (apical-4-chamber)",
            "frame_sampling": f"{self.num_frames} frames @ {self.frame_size}x{self.frame_size}",
        }

    @staticmethod
    def _demo_eda() -> Dict[str, Any]:
        """Fallback EDA when dataset is not yet downloaded."""
        return {
            "dataset": "EchoNet-Dynamic (Stanford AIMI) — DEMO MODE",
            "total_videos": 10030,
            "splits": {"TRAIN": 7465, "VAL": 1288, "TEST": 1277},
            "ef_stats": {"mean": 55.6, "std": 12.1, "min": 7.9, "max": 89.4},
            "esv_stats": {"mean": 52.3, "std": 30.8, "min": 4.0, "max": 280.0},
            "edv_stats": {"mean": 109.0, "std": 38.5, "min": 20.0, "max": 380.0},
            "ef_categories": {"HFrEF (EF<40)": 1187, "HFmrEF (40-50)": 1398, "Normal (EF>=50)": 7445},
            "video_format": "AVI (apical-4-chamber)",
            "frame_sampling": f"{NUM_FRAMES} frames @ {FRAME_SIZE}x{FRAME_SIZE}",
            "note": "Showing reference statistics. Download dataset from Stanford AIMI to enable real analysis.",
        }

    def __len__(self):
        return len(self.samples)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 2),
        "std":  round(float(arr.std()), 2),
        "min":  round(float(arr.min()), 2),
        "max":  round(float(arr.max()), 2),
    }


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------
_echonet_instance: Optional[EchoNetDataset] = None

def get_echonet_loader(dataset_dir: str = DEFAULT_DATASET_DIR) -> EchoNetDataset:
    global _echonet_instance
    if _echonet_instance is None:
        _echonet_instance = EchoNetDataset(dataset_dir=dataset_dir)
    return _echonet_instance
