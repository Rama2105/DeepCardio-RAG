"""
ECG Image Dataset Loader
========================
Loads the Kaggle ECG Images dataset (derived from MIT-BIH Arrhythmia Database).
109,445 labeled 256x256 grayscale images across 5 AAMI classes.

Classes:
    N (0): Normal beat
    S (1): Supraventricular premature beat
    V (2): Premature ventricular contraction
    F (3): Fusion of ventricular and normal beat
    Q (4): Unclassifiable beat

Dataset structure expected:
    ecg-images/
    ├── train/
    │   ├── N/    *.png
    │   ├── S/    *.png
    │   ├── V/    *.png
    │   ├── F/    *.png
    │   └── Q/    *.png
    └── test/
        ├── N/
        ├── S/
        ├── V/
        ├── F/
        └── Q/

Source: https://www.kaggle.com/datasets/analiviafr/ecg-images
Derived from: PhysioNet MIT-BIH (https://physionet.org/content/mitdb/1.0.0/)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ecg-images")
IMG_SIZE = 128           # Resize target (128x128 for efficiency)
NUM_CLASSES = 5

AAMI_CLASSES = {
    0: {"label": "N", "name": "Normal",           "description": "Normal beat"},
    1: {"label": "S", "name": "Supraventricular",  "description": "Supraventricular premature beat"},
    2: {"label": "V", "name": "Ventricular",       "description": "Premature ventricular contraction (PVC)"},
    3: {"label": "F", "name": "Fusion",            "description": "Fusion of ventricular and normal beat"},
    4: {"label": "Q", "name": "Unclassifiable",    "description": "Unclassifiable beat"},
}

CLASS_LABELS = ["N", "S", "V", "F", "Q"]


def _try_import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise ImportError("Pillow is required for ECG image loading. Install with: pip install Pillow")


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------
class ECGImageDataset:
    """
    Loads ECG waveform images from the Kaggle dataset and provides
    classification labels, EDA statistics, and batch loading.
    """

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR,
                 split: str = "train", img_size: int = IMG_SIZE):
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.split_dir = os.path.join(dataset_dir, split)

        self.samples: List[Tuple[str, int]] = []  # (filepath, class_idx)
        self.class_counts: Dict[str, int] = {}
        self._scan_directory()

    def _scan_directory(self):
        """Scan folder structure and index all images."""
        if not os.path.isdir(self.split_dir):
            print(f"[ECG-Images] Directory not found: {self.split_dir}. "
                  "Dataset not downloaded yet — loader operates in demo mode.")
            return

        for cls_idx, cls_label in enumerate(CLASS_LABELS):
            cls_dir = os.path.join(self.split_dir, cls_label)
            if not os.path.isdir(cls_dir):
                continue
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.class_counts[cls_label] = len(files)
            for f in files:
                self.samples.append((os.path.join(cls_dir, f), cls_idx))

        if self.samples:
            print(f"[ECG-Images] Loaded {len(self.samples)} images ({self.split}): "
                  f"{self.class_counts}")

    def load_image_tensor(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load a single image as tensor.
        Returns: (tensor [1, H, W], class_idx, class_label)
        """
        Image = _try_import_pil()
        filepath, cls_idx = self.samples[index]
        img = Image.open(filepath).convert("L")  # grayscale
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, cls_idx, CLASS_LABELS[cls_idx]

    def load_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of images. Returns (images [B,1,H,W], labels [B])."""
        tensors, labels = [], []
        for idx in indices:
            t, c, _ = self.load_image_tensor(idx)
            tensors.append(t)
            labels.append(c)
        return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)

    def get_random_sample(self, class_label: Optional[str] = None) -> Tuple[torch.Tensor, int, str]:
        """Get a random sample, optionally filtered by class."""
        if class_label:
            cls_idx = CLASS_LABELS.index(class_label)
            pool = [(i, s) for i, s in enumerate(self.samples) if s[1] == cls_idx]
        else:
            pool = list(enumerate(self.samples))
        if not pool:
            raise ValueError(f"No samples found for class '{class_label}'")
        idx, _ = random.choice(pool)
        return self.load_image_tensor(idx)

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def get_eda_summary(self) -> Dict[str, Any]:
        if not self.samples:
            return self._demo_eda()

        total = len(self.samples)
        distribution = {}
        for cls_label in CLASS_LABELS:
            count = self.class_counts.get(cls_label, 0)
            distribution[cls_label] = {
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0,
                "name": AAMI_CLASSES[CLASS_LABELS.index(cls_label)]["name"],
            }

        return {
            "dataset": "ECG Images (Kaggle — MIT-BIH derived)",
            "source": "MIT-BIH Arrhythmia Database via PhysioNet",
            "total_images": total,
            "split": self.split,
            "image_size": f"{self.img_size}x{self.img_size} grayscale",
            "num_classes": NUM_CLASSES,
            "class_distribution": distribution,
            "class_labels": CLASS_LABELS,
            "aami_standard": "AAMI EC57",
        }

    @staticmethod
    def _demo_eda() -> Dict[str, Any]:
        return {
            "dataset": "ECG Images (Kaggle — MIT-BIH derived) — DEMO MODE",
            "source": "MIT-BIH Arrhythmia Database via PhysioNet",
            "total_images": 109445,
            "split": "train",
            "image_size": f"{IMG_SIZE}x{IMG_SIZE} grayscale",
            "num_classes": NUM_CLASSES,
            "class_distribution": {
                "N": {"count": 90589, "percentage": 82.77, "name": "Normal"},
                "S": {"count": 2779,  "percentage": 2.54,  "name": "Supraventricular"},
                "V": {"count": 7236,  "percentage": 6.61,  "name": "Ventricular"},
                "F": {"count": 803,   "percentage": 0.73,  "name": "Fusion"},
                "Q": {"count": 8038,  "percentage": 7.34,  "name": "Unclassifiable"},
            },
            "class_labels": CLASS_LABELS,
            "aami_standard": "AAMI EC57",
            "note": "Reference statistics. Download dataset from Kaggle to enable real analysis.",
        }

    def __len__(self):
        return len(self.samples)


# ---------------------------------------------------------------------------
# 2D-CNN ECG Image Classifier
# ---------------------------------------------------------------------------
class ECGImageClassifier(nn.Module):
    """
    2D-CNN for classifying ECG waveform images into 5 AAMI classes.
    Input: (B, 1, 128, 128) grayscale ECG images
    Output: (B, 5) class logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
        )

        self.embedding = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x).flatten(1)
        emb = self.embedding(feat)
        logits = self.classifier(emb)
        return {"logits": logits, "embeddings": emb}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        self.eval()
        out = self.forward(x)
        probs = torch.softmax(out["logits"], dim=1)
        pred_idx = probs.argmax(dim=1)

        results = []
        for i in range(x.size(0)):
            idx = pred_idx[i].item()
            results.append({
                "class_index": idx,
                "class_label": CLASS_LABELS[idx],
                "class_name": AAMI_CLASSES[idx]["name"],
                "description": AAMI_CLASSES[idx]["description"],
                "confidence": round(probs[i, idx].item(), 4),
                "probabilities": {
                    CLASS_LABELS[j]: round(probs[i, j].item(), 4)
                    for j in range(NUM_CLASSES)
                },
            })
        return results


# ---------------------------------------------------------------------------
# Clinical guidelines per arrhythmia class
# ---------------------------------------------------------------------------
ECG_IMAGE_GUIDELINES = {
    "N": [
        "Normal sinus rhythm. No acute intervention required.",
        "Regular R-R intervals with consistent P-wave morphology.",
    ],
    "S": [
        "Supraventricular premature beat detected. Evaluate for atrial fibrillation risk.",
        "Guideline (ACC/AHA): Frequent SVPBs (>500/day) associated with increased AF risk.",
    ],
    "V": [
        "Premature Ventricular Contraction (PVC) detected. If >10% PVC burden, evaluate for cardiomyopathy.",
        "Guideline: Consider Holter monitoring and echocardiography for frequent PVCs.",
    ],
    "F": [
        "Fusion beat detected — simultaneous ventricular and supraventricular activation.",
        "May indicate underlying conduction abnormality. Correlate with clinical context.",
    ],
    "Q": [
        "Unclassifiable beat morphology. Recommend manual review by electrophysiologist.",
        "Consider artifact vs. true arrhythmia. Repeat ECG recording if clinically indicated.",
    ],
}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_ecg_img_dataset: Optional[ECGImageDataset] = None
_ecg_img_classifier: Optional[ECGImageClassifier] = None

def get_ecg_image_dataset(split: str = "train") -> ECGImageDataset:
    global _ecg_img_dataset
    if _ecg_img_dataset is None or _ecg_img_dataset.split != split:
        _ecg_img_dataset = ECGImageDataset(split=split)
    return _ecg_img_dataset

def get_ecg_image_classifier() -> ECGImageClassifier:
    global _ecg_img_classifier
    if _ecg_img_classifier is None:
        _ecg_img_classifier = ECGImageClassifier()
        _ecg_img_classifier.eval()
        # Load pretrained weights if available
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "data", "ecg_image_classifier.pt")
        # Callers MUST check this before rendering a class label or confidence:
        # with random weights the softmax still returns a confident-looking beat
        # class, which the API then paired with real clinical guidelines.
        _ecg_img_classifier.is_trained = False
        if os.path.isfile(weights_path):
            _ecg_img_classifier.load_state_dict(torch.load(weights_path, map_location="cpu"))
            _ecg_img_classifier.is_trained = True
            print("[ECG-Images] Loaded pretrained classifier weights.")
        else:
            print("[ECG-Images] No pretrained weights found. Using random initialization (demo mode).")
    return _ecg_img_classifier
