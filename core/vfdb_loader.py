"""
MIT-BIH Malignant Ventricular Ectopy Database (VFDB) Loader
============================================================
Loads the PhysioNet VFDB dataset for ventricular arrhythmia detection.

Dataset: https://physionet.org/content/vfdb/1.0.0/
Contains 22 half-hour ECG recordings of subjects with episodes of:
  - Ventricular Tachycardia (VT)
  - Ventricular Flutter (VFL)
  - Ventricular Fibrillation (VF/VFIB)
  - Other dangerous arrhythmias

File types:
    .dat  — Digitized ECG signal data (WFDB format)
    .hea  — Header files (metadata: leads, sampling rate, gain)
    .atr  — Rhythm annotations (rhythm changes only, no beat labels)

Rhythm annotation codes:
    N/NSR  — Normal sinus rhythm
    VT     — Ventricular tachycardia
    VF/VFIB — Ventricular fibrillation
    VFL    — Ventricular flutter
    AFIB   — Atrial fibrillation
    ASYS   — Asystole
    NOISE  — Noise/artifact
    SBR    — Sinus bradycardia
    SVTA   — Supraventricular tachyarrhythmia
    HGEA   — High-grade ventricular ectopic activity
    VER    — Ventricular escape rhythm
    BI     — First degree heart block
    B      — Ventricular bigeminy
    NOD    — Nodal/AV junctional rhythm
    PM     — Pacemaker/paced rhythm

Structure expected:
    vfdb/
    ├── 418.dat, 418.hea, 418.atr
    ├── 419.dat, 419.hea, 419.atr
    ├── ...
    └── RECORDS
"""

import os
import struct
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "vfdb"
)

VFDB_RECORDS = [
    "418", "419", "420", "421", "422", "423", "424",
    "425", "426", "427", "428", "429", "430", "602",
    "605", "607", "609", "610", "611", "612", "614", "615",
]

# Dangerous rhythms (the ones this DB specifically targets)
DANGEROUS_RHYTHMS = {"VT", "VF", "VFIB", "VFL", "ASYS", "HGEA", "VER"}
ALL_RHYTHMS = {
    "N": "Normal Sinus Rhythm", "NSR": "Normal Sinus Rhythm",
    "VT": "Ventricular Tachycardia", "VF": "Ventricular Fibrillation",
    "VFIB": "Ventricular Fibrillation", "VFL": "Ventricular Flutter",
    "AFIB": "Atrial Fibrillation", "ASYS": "Asystole",
    "NOISE": "Noise/Artifact", "SBR": "Sinus Bradycardia",
    "SVTA": "Supraventricular Tachyarrhythmia",
    "HGEA": "High-Grade Ventricular Ectopy",
    "VER": "Ventricular Escape Rhythm",
    "BI": "First Degree Heart Block", "B": "Ventricular Bigeminy",
    "NOD": "Nodal/AV Junctional Rhythm", "PM": "Pacemaker Rhythm",
}

SEGMENT_SECONDS = 10  # Default analysis window


# ---------------------------------------------------------------------------
# WFDB-lite reader (no wfdb library required)
# ---------------------------------------------------------------------------
def read_hea(filepath: str) -> Dict[str, Any]:
    """Parse a .hea header file for basic metadata."""
    info = {"n_signals": 0, "fs": 250, "n_samples": 0, "leads": []}
    with open(filepath, "r") as f:
        lines = f.readlines()
    if not lines:
        return info
    # First line: record_name n_signals fs n_samples
    parts = lines[0].strip().split()
    if len(parts) >= 3:
        info["n_signals"] = int(parts[1])
        info["fs"] = int(parts[2])
    if len(parts) >= 4:
        info["n_samples"] = int(parts[3])
    # Signal lines
    for i in range(1, 1 + info["n_signals"]):
        if i < len(lines):
            sig_parts = lines[i].strip().split()
            if len(sig_parts) >= 9:
                info["leads"].append(sig_parts[8])
            else:
                info["leads"].append(f"Lead_{i}")
    return info


def read_dat_212(filepath: str, n_signals: int, n_samples: int) -> np.ndarray:
    """Read a WFDB format 212 .dat file (common for PhysioNet)."""
    with open(filepath, "rb") as f:
        raw = f.read()

    # Format 212: 2 signals packed in 3 bytes
    if n_signals == 2:
        n_bytes = len(raw)
        n_sample_pairs = n_bytes // 3
        signals = np.zeros((2, n_sample_pairs), dtype=np.float32)
        for i in range(n_sample_pairs):
            b = raw[i * 3:(i + 1) * 3]
            if len(b) < 3:
                break
            # First sample: low byte + low nibble of middle
            s1 = b[0] | ((b[1] & 0x0F) << 8)
            if s1 >= 2048:
                s1 -= 4096
            # Second sample: high nibble of middle + high byte
            s2 = (b[2] << 4) | ((b[1] & 0xF0) >> 4)
            if s2 >= 2048:
                s2 -= 4096
            signals[0, i] = s1
            signals[1, i] = s2
        return signals[:, :min(n_samples, n_sample_pairs)]
    else:
        # Fallback: read as raw 16-bit
        samples = np.frombuffer(raw, dtype=np.int16)
        if n_signals > 0:
            trim = (len(samples) // n_signals) * n_signals
            return samples[:trim].reshape(-1, n_signals).T.astype(np.float32)
        return samples.astype(np.float32).reshape(1, -1)


def _normalise_rhythm(aux_text: str, current: str) -> str:
    """
    Turn a WFDB aux_note into a rhythm code.

    VFDB rhythm changes are stored as aux strings of the form '(VT', '(VF', '(N'.
    The leading '(' opens a rhythm; a bare ')' closes one and carries no new label.
    Anything empty or unparseable leaves the current rhythm in effect.
    """
    t = (aux_text or "").replace("\x00", "").strip()
    if not t:
        return current
    if t.startswith("("):
        t = t[1:].strip()
        return t.upper() if t else current
    if t.startswith(")"):
        return current
    return t.upper()


def read_atr_wfdb(filepath: str, fs: int = 250) -> List[Dict]:
    """
    Read rhythm annotations with the `wfdb` library (authoritative parser).

    Returns [] if wfdb is unavailable or the read fails, so the caller can fall
    back to the hand-rolled reader.
    """
    try:
        import wfdb
    except ImportError:
        print("[VFDB] wfdb not installed — falling back to the byte parser, whose "
              "sample positions are unreliable. Run: pip install wfdb")
        return []
    record_base = filepath[:-4] if filepath.endswith(".atr") else filepath
    try:
        ann = wfdb.rdann(record_base, "atr")
    except Exception as exc:
        # Never swallow this silently: the fallback parser yields plausible rhythm
        # strings with nonsensical sample offsets, which labels every window
        # 'Normal' and looks like clean data rather than a failure.
        print(f"[VFDB] wfdb.rdann failed on {record_base}: {type(exc).__name__}: {exc}")
        return []

    aux_notes = getattr(ann, "aux_note", None) or []
    annotations, current = [], "N"
    for samp, aux in zip(ann.sample, aux_notes):
        new_rhythm = _normalise_rhythm(aux, current)
        if new_rhythm == current and not (aux or "").strip():
            continue                      # no aux text -> beat annotation, not a rhythm change
        current = new_rhythm
        annotations.append({
            "sample": int(samp),
            "time": round(int(samp) / fs, 3),
            "rhythm": current,
            "dangerous": current in DANGEROUS_RHYTHMS,
        })
    return annotations


def read_atr(filepath: str, fs: int = 250) -> List[Dict]:
    """
    Parse a .atr annotation file for rhythm labels.
    Returns list of {sample, time, rhythm} dicts.

    Prefers the `wfdb` library; the byte-level reader below is a fallback for
    environments without it. The fallback mis-parses some VFDB files (it was
    yielding zero rhythm changes, which silently made every segment 'Normal'),
    so a parse that produces no dangerous rhythms is reported by the caller
    rather than passed to training.
    """
    if not os.path.isfile(filepath):
        return []

    annotations = read_atr_wfdb(filepath, fs)
    if annotations:
        return annotations

    with open(filepath, "rb") as f:
        raw = f.read()

    pos = 0
    sample_counter = 0
    current_rhythm = "N"

    while pos + 1 < len(raw):
        a = raw[pos]
        b = raw[pos + 1]
        anntype = (b >> 2) & 0x3F
        sample_delta = ((b & 0x03) << 8) | a
        pos += 2

        # Skip: type 0 is NOTQRS, handle special codes
        if anntype == 0 and sample_delta == 0:
            break

        # SKIP code
        if anntype == 59:  # SKIP
            if pos + 3 < len(raw):
                skip_val = struct.unpack_from("<I", raw, pos)[0]
                sample_counter += skip_val
                pos += 4
            continue

        # AUX code (contains rhythm label text)
        if anntype == 63:  # NOTE/AUX
            aux_len = sample_delta
            if aux_len % 2 == 1:
                aux_len += 1  # pad to even
            if pos + aux_len <= len(raw):
                aux_text = raw[pos:pos + sample_delta].decode("ascii", errors="ignore").strip("\x00").strip()
                current_rhythm = _normalise_rhythm(aux_text, current_rhythm)
                pos += aux_len
                annotations.append({
                    "sample": sample_counter,
                    "time": round(sample_counter / fs, 3),
                    "rhythm": current_rhythm,
                    "dangerous": current_rhythm in DANGEROUS_RHYTHMS,
                })
            continue

        sample_counter += sample_delta

    return annotations


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------
class VFDBDataset:
    """
    Loads MIT-BIH Malignant Ventricular Ectopy Database recordings.
    """

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.records: List[Dict] = []
        self._scan_records()

    def _scan_records(self):
        """Find and index all available VFDB records."""
        for rec_id in VFDB_RECORDS:
            hea_path = os.path.join(self.dataset_dir, f"{rec_id}.hea")
            dat_path = os.path.join(self.dataset_dir, f"{rec_id}.dat")
            atr_path = os.path.join(self.dataset_dir, f"{rec_id}.atr")

            if os.path.isfile(hea_path) and os.path.isfile(dat_path):
                info = read_hea(hea_path)
                annotations = read_atr(atr_path, info["fs"]) if os.path.isfile(atr_path) else []

                dangerous_events = [a for a in annotations if a.get("dangerous")]
                rhythm_types = list(set(a["rhythm"] for a in annotations))

                self.records.append({
                    "record_id": rec_id,
                    "hea_path": hea_path,
                    "dat_path": dat_path,
                    "atr_path": atr_path,
                    "fs": info["fs"],
                    "n_signals": info["n_signals"],
                    "n_samples": info["n_samples"],
                    "leads": info["leads"],
                    "annotations": annotations,
                    "dangerous_events": len(dangerous_events),
                    "rhythm_types": rhythm_types,
                })

        if self.records:
            n_ann = sum(len(r["annotations"]) for r in self.records)
            n_dang = sum(r["dangerous_events"] for r in self.records)
            print(f"[VFDB] Loaded {len(self.records)} records | "
                  f"{n_ann} rhythm annotations | {n_dang} dangerous episodes.")
            # A record set that parses to zero dangerous episodes is a parser
            # failure, not a property of VFDB — every recording in this database
            # contains malignant ventricular ectopy. Say so loudly: training on
            # it silently yields a single-class problem scoring accuracy 1.0.
            if n_dang == 0:
                print("[VFDB] ⚠ ZERO dangerous episodes parsed — annotations did NOT "
                      "load correctly. Install `wfdb` (pip install wfdb); training on "
                      "this would produce a meaningless single-class result.")
        else:
            print(f"[VFDB] No records found in {self.dataset_dir}. Demo mode.")

    def load_signal(self, index: int) -> Tuple[np.ndarray, Dict]:
        """
        Load full ECG signal for a record.
        Returns: (signal [n_signals, n_samples], record_info)
        """
        rec = self.records[index]
        signal = read_dat_212(rec["dat_path"], rec["n_signals"], rec["n_samples"])
        return signal, rec

    def load_segment_tensor(self, index: int, start_sec: float = 0,
                             duration: float = SEGMENT_SECONDS) -> Tuple[torch.Tensor, Dict]:
        """
        Load a segment of ECG as a tensor.
        Returns: (tensor [n_signals, segment_samples], record_info)
        """
        signal, rec = self.load_signal(index)
        fs = rec["fs"]
        start_sample = int(start_sec * fs)
        end_sample = start_sample + int(duration * fs)
        segment = signal[:, start_sample:end_sample]

        # Normalize
        for ch in range(segment.shape[0]):
            mean = segment[ch].mean()
            std = segment[ch].std()
            if std > 0:
                segment[ch] = (segment[ch] - mean) / std

        tensor = torch.from_numpy(segment).float()
        return tensor, rec

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def get_eda_summary(self) -> Dict[str, Any]:
        if not self.records:
            return self._demo_eda()

        rhythm_counts = {}
        total_dangerous = 0
        for rec in self.records:
            total_dangerous += rec["dangerous_events"]
            for ann in rec["annotations"]:
                r = ann["rhythm"]
                rhythm_counts[r] = rhythm_counts.get(r, 0) + 1

        return {
            "dataset": "MIT-BIH Malignant Ventricular Ectopy Database (VFDB)",
            "source": "PhysioNet",
            "total_records": len(self.records),
            "record_ids": [r["record_id"] for r in self.records],
            "total_dangerous_events": total_dangerous,
            "rhythm_distribution": rhythm_counts,
            "dangerous_rhythms": list(DANGEROUS_RHYTHMS),
            "all_rhythm_labels": {k: v for k, v in ALL_RHYTHMS.items()},
            "recording_duration": "30 minutes each",
        }

    @staticmethod
    def _demo_eda() -> Dict[str, Any]:
        return {
            "dataset": "MIT-BIH Malignant Ventricular Ectopy Database (VFDB) — DEMO MODE",
            "source": "PhysioNet",
            "total_records": 22,
            "record_ids": VFDB_RECORDS,
            "total_dangerous_events": 142,
            "rhythm_distribution": {
                "N": 65, "VT": 32, "VF": 18, "VFL": 12, "AFIB": 8,
                "ASYS": 5, "NOISE": 7, "SBR": 4, "HGEA": 6, "VER": 3,
                "SVTA": 2, "NOD": 1, "B": 2, "PM": 3,
            },
            "dangerous_rhythms": list(DANGEROUS_RHYTHMS),
            "all_rhythm_labels": {k: v for k, v in ALL_RHYTHMS.items()},
            "recording_duration": "30 minutes each",
            "note": "Reference stats. Download from PhysioNet for real analysis.",
        }

    def __len__(self):
        return len(self.records)


# ---------------------------------------------------------------------------
# Arrhythmia Detector (1D-CNN for rhythm classification)
# ---------------------------------------------------------------------------
class VentricularArrhythmiaDetector(nn.Module):
    """
    1D-CNN for detecting dangerous ventricular arrhythmias from ECG segments.
    Binary classification: Normal vs Dangerous.
    Plus multi-class rhythm identification.
    """

    RHYTHM_CLASSES = ["Normal", "VT", "VF/VFL", "Other_Dangerous"]

    def __init__(self, in_channels: int = 2, embed_dim: int = 256):
        super().__init__()
        # Trained-state flag (peer-review M1): a detector on random weights must not
        # surface a live alert. Set True only when genuine weights are loaded.
        self.is_trained = False
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, 11, stride=2, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, 7, stride=1, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim), nn.Dropout(0.3),
        )
        # Binary: dangerous or not
        self.binary_head = nn.Linear(embed_dim, 2)
        # Multi-class rhythm
        self.rhythm_head = nn.Linear(embed_dim, len(self.RHYTHM_CLASSES))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x).flatten(1)
        emb = self.embedding(feat)
        return {
            "binary_logits": self.binary_head(emb),
            "rhythm_logits": self.rhythm_head(emb),
            "embeddings": emb,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[Dict]:
        self.eval()
        out = self.forward(x)
        binary_probs = torch.softmax(out["binary_logits"], dim=1)
        rhythm_probs = torch.softmax(out["rhythm_logits"], dim=1)

        from core.safety_gating import gate_module_output

        results = []
        for i in range(x.size(0)):
            danger_prob = binary_probs[i, 1].item()
            rhythm_idx = rhythm_probs[i].argmax().item()
            # Safety gate: suppress the alert entirely if the detector is not
            # genuinely trained (M1 — no alerts from random weights).
            gate = gate_module_output("VFDB", getattr(self, "is_trained", False),
                                      confidence=danger_prob, value=danger_prob)
            if not gate.surfaced or gate.status == "SUPPRESSED":
                alert_level = "SUPPRESSED"
            else:
                alert_level = ("CRITICAL" if danger_prob > 0.8
                               else "WARNING" if danger_prob > 0.5 else "NORMAL")
            results.append({
                "is_dangerous": danger_prob > 0.5 and gate.surfaced,
                "danger_probability": round(danger_prob, 4),
                "alert_level": alert_level,
                "safety_status": gate.status,
                "safety_reasons": gate.reasons,
                "clinician_confirmation_required": True,
                "rhythm_class": self.RHYTHM_CLASSES[rhythm_idx],
                "rhythm_confidence": round(rhythm_probs[i, rhythm_idx].item(), 4),
                "rhythm_probabilities": {
                    self.RHYTHM_CLASSES[j]: round(rhythm_probs[i, j].item(), 4)
                    for j in range(len(self.RHYTHM_CLASSES))
                },
            })
        return results


# ---------------------------------------------------------------------------
# Clinical guidelines for ventricular arrhythmias
# ---------------------------------------------------------------------------
# NOTE (peer-review M1): these are ADVISORY findings for a clinician, not orders the
# system issues. Every entry is phrased as an observation requiring clinician
# confirmation; imperative treatment orders have been removed.
#
# These constants are curated HERE and are deliberately NOT passed through
# core/safety_gating.reframe_guideline_list() at surfacing time. They legitimately
# reference terms such as "defibrillation" while attributing the decision to a
# clinician, and a second substitution pass would rewrite those references
# mid-sentence. The runtime reframing pass is applied to UNCURATED text instead —
# see core/langchain_rag._gate_rag_result(). (An earlier version of this comment
# claimed the pipeline ran that second pass; it never did.)
VFDB_GUIDELINES = {
    "VT": [
        "[ADVISORY — requires clinician confirmation] Pattern consistent with Ventricular "
        "Tachycardia; clinician to assess urgency and hemodynamic stability.",
        "Reference (AHA/ACC): for confirmed sustained VT (>30s) with instability, clinicians "
        "may consider synchronized cardioversion — a clinician decision, not a system order.",
        "Reference: antiarrhythmic therapy and ICD candidacy are clinician-directed decisions.",
    ],
    "VF/VFL": [
        "[ADVISORY — requires clinician confirmation] Pattern that may correspond to a "
        "shockable rhythm (Ventricular Fibrillation/Flutter); clinician to confirm before any "
        "resuscitation decision.",
        "Reference (AHA ACLS): shockable-rhythm management (defibrillation, CPR, medications) "
        "is performed under clinician direction and is not recommended or ordered by this system.",
        "Reference: post-resuscitation considerations (temperature management, further imaging, "
        "ICD evaluation) are clinician-directed.",
    ],
    "Other_Dangerous": [
        "[ADVISORY — requires clinician confirmation] Pattern that a clinician may wish to "
        "evaluate; continuous monitoring is a reasonable precaution pending clinician review.",
        "Reference: assessment of hemodynamic stability is clinician-directed.",
    ],
    "Normal": [
        "No dangerous arrhythmia pattern detected in this segment (advisory; not a substitute "
        "for clinician interpretation).",
        "Continue monitoring per clinical protocol.",
    ],
}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_vfdb_dataset: Optional[VFDBDataset] = None
_vfdb_detector: Optional[VentricularArrhythmiaDetector] = None

def get_vfdb_dataset() -> VFDBDataset:
    global _vfdb_dataset
    if _vfdb_dataset is None:
        _vfdb_dataset = VFDBDataset()
    return _vfdb_dataset

def get_vfdb_detector() -> VentricularArrhythmiaDetector:
    global _vfdb_detector
    if _vfdb_detector is None:
        _vfdb_detector = VentricularArrhythmiaDetector()
        _vfdb_detector.eval()
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "data", "vfdb_detector.pt")
        if os.path.isfile(weights_path):
            _vfdb_detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
            _vfdb_detector.is_trained = True
            print("[VFDB] Loaded genuinely TRAINED detector (alerts enabled, advisory).")
        else:
            _vfdb_detector.is_trained = False
            print("[VFDB] No trained weights — UNTRAINED; alerts SUPPRESSED (run core/train_vfdb.py).")
    return _vfdb_detector
