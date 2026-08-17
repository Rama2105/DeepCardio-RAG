"""
PTB-XL Real 12-Lead ECG Signal Loader
=====================================
Loads the PhysioNet PTB-XL dataset to feed REAL clinical ECG waveforms into
the DeepCardio-RAG pipeline (replacing the synthetic torch.randn signal that
/api/analyze previously used).

Dataset: https://physionet.org/content/ptb-xl/1.0.3/
21,837 clinical 12-lead ECG records (10 s @ 500 Hz, i.e. 5000 samples/lead)
from 18,885 patients, annotated with SCP-ECG diagnostic statements that
aggregate into 5 diagnostic superclasses:
    NORM  — Normal ECG
    MI    — Myocardial Infarction
    STTC  — ST/T-wave Change
    CD    — Conduction Disturbance
    HYP   — Hypertrophy

Structure expected (PhysioNet layout):
    ptbxl/
    ├── ptbxl_database.csv      # record metadata + scp_codes (per-record)
    ├── scp_statements.csv      # SCP code → diagnostic superclass mapping
    ├── records100/             # 100 Hz waveforms (.hea/.dat, WFDB format)
    └── records500/             # 500 Hz waveforms (.hea/.dat, WFDB format)

Reading WFDB signal files requires the optional `wfdb` package. If it (or the
dataset) isn't available, this loader degrades gracefully to synthetic signals
generated from the published per-superclass statistics so the pipeline keeps
working end-to-end.
"""

import os
import ast
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOCAL_DATASET_DIR = os.path.join(DATA_DIR, "ptbxl")
KAGGLE_ID = "khyeh0719/ptb-xl-dataset"

ECG_LEADS = 12
ECG_LENGTH = 5000          # samples per lead (10 s @ 500 Hz) — matches core/pipeline.py
TARGET_SAMPLING_RATE = 500

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_INFO = {
    "NORM": {
        "name": "Normal ECG",
        "description": "No significant abnormalities detected in rhythm, conduction, or morphology.",
    },
    "MI": {
        "name": "Myocardial Infarction",
        "description": "ECG changes consistent with current or prior myocardial infarction (e.g. pathological Q waves, ST elevation).",
    },
    "STTC": {
        "name": "ST/T-Wave Change",
        "description": "Non-specific or ischemic ST-segment and T-wave abnormalities.",
    },
    "CD": {
        "name": "Conduction Disturbance",
        "description": "Abnormalities in electrical conduction (e.g. bundle branch block, AV block, fascicular block).",
    },
    "HYP": {
        "name": "Hypertrophy",
        "description": "Chamber enlargement / hypertrophy patterns (e.g. left ventricular hypertrophy).",
    },
}

# Reference per-superclass record counts published with PTB-XL (used for the
# demo/EDA fallback when the real CSVs aren't downloaded). Records can carry
# more than one diagnostic superclass, so these do not sum to the total.
PTBXL_REFERENCE_COUNTS = {"NORM": 9528, "MI": 5486, "STTC": 5250, "CD": 4907, "HYP": 2655}
PTBXL_REFERENCE_TOTAL = 21837
PTBXL_REFERENCE_PATIENTS = 18885


# ---------------------------------------------------------------------------
# Drive-mounted cache helpers (Colab persistence, mirrors other RAG loaders)
# ---------------------------------------------------------------------------
def _drive_cache_dir() -> Optional[str]:
    drive_root = "/content/drive/MyDrive"
    if not os.path.isdir(drive_root):
        return None
    cache_dir = os.path.join(drive_root, "DeepCardio-RAG", ".cache", "kagglehub")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _resolve_dataset_root() -> Optional[str]:
    """Find a directory containing ptbxl_database.csv, trying local → Drive cache → Kaggle."""
    if os.path.isfile(os.path.join(LOCAL_DATASET_DIR, "ptbxl_database.csv")):
        return LOCAL_DATASET_DIR

    drive_cache = _drive_cache_dir()
    if drive_cache:
        for root, _dirs, files in os.walk(drive_cache):
            if "ptbxl_database.csv" in files:
                print(f"[PTB-XL] Using Drive-cached dataset at {root}")
                return root

    try:
        import kagglehub
        if drive_cache:
            os.environ.setdefault("KAGGLEHUB_CACHE", drive_cache)
            print(f"[PTB-XL] Persisting Kaggle download to Drive cache: {drive_cache}")
        path = kagglehub.dataset_download(KAGGLE_ID)
        for root, _dirs, files in os.walk(path):
            if "ptbxl_database.csv" in files:
                print(f"[PTB-XL] Downloaded dataset, using {root}")
                return root
        print(f"[PTB-XL] Downloaded to {path} but ptbxl_database.csv not found — check dataset layout.")
    except Exception as e:
        print(f"[PTB-XL] Auto-download unavailable ({e}). "
              f"Manually place the PhysioNet PTB-XL files under: {LOCAL_DATASET_DIR}")
    return None


def _try_import_wfdb():
    try:
        import wfdb
        return wfdb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PTBXLDataset:
    """
    Indexes PTB-XL records and their diagnostic-superclass labels, and loads
    real 12-lead waveforms (via wfdb) on demand for inference/demo use.
    """

    def __init__(self, sampling_rate: int = TARGET_SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.root = _resolve_dataset_root()
        self.wfdb = _try_import_wfdb()
        self.records: List[Dict[str, Any]] = []
        self._scp_to_superclass: Dict[str, str] = {}
        if self.root:
            self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
    def _load_metadata(self):
        import pandas as pd
        try:
            scp_path = os.path.join(self.root, "scp_statements.csv")
            scp_df = pd.read_csv(scp_path, index_col=0)
            for code, row in scp_df.iterrows():
                superclass = row.get("diagnostic_class")
                if isinstance(superclass, str) and superclass in SUPERCLASSES:
                    self._scp_to_superclass[code] = superclass

            db_path = os.path.join(self.root, "ptbxl_database.csv")
            db_df = pd.read_csv(db_path, index_col="ecg_id")
            db_df["scp_codes"] = db_df["scp_codes"].apply(ast.literal_eval)

            folder = "records500" if self.sampling_rate == 500 else "records100"
            for ecg_id, row in db_df.iterrows():
                superclasses = sorted({
                    self._scp_to_superclass[code]
                    for code in row["scp_codes"].keys()
                    if code in self._scp_to_superclass
                })
                if not superclasses:
                    continue
                rel_path = row.get("filename_hr") if self.sampling_rate == 500 else row.get("filename_lr")
                if not isinstance(rel_path, str):
                    continue
                self.records.append({
                    "ecg_id": int(ecg_id),
                    "path": os.path.join(self.root, rel_path),
                    "superclasses": superclasses,
                    "primary_superclass": superclasses[0],
                    "age": _safe_num(row.get("age")),
                    "sex": "Male" if row.get("sex") == 0 else "Female",
                    "heart_rate": _safe_num(row.get("heart_axis")),
                    # patient_id + strat_fold enable a PATIENT-INDEPENDENT split
                    # (PTB-XL's official 10-fold assignment keeps each patient's
                    #  records inside a single fold — no patient leakage across splits).
                    "patient_id": _safe_num(row.get("patient_id")),
                    "strat_fold": _safe_num(row.get("strat_fold")),
                })
            print(f"[PTB-XL] Indexed {len(self.records):,} labeled records "
                  f"(sampling_rate={self.sampling_rate} Hz, wfdb={'available' if self.wfdb else 'MISSING'})")
        except Exception as e:
            print(f"[PTB-XL] Failed to index dataset metadata: {e}")
            self.records = []

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------
    def load_signal(self, record: Dict[str, Any]) -> np.ndarray:
        """Returns a (12, ECG_LENGTH) float32 waveform, resampled/padded as needed."""
        if self.wfdb is not None:
            try:
                signal, _meta = self.wfdb.rdsamp(record["path"])
                sig = signal.T.astype(np.float32)              # (12, n_samples)
                return _fit_length(sig, ECG_LENGTH)
            except Exception as e:
                print(f"[PTB-XL] Failed to read record {record.get('ecg_id')}: {e} — using synthetic fallback")
        return _synthetic_signal(record.get("primary_superclass", "NORM"))

    def get_random_sample(self) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
        """Returns (signal_tensor (12, ECG_LENGTH), primary_superclass_label, record_meta)."""
        if self.records:
            record = self.records[np.random.randint(len(self.records))]
            sig = self.load_signal(record)
            return torch.from_numpy(sig), record["primary_superclass"], record
        label = SUPERCLASSES[np.random.randint(len(SUPERCLASSES))]
        sig = _synthetic_signal(label)
        return torch.from_numpy(sig), label, {"ecg_id": None, "superclasses": [label], "mode": "synthetic"}

    def __len__(self):
        return len(self.records)

    # ------------------------------------------------------------------
    # Training-set construction (patient-independent)
    # ------------------------------------------------------------------
    def sample_records(self, n: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Return a class-stratified subsample of `n` indexed records (default: all).
        Sampling is proportional to each primary-superclass frequency so the
        label distribution is preserved.
        """
        if not self.records:
            return []
        if n is None or n >= len(self.records):
            return list(self.records)
        rng = np.random.RandomState(seed)
        by_cls: Dict[str, List[int]] = {}
        for i, r in enumerate(self.records):
            by_cls.setdefault(r["primary_superclass"], []).append(i)
        total = len(self.records)
        chosen: List[int] = []
        for cls, idxs in by_cls.items():
            k = max(1, int(round(n * len(idxs) / total)))
            k = min(k, len(idxs))
            chosen.extend(rng.choice(idxs, size=k, replace=False).tolist())
        rng.shuffle(chosen)
        return [self.records[i] for i in chosen[:n]]

    def split_patient_independent(
        self, records: List[Dict[str, Any]], seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split records into (train, val, test) with NO patient overlap.

        Preferred: PTB-XL official 10-fold assignment (folds 1-8 train, 9 val,
        10 test) — folds are patient-disjoint by construction. If folds are
        missing, fall back to a GroupShuffle by patient_id so a patient never
        appears in two splits (still leakage-free).
        """
        have_folds = all(r.get("strat_fold") is not None for r in records)
        if have_folds:
            train = [r for r in records if int(r["strat_fold"]) <= 8]
            val   = [r for r in records if int(r["strat_fold"]) == 9]
            test  = [r for r in records if int(r["strat_fold"]) == 10]
            if train and val and test:
                return train, val, test

        # Fallback: group-by-patient random split (60/20/20 of patients)
        rng = np.random.RandomState(seed)
        pids = sorted({r.get("patient_id") for r in records if r.get("patient_id") is not None})
        if not pids:  # no patient ids at all -> record-level split (last resort)
            idx = rng.permutation(len(records))
            n_tr, n_va = int(0.6 * len(records)), int(0.8 * len(records))
            R = [records[i] for i in idx]
            return R[:n_tr], R[n_tr:n_va], R[n_va:]
        rng.shuffle(pids)
        n_tr, n_va = int(0.6 * len(pids)), int(0.8 * len(pids))
        tr_p, va_p, te_p = set(pids[:n_tr]), set(pids[n_tr:n_va]), set(pids[n_va:])
        train = [r for r in records if r.get("patient_id") in tr_p]
        val   = [r for r in records if r.get("patient_id") in va_p]
        test  = [r for r in records if r.get("patient_id") in te_p]
        return train, val, test

    def build_xy(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load waveforms for `records` into arrays:
            X : (N, 12, ECG_LENGTH) float32 real 12-lead signals
            y : (N,) int64 primary-superclass index (into SUPERCLASSES)
        Uses the wfdb decoder; a record that fails to read falls back to a
        class-conditioned synthetic signal (flagged in load_signal).
        """
        X = np.zeros((len(records), ECG_LEADS, ECG_LENGTH), dtype=np.float32)
        y = np.zeros((len(records),), dtype=np.int64)
        for i, rec in enumerate(records):
            X[i] = self.load_signal(rec)
            y[i] = SUPERCLASSES.index(rec["primary_superclass"])
            if (i + 1) % 500 == 0:
                print(f"[PTB-XL] Loaded {i + 1}/{len(records)} waveforms")
        return X, y

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    def get_eda_summary(self) -> Dict[str, Any]:
        if not self.records:
            return self._demo_eda()

        dist = {cls: 0 for cls in SUPERCLASSES}
        sex_dist: Dict[str, int] = {}
        ages: List[float] = []
        for rec in self.records:
            for cls in rec["superclasses"]:
                dist[cls] += 1
            sex_dist[rec["sex"]] = sex_dist.get(rec["sex"], 0) + 1
            if rec["age"] is not None:
                ages.append(rec["age"])

        return {
            "dataset": "PTB-XL Clinical 12-Lead ECG (PhysioNet)",
            "source": "https://physionet.org/content/ptb-xl/1.0.3/",
            "total_records": len(self.records),
            "sampling_rate_hz": self.sampling_rate,
            "leads": LEAD_NAMES,
            "signal_length_samples": ECG_LENGTH,
            "superclass_distribution": {
                cls: {"count": dist[cls], "name": SUPERCLASS_INFO[cls]["name"]}
                for cls in SUPERCLASSES
            },
            "sex_distribution": sex_dist,
            "age_mean": round(float(np.mean(ages)), 1) if ages else None,
            "real_signals_available": self.wfdb is not None,
            "note": None if self.wfdb is not None else (
                "wfdb package not installed — predictions will use synthetic fallback signals. "
                "pip install wfdb to enable real waveform decoding."
            ),
        }

    @staticmethod
    def _demo_eda() -> Dict[str, Any]:
        return {
            "dataset": "PTB-XL Clinical 12-Lead ECG (PhysioNet) — DEMO MODE",
            "source": "https://physionet.org/content/ptb-xl/1.0.3/",
            "total_records": PTBXL_REFERENCE_TOTAL,
            "total_patients": PTBXL_REFERENCE_PATIENTS,
            "sampling_rate_hz": TARGET_SAMPLING_RATE,
            "leads": LEAD_NAMES,
            "signal_length_samples": ECG_LENGTH,
            "superclass_distribution": {
                cls: {"count": PTBXL_REFERENCE_COUNTS[cls], "name": SUPERCLASS_INFO[cls]["name"]}
                for cls in SUPERCLASSES
            },
            "real_signals_available": False,
            "note": "Reference stats from PTB-XL documentation. "
                    "Download the dataset (Kaggle mirror or PhysioNet) into data/ptbxl/ for real analysis.",
        }


def build_synthetic_dataset(n: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a small SYNTHETIC PTB-XL-shaped dataset for smoke-testing the training
    loop on machines without the real download. NOT for reporting results —
    signals come from `_synthetic_signal` (class-conditioned sine/noise), so any
    metric on this data is meaningless. Real runs use PTBXLDataset.build_xy().
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((n, ECG_LEADS, ECG_LENGTH), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        cls = rng.randint(len(SUPERCLASSES))
        X[i] = _synthetic_signal(SUPERCLASSES[cls])
        y[i] = cls
    return X, y


def _safe_num(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _fit_length(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Resamples (linear interp) or pads/truncates a (leads, n) signal to (leads, target_len)."""
    n = sig.shape[1]
    if n == target_len:
        return sig
    if n == 0:
        return np.zeros((sig.shape[0], target_len), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_len)
    resampled = np.stack([np.interp(x_new, x_old, sig[lead]) for lead in range(sig.shape[0])])
    return resampled.astype(np.float32)


def _synthetic_signal(superclass: str) -> np.ndarray:
    """Generates a plausible-looking 12-lead waveform tagged with mild class-conditioned variation."""
    t = np.linspace(0, 10.0, ECG_LENGTH, endpoint=False)
    hr = {"NORM": 70, "MI": 78, "STTC": 74, "CD": 60, "HYP": 82}.get(superclass, 70)
    beat_freq = hr / 60.0
    base = np.sin(2 * np.pi * beat_freq * t) * 0.6
    base += 0.15 * np.sin(2 * np.pi * beat_freq * 5 * t)            # QRS-ish higher harmonic
    noise_scale = {"NORM": 0.05, "MI": 0.12, "STTC": 0.10, "CD": 0.09, "HYP": 0.08}.get(superclass, 0.05)
    sig = np.stack([
        base * (0.7 + 0.3 * np.cos(lead_idx)) + np.random.randn(ECG_LENGTH) * noise_scale
        for lead_idx in range(ECG_LEADS)
    ]).astype(np.float32)
    return sig


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_dataset_singleton: Optional[PTBXLDataset] = None


def get_ptbxl_dataset() -> PTBXLDataset:
    global _dataset_singleton
    if _dataset_singleton is None:
        _dataset_singleton = PTBXLDataset()
    return _dataset_singleton
