"""
core/train_circor.py — Genuine training of the heart-sound (PCG) murmur module
==============================================================================
Fixes the reviewer's finding that the PCG module produced a "synthetic-signal
confidence only" and was marked Not-independently-trained in Table 14. This
module trains the `HeartSoundClassifier` on the REAL PhysioNet CirCor DigiScope
phonocardiogram dataset and reports genuine held-out metrics.

What this does honestly
-----------------------
- Real data:          PhysioNet CirCor DigiScope (5,272 PCG recordings, 1,568
                      subjects, 4 kHz), via core/heart_sound_loader.py.
- Task:               3-class murmur label {Absent, Present, Unknown}, plus the
                      clinically-relevant Present-vs-Absent binary AUC.
- Features:           log-mel spectrogram (numpy STFT, no librosa dependency).
- Split:              PATIENT-INDEPENDENT — each subject's recordings go to
                      exactly one split (a subject is auscultated at up to 4
                      locations; keeping them together prevents leakage).
- Imbalance:          class-weighted cross-entropy.
- Model selection:    early stopping on validation macro-F1.
- Genuine reporting:  held-out TEST macro-F1, per-class F1, and Present-vs-Absent
                      AUC. No synthetic-signal confidence presented as a result.

Usage (Colab GPU recommended; mel STFT is CPU-heavy)
----------------------------------------------------
    from core.train_circor import train_circor
    metrics = train_circor(n_max=3000, epochs=25)     # real CirCor
    # Mechanics-only smoke test (synthetic spectrograms — metrics meaningless):
    metrics = train_circor(epochs=2, synthetic=True)
"""
import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from core.heart_sound_loader import (
    HeartSoundDataset, HeartSoundClassifier, DEFAULT_DATASET_DIR,
    MURMUR_CLASSES, N_MELS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "heart_sound_classifier.pt")
METRICS_SAVE_PATH = os.path.join(DATA_DIR, "circor_metrics.json")

MEL_WIDTH = 160     # fixed spectrogram time width for batching


def _fix_mel_width(mel: np.ndarray, width: int = MEL_WIDTH) -> np.ndarray:
    """Pad/truncate a (n_mels, T) log-mel to a fixed T and standardise."""
    if mel.shape[1] >= width:
        mel = mel[:, :width]
    else:
        mel = np.pad(mel, ((0, 0), (0, width - mel.shape[1])), mode="edge")
    m, s = mel.mean(), mel.std()
    if s > 0:
        mel = (mel - m) / s
    return mel.astype(np.float32)


def _murmur_to_class(m: str) -> int:
    m = (m or "Unknown").strip().capitalize()
    return MURMUR_CLASSES.index(m) if m in MURMUR_CLASSES else MURMUR_CLASSES.index("Unknown")


# ---------------------------------------------------------------------------
# Lazy torch Dataset: reads wav + computes mel on demand
# ---------------------------------------------------------------------------
class _CirCorTorchDataset(Dataset):
    def __init__(self, ds: HeartSoundDataset, rec_indices):
        self.ds = ds
        self.rec_indices = rec_indices

    def __len__(self):
        return len(self.rec_indices)

    def __getitem__(self, i):
        idx = self.rec_indices[i]
        signal, rec = self.ds.load_audio(idx)
        mel = self.ds.compute_mel_spectrogram(signal)
        mel = _fix_mel_width(mel)
        x = torch.from_numpy(mel).unsqueeze(0)                 # (1, n_mels, T)
        y = torch.tensor(_murmur_to_class(rec.get("murmur")), dtype=torch.long)
        return x, y


class _SyntheticCirCor(Dataset):
    def __init__(self, n, seed):
        rng = np.random.RandomState(seed)
        self.y = rng.randint(0, len(MURMUR_CLASSES), size=n)
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 7919 + i)
        mel = rng.randn(N_MELS, MEL_WIDTH).astype(np.float32) + 0.7 * self.y[i]
        return torch.from_numpy(_fix_mel_width(mel)).unsqueeze(0), \
            torch.tensor(int(self.y[i]), dtype=torch.long)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _evaluate(model, loader, device):
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    model.eval()
    y_true, y_pred, probs_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out["logits"], 1).cpu().numpy()
            probs_all.append(p)
            y_pred.append(p.argmax(1))
            y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    probs = np.concatenate(probs_all)

    # Present-vs-Absent binary AUC (drop Unknown), the clinically useful signal.
    pa = np.isin(y_true, [MURMUR_CLASSES.index("Absent"), MURMUR_CLASSES.index("Present")])
    try:
        yb = (y_true[pa] == MURMUR_CLASSES.index("Present")).astype(int)
        sb = probs[pa][:, MURMUR_CLASSES.index("Present")]
        pa_auc = roc_auc_score(yb, sb) if len(set(yb)) > 1 else float("nan")
    except Exception:
        pa_auc = float("nan")

    per_class = f1_score(y_true, y_pred, average=None,
                         labels=list(range(len(MURMUR_CLASSES))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {MURMUR_CLASSES[j]: round(float(per_class[j]), 4)
                         for j in range(len(MURMUR_CLASSES))},
        "present_vs_absent_auc": float(pa_auc),
    }


def train_circor(n_max: int = 3000, epochs: int = 25, batch_size: int = 32,
                 lr: float = 3e-4, seed: int = 42, synthetic: bool = False,
                 dataset_dir: str = DEFAULT_DATASET_DIR, num_workers: int = 2,
                 save: bool = True) -> dict:
    """
    Train the CirCor murmur classifier on genuine data and save held-out metrics.
    Returns a metrics dict (also -> data/circor_metrics.json).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if synthetic:
        print("[CirCor] SYNTHETIC smoke-test mode — metrics are NOT meaningful.")
        train_loader = DataLoader(_SyntheticCirCor(400, seed), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(_SyntheticCirCor(100, seed + 1), batch_size=batch_size)
        test_loader = DataLoader(_SyntheticCirCor(150, seed + 2), batch_size=batch_size)
        n_tr, n_va, n_te = 400, 100, 150
        data_desc = "SYNTHETIC (smoke test)"
        split_desc = "random (synthetic)"
    else:
        ds = HeartSoundDataset(dataset_dir=dataset_dir)
        if len(ds) == 0:
            raise RuntimeError(
                "CirCor not found. Place PhysioNet CirCor DigiScope under "
                "data/circor-heart-sound/training_data/. For a mechanics-only "
                "smoke test call train_circor(synthetic=True).")
        # Patient-independent split by subject id.
        by_patient = {}
        for i, rec in enumerate(ds.recordings):
            by_patient.setdefault(rec["patient_id"], []).append(i)
        patients = sorted(by_patient.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(patients)
        n_te = max(1, int(round(0.20 * len(patients))))
        n_va = max(1, int(round(0.15 * len(patients))))
        te_p = set(patients[:n_te]); va_p = set(patients[n_te:n_te + n_va])
        tr_p = set(patients[n_te + n_va:])

        def recs_for(pset):
            out = []
            for p in pset:
                out.extend(by_patient[p])
            return out

        tr_idx, va_idx, te_idx = recs_for(tr_p), recs_for(va_p), recs_for(te_p)
        if n_max and len(tr_idx) > n_max:
            tr_idx = list(rng.choice(tr_idx, n_max, replace=False))
        train_loader = DataLoader(_CirCorTorchDataset(ds, tr_idx), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(_CirCorTorchDataset(ds, va_idx), batch_size=batch_size,
                                num_workers=num_workers)
        test_loader = DataLoader(_CirCorTorchDataset(ds, te_idx), batch_size=batch_size,
                                 num_workers=num_workers)
        n_tr, n_va, n_te = len(tr_idx), len(va_idx), len(te_idx)
        data_desc = (f"CirCor DigiScope (PhysioNet), {len(patients)} subjects; "
                     f"{n_tr} train / {n_va} val / {n_te} test recordings")
        split_desc = "patient-independent (subject-level)"
        print(f"[CirCor] {data_desc}")

        # class weights from the training split
        tr_labels = np.array([_murmur_to_class(ds.recordings[j].get("murmur")) for j in tr_idx])

    # ---- Model / optim ----
    model = HeartSoundClassifier(num_classes=len(MURMUR_CLASSES)).to(device)
    if synthetic:
        weights = torch.ones(len(MURMUR_CLASSES), device=device)
    else:
        counts = np.bincount(tr_labels, minlength=len(MURMUR_CLASSES)).astype(float)
        weights = torch.tensor(counts.sum() / (len(MURMUR_CLASSES) * np.maximum(counts, 1)),
                               dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Train w/ early stopping on val macro-F1 ----
    best_f1, best_state, patience, bad = -1.0, None, 6, 0
    for ep in range(epochs):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x)["logits"], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run += loss.item()
        scheduler.step()
        vm = _evaluate(model, val_loader, device)
        print(f"[CirCor] epoch {ep+1:02d}/{epochs} | loss {run/max(1,len(train_loader)):.4f} "
              f"| val_F1 {vm['f1_macro']:.4f} | PvA_AUC {vm['present_vs_absent_auc']:.4f}")
        if vm["f1_macro"] > best_f1:
            best_f1 = vm["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[CirCor] Early stop at epoch {ep+1} (best val_F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tm = _evaluate(model, test_loader, device)

    metrics = {
        "dataset": data_desc,
        "synthetic": synthetic,
        "task": "3-class murmur {Absent, Present, Unknown} + Present-vs-Absent AUC",
        "classes": MURMUR_CLASSES,
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "split": split_desc,
        "evaluation": "held-out TEST set only (no train-set eval)",
        "test_accuracy": round(tm["accuracy"], 4),
        "test_f1_macro": round(tm["f1_macro"], 4),
        "test_per_class_f1": tm["per_class_f1"],
        "test_present_vs_absent_auc": round(tm["present_vs_absent_auc"], 4),
        "best_val_f1_macro": round(best_f1, 4),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "training_time_s": round(time.time() - t0, 1),
        "device": str(device),
        "benchmark_note": "Oliveira et al. CirCor JBHI 2022 report ~0.79 weighted murmur "
                          "accuracy; this genuine run is reported without claiming that value.",
    }

    if save:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        with open(METRICS_SAVE_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n[CirCor] === GENUINE HELD-OUT TEST METRICS ===")
    for k in ["test_accuracy", "test_f1_macro", "test_present_vs_absent_auc"]:
        print(f"  {k:28s}: {metrics[k]}")
    print(f"  per-class F1: {metrics['test_per_class_f1']}")
    if save:
        print(f"  weights saved -> {MODEL_SAVE_PATH}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_max", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    train_circor(n_max=a.n_max, epochs=a.epochs, synthetic=a.synthetic)
