"""
core/train_vfdb.py — Genuine training of the ventricular-arrhythmia (VFDB) module
================================================================================
Fixes the reviewer's single most serious finding (M1b): the VFDB module ran on
RANDOM weights, and the flagship demo issued "Activate code blue / Immediate
defibrillation" from that random network on a synthetic signal. This module
trains the `VentricularArrhythmiaDetector` on the REAL PhysioNet MIT-BIH
Malignant Ventricular Ectopy Database (VFDB) and reports genuine held-out metrics.

What this does honestly
-----------------------
- Real data:          PhysioNet VFDB (22 half-hour 2-lead recordings), via
                      core/vfdb_loader.py (no wfdb dependency).
- Task:               (a) binary Dangerous-vs-Normal, (b) 4-class rhythm family
                      {Normal, VT, VF/VFL, Other_Dangerous}.
- Segmentation:       fixed-length windows; each window labelled by the rhythm
                      annotation active at its start.
- Split:              PATIENT-INDEPENDENT — whole recordings go to exactly one of
                      train/val/test, so no recording appears in two splits.
- Imbalance:          class-weighted cross-entropy (dangerous events are rare).
- Model selection:    early stopping on validation dangerous-class F1.
- Genuine reporting:  held-out TEST binary F1/AUC and 4-class macro-F1. No random
                      weights, no synthetic-signal alert presented as a result.

Usage (Colab GPU or CPU — dataset is small)
-------------------------------------------
    from core.train_vfdb import train_vfdb
    metrics = train_vfdb(epochs=30)                 # real VFDB under data/vfdb/
    # Mechanics-only smoke test (synthetic signals — metrics meaningless):
    metrics = train_vfdb(epochs=2, synthetic=True)
"""
import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.vfdb_loader import (
    VFDBDataset, VentricularArrhythmiaDetector, DEFAULT_DATASET_DIR,
    DANGEROUS_RHYTHMS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "vfdb_detector.pt")
METRICS_SAVE_PATH = os.path.join(DATA_DIR, "vfdb_metrics.json")

RHYTHM_CLASSES = VentricularArrhythmiaDetector.RHYTHM_CLASSES  # 4
SEGMENT_SECONDS = 5
IN_CHANNELS = 2


def _rhythm_to_class(rhythm: str) -> int:
    """Map an annotation rhythm string to one of the 4 detector classes."""
    r = rhythm.upper().strip()
    if r in ("VF", "VFIB", "VFL"):
        return 2   # VF/VFL
    if r == "VT":
        return 1   # VT
    if r in ("ASYS", "HGEA", "VER"):
        return 3   # Other_Dangerous
    return 0       # Normal / background (incl. non-targeted rhythms)


def _active_rhythm(annotations, sample):
    """Rhythm in effect at `sample` = the last annotation at or before it."""
    current = "N"
    for a in annotations:
        if a["sample"] <= sample:
            current = a["rhythm"]
        else:
            break
    return current


def _segments_from_record(ds: VFDBDataset, index: int, seg_seconds: int):
    """Return (X [n,2,L], y_rhythm [n], y_bin [n]) for one recording."""
    signal, rec = ds.load_signal(index)          # (n_signals, n_samples)
    fs = rec["fs"] or 250
    L = int(seg_seconds * fs)
    if signal.shape[0] < IN_CHANNELS:            # pad to 2 leads if needed
        signal = np.repeat(signal, IN_CHANNELS, axis=0)[:IN_CHANNELS]
    signal = signal[:IN_CHANNELS]
    anns = sorted(rec["annotations"], key=lambda a: a["sample"])

    xs, yr, yb = [], [], []
    n_total = signal.shape[1]
    for start in range(0, n_total - L, L):
        seg = signal[:, start:start + L].astype(np.float32)
        for ch in range(seg.shape[0]):           # per-channel z-norm
            m, s = seg[ch].mean(), seg[ch].std()
            if s > 0:
                seg[ch] = (seg[ch] - m) / s
        rhythm = _active_rhythm(anns, start)
        cls = _rhythm_to_class(rhythm)
        xs.append(seg)
        yr.append(cls)
        yb.append(1 if rhythm.upper().strip() in DANGEROUS_RHYTHMS else 0)
    if not xs:
        return (np.empty((0, IN_CHANNELS, L), np.float32),
                np.empty((0,), np.int64), np.empty((0,), np.int64))
    return np.stack(xs), np.array(yr, np.int64), np.array(yb, np.int64)


def _build_synthetic(n, seg_len, seed):
    """Random 2-lead segments; class weakly encoded in amplitude — mechanics only."""
    rng = np.random.RandomState(seed)
    yr = rng.randint(0, len(RHYTHM_CLASSES), size=n)
    X = rng.randn(n, IN_CHANNELS, seg_len).astype(np.float32)
    for i in range(n):
        X[i] *= (1.0 + 0.5 * yr[i])
    yb = (yr > 0).astype(np.int64)
    return X, yr.astype(np.int64), yb


def _evaluate(model, X, yb, yr, device, batch_size=64):
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    model.eval()
    bin_probs, rhy_pred = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            out = model(xb)
            bin_probs.append(torch.softmax(out["binary_logits"], 1)[:, 1].cpu().numpy())
            rhy_pred.append(out["rhythm_logits"].argmax(1).cpu().numpy())
    bin_probs = np.concatenate(bin_probs)
    rhy_pred = np.concatenate(rhy_pred)
    bin_pred = (bin_probs > 0.5).astype(int)
    try:
        bin_auc = roc_auc_score(yb, bin_probs) if len(set(yb)) > 1 else float("nan")
    except Exception:
        bin_auc = float("nan")
    return {
        "binary_accuracy": float(accuracy_score(yb, bin_pred)),
        "binary_f1_dangerous": float(f1_score(yb, bin_pred, pos_label=1, zero_division=0)),
        "binary_auc": float(bin_auc),
        "rhythm_f1_macro": float(f1_score(yr, rhy_pred, average="macro", zero_division=0)),
    }


def train_vfdb(epochs: int = 30, batch_size: int = 64, lr: float = 3e-4,
               seg_seconds: int = SEGMENT_SECONDS, seed: int = 42,
               synthetic: bool = False, dataset_dir: str = DEFAULT_DATASET_DIR,
               save: bool = True) -> dict:
    """
    Train the VFDB ventricular-arrhythmia detector on genuine data and save
    held-out metrics. Returns a metrics dict (also -> data/vfdb_metrics.json).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- Data ----
    if synthetic:
        print("[VFDB] SYNTHETIC smoke-test mode — metrics are NOT meaningful.")
        seg_len = seg_seconds * 250
        Xtr, ytr_r, ytr_b = _build_synthetic(400, seg_len, seed)
        Xva, yva_r, yva_b = _build_synthetic(100, seg_len, seed + 1)
        Xte, yte_r, yte_b = _build_synthetic(150, seg_len, seed + 2)
        data_desc = "SYNTHETIC (smoke test)"
        split_desc = "random (synthetic)"
    else:
        ds = VFDBDataset(dataset_dir=dataset_dir)
        if len(ds) == 0:
            raise RuntimeError(
                "VFDB not found. Place PhysioNet VFDB (.dat/.hea/.atr) under "
                "data/vfdb/. For a mechanics-only smoke test call "
                "train_vfdb(synthetic=True).")
        # Patient-independent split by recording.
        rng = np.random.RandomState(seed)
        order = list(range(len(ds)))
        rng.shuffle(order)
        n_te = max(1, int(round(0.20 * len(order))))
        n_va = max(1, int(round(0.15 * len(order))))
        te_idx = set(order[:n_te])
        va_idx = set(order[n_te:n_te + n_va])
        tr_idx = set(order[n_te + n_va:])

        def gather(idxs):
            Xs, Yr, Yb = [], [], []
            for j in idxs:
                x, yr, yb = _segments_from_record(ds, j, seg_seconds)
                if len(x):
                    Xs.append(x); Yr.append(yr); Yb.append(yb)
            if not Xs:
                raise RuntimeError("No segments extracted — check VFDB annotations.")
            return np.concatenate(Xs), np.concatenate(Yr), np.concatenate(Yb)

        Xtr, ytr_r, ytr_b = gather(tr_idx)
        Xva, yva_r, yva_b = gather(va_idx)
        Xte, yte_r, yte_b = gather(te_idx)
        data_desc = (f"MIT-BIH VFDB (PhysioNet), {len(ds)} recordings, "
                     f"{seg_seconds}s windows; {len(Xtr)} train / {len(Xva)} val / "
                     f"{len(Xte)} test segments")
        split_desc = "patient-independent (whole recordings per split)"
        print(f"[VFDB] {data_desc}")
        print(f"[VFDB] train dangerous frac {ytr_b.mean():.3f} | test dangerous frac {yte_b.mean():.3f}")

        # A split with only one class makes every metric meaningless: the model
        # predicts the single present class and scores accuracy 1.0 with
        # F1(dangerous) 0.0 and AUC nan. That is exactly the "too good to be
        # true" pattern the review flagged, so refuse to produce it rather than
        # writing it into genuine_results.json.
        for name, yb in (("train", ytr_b), ("val", yva_b), ("test", yte_b)):
            if len(np.unique(yb)) < 2:
                raise RuntimeError(
                    f"VFDB {name} split contains a single class "
                    f"(dangerous fraction {yb.mean():.3f}) — rhythm annotations did not "
                    "parse. Every VFDB recording contains malignant ventricular ectopy, "
                    "so this indicates an annotation-reading failure, not the data. "
                    "Install `wfdb` (pip install wfdb) and re-run; metrics from a "
                    "single-class split (accuracy 1.0, AUC nan) are not reportable.")

    seg_len = Xtr.shape[-1]

    # ---- Model / optim ----
    model = VentricularArrhythmiaDetector(in_channels=IN_CHANNELS).to(device)
    b_counts = np.bincount(ytr_b, minlength=2).astype(float)
    r_counts = np.bincount(ytr_r, minlength=len(RHYTHM_CLASSES)).astype(float)
    b_w = torch.tensor(b_counts.sum() / (2 * np.maximum(b_counts, 1)), dtype=torch.float32).to(device)
    r_w = torch.tensor(r_counts.sum() / (len(RHYTHM_CLASSES) * np.maximum(r_counts, 1)),
                       dtype=torch.float32).to(device)
    bin_crit = nn.CrossEntropyLoss(weight=b_w)
    rhy_crit = nn.CrossEntropyLoss(weight=r_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr_b, dtype=torch.long),
                      torch.tensor(ytr_r, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)

    # ---- Train w/ early stopping on val dangerous-class F1 ----
    best_f1, best_state, patience, bad = -1.0, None, 8, 0
    for ep in range(epochs):
        model.train()
        run = 0.0
        for xb, ybb, ybr in loader:
            xb, ybb, ybr = xb.to(device), ybb.to(device), ybr.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = bin_crit(out["binary_logits"], ybb) + rhy_crit(out["rhythm_logits"], ybr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run += loss.item()
        scheduler.step()
        vm = _evaluate(model, Xva, yva_b, yva_r, device, batch_size)
        print(f"[VFDB] epoch {ep+1:02d}/{epochs} | loss {run/len(loader):.4f} | "
              f"val_F1(dang) {vm['binary_f1_dangerous']:.4f} | val_AUC {vm['binary_auc']:.4f}")
        if vm["binary_f1_dangerous"] > best_f1:
            best_f1 = vm["binary_f1_dangerous"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[VFDB] Early stop at epoch {ep+1} (best val F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tm = _evaluate(model, Xte, yte_b, yte_r, device, batch_size)

    metrics = {
        "dataset": data_desc,
        "synthetic": synthetic,
        "task": "binary Dangerous-vs-Normal + 4-class rhythm family",
        "rhythm_classes": RHYTHM_CLASSES,
        "segment_seconds": seg_seconds,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "split": split_desc,
        "evaluation": "held-out TEST set only (no train-set eval, no random weights)",
        "test_binary_accuracy": round(tm["binary_accuracy"], 4),
        "test_binary_f1_dangerous": round(tm["binary_f1_dangerous"], 4),
        "test_binary_auc": round(tm["binary_auc"], 4),
        "test_rhythm_f1_macro": round(tm["rhythm_f1_macro"], 4),
        "best_val_f1_dangerous": round(best_f1, 4),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "training_time_s": round(time.time() - t0, 1),
        "device": str(device),
        "class_note": "Non-targeted rhythms (AFIB, SBR, etc.) grouped into the Normal/"
                      "background class; the binary head is the safety-relevant output.",
    }

    if save:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        with open(METRICS_SAVE_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n[VFDB] === GENUINE HELD-OUT TEST METRICS ===")
    for k in ["test_binary_accuracy", "test_binary_f1_dangerous", "test_binary_auc",
              "test_rhythm_f1_macro"]:
        print(f"  {k:26s}: {metrics[k]}")
    if save:
        print(f"  weights saved -> {MODEL_SAVE_PATH}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    train_vfdb(epochs=a.epochs, synthetic=a.synthetic)
