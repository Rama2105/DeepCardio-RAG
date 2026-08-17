"""
core/train_cardiofusion.py — Genuine training for the CardioFusion hybrid model
===============================================================================
Honest scope (stated plainly for the RR review)
------------------------------------------------
CardioFusion is designed to fuse 4 modalities (echo video, ECG image, heart
sound, ECG signal) with a cross-modal CONTRASTIVE alignment loss. That loss —
and any genuine "multi-modal fusion" claim — requires SAME-PATIENT PAIRED data
across all modalities, which NO public benchmark provides (EchoNet, CirCor,
MIT-BIH, VFDB are each single-modality with disjoint patients). We therefore do
NOT fabricate a jointly-trained 4-modality result.

What this trainer does genuinely:
  - Trains the CardioFusion ECG-signal path (encoder -> shared projection ->
    cross-modal Transformer with a single modality token -> MMoE -> arrhythmia
    head) on the REAL MIT-BIH benchmark (5 AAMI classes).
  - Uses the LEARNED homoscedastic uncertainty weighting (Kendall et al., CVPR
    2018) — the fusion's task weights are trained, not hand-set (reviewer Q6).
  - Reports HELD-OUT MIT-BIH test metrics only, and saves genuine weights to
    data/cardiofusion_weights.pt (loaded by get_cardiofusion_model()).

MIT-BIH beats are 187 samples, single lead. The CardioFusion ECG-signal encoder
(a VFDB-style 1D-CNN) needs a longer 2-lead segment, so each beat is linearly
resampled to 512 samples and the single lead is duplicated to 2 channels. This
is a documented preprocessing adaptation, not a change to the reported labels.
"""
import os
import time
import json
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from core.hybrid_model import CardioFusion, CardioFusionTrainer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
WEIGHTS_PATH = os.path.join(DATA_DIR, "cardiofusion_weights.pt")
METRICS_PATH = os.path.join(DATA_DIR, "cardiofusion_metrics.json")
SEG_LEN = 512
NUM_AAMI = 5


def _beats_to_ecg_signal(X187: np.ndarray) -> np.ndarray:
    """(N,187) single-lead beats -> (N, 2, 512) two-lead resampled segments."""
    n = X187.shape[0]
    x_old = np.linspace(0.0, 1.0, X187.shape[1])
    x_new = np.linspace(0.0, 1.0, SEG_LEN)
    out = np.zeros((n, 2, SEG_LEN), dtype=np.float32)
    for i in range(n):
        r = np.interp(x_new, x_old, X187[i]).astype(np.float32)
        out[i, 0] = r
        out[i, 1] = r  # duplicate lead (documented adaptation)
    return out


def _load_mitbih(n_records: int, seed: int):
    import pandas as pd
    train_path = os.path.join(DATA_DIR, "mitbih_train.csv")
    test_path = os.path.join(DATA_DIR, "mitbih_test.csv")
    if not os.path.exists(train_path):
        raise RuntimeError(f"MIT-BIH not found at {train_path}. It should be in data/ on Drive.")
    tr = pd.read_csv(train_path, header=None)
    te = pd.read_csv(test_path, header=None) if os.path.exists(test_path) else None
    if te is None:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(tr, test_size=0.2, random_state=seed, stratify=tr.iloc[:, -1])
    label_col = tr.columns[-1]
    rng = np.random.RandomState(seed)
    if n_records and len(tr) > n_records:  # class-stratified cap
        parts = []
        for _, g in tr.groupby(label_col):
            k = min(len(g), max(1, int(round(n_records * len(g) / len(tr)))))
            parts.append(g.iloc[rng.choice(len(g), k, replace=False)])
        tr = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    Xtr = _beats_to_ecg_signal(tr.iloc[:, :187].values.astype(np.float32))
    ytr = tr.iloc[:, -1].values.astype(np.int64)
    Xte = _beats_to_ecg_signal(te.iloc[:, :187].values.astype(np.float32))
    yte = te.iloc[:, -1].values.astype(np.int64)
    return Xtr, ytr, Xte, yte


@torch.no_grad()
def _evaluate(model, X, y, device, batch_size=256):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        out = model({"ecg_signal": xb})
        probs.append(torch.softmax(out["arrhythmia_logits"], dim=-1).cpu().numpy())
    probs = np.concatenate(probs)
    preds = probs.argmax(axis=1)
    try:
        auc = roc_auc_score(np.eye(NUM_AAMI)[y], probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"accuracy": float(accuracy_score(y, preds)),
            "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
            "auc_macro_ovr": float(auc)}


def train_cardiofusion(n_records: int = 3000, epochs: int = 20, batch_size: int = 128,
                       lr: float = 1e-3, seed: int = 42, synthetic: bool = False) -> dict:
    """Train the CardioFusion ECG-signal path on real MIT-BIH; save genuine weights."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if synthetic:
        print("[CardioFusion] SYNTHETIC smoke test — metrics NOT meaningful.")
        Xtr = np.random.randn(min(n_records, 200), 2, SEG_LEN).astype(np.float32)
        ytr = np.random.randint(0, NUM_AAMI, len(Xtr))
        Xte, yte = Xtr[:50], ytr[:50]
        data_desc = "SYNTHETIC (smoke test)"
    else:
        Xtr, ytr, Xte, yte = _load_mitbih(n_records, seed)
        data_desc = "MIT-BIH Arrhythmia (5 AAMI classes), ECG-signal modality"

    # Carve a stratified VALIDATION split out of TRAIN for model selection.
    # The held-out TEST set is NEVER used to pick a checkpoint — using it for
    # that (as the previous version did) is test-set peeking / selection bias.
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
    print(f"[CardioFusion] train {len(Xtr)} | val {len(Xval)} | test {len(Xte)} | "
          f"train-classes {np.bincount(ytr, minlength=NUM_AAMI).tolist()}")

    # Inverse-frequency class weights counter the heavy Normal-class imbalance
    # so the model does not collapse to predicting the majority class.
    counts = np.bincount(ytr, minlength=NUM_AAMI).astype(np.float64)
    class_weights = torch.tensor(
        (counts.sum() / (NUM_AAMI * np.clip(counts, 1.0, None))),
        dtype=torch.float32)
    print(f"[CardioFusion] class weights {np.round(class_weights.numpy(), 3).tolist()}")

    model = CardioFusion(freeze_encoders=False).to(device)
    trainer = CardioFusionTrainer(model, lr=lr, class_weights=class_weights)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, drop_last=False)

    best_f1, best_state = -1.0, None
    for ep in range(epochs):
        run = 0.0
        for xb, yb in loader:
            losses = trainer.train_step({"ecg_signal": xb.to(device)},
                                        {"arrhythmia_labels": yb.to(device)})
            run += losses["total"]
        val = _evaluate(model, Xval, yval, device, batch_size)   # selection on VAL only
        print(f"[CardioFusion] epoch {ep+1:02d}/{epochs} | loss {run/len(loader):.4f} | "
              f"val_f1 {val['f1_macro']:.4f} | val_acc {val['accuracy']:.4f}")
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = _evaluate(model, Xte, yte, device, batch_size)   # TEST evaluated once, after selection

    metrics = {
        "dataset": data_desc,
        "synthetic": synthetic,
        "modality_trained": "ecg_signal",
        "task": "arrhythmia (5 AAMI classes)",
        "n_train": int(len(Xtr)), "n_val": int(len(Xval)), "n_test": int(len(Xte)),
        "evaluation": "held-out MIT-BIH test set",
        "model_selection": "best macro-F1 on a stratified validation split "
                           "carved from train (test set never used for selection)",
        "val_f1_macro_best": round(float(best_f1), 4),
        "class_weighting": "inverse-frequency class weights on the arrhythmia CE loss",
        "test_accuracy": round(test["accuracy"], 4),
        "test_f1_macro": round(test["f1_macro"], 4),
        "test_auc_macro_ovr": round(test["auc_macro_ovr"], 4),
        "loss_weighting": "learned homoscedastic uncertainty (Kendall et al., CVPR 2018)",
        "learned_task_weights": trainer.loss_weighter.current_weights(),
        "multimodal_note": "Single-modality training. Joint 4-modality fusion + contrastive "
                           "alignment NOT trained — requires same-patient paired data absent "
                           "from public benchmarks.",
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "training_time_s": round(time.time() - t0, 1),
        "device": str(device),
    }
    if not synthetic:
        torch.save(model.state_dict(), WEIGHTS_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[CardioFusion] weights saved -> {WEIGHTS_PATH}")

    print("\n[CardioFusion] === GENUINE HELD-OUT TEST METRICS (ECG-signal / arrhythmia) ===")
    for k in ["test_accuracy", "test_f1_macro", "test_auc_macro_ovr"]:
        print(f"  {k:22s}: {metrics[k]}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    train_cardiofusion(n_records=a.n, epochs=a.epochs, synthetic=a.synthetic)
