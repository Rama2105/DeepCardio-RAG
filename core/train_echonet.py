"""
core/train_echonet.py — Genuine training of the echocardiography LVEF module
============================================================================
Fixes the reviewer's M1 / Table-2-#1 objection: the echo module previously ran
on RANDOM weights over a SYNTHETIC tensor, yet the paper claimed "MAE 4.1% on the
held-out EchoNet test set" (a number it never measured) and produced a physically
impossible EF = -0.1% that propagated unchecked into the risk score.

This module trains the 3D-CNN EF regressor (`EchoNetModel`) on the REAL
EchoNet-Dynamic dataset and reports genuine held-out TEST metrics. Predicted EF
is clamped to the physiological range [0, 100]% (input-validity gating) so an
impossible value can never reach a downstream risk score again.

What this does honestly
-----------------------
- Real data:          Stanford EchoNet-Dynamic (apical-4-chamber AVI clips), via
                      core/echonet_loader.py.
- Task:               Left-ventricular ejection-fraction (LVEF) regression, plus
                      the derived 3-class HF category (HFrEF/HFmrEF/Normal).
- Split:              EchoNet's OWN official TRAIN/VAL/TEST split (patient-level,
                      as released in FileList.csv) — no leakage across splits.
- Model selection:    early stopping on validation MAE.
- Genuine reporting:  held-out TEST MAE, RMSE, R^2 for EF; macro-F1 and macro-AUC
                      for the HF category. No train-set eval, no fabricated MAE.

Usage (Colab GPU strongly recommended — 3D-CNN over video)
----------------------------------------------------------
    from core.train_echonet import train_echonet
    metrics = train_echonet(n_train=3000, epochs=25)     # real EchoNet-Dynamic
    # Mechanics-only smoke test on any CPU (synthetic clips — metrics meaningless):
    metrics = train_echonet(n_train=64, epochs=2, synthetic=True)
"""
import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from core.video_encoder import EchoNetModel
from core.echonet_loader import (
    EchoNetDataset, DEFAULT_DATASET_DIR,
    NUM_FRAMES, FRAME_SIZE, MEAN, STD,
    EF_REDUCED, EF_MIDRANGE,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "echonet_ef_model.pt")
METRICS_SAVE_PATH = os.path.join(DATA_DIR, "echonet_metrics.json")

EMBED_DIM = 384
EF_CATEGORIES = ["HFrEF", "HFmrEF", "Normal"]   # <40, 40-50, >=50


def _ef_to_class(ef: float) -> int:
    if ef < EF_REDUCED:
        return 0
    if ef < EF_MIDRANGE:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Lazy torch Dataset wrapping the AVI loader (videos read on demand)
# ---------------------------------------------------------------------------
class _EchoTorchDataset(Dataset):
    """Reads EchoNet AVI clips lazily and returns (video_tensor, ef)."""

    def __init__(self, split: str, dataset_dir: str, max_n: int | None, seed: int):
        self.ds = EchoNetDataset(dataset_dir=dataset_dir, split=split)
        idx = [i for i, s in enumerate(self.ds.samples) if s.get("ef") is not None]
        if max_n and len(idx) > max_n:
            rng = np.random.RandomState(seed)
            idx = list(rng.choice(idx, max_n, replace=False))
        self.indices = idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        video, sample = self.ds.load_video_tensor(self.indices[i])   # (1, T, H, W)
        return video, torch.tensor(float(sample["ef"]), dtype=torch.float32)


class _SyntheticEchoDataset(Dataset):
    """Random clips with EF weakly encoded in mean intensity — mechanics only."""

    def __init__(self, n: int, seed: int):
        self.n = n
        rng = np.random.RandomState(seed)
        self.efs = rng.uniform(10, 80, size=n).astype(np.float32)
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 100003 + i)
        base = (self.efs[i] - 45.0) / 100.0
        clip = rng.randn(1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE).astype(np.float32) + base
        clip = (clip - MEAN) / STD
        return torch.from_numpy(clip), torch.tensor(self.efs[i], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _evaluate(model, loader, device):
    from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                                 f1_score, roc_auc_score)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for video, ef in loader:
            video = video.to(device)
            out = model.encoder(video)
            pred = model.ef_head(out).squeeze(-1).cpu().numpy()
            # Physiological gating: EF is a percentage, clamp to [0, 100].
            pred = np.clip(pred, 0.0, 100.0)
            y_pred.append(pred)
            y_true.append(ef.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cls_true = np.array([_ef_to_class(v) for v in y_true])
    cls_pred = np.array([_ef_to_class(v) for v in y_pred])
    # 3-class OvR AUC from a soft score built off the EF distance to thresholds.
    try:
        # crude per-class score: closeness to each category's centre
        centres = np.array([30.0, 45.0, 65.0])
        dist = np.abs(y_pred[:, None] - centres[None, :])
        soft = np.exp(-dist / 10.0)
        soft = soft / soft.sum(axis=1, keepdims=True)
        macro_auc = roc_auc_score(np.eye(3)[cls_true], soft,
                                  multi_class="ovr", average="macro")
    except Exception:
        macro_auc = float("nan")

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "cat_f1_macro": float(f1_score(cls_true, cls_pred, average="macro", zero_division=0)),
        "cat_auc_macro_ovr": float(macro_auc),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_echonet(n_train: int = 3000, n_eval: int = 1000, epochs: int = 25,
                  batch_size: int = 8, lr: float = 1e-4, seed: int = 42,
                  synthetic: bool = False, dataset_dir: str = DEFAULT_DATASET_DIR,
                  num_workers: int = 2, save: bool = True) -> dict:
    """
    Train the EchoNet EF regressor and save genuine held-out TEST metrics.
    Returns a metrics dict (also written to data/echonet_metrics.json).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- Data ----
    if synthetic:
        print("[EchoNet] SYNTHETIC smoke-test mode — metrics are NOT meaningful.")
        train_ds = _SyntheticEchoDataset(int(n_train), seed)
        val_ds = _SyntheticEchoDataset(max(16, int(n_train * 0.15)), seed + 1)
        test_ds = _SyntheticEchoDataset(max(16, int(n_eval)), seed + 2)
        data_desc = "SYNTHETIC (smoke test)"
    else:
        train_ds = _EchoTorchDataset("TRAIN", dataset_dir, n_train, seed)
        val_ds = _EchoTorchDataset("VAL", dataset_dir, max(1, int(n_eval * 0.5)), seed)
        test_ds = _EchoTorchDataset("TEST", dataset_dir, n_eval, seed)
        if len(train_ds) == 0:
            raise RuntimeError(
                "EchoNet-Dynamic not found. Place it under data/EchoNet-Dynamic/ "
                "(Videos/ + FileList.csv) or request access from Stanford AIMI. "
                "For a mechanics-only smoke test call train_echonet(synthetic=True).")
        data_desc = (f"EchoNet-Dynamic (Stanford AIMI), "
                     f"{len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test clips")
        print(f"[EchoNet] {data_desc}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    # ---- Model / optim ----
    model = EchoNetModel(embed_dim=EMBED_DIM).to(device)
    criterion = nn.SmoothL1Loss()          # robust to EF outliers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Train w/ early stopping on val MAE ----
    best_mae, best_state, patience, bad = float("inf"), None, 6, 0
    for ep in range(epochs):
        model.train()
        run = 0.0
        for video, ef in train_loader:
            video, ef = video.to(device), ef.to(device)
            optimizer.zero_grad()
            pred = model.ef_head(model.encoder(video)).squeeze(-1)
            loss = criterion(pred, ef)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run += loss.item()
        scheduler.step()
        vm = _evaluate(model, val_loader, device)
        print(f"[EchoNet] epoch {ep+1:02d}/{epochs} | loss {run/max(1,len(train_loader)):.4f} "
              f"| val_MAE {vm['mae']:.3f} | val_R2 {vm['r2']:.3f}")
        if vm["mae"] < best_mae:
            best_mae = vm["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EchoNet] Early stop at epoch {ep+1} (best val_MAE={best_mae:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Genuine held-out TEST evaluation ----
    tm = _evaluate(model, test_loader, device)

    metrics = {
        "dataset": data_desc,
        "synthetic": synthetic,
        "task": "LVEF regression + 3-class HF category (HFrEF/HFmrEF/Normal)",
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "split": "EchoNet official TRAIN/VAL/TEST (patient-level)"
                 if not synthetic else "random (synthetic)",
        "evaluation": "held-out TEST set only (no train-set eval)",
        "ef_clamped_range": [0.0, 100.0],
        "test_mae": round(tm["mae"], 3),
        "test_rmse": round(tm["rmse"], 3),
        "test_r2": round(tm["r2"], 4),
        "test_cat_f1_macro": round(tm["cat_f1_macro"], 4),
        "test_cat_auc_macro_ovr": round(tm["cat_auc_macro_ovr"], 4),
        "best_val_mae": round(best_mae, 3),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "training_time_s": round(time.time() - t0, 1),
        "device": str(device),
        "benchmark_note": "Ouyang et al. Nature 2020 report MAE ~4.05 on full EchoNet; "
                          "this genuine subsample run will not match that and is not claimed to.",
    }

    if save:
        torch.save({"encoder": model.encoder.state_dict(),
                    "ef_head": model.ef_head.state_dict(),
                    "metrics": metrics}, MODEL_SAVE_PATH)
        with open(METRICS_SAVE_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n[EchoNet] === GENUINE HELD-OUT TEST METRICS ===")
    for k in ["test_mae", "test_rmse", "test_r2", "test_cat_f1_macro", "test_cat_auc_macro_ovr"]:
        print(f"  {k:24s}: {metrics[k]}")
    if save:
        print(f"  weights saved -> {MODEL_SAVE_PATH}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    train_echonet(n_train=a.n_train, n_eval=a.n_eval, epochs=a.epochs, synthetic=a.synthetic)
