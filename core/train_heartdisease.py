"""
core/train_heartdisease.py — Genuine training of the heart-disease module
=========================================================================
Fixes the reviewer's M5 + Table-2-#5 findings:
  - M5: the 1,025-row Kaggle "heart.csv" is the 303-record UCI Cleveland cohort
        expanded with 700+ DUPLICATE rows; identical rows land on both sides of
        any random split, inflating metrics. -> we DEDUPLICATE to the unique rows.
  - Table-2-#5: the paper reported "Confidence 99.9%" from a FULL-DATASET
        (train-set) evaluation. -> we report leakage-free repeated-CV mean±95% CI
        on the deduplicated data, with NO train-set evaluation.

It also answers reviewer R8 (arXiv:2409.12116): a five-model / BERT-MoE stack must
justify its complexity against plain baselines, so we report plain Logistic
Regression and Gradient Boosting alongside the TabularBERT-MoE on the same folds.

Usage (CPU is fine — deduplicated cohort is ~300 records)
---------------------------------------------------------
    from core.train_heartdisease import train_heartdisease
    metrics = train_heartdisease(n_repeats=5, n_splits=3)
"""
import os
import time
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss

from core.heart_disease_pipeline import (
    HeartDiseaseDataLoader, HeartDiseaseFeatureEngineer, HeartDiseaseBERTMoE,
    FEATURE_COLUMNS, TARGET_COLUMN, DATA_DIR,
)

MODEL_SAVE_PATH = os.path.join(DATA_DIR, "heart_disease_bert_moe_model.pt")
SCALER_SAVE_PATH = os.path.join(DATA_DIR, "heart_disease_bert_moe_scaler.pkl")
METRICS_SAVE_PATH = os.path.join(DATA_DIR, "heart_disease_metrics.json")


def _ci95(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return {"mean": None, "std": None, "ci95": [None, None]}
    m, s = float(a.mean()), float(a.std())
    half = 1.96 * s / np.sqrt(len(a))
    return {"mean": round(m, 4), "std": round(s, 4),
            "ci95": [round(m - half, 4), round(m + half, 4)]}


def _train_bertmoe(Xtr, ytr, device, epochs=80, seed=42):
    torch.manual_seed(seed)
    model = HeartDiseaseBERTMoE(num_features=Xtr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                      torch.tensor(ytr, dtype=torch.float32)),
                        batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _bertmoe_prob(model, X, device):
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32).to(device)
    return torch.sigmoid(model(xt)).cpu().numpy()


def train_heartdisease(n_repeats: int = 5, n_splits: int = 3, epochs: int = 80,
                       seed: int = 42, save: bool = True) -> dict:
    """
    Deduplicate the Cleveland file, run leakage-free repeated-CV for the BERT-MoE
    plus Logistic and GradientBoosting baselines, and a single held-out split.
    Returns a metrics dict (also -> data/heart_disease_metrics.json).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    t0 = time.time()

    # ---- Load + DEDUPLICATE (M5) ----
    loader = HeartDiseaseDataLoader()
    df = loader.load()
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_unique = len(df)
    print(f"[HeartDisease] Deduplicated {n_raw} -> {n_unique} unique records "
          f"(UCI Cleveland origin is 303; Kaggle file duplicates them).")

    y = df[TARGET_COLUMN].values.astype(float)
    X_raw = df[FEATURE_COLUMNS].copy()
    fe = HeartDiseaseFeatureEngineer()

    # ---- Repeated Stratified CV with preproc INSIDE folds (no leakage) ----
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = {m: {"acc": [], "auc": [], "f1": [], "brier": []}
              for m in ["bert_moe", "logistic", "gbt"]}

    for tr_idx, te_idx in rskf.split(X_raw, y):
        Xtr_raw, Xte_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # fit feature engineering is stateless; impute+scale fit on TRAIN only
        Xtr_eng = fe.transform(Xtr_raw)
        Xte_eng = fe.transform(Xte_raw)
        imp = KNNImputer(n_neighbors=5)
        sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(Xtr_eng))
        Xte = sc.transform(imp.transform(Xte_eng))

        # BERT-MoE
        m = _train_bertmoe(Xtr, ytr, device, epochs=epochs, seed=seed)
        p = _bertmoe_prob(m, Xte, device)
        _acc(scores["bert_moe"], yte, p)

        # Logistic baseline
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xtr, ytr)
        _acc(scores["logistic"], yte, lr.predict_proba(Xte)[:, 1])

        # Gradient Boosting baseline
        gb = GradientBoostingClassifier(random_state=seed)
        gb.fit(Xtr, ytr)
        _acc(scores["gbt"], yte, gb.predict_proba(Xte)[:, 1])

    cv = {model: {k: _ci95(v) for k, v in d.items()} for model, d in scores.items()}

    # ---- Single held-out split (stratified 25%) for the BERT-MoE ----
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.25, stratify=y, random_state=seed)
    imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(fe.transform(Xtr_raw)))
    Xte = sc.transform(imp.transform(fe.transform(Xte_raw)))
    holdout_model = _train_bertmoe(Xtr, ytr, device, epochs=epochs, seed=seed)
    hp = _bertmoe_prob(holdout_model, Xte, device)
    holdout = {
        "accuracy": round(float(accuracy_score(yte, (hp > 0.5).astype(int))), 4),
        "auc_roc": round(float(roc_auc_score(yte, hp)), 4),
        "f1": round(float(f1_score(yte, (hp > 0.5).astype(int), average="macro", zero_division=0)), 4),
        "brier": round(float(brier_score_loss(yte, hp)), 4),
    }

    metrics = {
        "dataset": "UCI Cleveland Heart Disease (Detrano et al. 1989), Kaggle johnsmith88 mirror",
        "n_raw_rows": n_raw,
        "n_unique_records": n_unique,
        "dedup_note": "Kaggle file duplicates the 303-record Cleveland cohort; trained on unique rows only.",
        "num_features": len(fe.transform(X_raw.head(1)).columns),
        "cv_protocol": f"{n_repeats}x{n_splits} Repeated Stratified K-Fold (impute+scale inside folds)",
        "cv": cv,
        "holdout_25pct_bert_moe": holdout,
        "evaluation": "repeated-CV mean±95% CI (headline) + 25% held-out; NO train-set eval",
        "baseline_comparison_note": "Logistic and GBT reported on identical folds (reviewer R8): "
                                    "a BERT-MoE must beat plain baselines to justify its complexity.",
        "training_time_s": round(time.time() - t0, 1),
        "device": str(device),
    }

    if save:
        import joblib
        # retrain on ALL unique records for the deployed checkpoint
        imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
        X_all = sc.fit_transform(imp.fit_transform(fe.transform(X_raw)))
        final = _train_bertmoe(X_all, y, device, epochs=epochs, seed=seed)
        torch.save(final.state_dict(), MODEL_SAVE_PATH)
        joblib.dump({"scaler": sc, "imputer": imp,
                     "feature_cols": fe.transform(X_raw.head(1)).columns.tolist()},
                    SCALER_SAVE_PATH)
        with open(METRICS_SAVE_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    print("\n[HeartDisease] === GENUINE LEAKAGE-FREE RESULTS (deduplicated) ===")
    for model in ["bert_moe", "logistic", "gbt"]:
        a, u = cv[model]["acc"], cv[model]["auc"]
        print(f"  {model:9s}  CV acc {a['mean']} CI{a['ci95']} | CV AUC {u['mean']} CI{u['ci95']}")
    print(f"  held-out (BERT-MoE): {holdout}")
    if save:
        print(f"  weights saved -> {MODEL_SAVE_PATH}")
    return metrics


def _acc(bucket, y_true, prob):
    pred = (prob > 0.5).astype(int)
    bucket["acc"].append(accuracy_score(y_true, pred))
    bucket["f1"].append(f1_score(y_true, pred, average="macro", zero_division=0))
    bucket["brier"].append(brier_score_loss(y_true, prob))
    try:
        bucket["auc"].append(roc_auc_score(y_true, prob))
    except ValueError:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    train_heartdisease(n_repeats=a.n_repeats, n_splits=a.n_splits, epochs=a.epochs)
