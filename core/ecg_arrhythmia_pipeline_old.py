"""
ECG Arrhythmia Classification Pipeline (MIT-BIH / Kaggle)
==========================================================
Dataset: https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset
109,446 ECG heartbeat samples · 187 features · 5 AAMI classes
Derived from MIT-BIH Arrhythmia Database (PhysioNet)

Uses a 1D-CNN encoder with optional RAG enrichment for clinical report generation.
Train: 87,554 · Test: 21,892 · Pre-split CSV files
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score,
    confusion_matrix,
)

# ==============================================================================
# Dataset Configuration
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "mitbih_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "mitbih_test.csv")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "ecg_arrhythmia_cnn_model.pt")
SCALER_SAVE_PATH = os.path.join(DATA_DIR, "ecg_arrhythmia_scaler.pkl")

# MIT-BIH AAMI heartbeat classes
ARRHYTHMIA_CLASSES = {
    0: "Normal (N)",
    1: "Supraventricular Ectopic Beat (SVEB)",
    2: "Ventricular Ectopic Beat (VEB)",
    3: "Fusion Beat (F)",
    4: "Unknown / Paced (Q)",
}

CLASS_NAMES = list(ARRHYTHMIA_CLASSES.values())
NUM_CLASSES = 5
SIGNAL_LENGTH = 187  # 187 time-step features per heartbeat

# Clinical significance
CLASS_SEVERITY = {
    0: "benign",
    1: "moderate",
    2: "critical",
    3: "moderate",
    4: "investigate",
}

CLASS_CLINICAL_NOTES = {
    0: "Normal sinus rhythm. No intervention required.",
    1: "Supraventricular ectopic activity detected. Consider Holter monitoring if symptomatic (palpitations, dizziness).",
    2: "Ventricular ectopic beats detected. Potentially dangerous — evaluate for structural heart disease. May require antiarrhythmic therapy.",
    3: "Fusion beats present — simultaneous supraventricular and ventricular activation. Often benign but warrants monitoring.",
    4: "Unclassifiable or paced rhythm. Verify pacemaker status if applicable. Manual review recommended.",
}


# ==============================================================================
# Data Loader with EDA
# ==============================================================================
class ECGArrhythmiaDataLoader:
    def __init__(self):
        self.train_df = None
        self.test_df = None

    def load(self) -> tuple:
        if not os.path.exists(TRAIN_PATH):
            print(f"[ECGArrhythmia] Dataset not found — attempting auto-download...")
            self._auto_download()
        self.train_df = pd.read_csv(TRAIN_PATH, header=None)
        if os.path.exists(TEST_PATH):
            self.test_df = pd.read_csv(TEST_PATH, header=None)
        else:
            # Split train 80/20 if test file not available
            from sklearn.model_selection import train_test_split as tts
            train_part, test_part = tts(self.train_df, test_size=0.2, random_state=42,
                                        stratify=self.train_df.iloc[:, -1])
            self.test_df  = test_part.reset_index(drop=True)
            self.train_df = train_part.reset_index(drop=True)
            print("[ECGArrhythmia] Test file not found — using 80/20 split of train set")
        print(f"[ECGArrhythmia] Train: {len(self.train_df):,} | Test: {len(self.test_df):,} samples")
        return self.train_df, self.test_df

    def _auto_download(self):
        import sys
        sys.path.insert(0, os.path.dirname(DATA_DIR))
        try:
            from data.download_datasets import download_mitbih
            download_mitbih()
        except Exception as e:
            print(f"[ECGArrhythmia] Auto-download failed: {e}")
            raise FileNotFoundError(
                f"mitbih_train.csv not found at {TRAIN_PATH}.\n"
                "Run: python data/download_datasets.py --mitbih\n"
                "Or download from: https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset"
            )

    def get_eda_summary(self) -> dict:
        if self.train_df is None:
            self.load()
        train = self.train_df
        target_col = train.columns[-1]  # Last column is the label
        class_counts = train[target_col].value_counts().sort_index()

        summary = {
            "total_train_samples": int(len(train)),
            "total_test_samples": int(len(self.test_df)) if self.test_df is not None else 0,
            "total_samples": int(len(train)) + (int(len(self.test_df)) if self.test_df is not None else 0),
            "signal_length": SIGNAL_LENGTH,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "class_distribution": {},
            "class_imbalance": {},
            "signal_statistics": {},
        }

        # Class distribution
        for cls_id, cls_name in ARRHYTHMIA_CLASSES.items():
            count = int(class_counts.get(float(cls_id), 0))
            summary["class_distribution"][cls_name] = count
            summary["class_imbalance"][cls_name] = round(count / len(train) * 100, 2)

        # Signal amplitude stats (across first 187 columns)
        signal_data = train.iloc[:, :SIGNAL_LENGTH]
        summary["signal_statistics"] = {
            "mean_amplitude": round(float(signal_data.values.mean()), 6),
            "std_amplitude": round(float(signal_data.values.std()), 6),
            "min_amplitude": round(float(signal_data.values.min()), 6),
            "max_amplitude": round(float(signal_data.values.max()), 6),
            "mean_peak": round(float(signal_data.max(axis=1).mean()), 6),
        }

        # Test set distribution
        if self.test_df is not None:
            test_counts = self.test_df[self.test_df.columns[-1]].value_counts().sort_index()
            summary["test_class_distribution"] = {
                ARRHYTHMIA_CLASSES.get(int(k), f"Class {int(k)}"): int(v)
                for k, v in test_counts.items()
            }

        return summary

    def get_sample_beats(self, n: int = 5) -> dict:
        """Return sample heartbeat waveforms for visualization."""
        if self.train_df is None:
            self.load()
        samples = {}
        target_col = self.train_df.columns[-1]
        for cls_id, cls_name in ARRHYTHMIA_CLASSES.items():
            cls_data = self.train_df[self.train_df[target_col] == float(cls_id)]
            if len(cls_data) > 0:
                sample = cls_data.iloc[:min(n, len(cls_data)), :SIGNAL_LENGTH]
                samples[cls_name] = sample.values.tolist()
        return samples


# ==============================================================================
# 1D-CNN Encoder for ECG Heartbeat Classification
# ==============================================================================
class ECGArrhythmiaCNN(nn.Module):
    """
    1D Convolutional Neural Network for 5-class arrhythmia classification.
    Input: (batch_size, 1, 187) — single-lead heartbeat signal
    Output: (batch_size, 5) — class logits
    """

    def __init__(self, signal_length: int = SIGNAL_LENGTH, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.signal_length = signal_length

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Block 2: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            # Block 3: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 4: 128 → 256 channels
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch_size, 1, signal_length)
        features = self.features(x)  # (batch_size, 256, 1)
        features = features.squeeze(-1)  # (batch_size, 256)
        logits = self.classifier(features)  # (batch_size, num_classes)
        return logits

    def extract_embedding(self, x):
        """Extract 256-dim feature embedding (for RAG vector store)."""
        features = self.features(x)
        return features.squeeze(-1)


# ==============================================================================
# Training and Prediction Pipeline
# ==============================================================================
class ECGArrhythmiaPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_data(self, df: pd.DataFrame):
        """Split features and labels from raw CSV."""
        X = df.iloc[:, :SIGNAL_LENGTH].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None,
              num_epochs: int = 30, batch_size: int = 256) -> dict:
        start_time = time.time()

        X_train, y_train = self._prepare_data(train_df)

        # Scale signal amplitudes
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Reshape for Conv1d: (N, 1, 187)
        X_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Handle class imbalance with weighted loss
        class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.model = ECGArrhythmiaCNN().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Training loop
        self.model.train()
        train_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            train_losses.append(epoch_loss / len(train_loader))

        # Evaluate on test set (or train set if no test)
        eval_df = test_df if test_df is not None else train_df
        X_eval, y_eval = self._prepare_data(eval_df)
        X_eval_scaled = self.scaler.transform(X_eval)
        X_eval_t = torch.tensor(X_eval_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_eval_t), batch_size):
                batch = X_eval_t[i:i + batch_size]
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)

        accuracy = accuracy_score(y_eval, y_pred)
        f1_macro = f1_score(y_eval, y_pred, average="macro")
        f1_weighted = f1_score(y_eval, y_pred, average="weighted")

        try:
            auc_ovr = roc_auc_score(y_eval, y_probs, multi_class="ovr", average="macro")
        except:
            auc_ovr = 0.0

        cm = confusion_matrix(y_eval, y_pred, labels=list(range(NUM_CLASSES)))

        # Per-class metrics
        per_class = {}
        for cls_id, cls_name in ARRHYTHMIA_CLASSES.items():
            mask = y_eval == cls_id
            if mask.sum() > 0:
                cls_preds = y_pred[mask]
                per_class[cls_name] = {
                    "total": int(mask.sum()),
                    "correct": int((cls_preds == cls_id).sum()),
                    "accuracy": round(float((cls_preds == cls_id).mean()), 4),
                    "severity": CLASS_SEVERITY[cls_id],
                }

        training_time = time.time() - start_time
        self.is_trained = True
        self.metrics = {
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "auc_roc_ovr": round(float(auc_ovr), 4),
            "train_samples": int(len(X_train)),
            "eval_samples": int(len(X_eval)),
            "eval_set": "test" if test_df is not None else "train",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist(),
            "model_type": "1D-CNN (4-block, 256-dim embedding)",
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "signal_length": SIGNAL_LENGTH,
            "training_epochs": num_epochs,
            "training_time_seconds": round(training_time, 2),
            "final_train_loss": round(train_losses[-1], 6) if train_losses else None,
            "classification_report": classification_report(
                y_eval, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
            ),
        }

        # Save model and scaler
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        joblib.dump({"scaler": self.scaler}, SCALER_SAVE_PATH)

        return self.metrics

    def predict(self, signal: np.ndarray) -> dict:
        """
        Classify a single heartbeat signal.
        signal: 1D array of length 187 (or close) representing one heartbeat.
        """
        if not self.is_trained:
            if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
                saved = joblib.load(SCALER_SAVE_PATH)
                self.scaler = saved["scaler"]
                self.model = ECGArrhythmiaCNN().to(self.device)
                self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
                self.model.eval()
                self.is_trained = True
            else:
                raise RuntimeError("Model not trained. Call /api/ecg-arrhythmia/train first.")

        # Normalize signal length
        if len(signal) < SIGNAL_LENGTH:
            signal = np.pad(signal, (0, SIGNAL_LENGTH - len(signal)))
        elif len(signal) > SIGNAL_LENGTH:
            signal = signal[:SIGNAL_LENGTH]

        signal = signal.reshape(1, -1).astype(np.float32)
        signal_scaled = self.scaler.transform(signal)

        with torch.no_grad():
            x = torch.tensor(signal_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(logits.argmax(dim=-1).item())

        class_name = ARRHYTHMIA_CLASSES.get(pred_class, f"Class {pred_class}")
        confidence = float(probs[pred_class])
        severity = CLASS_SEVERITY.get(pred_class, "unknown")
        clinical_note = CLASS_CLINICAL_NOTES.get(pred_class, "Manual review recommended.")

        return {
            "predicted_class": pred_class,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "severity": severity,
            "probabilities": {
                ARRHYTHMIA_CLASSES[i]: round(float(probs[i]), 4)
                for i in range(NUM_CLASSES)
            },
            "clinical_note": clinical_note,
            "signal_length": len(signal.flatten()),
        }

    def predict_batch(self, signals: np.ndarray, batch_size: int = 256) -> list:
        """Classify a batch of heartbeat signals."""
        if not self.is_trained:
            self.predict(np.zeros(SIGNAL_LENGTH))  # trigger model load

        results = []
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i + batch_size]
            if batch.shape[1] < SIGNAL_LENGTH:
                batch = np.pad(batch, ((0, 0), (0, SIGNAL_LENGTH - batch.shape[1])))
            elif batch.shape[1] > SIGNAL_LENGTH:
                batch = batch[:, :SIGNAL_LENGTH]

            batch_scaled = self.scaler.transform(batch.astype(np.float32))

            with torch.no_grad():
                x = torch.tensor(batch_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()

            for j in range(len(preds)):
                pred_cls = int(preds[j])
                results.append({
                    "predicted_class": pred_cls,
                    "class_name": ARRHYTHMIA_CLASSES.get(pred_cls, f"Class {pred_cls}"),
                    "confidence": round(float(probs[j][pred_cls]), 4),
                    "severity": CLASS_SEVERITY.get(pred_cls, "unknown"),
                })
        return results


# ==============================================================================
# Global Lazy Init
# ==============================================================================
_loader_instance = None
_predictor_instance = None


def get_ecg_arrhythmia_loader() -> ECGArrhythmiaDataLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ECGArrhythmiaDataLoader()
        try:
            _loader_instance.load()
        except FileNotFoundError:
            pass  # Dataset not yet downloaded
    return _loader_instance


def get_ecg_arrhythmia_predictor() -> ECGArrhythmiaPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ECGArrhythmiaPredictor()
    return _predictor_instance
