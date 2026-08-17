"""
Heart Disease Prediction Pipeline (UCI Cleveland / Kaggle)
==========================================================
Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
1,025 samples · 13 features + binary target
Sources: Cleveland, Hungarian, Swiss, VA Long Beach databases

Uses TabularBERT + MoE architecture (shared with arthritis pipeline)
for consistency across the DeepCardio-RAG framework.
"""

import os
import numpy as np
import pandas as pd
import json
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ==============================================================================
# Dataset Configuration
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "heart.csv")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "heart_disease_bert_moe_model.pt")
SCALER_SAVE_PATH = os.path.join(DATA_DIR, "heart_disease_bert_moe_scaler.pkl")

# UCI Cleveland column schema
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"

# Written into the saved scaler bundle by train(). Its ABSENCE marks a checkpoint
# fitted before the polarity fix, i.e. on labels where 1 meant "no disease".
TARGET_POLARITY_CANONICAL = "1_is_disease"

# Clinical groupings
DEMOGRAPHIC_FEATURES = ["age", "sex"]
CARDIAC_FEATURES = ["cp", "trestbps", "thalach", "exang", "oldpeak", "slope"]
LAB_FEATURES = ["chol", "fbs"]
ANGIOGRAPHIC_FEATURES = ["ca", "thal", "restecg"]

# Feature descriptions (for EDA and reports)
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0–3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar > 120 mg/dL (1 = true)",
    "restecg": "Resting ECG results (0–2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0–2)",
    "ca": "Number of major vessels colored by fluoroscopy (0–3)",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)",
    "target": "Heart disease presence (1 = disease, 0 = no disease)",
}


# ==============================================================================
# Data Loader with EDA
# ==============================================================================
class HeartDiseaseDataLoader:
    def __init__(self):
        self.df = None

    def load(self) -> pd.DataFrame:
        if not os.path.exists(DATASET_PATH):
            print(f"[HeartDisease] Dataset not found — attempting auto-download...")
            self._auto_download()
        self.df = pd.read_csv(DATASET_PATH)
        # Standardize column names to lowercase
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        # Binarise target if values 0-4 (UCI original multi-class)
        if self.df[TARGET_COLUMN].max() > 1:
            self.df[TARGET_COLUMN] = (self.df[TARGET_COLUMN] > 0).astype(int)
        self._normalise_target_polarity()
        print(f"[HeartDisease] Loaded {len(self.df)} samples, {len(self.df.columns)} columns")
        return self.df

    def _normalise_target_polarity(self):
        """
        Force the canonical encoding 1 = disease, 0 = no disease.

        The Kaggle johnsmith88 mirror of UCI Cleveland ships the target INVERTED
        relative to this module's own documented meaning (see TARGET_COLUMN docs):
        measured on the shipped file, the target==0 group has the diseased profile
        (ca 1.16, oldpeak 1.60, exang 0.55, thalach 139) and target==1 has the
        healthy one (ca 0.37, oldpeak 0.57, exang 0.13, thalach 159). Training on
        it verbatim produced a model that called textbook high-risk patients LOW
        with p(disease)=0.0000 and textbook low-risk patients HIGH at 1.0000.

        Decided from clinical priors rather than hardcoded, so a corrected upstream
        file is left alone: diseased patients have more diseased vessels, greater
        exercise ST depression, more exercise angina and lower max heart rate.
        """
        df, t = self.df, TARGET_COLUMN
        needed = [c for c in ("ca", "oldpeak", "exang", "thalach") if c in df.columns]
        if t not in df.columns or len(needed) < 3 or df[t].nunique() < 2:
            return

        pos, neg = df[df[t] == 1], df[df[t] == 0]
        votes = 0
        for col, higher_means_disease in (("ca", True), ("oldpeak", True),
                                          ("exang", True), ("thalach", False)):
            if col not in df.columns:
                continue
            pos_greater = pos[col].mean() > neg[col].mean()
            # A vote for "inverted" whenever the label-1 group looks HEALTHIER.
            votes += 1 if pos_greater != higher_means_disease else -1

        if votes > 0:
            df[t] = 1 - df[t]
            self.target_was_inverted = True
            print("[HeartDisease] Target polarity INVERTED in source file "
                  "(1 meant 'no disease') — flipped to canonical 1 = disease.")
        else:
            self.target_was_inverted = False

    def _auto_download(self):
        """Attempt to auto-download via kagglehub, then kaggle API, then UCI URL."""
        import sys
        sys.path.insert(0, os.path.dirname(DATA_DIR))
        try:
            from data.download_datasets import download_heart_disease
            download_heart_disease()
        except Exception as e:
            print(f"[HeartDisease] Auto-download failed: {e}")
            raise FileNotFoundError(
                f"heart.csv not found at {DATASET_PATH}.\n"
                "Run: python data/download_datasets.py --heart\n"
                "Or download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
            )

    def get_eda_summary(self) -> dict:
        if self.df is None:
            self.load()
        df = self.df
        target_counts = df[TARGET_COLUMN].value_counts()
        summary = {
            "total_samples": int(len(df)),
            "total_features": len(FEATURE_COLUMNS),
            "feature_names": FEATURE_COLUMNS,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
            "missing_values": df[FEATURE_COLUMNS].isnull().sum().to_dict(),
            "missing_percentage": (df[FEATURE_COLUMNS].isnull().sum() / len(df) * 100).round(2).to_dict(),
            "class_distribution": {
                "no_disease": int(target_counts.get(0, 0)),
                "disease": int(target_counts.get(1, 0)),
                "disease_ratio": round(float(df[TARGET_COLUMN].mean()), 4),
            },
            "gender_distribution": {
                "male": int(df["sex"].sum()) if "sex" in df.columns else None,
                "female": int(len(df) - df["sex"].sum()) if "sex" in df.columns else None,
            },
            "age_stats": {
                "mean": round(float(df["age"].mean()), 1),
                "min": int(df["age"].min()),
                "max": int(df["age"].max()),
                "std": round(float(df["age"].std()), 1),
            },
            "statistics": {},
            "cardiac_features": {},
            "lab_features": {},
            "angiographic_features": {},
        }
        # Per-feature statistics
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                desc = df[col].describe()
                summary["statistics"][col] = {
                    "mean": round(float(desc["mean"]), 2),
                    "std": round(float(desc["std"]), 2),
                    "min": round(float(desc["min"]), 2),
                    "max": round(float(desc["max"]), 2),
                }
        # Grouped marker stats
        for marker in CARDIAC_FEATURES:
            if marker in df.columns:
                vals = df[marker].dropna()
                summary["cardiac_features"][marker] = {
                    "mean": round(float(vals.mean()), 2),
                    "available_samples": int(len(vals)),
                }
        for marker in LAB_FEATURES:
            if marker in df.columns:
                vals = df[marker].dropna()
                summary["lab_features"][marker] = {
                    "mean": round(float(vals.mean()), 2),
                    "available_samples": int(len(vals)),
                }
        for marker in ANGIOGRAPHIC_FEATURES:
            if marker in df.columns:
                vals = df[marker].dropna()
                summary["angiographic_features"][marker] = {
                    "mean": round(float(vals.mean()), 2),
                    "available_samples": int(len(vals)),
                }
        return summary

    def get_correlation_matrix(self) -> dict:
        if self.df is None:
            self.load()
        cols = FEATURE_COLUMNS + [TARGET_COLUMN]
        corr = self.df[cols].corr().round(3)
        return {"columns": corr.columns.tolist(), "data": corr.values.tolist()}

    def get_patients(self, page: int = 0, page_size: int = 20) -> dict:
        if self.df is None:
            self.load()
        start = page * page_size
        end = start + page_size
        subset = self.df.iloc[start:end]
        return {
            "total": int(len(self.df)),
            "page": page,
            "page_size": page_size,
            "records": subset.to_dict(orient="records"),
        }


# ==============================================================================
# Feature Engineering for Heart Disease
# ==============================================================================
class HeartDiseaseFeatureEngineer:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 1. Age-risk bucket (risk increases >55)
        if "age" in out.columns:
            out["age_risk"] = (out["age"].fillna(55) / 100.0) ** 2

        # 2. Cholesterol-to-thalach ratio (high chol + low HR → risk)
        if "chol" in out.columns and "thalach" in out.columns:
            thal = out["thalach"].fillna(150).replace(0, 1)
            out["chol_hr_ratio"] = out["chol"].fillna(240) / thal

        # 3. ST depression severity index
        if "oldpeak" in out.columns and "slope" in out.columns:
            out["st_severity"] = out["oldpeak"].fillna(0) * (out["slope"].fillna(1) + 1)

        # 4. Exercise capacity: thalach / (age + 1)
        if "thalach" in out.columns and "age" in out.columns:
            out["exercise_capacity"] = out["thalach"].fillna(150) / (out["age"].fillna(55) + 1)

        # 5. Resting BP risk (hypertension marker)
        if "trestbps" in out.columns:
            out["bp_elevated"] = (out["trestbps"].fillna(120) > 140).astype(float)

        # 6. Combined angina score
        if "exang" in out.columns and "cp" in out.columns:
            out["angina_score"] = out["exang"].fillna(0) * (4 - out["cp"].fillna(0))

        # 7. Vessel-thal interaction (angiographic severity)
        if "ca" in out.columns and "thal" in out.columns:
            out["vessel_thal"] = out["ca"].fillna(0) * out["thal"].fillna(2)

        # 8. Metabolic syndrome proxy (age + chol + bp + fbs)
        if all(c in out.columns for c in ["age", "chol", "trestbps", "fbs"]):
            out["metabolic_proxy"] = (
                out["age"].fillna(55) / 80.0 +
                out["chol"].fillna(240) / 400.0 +
                out["trestbps"].fillna(130) / 200.0 +
                out["fbs"].fillna(0)
            ) / 4.0

        return out


# ==============================================================================
# PyTorch Tabular BERT + MoE Architecture (adapted for heart disease)
# ==============================================================================
class HeartDiseaseBERTMoE(nn.Module):
    """Tabular BERT + Mixture-of-Experts for heart disease binary classification."""

    def __init__(self, num_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, num_experts: int = 4):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.num_experts = num_experts

        # Feature value projection (1 → d_model)
        self.val_proj = nn.Linear(1, d_model)

        # Feature ID embeddings (num_features → d_model)
        self.feat_emb = nn.Embedding(num_features, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=0.3, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Mixture of Experts
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(d_model, d_model),
            ) for _ in range(num_experts)
        ])

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_val = x.unsqueeze(-1)
        val_emb = self.val_proj(x_val)

        feat_ids = torch.arange(self.num_features, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.feat_emb(feat_ids)

        seq = val_emb + pos_emb
        encoded_seq = self.transformer(seq)
        pooled = encoded_seq.mean(dim=1)

        route_logits = self.router(pooled)
        route_probs = torch.softmax(route_logits, dim=-1)

        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        moe_out = torch.sum(route_probs.unsqueeze(-1) * expert_outputs, dim=1)

        logits = self.head(moe_out).squeeze(-1)
        return logits


# ==============================================================================
# Model Training and Prediction Pipeline
# ==============================================================================
class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_engineer = HeartDiseaseFeatureEngineer()
        self.feature_cols = []
        self.is_trained = False
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_pytorch_model(self, X_train, y_train, num_epochs=100, batch_size=32):
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = HeartDiseaseBERTMoE(num_features=X_train.shape[1]).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def train(self, df: pd.DataFrame) -> dict:
        import time
        start_time = time.time()

        y = df[TARGET_COLUMN].values.astype(float)
        X_base = df[FEATURE_COLUMNS].copy()

        # Feature engineering
        X_eng = self.feature_engineer.transform(X_base)
        self.feature_cols = X_eng.columns.tolist()

        # Impute + scale
        X_imputed = self.imputer.fit_transform(X_eng)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Train full model
        self._train_pytorch_model(X_scaled, y, num_epochs=100)

        # 5-Fold Stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        cv_aucs = []

        for train_idx, test_idx in skf.split(X_scaled, y):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            val_model = HeartDiseaseBERTMoE(num_features=X_scaled.shape[1]).to(self.device)
            optimizer = optim.AdamW(val_model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(self.device)
            val_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

            val_model.train()
            for epoch in range(80):
                for batch_x, batch_y in val_loader:
                    optimizer.zero_grad()
                    logits = val_model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            val_model.eval()
            with torch.no_grad():
                X_te_t = torch.tensor(X_te, dtype=torch.float32).to(self.device)
                val_logits = val_model(X_te_t)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_preds = (val_probs > 0.5).astype(int)
                cv_scores.append(accuracy_score(y_te, val_preds))
                try:
                    cv_aucs.append(roc_auc_score(y_te, val_probs))
                except:
                    pass

        # Full-dataset evaluation
        self.model.eval()
        with torch.no_grad():
            X_full = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            all_logits = self.model(X_full)
            all_probs = torch.sigmoid(all_logits).cpu().numpy()
            all_preds = (all_probs > 0.5).astype(int)

        accuracy = accuracy_score(y, all_preds)
        try:
            from sklearn.metrics import f1_score as _f1
            f1 = _f1(y, all_preds, average="macro", zero_division=0)
        except:
            f1 = 0.0
        try:
            auc = roc_auc_score(y, all_probs)
        except:
            auc = 0.0

        training_time = time.time() - start_time
        self.is_trained = True
        self.metrics = {
            "accuracy": round(float(accuracy), 4),
            "f1": round(float(f1), 4),
            "cv_mean_accuracy": round(float(np.mean(cv_scores)), 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
            "cv_mean_auc": round(float(np.mean(cv_aucs)), 4) if cv_aucs else 0.0,
            "auc_roc": round(float(auc), 4),
            "train_samples": int(len(X_scaled)),
            "positive_class_ratio": round(float(y.mean()), 4),
            "model_type": "Tabular BERT + Mixture of Experts (PyTorch)",
            "dataset": "UCI Heart Disease (Cleveland+Hungarian+Swiss+VA)",
            "feature_engineering": "8 derived clinical features + Transformer self-attention",
            "imputation": "KNN (k=5)",
            "total_features": len(self.feature_cols),
            "training_time_seconds": round(training_time, 2),
            "classification_report": classification_report(y, all_preds, output_dict=True),
        }

        # Save
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        joblib.dump({
            "scaler": self.scaler,
            "imputer": self.imputer,
            "feature_cols": self.feature_cols,
            # Stamp the label convention this model was fitted under. Checkpoints
            # WITHOUT this key predate _normalise_target_polarity() and were fitted
            # on the inverted Kaggle labels, so predict() must flip their output.
            "target_polarity": TARGET_POLARITY_CANONICAL,
        }, SCALER_SAVE_PATH)

        return self.metrics

    def predict(self, patient_data: dict) -> dict:
        if not self.is_trained:
            if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
                saved = joblib.load(SCALER_SAVE_PATH)
                self.scaler = saved["scaler"]
                self.imputer = saved["imputer"]
                self.feature_cols = saved["feature_cols"]
                # Legacy checkpoint (no polarity stamp) => fitted on inverted labels.
                self.legacy_inverted = saved.get("target_polarity") != TARGET_POLARITY_CANONICAL
                if self.legacy_inverted:
                    print("[HeartDisease] Loaded a PRE-POLARITY-FIX checkpoint — "
                          "flipping its output to canonical 1 = disease. "
                          "Retrain on Colab to remove the flip.")
                self.model = HeartDiseaseBERTMoE(num_features=len(self.feature_cols)).to(self.device)
                self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
                self.model.eval()
                self.is_trained = True
            else:
                raise RuntimeError("Model not trained. Call /api/heart-disease/train first.")

        row_df = pd.DataFrame([patient_data])
        row_df.columns = [c.strip().lower() for c in row_df.columns]
        row_eng = self.feature_engineer.transform(row_df)

        for col in self.feature_cols:
            if col not in row_eng.columns:
                row_eng[col] = np.nan
        row_eng = row_eng[self.feature_cols]

        X = row_eng.values
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logit = self.model(X_t)[0]
            prob = torch.sigmoid(logit).item()

        # This model was fitted when 1 meant "no disease"; its p(1) is therefore
        # p(no disease). Flip so every caller sees p(disease).
        if getattr(self, "legacy_inverted", False):
            prob = 1.0 - prob

        prediction = 1 if prob > 0.5 else 0
        risk_level = "HIGH" if prediction == 1 else "LOW"
        confidence = prob if prediction == 1 else (1.0 - prob)

        interpretation = self._interpret(patient_data, risk_level, confidence)

        return {
            "risk_level": risk_level,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "no_disease": round(1.0 - prob, 4),
                "disease": round(prob, 4),
            },
            "clinical_interpretation": interpretation,
        }

    def _interpret(self, data: dict, risk: str, conf: float) -> str:
        lines = [f"Heart Disease Risk Assessment: {risk} (Confidence: {conf:.1%})", ""]

        age = data.get("age")
        if age and age > 60:
            lines.append(f"- Age {int(age)}: advanced age is an independent cardiac risk factor.")

        chol = data.get("chol")
        if chol and chol > 240:
            lines.append(f"- Cholesterol {chol} mg/dL is elevated (desirable <200, high >240).")

        trestbps = data.get("trestbps")
        if trestbps and trestbps > 140:
            lines.append(f"- Resting BP {int(trestbps)} mm Hg indicates hypertension (>140).")

        thalach = data.get("thalach")
        if thalach and age:
            max_hr = 220 - age
            pct = thalach / max_hr * 100
            if pct < 85:
                lines.append(f"- Max HR {int(thalach)} is {pct:.0f}% of age-predicted max ({int(max_hr)}), suggesting chronotropic incompetence.")

        oldpeak = data.get("oldpeak")
        if oldpeak and oldpeak > 1.0:
            lines.append(f"- ST depression of {oldpeak} mm during exercise suggests myocardial ischaemia.")

        ca = data.get("ca")
        if ca and ca > 0:
            lines.append(f"- {int(ca)} major vessel(s) narrowed on fluoroscopy — coronary artery disease indicator.")

        exang = data.get("exang")
        if exang and exang == 1:
            lines.append("- Exercise-induced angina present — indicative of obstructive CAD.")

        if risk == "HIGH":
            lines += ["", "RECOMMENDATION: Cardiology referral recommended. Consider stress testing, coronary angiography, and initiation of statin/antiplatelet therapy per ACC/AHA guidelines."]
        else:
            lines += ["", "RECOMMENDATION: Continue cardiovascular risk factor management. Annual lipid panel and BP monitoring recommended."]

        return "\n".join(lines)


# ==============================================================================
# Global Lazy Init
# ==============================================================================
_loader_instance = None
_predictor_instance = None


def get_heart_disease_loader() -> HeartDiseaseDataLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = HeartDiseaseDataLoader()
        try:
            _loader_instance.load()
        except FileNotFoundError:
            pass  # Dataset not yet downloaded — EDA will fail gracefully
    return _loader_instance


def get_heart_disease_predictor() -> HeartDiseasePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HeartDiseasePredictor()
    return _predictor_instance


# ─────────────────────────────────────────────────────────────────────────────
# HeartDiseasePipeline — unified wrapper used by the Colab notebook / main.py
# Provides: load() → preprocess() → train() → evaluate()
# ─────────────────────────────────────────────────────────────────────────────
class HeartDiseasePipeline:
    """
    Thin facade over HeartDiseaseDataLoader + HeartDiseasePredictor.
    Exposes the load / preprocess / train / evaluate interface expected by
    the Colab notebook (Step 7) and the /api/arthritis/train endpoint.
    """

    def __init__(self):
        self.loader    = HeartDiseaseDataLoader()
        self.predictor = HeartDiseasePredictor()
        self.df        = None

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load heart.csv (auto-downloads if missing). Returns DataFrame."""
        self.df = self.loader.load()
        return self.df

    # ------------------------------------------------------------------
    def preprocess(self) -> pd.DataFrame:
        """
        Feature engineering is handled inside HeartDiseasePredictor.train(),
        so this is a lightweight validation step only.
        """
        if self.df is None:
            raise RuntimeError("Call load() before preprocess()")
        # Binarise target if dataset has 0-4 scale
        if self.df[TARGET_COLUMN].max() > 1:
            self.df = self.df.copy()
            self.df[TARGET_COLUMN] = (self.df[TARGET_COLUMN] > 0).astype(int)
        return self.df

    # ------------------------------------------------------------------
    def train(self) -> dict:
        """Train TabularBERT + MoE on the loaded DataFrame. Returns metrics dict."""
        if self.df is None:
            raise RuntimeError("Call load() before train()")
        return self.predictor.train(self.df)

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        """
        Re-run inference on the full dataset and return y_true / y_pred / y_prob
        for use with MetricsCalculator.
        """
        if not self.predictor.is_trained:
            self.train()

        import torch
        from sklearn.preprocessing import StandardScaler

        fe     = HeartDiseaseFeatureEngineer()
        X_eng  = fe.transform(self.df[FEATURE_COLUMNS].copy())
        X_imp  = self.predictor.imputer.transform(X_eng)
        X_sc   = self.predictor.scaler.transform(X_imp)
        y_true = self.df[TARGET_COLUMN].values.astype(int).tolist()

        self.predictor.model.eval()
        with torch.no_grad():
            X_t    = torch.tensor(X_sc, dtype=torch.float32).to(self.predictor.device)
            logits = self.predictor.model(X_t)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            preds  = (probs > 0.5).astype(int).tolist()

        return {
            "y_true": y_true,
            "y_pred": preds,
            "y_prob": [[1 - p, p] for p in probs],   # shape (N, 2) for MetricsCalculator
        }
