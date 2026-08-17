"""
Advanced Arthritis Profile Dataset (APD) Analysis Pipeline
Uses a PyTorch-based Tabular BERT (Transformer) with a Mixture of Experts (MoE) classifier for high accuracy.
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
DATASET_PATH = os.path.join(DATA_DIR, "APDDataset.xlsx")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "arthritis_bert_moe_model.pt")
SCALER_SAVE_PATH = os.path.join(DATA_DIR, "arthritis_bert_moe_scaler.pkl")

INFLAMMATORY_MARKERS = ["ESRh", "ESRo", "CRP", "ASO", "RA"]
HEMATOLOGY_MARKERS = ["TC", "Hb", "RBC", "PCV", "MCV", "MCH", "MCHC", "P", "L", "E", "PC"]
BIOCHEMISTRY_MARKERS = ["Urea", "Creatinine", "Calcium", "Uric_Acid", "RBS"]
DEMOGRAPHIC_FEATURES = ["Gender_M", "Age"]

# ==============================================================================
# Data Loader with EDA
# ==============================================================================
class ArthritisDataLoader:
    def __init__(self):
        self.df = None

    def load(self) -> pd.DataFrame:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")
        self.df = pd.read_excel(DATASET_PATH)
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])
        return self.df

    def get_eda_summary(self) -> dict:
        if self.df is None:
            self.load()
        df = self.df
        summary = {
            "total_samples": int(len(df)),
            "total_features": int(len(df.columns)),
            "feature_names": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "statistics": {},
            "gender_distribution": {
                "male": int(df["Gender_M"].sum()),
                "female": int(len(df) - df["Gender_M"].sum())
            },
            "age_stats": {
                "mean": round(float(df["Age"].mean()), 1) if not df["Age"].isna().all() else None,
                "min": round(float(df["Age"].min()), 1) if not df["Age"].isna().all() else None,
                "max": round(float(df["Age"].max()), 1) if not df["Age"].isna().all() else None,
            },
            "inflammatory_markers": {},
            "hematology_markers": {},
            "biochemistry_markers": {},
        }
        for col in df.select_dtypes(include=[np.number]).columns:
            desc = df[col].describe()
            summary["statistics"][col] = {
                "mean": round(float(desc["mean"]), 2) if not pd.isna(desc["mean"]) else None,
                "std": round(float(desc["std"]), 2) if not pd.isna(desc["std"]) else None,
                "min": round(float(desc["min"]), 2) if not pd.isna(desc["min"]) else None,
                "max": round(float(desc["max"]), 2) if not pd.isna(desc["max"]) else None,
            }
        for marker in INFLAMMATORY_MARKERS:
            if marker in df.columns:
                vals = df[marker].dropna()
                summary["inflammatory_markers"][marker] = {
                    "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                    "elevated_count": int((vals > vals.quantile(0.75)).sum()) if len(vals) > 0 else 0,
                    "available_samples": int(len(vals))
                }
        for marker in HEMATOLOGY_MARKERS:
            if marker in df.columns:
                vals = df[marker].dropna()
                summary["hematology_markers"][marker] = {
                    "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                    "available_samples": int(len(vals))
                }
        for marker in BIOCHEMISTRY_MARKERS:
            if marker in df.columns:
                vals = df[marker].dropna()
                summary["biochemistry_markers"][marker] = {
                    "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                    "available_samples": int(len(vals))
                }
        return summary

    def get_correlation_matrix(self) -> dict:
        if self.df is None:
            self.load()
        corr = self.df.select_dtypes(include=[np.number]).corr().round(3)
        return {"columns": corr.columns.tolist(), "data": corr.values.tolist()}


# ==============================================================================
# Advanced Feature Engineering
# ==============================================================================
class AdvancedFeatureEngineer:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "ESRh" in out.columns and "CRP" in out.columns:
            esr_norm = out["ESRh"].fillna(out["ESRh"].median()) / 40.0
            crp_norm = out["CRP"].fillna(out["CRP"].median()) / 10.0
            out["inflammation_index"] = (esr_norm + crp_norm) / 2.0
        if "ESRh" in out.columns and "ESRo" in out.columns:
            esh = out["ESRh"].fillna(1)
            eso = out["ESRo"].fillna(1).replace(0, 1)
            out["esr_ratio"] = esh / eso
        if "Hb" in out.columns and "RBC" in out.columns:
            out["anemia_score"] = (15.0 - out["Hb"].fillna(13.5)) * (5.0 - out["RBC"].fillna(4.5))
        if all(c in out.columns for c in ["MCV", "MCH", "MCHC"]):
            out["rbc_index"] = (
                out["MCV"].fillna(85) / 100.0 +
                out["MCH"].fillna(28) / 33.0 +
                out["MCHC"].fillna(33) / 36.0
            ) / 3.0
        if "Urea" in out.columns and "Creatinine" in out.columns:
            out["kidney_index"] = out["Urea"].fillna(25) / 40.0 + out["Creatinine"].fillna(0.9) / 1.2
        if "Uric_Acid" in out.columns:
            out["uric_elevated"] = (out["Uric_Acid"].fillna(5.5) > 7.0).astype(float)
        if "P" in out.columns and "L" in out.columns:
            p = out["P"].fillna(60)
            l = out["L"].fillna(35).replace(0, 1)
            out["nlr"] = p / l
        if "Age" in out.columns and "Gender_M" in out.columns:
            out["age_gender"] = out["Age"].fillna(45) * out["Gender_M"].fillna(0)
        if "Calcium" in out.columns and "Uric_Acid" in out.columns:
            ca = out["Calcium"].fillna(9.0)
            ua = out["Uric_Acid"].fillna(5.5).replace(0, 1)
            out["ca_ua_ratio"] = ca / ua
        return out


# ==============================================================================
# PyTorch Tabular BERT + MoE Architecture
# ==============================================================================
class TabularBERTMoE(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, num_experts: int = 4):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.num_experts = num_experts

        # Feature value projection (1 -> d_model)
        self.val_proj = nn.Linear(1, d_model)
        
        # Feature ID embeddings (num_features -> d_model)
        self.feat_emb = nn.Embedding(num_features, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Mixture of Experts
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_experts)
        ])

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_features)
        batch_size = x.size(0)
        
        # Reshape to (batch_size, num_features, 1) for projection
        x_val = x.unsqueeze(-1)
        val_emb = self.val_proj(x_val) # (batch_size, num_features, d_model)

        # Get feature ID embeddings
        feat_ids = torch.arange(self.num_features, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.feat_emb(feat_ids) # (batch_size, num_features, d_model)

        # Combine
        seq = val_emb + pos_emb

        # Transformer
        encoded_seq = self.transformer(seq) # (batch_size, num_features, d_model)

        # Mean pooling
        pooled = encoded_seq.mean(dim=1) # (batch_size, d_model)

        # MoE Routing
        route_logits = self.router(pooled) # (batch_size, num_experts)
        route_probs = torch.softmax(route_logits, dim=-1) # (batch_size, num_experts)

        # MoE Experts output
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1) # (batch_size, num_experts, d_model)
        
        # Weighted sum of experts
        moe_out = torch.sum(route_probs.unsqueeze(-1) * expert_outputs, dim=1) # (batch_size, d_model)

        # Classification
        logits = self.head(moe_out).squeeze(-1) # (batch_size,)
        return logits


# ==============================================================================
# Model Training and Prediction Pipeline
# ==============================================================================
class ArthritisPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_cols = []
        self.is_trained = False
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        ra_positive = df["RA"].fillna(0) > 14
        crp_positive = df["CRP"].fillna(0) > 6
        return (ra_positive | crp_positive).astype(int)

    def _train_pytorch_model(self, X_train, y_train, num_epochs=80, batch_size=16):
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = TabularBERTMoE(num_features=X_train.shape[1]).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def train(self, df: pd.DataFrame) -> dict:
        y = self._create_target(df).values

        exclude_cols = ["RA", "CRP"]
        base_cols = [c for c in df.columns if c not in exclude_cols and c != "Unnamed: 0"]
        X_base = df[base_cols].copy()

        X_eng = self.feature_engineer.transform(X_base)
        self.feature_cols = X_eng.columns.tolist()

        X_imputed = self.imputer.fit_transform(X_eng)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Train full model for saving
        self._train_pytorch_model(X_scaled, y, num_epochs=100)

        # K-Fold CV Evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # Train validation model
            val_model = TabularBERTMoE(num_features=X_scaled.shape[1]).to(self.device)
            optimizer = optim.AdamW(val_model.parameters(), lr=0.002, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            
            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_tr_t, y_tr_t)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
            
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
                val_preds = (torch.sigmoid(val_logits) > 0.5).int().cpu().numpy()
                cv_scores.append(accuracy_score(y_te, val_preds))

        # Overall Metrics on full dataset
        self.model.eval()
        with torch.no_grad():
            X_full = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            all_logits = self.model(X_full)
            all_probs = torch.sigmoid(all_logits).cpu().numpy()
            all_preds = (all_probs > 0.5).astype(int)

        accuracy = accuracy_score(y, all_preds)
        try:
            auc = roc_auc_score(y, all_probs)
        except:
            auc = 0.0

        self.is_trained = True
        self.metrics = {
            "accuracy": round(float(accuracy), 4),
            "cv_mean_accuracy": round(float(np.mean(cv_scores)), 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
            "auc_roc": round(float(auc), 4),
            "train_samples": int(len(X_scaled)),
            "test_samples": 0, # Evaluated whole for metrics here due to small dataset
            "positive_class_ratio": round(float(y.mean()), 4),
            "model_type": "Tabular BERT + Mixture of Experts (PyTorch)",
            "feature_engineering": "10 derived clinical features + Transformer self-attention",
            "imputation": "KNN (k=5)",
            "total_features": len(self.feature_cols),
            "top_features": [], # Complex to extact from BERT
            "classification_report": classification_report(y, all_preds, output_dict=True)
        }

        # Save weights and state
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        joblib.dump({
            "scaler": self.scaler, "imputer": self.imputer,
            "feature_cols": self.feature_cols
        }, SCALER_SAVE_PATH)

        return self.metrics

    def predict(self, patient_data: dict) -> dict:
        if not self.is_trained:
            if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
                saved = joblib.load(SCALER_SAVE_PATH)
                self.scaler = saved["scaler"]
                self.imputer = saved["imputer"]
                self.feature_cols = saved["feature_cols"]
                
                self.model = TabularBERTMoE(num_features=len(self.feature_cols)).to(self.device)
                self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
                self.model.eval()
                self.is_trained = True
            else:
                raise RuntimeError("Model not trained. Call /api/arthritis/train first.")

        row_df = pd.DataFrame([patient_data])
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
            
        prediction = 1 if prob > 0.5 else 0

        risk_level = "HIGH" if prediction == 1 else "LOW"
        confidence = prob if prediction == 1 else (1.0 - prob)

        interpretation = self._interpret(patient_data, risk_level, confidence)

        return {
            "risk_level": risk_level,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "low_risk": round(1.0 - prob, 4),
                "high_risk": round(prob, 4)
            },
            "clinical_interpretation": interpretation
        }

    def _interpret(self, data: dict, risk: str, conf: float) -> str:
        lines = [f"Arthritis Risk Assessment: {risk} (Confidence: {conf:.1%})", ""]
        esr = data.get("ESRh") or data.get("ESRo")
        if esr and esr > 20:
            lines.append(f"- ESR elevated at {esr} mm/hr (normal <20), suggesting active inflammation.")
        uric = data.get("Uric_Acid")
        if uric and uric > 7.0:
            lines.append(f"- Uric Acid elevated at {uric} mg/dL (normal <7.0), monitor for gout.")
        hb = data.get("Hb")
        if hb and hb < 12:
            lines.append(f"- Hemoglobin low at {hb} g/dL, anemia of chronic disease may be present.")
        calcium = data.get("Calcium")
        if calcium and calcium < 8.5:
            lines.append(f"- Calcium low at {calcium} mg/dL, consider bone metabolism evaluation.")
        aso = data.get("ASO")
        if aso and aso > 200:
            lines.append(f"- ASO elevated at {aso} IU/mL (normal <200), suggestive of recent streptococcal infection.")
        if risk == "HIGH":
            lines += ["", "RECOMMENDATION: Urgent referral to rheumatologist. Consider anti-CCP antibodies, joint imaging, and DMARD initiation per ACR/EULAR guidelines."]
        else:
            lines += ["", "RECOMMENDATION: Continue routine monitoring. Re-evaluate if joint pain, morning stiffness, or joint swelling develops."]
        return "\n".join(lines)


# ==============================================================================
# Global Lazy Init
# ==============================================================================
_loader_instance = None
_predictor_instance = None

def get_arthritis_loader() -> ArthritisDataLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ArthritisDataLoader()
        _loader_instance.load()
    return _loader_instance

def get_arthritis_predictor() -> ArthritisPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ArthritisPredictor()
    return _predictor_instance
