"""
Advanced Arthritis Risk Analysis Pipeline
==========================================
Supports multiple real datasets:
  1. Cardiovascular Disease Dataset (Kaggle, 70,000 records) — PRIMARY
     Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
  2. BRFSS Health Indicators Dataset (Kaggle, 253,680 records) — ALTERNATE
     Source: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-behavioral-risk-factor-surveillance-system-brfss
  3. APD (Arthritis Profile Dataset, 102 records) — FALLBACK

Architecture: PyTorch Tabular BERT + Mixture of Experts (MoE) Classifier
"""

import os
import numpy as np
import pandas as pd
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# Dataset Configuration
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "APDDataset.xlsx")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "arthritis_bert_moe_model.pt")
SCALER_SAVE_PATH = os.path.join(DATA_DIR, "arthritis_bert_moe_scaler.pkl")

# Real dataset info with source links
# Priority order: arthritis_nih > brfss_full > cardiovascular
REAL_DATASETS = {
    "arthritis_nih": {
        "name": "NHANES Medical Conditions Questionnaire (CDC/Kaggle)",
        "kaggle_id": "cdc/national-health-and-nutrition-examination-survey",
        "url": "https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey",
        "records": 10175,
        "file": "questionnaire.csv",   # 953-column MCQ file confirmed present
        "alt_files": [],
        "description": "NHANES Medical Conditions Questionnaire — MCQ160A (doctor-diagnosed arthritis), MCQ195 (RA/OA type), MCQ160N, demographics",
        "separator": ",",
        "arthritis_col": "MCQ160A",    # 1=Yes, 2=No (confirmed uppercase)
        # questionnaire.csv has no demographic/exam fields — join them in from the
        # companion NHANES files (same Kaggle dataset) on the shared patient key SEQN
        "join_on": "SEQN",
        "join_files": [
            {"file": "demographic.csv", "columns": ["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "DMDEDUC2"]},
            {"file": "examination.csv", "columns": ["SEQN", "BMXBMI", "BPXSY1", "BPXDI1"]},
        ],
    },
    "brfss_full": {
        "name": "CDC BRFSS Full Survey (2011-2015, with arthritis indicator)",
        "kaggle_id": "cdc/behavioral-risk-factor-surveillance-system",
        "url": "https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system",
        "records": 441456,
        "file": "2015.csv",            # try latest year first
        "alt_files": ["2014.csv", "2013.csv", "2012.csv", "2011.csv"],
        "description": "441K+ CDC survey records — HAVARTH3 (doctor-diagnosed arthritis, confirmed present in all years 2011-2015)",
        "separator": ",",
        "arthritis_col": "HAVARTH3",   # 1=Yes, 2=No, 7/9=Don't know/Refused
    },
    "cardiovascular": {
        "name": "Cardiovascular Disease Dataset",
        "kaggle_id": "sulianova/cardiovascular-disease-dataset",
        "url": "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
        "records": 70000,
        "file": "cardio_train.csv",
        "alt_files": [],
        "description": "70,000 patient records — age, BP, cholesterol, glucose, BMI, lifestyle factors (metabolic risk proxy)",
        "separator": ";",
    },
}

INFLAMMATORY_MARKERS = ["ESRh", "ESRo", "CRP", "ASO", "RA"]
HEMATOLOGY_MARKERS   = ["TC", "Hb", "RBC", "PCV", "MCV", "MCH", "MCHC", "P", "L", "E", "PC"]
BIOCHEMISTRY_MARKERS = ["Urea", "Creatinine", "Calcium", "Uric_Acid", "RBS"]
DEMOGRAPHIC_FEATURES = ["Gender_M", "Age"]


# ==============================================================================
# Real Dataset Downloader
# ==============================================================================
class RealDatasetManager:
    """
    Downloads and maps real large-scale datasets to the arthritis prediction task.
    Tries Kaggle API via kagglehub, falls back to APD on failure.
    """

    @staticmethod
    def download_and_load(dataset_key: str = "cardiovascular") -> pd.DataFrame:
        """
        Download a real dataset from Kaggle and return a raw DataFrame.
        Returns None if download fails.
        """
        info = REAL_DATASETS.get(dataset_key)
        if not info:
            return None
        try:
            import kagglehub
            print(f"[Arthritis] Downloading {info['name']} from Kaggle...")
            path = kagglehub.dataset_download(info["kaggle_id"])

            # Try the primary file first, then any listed alt_files
            candidates = [info["file"]] + info.get("alt_files", [])
            csv_file = None
            for candidate in candidates:
                p = os.path.join(path, candidate)
                if os.path.exists(p):
                    csv_file = p
                    break
            # Walk subdirectories as fallback
            if csv_file is None:
                for root, _, files in os.walk(path):
                    for f in files:
                        if f.endswith(".csv"):
                            csv_file = os.path.join(root, f)
                            break
                    if csv_file:
                        break

            if not csv_file or not os.path.exists(csv_file):
                print(f"[Arthritis] No CSV found in downloaded {info['name']}")
                return None

            df = pd.read_csv(csv_file, sep=info.get("separator", ","), low_memory=False)
            print(f"[Arthritis] Downloaded {len(df):,} records from {info['name']} ({os.path.basename(csv_file)})")

            # Join supplementary files (e.g. NHANES demographic/examination data)
            # from the same Kaggle dataset on a shared patient key
            join_on = info.get("join_on")
            for jf in info.get("join_files", []):
                jf_path = os.path.join(path, jf["file"])
                if not os.path.exists(jf_path):
                    print(f"[Arthritis] Join file not found: {jf['file']} — skipping")
                    continue
                try:
                    join_df = pd.read_csv(jf_path, usecols=jf["columns"], low_memory=False)
                    df = df.merge(join_df, on=join_on, how="left")
                    print(f"[Arthritis] Joined {jf['file']} on {join_on} "
                          f"({len(jf['columns']) - 1} columns)")
                except Exception as e:
                    print(f"[Arthritis] Failed to join {jf['file']}: {e}")

            return df
        except Exception as e:
            print(f"[Arthritis] Dataset download failed ({dataset_key}): {e}")
            return None

    @staticmethod
    def map_cardiovascular_to_arthritis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Cardiovascular Disease Dataset features to arthritis risk prediction.
        Shared risk factors: age, BMI, BP, cholesterol, glucose, smoking, alcohol, activity.

        Cardiovascular -> Arthritis mapping:
          age (days) -> Age (years)
          gender (1=woman, 2=man) -> Gender_M (0/1)
          height (cm), weight (kg) -> BMI
          ap_hi -> Systolic BP (inflammation proxy)
          ap_lo -> Diastolic BP
          cholesterol (1-3) -> Cholesterol level
          gluc (1-3) -> Glucose/metabolic marker
          smoke -> Smoking (arthritis risk factor)
          alco -> Alcohol (uric acid/gout risk)
          active -> Physical activity (protective)
          cardio -> Target (cardiovascular/metabolic disease risk)
        """
        out = pd.DataFrame()
        # Demographics
        out["Age"] = (df["age"] / 365.25).round(1) if "age" in df.columns else 50.0
        out["Gender_M"] = (df["gender"] == 2).astype(float) if "gender" in df.columns else 0.0
        # BMI (derived from height/weight)
        if "height" in df.columns and "weight" in df.columns:
            h = df["height"] / 100.0  # cm to m
            out["BMI"] = (df["weight"] / (h ** 2)).round(1).clip(10, 60)
        # Blood pressure — inflammatory/vascular markers
        if "ap_hi" in df.columns:
            out["SystolicBP"] = df["ap_hi"].clip(60, 250)
        if "ap_lo" in df.columns:
            out["DiastolicBP"] = df["ap_lo"].clip(40, 150)
        # Metabolic markers (mapped to arthritis biomarkers)
        if "cholesterol" in df.columns:
            out["CholesterolLevel"] = df["cholesterol"].astype(float)  # 1=normal,2=above,3=well above
        if "gluc" in df.columns:
            out["GlucoseLevel"] = df["gluc"].astype(float)  # 1=normal,2=above,3=well above
        # Lifestyle risk factors
        if "smoke" in df.columns:
            out["Smoking"] = df["smoke"].astype(float)
        if "alco" in df.columns:
            out["AlcoholUse"] = df["alco"].astype(float)  # gout/uric acid risk
        if "active" in df.columns:
            out["PhysicallyActive"] = df["active"].astype(float)
        # Target: cardiovascular metabolic disease (surrogate for arthritis risk)
        if "cardio" in df.columns:
            out["arthritis_risk"] = df["cardio"].astype(int)
        # Derived inflammation index (proxy for ESR/CRP)
        if "cholesterol" in df.columns and "ap_hi" in df.columns:
            chol_norm = (df["cholesterol"] - 1) / 2.0  # 0-1
            bp_norm = ((df["ap_hi"].clip(60, 250) - 60) / 190.0)
            out["InflammationProxy"] = ((chol_norm + bp_norm) / 2.0).round(3)
        # Metabolic risk score
        if "BMI" in out.columns:
            out["MetabolicRisk"] = (out["BMI"] > 30).astype(float)
        return out

    @staticmethod
    def map_nhanes_to_arthritis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map NHANES Medical Conditions Questionnaire (questionnaire.csv) to arthritis prediction.

        Confirmed columns in the Kaggle NHANES dataset:
          MCQ160A  — Doctor-diagnosed arthritis (1=Yes, 2=No)
          MCQ195   — Arthritis type (1=osteoarthritis, 2=rheumatoid arthritis, 3=psoriatic, 4=other)
          MCQ160N  — Doctor-diagnosed gout (1=Yes, 2=No)
          MCQ160B  — Doctor-diagnosed osteoporosis
          RIDAGEYR — Age in years (joined in from demographic.csv via SEQN)
          RIAGENDR — Gender (1=Male, 2=Female; joined in from demographic.csv)
          INDFMPIR — Poverty income ratio (socioeconomic proxy; from demographic.csv)
          DMDEDUC2 — Education level (from demographic.csv)
          BMXBMI   — Body Mass Index (joined in from examination.csv via SEQN)
          BPXSY1/BPXDI1 — Systolic/diastolic BP (joined in from examination.csv)
        """
        out = pd.DataFrame()

        # Demographics — present if questionnaire.csv was joined with demographic
        for src, dst in [("RIDAGEYR", "Age"), ("RIAGENDR", "Gender_raw")]:
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce")
        if "Gender_raw" in out.columns:
            out["Gender_M"] = (out["Gender_raw"] == 1).astype(float)
            out = out.drop(columns=["Gender_raw"])

        # Socioeconomic / lifestyle
        for src, dst in [("INDFMPIR", "PovertyRatio"), ("DMDEDUC2", "EducationLevel"),
                         ("BMXBMI", "BMI"), ("BPXSY1", "SystolicBP"), ("BPXDI1", "DiastolicBP")]:
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce")
        if "BMI" in out.columns:
            out["BMI"] = out["BMI"].clip(10, 70)
        if "SystolicBP" in out.columns:
            out["SystolicBP"] = out["SystolicBP"].clip(60, 250)
        if "DiastolicBP" in out.columns:
            out["DiastolicBP"] = out["DiastolicBP"].clip(40, 150)

        # Musculoskeletal comorbidities (asked of ALL respondents — valid predictors).
        # NOTE: MCQ195 ("which type of arthritis?") is deliberately EXCLUDED — it is a
        # skip-pattern question asked ONLY of patients who already answered YES to
        # arthritis (MCQ160A, the target), so including it leaks the label and inflates
        # AUC to ~0.99. Dropping it is required for an honest generalisation estimate.
        for src, dst in [("MCQ160N", "HasGout"), ("MCQ160B", "HasOsteoporosis")]:
            if src in df.columns:
                raw = pd.to_numeric(df[src], errors="coerce")
                out[dst] = raw.map({1: 1, 2: 0})

        # Derived inflammation proxy (from available columns)
        if "HasGout" in out.columns and "HasOsteoporosis" in out.columns:
            out["InflammationProxy"] = (
                out["HasGout"].fillna(0) + out["HasOsteoporosis"].fillna(0)
            ) / 2.0

        # Target: MCQ160A — Doctor-diagnosed arthritis (confirmed uppercase in this dataset)
        # 1=Yes → 1, 2=No → 0; 7=Don't know / 9=Refused → NaN
        if "MCQ160A" not in df.columns:
            print("[Arthritis] NHANES: MCQ160A column not found in questionnaire.csv — skipping")
            return pd.DataFrame()

        raw = pd.to_numeric(df["MCQ160A"], errors="coerce")
        out["arthritis_risk"] = raw.map({1: 1, 2: 0})
        out = out.dropna(subset=["arthritis_risk"]).copy()
        out["arthritis_risk"] = out["arthritis_risk"].astype(int)
        print(f"[Arthritis] NHANES: {out['arthritis_risk'].sum():,} arthritis-positive / {len(out):,} total")
        return out

    @staticmethod
    def map_brfss_full_to_arthritis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map the full CDC BRFSS 2015 survey to arthritis prediction.
        Uses HAVARTH3 — the direct doctor-diagnosed arthritis column.

        HAVARTH3: 1=Yes, 2=No, 7=Don't know, 9=Refused, blank=missing
        """
        out = pd.DataFrame()

        # Target — must exist
        if "HAVARTH3" not in df.columns:
            print("[Arthritis] Full BRFSS: HAVARTH3 column not found — skipping dataset")
            return pd.DataFrame()

        raw = pd.to_numeric(df["HAVARTH3"], errors="coerce")
        out["arthritis_risk"] = raw.map({1: 1, 2: 0})  # 7/9/NaN → NaN
        valid_mask = out["arthritis_risk"].notna()
        out = out[valid_mask].copy()
        raw = raw[valid_mask]
        df  = df[valid_mask].copy()
        out["arthritis_risk"] = out["arthritis_risk"].astype(int)

        # Demographics
        if "_AGE80" in df.columns:
            out["Age"] = pd.to_numeric(df["_AGE80"], errors="coerce")
        elif "_AGE_G" in df.columns:
            out["Age"] = pd.to_numeric(df["_AGE_G"], errors="coerce") * 10  # age group → approx years

        if "SEX" in df.columns:
            out["Gender_M"] = (pd.to_numeric(df["SEX"], errors="coerce") == 1).astype(float)

        if "_BMI5" in df.columns:
            out["BMI"] = (pd.to_numeric(df["_BMI5"], errors="coerce") / 100.0).clip(10, 70)
        elif "BMI5" in df.columns:
            out["BMI"] = (pd.to_numeric(df["BMI5"], errors="coerce") / 100.0).clip(10, 70)

        # Health indicators
        for src, dst in [("BPHIGH4", "HighBP"), ("TOLDHI2", "HighChol"),
                         ("SMOKE100", "Smoking"), ("_RFDRHV5", "AlcoholUse"),
                         ("EXERANY2", "PhysicallyActive"), ("DIFWALK", "DiffWalk"),
                         ("CVDSTRK3", "Stroke"), ("CVDCRHD4", "HeartDisease"),
                         ("GENHLTH", "GenHlth"), ("PHYSHLTH", "PhysHlth"),
                         ("MENTHLTH", "MentHlth")]:
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce")
                # Recode 1/2 binary columns to 0/1
                if dst in {"HighBP", "HighChol", "Smoking", "AlcoholUse",
                           "PhysicallyActive", "DiffWalk", "Stroke", "HeartDisease"}:
                    out[dst] = out[dst].map({1: 1, 2: 0})

        # Inflammation proxy
        if "HighBP" in out.columns and "HighChol" in out.columns:
            out["InflammationProxy"] = (
                out["HighBP"].fillna(0) + out["HighChol"].fillna(0)
            ) / 2.0

        print(f"[Arthritis] Full BRFSS: {out['arthritis_risk'].sum():,} arthritis-positive / {len(out):,} total")
        return out

    @staticmethod
    def map_brfss_to_arthritis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map BRFSS Health Indicators (heart-disease subset, alexteboul) to arthritis prediction.
        This version does NOT have a direct arthritis column — uses musculoskeletal proxies.

        BRFSS columns: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                       HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                       HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
                       MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
        """
        out = pd.DataFrame()
        col_map = {
            "Age": "Age", "Sex": "Gender_M", "BMI": "BMI",
            "HighBP": "HighBP", "HighChol": "HighChol",
            "Smoker": "Smoking", "HvyAlcoholConsump": "AlcoholUse",
            "PhysActivity": "PhysicallyActive", "DiffWalk": "DiffWalk",
            "GenHlth": "GenHlth", "PhysHlth": "PhysHlth",
            "Stroke": "Stroke", "HeartDiseaseorAttack": "HeartDisease",
        }
        for src, dst in col_map.items():
            if src in df.columns:
                out[dst] = df[src]
        # Derived inflammation proxy
        if "HighBP" in df.columns and "HighChol" in df.columns:
            out["InflammationProxy"] = ((df["HighBP"] + df["HighChol"]) / 2.0)
        # Target: musculoskeletal disability proxy
        # DiffWalk (difficulty walking/climbing stairs) is the strongest arthritis proxy
        # in this subset; PhysHlth > 14 days = chronic poor physical health
        if "DiffWalk" in df.columns:
            diff = pd.to_numeric(df["DiffWalk"], errors="coerce").fillna(0)
            phys = pd.to_numeric(df.get("PhysHlth", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
            out["arthritis_risk"] = ((diff == 1) | (phys > 14)).astype(int)
        elif "HeartDiseaseorAttack" in df.columns:
            out["arthritis_risk"] = pd.to_numeric(df["HeartDiseaseorAttack"], errors="coerce").fillna(0).astype(int)
        return out


# ==============================================================================
# Data Loader with EDA — supports multiple datasets
# ==============================================================================
class ArthritisDataLoader:
    """
    Loads the best available dataset:
    1. Cardiovascular Disease Dataset (70K records) via Kaggle
    2. BRFSS Health Indicators (253K records) via Kaggle
    3. APD Excel dataset (102 records) — local fallback
    """

    def __init__(self):
        self.df = None
        self.dataset_info = None

    # Priority order for dataset loading (most arthritis-specific first)
    _LOAD_PIPELINE = [
        (
            "arthritis_nih",
            lambda raw: RealDatasetManager.map_nhanes_to_arthritis(raw),
        ),
        (
            "brfss_full",
            lambda raw: RealDatasetManager.map_brfss_full_to_arthritis(raw),
        ),
        (
            "cardiovascular",
            lambda raw: RealDatasetManager.map_cardiovascular_to_arthritis(raw),
        ),
    ]

    @staticmethod
    def _stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
        """Class-stratified subsample to n rows on arthritis_risk (preserves prevalence)."""
        if n is None or "arthritis_risk" not in df.columns or len(df) <= n:
            return df
        rng = np.random.RandomState(seed)
        parts = []
        for cls, grp in df.groupby("arthritis_risk"):
            k = max(1, int(round(n * len(grp) / len(df))))
            k = min(k, len(grp))
            parts.append(grp.iloc[rng.choice(len(grp), k, replace=False)])
        out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out.iloc[:n].reset_index(drop=True)

    def load(self, prefer_real: bool = True, sample_n: int = None) -> pd.DataFrame:
        """
        Load the best available real dataset from Kaggle.

        Priority (highest arthritis specificity first):
          1. NHANES MCQ — doctor-diagnosed arthritis (most specific)
          2. Full BRFSS 2015 — HAVARTH3 direct arthritis column
          3. Cardiovascular Disease Dataset — 70K records, metabolic risk factors
          4. BRFSS HDI subset — musculoskeletal disability proxy

        `sample_n` (e.g. 3000) class-stratified-caps the loaded real dataset to a
        fixed size for a manageable, reproducible training run.

        APD (102 records) is only used when Kaggle is completely unavailable AND
        the local file already exists — it is never the default.
        """
        if prefer_real:
            for dataset_key, mapper in self._LOAD_PIPELINE:
                raw = RealDatasetManager.download_and_load(dataset_key)
                if raw is None or len(raw) < 500:
                    continue
                mapped = mapper(raw)
                if mapped is None or len(mapped) < 500:
                    print(f"[Arthritis] {dataset_key} mapped to <500 usable rows — skipping")
                    continue
                # Drop rows with no target
                if "arthritis_risk" in mapped.columns:
                    mapped = mapped.dropna(subset=["arthritis_risk"])
                if len(mapped) < 500:
                    continue
                if sample_n:
                    mapped = self._stratified_subsample(mapped, sample_n)
                    print(f"[Arthritis] Stratified-subsampled to {len(mapped):,} records")
                self.df = mapped.reset_index(drop=True)
                self.dataset_info = {
                    **REAL_DATASETS[dataset_key],
                    "loaded_records": len(self.df),
                    "has_direct_arthritis_label": dataset_key in {"arthritis_nih", "brfss_full"},
                }
                print(f"[Arthritis] Using {REAL_DATASETS[dataset_key]['name']}: {len(self.df):,} records")
                return self.df

        # Last resort: local APD file (102 records) — only if it exists
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                "No Kaggle dataset could be downloaded and the local APD fallback "
                f"is missing at {DATASET_PATH}.\n"
                "Ensure kagglehub is installed and Kaggle API credentials are configured:\n"
                "  pip install kagglehub\n"
                "  Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or place\n"
                "  kaggle.json in ~/.kaggle/kaggle.json"
            )
        print("[Arthritis] WARNING: All Kaggle downloads failed — using local APD (102 records). "
              "Install kagglehub and configure Kaggle credentials for production-quality data.")
        self.df = pd.read_excel(DATASET_PATH)
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])
        numeric_cols = self.df.select_dtypes(include="number").columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        self.dataset_info = {
            "name": "Arthritis Profile Dataset (APD) — LOCAL FALLBACK ONLY",
            "url": "https://www.kaggle.com/datasets/santhoshkumarsundar/arthritis-profile-dataset",
            "records": len(self.df),
            "loaded_records": len(self.df),
            "description": "Clinical blood markers from 102 rheumatology patients (fallback — configure Kaggle credentials for real data)",
            "has_direct_arthritis_label": True,
            "is_fallback": True,
        }
        print(f"[Arthritis] APD fallback: {len(self.df)} records")
        return self.df

    def get_eda_summary(self) -> dict:
        if self.df is None:
            self.load()
        df = self.df
        info = self.dataset_info or {}
        is_real_dataset = info.get("loaded_records", 0) > 500
        has_direct_label = info.get("has_direct_arthritis_label", False)

        # Dataset source info
        source_info = {
            "dataset_name": info.get("name", "Arthritis Profile Dataset"),
            "dataset_url": info.get("url", ""),
            "loaded_records": int(len(df)),
            "description": info.get("description", ""),
            "is_large_dataset": is_real_dataset,
            "has_direct_arthritis_label": has_direct_label,
            "is_fallback": info.get("is_fallback", False),
        }

        # Basic stats
        numeric_df = df.select_dtypes(include=[np.number])
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

        summary = {
            **source_info,
            "total_samples": int(len(df)),
            "total_features": int(len(df.columns)),
            "feature_names": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": missing_pct,
            "low_missing_data": all(v < 20 for v in missing_pct.values()),
            "statistics": {},
        }

        # Per-column statistics
        for col in numeric_df.columns:
            desc = df[col].describe()
            summary["statistics"][col] = {
                "mean": round(float(desc["mean"]), 2) if not pd.isna(desc["mean"]) else None,
                "std": round(float(desc["std"]), 2) if not pd.isna(desc["std"]) else None,
                "min": round(float(desc["min"]), 2) if not pd.isna(desc["min"]) else None,
                "max": round(float(desc["max"]), 2) if not pd.isna(desc["max"]) else None,
            }

        # Demographics
        gender_col = "Gender_M" if "Gender_M" in df.columns else "Sex"
        age_col = "Age"
        if gender_col in df.columns:
            summary["gender_distribution"] = {
                "male": int((df[gender_col] == 1).sum()),
                "female": int((df[gender_col] == 0).sum()),
            }
        if age_col in df.columns:
            vals = df[age_col].dropna()
            summary["age_stats"] = {
                "mean": round(float(vals.mean()), 1),
                "min": round(float(vals.min()), 1),
                "max": round(float(vals.max()), 1),
            }

        # Marker groups — adapt to whichever dataset was loaded
        summary["inflammatory_markers"] = {}
        summary["hematology_markers"] = {}
        summary["biochemistry_markers"] = {}

        if is_real_dataset:
            # Inflammation / vascular markers common to real datasets
            for col in ["InflammationProxy", "CholesterolLevel", "HighChol", "HighBP"]:
                if col in df.columns:
                    vals = df[col].dropna()
                    summary["inflammatory_markers"][col] = {
                        "mean": round(float(vals.mean()), 3),
                        "elevated_count": int((vals > vals.median()).sum()),
                        "available_samples": int(len(vals)),
                    }
            # NHANES joint pain indicators
            joint_cols = [c for c in df.columns if c.startswith("JointPain_")]
            for col in joint_cols:
                vals = df[col].dropna()
                summary["inflammatory_markers"][col] = {
                    "mean": round(float(vals.mean()), 3),
                    "elevated_count": int(vals.sum()),
                    "available_samples": int(len(vals)),
                }
            for col in ["BMI", "SystolicBP", "DiastolicBP", "MetabolicRisk"]:
                if col in df.columns:
                    vals = df[col].dropna()
                    summary["hematology_markers"][col] = {
                        "mean": round(float(vals.mean()), 3),
                        "available_samples": int(len(vals)),
                    }
            for col in ["GlucoseLevel", "AlcoholUse", "Smoking", "PhysicallyActive",
                        "GenHlth", "PhysHlth", "DiffWalk"]:
                if col in df.columns:
                    vals = df[col].dropna()
                    summary["biochemistry_markers"][col] = {
                        "mean": round(float(vals.mean()), 3),
                        "available_samples": int(len(vals)),
                    }
        else:
            # APD dataset — clinical blood markers
            for marker in INFLAMMATORY_MARKERS:
                if marker in df.columns:
                    vals = df[marker].dropna()
                    summary["inflammatory_markers"][marker] = {
                        "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                        "elevated_count": int((vals > vals.quantile(0.75)).sum()) if len(vals) > 0 else 0,
                        "available_samples": int(len(vals)),
                    }
            for marker in HEMATOLOGY_MARKERS:
                if marker in df.columns:
                    vals = df[marker].dropna()
                    summary["hematology_markers"][marker] = {
                        "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                        "available_samples": int(len(vals)),
                    }
            for marker in BIOCHEMISTRY_MARKERS:
                if marker in df.columns:
                    vals = df[marker].dropna()
                    summary["biochemistry_markers"][marker] = {
                        "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                        "available_samples": int(len(vals)),
                    }

        # Target distribution
        if "arthritis_risk" in df.columns:
            counts = df["arthritis_risk"].value_counts().to_dict()
            summary["target_distribution"] = {
                "high_risk": int(counts.get(1, 0)),
                "low_risk": int(counts.get(0, 0)),
                "prevalence": round(float(df["arthritis_risk"].mean()), 4),
                "label_type": "Direct arthritis diagnosis" if has_direct_label else "Musculoskeletal risk proxy",
            }
        return summary

    def get_correlation_matrix(self) -> dict:
        if self.df is None:
            self.load()
        numeric = self.df.select_dtypes(include=[np.number]).head(5000)  # limit for speed
        corr = numeric.corr().round(3)
        return {"columns": corr.columns.tolist(), "data": corr.values.tolist()}


# ==============================================================================
# Feature Engineering — adapts to dataset source
# ==============================================================================
class AdvancedFeatureEngineer:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # APD-specific features
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
        # Real dataset derived features
        if "BMI" in out.columns:
            out["obesity_flag"] = (out["BMI"].fillna(25) > 30).astype(float)
            out["bmi_squared"] = (out["BMI"].fillna(25) ** 2) / 1000.0
        if "Age" in out.columns:
            out["age_risk"] = (out["Age"].fillna(45) > 55).astype(float)
            out["age_squared"] = (out["Age"].fillna(45) ** 2) / 10000.0
        if "SystolicBP" in out.columns and "DiastolicBP" in out.columns:
            out["pulse_pressure"] = out["SystolicBP"].fillna(120) - out["DiastolicBP"].fillna(80)
            out["map"] = (out["DiastolicBP"].fillna(80) + out["pulse_pressure"].fillna(40) / 3.0)
        if "CholesterolLevel" in out.columns:
            out["chol_elevated"] = (out["CholesterolLevel"].fillna(1) > 1).astype(float)
        if "Smoking" in out.columns and "AlcoholUse" in out.columns:
            out["lifestyle_risk"] = out["Smoking"].fillna(0) + out["AlcoholUse"].fillna(0)
        return out


# ==============================================================================
# PyTorch Tabular BERT + MoE Architecture (Enhanced)
# ==============================================================================
class TabularBERTMoE(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, num_experts: int = 6, dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.num_experts = num_experts

        # Feature value projection (1 -> d_model)
        self.val_proj = nn.Linear(1, d_model)
        # Feature ID embeddings
        self.feat_emb = nn.Embedding(num_features, d_model)

        # Transformer Encoder (3 layers for better representation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Enhanced Mixture of Experts (6 experts)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(num_experts)
        ])

        # Classification Head with residual
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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

        # Top-2 MoE routing
        route_logits = self.router(pooled)
        route_probs = torch.softmax(route_logits, dim=-1)

        # Sparse top-2 routing
        top2_vals, top2_idx = torch.topk(route_probs, 2, dim=-1)
        top2_vals = top2_vals / top2_vals.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([exp(pooled) for exp in self.experts], dim=1)
        # Gather top-2 experts
        idx_expanded = top2_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        selected = expert_outputs.gather(1, idx_expanded)  # (B, 2, d_model)
        moe_out = (top2_vals.unsqueeze(-1) * selected).sum(dim=1)  # (B, d_model)

        logits = self.head(moe_out).squeeze(-1)
        return logits


# ==============================================================================
# Model Training and Prediction Pipeline
# Hybrid Stacking Ensemble:
#   Layer 0:  GradientBoosting + RandomForest + ExtraTrees + Calibrated-SVC
#             + Tabular BERT-MoE  (PyTorch)
#   Layer 1:  LogisticRegression meta-learner trained on OOF predictions
#   Balancing: SMOTE-lite (minority class synthetic oversampling)
# Target accuracy: >95% (APD), best-effort on large datasets
# ==============================================================================
class ArthritisPredictor:
    def __init__(self):
        self.model = None                  # TabularBERTMoE (kept for compat)
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_cols = []
        self.is_trained = False
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_real_dataset = False
        # Hybrid stacking components
        self.base_learners: dict = {}      # name -> fitted sklearn clf
        self.meta_learner = None           # LogisticRegression stacking head

    # ------------------------------------------------------------------
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        if "arthritis_risk" in df.columns:
            return df["arthritis_risk"].fillna(0).astype(int)
        ra_positive  = df["RA"].fillna(0)  > 14 if "RA"  in df.columns else pd.Series([False]*len(df))
        crp_positive = df["CRP"].fillna(0) > 6  if "CRP" in df.columns else pd.Series([False]*len(df))
        return (ra_positive | crp_positive).astype(int)

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        exclude = {"arthritis_risk", "RA", "CRP", "Unnamed: 0", "id"}
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    # ------------------------------------------------------------------
    def _smote_lite(self, X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
        """Oversample every minority class to match the majority count."""
        rng = np.random.RandomState(random_state)
        classes, counts = np.unique(y.astype(int), return_counts=True)
        max_cnt = counts.max()
        X_out, y_out = [X], [y]
        for cls, cnt in zip(classes, counts):
            if cnt >= max_cnt:
                continue
            cls_X = X[y.astype(int) == cls]
            n_gen = max_cnt - cnt
            synth = []
            for _ in range(n_gen):
                i, j = rng.choice(len(cls_X), 2, replace=True)
                alpha = rng.random()
                synth.append(cls_X[i] * alpha + cls_X[j] * (1 - alpha))
            X_out.append(np.array(synth))
            y_out.append(np.full(n_gen, cls, dtype=int))
        X_bal = np.vstack(X_out)
        y_bal = np.concatenate(y_out).astype(int)
        perm  = rng.permutation(len(X_bal))
        return X_bal[perm], y_bal[perm]

    # ------------------------------------------------------------------
    def _sklearn_learners(self, n_samples: int) -> dict:
        """Return sklearn base-learner dict tuned to dataset size."""
        if n_samples <= 500:          # APD — small, powerful settings
            return {
                "gbm": GradientBoostingClassifier(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=2, random_state=42),
                "rf":  RandomForestClassifier(
                    n_estimators=500, class_weight="balanced",
                    min_samples_leaf=1, random_state=42),
                "et":  ExtraTreesClassifier(
                    n_estimators=400, class_weight="balanced",
                    min_samples_leaf=1, random_state=42),
                "svc": CalibratedClassifierCV(
                    SVC(kernel="rbf", C=10, gamma="scale"), cv=3),
            }
        else:                         # large dataset — faster settings
            return {
                "gbm": GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, random_state=42),
                "rf":  RandomForestClassifier(
                    n_estimators=300, class_weight="balanced",
                    random_state=42, n_jobs=1),
                "et":  ExtraTreesClassifier(
                    n_estimators=200, class_weight="balanced",
                    random_state=42, n_jobs=1),
            }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _train_pytorch_model(self, X_train, y_train, num_epochs=80, batch_size=256):
        n_feat = X_train.shape[1]
        if len(X_train) > 10000:
            batch_size = 512
        elif len(X_train) <= 200:
            batch_size = max(16, len(X_train) // 4)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, drop_last=False)

        self.model = TabularBERTMoE(num_features=n_feat).to(self.device)
        optimizer  = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        pos_ratio  = float(y_train.mean())
        pos_weight = torch.tensor([(1 - pos_ratio) / max(pos_ratio, 0.01)]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model.train()
        for epoch in range(num_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

    def _bert_oof_probs(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> np.ndarray:
        """OOF probabilities from BERT-MoE without storing the fold models."""
        skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof  = np.zeros(len(X))
        n_feat = X.shape[1]
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr = y[tr_idx]
            bs   = max(16, min(256, len(X_tr) // 4))
            m    = TabularBERTMoE(num_features=n_feat).to(self.device)
            pos_ratio  = float(y_tr.mean())
            pos_weight = torch.tensor([(1 - pos_ratio) / max(pos_ratio, 0.01)]).to(self.device)
            opt  = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            ld   = DataLoader(TensorDataset(
                       torch.tensor(X_tr, dtype=torch.float32).to(self.device),
                       torch.tensor(y_tr, dtype=torch.float32).to(self.device)),
                   batch_size=bs, shuffle=True)
            m.train()
            for _ in range(50):
                for bx, by in ld:
                    opt.zero_grad()
                    loss = crit(m(bx), by)
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
            m.eval()
            with torch.no_grad():
                X_te_t = torch.tensor(X_te, dtype=torch.float32).to(self.device)
                oof[te_idx] = torch.sigmoid(m(X_te_t)).cpu().numpy()
        return oof

    # ------------------------------------------------------------------
    def _predict_hybrid(self, X_scaled: np.ndarray) -> tuple:
        """Combine all base learners via the meta-learner."""
        preds_list = []
        for clf in self.base_learners.values():
            preds_list.append(clf.predict_proba(X_scaled)[:, 1])
        # BERT-MoE
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            preds_list.append(torch.sigmoid(self.model(X_t)).cpu().numpy())
        X_meta      = np.column_stack(preds_list)
        final_probs = self.meta_learner.predict_proba(X_meta)[:, 1]
        return (final_probs > 0.5).astype(int), final_probs

    # ------------------------------------------------------------------
    def _rigorous_cv_report(self, X_raw: np.ndarray, y: np.ndarray,
                            n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> dict:
        """
        Honest generalisation estimate: Repeated Stratified K-Fold CV with 95% CIs.
        Imputation, scaling and SMOTE are fit INSIDE each training fold only, so the
        held-out fold is never seen during preprocessing or augmentation (no leakage,
        no train-set evaluation). Reports the sklearn stacking ensemble plus a
        Logistic-Regression baseline and calibration (Brier score).
        """
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
        from sklearn.metrics import brier_score_loss
        if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < n_splits:
            return {"note": "Too few samples/'positives' for repeated CV — skipped."}

        def _ci(vals):
            v = np.array([x for x in vals if not np.isnan(x)], dtype=float)
            if len(v) == 0:
                return {"mean": None, "std": None, "ci95": [None, None]}
            m, s = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            h = 1.96 * s / np.sqrt(len(v)) if len(v) > 1 else 0.0
            return {"mean": round(m, 4), "std": round(s, 4),
                    "ci95": [round(float(m - h), 4), round(float(m + h), 4)]}

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        acc, auc, f1s, brier = [], [], [], []
        n_samples = len(y)
        for tr, te in rskf.split(X_raw, y):
            imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(X_raw[tr]))
            Xte = sc.transform(imp.transform(X_raw[te]))
            Xb, yb = self._smote_lite(Xtr, y[tr])
            clfs = self._sklearn_learners(n_samples)
            oof = [cross_val_predict(c, Xb, yb, cv=3, method="predict_proba", n_jobs=1)[:, 1]
                   for c in clfs.values()]
            meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            meta.fit(np.column_stack(oof), yb)
            test_cols = []
            for c in clfs.values():
                c.fit(Xb, yb); test_cols.append(c.predict_proba(Xte)[:, 1])
            p = meta.predict_proba(np.column_stack(test_cols))[:, 1]
            pred = (p > 0.5).astype(int)
            acc.append(accuracy_score(y[te], pred))
            auc.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
            brier.append(brier_score_loss(y[te], p))
        return {
            "protocol": f"{n_splits}x{n_repeats} Repeated Stratified K-Fold (preproc+SMOTE inside folds)",
            "n_samples": int(n_samples),
            "accuracy": _ci(acc), "auc_roc": _ci(auc),
            "f1_macro": _ci(f1s), "brier": _ci(brier),
        }

    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, cross_validate: bool = True) -> dict:
        self.is_real_dataset = "arthritis_risk" in df.columns
        n_samples = len(df)
        print(f"[Arthritis] Hybrid ensemble training — {n_samples} records, dataset={'real' if self.is_real_dataset else 'APD'}")

        y         = self._create_target(df).values
        base_cols = self._get_feature_cols(df)
        X_eng     = self.feature_engineer.transform(df[base_cols].copy())
        self.feature_cols = [c for c in X_eng.columns
                             if c not in {"arthritis_risk", "RA", "CRP"}
                             and X_eng[c].dtype in [np.float64, np.int64, float, int]]

        X_arr     = X_eng[self.feature_cols].values.astype(np.float64)
        X_imputed = self.imputer.fit_transform(X_arr)
        X_scaled  = self.scaler.fit_transform(X_imputed)

        # Honest generalisation estimate BEFORE fitting the final model on all data.
        cv_report = self._rigorous_cv_report(X_arr, y) if cross_validate else None
        if cv_report and "accuracy" in cv_report:
            print(f"[Arthritis] Repeated-CV — acc {cv_report['accuracy']['mean']} "
                  f"CI{cv_report['accuracy']['ci95']} | AUC {cv_report['auc_roc']['mean']} "
                  f"CI{cv_report['auc_roc']['ci95']}")

        # Stratified subsample for very large datasets (keep ≤30 K)
        MAX_TR = 30_000
        if len(X_scaled) > MAX_TR:
            rng  = np.random.RandomState(42)
            idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
            n0   = min(MAX_TR // 2, len(idx0)); n1 = min(MAX_TR // 2, len(idx1))
            idx  = np.concatenate([rng.choice(idx0, n0, replace=False),
                                   rng.choice(idx1, n1, replace=False)])
            rng.shuffle(idx)
            X_tr, y_tr = X_scaled[idx], y[idx]
        else:
            X_tr, y_tr = X_scaled.copy(), y.copy()

        # ── Hold-out test split (stratified, before SMOTE) ───────────────
        # Ensures evaluation is always on unseen data, never on training samples.
        test_size = 0.2 if len(X_tr) >= 50 else max(1, int(len(X_tr) * 0.15))
        try:
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=42, stratify=y_tr
            )
        except ValueError:
            # Fall back to non-stratified if classes are too rare
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=42
            )
        print(f"[Arthritis] Train: {len(X_train_split)} | Hold-out test: {len(X_test)}")

        # SMOTE-lite: balance classes on TRAIN split only (never touches test set)
        X_bal, y_bal = self._smote_lite(X_train_split, y_train_split)
        n_cv = 5 if len(X_bal) >= 50 else 3
        print(f"[Arthritis] After SMOTE: {len(X_bal)} samples | CV folds: {n_cv}")

        # ── Phase 1: OOF predictions from each base learner ──────────────
        sklearn_clfs = self._sklearn_learners(n_samples)
        oof_cols = []

        print("[Arthritis] Computing OOF predictions for stacking meta-learner...")
        for name, clf in sklearn_clfs.items():
            print(f"  → {name}")
            probs = cross_val_predict(clf, X_bal, y_bal, cv=n_cv,
                                      method="predict_proba", n_jobs=1)[:, 1]
            oof_cols.append(probs)

        print("  → bert_moe (OOF)")
        oof_cols.append(self._bert_oof_probs(X_bal, y_bal, n_splits=n_cv))

        # ── Phase 2: Train meta-learner on OOF stack ──────────────────────
        X_meta = np.column_stack(oof_cols)
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_learner.fit(X_meta, y_bal)
        print("[Arthritis] Meta-learner trained.")

        # ── Phase 3: Refit all base learners on full balanced data ────────
        self.base_learners = {}
        for name, clf in sklearn_clfs.items():
            print(f"[Arthritis] Final fit: {name}")
            clf.fit(X_bal, y_bal)
            self.base_learners[name] = clf

        print("[Arthritis] Final BERT-MoE training...")
        self._train_pytorch_model(X_bal, y_bal, num_epochs=80)

        # ── Phase 4: Evaluate on HELD-OUT test set only ───────────────────
        # Using the test set from the pre-SMOTE split — these samples were
        # never seen during training or SMOTE augmentation.
        preds, probs_arr = self._predict_hybrid(X_test)
        eval_y = y_test

        accuracy  = accuracy_score(eval_y, preds)
        f1        = f1_score(eval_y, preds, average="macro", zero_division=0)
        precision = precision_score(eval_y, preds, average="macro", zero_division=0)
        recall    = recall_score(eval_y, preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(eval_y, probs_arr) if len(np.unique(eval_y)) > 1 else 0.0
        except Exception:
            auc = 0.0

        # CV mean accuracy (from OOF on training set — honest cross-validation estimate)
        # Average OOF accuracy from the first sklearn learner as a representative CV metric
        oof_all = np.column_stack(oof_cols)
        oof_preds = (oof_all.mean(axis=1) > 0.5).astype(int)
        cv_mean_accuracy = round(float(accuracy_score(y_bal, oof_preds)), 4)

        # Feature importances from RF (most interpretable)
        top_features = []
        if "rf" in self.base_learners:
            imp     = self.base_learners["rf"].feature_importances_
            top_idx = np.argsort(imp)[::-1][:12]
            top_features = [{"feature": self.feature_cols[i], "importance": round(float(imp[i]), 4)}
                            for i in top_idx]

        self.is_trained = True
        self.metrics = {
            "accuracy":              round(float(accuracy), 4),
            "cv_mean_accuracy":      cv_mean_accuracy,
            "f1":                    round(float(f1), 4),
            "precision":             round(float(precision), 4),
            "recall":                round(float(recall), 4),
            "auc_roc":               round(float(auc), 4),
            "train_samples":         int(len(X_bal)),
            "test_samples":          int(len(X_test)),
            "total_dataset_records": int(n_samples),
            "positive_class_ratio":  round(float(y.mean()), 4),
            "evaluation_method":     "Held-out test set (20% stratified split, pre-SMOTE)",
            "cross_validation":      cv_report,   # honest repeated-CV mean±95% CI (headline metric)
            "model_type":            "Hybrid Stack: BERT-MoE + GBM + RF + ExtraTrees + SVC → LogReg Meta-Learner",
            "feature_engineering":   "Advanced clinical features + SMOTE-lite balancing + KNN imputation",
            "imputation":            "KNN (k=5)",
            "total_features":        len(self.feature_cols),
            "top_features":          top_features,
            "classification_report": classification_report(eval_y, preds, output_dict=True, zero_division=0),
            "dataset_source":        "Kaggle Real Dataset" if self.is_real_dataset else "APD Local Dataset",
        }
        print(f"[Arthritis] Test set — Accuracy: {accuracy:.4f}  AUC: {auc:.4f}  F1: {f1:.4f}")

        # ── Save ──────────────────────────────────────────────────────────
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        joblib.dump({
            "scaler":        self.scaler,
            "imputer":       self.imputer,
            "feature_cols":  self.feature_cols,
            "is_real_dataset": self.is_real_dataset,
            "base_learners": self.base_learners,
            "meta_learner":  self.meta_learner,
        }, SCALER_SAVE_PATH)

        return self.metrics

    def predict(self, patient_data: dict) -> dict:
        if not self.is_trained:
            if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
                try:
                    saved = joblib.load(SCALER_SAVE_PATH)
                    self.scaler         = saved["scaler"]
                    self.imputer        = saved["imputer"]
                    self.feature_cols   = saved["feature_cols"]
                    self.is_real_dataset= saved.get("is_real_dataset", False)
                    self.base_learners  = saved.get("base_learners", {})
                    self.meta_learner   = saved.get("meta_learner", None)
                    self.model = TabularBERTMoE(num_features=len(self.feature_cols)).to(self.device)
                    self.model.load_state_dict(
                        torch.load(MODEL_SAVE_PATH, map_location=self.device, weights_only=True))
                    self.model.eval()
                    self.is_trained = True
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load saved model ({e}). Retrain via POST /api/arthritis/train.")
            else:
                raise RuntimeError("Model not trained yet. Please train via POST /api/arthritis/train.")

        # ── Feature preparation ───────────────────────────────────────────
        row_df  = pd.DataFrame([patient_data])
        row_eng = self.feature_engineer.transform(row_df)
        for col in self.feature_cols:
            if col not in row_eng.columns:
                row_eng[col] = 0.0
        row_eng = row_eng[self.feature_cols]

        # Fill remaining NaN with scaler column means (safe neutral)
        means = getattr(self.scaler, "mean_", None)
        if means is not None:
            for j, col in enumerate(self.feature_cols):
                if pd.isna(row_eng.iloc[0, j]):
                    row_eng.iloc[0, j] = float(means[j])

        X        = row_eng.values.astype(np.float64)
        X_imp    = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)

        # ── Hybrid prediction (stacking) if meta-learner available ───────
        if self.meta_learner is not None and self.base_learners:
            _, probs_arr = self._predict_hybrid(X_scaled)
            prob = float(probs_arr[0])
        else:
            # Fallback: BERT-MoE only
            with torch.no_grad():
                X_t  = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
                prob = torch.sigmoid(self.model(X_t))[0].item()

        prediction = 1 if prob > 0.5 else 0
        risk_level = "HIGH" if prediction == 1 else "LOW"
        confidence = prob if prediction == 1 else (1.0 - prob)

        return {
            "risk_level":   risk_level,
            "prediction":   prediction,
            "confidence":   round(confidence, 4),
            "probabilities":{"low_risk": round(1.0 - prob, 4), "high_risk": round(prob, 4)},
            "clinical_interpretation": self._interpret(patient_data, risk_level, confidence),
        }

    def _interpret(self, data: dict, risk: str, conf: float) -> str:
        lines = [f"Arthritis Risk Assessment: {risk} (Confidence: {conf:.1%})", ""]
        # APD-specific interpretations
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
        # Real dataset interpretations
        bmi = data.get("BMI")
        if bmi and bmi > 30:
            lines.append(f"- BMI {bmi:.1f} kg/m² — obesity is a significant arthritis risk factor.")
        bp = data.get("SystolicBP")
        if bp and bp > 140:
            lines.append(f"- Systolic BP {bp} mmHg — hypertension correlates with inflammatory arthritis.")
        if risk == "HIGH":
            lines += ["", "RECOMMENDATION: Urgent referral to rheumatologist. Consider anti-CCP antibodies, joint imaging, and DMARD initiation per ACR/EULAR guidelines."]
        else:
            lines += ["", "RECOMMENDATION: Continue routine monitoring. Re-evaluate if joint pain, morning stiffness, or joint swelling develops."]
        return "\n".join(lines)


# ==============================================================================
# Performance Comparison: Model vs Clinical Benchmarks
# ==============================================================================
CARDIOLOGIST_BENCHMARKS = {
    "arthritis_screening": {
        "source": "ACR 2010 Classification Criteria for RA",
        "expert_sensitivity": 0.82,
        "expert_specificity": 0.91,
        "expert_auc": 0.87,
        "note": "Based on 2010 ACR/EULAR RA classification criteria validation studies",
    }
}

def compare_with_benchmark(model_metrics: dict) -> dict:
    """Compare model performance against published clinical benchmarks."""
    bench = CARDIOLOGIST_BENCHMARKS["arthritis_screening"]
    model_auc = model_metrics.get("auc_roc", 0.0)
    expert_auc = bench["expert_auc"]
    return {
        "benchmark_source": bench["source"],
        "model_auc_roc": model_auc,
        "expert_reference_auc": expert_auc,
        "difference": round(model_auc - expert_auc, 4),
        "model_vs_expert": "Comparable" if abs(model_auc - expert_auc) < 0.05 else
                           ("Superior" if model_auc > expert_auc else "Below Expert"),
        "note": bench["note"],
        "validation_status": "Research prototype — prospective clinical validation required",
    }


# ==============================================================================
# Global Lazy Init
# ==============================================================================
_loader_instance = None
_predictor_instance = None

def get_arthritis_loader() -> ArthritisDataLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ArthritisDataLoader()
        _loader_instance.load(prefer_real=True)
    return _loader_instance

def get_arthritis_predictor() -> ArthritisPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ArthritisPredictor()
    return _predictor_instance


# ==============================================================================
# ArthritisPipeline — unified wrapper used by the Colab notebook / main.py
# ==============================================================================
class ArthritisPipeline:
    """
    Thin facade over ArthritisDataLoader + ArthritisPredictor.
    Supports real datasets (70K-253K records) and APD fallback.
    """

    def __init__(self):
        self.loader    = ArthritisDataLoader()
        self.predictor = ArthritisPredictor()
        self.df        = None

    def load(self, prefer_real: bool = True) -> pd.DataFrame:
        self.df = self.loader.load(prefer_real=prefer_real)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load() before preprocess()")
        return self.df

    def train(self) -> dict:
        if self.df is None:
            raise RuntimeError("Call load() before train()")
        return self.predictor.train(self.df)

    def predict(self, patient_data: dict) -> dict:
        return self.predictor.predict(patient_data)

    def get_dataset_info(self) -> dict:
        return self.loader.dataset_info or {}
