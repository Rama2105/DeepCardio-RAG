"""
core/model_cards.py — Model Cards & Transparency Documentation
==============================================================
Each model card documents: training data, architecture, intended use,
known limitations, demographic biases, and clinical disclaimer.

Access via API:
    GET /api/model-cards          → list all cards
    GET /api/model-cards/{model}  → specific card

References:
    - Mitchell et al. (2019) "Model Cards for Model Reporting"
    - FDA AI/ML-Based SaMD Action Plan (2021)
"""

from typing import Dict, Any

CLINICAL_DISCLAIMER = (
    "⚠ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE. "
    "This model has not been validated in a clinical setting, approved by any regulatory body "
    "(FDA, CE Mark, etc.), or tested on diverse patient populations. "
    "All outputs are for research and educational purposes only. "
    "Never use these predictions to guide medical decisions without independent verification "
    "by a qualified healthcare professional."
)

MODEL_CARDS: Dict[str, Dict[str, Any]] = {

    # ─────────────────────────────────────────────────────────────────────────
    "ecg_cnn": {
        "model_name":       "ECG 1D-CNN Encoder",
        "version":          "1.0.0",
        "modality":         "ECG Signal (12-lead, 1250 samples)",
        "task":             "Feature extraction + RAG-augmented report generation",
        "architecture":     "3-block 1D Convolutional Network → 384-dim embedding → GPT-2 report",
        "output":           "Clinical narrative report (text)",
        "training_data":    "No real ECG data used in current version; model uses random weights.",
        "intended_use":     (
            "Research demonstration of ECG-to-report pipeline. "
            "Intended for AI/ML researchers exploring clinical NLP."
        ),
        "out_of_scope":     [
            "Clinical diagnosis of any arrhythmia or cardiac condition",
            "Real-time patient monitoring",
            "Paediatric ECG interpretation",
            "Any regulatory submission",
        ],
        "limitations":      [
            "Random weights — not trained on real ECG data",
            "BLEU score of 0.847 is hardcoded, not a real measurement",
            "No validation on diverse demographics or pathologies",
            "12-lead input only; no support for single-lead wearable data",
        ],
        "known_biases":     [
            "No demographic data considered",
            "No sex or age bias analysis performed",
        ],
        "metrics":          {
            "note": "Model not trained — no real metrics available. "
                    "Benchmark against MIT-BIH / PTB-XL datasets when weights are trained."
        },
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "echonet_3dcnn": {
        "model_name":       "EchoNet 3D-CNN EF Regressor",
        "version":          "1.0.0",
        "modality":         "Echocardiogram Video (AVI, apical-4-chamber)",
        "task":             "Ejection Fraction (EF) regression",
        "architecture":     "R2+1D spatiotemporal CNN → 384-dim → MLP → EF scalar",
        "output":           "EF (%), EF category (HFrEF / HFmrEF / Normal), clinical report",
        "training_data":    (
            "Designed for EchoNet-Dynamic (Stanford AIMI, 10,030 echocardiogram videos). "
            "Dataset requires a research use agreement: https://echonet.github.io/dynamic/"
        ),
        "intended_use":     (
            "Research exploration of video-based cardiac function assessment. "
            "Companion to clinical echocardiography analysis workflows."
        ),
        "out_of_scope":     [
            "Replacing clinical echocardiography interpretation",
            "Non-apical-4-chamber views",
            "Paediatric or congenital heart disease",
        ],
        "limitations":      [
            "Current weights are random (not trained on EchoNet-Dynamic)",
            "Only grayscale 112×112 input; colour echo not supported",
            "EF estimation accuracy strongly depends on image quality",
            "Not validated on non-Stanford acquisition protocols",
        ],
        "known_biases":     [
            "EchoNet-Dynamic is predominantly from one academic medical centre (Stanford)",
            "May not generalise to community hospital echo quality",
        ],
        "metrics":          {
            "reference_paper_mae": "< 6% EF MAE reported in Ouyang et al. (2020)",
            "current_model_mae":   "Not evaluated (random weights)",
        },
        "paper":            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8979576/",
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "arrhythmia_video_cnn": {
        "model_name":       "Cardiac Arrhythmia Video 3D-CNN",
        "version":          "1.0.0",
        "modality":         "Echocardiogram Video (AVI/MP4)",
        "task":             "Binary arrhythmia classification (Normal vs Arrhythmia)",
        "architecture":     "3-block 3D-CNN → AdaptiveAvgPool3d → MLP → 2-class softmax",
        "output":           "Arrhythmia label (0/1), confidence, clinical guideline",
        "training_data":    (
            "Designed for EchoNet-Dynamic with arrhythmia labels derived from EF thresholds "
            "(EF < 35 or > 75 flagged as arrhythmia proxy). "
            "For explicit arrhythmia labels, an 'Arrhythmia' column in FileList.csv is supported."
        ),
        "intended_use":     (
            "Research prototype for video-based arrhythmia screening. "
            "Exploratory tool for cardiac motion analysis."
        ),
        "out_of_scope":     [
            "Specific arrhythmia subtype diagnosis (AF vs VT vs flutter)",
            "Real-time clinical screening",
            "Paediatric populations",
        ],
        "limitations":      [
            "EF-proxy labels are an imperfect surrogate for true arrhythmia diagnosis",
            "Binary classification cannot distinguish between arrhythmia subtypes",
            "Random weights — requires training on labelled data",
            "No multi-class output (AF, VT, VFL, etc.) yet",
        ],
        "known_biases":     [
            "Class imbalance: ~15% arrhythmia, ~85% normal in EchoNet-Dynamic",
            "No sex, age, or ethnicity bias analysis performed",
        ],
        "metrics":          {
            "note": "Model not trained — no real metrics. "
                    "Target metrics: AUC ≥ 0.85, F1 ≥ 0.75 on held-out test set."
        },
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "ecg_image_cnn": {
        "model_name":       "ECG Image 2D-CNN Classifier (AAMI 5-class)",
        "version":          "1.0.0",
        "modality":         "ECG Waveform Image (PNG/JPG, 128×128 grayscale)",
        "task":             "5-class AAMI beat classification (N, S, V, F, Q)",
        "architecture":     "4-block 2D-CNN → GlobalAvgPool → MLP → 5-class softmax",
        "output":           "AAMI class label, confidence, clinical guidelines",
        "training_data":    (
            "Designed for MIT-BIH Arrhythmia Database ECG images (Kaggle). "
            "Dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat"
        ),
        "intended_use":     "Research on image-based ECG beat classification.",
        "out_of_scope":     [
            "Rhythm analysis (requires waveform data, not images)",
            "12-lead interpretation from a single image",
        ],
        "limitations":      [
            "Image quality and scan resolution heavily affect accuracy",
            "MIT-BIH dataset is from a single centre (Beth Israel Hospital, 1970s recordings)",
        ],
        "known_biases":     ["MIT-BIH is skewed toward male patients aged 23–89"],
        "metrics":          {
            "reference_accuracy": "~99% reported on MIT-BIH (imbalanced classes)",
            "note": "High accuracy on MIT-BIH does NOT imply real-world generalisability",
        },
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "heart_sound_cnn": {
        "model_name":       "Heart Sound 2D-CNN Murmur Detector",
        "version":          "1.0.0",
        "modality":         "Phonocardiogram Audio (WAV, 4000 Hz)",
        "task":             "3-class murmur detection (Present / Absent / Unknown)",
        "architecture":     "Mel spectrogram (64 mels) → 3-block 2D-CNN → MLP → 3-class softmax",
        "output":           "Murmur class, confidence, clinical guidelines",
        "training_data":    (
            "Designed for CirCor DigiScope Heart Sound Dataset (PhysioNet 2022). "
            "Dataset: https://physionet.org/content/circor-heart-sound/1.0.3/"
        ),
        "intended_use":     "Research on automated cardiac auscultation.",
        "out_of_scope":     [
            "Specific structural diagnosis (MVR, AS, etc.)",
            "Paediatric congenital heart disease screening without retraining",
        ],
        "limitations":      [
            "Mel spectrogram parameters (n_mels=64) are heuristic",
            "Background noise and recording quality significantly affect performance",
        ],
        "known_biases":     ["CirCor dataset skewed toward paediatric patients"],
        "metrics":          {
            "note": "Model not trained — no real metrics. "
                    "CirCor challenge winner achieved F1=0.79 on validation set."
        },
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "vfdb_detector": {
        "model_name":       "VFDB Ventricular Arrhythmia 1D-CNN",
        "version":          "1.0.0",
        "modality":         "Raw ECG Signal (PhysioNet WFDB format, 2 leads)",
        "task":             "Ventricular arrhythmia detection (16-class rhythm classification)",
        "architecture":     "3-block 1D-CNN → GlobalAvgPool → MLP → 16-class softmax",
        "output":           "Rhythm class, is_dangerous flag, confidence, clinical guidelines",
        "training_data":    (
            "Designed for MIT-BIH Malignant Ventricular Ectopy Database (VFDB). "
            "Dataset: https://physionet.org/content/vfdb/1.0.0/ (22 half-hour recordings)"
        ),
        "intended_use":     "Research on ventricular fibrillation and tachycardia detection.",
        "out_of_scope":     [
            "Real-time ICU alarm fatigue reduction without extensive clinical validation",
            "Ambulatory monitoring (VFDB is from ICU/telemetry patients only)",
        ],
        "limitations":      [
            "Only 22 recordings in VFDB — extremely small dataset for deep learning",
            "Class imbalance: Normal rhythm dominates",
            "No file upload support — dataset-indexed only",
        ],
        "known_biases":     ["VFDB is entirely from adults in critical care settings"],
        "metrics":          {"note": "Model not trained — no real metrics available."},
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "arthritis_bert_moe": {
        "model_name":       "Arthritis Risk Predictor (Tabular BERT + MoE)",
        "version":          "1.0.0",
        "modality":         "Tabular blood test features (22 fields)",
        "task":             "Arthritis risk binary classification",
        "architecture":     "Tabular BERT embedding → Mixture of Experts (MoE) → risk score",
        "output":           "Risk label, probability, top contributing features",
        "training_data":    (
            "Arthritis Profile Dataset (APD) — 102 patients, 22 blood markers. "
            "Source: Kaggle (santhoshkumarsundar/arthritis-profile-dataset)"
        ),
        "intended_use":     "Research on tabular clinical data classification.",
        "out_of_scope":     [
            "Specific arthritis subtype diagnosis (RA, OA, gout, etc.)",
            "Treatment recommendation",
            "Paediatric patients",
        ],
        "limitations":      [
            "Only 102 patients — severely underpowered for a clinical study",
            "No external validation cohort",
            "Missing data handled by zero-imputation — may introduce bias",
            "All fields Optional[float] with no bounds (validation gap)",
        ],
        "known_biases":     [
            "Small, single-centre dataset of unknown demographic composition",
            "No sex- or age-stratified performance reported",
        ],
        "metrics":          {
            "reported_accuracy": "See /api/arthritis/train for live metrics",
            "note": "Accuracy alone is misleading on imbalanced datasets — check F1 and AUC.",
        },
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },

    # ─────────────────────────────────────────────────────────────────────────
    "cardiofusion": {
        "model_name":       "CardioFusion Multi-Modal Transformer",
        "version":          "1.0.0",
        "modality":         "Echo Video + ECG Image + Heart Sound (any combination)",
        "task":             "6-head multi-task cardiac assessment",
        "architecture":     (
            "Modality encoders (3D-CNN / 2D-CNN / mel-CNN) → "
            "Cross-modal Transformer fusion → 6 task heads "
            "(EF regression, arrhythmia, murmur, rhythm, risk score, report)"
        ),
        "output":           "6 simultaneous clinical assessments + fusion confidence",
        "training_data":    "Not trained — architecture demonstration only.",
        "intended_use":     "Research demonstration of multi-modal cardiac AI fusion.",
        "out_of_scope":     ["Any clinical use"],
        "limitations":      [
            "Random weights — all outputs are meaningless without training",
            "No training data, loss function, or optimisation defined",
            "No evaluation metrics available",
            "Fusion quality untested on real multi-modal datasets",
        ],
        "known_biases":     ["Not applicable — model not trained"],
        "metrics":          {"note": "No evaluation performed."},
        "mlflow_run_id":    None,
        "disclaimer":       CLINICAL_DISCLAIMER,
    },
}


def get_all_cards() -> Dict[str, Dict]:
    """Return summary of all model cards (without full detail fields)."""
    return {
        key: {
            "model_name": card["model_name"],
            "modality":   card["modality"],
            "task":       card["task"],
            "version":    card["version"],
        }
        for key, card in MODEL_CARDS.items()
    }


def get_card(model_id: str) -> Dict[str, Any]:
    """Return the full model card for a given model_id, or raise KeyError."""
    if model_id not in MODEL_CARDS:
        raise KeyError(f"No model card found for '{model_id}'. "
                       f"Available: {list(MODEL_CARDS.keys())}")
    return MODEL_CARDS[model_id]
