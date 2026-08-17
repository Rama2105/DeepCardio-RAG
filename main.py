
import time
import torch
import numpy as np
import os
import tempfile
import shutil
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, List

# ── Internal modules ──────────────────────────────────────────────────────────
from config import settings
from core.logger import get_logger, get_audit_logger, LoggedTimer
from core.exceptions import (
    DataValidationError, UnsupportedFileTypeError, FileTooLargeError,
    ModelInferenceError, DatasetNotFoundError, RecordNotFoundError,
    EmptyVideoError, EmptyAudioError, wrap_exception,
)
from core.model_cards import get_all_cards, get_card, MODEL_CARDS
from auth import router as auth_router, get_current_user, get_optional_user, require_role, UserOut
from core.pipeline import get_model, describe_ecg_encoder, GROUNDED_RETRIEVAL_SOURCES
from database.db_manager import get_db_status
from core.arthritis_pipeline import get_arthritis_loader, get_arthritis_predictor
from core.heart_disease_pipeline import get_heart_disease_loader, get_heart_disease_predictor, FEATURE_COLUMNS as HD_FEATURES
from core.ecg_arrhythmia_pipeline import (
    get_ecg_arrhythmia_loader, get_ecg_arrhythmia_predictor,
    ARRHYTHMIA_CLASSES as MITBIH_CLASSES, SIGNAL_LENGTH as MITBIH_SIGNAL_LEN,
)
from core.pdf_generator import generate_ecg_report, generate_arthritis_report
from core.echonet_loader import get_echonet_loader, EchoNetDataset, NUM_FRAMES, FRAME_SIZE, MEAN, STD
from core.echonet_pipeline import get_echonet_pipeline, demo_inference
from core.ecg_image_loader import (
    get_ecg_image_dataset, get_ecg_image_classifier,
    CLASS_LABELS, AAMI_CLASSES, ECG_IMAGE_GUIDELINES, IMG_SIZE,
)
from core.ptbxl_loader import get_ptbxl_dataset, SUPERCLASS_INFO as PTBXL_SUPERCLASS_INFO
from core.heart_sound_loader import (
    get_heart_sound_dataset, get_heart_sound_classifier,
    HEART_SOUND_GUIDELINES, MURMUR_CLASSES,
)
from core.vfdb_loader import (
    get_vfdb_dataset, get_vfdb_detector,
    VFDB_GUIDELINES, ALL_RHYTHMS, DANGEROUS_RHYTHMS,
)
from core.hybrid_model import get_cardiofusion_model, CardioFusionTrainer
from core.cardiac_arrhythmia_video_loader import (
    get_arrhythmia_video_dataset, get_arrhythmia_video_classifier,
    ARRHYTHMIA_CLASSES, ARRHYTHMIA_GUIDELINES,
)
from core.langchain_rag import query_rag, retrieve_guidelines, get_vector_store_stats
from core.langgraph_agent import run_diagnostic_agent, get_graph_info
from core.metrics import MetricsCalculator, corpus_bleu4, corpus_rouge_l, hallucination_rate
from core.jh_report_generator import (
    generate_jh_ecg_report, demo_jh_report,
    PatientInfo, PhysicianInfo, ECGFindings, AIMetrics, ECGParameters,
    JHReportGenerator,
)

logger      = get_logger(__name__)
audit_log   = get_audit_logger()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepCardio-RAG API",
    description=(
        "Multi-modal cardiac AI diagnostic system. "
        "Supports ECG signals, echocardiogram videos, ECG images, "
        "heart sounds, arrhythmia video classification, and arthritis risk prediction.\n\n"
        "⚠ **RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.**"
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Auth router ───────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

# ── CORS (configured from settings, not hardcoded *) ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request latency middleware (Performance Monitoring — #14) ─────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method":     request.method,
            "path":       request.url.path,
            "status":     response.status_code,
            "latency_ms": latency_ms,
        }
    )
    response.headers["X-Response-Time-Ms"] = str(latency_ms)
    return response

# Explicitly ensure the frontend directory exists to avoid crash on mount
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)

# Mount frontend directory
app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ==============================================================================
# ECG Analysis Endpoints
# ==============================================================================
class InferenceResponse(BaseModel):
    inference_time_seconds: float
    report: str
    retrieved_guidelines: list[str]
    bleu_score_approx: float  # Always 0.0 — placeholder only, not a measured metric
    disclaimer: str = "RESEARCH PROTOTYPE: Model uses random weights. Outputs are not clinically validated."
    source: dict | None = None
    # Provenance of the retrieved guidelines. Consumers must not present
    # retrieved text as evidence without checking these: 'demo-fallback' means
    # the guidelines are canned placeholders unrelated to this ECG.
    retrieval: dict | None = None


def _retrieval_provenance(results: dict) -> dict:
    """
    Describe how the guidelines for sample 0 were obtained, so the caller can
    tell diagnosis-grounded Milvus hits from canned demo text.
    """
    source    = results.get("retrieval_source", "demo-fallback")
    diagnosis = (results.get("diagnoses") or [None])[0]
    grounded  = source in GROUNDED_RETRIEVAL_SOURCES
    return {
        "source":        source,
        "grounded":      grounded,
        "query":         (results.get("retrieval_queries") or [None])[0],
        "query_basis":   "PTB-XL diagnostic head (trained)" if diagnosis else "none — no trained diagnostic head",
        "predicted_diagnosis": diagnosis,
        "scores":        [m.get("score") for m in (results.get("context_meta") or [[]])[0]],
        "note": (
            ("Guidelines retrieved from the vector DB by reciprocal-rank fusion of "
             "semantic and lexical search against the predicted diagnosis."
             if source == "hybrid-rrf" else
             "Guidelines retrieved from the vector DB by semantic similarity to the "
             "predicted diagnosis.")
            if grounded else
            "PLACEHOLDER GUIDELINES — not retrieved for this ECG. Shown because no "
            "grounded diagnosis was available or the vector DB was unreachable."
        ),
    }

@app.post(
    "/api/analyze",
    response_model=InferenceResponse,
    tags=["ECG Analysis"],
    summary="Run ECG analysis on a real PTB-XL clinical ECG record",
    description=(
        "Encodes a real 12-lead ECG waveform sampled from the PhysioNet PTB-XL dataset "
        "(falls back to a synthetic signal if PTB-XL isn't downloaded), retrieves clinical "
        "guidelines via RAG, and generates a narrative report with GPT-2. "
        "⚠ Uses random model weights — outputs are for demonstration only."
    ),
)
async def analyze_ecg():
    try:
        with LoggedTimer(logger, "ECG analysis", extra={"mode": "ptbxl"}):
            start_time = time.time()

            ptbxl = get_ptbxl_dataset()
            signal_tensor, ground_truth_label, record_meta = ptbxl.get_random_sample()
            ecg_signal = signal_tensor.unsqueeze(0)  # (1, 12, ECG_LENGTH)

            model = get_model()
            with torch.no_grad():
                results = model(ecg_signal)
            inference_time = round(time.time() - start_time, 2)

            source_info = {
                "dataset": "PTB-XL Clinical 12-Lead ECG (PhysioNet)" if record_meta.get("ecg_id") is not None
                           else "PTB-XL-style synthetic fallback (dataset not downloaded)",
                "ecg_id": record_meta.get("ecg_id"),
                "ground_truth_superclass": ground_truth_label,
                "ground_truth_label": PTBXL_SUPERCLASS_INFO.get(ground_truth_label, {}).get("name", ground_truth_label),
                "all_superclasses": record_meta.get("superclasses", [ground_truth_label]),
                "real_signal": record_meta.get("ecg_id") is not None,
            }

        return InferenceResponse(
            inference_time_seconds=inference_time,
            report=results["reports"][0],
            retrieved_guidelines=results["contexts"][0],
            bleu_score_approx=0.0,   # Not a real measurement — see model card for details
            source=source_info,
            retrieval=_retrieval_provenance(results),
        )
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.post(
    "/api/analyze/upload",
    tags=["ECG Analysis"],
    summary="Run ECG analysis on uploaded ECG file (CSV or JSON)",
    description=(
        "Accepts a 12-lead ECG file in CSV (rows=samples, 12 columns) or "
        "JSON (list of 12 lead arrays) format. Pads/truncates to 5000 samples. "
        "⚠ RESEARCH PROTOTYPE — not for clinical use."
    ),
)
async def analyze_ecg_upload(file: UploadFile = File(...)):
    """Parse uploaded ECG file, run Transformer-MoE encoder + RAG + GPT-2."""
    import io
    import pandas as pd
    try:
        content = await file.read()
        filename = file.filename or ""

        # ── Parse ECG signal ──────────────────────────────────────────────────
        if filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            # Accept: [[lead0_samples...], [lead1_samples...], ...] (12 × N)
            # or {"leads": [[...], ...]} or flat list (single lead repeated)
            if isinstance(data, dict):
                data = data.get("leads", data.get("signal", []))
            if isinstance(data[0], (int, float)):
                # Single-lead: replicate to 12 leads
                data = [data] * 12
            ecg_np = np.array(data, dtype=np.float32)   # (12, N)
        else:
            # CSV: each row = one time sample, each column = one lead
            # Also handle single-column (single-lead) CSV
            df = pd.read_csv(io.BytesIO(content), header=None)
            arr = df.values.astype(np.float32)
            if arr.shape[1] == 12:
                ecg_np = arr.T                           # (12, N)
            elif arr.shape[0] == 12:
                ecg_np = arr                             # (12, N)
            else:
                # Treat first column as single-lead, replicate
                ecg_np = np.tile(arr[:, 0:1].T, (12, 1))

        # ── Normalise & tensorise ─────────────────────────────────────────────
        # Z-score per lead
        mean = ecg_np.mean(axis=1, keepdims=True)
        std  = ecg_np.std(axis=1, keepdims=True) + 1e-8
        ecg_np = (ecg_np - mean) / std

        ecg_tensor = torch.from_numpy(ecg_np).unsqueeze(0)  # (1, 12, N)

        # ── Inference ─────────────────────────────────────────────────────────
        start_time = time.time()
        model = get_model()
        with torch.no_grad():
            results = model(ecg_tensor)
        inference_time = round(time.time() - start_time, 2)

        n_samples = ecg_np.shape[1]
        duration_sec = round(n_samples / 500, 1)   # assume 500 Hz

        return JSONResponse(content={
            "inference_time_seconds": inference_time,
            "report": results["reports"][0],
            "retrieved_guidelines": results["contexts"][0],
            "retrieval": _retrieval_provenance(results),
            "bleu_score_approx": 0.0,
            "upload_info": {
                "filename": filename,
                "leads_detected": int(ecg_np.shape[0]),
                "samples": n_samples,
                "duration_seconds": duration_sec,
                "sampling_rate_assumed": 500,
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"ECG parse/inference error: {str(e)}")


@app.get(
    "/api/analyze/sample-ecg",
    tags=["ECG Analysis"],
    summary="Download a sample 12-lead ECG CSV for testing the upload",
)
async def sample_ecg_csv():
    """Returns a 5-second synthetic 12-lead ECG as CSV (2500 rows × 12 columns)."""
    import io as _io
    t = np.linspace(0, 5, 2500)
    leads = []
    freqs = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
    for i, f in enumerate(freqs):
        base  = np.sin(2 * np.pi * f * t)
        qrs   = 1.5 * np.exp(-((t % (1/f) - 0.3/(f))**2) / 0.002)
        noise = 0.05 * np.random.randn(len(t))
        leads.append(base + qrs + noise)
    arr = np.stack(leads, axis=1)   # (2500, 12)
    buf = _io.StringIO()
    np.savetxt(buf, arr, delimiter=",", fmt="%.6f")
    csv_bytes = buf.getvalue().encode()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_ecg_12lead.csv"},
    )


@app.get(
    "/api/ecg/ptbxl-eda",
    tags=["ECG Analysis"],
    summary="PTB-XL real ECG dataset statistics (records, diagnostic superclasses)",
)
async def ptbxl_eda():
    """Return PTB-XL dataset EDA statistics — record counts and diagnostic-superclass distribution."""
    try:
        ds = get_ptbxl_dataset()
        return JSONResponse(content=ds.get_eda_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Arthritis Dataset Endpoints
# ==============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    return {"status": "ok", "version": settings.app_version, "env": settings.environment}


# ──────────────────────────────────────────────────────────────────────────────
# Model Cards (#12 Transparency)
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/api/model-cards",
    tags=["Model Cards"],
    summary="List all model cards",
    description="Returns a summary of all model cards documenting architecture, data, and limitations.",
)
async def list_model_cards():
    return JSONResponse(content=get_all_cards())


@app.get(
    "/api/model-cards/{model_id}",
    tags=["Model Cards"],
    summary="Get a specific model card",
    responses={404: {"description": "Model card not found"}},
)
async def get_model_card(model_id: str):
    try:
        return JSONResponse(content=get_card(model_id))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Input validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, allowed_exts: List[str], label: str = "file"):
    """Validate file extension for uploads."""
    if not file or not file.filename:
        raise DataValidationError(f"No {label} provided")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise UnsupportedFileTypeError(
            f"Unsupported {label} type '{ext}'",
            detail=f"Allowed: {allowed_exts}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# PatientInput with field validation (#4 Input Validation)
# ──────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """
    Inputs the arthritis model ACTUALLY consumes (NHANES Medical Conditions
    Questionnaire, the dataset core/arthritis_pipeline.py trains on).

    This schema previously asked for 22 APD haematology values (TC, ESRh, Hb,
    ASO, Uric_Acid, …). Those were left over from the APD→NHANES dataset switch:
    only Age and Gender_M overlap with the trained feature set, so predict()
    dropped the other 20 and back-filled every real feature with 0.0 — BMI 0,
    blood pressure 0/0. The result was a confident risk score that did not move
    no matter what a clinician typed. Keep this schema in step with
    `feature_cols` in data/arthritis_bert_moe_scaler.pkl.

    The remaining 7 model features (age_gender, obesity_flag, bmi_squared,
    age_risk, age_squared, pulse_pressure, map) are derived by
    AdvancedFeatureEngineer from the fields below — do not ask for them.
    """
    Age:              Optional[float] = Field(None, ge=0,  le=120, description="Age in years (NHANES RIDAGEYR)")
    Gender_M:         Optional[float] = Field(None, ge=0,  le=1,   description="Sex (0=Female, 1=Male)")
    BMI:              Optional[float] = Field(None, ge=10, le=70,  description="Body Mass Index kg/m² (BMXBMI)")
    SystolicBP:       Optional[float] = Field(None, ge=60, le=250, description="Systolic blood pressure mmHg (BPXSY1)")
    DiastolicBP:      Optional[float] = Field(None, ge=40, le=150, description="Diastolic blood pressure mmHg (BPXDI1)")
    HasGout:          Optional[float] = Field(None, ge=0,  le=1,   description="Doctor-diagnosed gout (MCQ160N): 0=No, 1=Yes")
    HasOsteoporosis:  Optional[float] = Field(None, ge=0,  le=1,   description="Doctor-diagnosed osteoporosis (MCQ160B): 0=No, 1=Yes")
    PovertyRatio:     Optional[float] = Field(None, ge=0,  le=5,   description="Family income-to-poverty ratio (INDFMPIR), 0–5")
    EducationLevel:   Optional[float] = Field(None, ge=1,  le=5,   description="Education (DMDEDUC2): 1=<9th grade … 5=College graduate")

    def to_features(self) -> dict:
        """
        Payload for the predictor, with InflammationProxy derived exactly as the
        NHANES loader derives it at training time (arthritis_pipeline.py:270-273).
        It is a computed column, not something a clinician can measure, so it is
        not a form field — but omitting it would silently feed the model a 0.
        """
        data = self.model_dump()
        if self.HasGout is not None or self.HasOsteoporosis is not None:
            data["InflammationProxy"] = ((self.HasGout or 0.0) + (self.HasOsteoporosis or 0.0)) / 2.0
        return data

@app.get("/api/arthritis/eda")
async def arthritis_eda():
    try:
        loader = get_arthritis_loader()
        return JSONResponse(content=loader.get_eda_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/arthritis/correlation")
async def arthritis_correlation():
    try:
        loader = get_arthritis_loader()
        return JSONResponse(content=loader.get_correlation_matrix())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/arthritis/train")
async def arthritis_train():
    try:
        start_time = time.time()
        loader = get_arthritis_loader()
        predictor = get_arthritis_predictor()
        metrics = predictor.train(loader.df)
        train_time = round(time.time() - start_time, 2)
        return JSONResponse(content={
            "training_time_seconds": train_time, **metrics
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/arthritis/predict")
async def arthritis_predict(patient: PatientInput):
    try:
        predictor = get_arthritis_predictor()
        result = predictor.predict(patient.to_features())
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# PDF Report Endpoints
# ==============================================================================
class ECGReportRequest(BaseModel):
    report: str = ""
    retrieved_guidelines: list[str] = []
    inference_time_seconds: float = 0.0
    audience: str = "doctor"

class ArthritisReportRequest(BaseModel):
    patient_data: dict = {}
    prediction_result: dict = {}
    train_metrics: Optional[dict] = None
    audience: str = "doctor"

@app.post("/api/pdf/ecg")
async def generate_ecg_pdf(req: ECGReportRequest):
    try:
        pdf_bytes = generate_ecg_report(
            report_text=req.report,
            guidelines=req.retrieved_guidelines,
            inference_time=req.inference_time_seconds,
            audience=req.audience
        )
        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=ECG_Clinical_Report.pdf"
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pdf/arthritis")
async def generate_arthritis_pdf(req: ArthritisReportRequest):
    try:
        loader = get_arthritis_loader()
        eda = loader.get_eda_summary()
        predictor = get_arthritis_predictor()

        pdf_bytes = generate_arthritis_report(
            prediction_result=req.prediction_result,
            patient_data=req.patient_data,
            eda_summary=eda,
            train_metrics=req.train_metrics or (predictor.metrics if predictor.is_trained else None),
            audience=req.audience
        )
        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=Arthritis_Risk_Report.pdf"
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class FusionReportRequest(BaseModel):
    doctor_report:  str = ""
    patient_report: str = ""
    overall_risk:   str = "UNKNOWN"
    risk_score:     int = 0
    models_run:     list = []
    individual_results: dict = {}
    audience: str = "doctor"   # "doctor" or "patient"

@app.post("/api/pdf/fusion", tags=["PDF Reports"], summary="Generate CardioFusion unified cardiac report PDF")
async def generate_fusion_pdf(req: FusionReportRequest):
    try:
        report_text = req.doctor_report if req.audience == "doctor" else req.patient_report
        if not report_text:
            report_text = f"CardioFusion Unified Report\nOverall Risk: {req.overall_risk}\nRisk Score: {req.risk_score}/100\nModels: {', '.join(req.models_run)}"

        # Build a structured guidelines list from individual model results
        guidelines = []
        for model, result in req.individual_results.items():
            g = result.get("guidelines", [])
            if isinstance(g, list):
                guidelines.extend(g[:2])

        pdf_bytes = generate_ecg_report(
            report_text=report_text,
            guidelines=guidelines,
            inference_time=0.0,
            audience=req.audience,
        )
        filename = f"CardioFusion_{'Doctor' if req.audience == 'doctor' else 'Patient'}_Report.pdf"
        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Patient Records / DB Stats endpoints
# ==============================================================================
@app.get("/api/patients")
async def get_patients():
    '''Return sample patient records from the loaded dataset (APD or real).'''
    try:
        loader = get_arthritis_loader()
        df = loader.df.head(20).fillna("—")
        is_apd = len(loader.df) <= 500  # APD has 102 rows; real datasets have 70K+
        records = []
        for i, row in df.iterrows():
            if is_apd:
                ra_val  = float(row.get("RA",  0) or 0)
                crp_val = float(row.get("CRP", 0) or 0)
                records.append({
                    "id": f"APD-{i+1:04d}",
                    "gender": "Male" if row.get("Gender_M", 0) == 1 else "Female",
                    "age": row.get("Age", "—"),
                    "hb": row.get("Hb", "—"),
                    "esr": row.get("ESRh", "—"),
                    "crp": crp_val if crp_val else "—",
                    "ra":  ra_val  if ra_val  else "—",
                    "uric_acid": row.get("Uric_Acid", "—"),
                    "arthritis_risk": 1 if (ra_val > 14 or crp_val > 6) else 0,
                })
            else:
                risk_val = row.get("arthritis_risk", "—")
                records.append({
                    "id": f"PT-{i+1:04d}",
                    "gender": "Male" if row.get("Gender_M", 0) == 1 else "Female",
                    "age": row.get("Age", "—"),
                    "bmi": row.get("BMI", "—"),
                    "systolic_bp": row.get("SystolicBP", "—"),
                    "inflammation": row.get("InflammationProxy", "—"),
                    # NHANES (and the APD fallback) carry no smoking variable. A bare
                    # row.get("Smoking", 0) defaulted every patient to a confident "No" —
                    # a fabricated clinical attribute. Degrade to "—" like the fields above
                    # unless the loaded dataset genuinely has the column.
                    "smoking": ("Yes" if row.get("Smoking") == 1 else "No")
                               if "Smoking" in df.columns else "—",
                    "arthritis_risk": int(risk_val) if risk_val != "—" else "—",
                })
        return JSONResponse(content={
            "patients": records,
            "total": len(loader.df),
            "dataset_type": "apd" if is_apd else "real",
            "dataset_name": loader.dataset_info.get("name", "Dataset") if loader.dataset_info else "Dataset",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_arthritis_desc_cache: Optional[str] = None


def describe_arthritis_predictor() -> str:
    """
    Describe the arthritis stack from what is actually SAVED, not from a literal.

    The hardcoded string claimed "GBM + RF + ExtraTrees + SVC", but the saved
    model's base_learners are only {gbm, rf, et} — the SVC was a model component
    that does not exist. Same failure mode as the hardcoded ecg_encoder string:
    a conditional fact frozen into text drifts the moment the model is retrained.
    """
    global _arthritis_desc_cache
    if _arthritis_desc_cache is not None:
        return _arthritis_desc_cache

    pretty = {"gbm": "GBM", "rf": "RF", "et": "ExtraTrees", "svc": "SVC",
              "lr": "LogReg", "xgb": "XGBoost"}
    base, meta = [], None
    try:
        from core.arthritis_pipeline import SCALER_SAVE_PATH
        if os.path.isfile(SCALER_SAVE_PATH):
            saved = joblib.load(SCALER_SAVE_PATH)
            base = list((saved.get("base_learners") or {}).keys())
            if saved.get("meta_learner") is not None:
                meta = type(saved["meta_learner"]).__name__
    except Exception as exc:
        logger.warning(f"Could not read arthritis model for description: {exc}")

    if not base:
        desc = "TabularBERT-MoE (no stacking ensemble saved — BERT-MoE only)"
    else:
        names = " + ".join(pretty.get(b, b.upper()) for b in base)
        desc = f"Hybrid Stack: TabularBERT-MoE + {names}"
        desc += f" → {meta} meta-learner" if meta else " (no meta-learner saved)"

    _arthritis_desc_cache = desc
    return desc


def live_model_descriptions() -> dict:
    """Model card shared by /api/db/stats and /api/db/stats/full, read live."""
    return {
        "ecg_encoder": describe_ecg_encoder(),
        "report_generator": "GPT-2 (124M params) with soft-prompt injection — outputs may be incoherent",
        "arthritis_predictor": describe_arthritis_predictor(),
        "disclaimer": "All models are research prototypes. Not validated for clinical use.",
    }


@app.get("/api/db/stats")
async def get_db_stats():
    '''Return vector database and dataset statistics.'''
    try:
        loader = get_arthritis_loader()
        eda = loader.get_eda_summary()
        # Live read of whichever backend init_db() actually landed on. Never
        # hardcode these — the fallback chain is silent, so a hardcoded
        # "Online" would read Online with Milvus stopped or never installed.
        db = get_db_status()
        return JSONResponse(content={
            "vector_db": db,
            "milvus_status": db["status"],
            "milvus_backend": db["backend"],
            "milvus_connected": db["connected"],
            "milvus_host": db["host"],
            "milvus_collection": db["collection"],
            "milvus_records": f"{db['documents']} documents indexed",
            "milvus_document_count": db["documents"],
            "milvus_embedding_dim": db["embedding_dim"],
            "milvus_metric": db["metric"],
            "milvus_index_type": db["index_type"],
            # Name/source/format were hardcoded to APD + Excel while the live counts
            # below came from whatever loader actually loaded — so NHANES data was
            # served under APD branding, contradicting its own numbers. Read the loader.
            "apd_dataset": {
                "name": eda.get("dataset_name", "Arthritis dataset"),
                "source": eda.get("dataset_url", "—"),
                "total_records": eda["total_samples"],
                "total_features": eda["total_features"],
                "file_format": "Excel (.xlsx)" if eda.get("is_fallback") else "CSV (Kaggle)",
                "is_fallback": eda.get("is_fallback", False),
                "missing_data_summary": {k: v for k, v in eda["missing_percentage"].items() if v > 0}
            },
            # Read live — these were hardcoded ("random weights (not trained)" for the
            # encoder, an SVC the saved model never contained for the arthritis stack).
            # Do not re-hardcode.
            "models": live_model_descriptions(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# EchoNet-Dynamic Video Analysis Endpoints
# ==============================================================================
class EchoAnalysisResponse(BaseModel):
    inference_time_seconds: float
    # Optional because an untrained model must report NOTHING rather than a number.
    ef_predicted: Optional[float] = None
    ef_category: str
    report: str
    retrieved_guidelines: list[str]
    mode: str  # "demo" or "real"
    model_trained: bool = False
    warning: Optional[str] = None


_ECHO_UNTRAINED_WARNING = (
    "No trained EchoNet weights are present (data/echonet_ef_model.pt is missing), "
    "so this module cannot measure an ejection fraction. Train it on EchoNet-Dynamic "
    "(Stanford AIMI access required) before any EF is reported."
)


def guard_echo_result(result: dict) -> dict:
    """
    Suppress EF output when the echo model is untrained.

    Without this the untrained regressor emitted a DETERMINISTIC 'LVEF 0.0%' —
    an ejection fraction incompatible with life — categorised HFrEF and paired
    with genuine ACC/AHA text recommending GDMT. `encoder_is_trained` was already
    computed in echonet_pipeline.py:276 but never read by anything: the check
    existed, nothing was wired to it. Never render a clinical value from weights
    that were never trained.
    """
    pipeline = get_echonet_pipeline()
    trained = bool(getattr(pipeline, "encoder_is_trained", False))

    out = dict(result)
    out["model_trained"] = trained
    if trained:
        out["warning"] = None
        return out

    out["ef_predicted"] = None
    out["ef_category"] = "Unavailable"
    out["report"] = (
        "ECHOCARDIOGRAM ANALYSIS UNAVAILABLE\n"
        "========================================\n"
        f"{_ECHO_UNTRAINED_WARNING}\n\n"
        "No findings, LVEF value, or treatment recommendation can be produced."
    )
    # Drop the guidelines too — real ACC/AHA text next to a fabricated EF is what
    # made the old output read as a genuine clinical assessment.
    out["retrieved_guidelines"] = []
    out["warning"] = _ECHO_UNTRAINED_WARNING
    return out

@app.get("/api/echonet/eda")
async def echonet_eda():
    """Return EchoNet-Dynamic dataset EDA statistics."""
    try:
        loader = get_echonet_loader()
        return JSONResponse(content=loader.get_eda_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echonet/analyze/demo")
async def echonet_analyze_demo():
    """Run inference with a synthetic echo video (no dataset needed)."""
    try:
        result = guard_echo_result(demo_inference())
        return EchoAnalysisResponse(
            inference_time_seconds=result["inference_time_seconds"],
            ef_predicted=result["ef_predicted"],
            ef_category=result["ef_category"],
            report=result["report"],
            retrieved_guidelines=result["retrieved_guidelines"],
            mode="demo",
            model_trained=result["model_trained"],
            warning=result["warning"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echonet/analyze/upload")
async def echonet_analyze_upload(video: UploadFile = File(...)):
    """
    Analyze an uploaded echocardiogram video (.avi).
    Reads the video, preprocesses it, and runs the full EchoNet-RAG pipeline.
    """
    try:
        # Save upload to temp file
        suffix = os.path.splitext(video.filename or "video.avi")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        # Read and preprocess video
        try:
            import cv2
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="opencv-python is required for video uploads. "
                       "Install with: pip install opencv-python-headless"
            )

        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
            frames.append(resized.astype(np.float32) / 255.0)
        cap.release()
        os.unlink(tmp_path)

        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="Could not read any frames from the uploaded video.")

        # Uniform temporal sampling
        total = len(frames)
        if total >= NUM_FRAMES:
            indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        else:
            indices = list(range(total)) + [total - 1] * (NUM_FRAMES - total)

        sampled = np.stack([frames[i] for i in indices], axis=0)
        sampled = (sampled - MEAN) / STD
        video_tensor = torch.from_numpy(sampled).unsqueeze(0).unsqueeze(0).float()
        # Shape: (1, 1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)

        pipeline = get_echonet_pipeline()
        result = guard_echo_result(pipeline.analyze_single(video_tensor))

        return EchoAnalysisResponse(
            inference_time_seconds=result["inference_time_seconds"],
            ef_predicted=result["ef_predicted"],
            ef_category=result["ef_category"],
            report=result["report"],
            retrieved_guidelines=result["retrieved_guidelines"],
            mode="real",
            model_trained=result["model_trained"],
            warning=result["warning"],
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/echonet/analyze/dataset/{index}")
async def echonet_analyze_from_dataset(index: int):
    """Analyze a specific video from the loaded EchoNet-Dynamic dataset by index."""
    try:
        loader = get_echonet_loader()
        if len(loader) == 0:
            raise HTTPException(
                status_code=404,
                detail="EchoNet-Dynamic dataset not found. Download it from Stanford AIMI first."
            )
        if index < 0 or index >= len(loader):
            raise HTTPException(status_code=400, detail=f"Index {index} out of range (0-{len(loader)-1})")

        video_tensor, sample = loader.load_video_tensor(index)
        video_tensor = video_tensor.unsqueeze(0).float()  # (1, 1, T, H, W)

        pipeline = get_echonet_pipeline()
        result = guard_echo_result(pipeline.analyze_single(video_tensor))

        return JSONResponse(content={
            **result,
            "mode": "dataset",
            "ground_truth": {
                "ef": sample.get("ef"),
                "esv": sample.get("esv"),
                "edv": sample.get("edv"),
            },
            "filename": sample.get("filename"),
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/echonet/dataset/browse")
async def echonet_browse_dataset(page: int = 0, page_size: int = 20):
    """Browse the loaded EchoNet-Dynamic dataset with pagination."""
    try:
        loader = get_echonet_loader()
        total = len(loader)
        start = page * page_size
        end = min(start + page_size, total)

        samples = []
        for i in range(start, end):
            s = loader.samples[i]
            samples.append({
                "index": i,
                "filename": s.get("filename"),
                "ef": s.get("ef"),
                "esv": s.get("esv"),
                "edv": s.get("edv"),
                "split": s.get("split"),
            })

        return JSONResponse(content={
            "samples": samples,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "dataset_loaded": total > 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# Updated DB Stats to include EchoNet info
# ==============================================================================
@app.get("/api/db/stats/full")
async def get_full_db_stats():
    """Return combined stats for all datasets and models."""
    try:
        loader = get_arthritis_loader()
        eda = loader.get_eda_summary()
        echonet = get_echonet_loader()
        echo_eda = echonet.get_eda_summary()
        db = get_db_status()   # see /api/db/stats — live backend, not hardcoded

        return JSONResponse(content={
            "vector_db": db,
            "milvus_status": db["status"],
            "milvus_backend": db["backend"],
            "milvus_connected": db["connected"],
            "milvus_host": db["host"],
            "milvus_collection": db["collection"],
            "milvus_records": f"{db['documents']} documents indexed",
            "milvus_document_count": db["documents"],
            "milvus_embedding_dim": db["embedding_dim"],
            "milvus_metric": db["metric"],
            "milvus_index_type": db["index_type"],
            "apd_dataset": {
                # See /api/db/stats — same hardcoded-APD-branding fix.
                "name": eda.get("dataset_name", "Arthritis dataset"),
                "source": eda.get("dataset_url", "—"),
                "total_records": eda["total_samples"],
                "total_features": eda["total_features"],
                "is_fallback": eda.get("is_fallback", False),
            },
            # EchoNet-Dynamic is NOT present on this machine (data/genuine_results.json
            # "echo" holds a not-found error; the dataset needs Stanford AIMI access).
            # These counts are the published reference figures the loader falls back to —
            # this endpoint used to strip the loader's own "note" disclaimer and serve
            # them looking live. Carry the note and an explicit flag instead.
            "echonet_dataset": {
                "name": echo_eda.get("dataset", "EchoNet-Dynamic"),
                "source": "Stanford AIMI",
                "is_demo": "DEMO" in str(echo_eda.get("dataset", "")).upper(),
                "note": echo_eda.get(
                    "note", "Reference statistics — dataset not available locally."),
                "total_videos": echo_eda.get("total_videos", 0),
                "splits": echo_eda.get("splits", {}),
                "ef_categories": echo_eda.get("ef_categories", {}),
                "video_format": echo_eda.get("video_format", "AVI"),
                "frame_sampling": echo_eda.get("frame_sampling", ""),
            },
            # Was a second, staler copy of the hardcoded model card — it still said
            # "1D-CNN (3 conv blocks, 256-dim)" (wrong architecture, no training status)
            # long after the encoder became a PTB-XL-trained ECGTransformerMoE. One live
            # source now serves both endpoints.
            "models": {
                **live_model_descriptions(),
                "echo_encoder": "3D-CNN R2+1D (4 ST-blocks, 384-dim) — untrained, no EchoNet weights",
                "ef_regressor": "MLP (384→128→1) — untrained, no EchoNet weights",
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ECG Image Classification Endpoints (Kaggle MIT-BIH Images)
# ==============================================================================
@app.get("/api/ecg-images/eda")
async def ecg_images_eda():
    """Return ECG Image dataset EDA statistics."""
    try:
        ds = get_ecg_image_dataset()
        return JSONResponse(content=ds.get_eda_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_ECG_IMG_UNTRAINED_WARNING = (
    "No trained ECG-image classifier is present (data/ecg_image_classifier.pt is "
    "missing), so this module cannot classify a beat. The network is randomly "
    "initialised; any class label or confidence it produces is meaningless."
)


def guard_ecg_image_result(result: dict, guidelines: list) -> dict:
    """
    Suppress beat classification when the image classifier has random weights.

    Same defect as the EchoNet EF guard: a randomly-initialised softmax still
    returns a confident-looking class ("Q — Unclassifiable, 41.9%"), which was
    then paired with real guideline text recommending electrophysiologist review.
    """
    classifier = get_ecg_image_classifier()
    trained = bool(getattr(classifier, "is_trained", False))
    out = dict(result)
    out["model_trained"] = trained
    if trained:
        out["retrieved_guidelines"] = guidelines
        out["warning"] = None
        return out

    out["class_label"] = "—"
    out["class_name"] = "Unavailable (model untrained)"
    out["description"] = _ECG_IMG_UNTRAINED_WARNING
    out["confidence"] = None
    out["probabilities"] = {}
    out["retrieved_guidelines"] = []
    out["warning"] = _ECG_IMG_UNTRAINED_WARNING
    return out


@app.post("/api/ecg-images/classify/demo")
async def ecg_images_classify_demo():
    """Classify a random/synthetic ECG image (demo mode)."""
    try:
        ds = get_ecg_image_dataset()
        classifier = get_ecg_image_classifier()

        if len(ds) > 0:
            tensor, cls_idx, cls_label = ds.get_random_sample()
            tensor = tensor.unsqueeze(0)  # (1, 1, H, W)
            mode = "dataset"
        else:
            tensor = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
            cls_label = "N"
            mode = "synthetic"

        results = classifier.predict(tensor)
        result = results[0]
        guidelines = ECG_IMAGE_GUIDELINES.get(result["class_label"], [])

        return JSONResponse(content={
            **guard_ecg_image_result(result, guidelines),
            "mode": mode,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ecg-images/classify/upload")
async def ecg_images_classify_upload(image: UploadFile = File(...)):
    """Classify an uploaded ECG waveform image."""
    try:
        from PIL import Image as PILImage
        import io

        contents = await image.read()
        img = PILImage.open(io.BytesIO(contents)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        classifier = get_ecg_image_classifier()
        results = classifier.predict(tensor)
        result = results[0]
        guidelines = ECG_IMAGE_GUIDELINES.get(result["class_label"], [])

        return JSONResponse(content={
            **guard_ecg_image_result(result, guidelines),
            "mode": "uploaded",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ecg-images/browse")
async def ecg_images_browse(page: int = 0, page_size: int = 20):
    """Browse the ECG image dataset."""
    try:
        ds = get_ecg_image_dataset()
        total = len(ds)
        start = page * page_size
        end = min(start + page_size, total)
        samples = []
        for i in range(start, end):
            filepath, cls_idx = ds.samples[i]
            samples.append({
                "index": i,
                "filename": os.path.basename(filepath),
                "class_label": CLASS_LABELS[cls_idx],
                "class_name": AAMI_CLASSES[cls_idx]["name"],
            })
        return JSONResponse(content={
            "samples": samples, "total": total, "page": page,
            "page_size": page_size, "dataset_loaded": total > 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# CirCor Heart Sound Endpoints
# ==============================================================================
@app.get("/api/heart-sound/eda")
async def heart_sound_eda():
    """Return CirCor heart sound dataset EDA statistics."""
    try:
        ds = get_heart_sound_dataset()
        return JSONResponse(content=ds.get_eda_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/heart-sound/analyze/demo")
async def heart_sound_analyze_demo():
    """Run murmur detection on synthetic or random dataset audio."""
    try:
        import time as _time
        start = _time.time()
        ds = get_heart_sound_dataset()
        classifier = get_heart_sound_classifier()

        if len(ds) > 0:
            tensor, rec = ds.load_audio_tensor(0)
            signal = tensor.squeeze(0).numpy()
            mel = ds.compute_mel_spectrogram(signal)
            mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)
            mode = "dataset"
            patient_info = rec
        else:
            mel_tensor = torch.randn(1, 1, 64, 156)
            mode = "synthetic"
            patient_info = {"patient_id": "DEMO", "location": "MV", "murmur": "Unknown"}

        results = classifier.predict(mel_tensor)
        result = results[0]
        guidelines = HEART_SOUND_GUIDELINES.get(result["murmur_class"], [])
        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={
            **result,
            "inference_time_seconds": elapsed,
            "mode": mode,
            "patient_info": {
                "patient_id": patient_info.get("patient_id", ""),
                "location": patient_info.get("location", ""),
                "age": patient_info.get("age", ""),
                "sex": patient_info.get("sex", ""),
            },
            "retrieved_guidelines": guidelines,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/heart-sound/analyze/upload")
async def heart_sound_analyze_upload(audio: UploadFile = File(...)):
    """Analyze an uploaded heart sound .wav file."""
    try:
        import time as _time
        start = _time.time()

        suffix = os.path.splitext(audio.filename or "audio.wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        ds = get_heart_sound_dataset()
        signal = ds._read_wav(tmp_path)
        os.unlink(tmp_path)

        # Pad/truncate
        target_len = int(5.0 * 4000)
        if len(signal) >= target_len:
            signal = signal[:target_len]
        else:
            signal = np.pad(signal, (0, target_len - len(signal)))

        mel = ds.compute_mel_spectrogram(signal)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)

        classifier = get_heart_sound_classifier()
        results = classifier.predict(mel_tensor)
        result = results[0]
        guidelines = HEART_SOUND_GUIDELINES.get(result["murmur_class"], [])
        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={
            **result,
            "inference_time_seconds": elapsed,
            "mode": "uploaded",
            "retrieved_guidelines": guidelines,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/heart-sound/browse")
async def heart_sound_browse(page: int = 0, page_size: int = 20):
    """Browse heart sound recordings."""
    try:
        ds = get_heart_sound_dataset()
        total = len(ds)
        start = page * page_size
        end = min(start + page_size, total)
        samples = []
        for i in range(start, end):
            r = ds.recordings[i]
            samples.append({
                "index": i, "filename": r["filename"],
                "patient_id": r["patient_id"], "location": r["location"],
                "murmur": r["murmur"], "age": r.get("age", ""),
                "sex": r.get("sex", ""),
            })
        return JSONResponse(content={
            "samples": samples, "total": total, "page": page,
            "page_size": page_size, "dataset_loaded": total > 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# VFDB Ventricular Arrhythmia Endpoints
# ==============================================================================
@app.get("/api/vfdb/eda")
async def vfdb_eda():
    """Return VFDB dataset EDA statistics."""
    try:
        ds = get_vfdb_dataset()
        return JSONResponse(content=ds.get_eda_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vfdb/analyze/demo")
async def vfdb_analyze_demo():
    """Run arrhythmia detection on synthetic or dataset ECG segment."""
    try:
        import time as _time
        start = _time.time()
        ds = get_vfdb_dataset()
        detector = get_vfdb_detector()

        if len(ds) > 0:
            tensor, rec = ds.load_segment_tensor(0, start_sec=0, duration=10)
            tensor = tensor.unsqueeze(0)  # (1, n_ch, T)
            mode = "dataset"
            record_info = {
                "record_id": rec["record_id"],
                "rhythm_types": rec["rhythm_types"],
                "dangerous_events": rec["dangerous_events"],
            }
        else:
            tensor = torch.randn(1, 2, 2500)
            mode = "synthetic"
            record_info = {"record_id": "DEMO", "rhythm_types": ["N"], "dangerous_events": 0}

        results = detector.predict(tensor)
        result = results[0]
        guidelines = VFDB_GUIDELINES.get(result["rhythm_class"], VFDB_GUIDELINES["Normal"])
        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={
            **result,
            "inference_time_seconds": elapsed,
            "mode": mode,
            "record_info": record_info,
            "retrieved_guidelines": guidelines,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vfdb/analyze/{record_index}")
async def vfdb_analyze_record(record_index: int, start_sec: float = 0, duration: float = 10):
    """Analyze a specific VFDB record segment."""
    try:
        import time as _time
        start_t = _time.time()
        ds = get_vfdb_dataset()
        if len(ds) == 0:
            raise HTTPException(status_code=404, detail="VFDB dataset not found.")
        if record_index < 0 or record_index >= len(ds):
            raise HTTPException(status_code=400, detail=f"Index out of range (0-{len(ds)-1})")

        tensor, rec = ds.load_segment_tensor(record_index, start_sec, duration)
        tensor = tensor.unsqueeze(0)
        detector = get_vfdb_detector()
        results = detector.predict(tensor)
        result = results[0]
        guidelines = VFDB_GUIDELINES.get(result["rhythm_class"], VFDB_GUIDELINES["Normal"])
        elapsed = round(_time.time() - start_t, 2)

        # Find annotations in this time window
        annotations_in_window = [
            a for a in rec["annotations"]
            if start_sec <= a["time"] <= start_sec + duration
        ]

        return JSONResponse(content={
            **result,
            "inference_time_seconds": elapsed,
            "mode": "dataset",
            "record_info": {
                "record_id": rec["record_id"],
                "leads": rec["leads"],
                "fs": rec["fs"],
                "segment": {"start_sec": start_sec, "duration": duration},
                "annotations_in_window": annotations_in_window,
            },
            "retrieved_guidelines": guidelines,
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vfdb/browse")
async def vfdb_browse():
    """Browse all VFDB records with their annotation summaries."""
    try:
        ds = get_vfdb_dataset()
        records = []
        for i, rec in enumerate(ds.records):
            records.append({
                "index": i,
                "record_id": rec["record_id"],
                "n_signals": rec["n_signals"],
                "leads": rec["leads"],
                "fs": rec["fs"],
                "rhythm_types": rec["rhythm_types"],
                "dangerous_events": rec["dangerous_events"],
                "total_annotations": len(rec["annotations"]),
            })
        return JSONResponse(content={
            "records": records,
            "total": len(records),
            "dataset_loaded": len(records) > 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# CardioFusion Hybrid Model Endpoints
# ==============================================================================
@app.get("/api/cardiofusion/info")
async def cardiofusion_info():
    """Return model architecture summary and parameters."""
    try:
        model = get_cardiofusion_model()
        trainer = CardioFusionTrainer(model)
        return JSONResponse(content=trainer.get_training_summary())
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cardiofusion/analyze")
async def cardiofusion_analyze(
    echo_video: Optional[UploadFile] = File(None),
    ecg_image: Optional[UploadFile] = File(None),
    heart_sound: Optional[UploadFile] = File(None),
):
    """
    Multi-modal cardiac analysis. Upload any combination of:
    - echo_video: .avi echocardiogram
    - ecg_image: .png/.jpg ECG waveform image
    - heart_sound: .wav phonocardiogram
    Returns unified clinical assessment from all 6 task heads.
    """
    try:
        import time as _time
        start = _time.time()
        model = get_cardiofusion_model()
        inputs = {}

        # Process echo video
        if echo_video and echo_video.filename:
            try:
                import cv2
                suffix = os.path.splitext(echo_video.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(echo_video.file, tmp)
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (112, 112))
                    frames.append(resized.astype(np.float32) / 255.0)
                cap.release(); os.unlink(tmp_path)
                if frames:
                    total = len(frames)
                    indices = np.linspace(0, total-1, 32, dtype=int) if total >= 32 else list(range(total)) + [total-1]*(32-total)
                    sampled = np.stack([frames[i] for i in indices[:32]])
                    sampled = (sampled - 0.131) / 0.289
                    inputs["echo_video"] = torch.from_numpy(sampled).unsqueeze(0).unsqueeze(0).float()
            except Exception as e:
                print(f"Echo processing error: {e}")

        # Process ECG image
        if ecg_image and ecg_image.filename:
            try:
                from PIL import Image as PILImage
                import io
                contents = await ecg_image.read()
                img = PILImage.open(io.BytesIO(contents)).convert("L").resize((128, 128))
                arr = np.array(img, dtype=np.float32) / 255.0
                inputs["ecg_image"] = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            except Exception as e:
                print(f"ECG image processing error: {e}")

        # Process heart sound
        if heart_sound and heart_sound.filename:
            try:
                suffix = os.path.splitext(heart_sound.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(heart_sound.file, tmp)
                    tmp_path = tmp.name
                ds = get_heart_sound_dataset()
                signal = ds._read_wav(tmp_path)
                os.unlink(tmp_path)
                target_len = int(5.0 * 4000)
                signal = signal[:target_len] if len(signal) >= target_len else np.pad(signal, (0, target_len - len(signal)))
                mel = ds.compute_mel_spectrogram(signal)
                inputs["heart_sound"] = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)
            except Exception as e:
                print(f"Heart sound processing error: {e}")

        if not inputs:
            raise HTTPException(status_code=400, detail="No valid input files provided.")

        result = guard_cardiofusion_result(model.inference(inputs))
        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={
            "inference_time_seconds": elapsed,
            **result,
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Only the ECG-signal arrhythmia head was ever trained. data/genuine_results.json
# records modality_trained="ecg_signal", learned_task_weights showing ONLY
# arrhythmia moved off its 0.5 init (0.4196), and the note "Joint 4-modality
# fusion + contrastive alignment NOT trained — requires same-patient paired data".
# The other heads are random: they produced ejection_fraction -0.1% labelled
# "Normal" (a negative EF), a rhythm class at 25.9% over four near-uniform
# probabilities, and fed both into a composite cardiac_risk_score of 53.1.
CARDIOFUSION_TRAINED_TASKS = {"arrhythmia"}
_CARDIOFUSION_UNTRAINED_NOTE = (
    "Head not trained — CardioFusion was fitted on the ECG-signal modality only. "
    "No value is reported."
)


def guard_cardiofusion_result(result: dict) -> dict:
    """Blank every CardioFusion task output that came from an untrained head."""
    out = dict(result)
    suppressed = []
    for task in ("ejection_fraction", "murmur", "rhythm", "ventricular_danger"):
        if task in out and task not in CARDIOFUSION_TRAINED_TASKS:
            out[task] = {"available": False, "reason": _CARDIOFUSION_UNTRAINED_NOTE}
            suppressed.append(task)

    # The composite score and alert aggregate the untrained heads, so they inherit
    # their meaninglessness — withdraw them rather than showing a WARNING built
    # mostly from noise.
    if suppressed:
        for k in ("cardiac_risk_score", "alert_level"):
            out.pop(k, None)
        out["cardiac_risk_score_note"] = (
            "Withdrawn: the composite score aggregates untrained heads "
            f"({', '.join(suppressed)}). Only the arrhythmia head is trained."
        )
    out["trained_tasks"] = sorted(CARDIOFUSION_TRAINED_TASKS)
    out["untrained_tasks"] = suppressed
    return out


@app.post("/api/cardiofusion/demo")
async def cardiofusion_demo():
    """Run CardioFusion with synthetic inputs for all 4 modalities."""
    try:
        import time as _time
        start = _time.time()
        model = get_cardiofusion_model()

        inputs = {
            "echo_video": torch.randn(1, 1, 32, 112, 112),
            "ecg_image": torch.randn(1, 1, 128, 128),
            "heart_sound": torch.randn(1, 1, 64, 156),
            "ecg_signal": torch.randn(1, 2, 2500),
        }
        result = guard_cardiofusion_result(model.inference(inputs))
        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={
            "inference_time_seconds": elapsed,
            **result,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cardiofusion/demo/single/{modality}")
async def cardiofusion_demo_single(modality: str):
    """Run CardioFusion with synthetic input for a single modality."""
    try:
        import time as _time
        start = _time.time()
        model = get_cardiofusion_model()

        synth = {
            "echo_video": lambda: torch.randn(1, 1, 32, 112, 112),
            "ecg_image": lambda: torch.randn(1, 1, 128, 128),
            "heart_sound": lambda: torch.randn(1, 1, 64, 156),
            "ecg_signal": lambda: torch.randn(1, 2, 2500),
        }
        if modality not in synth:
            raise HTTPException(status_code=400, detail=f"Unknown modality: {modality}")

        inputs = {modality: synth[modality]()}
        result = model.inference(inputs)
        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={"inference_time_seconds": elapsed, **result})
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Cardiac Arrhythmia Video Endpoints (NEW — #6 dataset integration)
# ==============================================================================

class ArrhythmiaVideoResponse(BaseModel):
    prediction:        int
    label:             str
    confidence:        float
    probabilities:     dict
    guideline:         str
    inference_time_seconds: float
    mode:              str

@app.get(
    "/api/arrhythmia-video/eda",
    tags=["Arrhythmia Video"],
    summary="Cardiac arrhythmia video dataset EDA statistics",
)
async def arrhythmia_video_eda():
    """Return dataset-level statistics for the cardiac arrhythmia video dataset."""
    try:
        ds = get_arrhythmia_video_dataset()
        return JSONResponse(content=ds.get_eda_summary())
    except Exception as e:
        raise wrap_exception(e)


@app.post(
    "/api/arrhythmia-video/analyze/demo",
    response_model=ArrhythmiaVideoResponse,
    tags=["Arrhythmia Video"],
    summary="Arrhythmia classification on synthetic/demo video",
    description=(
        "Runs the 3D-CNN arrhythmia classifier on a synthetic video tensor. "
        "No real video needed. ⚠ Random weights — outputs are for demonstration only."
    ),
)
async def arrhythmia_video_demo():
    """Run arrhythmia video classification in demo mode."""
    try:
        start = time.time()
        ds         = get_arrhythmia_video_dataset()
        classifier = get_arrhythmia_video_classifier()

        if len(ds) > 0:
            tensor, sample = ds.load_video_tensor(0)
            tensor = tensor.unsqueeze(0).float()
            mode   = "demo_dataset"
        else:
            tensor = torch.randn(1, 1, 32, 112, 112)
            mode   = "synthetic"

        result = classifier.predict(tensor)
        elapsed = round(time.time() - start, 3)
        logger.info("Arrhythmia video demo inference", extra={"mode": mode, "latency_ms": elapsed*1000})

        return ArrhythmiaVideoResponse(
            **result,
            inference_time_seconds=elapsed,
            mode=mode,
        )
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.post(
    "/api/arrhythmia-video/analyze/upload",
    tags=["Arrhythmia Video"],
    summary="Classify an uploaded cardiac video for arrhythmia",
    description=(
        "Upload an echocardiogram .avi or .mp4 video. "
        "Returns binary arrhythmia prediction with confidence and clinical guideline. "
        "⚠ RESEARCH ONLY — not validated for clinical use."
    ),
    responses={
        415: {"description": "Unsupported file type"},
        422: {"description": "Empty video — no frames readable"},
    },
)
async def arrhythmia_video_upload(video: UploadFile = File(...)):
    """Classify an uploaded cardiac echocardiogram video for arrhythmia."""
    _validate_upload(video, settings.allowed_video_ext_list, "video")

    try:
        import cv2
    except ImportError:
        raise HTTPException(status_code=500, detail="opencv-python-headless not installed")

    suffix = os.path.splitext(video.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        start = time.time()
        ds = get_arrhythmia_video_dataset()
        tensor = ds.load_video_from_path(tmp_path)
    except RuntimeError as e:
        raise EmptyVideoError(str(e))
    finally:
        os.unlink(tmp_path)

    tensor = tensor.unsqueeze(0).float()
    classifier = get_arrhythmia_video_classifier()
    result  = classifier.predict(tensor)
    elapsed = round(time.time() - start, 3)

    audit_log.info("Arrhythmia video upload analysed", extra={
        "filename": video.filename, "prediction": result["label"], "latency_ms": elapsed*1000
    })

    return JSONResponse(content={**result, "inference_time_seconds": elapsed, "mode": "uploaded"})


@app.post(
    "/api/arrhythmia-video/analyze/dataset/{index}",
    tags=["Arrhythmia Video"],
    summary="Classify a video from the loaded arrhythmia dataset by index",
)
async def arrhythmia_video_from_dataset(index: int):
    """Classify a specific sample from the loaded cardiac arrhythmia video dataset."""
    try:
        ds = get_arrhythmia_video_dataset()
        if len(ds) == 0:
            raise DatasetNotFoundError(
                "Cardiac arrhythmia video dataset not found.",
                detail="Download EchoNet-Dynamic from Stanford AIMI and set ECHONET_DIR.",
            )
        if index < 0 or index >= len(ds):
            raise RecordNotFoundError(f"Index {index} out of range (0–{len(ds)-1})")

        start = time.time()
        tensor, sample = ds.load_video_tensor(index)
        tensor = tensor.unsqueeze(0).float()

        classifier = get_arrhythmia_video_classifier()
        result     = classifier.predict(tensor)
        elapsed    = round(time.time() - start, 3)

        return JSONResponse(content={
            **result,
            "inference_time_seconds": elapsed,
            "mode":         "dataset",
            "ground_truth": {
                "ef":              sample.get("ef"),
                "arrhythmia":      sample.get("arrhythmia"),
                "arrhythmia_label":sample.get("arrhythmia_label"),
            },
            "filename": sample.get("filename"),
        })
    except (DatasetNotFoundError, RecordNotFoundError):
        raise
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.get(
    "/api/arrhythmia-video/browse",
    tags=["Arrhythmia Video"],
    summary="Browse the cardiac arrhythmia video dataset",
)
async def arrhythmia_video_browse(page: int = 0, page_size: int = 20):
    """Paginated list of cardiac arrhythmia video dataset samples."""
    try:
        ds    = get_arrhythmia_video_dataset()
        total = len(ds)
        start = page * page_size
        end   = min(start + page_size, total)
        samples = [
            {
                "index":            i,
                "filename":         ds.samples[i].get("filename"),
                "ef":               ds.samples[i].get("ef"),
                "arrhythmia":       ds.samples[i].get("arrhythmia"),
                "arrhythmia_label": ds.samples[i].get("arrhythmia_label"),
                "split":            ds.samples[i].get("split"),
            }
            for i in range(start, end)
        ]
        return JSONResponse(content={
            "samples": samples, "total": total, "page": page,
            "page_size": page_size, "dataset_loaded": total > 0,
        })
    except Exception as e:
        raise wrap_exception(e)


# ==============================================================================
# LangChain RAG Endpoints
# ==============================================================================

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          description="Clinical question to query the knowledge base")
    category: Optional[str] = Field(None,
                          description="Optional filter: ecg | echo | heart_sound | arrhythmia_video | arthritis")

@app.get(
    "/api/langchain/info",
    tags=["LangChain RAG"],
    summary="LangChain RAG vector store info and status",
)
async def langchain_info():
    """Return information about the LangChain RAG vector store."""
    try:
        return JSONResponse(content=get_vector_store_stats())
    except Exception as e:
        raise wrap_exception(e)


@app.post(
    "/api/langchain/query",
    tags=["LangChain RAG"],
    summary="Query the clinical knowledge base via LangChain RAG",
    description=(
        "Submit a clinical question and get an LLM-augmented answer "
        "retrieved from the embedded clinical guidelines corpus. "
        "Uses FAISS vector store + GPT-2 (or keyword fallback if not installed)."
    ),
)
async def langchain_query(req: RAGQueryRequest):
    """Query the LangChain RAG pipeline with a clinical question."""
    try:
        with LoggedTimer(logger, "LangChain RAG query", extra={"question": req.question[:60]}):
            result = query_rag(req.question, category=req.category)
        return JSONResponse(content=result)
    except Exception as e:
        raise wrap_exception(e)


@app.get(
    "/api/langchain/guidelines",
    tags=["LangChain RAG"],
    summary="Retrieve top clinical guidelines for a query term",
)
async def langchain_guidelines(q: str, category: Optional[str] = None, top_k: int = 3):
    """Retrieve top-k clinical guideline excerpts matching the query."""
    try:
        guidelines = retrieve_guidelines(q, category=category, top_k=top_k)
        return JSONResponse(content={"guidelines": guidelines, "query": q, "top_k": top_k})
    except Exception as e:
        raise wrap_exception(e)


# ==============================================================================
# LangGraph Multi-Modal Diagnostic Agent Endpoints
# ==============================================================================

class AgentRequest(BaseModel):
    patient_id:   str   = Field("DEMO-001", description="Patient identifier")
    run_ecg:      bool  = Field(True,  description="Include ECG analysis node")
    run_echo:     bool  = Field(False, description="Include EchoNet analysis node")
    run_sound:    bool  = Field(False, description="Include heart sound analysis node")
    run_arrhythmia: bool = Field(False, description="Include arrhythmia video node")
    patient_data: Optional[dict] = Field(None, description="Arthritis blood test fields (22 features)")


@app.get(
    "/api/langgraph/info",
    tags=["LangGraph Agent"],
    summary="LangGraph diagnostic agent info and graph structure",
)
async def langgraph_info():
    """Return graph topology, node list, and availability status."""
    try:
        return JSONResponse(content=get_graph_info())
    except Exception as e:
        raise wrap_exception(e)


@app.post(
    "/api/langgraph/diagnose",
    tags=["LangGraph Agent"],
    summary="Run the full multi-modal LangGraph diagnostic agent",
    description=(
        "Triggers the LangGraph StateGraph. The supervisor node routes to "
        "the selected specialist nodes (ECG, Echo, Heart Sound, Arrhythmia, Arthritis), "
        "RAG enriches findings with clinical guidelines, and the aggregator "
        "synthesises a final report with risk stratification.\n\n"
        "All modalities run on **synthetic/demo data** when no real tensors are provided.\n\n"
        "⚠ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE."
    ),
    responses={200: {"description": "Diagnostic report with risk level and recommendations"}},
)
async def langgraph_diagnose(req: AgentRequest):
    """Run the LangGraph multi-modal diagnostic agent (demo mode)."""
    try:
        import torch
        inputs: dict = {"patient_id": req.patient_id}

        # Build synthetic tensors for each requested modality
        if req.run_ecg:
            inputs["ecg_tensor"] = torch.randn(1, 12, 1250)
        if req.run_echo:
            inputs["echo_tensor"] = torch.randn(1, 1, 32, 112, 112)
        if req.run_sound:
            inputs["mel_tensor"] = torch.randn(1, 1, 64, 156)
        if req.run_arrhythmia:
            inputs["arrh_tensor"] = torch.randn(1, 1, 32, 112, 112)
        if req.patient_data:
            inputs["patient_data"] = req.patient_data

        with LoggedTimer(logger, "LangGraph agent", extra={"patient_id": req.patient_id}):
            result = run_diagnostic_agent(inputs)

        # Serialise tensors and non-JSON types
        safe_result = {
            "patient_id":          result.get("patient_id"),
            "final_report":        result.get("final_report"),
            "risk_level":          result.get("risk_level"),
            "recommendations":     result.get("recommendations"),
            "active_modalities":   result.get("active_modalities"),
            "completed_nodes":     result.get("completed_nodes"),
            "retrieved_guidelines":result.get("retrieved_guidelines"),
            "rag_backend":         result.get("rag_backend"),
            "backend":             result.get("backend"),
            "inference_time_ms":   result.get("inference_time_ms"),
            "ecg_findings":        {k: v for k, v in result.get("ecg_findings", {}).items()
                                    if isinstance(v, (str, int, float, list, dict, bool, type(None)))},
            "echo_findings":       {k: v for k, v in result.get("echo_findings", {}).items()
                                    if isinstance(v, (str, int, float, list, dict, bool, type(None)))},
            "arrhythmia_findings": {k: v for k, v in result.get("arrhythmia_findings", {}).items()
                                    if isinstance(v, (str, int, float, list, dict, bool, type(None)))},
            "heart_sound_findings":{k: v for k, v in result.get("heart_sound_findings", {}).items()
                                    if isinstance(v, (str, int, float, list, dict, bool, type(None)))},
            "arthritis_findings":  {k: v for k, v in result.get("arthritis_findings", {}).items()
                                    if isinstance(v, (str, int, float, list, dict, bool, type(None)))},
        }

        audit_log.info("LangGraph diagnosis completed", extra={
            "patient_id":  req.patient_id,
            "risk_level":  result.get("risk_level"),
            "backend":     result.get("backend"),
            "latency_ms":  result.get("inference_time_ms"),
        })

        return JSONResponse(content=safe_result)

    except Exception as e:
        raise wrap_exception(e)


@app.post(
    "/api/langgraph/diagnose/full",
    tags=["LangGraph Agent"],
    summary="Run all modalities through the LangGraph agent",
    description="Convenience endpoint — runs all 5 modalities simultaneously in demo mode.",
)
async def langgraph_diagnose_full(patient_id: str = "DEMO-FULL"):
    """Run LangGraph agent with all modalities active (full demo)."""
    return await langgraph_diagnose(AgentRequest(
        patient_id=patient_id,
        run_ecg=True,
        run_echo=True,
        run_sound=True,
        run_arrhythmia=True,
        patient_data={
            "Age": 58.0, "Gender_M": 1.0, "Hb": 11.5, "ESRh": 45.0,
            "Uric_Acid": 7.2, "Creatinine": 1.3, "TC": 380.0,
        }
    ))


# ══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE ENDPOINTS (UCI Cleveland / Kaggle)
# ══════════════════════════════════════════════════════════════════════════════

class HeartDiseasePatientInput(BaseModel):
    age: Optional[float] = Field(None, ge=0, le=120, description="Age in years")
    sex: Optional[float] = Field(None, ge=0, le=1, description="1=male, 0=female")
    cp: Optional[float] = Field(None, ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: Optional[float] = Field(None, ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: Optional[float] = Field(None, ge=50, le=600, description="Serum cholesterol (mg/dL)")
    fbs: Optional[float] = Field(None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dL")
    restecg: Optional[float] = Field(None, ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: Optional[float] = Field(None, ge=50, le=250, description="Max heart rate achieved")
    exang: Optional[float] = Field(None, ge=0, le=1, description="Exercise-induced angina")
    oldpeak: Optional[float] = Field(None, ge=0, le=10, description="ST depression (exercise)")
    slope: Optional[float] = Field(None, ge=0, le=2, description="Slope of peak ST segment")
    ca: Optional[float] = Field(None, ge=0, le=4, description="Major vessels colored by fluoroscopy")
    thal: Optional[float] = Field(None, ge=0, le=3, description="Thalassemia type")


@app.get("/api/heart-disease/eda", tags=["Heart Disease"], summary="EDA summary for UCI Heart Disease dataset")
async def heart_disease_eda():
    try:
        loader = get_heart_disease_loader()
        return loader.get_eda_summary()
    except Exception as e:
        raise wrap_exception(e, DatasetNotFoundError)


@app.get("/api/heart-disease/correlation", tags=["Heart Disease"], summary="Feature correlation matrix")
async def heart_disease_correlation():
    try:
        loader = get_heart_disease_loader()
        return loader.get_correlation_matrix()
    except Exception as e:
        raise wrap_exception(e)


@app.get("/api/heart-disease/patients", tags=["Heart Disease"], summary="Browse patient records")
async def heart_disease_patients(page: int = 0, page_size: int = 20):
    try:
        loader = get_heart_disease_loader()
        return loader.get_patients(page, page_size)
    except Exception as e:
        raise wrap_exception(e)


@app.post("/api/heart-disease/train", tags=["Heart Disease"], summary="Train TabularBERT+MoE on heart disease data")
async def heart_disease_train():
    try:
        with LoggedTimer(logger, "heart_disease_train"):
            loader = get_heart_disease_loader()
            df = loader.load()
            predictor = get_heart_disease_predictor()
            metrics = predictor.train(df)
            audit_log.info("heart_disease_train", extra={"accuracy": metrics["accuracy"]})
            return metrics
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.post("/api/heart-disease/predict", tags=["Heart Disease"], summary="Predict heart disease risk")
async def heart_disease_predict(patient: HeartDiseasePatientInput):
    try:
        with LoggedTimer(logger, "heart_disease_predict"):
            predictor = get_heart_disease_predictor()
            patient_dict = patient.dict(exclude_none=True)
            result = predictor.predict(patient_dict)
            audit_log.info("heart_disease_predict", extra={"risk": result["risk_level"]})
            return result
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


# ══════════════════════════════════════════════════════════════════════════════
# ECG ARRHYTHMIA ENDPOINTS (MIT-BIH / Kaggle)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/ecg-arrhythmia/eda", tags=["ECG Arrhythmia"], summary="EDA summary for MIT-BIH ECG dataset")
async def ecg_arrhythmia_eda():
    try:
        loader = get_ecg_arrhythmia_loader()
        return loader.get_eda_summary()
    except Exception as e:
        raise wrap_exception(e, DatasetNotFoundError)


@app.get("/api/ecg-arrhythmia/samples", tags=["ECG Arrhythmia"], summary="Sample heartbeat waveforms per class")
async def ecg_arrhythmia_samples(n: int = 3):
    try:
        loader = get_ecg_arrhythmia_loader()
        return loader.get_sample_beats(n)
    except Exception as e:
        raise wrap_exception(e)


@app.post("/api/ecg-arrhythmia/train", tags=["ECG Arrhythmia"], summary="Train 1D-CNN on MIT-BIH heartbeats")
async def ecg_arrhythmia_train(epochs: int = 30, batch_size: int = 256):
    try:
        with LoggedTimer(logger, "ecg_arrhythmia_train"):
            loader = get_ecg_arrhythmia_loader()
            train_df, test_df = loader.load()
            predictor = get_ecg_arrhythmia_predictor()
            metrics = predictor.train(train_df, test_df, num_epochs=epochs, batch_size=batch_size)
            audit_log.info("ecg_arrhythmia_train", extra={"accuracy": metrics["accuracy"]})
            return metrics
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


class ECGBeatInput(BaseModel):
    signal: List[float] = Field(..., min_length=50, max_length=300, description="Heartbeat signal (187 time-steps)")


@app.post("/api/ecg-arrhythmia/predict", tags=["ECG Arrhythmia"], summary="Classify a single heartbeat")
async def ecg_arrhythmia_predict(beat: ECGBeatInput):
    try:
        with LoggedTimer(logger, "ecg_arrhythmia_predict"):
            predictor = get_ecg_arrhythmia_predictor()
            signal = np.array(beat.signal, dtype=np.float32)
            result = predictor.predict(signal)
            audit_log.info("ecg_arrhythmia_predict", extra={"class": result["class_name"]})
            return result
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.post("/api/ecg-arrhythmia/predict/demo", tags=["ECG Arrhythmia"], summary="Demo: classify random heartbeat from dataset")
async def ecg_arrhythmia_predict_demo():
    try:
        with LoggedTimer(logger, "ecg_arrhythmia_predict_demo"):
            loader = get_ecg_arrhythmia_loader()
            train_df, _ = loader.load()
            # Pick a random sample
            idx = np.random.randint(0, len(train_df))
            signal = train_df.iloc[idx, :MITBIH_SIGNAL_LEN].values.astype(np.float32)
            true_label = int(train_df.iloc[idx, -1])

            predictor = get_ecg_arrhythmia_predictor()
            result = predictor.predict(signal)
            result["true_class"] = true_label
            result["true_class_name"] = MITBIH_CLASSES.get(true_label, f"Class {true_label}")
            result["signal"] = signal.tolist()
            return result
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


# ──────────────────────────────────────────────────────────────────────────────
# Performance Metrics Endpoints (10 metrics across all pipelines)
# ──────────────────────────────────────────────────────────────────────────────

class MetricsRequest(BaseModel):
    """Payload for computing performance metrics."""
    y_true:             List[int]            = Field(..., description="Ground-truth class labels")
    y_pred:             List[int]            = Field(..., description="Predicted class labels")
    y_prob:             Optional[List[List[float]]] = Field(None, description="Class probabilities (N x K)")
    generated_reports:  Optional[List[str]]  = Field(None, description="Generated clinical report strings")
    reference_reports:  Optional[List[str]]  = Field(None, description="Reference clinical report strings")
    retrieved_contexts: Optional[List[str]]  = Field(None, description="Retrieved context strings (for hallucination rate)")
    pipeline:           str                  = Field("deepcardio", description="Pipeline: deepcardio | ecg_arrhythmia | heart_disease")


class NLGMetricsRequest(BaseModel):
    """Payload for NLG-only metrics (BLEU-4, ROUGE-L, Hallucination Rate)."""
    generated_reports:  List[str] = Field(..., description="Generated report strings")
    reference_reports:  List[str] = Field(..., description="Reference report strings")
    retrieved_contexts: Optional[List[str]] = Field(None, description="Retrieved context strings")


@app.get("/api/metrics/info", tags=["Metrics"], summary="List of all 10 performance metrics with descriptions")
async def metrics_info():
    return {
        "metrics": [
            {"id": 1,  "name": "Accuracy",           "type": "classification", "formula": "(TP+TN)/N",                          "target": ">= 0.95"},
            {"id": 2,  "name": "Precision",           "type": "classification", "formula": "TP/(TP+FP) macro-avg",               "target": ">= 0.90"},
            {"id": 3,  "name": "Recall/Sensitivity",  "type": "classification", "formula": "TP/(TP+FN) macro-avg",               "target": ">= 0.90"},
            {"id": 4,  "name": "F1-Score",             "type": "classification", "formula": "2·P·R/(P+R) macro-avg",             "target": ">= 0.90"},
            {"id": 5,  "name": "AUC-ROC",              "type": "classification", "formula": "OvR macro AUC, trapezoidal rule",   "target": ">= 0.90"},
            {"id": 6,  "name": "Specificity",          "type": "classification", "formula": "TN/(TN+FP) macro-avg",              "target": ">= 0.90"},
            {"id": 7,  "name": "NPV",                  "type": "classification", "formula": "TN/(TN+FN) macro-avg",              "target": ">= 0.85"},
            {"id": 8,  "name": "BLEU-4",               "type": "nlg",            "formula": "4-gram corpus BLEU (Papineni 2002)","target": ">= 0.25"},
            {"id": 9,  "name": "ROUGE-L",              "type": "nlg",            "formula": "LCS-based F1 (Lin 2004)",           "target": ">= 0.35"},
            {"id": 10, "name": "Hallucination Rate",   "type": "safety",         "formula": "Fraction of tokens absent from RAG context", "target": "< 0.05"},
        ],
        "pipeline_configs": {
            "deepcardio":     {"n_classes": 6, "classes": ["Normal","SVEB","VEB","Fusion","Unknown/Paced","ST-Elevation"]},
            "ecg_arrhythmia": {"n_classes": 5, "classes": ["Normal","SVEB","VEB","Fusion","Unknown/Paced"]},
            "heart_disease":  {"n_classes": 2, "classes": ["No CAD","CAD Present"]},
        }
    }


@app.post("/api/metrics/compute", tags=["Metrics"], summary="Compute all 10 performance metrics")
async def compute_metrics(req: MetricsRequest):
    try:
        with LoggedTimer(logger, "compute_metrics"):
            # Select pipeline-specific calculator
            pipeline_map = {
                "deepcardio":     MetricsCalculator.deepcardio,
                "ecg_arrhythmia": MetricsCalculator.ecg_arrhythmia,
                "heart_disease":  MetricsCalculator.heart_disease,
            }
            factory = pipeline_map.get(req.pipeline, MetricsCalculator.deepcardio)
            calc    = factory()

            if req.generated_reports and req.reference_reports:
                result = calc.full_report(
                    y_true=req.y_true,
                    y_pred=req.y_pred,
                    y_prob=req.y_prob,
                    generated_reports=req.generated_reports,
                    reference_reports=req.reference_reports,
                    retrieved_contexts=req.retrieved_contexts,
                )
            else:
                result = calc.classification_metrics(
                    y_true=req.y_true,
                    y_pred=req.y_pred,
                    y_prob=req.y_prob,
                )
            return result
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.post("/api/metrics/nlg", tags=["Metrics"], summary="Compute NLG metrics: BLEU-4, ROUGE-L, Hallucination Rate")
async def compute_nlg_metrics(req: NLGMetricsRequest):
    try:
        with LoggedTimer(logger, "compute_nlg_metrics"):
            calc = MetricsCalculator()
            return calc.nlg_metrics(
                generated_reports=req.generated_reports,
                reference_reports=req.reference_reports,
                retrieved_contexts=req.retrieved_contexts,
            )
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.get("/api/metrics/demo", tags=["Metrics"], summary="Demo: compute all 10 metrics on synthetic data")
async def metrics_demo():
    """Returns a demonstration of all 10 metrics computed on synthetic data."""
    try:
        np.random.seed(42)
        N = 200
        n_classes = 5
        y_true = np.random.randint(0, n_classes, N)
        y_pred = y_true.copy()
        # Introduce ~5% errors
        flip_idx = np.random.choice(N, int(N * 0.05), replace=False)
        y_pred[flip_idx] = np.random.randint(0, n_classes, len(flip_idx))
        y_prob = np.eye(n_classes)[y_pred] * 0.9 + 0.02  # near-perfect probs

        generated  = ["Normal sinus rhythm. No acute changes. Rate 72 bpm."] * N
        references = ["Normal sinus rhythm. Rate 72 bpm. No ST changes noted."] * N
        contexts   = ["Guideline: Normal sinus rhythm is defined by regular P waves and PR interval 120-200ms."] * N

        calc = MetricsCalculator.ecg_arrhythmia()
        result = calc.full_report(
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            y_prob=y_prob.tolist(),
            generated_reports=generated,
            reference_reports=references,
            retrieved_contexts=contexts,
        )
        result["note"] = "Demo with synthetic data (5% artificial error rate)"
        return result
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


# ──────────────────────────────────────────────────────────────────────────────
# Johns Hopkins–Style Clinical Report Endpoints
# ──────────────────────────────────────────────────────────────────────────────

class JHReportRequest(BaseModel):
    """Full structured payload for a Johns Hopkins–style clinical ECG PDF report."""
    # Patient info
    patient_name:           str   = "Anonymous Patient"
    mrn:                    str   = "N/A"
    dob:                    str   = "N/A"
    age:                    int   = 0
    sex:                    str   = "N/A"
    encounter_id:           str   = "N/A"
    ward:                   str   = "Cardiology"
    height_cm:              float = 0.0
    weight_kg:              float = 0.0
    # Physician info
    ordering_name:          str   = "Ordering Physician"
    interpreting_name:      str   = "Interpreting Cardiologist"
    department:             str   = "Division of Cardiology"
    clinical_indication:    str   = "Routine ECG evaluation"
    # Findings
    rhythm:                 str   = "Sinus Rhythm"
    heart_rate_bpm:         float = 72.0
    regularity:             str   = "Regular"
    pr_interval_ms:         float = 160.0
    qrs_duration_ms:        float = 88.0
    qt_interval_ms:         float = 380.0
    qtc_interval_ms:        float = 410.0
    st_segment:             str   = "Isoelectric — no significant deviation"
    t_waves:                str   = "Upright in all lateral leads"
    primary_impression:     str   = "Normal Sinus Rhythm. No acute ischaemic changes."
    risk_category:          str   = "LOW"
    risk_rationale:         str   = "No life-threatening findings detected."
    icd10_codes:            Optional[List[str]] = None
    differential:           Optional[List[str]] = None
    recommendations:        Optional[List[str]] = None
    stemi_equivalent:       bool  = False
    emergent_action:        bool  = False
    # AI metrics (optional)
    overall_confidence:     float = 0.0
    retrieved_contexts:     Optional[List[str]] = None
    bleu4:                  float = 0.0
    rouge_l:                float = 0.0
    hallucination_rate_val: float = 0.0
    inference_time_ms:      float = 0.0
    # Output
    audience:               str   = "clinical"   # "clinical" | "patient"


@app.get("/api/jh-report/demo", tags=["JH Report"], summary="Generate demo Johns Hopkins–style ECG report (clinical + patient PDFs)")
async def jh_report_demo():
    """Returns download URLs for pre-built demo reports."""
    try:
        with LoggedTimer(logger, "jh_report_demo"):
            import tempfile, os
            clinical_path = "/tmp/demo_jh_report_clinical.pdf"
            patient_path  = "/tmp/demo_jh_report_patient.pdf"

            from core.jh_report_generator import demo_jh_report
            demo_jh_report(output_path=clinical_path)

            # Also make a patient version
            return {
                "status": "success",
                "message": "Demo Johns Hopkins–style reports generated",
                "reports": {
                    "clinical": "/api/jh-report/download/clinical",
                    "patient":  "/api/jh-report/download/patient",
                },
                "note": (
                    "Reports follow Johns Hopkins Hospital ECG format: "
                    "8 structured sections, risk stratification, AI confidence scores, "
                    "evidence-based references, physician e-signature block."
                ),
            }
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


@app.get("/api/jh-report/download/{audience}", tags=["JH Report"], summary="Download demo JH report PDF")
async def jh_report_download(audience: str = "clinical"):
    """Serve the pre-built demo PDF."""
    from fastapi.responses import FileResponse
    path = f"/tmp/demo_jh_report_{audience}.pdf"
    if not os.path.exists(path):
        # Build on demand
        demo_jh_report(output_path="/tmp/demo_jh_report_clinical.pdf")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Report not found for audience: {audience}")
    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename=f"DeepCardio_JH_Report_{audience}.pdf",
    )


@app.post("/api/jh-report/generate", tags=["JH Report"], summary="Generate custom Johns Hopkins–style ECG PDF report")
async def jh_report_generate(req: JHReportRequest):
    """
    Generate a full Johns Hopkins–quality clinical or patient ECG PDF report.
    Returns the PDF as a binary response.
    """
    try:
        with LoggedTimer(logger, "jh_report_generate"):
            patient_data = {
                "name": req.patient_name, "mrn": req.mrn, "dob": req.dob,
                "age": req.age, "sex": req.sex, "encounter_id": req.encounter_id,
                "ward": req.ward, "height_cm": req.height_cm, "weight_kg": req.weight_kg,
                "ordering_name": req.ordering_name,
                "interpreting_name": req.interpreting_name,
                "clinical_indication": req.clinical_indication,
            }
            findings_data = {
                "rhythm": req.rhythm, "heart_rate_bpm": req.heart_rate_bpm,
                "regularity": req.regularity, "pr_interval_ms": req.pr_interval_ms,
                "qrs_duration_ms": req.qrs_duration_ms, "qt_interval_ms": req.qt_interval_ms,
                "qtc_interval_ms": req.qtc_interval_ms, "st_segment": req.st_segment,
                "t_waves": req.t_waves, "primary_impression": req.primary_impression,
                "risk_category": req.risk_category, "risk_rationale": req.risk_rationale,
                "stemi_equivalent": req.stemi_equivalent, "emergent_action": req.emergent_action,
            }
            if req.icd10_codes:
                findings_data["icd10_codes"] = req.icd10_codes
            if req.differential:
                findings_data["differential"] = req.differential
            if req.recommendations:
                findings_data["recommendations"] = req.recommendations

            ai_data = {
                "overall_confidence": req.overall_confidence,
                "bleu4_score": req.bleu4, "rouge_l_score": req.rouge_l,
                "hallucination_rate": req.hallucination_rate_val,
                "inference_time_ms": req.inference_time_ms,
            }
            if req.retrieved_contexts:
                ai_data["retrieved_contexts"] = req.retrieved_contexts

            pdf_bytes = generate_jh_ecg_report(
                patient_data=patient_data,
                findings_data=findings_data,
                ai_data=ai_data,
                audience=req.audience,
            )

            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="JH_ECG_Report_{req.audience}.pdf"',
                    "X-Patient-MRN": req.mrn,
                    "X-Risk-Category": req.risk_category,
                }
            )
    except Exception as e:
        raise wrap_exception(e, ModelInferenceError)


# ==============================================================================
# Cardiologist Validation & Benchmark Comparison Endpoints
# ==============================================================================
from core.cardio_validator import get_cardio_validator, CARDIOLOGIST_BENCHMARKS

@app.get(
    "/api/validation/benchmarks",
    tags=["Validation"],
    summary="Published cardiologist benchmark performance data",
)
async def validation_benchmarks():
    """Return all published cardiologist benchmarks used for model comparison."""
    try:
        validator = get_cardio_validator()
        return JSONResponse(content=validator.get_all_benchmarks())
    except Exception as e:
        raise wrap_exception(e)


@app.post(
    "/api/validation/compare",
    tags=["Validation"],
    summary="Compare model metrics against cardiologist benchmarks",
)
async def validation_compare(
    task: str,
    accuracy: Optional[float] = None,
    sensitivity: Optional[float] = None,
    specificity: Optional[float] = None,
    auc_roc: Optional[float] = None,
    f1_score: Optional[float] = None,
):
    """
    Compare provided model metrics against published cardiologist benchmarks.
    Task options: ecg_arrhythmia | heart_murmur | echo_ef | ventricular_arrhythmia | arthritis_risk
    """
    try:
        validator = get_cardio_validator()
        model_metrics = {}
        if accuracy    is not None: model_metrics["accuracy"]    = accuracy
        if sensitivity is not None: model_metrics["sensitivity"] = sensitivity
        if specificity is not None: model_metrics["specificity"] = specificity
        if auc_roc     is not None: model_metrics["auc_roc"]     = auc_roc
        if f1_score    is not None: model_metrics["f1_score"]    = f1_score
        result = validator.compare_model_to_experts(model_metrics, task)
        return JSONResponse(content=result)
    except Exception as e:
        raise wrap_exception(e)


@app.get(
    "/api/validation/unified-report",
    tags=["Validation"],
    summary="Full multi-modal clinical validation report vs cardiologist benchmarks",
)
async def validation_unified_report():
    """
    Generate a comprehensive validation report comparing all DeepCardio-RAG
    model components against published cardiologist performance benchmarks.
    """
    try:
        validator = get_cardio_validator()
        report = validator.generate_unified_validation_report()
        return JSONResponse(content=report)
    except Exception as e:
        raise wrap_exception(e)


# ==============================================================================
# CardioFusion Unified Report — All 4 Models Combined
# ==============================================================================
@app.post(
    "/api/cardiofusion/unified-report",
    tags=["CardioFusion"],
    summary="Run all 4 models in demo mode and generate unified multi-modal report",
    description=(
        "Runs ECG, EchoNet, HeartSound, and VFDB models simultaneously on demo data. "
        "Returns unified clinical report with individual model results, combined risk score, "
        "and both doctor and patient summary sections. "
        "⚠ Demo mode — all inputs are synthetic."
    ),
)
async def cardiofusion_unified_report():
    """Run all 4 cardiac AI models and produce a comprehensive unified report."""
    try:
        import time as _time
        start = _time.time()

        results = {}

        # 1. ECG Signal analysis (Transformer-MoE)
        try:
            dummy_ecg = torch.randn(1, 12, 5000)
            ecg_model = get_model()
            with torch.no_grad():
                ecg_out = ecg_model(dummy_ecg)
            ecg_probs = ecg_out["probs"][0].tolist()
            ecg_classes = ["Normal", "SVEB", "VEB", "Fusion", "Unknown/Paced", "ST-Elevation"]
            ecg_pred_idx = int(ecg_out["probs"][0].argmax().item())
            results["ecg_signal"] = {
                "model": "ECG Transformer-MoE (6 layers, 4 experts)",
                "prediction": ecg_classes[ecg_pred_idx],
                "confidence": round(ecg_probs[ecg_pred_idx], 4),
                "probabilities": {ecg_classes[i]: round(p, 4) for i, p in enumerate(ecg_probs)},
                "report_excerpt": ecg_out["reports"][0][:400] if ecg_out.get("reports") else "",
                "retrieved_guidelines": ecg_out.get("contexts", [[]])[0][:2],
            }
        except Exception as e:
            results["ecg_signal"] = {"error": str(e), "model": "ECG Transformer-MoE"}

        # 2. EchoNet video analysis
        try:
            dummy_video = torch.randn(1, 1, 32, 112, 112)
            echo_pipe = get_echonet_pipeline()
            echo_out = echo_pipe.analyze_single(dummy_video)
            results["echo_video"] = {
                "model": "EchoNet 3D-CNN (R2+1D)",
                "ef_predicted": echo_out["ef_predicted"],
                "ef_category": echo_out["ef_category"],
                "report_excerpt": echo_out["report"][:400],
                "retrieved_guidelines": echo_out["retrieved_guidelines"][:2],
            }
        except Exception as e:
            results["echo_video"] = {"error": str(e), "model": "EchoNet 3D-CNN"}

        # 3. Heart Sound / Murmur Detection
        try:
            dummy_mel = torch.randn(1, 1, 64, 156)
            hs_classifier = get_heart_sound_classifier()
            hs_results = hs_classifier.predict(dummy_mel)
            hs_r = hs_results[0]
            results["heart_sound"] = {
                "model": "HeartSound CNN (Mel-spectrogram, CirCor PhysioNet)",
                "murmur_detection": hs_r["murmur_class"],
                "confidence": hs_r["confidence"],
                "probabilities": hs_r["probabilities"],
                "guidelines": HEART_SOUND_GUIDELINES.get(hs_r["murmur_class"], [])[:2],
            }
        except Exception as e:
            results["heart_sound"] = {"error": str(e), "model": "HeartSound CNN"}

        # 4. Ventricular Arrhythmia Detection (VFDB)
        try:
            dummy_ecg_seg = torch.randn(1, 2, 2500)
            vfdb_det = get_vfdb_detector()
            vfdb_results = vfdb_det.predict(dummy_ecg_seg)
            vfdb_r = vfdb_results[0]
            results["ventricular_arrhythmia"] = {
                "model": "VFDB 1D-CNN (MIT-BIH VFDB, PhysioNet)",
                "rhythm_class": vfdb_r["rhythm_class"],
                "is_dangerous": vfdb_r.get("is_dangerous", False),
                "confidence": vfdb_r.get("rhythm_confidence", vfdb_r.get("confidence", 0.0)),
                "malignant_va_detected": vfdb_r.get("is_dangerous", False),
                "probabilities": vfdb_r.get("rhythm_probabilities", vfdb_r.get("probabilities", {})),
                "guidelines": VFDB_GUIDELINES.get(vfdb_r["rhythm_class"], [])[:2],
            }
        except Exception as e:
            results["ventricular_arrhythmia"] = {"error": str(e), "model": "VFDB 1D-CNN"}

        elapsed = round(_time.time() - start, 2)

        # Compute combined cardiac risk score
        risk_factors = []
        if "ecg_signal" in results and not results["ecg_signal"].get("error"):
            if results["ecg_signal"]["prediction"] not in ("Normal",):
                risk_factors.append(f"ECG: {results['ecg_signal']['prediction']}")
        if "echo_video" in results and not results["echo_video"].get("error"):
            if results["echo_video"]["ef_category"] == "HFrEF":
                risk_factors.append(f"Echo: Reduced EF ({results['echo_video']['ef_predicted']:.1f}%)")
            elif results["echo_video"]["ef_category"] == "HFmrEF":
                risk_factors.append(f"Echo: Mildly reduced EF ({results['echo_video']['ef_predicted']:.1f}%)")
        if "heart_sound" in results and not results["heart_sound"].get("error"):
            if results["heart_sound"]["murmur_detection"] == "Present":
                risk_factors.append("Murmur: Detected")
        if "ventricular_arrhythmia" in results and not results["ventricular_arrhythmia"].get("error"):
            if results["ventricular_arrhythmia"].get("is_dangerous"):
                risk_factors.append(f"DANGEROUS: Malignant VA — {results['ventricular_arrhythmia']['rhythm_class']}")

        overall_risk = "HIGH" if len(risk_factors) >= 2 else ("MODERATE" if len(risk_factors) == 1 else "LOW")
        risk_score = min(100, len(risk_factors) * 30 + 10)

        # Doctor report
        doctor_report = (
            "=== UNIFIED MULTI-MODAL CARDIAC ANALYSIS REPORT ===\n"
            f"Overall Cardiac Risk: {overall_risk} (Score: {risk_score}/100)\n"
            f"Inference Time: {elapsed}s\n\n"
            "MODULE FINDINGS:\n"
        )
        for key, r in results.items():
            if not r.get("error"):
                doctor_report += f"\n[{key.upper()}]\n"
                if key == "ecg_signal":
                    doctor_report += f"  Arrhythmia: {r['prediction']} (confidence {r['confidence']:.1%})\n"
                elif key == "echo_video":
                    doctor_report += f"  LVEF: {r['ef_predicted']:.1f}% — {r['ef_category']}\n"
                elif key == "heart_sound":
                    doctor_report += f"  Murmur: {r['murmur_detection']} (confidence {r['confidence']:.1%})\n"
                elif key == "ventricular_arrhythmia":
                    danger = "⚠ DANGEROUS" if r.get("is_dangerous") else "Low Risk"
                    doctor_report += f"  VA: {r['rhythm_class']} — {danger}\n"

        if risk_factors:
            doctor_report += f"\nRISK FACTORS IDENTIFIED:\n"
            for rf in risk_factors:
                doctor_report += f"  • {rf}\n"

        doctor_report += (
            "\nCLINICAL RECOMMENDATIONS:\n"
            + ("  URGENT: Immediate cardiology consult required.\n" if overall_risk == "HIGH" else
               "  MODERATE risk: Schedule cardiology follow-up within 2 weeks.\n" if overall_risk == "MODERATE" else
               "  LOW risk: Routine follow-up per clinical protocol.\n")
        )

        # Patient report
        patient_report = (
            "=== YOUR CARDIAC HEALTH SUMMARY ===\n"
            f"Overall Heart Health: {overall_risk} Risk\n\n"
        )
        patient_report += "What we found:\n"
        for key, r in results.items():
            if not r.get("error"):
                if key == "ecg_signal":
                    patient_report += f"  • Heart rhythm check: {r['prediction']}\n"
                elif key == "echo_video":
                    cat = r["ef_category"]
                    desc = {"HFrEF": "heart pumping weakly", "HFmrEF": "heart pumping borderline", "Normal": "heart pumping normally"}.get(cat, cat)
                    patient_report += f"  • Heart scan: {desc} ({r['ef_predicted']:.0f}% pumping strength)\n"
                elif key == "heart_sound":
                    patient_report += f"  • Heart sounds: {'Murmur heard — follow-up needed' if r['murmur_detection'] == 'Present' else 'No murmur detected'}\n"
                elif key == "ventricular_arrhythmia":
                    patient_report += f"  • Heart rhythm pattern: {'Abnormal rhythm detected' if r.get('is_dangerous') else 'Normal rhythm'}\n"

        if overall_risk == "HIGH":
            patient_report += "\nIMPORTANT: Please see your doctor soon.\n"
        elif overall_risk == "MODERATE":
            patient_report += "\nPlease schedule a follow-up appointment within the next 2 weeks.\n"
        else:
            patient_report += "\nYour heart health looks good. Keep up with regular check-ups.\n"

        return JSONResponse(content={
            "inference_time_seconds": elapsed,
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "risk_factors_identified": risk_factors,
            "models_run": list(results.keys()),
            "individual_results": results,
            "doctor_report": doctor_report,
            "patient_report": patient_report,
            "mode": "demo",
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ECG Image Pipeline Display Endpoint
# ==============================================================================
@app.post(
    "/api/ecg-images/classify/pipeline",
    tags=["ECG Images"],
    summary="ECG image classification with input→processing→output pipeline display",
)
async def ecg_images_classify_pipeline(image: Optional[UploadFile] = File(None)):
    """
    Classify ECG image and return full pipeline stages for visualization:
    Stage 1: Input image (base64 or metadata)
    Stage 2: Preprocessing steps (resize, normalize)
    Stage 3: Feature extraction (CNN layer outputs)
    Stage 4: Classification result with probabilities
    """
    try:
        from PIL import Image as PILImage
        import io, base64

        classifier = get_ecg_image_classifier()
        ds = get_ecg_image_dataset()

        if image and image.filename:
            # Use uploaded image
            contents = await image.read()
            pil_img = PILImage.open(io.BytesIO(contents)).convert("L")
            mode = "uploaded"
        elif len(ds) > 0:
            # Use random sample from dataset
            tensor_orig, cls_idx, cls_label = ds.get_random_sample()
            filepath, _ = ds.samples[ds.samples.index((ds.samples[0][0], cls_idx)) if len(ds.samples) > 0 else 0]
            pil_img = PILImage.open(filepath).convert("L")
            mode = "dataset"
        else:
            # Create synthetic ECG-like image
            import numpy as np
            arr = np.zeros((128, 128), dtype=np.uint8)
            # Draw synthetic ECG waveform
            for i in range(128):
                cycle = i % 25
                if cycle == 5: y = 30
                elif cycle == 6: y = 80
                elif cycle == 7: y = 20
                elif cycle == 10: y = 60
                else: y = 50
                if 0 <= y < 128:
                    arr[y, i] = 255
            pil_img = PILImage.fromarray(arr)
            mode = "synthetic"

        # Stage 1: Original image
        orig_resized = pil_img.resize((128, 128))
        buf = io.BytesIO()
        orig_resized.save(buf, format="PNG")
        input_b64 = base64.b64encode(buf.getvalue()).decode()

        # Stage 2: Preprocessing
        import numpy as np
        arr = np.array(orig_resized, dtype=np.float32) / 255.0
        # Contrast-enhanced version for display
        enhanced = np.clip(arr * 1.5, 0, 1)
        enh_img = PILImage.fromarray((enhanced * 255).astype(np.uint8))
        buf2 = io.BytesIO()
        enh_img.save(buf2, format="PNG")
        preprocessed_b64 = base64.b64encode(buf2.getvalue()).decode()

        # Stage 3: Tensor for model
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        # Stage 4: Classification
        results = classifier.predict(tensor)
        result = results[0]
        guidelines = ECG_IMAGE_GUIDELINES.get(result["class_label"], [])

        # Stage 5: Highlighted output image (mark predicted class region)
        output_arr = (arr * 255).astype(np.uint8)
        # Add confidence bar as a visual annotation
        conf_width = int(result["confidence"] * 128)
        output_arr[-8:, :conf_width] = 180  # bottom bar = confidence
        out_img = PILImage.fromarray(output_arr)
        buf3 = io.BytesIO()
        out_img.save(buf3, format="PNG")
        output_b64 = base64.b64encode(buf3.getvalue()).decode()

        return JSONResponse(content={
            "mode": mode,
            "pipeline_stages": {
                "stage1_input": {
                    "label": "Input ECG Image",
                    "description": f"Raw ECG waveform image ({pil_img.size[0]}×{pil_img.size[1]} px, resized to 128×128)",
                    "image_base64": input_b64,
                },
                "stage2_preprocessing": {
                    "label": "Preprocessing",
                    "description": "Grayscale conversion → Resize (128×128) → Contrast normalisation → [0,1] scaling",
                    "image_base64": preprocessed_b64,
                    "steps": ["Grayscale", "Resize 128×128", "Normalize /255", "Contrast enhance"],
                },
                "stage3_feature_extraction": {
                    "label": "CNN Feature Extraction",
                    "description": "4-block 2D-CNN: Conv(32)→Conv(64)→Conv(128)→Conv(256)→AdaptiveAvgPool→256-dim embedding",
                    "layers": ["Block1: 32 filters", "Block2: 64 filters", "Block3: 128 filters", "Block4: 256 filters", "Embedding: 256-dim"],
                },
                "stage4_classification": {
                    "label": "Arrhythmia Classification",
                    "description": "Linear(256→5) MoE head → Softmax → AAMI 5-class prediction",
                    "image_base64": output_b64,
                    **result,
                    "retrieved_guidelines": guidelines,
                },
            },
            "dataset_info": {
                "name": "ECG Images Dataset (Kaggle — MIT-BIH derived)",
                "url": "https://www.kaggle.com/datasets/analiviafr/ecg-images",
                "total_images": 109445,
                "classes": ["N (Normal)", "S (Supraventricular)", "V (Ventricular/PVC)", "F (Fusion)", "Q (Unclassifiable)"],
                "aami_standard": "AAMI EC57",
            },
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Heart Sound — Combined Murmur + Arrhythmia Detection Endpoint
# ==============================================================================
@app.post(
    "/api/heart-sound/analyze/combined",
    tags=["Heart Sound"],
    summary="Combined murmur detection + ventricular arrhythmia detection",
    description=(
        "Runs both the CirCor murmur classifier AND the VFDB arrhythmia detector. "
        "Returns 'Murmur Detection Result' and 'Malignant Ventricular Arrhythmia Detection Result'."
    ),
)
async def heart_sound_analyze_combined():
    """Combined analysis: murmur detection + malignant ventricular arrhythmia detection."""
    try:
        import time as _time
        start = _time.time()

        # Murmur Detection (CirCor)
        ds         = get_heart_sound_dataset()
        classifier = get_heart_sound_classifier()

        if len(ds) > 0:
            tensor, rec = ds.load_audio_tensor(0)
            signal = tensor.squeeze(0).numpy()
            mel = ds.compute_mel_spectrogram(signal)
            mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)
            mode = "dataset"
            patient_info = {"patient_id": rec.get("patient_id", "CirCor-Demo"), "location": rec.get("location", "MV")}
        else:
            mel_tensor = torch.randn(1, 1, 64, 156)
            mode = "synthetic"
            patient_info = {"patient_id": "CirCor-Demo", "location": "Mitral Valve (MV)"}

        murmur_results = classifier.predict(mel_tensor)
        murmur_r       = murmur_results[0]
        murmur_guidelines = HEART_SOUND_GUIDELINES.get(murmur_r["murmur_class"], [])

        # Ventricular Arrhythmia Detection (VFDB)
        dummy_ecg_seg = torch.randn(1, 2, 2500)
        vfdb_det = get_vfdb_detector()
        vfdb_results = vfdb_det.predict(dummy_ecg_seg)
        vfdb_r       = vfdb_results[0]
        va_guidelines = VFDB_GUIDELINES.get(vfdb_r["rhythm_class"], VFDB_GUIDELINES.get("Normal", []))

        elapsed = round(_time.time() - start, 2)

        return JSONResponse(content={
            "inference_time_seconds": elapsed,
            "mode": mode,
            "patient_info": patient_info,

            # === MURMUR DETECTION RESULT ===
            "murmur_detection": {
                "result": murmur_r["murmur_class"],
                "confidence": murmur_r["confidence"],
                "probabilities": murmur_r["probabilities"],
                "interpretation": (
                    "Cardiac murmur DETECTED — echocardiography recommended."
                    if murmur_r["murmur_class"] == "Present" else
                    "No cardiac murmur detected — normal auscultation."
                    if murmur_r["murmur_class"] == "Absent" else
                    "Murmur status inconclusive — repeat auscultation advised."
                ),
                "clinical_guidelines": murmur_guidelines,
                "dataset": "CirCor DigiScope PhysioNet",
                "dataset_url": "https://physionet.org/content/circor-heart-sound/1.0.1/",
                "auscultation_sites": ["Aortic Valve (AV)", "Pulmonary Valve (PV)", "Tricuspid Valve (TV)", "Mitral Valve (MV)"],
            },

            # === MALIGNANT VENTRICULAR ARRHYTHMIA DETECTION RESULT ===
            "ventricular_arrhythmia_detection": {
                "result": vfdb_r["rhythm_class"],
                "is_malignant": vfdb_r.get("is_dangerous", False),
                "confidence": vfdb_r.get("rhythm_confidence", vfdb_r.get("confidence", 0.0)),
                "probabilities": vfdb_r.get("rhythm_probabilities", vfdb_r.get("probabilities", {})),
                "alert_level": (
                    "CRITICAL — IMMEDIATE INTERVENTION REQUIRED"
                    if vfdb_r.get("is_dangerous") else "Normal"
                ),
                "malignant_types_detected": [
                    vfdb_r["rhythm_class"]
                ] if vfdb_r.get("is_dangerous") else [],
                "interpretation": (
                    f"MALIGNANT VENTRICULAR ARRHYTHMIA DETECTED: {vfdb_r['rhythm_class']}. "
                    f"Immediate defibrillation/cardioversion may be required."
                    if vfdb_r.get("is_dangerous") else
                    f"No malignant arrhythmia detected. Rhythm: {vfdb_r['rhythm_class']}."
                ),
                "clinical_guidelines": va_guidelines,
                "dataset": "MIT-BIH Malignant Ventricular Ectopy Database (VFDB)",
                "dataset_url": "https://physionet.org/content/vfdb/1.0.0/",
                "dangerous_rhythms": list(ALL_RHYTHMS),
            },

            # Combined clinical summary
            "combined_clinical_summary": {
                "overall_alert": (
                    "CRITICAL" if vfdb_r.get("is_dangerous") else
                    "WARNING" if murmur_r["murmur_class"] == "Present" else
                    "NORMAL"
                ),
                "findings": [
                    f"Murmur: {murmur_r['murmur_class']}",
                    f"Rhythm: {vfdb_r['rhythm_class']} ({'DANGEROUS' if vfdb_r.get('is_dangerous') else 'Safe'})",
                ],
            },
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

