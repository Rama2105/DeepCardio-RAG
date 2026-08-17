"""
run_tests.py — Standalone test runner (no pytest required)
===========================================================
Run on Windows in your venv:
    python run_tests.py

Or with pytest if available:
    python -m pytest tests/ -v --tb=short

Or install pytest first:
    pip install pytest pytest-cov httpx
    python -m pytest tests/ -v --cov=core --cov-report=term-missing
"""

import sys
import time
import traceback
import importlib
import os

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

results = []

def run_test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except ImportError as e:
        results.append((SKIP, name))
        print(f"  {SKIP}  {name}  [{e}]")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DeepCardio-RAG — Standalone Test Suite")
print("="*60)

# ─── config ──────────────────────────────────────────────────────────────────
print("\n[config]")

def test_config_importable():
    from config import settings
    assert settings.app_name == "DeepCardio-RAG"
    assert settings.embedding_dim > 0
    assert settings.num_frames > 0
    assert isinstance(settings.cors_origins, list)

run_test("config importable and valid", test_config_importable)

# ─── exceptions ──────────────────────────────────────────────────────────────
print("\n[core.exceptions]")

def test_exceptions():
    from core.exceptions import (
        DataValidationError, ModelInferenceError,
        DatabaseConnectionError, UnsupportedFileTypeError,
        FileTooLargeError, wrap_exception
    )
    e1 = DataValidationError("bad age")
    assert e1.status_code == 400
    assert e1.detail["error_code"] == "VALIDATION_ERROR"

    e2 = ModelInferenceError("OOM")
    assert e2.status_code == 500

    e3 = DatabaseConnectionError("unreachable")
    assert e3.status_code == 503

    e4 = UnsupportedFileTypeError(".exe not allowed")
    assert e4.status_code == 415

    e5 = FileTooLargeError("too big")
    assert e5.status_code == 413

    # wrap_exception passthrough
    assert wrap_exception(e1) is e1

    # wrap_exception converts generic
    wrapped = wrap_exception(RuntimeError("oops"), default_cls=ModelInferenceError)
    assert isinstance(wrapped, ModelInferenceError)

run_test("exception hierarchy + status codes", test_exceptions)

# ─── logger ──────────────────────────────────────────────────────────────────
print("\n[core.logger]")

def test_logger():
    import logging
    from core.logger import get_logger, get_audit_logger, redact_pii, LoggedTimer

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)

    audit = get_audit_logger()
    assert isinstance(audit, logging.Logger)

    data = {"name": "Alice", "age": 45, "email": "a@b.com"}
    redacted = redact_pii(data)
    assert redacted["name"] == "[REDACTED]"
    assert redacted["email"] == "[REDACTED]"
    assert redacted["age"] == 45

    with LoggedTimer(logger, "test op"):
        time.sleep(0.001)

run_test("structured logger + PII redaction + timer", test_logger)

# ─── model cards ─────────────────────────────────────────────────────────────
print("\n[core.model_cards]")

def test_model_cards():
    from core.model_cards import get_all_cards, get_card, MODEL_CARDS
    cards = get_all_cards()
    assert len(cards) >= 7

    card = get_card("arrhythmia_video_cnn")
    assert "model_name" in card
    assert "intended_use" in card
    assert "limitations" in card
    assert "disclaimer" in card
    assert "NOT FOR CLINICAL USE" in card["disclaimer"]

    try:
        get_card("nonexistent_model")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

run_test("model cards completeness + disclaimer", test_model_cards)

# ─── arrhythmia video dataset ─────────────────────────────────────────────────
print("\n[core.cardiac_arrhythmia_video_loader]")

def test_arrhythmia_dataset():
    from core.cardiac_arrhythmia_video_loader import (
        CardiacArrhythmiaVideoDataset, EF_ARRHYTHMIA_LOW, EF_ARRHYTHMIA_HIGH
    )
    ds = CardiacArrhythmiaVideoDataset()
    assert len(ds) > 0
    assert all("arrhythmia" in s for s in ds.samples)
    assert all(s["arrhythmia"] in [0, 1] for s in ds.samples)

    # EF proxy labels correct
    for s in ds.samples:
        if s["ef"] is not None:
            if s["ef"] < EF_ARRHYTHMIA_LOW or s["ef"] > EF_ARRHYTHMIA_HIGH:
                assert s["arrhythmia"] == 1
            else:
                assert s["arrhythmia"] == 0

    # EDA counts match
    eda = ds.get_eda_summary()
    assert eda["total_videos"] == len(ds.samples)
    assert eda["arrhythmia_count"] + eda["normal_count"] == len(ds.samples)

run_test("arrhythmia dataset demo mode + label correctness + EDA", test_arrhythmia_dataset)


def test_arrhythmia_tensor():
    import torch
    from core.cardiac_arrhythmia_video_loader import CardiacArrhythmiaVideoDataset
    ds = CardiacArrhythmiaVideoDataset()
    tensor, labels = ds.load_video_tensor(0)
    assert tensor.dim() == 4          # (1, T, H, W)
    assert tensor.shape[0] == 1
    assert "arrhythmia" in labels

run_test("arrhythmia video tensor shape", test_arrhythmia_tensor)


def test_arrhythmia_cnn():
    import torch
    from core.cardiac_arrhythmia_video_loader import ArrhythmiaVideoCNN
    model = ArrhythmiaVideoCNN()
    model.eval()
    x = torch.randn(1, 1, 32, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)

    result = model.predict(x)
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4

run_test("ArrhythmiaVideoCNN output shape + predict()", test_arrhythmia_cnn)

# ─── ECG image classifier ─────────────────────────────────────────────────────
print("\n[core.ecg_image_loader]")

def test_ecg_image():
    import torch
    from core.ecg_image_loader import get_ecg_image_classifier, CLASS_LABELS
    clf = get_ecg_image_classifier()
    x   = torch.randn(1, 1, 128, 128)
    res = clf.predict(x)
    assert len(res) == 1
    assert res[0]["class_label"] in CLASS_LABELS
    assert 0.0 <= res[0]["confidence"] <= 1.0

run_test("ECG image classifier predict()", test_ecg_image)

# ─── Heart sound classifier ───────────────────────────────────────────────────
print("\n[core.heart_sound_loader]")

def test_heart_sound():
    import torch
    from core.heart_sound_loader import get_heart_sound_classifier, MURMUR_CLASSES
    clf = get_heart_sound_classifier()
    x   = torch.randn(1, 1, 64, 156)
    res = clf.predict(x)
    assert len(res) == 1
    assert res[0]["murmur_class"] in MURMUR_CLASSES

run_test("Heart sound classifier predict()", test_heart_sound)

# ─── VFDB detector ────────────────────────────────────────────────────────────
print("\n[core.vfdb_loader]")

def test_vfdb():
    import torch
    from core.vfdb_loader import get_vfdb_detector, ALL_RHYTHMS
    det = get_vfdb_detector()
    x   = torch.randn(1, 2, 2500)
    res = det.predict(x)
    assert len(res) == 1
    assert res[0]["rhythm_class"] in ALL_RHYTHMS

run_test("VFDB detector predict()", test_vfdb)

# ─── CardioFusion ─────────────────────────────────────────────────────────────
print("\n[core.hybrid_model]")

def test_cardiofusion():
    import torch
    from core.hybrid_model import get_cardiofusion_model
    model = get_cardiofusion_model()
    result = model.inference({
        "echo_video":  torch.randn(1, 1, 32, 112, 112),
        "ecg_image":   torch.randn(1, 1, 128, 128),
        "heart_sound": torch.randn(1, 1, 64, 156),
    })
    assert isinstance(result, dict)
    assert "modalities_used" in result

run_test("CardioFusion multi-modal inference", test_cardiofusion)

# ─── FastAPI endpoints (requires httpx) ──────────────────────────────────────
print("\n[FastAPI endpoints]")

def test_api_endpoints():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        raise ImportError("httpx not installed — run: pip install httpx")

    from main import app
    client = TestClient(app, raise_server_exceptions=False)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Model cards
    r = client.get("/api/model-cards")
    assert r.status_code == 200

    r = client.get("/api/model-cards/arrhythmia_video_cnn")
    assert r.status_code == 200

    # ECG analyze
    r = client.post("/api/analyze")
    assert r.status_code == 200
    assert "report" in r.json()

    # Arrhythmia video EDA
    r = client.get("/api/arrhythmia-video/eda")
    assert r.status_code == 200
    data = r.json()
    assert "total_videos" in data

    # Arrhythmia video demo
    r = client.post("/api/arrhythmia-video/analyze/demo")
    assert r.status_code == 200
    assert "prediction" in r.json()

    # Arrhythmia video browse
    r = client.get("/api/arrhythmia-video/browse")
    assert r.status_code == 200

    # Echonet demo
    r = client.post("/api/echonet/analyze/demo")
    assert r.status_code == 200

    # Heart sound demo
    r = client.post("/api/heart-sound/analyze/demo")
    assert r.status_code == 200

    # VFDB demo
    r = client.post("/api/vfdb/analyze/demo")
    assert r.status_code == 200

    # CardioFusion demo
    r = client.post("/api/cardiofusion/demo")
    assert r.status_code == 200

    # Auth login
    r = client.post("/api/auth/token",
                    data={"username": "doctor", "password": "doctor123"})
    assert r.status_code == 200
    assert "access_token" in r.json()

    # Wrong password
    r = client.post("/api/auth/token",
                    data={"username": "doctor", "password": "wrongpass"})
    assert r.status_code == 401

run_test("FastAPI endpoints (health, ECG, arrhythmia-video, auth, etc.)", test_api_endpoints)

# ─── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
n_skip = sum(1 for r in results if r[0] == SKIP)

print("\n" + "="*60)
print(f"  Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
print("="*60 + "\n")

if n_fail > 0:
    sys.exit(1)
