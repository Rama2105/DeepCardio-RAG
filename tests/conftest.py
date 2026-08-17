"""
tests/conftest.py — Shared fixtures for pytest
================================================
Run all tests with:
    pytest tests/ -v --cov=core --cov=main --cov-report=term-missing

Install test dependencies:
    pip install pytest pytest-cov httpx
"""

import numpy as np
import torch
import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic fixture data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ecg_tensor():
    """12-lead ECG tensor (batch=1, leads=12, length=1250)."""
    return torch.randn(1, 12, 1250)

@pytest.fixture
def synthetic_echo_tensor():
    """Echocardiogram video tensor (batch=1, channels=1, frames=32, H=112, W=112)."""
    return torch.randn(1, 1, 32, 112, 112)

@pytest.fixture
def synthetic_ecg_image_tensor():
    """ECG waveform image tensor (batch=1, channels=1, H=128, W=128)."""
    return torch.randn(1, 1, 128, 128)

@pytest.fixture
def synthetic_mel_tensor():
    """Heart sound mel spectrogram (batch=1, channels=1, mels=64, frames=156)."""
    return torch.randn(1, 1, 64, 156)

@pytest.fixture
def synthetic_vfdb_tensor():
    """2-lead ECG signal (batch=1, leads=2, samples=2500)."""
    return torch.randn(1, 2, 2500)

@pytest.fixture
def sample_patient_data():
    """Minimal valid arthritis patient feature dict."""
    return {
        "Age": 45.0,
        "Gender_M": 1.0,
        "TC": 200.0,
        "Hb": 13.5,
        "ESRh": 25.0,
        "ESRo": 30.0,
        "RBC": 4.5,
        "PCV": 40.0,
        "MCV": 85.0,
        "MCH": 28.0,
        "MCHC": 33.0,
        "Urea": 30.0,
        "Creatinine": 1.0,
        "Uric_Acid": 5.5,
        "Calcium": 9.0,
        "RBS": 100.0,
        "P": 2.5,
        "L": 1.8,
        "E": 0.2,
        "Abs": 0.1,
        "PC": 250.0,
        "ASO": 200.0,
    }

@pytest.fixture
def synthetic_arrhythmia_sample():
    """A single arrhythmia dataset sample dict."""
    return {
        "filename": "DEMO_0001",
        "video_path": None,
        "ef": 28.0,
        "esv": 90.0,
        "edv": 140.0,
        "fps": 30.0,
        "num_raw_frames": 60,
        "split": "TEST",
        "arrhythmia": 1,
        "arrhythmia_label": "Arrhythmia",
        "_demo": True,
    }

@pytest.fixture
def app_client():
    """FastAPI TestClient (no auth by default)."""
    from main import app
    return TestClient(app, raise_server_exceptions=False)

@pytest.fixture
def auth_headers(app_client):
    """Return auth headers for the demo 'doctor' user."""
    resp = app_client.post(
        "/api/auth/token",
        data={"username": "doctor", "password": "doctor123"},
    )
    if resp.status_code == 200:
        token = resp.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}
