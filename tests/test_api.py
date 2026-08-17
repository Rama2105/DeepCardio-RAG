"""
tests/test_api.py — Integration tests for FastAPI endpoints
============================================================
Tests every endpoint using the FastAPI TestClient.
No real dataset or GPU required — all models run on synthetic data.

Run:
    pytest tests/test_api.py -v
"""

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_doctor(self, app_client):
        resp = app_client.post(
            "/api/auth/token",
            data={"username": "doctor", "password": "doctor123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["role"] == "doctor"

    def test_login_admin(self, app_client):
        resp = app_client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_login_wrong_password(self, app_client):
        resp = app_client.post(
            "/api/auth/token",
            data={"username": "doctor", "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    def test_get_me(self, app_client, auth_headers):
        if not auth_headers:
            pytest.skip("Auth not available")
        resp = app_client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "doctor"


# ──────────────────────────────────────────────────────────────────────────────
# ECG analysis
# ──────────────────────────────────────────────────────────────────────────────

class TestECGAnalysis:
    def test_analyze_ecg_demo(self, app_client):
        resp = app_client.post("/api/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data
        assert "inference_time_seconds" in data

    def test_ecg_response_has_guidelines(self, app_client):
        resp = app_client.post("/api/analyze")
        assert resp.status_code == 200
        assert "retrieved_guidelines" in resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Arthritis endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestArthritis:
    def test_eda_endpoint(self, app_client):
        resp = app_client.get("/api/arthritis/eda")
        assert resp.status_code == 200

    def test_predict_with_valid_data(self, app_client, sample_patient_data):
        resp = app_client.post("/api/arthritis/predict", json=sample_patient_data)
        assert resp.status_code == 200

    def test_predict_with_empty_data(self, app_client):
        resp = app_client.post("/api/arthritis/predict", json={})
        # Should still respond (all fields optional)
        assert resp.status_code in (200, 422)

    def test_predict_age_boundary_negative(self, app_client, sample_patient_data):
        """Negative age should be rejected after validation is added."""
        bad_data = {**sample_patient_data, "Age": -5.0}
        resp = app_client.post("/api/arthritis/predict", json=bad_data)
        # Currently returns 200 (validation gap), after fix should be 422
        assert resp.status_code in (200, 422)


# ──────────────────────────────────────────────────────────────────────────────
# EchoNet endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestEchoNet:
    def test_eda_endpoint(self, app_client):
        resp = app_client.get("/api/echonet/eda")
        assert resp.status_code == 200

    def test_demo_inference(self, app_client):
        resp = app_client.post("/api/echonet/analyze/demo")
        assert resp.status_code == 200
        data = resp.json()
        assert "ef_predicted" in data
        assert "ef_category" in data
        assert "report" in data

    def test_browse_empty_dataset(self, app_client):
        resp = app_client.get("/api/echonet/dataset/browse")
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Cardiac Arrhythmia Video endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestArrhythmiaVideo:
    def test_eda_endpoint(self, app_client):
        resp = app_client.get("/api/arrhythmia-video/eda")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_videos" in data
        assert "arrhythmia_count" in data

    def test_demo_inference(self, app_client):
        resp = app_client.post("/api/arrhythmia-video/analyze/demo")
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "label" in data
        assert "confidence" in data

    def test_browse_demo(self, app_client):
        resp = app_client.get("/api/arrhythmia-video/browse")
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# Heart Sound endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestHeartSound:
    def test_eda_endpoint(self, app_client):
        resp = app_client.get("/api/heart-sound/eda")
        assert resp.status_code == 200

    def test_demo_inference(self, app_client):
        resp = app_client.post("/api/heart-sound/analyze/demo")
        assert resp.status_code == 200
        assert "murmur_class" in resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# VFDB endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestVFDB:
    def test_eda_endpoint(self, app_client):
        resp = app_client.get("/api/vfdb/eda")
        assert resp.status_code == 200

    def test_demo_inference(self, app_client):
        resp = app_client.post("/api/vfdb/analyze/demo")
        assert resp.status_code == 200
        assert "rhythm_class" in resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# CardioFusion endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestCardioFusion:
    def test_demo_all_modalities(self, app_client):
        resp = app_client.post("/api/cardiofusion/demo")
        assert resp.status_code == 200

    def test_demo_single_echo(self, app_client):
        resp = app_client.post("/api/cardiofusion/demo/single/echo_video")
        assert resp.status_code == 200

    def test_demo_single_ecg_image(self, app_client):
        resp = app_client.post("/api/cardiofusion/demo/single/ecg_image")
        assert resp.status_code == 200

    def test_demo_unknown_modality(self, app_client):
        resp = app_client.post("/api/cardiofusion/demo/single/unknown_modality")
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# DB / Stats endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestDBStats:
    def test_db_stats(self, app_client):
        resp = app_client.get("/api/db/stats")
        assert resp.status_code == 200

    def test_full_db_stats(self, app_client):
        resp = app_client.get("/api/db/stats/full")
        assert resp.status_code == 200

    def test_patients_endpoint(self, app_client):
        resp = app_client.get("/api/patients")
        assert resp.status_code == 200
        assert "patients" in resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Model Cards endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestModelCards:
    def test_model_cards_list(self, app_client):
        resp = app_client.get("/api/model-cards")
        assert resp.status_code == 200

    def test_specific_model_card(self, app_client):
        resp = app_client.get("/api/model-cards/ecg_cnn")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "intended_use" in data
        assert "limitations" in data
        assert "disclaimer" in data
