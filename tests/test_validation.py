"""
tests/test_validation.py — Input validation and exception tests
===============================================================
Tests Pydantic constraints, file upload validation, and exception hierarchy.
"""

import pytest
import io
from core.exceptions import (
    DataValidationError, UnsupportedFileTypeError, FileTooLargeError,
    ModelInferenceError, DatabaseConnectionError, wrap_exception,
)


# ──────────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ──────────────────────────────────────────────────────────────────────────────

class TestExceptions:
    def test_data_validation_error_status(self):
        exc = DataValidationError("Age cannot be negative")
        assert exc.status_code == 400
        assert exc.detail["error_code"] == "VALIDATION_ERROR"

    def test_model_inference_error_status(self):
        exc = ModelInferenceError("OOM on GPU")
        assert exc.status_code == 500
        assert exc.detail["error_code"] == "MODEL_INFERENCE_ERROR"

    def test_database_connection_error_status(self):
        exc = DatabaseConnectionError("Milvus unreachable")
        assert exc.status_code == 503

    def test_unsupported_file_type_status(self):
        exc = UnsupportedFileTypeError("Only .wav files accepted")
        assert exc.status_code == 415

    def test_file_too_large_status(self):
        exc = FileTooLargeError("File exceeds 100MB limit")
        assert exc.status_code == 413

    def test_wrap_exception_passthrough(self):
        original = ModelInferenceError("original")
        wrapped  = wrap_exception(original)
        assert wrapped is original

    def test_wrap_exception_converts(self):
        generic = RuntimeError("something broke")
        wrapped = wrap_exception(generic, default_cls=ModelInferenceError)
        assert isinstance(wrapped, ModelInferenceError)
        assert "something broke" in wrapped.detail.get("detail", "")

    def test_exception_with_context(self):
        exc = DataValidationError(
            "Invalid age", detail="Must be 0–150", context={"field": "Age", "value": -1}
        )
        assert exc.detail["context"]["field"] == "Age"


# ──────────────────────────────────────────────────────────────────────────────
# Config settings
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_importable(self):
        from config import settings
        assert settings.app_name == "DeepCardio-RAG"

    def test_embedding_dim_positive(self):
        from config import settings
        assert settings.embedding_dim > 0

    def test_num_frames_positive(self):
        from config import settings
        assert settings.num_frames > 0

    def test_max_upload_bytes(self):
        from config import settings
        assert settings.max_upload_bytes > 0

    def test_cors_origins_is_list(self):
        from config import settings
        assert isinstance(settings.cors_origins, list)


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

class TestLogger:
    def test_get_logger_returns_logger(self):
        from core.logger import get_logger
        import logging
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_audit_logger(self):
        from core.logger import get_audit_logger
        import logging
        audit = get_audit_logger()
        assert isinstance(audit, logging.Logger)

    def test_redact_pii(self):
        from core.logger import redact_pii
        data = {"name": "John Doe", "age": 45, "email": "john@example.com"}
        redacted = redact_pii(data)
        assert redacted["name"] == "[REDACTED]"
        assert redacted["email"] == "[REDACTED]"
        assert redacted["age"] == 45   # non-PII field unchanged

    def test_logged_timer(self):
        from core.logger import LoggedTimer, get_logger
        import time
        logger = get_logger("test.timer")
        with LoggedTimer(logger, "test operation"):
            time.sleep(0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Arrhythmia dataset label correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestArrhythmiaLabels:
    def test_low_ef_gets_arrhythmia_label(self):
        from core.cardiac_arrhythmia_video_loader import (
            CardiacArrhythmiaVideoDataset, EF_ARRHYTHMIA_LOW
        )
        ds = CardiacArrhythmiaVideoDataset()
        low_ef_samples = [s for s in ds.samples if s["ef"] is not None and s["ef"] < EF_ARRHYTHMIA_LOW]
        for s in low_ef_samples:
            assert s["arrhythmia"] == 1, f"EF={s['ef']} should be arrhythmia"

    def test_normal_ef_gets_normal_label(self):
        from core.cardiac_arrhythmia_video_loader import (
            CardiacArrhythmiaVideoDataset, EF_ARRHYTHMIA_LOW, EF_ARRHYTHMIA_HIGH
        )
        ds = CardiacArrhythmiaVideoDataset()
        normal_samples = [
            s for s in ds.samples
            if s["ef"] is not None
            and EF_ARRHYTHMIA_LOW <= s["ef"] <= EF_ARRHYTHMIA_HIGH
        ]
        for s in normal_samples:
            assert s["arrhythmia"] == 0, f"EF={s['ef']} should be normal"

    def test_eda_counts_match(self):
        from core.cardiac_arrhythmia_video_loader import CardiacArrhythmiaVideoDataset
        ds = CardiacArrhythmiaVideoDataset()
        eda = ds.get_eda_summary()
        actual_arrh  = sum(s["arrhythmia"] for s in ds.samples)
        actual_normal = len(ds.samples) - actual_arrh
        assert eda["arrhythmia_count"] == actual_arrh
        assert eda["normal_count"]     == actual_normal
        assert eda["total_videos"]     == len(ds.samples)
