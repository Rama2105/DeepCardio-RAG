"""
tests/test_models.py — Unit tests for all ML model encoders
============================================================
Tests run entirely with synthetic tensors — no real datasets needed.
"""

import pytest
import torch
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Arrhythmia Video CNN
# ──────────────────────────────────────────────────────────────────────────────

class TestArrhythmiaVideoCNN:
    def test_output_shape(self, synthetic_echo_tensor):
        from core.cardiac_arrhythmia_video_loader import ArrhythmiaVideoCNN
        model = ArrhythmiaVideoCNN()
        model.eval()
        with torch.no_grad():
            out = model(synthetic_echo_tensor)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"

    def test_predict_returns_dict(self, synthetic_echo_tensor):
        from core.cardiac_arrhythmia_video_loader import ArrhythmiaVideoCNN
        model = ArrhythmiaVideoCNN()
        result = model.predict(synthetic_echo_tensor)
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_prediction_is_binary(self, synthetic_echo_tensor):
        from core.cardiac_arrhythmia_video_loader import ArrhythmiaVideoCNN
        model = ArrhythmiaVideoCNN()
        result = model.predict(synthetic_echo_tensor)
        assert result["prediction"] in [0, 1]

    def test_probabilities_sum_to_one(self, synthetic_echo_tensor):
        from core.cardiac_arrhythmia_video_loader import ArrhythmiaVideoCNN
        model = ArrhythmiaVideoCNN()
        result = model.predict(synthetic_echo_tensor)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected 1.0"


# ──────────────────────────────────────────────────────────────────────────────
# 2. ECG Image Classifier
# ──────────────────────────────────────────────────────────────────────────────

class TestECGImageClassifier:
    def test_output_shape(self, synthetic_ecg_image_tensor):
        from core.ecg_image_loader import get_ecg_image_classifier, AAMI_CLASSES
        clf = get_ecg_image_classifier()
        results = clf.predict(synthetic_ecg_image_tensor)
        assert len(results) == 1
        assert "class_label" in results[0]
        assert "confidence" in results[0]

    def test_confidence_range(self, synthetic_ecg_image_tensor):
        from core.ecg_image_loader import get_ecg_image_classifier
        clf = get_ecg_image_classifier()
        results = clf.predict(synthetic_ecg_image_tensor)
        assert 0.0 <= results[0]["confidence"] <= 1.0

    def test_class_label_valid(self, synthetic_ecg_image_tensor):
        from core.ecg_image_loader import get_ecg_image_classifier, CLASS_LABELS
        clf = get_ecg_image_classifier()
        results = clf.predict(synthetic_ecg_image_tensor)
        assert results[0]["class_label"] in CLASS_LABELS


# ──────────────────────────────────────────────────────────────────────────────
# 3. Heart Sound Classifier
# ──────────────────────────────────────────────────────────────────────────────

class TestHeartSoundClassifier:
    def test_output_shape(self, synthetic_mel_tensor):
        from core.heart_sound_loader import get_heart_sound_classifier, MURMUR_CLASSES
        clf = get_heart_sound_classifier()
        results = clf.predict(synthetic_mel_tensor)
        assert len(results) == 1
        assert "murmur_class" in results[0]

    def test_murmur_class_valid(self, synthetic_mel_tensor):
        from core.heart_sound_loader import get_heart_sound_classifier, MURMUR_CLASSES
        clf = get_heart_sound_classifier()
        results = clf.predict(synthetic_mel_tensor)
        assert results[0]["murmur_class"] in MURMUR_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# 4. VFDB Arrhythmia Detector
# ──────────────────────────────────────────────────────────────────────────────

class TestVFDBDetector:
    def test_output_shape(self, synthetic_vfdb_tensor):
        from core.vfdb_loader import get_vfdb_detector, ALL_RHYTHMS
        detector = get_vfdb_detector()
        results = detector.predict(synthetic_vfdb_tensor)
        assert len(results) == 1
        assert "rhythm_class" in results[0]

    def test_rhythm_class_valid(self, synthetic_vfdb_tensor):
        from core.vfdb_loader import get_vfdb_detector, ALL_RHYTHMS
        detector = get_vfdb_detector()
        results = detector.predict(synthetic_vfdb_tensor)
        assert results[0]["rhythm_class"] in ALL_RHYTHMS


# ──────────────────────────────────────────────────────────────────────────────
# 5. CardioFusion Hybrid Model
# ──────────────────────────────────────────────────────────────────────────────

class TestCardioFusion:
    def test_demo_inference(self, synthetic_echo_tensor, synthetic_ecg_image_tensor):
        from core.hybrid_model import get_cardiofusion_model
        model = get_cardiofusion_model()
        inputs = {
            "echo_video": synthetic_echo_tensor,
            "ecg_image":  synthetic_ecg_image_tensor,
        }
        result = model.inference(inputs)
        assert isinstance(result, dict)
        assert "modalities_used" in result

    def test_single_modality_echo(self, synthetic_echo_tensor):
        from core.hybrid_model import get_cardiofusion_model
        model = get_cardiofusion_model()
        result = model.inference({"echo_video": synthetic_echo_tensor})
        assert isinstance(result, dict)

    def test_single_modality_ecg_image(self, synthetic_ecg_image_tensor):
        from core.hybrid_model import get_cardiofusion_model
        model = get_cardiofusion_model()
        result = model.inference({"ecg_image": synthetic_ecg_image_tensor})
        assert isinstance(result, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Arrhythmia Dataset (demo mode)
# ──────────────────────────────────────────────────────────────────────────────

class TestArrhythmiaVideoDataset:
    def test_demo_loads(self):
        from core.cardiac_arrhythmia_video_loader import CardiacArrhythmiaVideoDataset
        ds = CardiacArrhythmiaVideoDataset()  # no real data → demo mode
        assert len(ds) > 0

    def test_demo_has_labels(self):
        from core.cardiac_arrhythmia_video_loader import CardiacArrhythmiaVideoDataset
        ds = CardiacArrhythmiaVideoDataset()
        assert all("arrhythmia" in s for s in ds.samples)
        assert all(s["arrhythmia"] in [0, 1] for s in ds.samples)

    def test_load_tensor_demo(self):
        from core.cardiac_arrhythmia_video_loader import CardiacArrhythmiaVideoDataset
        ds = CardiacArrhythmiaVideoDataset()
        tensor, labels = ds.load_video_tensor(0)
        assert tensor.dim() == 4  # (1, T, H, W)
        assert "arrhythmia" in labels

    def test_eda_summary(self):
        from core.cardiac_arrhythmia_video_loader import CardiacArrhythmiaVideoDataset
        ds = CardiacArrhythmiaVideoDataset()
        summary = ds.get_eda_summary()
        assert "total_videos" in summary
        assert "arrhythmia_count" in summary
        assert "normal_count" in summary
