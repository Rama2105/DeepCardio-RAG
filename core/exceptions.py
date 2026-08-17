"""
core/exceptions.py — Custom Exception Hierarchy
================================================
Provides typed exceptions for the DeepCardio-RAG system so that each
error surfaces with the correct HTTP status code and a structured message.

Usage:
    from core.exceptions import ModelInferenceError, DataValidationError

    raise DataValidationError("Age must be between 0 and 150")
    raise ModelInferenceError("ECG encoder failed", detail="OOM on GPU")
"""

from fastapi import HTTPException
from typing import Optional, Any, Dict


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class DeepCardioError(HTTPException):
    """Base class for all DeepCardio application exceptions."""
    status_code: int = 500
    error_code:  str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        detail: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_detail = {
            "error_code": self.error_code,
            "message":    message,
        }
        if detail:
            full_detail["detail"] = detail
        if context:
            full_detail["context"] = context
        super().__init__(status_code=self.status_code, detail=full_detail)


# ──────────────────────────────────────────────────────────────────────────────
# 400 — Client / Validation Errors
# ──────────────────────────────────────────────────────────────────────────────

class DataValidationError(DeepCardioError):
    """Raised when input data fails validation checks (bad field value, wrong type, etc.)."""
    status_code = 400
    error_code  = "VALIDATION_ERROR"


class UnsupportedFileTypeError(DeepCardioError):
    """Raised when an uploaded file has an unsupported extension or MIME type."""
    status_code = 415
    error_code  = "UNSUPPORTED_FILE_TYPE"


class FileTooLargeError(DeepCardioError):
    """Raised when an uploaded file exceeds the maximum size limit."""
    status_code = 413
    error_code  = "FILE_TOO_LARGE"


class InvalidTensorShapeError(DeepCardioError):
    """Raised when a tensor has an unexpected shape before model inference."""
    status_code = 422
    error_code  = "INVALID_TENSOR_SHAPE"


class EmptyVideoError(DeepCardioError):
    """Raised when a video file yields zero readable frames."""
    status_code = 422
    error_code  = "EMPTY_VIDEO"


class EmptyAudioError(DeepCardioError):
    """Raised when an audio file yields an empty signal."""
    status_code = 422
    error_code  = "EMPTY_AUDIO"


# ──────────────────────────────────────────────────────────────────────────────
# 401 / 403 — Auth Errors
# ──────────────────────────────────────────────────────────────────────────────

class AuthenticationError(DeepCardioError):
    """Raised when credentials are missing or invalid."""
    status_code = 401
    error_code  = "AUTHENTICATION_FAILED"


class AuthorizationError(DeepCardioError):
    """Raised when a user lacks permission for the requested resource."""
    status_code = 403
    error_code  = "AUTHORIZATION_FAILED"


class TokenExpiredError(DeepCardioError):
    """Raised when a JWT token has expired."""
    status_code = 401
    error_code  = "TOKEN_EXPIRED"


# ──────────────────────────────────────────────────────────────────────────────
# 404 — Not Found
# ──────────────────────────────────────────────────────────────────────────────

class DatasetNotFoundError(DeepCardioError):
    """Raised when a required dataset directory or file is missing."""
    status_code = 404
    error_code  = "DATASET_NOT_FOUND"


class RecordNotFoundError(DeepCardioError):
    """Raised when a specific record / index does not exist in the dataset."""
    status_code = 404
    error_code  = "RECORD_NOT_FOUND"


class ModelWeightsNotFoundError(DeepCardioError):
    """Raised when saved model weights (.pt / .pkl) cannot be located."""
    status_code = 404
    error_code  = "MODEL_WEIGHTS_NOT_FOUND"


# ──────────────────────────────────────────────────────────────────────────────
# 500 / 503 — Server / Inference Errors
# ──────────────────────────────────────────────────────────────────────────────

class ModelInferenceError(DeepCardioError):
    """Raised when model forward pass fails (OOM, NaN output, etc.)."""
    status_code = 500
    error_code  = "MODEL_INFERENCE_ERROR"


class ModelNotTrainedError(DeepCardioError):
    """Raised when prediction is requested but the model has not been trained yet."""
    status_code = 503
    error_code  = "MODEL_NOT_TRAINED"


class DatabaseConnectionError(DeepCardioError):
    """Raised when the vector database (Milvus / ChromaDB / FAISS) cannot be reached."""
    status_code = 503
    error_code  = "DATABASE_CONNECTION_ERROR"


class EmbeddingError(DeepCardioError):
    """Raised when sentence embedding fails."""
    status_code = 500
    error_code  = "EMBEDDING_ERROR"


class ReportGenerationError(DeepCardioError):
    """Raised when PDF or clinical report generation fails."""
    status_code = 500
    error_code  = "REPORT_GENERATION_ERROR"


class VideoProcessingError(DeepCardioError):
    """Raised when OpenCV fails to decode/process a video file."""
    status_code = 500
    error_code  = "VIDEO_PROCESSING_ERROR"


class AudioProcessingError(DeepCardioError):
    """Raised when audio signal processing (FFT, mel spectrogram) fails."""
    status_code = 500
    error_code  = "AUDIO_PROCESSING_ERROR"


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def wrap_exception(exc: Exception, default_cls=ModelInferenceError) -> DeepCardioError:
    """
    Convert a generic Python exception into a typed DeepCardioError.
    Useful inside except blocks where the exact exception type is unknown.

    Example:
        try:
            result = model(tensor)
        except Exception as e:
            raise wrap_exception(e)
    """
    if isinstance(exc, DeepCardioError):
        return exc
    return default_cls(
        message=type(exc).__name__,
        detail=str(exc),
    )
