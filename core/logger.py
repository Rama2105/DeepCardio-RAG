"""
core/logger.py — Structured Logging
=====================================
Replaces all print() calls with a proper logging setup:
  - Rotating file handler (log/deepcardio.log)
  - Console handler
  - Configurable log level from settings
  - Context fields: patient_id, endpoint, model, latency

Usage:
    from core.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Inference complete", extra={"patient_id": "APD-0001", "latency_ms": 123})
    logger.warning("EF below threshold", extra={"ef": 28.5})
    logger.error("Model failed", exc_info=True)
"""

import logging
import logging.handlers
import os
import json
import time
from typing import Optional, Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# JSON formatter for structured log lines
# ──────────────────────────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON line:
        {"ts": "...", "level": "INFO", "logger": "...", "msg": "...", "patient_id": "..."}
    """

    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "id", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        log_dict: Dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.message,
        }
        # Append any extra context fields added via logger.info(..., extra={...})
        for key, val in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_dict[key] = val

        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Logger factory
# ──────────────────────────────────────────────────────────────────────────────

_configured = False

def _configure_root_logger():
    """Configure the root logger once on first call."""
    global _configured
    if _configured:
        return
    _configured = True

    try:
        from config import settings
        log_level    = getattr(logging, settings.log_level.upper(), logging.INFO)
        log_dir      = settings.log_dir
        max_bytes    = settings.log_max_bytes
        backup_count = settings.log_backup_count
    except Exception:
        log_level    = logging.INFO
        log_dir      = "./logs"
        max_bytes    = 10 * 1024 * 1024
        backup_count = 5

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "deepcardio.log")

    root = logging.getLogger()
    root.setLevel(log_level)

    if root.handlers:
        return  # Already configured (e.g., uvicorn already set up handlers)

    # ── Rotating file handler (JSON lines) ────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(log_level)

    # ── Console handler (human-readable) ──────────────────────────────────
    console_handler = logging.StreamHandler()
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(log_level)

    root.addHandler(file_handler)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Call once per module:
        logger = get_logger(__name__)
    """
    _configure_root_logger()
    return logging.getLogger(name)


# ──────────────────────────────────────────────────────────────────────────────
# Audit logger (separate file for PII-sensitive access events)
# ──────────────────────────────────────────────────────────────────────────────

def get_audit_logger() -> logging.Logger:
    """
    Returns a dedicated audit logger that writes to logs/audit.log.
    Use for patient data access, auth events, and report generation.
    """
    _configure_root_logger()
    audit = logging.getLogger("deepcardio.audit")
    if audit.handlers:
        return audit

    try:
        from config import settings
        log_dir = settings.log_dir
    except Exception:
        log_dir = "./logs"

    os.makedirs(log_dir, exist_ok=True)
    audit_file = os.path.join(log_dir, "audit.log")

    handler = logging.handlers.RotatingFileHandler(
        audit_file, maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    handler.setFormatter(JsonFormatter())
    audit.addHandler(handler)
    audit.setLevel(logging.INFO)
    audit.propagate = False
    return audit


# ──────────────────────────────────────────────────────────────────────────────
# PII Redaction helper
# ──────────────────────────────────────────────────────────────────────────────

_PII_FIELDS = {"name", "email", "phone", "address", "ssn", "dob", "patient_name"}

def redact_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of *data* with PII fields replaced by '[REDACTED]'.
    Safe to log without exposing personal information.
    """
    return {
        k: "[REDACTED]" if k.lower() in _PII_FIELDS else v
        for k, v in data.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Latency context manager
# ──────────────────────────────────────────────────────────────────────────────

class LoggedTimer:
    """
    Context manager that logs elapsed time on exit.

    Usage:
        with LoggedTimer(logger, "ECG inference", extra={"patient_id": "123"}):
            result = model(tensor)
    """
    def __init__(self, logger: logging.Logger, label: str,
                 extra: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.label  = label
        self.extra  = extra or {}

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        ctx = {**self.extra, "latency_ms": elapsed_ms}
        if exc_type is None:
            self.logger.info(f"{self.label} completed", extra=ctx)
        else:
            self.logger.error(
                f"{self.label} failed after {elapsed_ms}ms: {exc_val}",
                extra=ctx,
                exc_info=True,
            )
        return False  # re-raise exceptions
