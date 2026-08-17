"""
config.py — Centralized Configuration Management
=================================================
Reads from environment variables / .env file.
All hardcoded paths, dimensions, and secrets are consolidated here.

Usage:
    from config import settings
    print(settings.database_url)

Create a .env file in the project root (never commit it):
    SECRET_KEY=your-secret-key
    MILVUS_URI=./database/deepcardio.db
    ECHONET_DIR=./data/EchoNet-Dynamic
    ALLOWED_ORIGINS=http://localhost:3000,https://xxxx.loca.lt
"""

import os
from typing import List, Optional
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False


BASE_DIR = Path(__file__).parent


if _USE_PYDANTIC:
    class Settings(BaseSettings):
        # ── App ──────────────────────────────────────────────────────────────
        app_name:    str = "DeepCardio-RAG"
        app_version: str = "1.0.0"
        debug:       bool = False
        environment: str = "development"   # development | production | testing

        # ── Security ─────────────────────────────────────────────────────────
        secret_key:           str  = "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING"
        algorithm:            str  = "HS256"
        access_token_expire_minutes: int = 60
        allowed_origins:      str  = "*"   # comma-separated list for CORS

        # ── Vector DB (External Milvus Production) ──────────────────────────────
        milvus_host:          str  = "10.228.1.9"
        milvus_port:          str  = "19530"
        milvus_uri:           str  = str(BASE_DIR / "database" / "deepcardio.db")  # Lite fallback
        collection_name:      str  = "cardio_knowledge_base"
        embedding_dim:        int  = 768   # Sentence-BERT all-mpnet-base-v2
        vector_db_backend:    str  = "milvus"   # milvus | faiss | chromadb
        milvus_index_type:    str  = "IVF_FLAT"
        milvus_nlist:         int  = 2048
        milvus_metric:        str  = "L2"

        # ── Dataset paths ─────────────────────────────────────────────────────
        echonet_dir:          str  = str(BASE_DIR / "data" / "EchoNet-Dynamic")
        vfdb_dir:             str  = str(BASE_DIR / "data" / "vfdb")
        apd_xlsx:             str  = str(BASE_DIR / "data" / "APDDataset.xlsx")
        arrhythmia_model_pt:  str  = str(BASE_DIR / "data" / "arthritis_bert_moe_model.pt")

        # ── New Kaggle datasets ──────────────────────────────────────────────
        heart_disease_csv:    str  = str(BASE_DIR / "data" / "heart.csv")
        mitbih_train_csv:     str  = str(BASE_DIR / "data" / "mitbih_train.csv")
        mitbih_test_csv:      str  = str(BASE_DIR / "data" / "mitbih_test.csv")

        # ── Video / Frame settings ────────────────────────────────────────────
        num_frames:           int  = 32
        frame_size:           int  = 112
        echo_mean:            float = 0.131
        echo_std:             float = 0.289

        # ── ECG settings ──────────────────────────────────────────────────────
        ecg_leads:            int  = 12
        ecg_length:           int  = 1250
        ecg_embed_dim:        int  = 384

        # ── Upload limits ────────────────────────────────────────────────────
        max_upload_mb:        int  = 100
        allowed_video_exts:   str  = ".avi,.mp4,.mov"
        allowed_image_exts:   str  = ".png,.jpg,.jpeg,.bmp"
        allowed_audio_exts:   str  = ".wav,.mp3,.flac"

        # ── Logging ───────────────────────────────────────────────────────────
        log_level:            str  = "INFO"
        log_dir:              str  = str(BASE_DIR / "logs")
        log_max_bytes:        int  = 10 * 1024 * 1024   # 10 MB
        log_backup_count:     int  = 5

        # ── MLflow ────────────────────────────────────────────────────────────
        mlflow_tracking_uri:  str  = "http://localhost:5000"
        mlflow_experiment:    str  = "DeepCardio"
        enable_mlflow:        bool = False

        # ── Prometheus ────────────────────────────────────────────────────────
        enable_metrics:       bool = True
        metrics_path:         str  = "/metrics"

        # ── Roles ─────────────────────────────────────────────────────────────
        admin_username:       str  = "admin"
        admin_password_hash:  str  = ""   # bcrypt hash; set in .env

        class Config:
            env_file = str(BASE_DIR / ".env")
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore" # silently ignore env vars not defined in settings

        @property
        def cors_origins(self) -> List[str]:
            if self.allowed_origins == "*":
                return ["*"]
            return [o.strip() for o in self.allowed_origins.split(",")]

        @property
        def max_upload_bytes(self) -> int:
            return self.max_upload_mb * 1024 * 1024

        @property
        def allowed_video_ext_list(self) -> List[str]:
            return [e.strip() for e in self.allowed_video_exts.split(",")]

        @property
        def allowed_image_ext_list(self) -> List[str]:
            return [e.strip() for e in self.allowed_image_exts.split(",")]

        @property
        def allowed_audio_ext_list(self) -> List[str]:
            return [e.strip() for e in self.allowed_audio_exts.split(",")]

    settings = Settings()

else:
    # Fallback: plain object reading from os.environ when pydantic-settings not installed
    class _SimpleSettings:
        app_name             = "DeepCardio-RAG"
        app_version          = "1.0.0"
        debug                = os.getenv("DEBUG", "false").lower() == "true"
        environment          = os.getenv("ENVIRONMENT", "development")
        secret_key           = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
        algorithm            = "HS256"
        access_token_expire_minutes = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
        allowed_origins      = os.getenv("ALLOWED_ORIGINS", "*")
        milvus_host          = os.getenv("MILVUS_HOST", "10.228.1.9")
        milvus_port          = os.getenv("MILVUS_PORT", "19530")
        milvus_uri           = os.getenv("MILVUS_URI", str(BASE_DIR / "database" / "deepcardio.db"))
        collection_name      = "cardio_knowledge_base"
        embedding_dim        = 768
        vector_db_backend    = os.getenv("VECTOR_DB_BACKEND", "milvus")
        milvus_index_type    = "IVF_FLAT"
        milvus_nlist         = 2048
        milvus_metric        = "L2"
        echonet_dir          = os.getenv("ECHONET_DIR", str(BASE_DIR / "data" / "EchoNet-Dynamic"))
        vfdb_dir             = os.getenv("VFDB_DIR", str(BASE_DIR / "data" / "vfdb"))
        apd_xlsx             = os.getenv("APD_XLSX", str(BASE_DIR / "data" / "APDDataset.xlsx"))
        num_frames           = 32
        frame_size           = 112
        echo_mean            = 0.131
        echo_std             = 0.289
        ecg_leads            = 12
        ecg_length           = 1250
        max_upload_mb        = int(os.getenv("MAX_UPLOAD_MB", "100"))
        log_level            = os.getenv("LOG_LEVEL", "INFO")
        log_dir              = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
        log_max_bytes        = 10 * 1024 * 1024
        log_backup_count     = 5
        enable_mlflow        = False
        enable_metrics       = True
        metrics_path         = "/metrics"

        @property
        def cors_origins(self):
            return ["*"] if self.allowed_origins == "*" else self.allowed_origins.split(",")

        @property
        def max_upload_bytes(self):
            return self.max_upload_mb * 1024 * 1024

    settings = _SimpleSettings()
