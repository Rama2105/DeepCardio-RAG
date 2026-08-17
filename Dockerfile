FROM python:3.10-slim

LABEL maintainer="DeepCardio-RAG Research Team"
LABEL description="DeepCardio-RAG: Multi-modal cardiac AI diagnostic system"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt requirements-dev.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Create required directories ───────────────────────────────────────────────
RUN mkdir -p logs data database frontend

# ── Environment defaults (override in docker-compose or at runtime) ───────────
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    VECTOR_DB_BACKEND=faiss \
    MAX_UPLOAD_MB=100 \
    PORT=8000

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
