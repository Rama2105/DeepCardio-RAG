# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeepCardio-RAG** is an AI-powered clinical diagnostic system with two main capabilities:
- **ECG Analysis**: 1D-CNN encoder → Milvus vector DB (RAG) → GPT-2 report generation
- **Arthritis Risk Prediction**: Tabular BERT + Mixture of Experts (MoE) classifier trained on the Arthritis Profile Dataset (APD, 102 patients, 22+ features)

## Commands

### Backend

```bash
# Quick start (Windows — activates venv, installs deps, starts server)
run_demo.bat

# Manual start
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py          # FastAPI on http://localhost:8000
```

### Milvus (local, Docker Desktop)

Milvus standalone `v2.6.2` runs locally in container `milvus-standalone` on `localhost:19530`, managed from `C:\milvus`:

```bash
cd /d C:\milvus
standalone.bat start      # also: stop | delete  (delete WIPES all data)
```

Data lives in `C:\milvus\volumes` — deliberately outside `G:\My Drive\`, since Drive sync corrupts live database files.

The container does not auto-start. After a reboot it sits in `Exited (255)` — launch Docker Desktop, then `docker start milvus-standalone` (data survives; only `standalone.bat delete` wipes it).

Pin the server to the **2.6** line. The `standalone_embed.bat` script from the repo's `master` branch pulls `v3.0-beta`, whose API breaks the 2.x ORM calls in `db_manager.py`; fetch it from the `v2.6.9` tag instead. The client is `pymilvus==2.6.9`.

Seed the knowledge base (16 documents, not the "50,000+" the code comments claim):

```bash
python -m database.seed_data
```

Verify from a **separate process** rather than trusting the startup log, because the backend chain fails silently:

```bash
python -c "from pymilvus import connections, Collection; connections.connect(host='localhost', port='19530'); c=Collection('cardio_knowledge_base'); c.load(); print(c.num_entities)"
```

`MILVUS_HOST`/`MILVUS_PORT` come from `.env`. Real environment variables take precedence over `.env` values — worth remembering when a `set MILVUS_HOST=...` appears to half-work. Note `connect_colab_milvus.py` (no args) rewrites `.env` from `milvus_tunnel.json`, which is only for Colab-tunnel setups.

### Frontend (React, in frontend-react/)

```bash
cd frontend-react
npm install
npm run dev        # Vite dev server
npm run build      # Production build to dist/
npm run lint       # ESLint
npm run preview    # Preview production build
```

The vanilla HTML/JS frontend lives in `frontend/` and is served statically by FastAPI at `/dashboard/`. The React frontend in `frontend-react/` is a newer alternative.

## Architecture

### Entry Points

- `main.py` — FastAPI server; all REST endpoints are defined here
- `frontend/app.js` — All client-side logic for the vanilla SPA
- `frontend-react/src/` — React + Vite frontend

### Core ML Pipelines (`core/`)

- `pipeline.py` — ECG pipeline: encodes ECG signal via 1D-CNN/ResNet-34 encoder (`ECG_EMBED_DIM = 256`), projects 256 → 768 to query Milvus (`MILVUS_DIM = 768`), retrieves context, generates report with GPT-2 using soft-prompt injection
- `arthritis_pipeline.py` — Arthritis pipeline: loads `data/APDDataset.xlsx`, trains Tabular BERT + MoE, handles EDA and prediction; saves weights to `data/arthritis_bert_moe_model.pt`
- `pdf_generator.py` — Generates dual-audience PDFs (clinical vs. patient) using fpdf2
- `hybrid_model.py` — Multi-modal model combining ECG, echo, and sound inputs

### Database Layer (`database/`)

- `db_manager.py` — Vector DB layer. Collection `cardio_knowledge_base`: **768-dim, L2 metric, IVF_FLAT index (nlist=2048)**, per `config.py`. Backends are tried in order and **fall back silently** — external Milvus → Milvus Lite → ChromaDB → FAISS in-memory. `init_db()` logs `✓ Vector DB ready` even on the last rung, so always check the backend name in the log line before trusting retrieval results.
- `seed_data.py` — Seeds clinical guidelines into Milvus at startup

### Data & Saved Models (`data/`)

- `APDDataset.xlsx` — Source dataset for arthritis (do not modify)
- `arthritis_bert_moe_model.pt` + `arthritis_bert_moe_scaler.pkl` — Primary trained model
- `arthritis_model.pkl` + `arthritis_scaler.pkl` — GradientBoosting fallback models
- `arthritis_vector_db.pkl` — Pickled patient embeddings for local vector search

### API Surface (main.py)

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Full ECG analysis pipeline |
| `GET /api/arthritis/eda` | EDA summary stats |
| `GET /api/arthritis/correlation` | Feature correlation matrix |
| `POST /api/arthritis/train` | Train BERT+MoE model |
| `POST /api/arthritis/predict` | Arthritis risk prediction |
| `POST /api/pdf/ecg` | Generate ECG PDF report |
| `POST /api/pdf/arthritis` | Generate arthritis PDF report |
| `GET /api/patients` | Patient records from APD |
| `GET /api/db/stats` | Milvus + system statistics |

### Frontend Pages (6-page SPA)

1. ECG Dashboard — signal visualization, run analysis, RAG context, PDF download
2. Arthritis Analysis — EDA charts (Chart.js), model training, feature importances
3. Patient Predictor — 22-field blood test form → risk prediction
4. Patient Records — tabular view of APD dataset
5. Vector DB Stats — Milvus status, collection info, model metadata
6. Settings — backend config, physician profile

## Google Colab Deployment

Use `DeepCardio_RAG_Google_Colab.ipynb`. Requires Google Drive mount and LocalTunnel for external access. See `COLAB_GUIDE.md` for step-by-step instructions.
