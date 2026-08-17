# DeepCardio-RAG

AI-assisted clinical diagnostic research prototype combining ECG analysis, heart-sound
and echo modules, arthritis risk prediction, and retrieval-grounded report generation.

> **Research prototype. Not validated for clinical use.** No output from this system
> should inform patient care. Several modules are untrained or under-evaluated — see
> [Module status](#module-status).

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Rama2105/DeepCardio-RAG.git
cd DeepCardio-RAG
```

### 2. Python environment

Python 3.10–3.12 recommended.

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment file

```bash
# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
```

Then open `.env` and set `SECRET_KEY` to your own value:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**The app will not start without a `.env`.**

### 4. Vector database (Milvus)

Milvus standalone `v2.6.2` runs locally in Docker on `localhost:19530`.

```bash
cd C:\milvus
standalone.bat start      # also: stop | delete  (delete WIPES all data)
```

Then seed the knowledge base:

```bash
python -m database.seed_data
```

Verify from a **separate process** — the backend chain fails silently, so the
startup log is not trustworthy:

```bash
python -c "from pymilvus import connections, Collection; connections.connect(host='localhost', port='19530'); c=Collection('cardio_knowledge_base'); c.load(); print(c.num_entities)"
```

Expect `67`.

> The container does not auto-start. After a reboot it sits in `Exited`; launch
> Docker Desktop, then `docker start milvus-standalone`. Data survives — only
> `standalone.bat delete` wipes it.
>
> Pin the server to the **2.6** line. `standalone_embed.bat` from the repo's
> `master` branch pulls `v3.0-beta`, whose API breaks the 2.x ORM calls in
> `db_manager.py`. Fetch it from the `v2.6.9` tag instead. Client is
> `pymilvus==2.6.9`.

### 5. Run

```bash
python main.py
```

Dashboard at http://localhost:8000/dashboard/

---

## What is NOT in this repository

Three categories are deliberately excluded — the repo is code only (~26 MB).

| Missing | Why | How to get it |
|---|---|---|
| `data/*.csv`, `data/vfdb/`, `data/circor-heart-sound/` | ~950 MB of public benchmarks; two files exceed GitHub's 100 MB limit | `python data/download_datasets.py` (needs a Kaggle API key) |
| `*.pt` / `*.pkl` trained weights | Up to 110 MB each | Ask the maintainer for the shared Drive artifacts folder |
| `.env` | Contains `SECRET_KEY` | Copy from `.env.example` |

### Kaggle credentials

`data/download_datasets.py` needs a Kaggle API token. Get one from
kaggle.com → Account → Create New API Token, then:

```bash
# Mac/Linux
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
# Windows PowerShell
$env:KAGGLE_USERNAME="your_username"; $env:KAGGLE_KEY="your_key"
```

---

## Training

**All training runs on Colab GPU — do not train locally, even small modules.**

Use `DeepCardio_Genuine_Training.ipynb`. Set **Runtime → Change runtime type → GPU**
before running. Trained weights and `data/genuine_results.json` should be saved to
the shared Drive artifacts folder, not committed here.

---

## Layout

```
core/          ML pipelines — one module per clinical task, plus train_*.py scripts
database/      Vector DB layer (db_manager.py) and knowledge-base seeding
frontend/      Vanilla HTML/JS dashboard, served at /dashboard/
frontend-react/  Newer React + Vite frontend
tests/         pytest suite
data/          Datasets and trained artifacts (mostly gitignored)
documentation/ Design notes
```

Key entry points:

- `main.py` — FastAPI server, all REST endpoints
- `core/pipeline.py` — ECG pipeline (1D-CNN encoder → Milvus retrieval → GPT-2 report)
- `core/arthritis_pipeline.py` — TabularBERT-MoE + stacking ensemble
- `core/safety_gating.py` — physiological validation and directive reframing
- `database/db_manager.py` — vector store with a **silent** fallback chain
  (external Milvus → Milvus Lite → ChromaDB → FAISS). `init_db()` logs
  `✓ Vector DB ready` even on the last rung, so **always check the backend name
  in the log** before trusting retrieval results.

See `PROJECT_ARCHITECTURE.md` for detail and `CLAUDE.md` for working conventions.

---

## Module status

Trained and evaluated on public benchmarks: PTB-XL ECG, ECG arrhythmia, VFDB,
PCG/CirCor, heart disease, arthritis (NHANES).

**Echo (EchoNet) is not trained** — its outputs are suppressed rather than reported.
No module currently matches its published benchmark. Report-generation quality has
never been formally evaluated. Genuine measured values live in
`data/genuine_results.json`; do not quote figures from any manuscript draft as if
they were measurements.

---

## Contributing

`main` is the shared branch. Work on your own branch and open a pull request.

```bash
git checkout -b feature/your-change
# ... work ...
git add -A
git commit -m "describe what changed"
git push -u origin feature/your-change
```

**Notebooks:** run **Edit → Clear all outputs** before committing. Notebook outputs
produce huge diffs and conflicts that cannot be merged.

**Never commit:** API keys or tokens, `.env`, dataset files, trained weights.
