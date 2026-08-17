# Running DeepCardio-RAG v3.0 on Google Colab

Full AI-powered cardiac diagnostic system running from a Google Colab notebook,
exposed via a public tunnel so you can open the dashboard in any browser.

---

## Prerequisites

- Google account (for Drive + Colab)
- The `DeepCardio-RAG` project folder in your Google Drive

---

## Step 1: Upload Project to Google Drive

1. Zip your local `DeepCardio-RAG` folder **excluding** `venv/`, `__pycache__/`, `*.pt` model files, and `data/arthritis_vector_db.pkl`
2. Upload the zip to Google Drive and extract it so the path is:
   ```
   MyDrive/DeepCardio-RAG/
   ```
3. Verify these files exist in the folder:
   - `main.py`
   - `requirements.txt`
   - `DeepCardio_Colab.ipynb`
   - `frontend/index.html`
   - `core/pipeline.py`

---

## Step 2: Open the Notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → Google Drive**
3. Navigate to `MyDrive/DeepCardio-RAG/` and open `DeepCardio_Colab.ipynb`
4. **Runtime → Change runtime type → T4 GPU** (recommended for faster inference)

---

## Step 3: Run Cells in Order

Run each cell top-to-bottom. Do **not** skip cells.

| Cell | What it does |
|---|---|
| **Cell 1** | GPU check + mount Google Drive + `cd` into project |
| **Cell 2** | Install all Python dependencies (PyTorch, FastAPI, Transformers, Milvus, kagglehub, fpdf2, etc.) |
| **Cell 3** | Kaggle credentials setup (needed for 70K Cardiovascular dataset download) |
| **Cell 4** | Download real datasets via kagglehub |
| **Cell 5** | EDA visualisations |
| **Cell 6** | Configure environment (PORT, device, tunnel flag) |
| **Cell 6B** | Start Milvus server + expose via bore.pub tunnel *(optional)* |
| **Cell 7** | Train models on real data |
| **Cell 8** | **Start FastAPI server + create public URL** ← main launch cell |
| **Cell 9** | Test all API endpoints |
| **Cell 10** | Generate PDF reports |
| **Cell 11** | Full evaluation metrics |

---

## Step 4: Access the Dashboard

After **Cell 8** runs you will see output like:

```
Server started on http://0.0.0.0:8000
Public URL: https://xxxx-xxxx.loca.lt
```

1. Click the `https://xxxx-xxxx.loca.lt` link
2. You may see a LocalTunnel warning page — paste the **IP password** shown in the cell output and click Submit
3. Add `/dashboard/` to the URL:
   ```
   https://xxxx-xxxx.loca.lt/dashboard/
   ```

You should see the full DeepCardio-RAG v3.0 dashboard.

---

## New Features in v3.0 (accessible from the dashboard)

| Page | Feature |
|---|---|
| ECG Dashboard | Upload real ECG files (.csv / .json) via "Upload ECG" button |
| ECG Images | Input → Processing → Output 4-stage pipeline visualisation |
| Heart Sound | Combined murmur detection + malignant ventricular arrhythmia detection |
| CardioFusion | Run all 4 models → unified doctor & patient reports → PDF download |
| Validation | Compare model vs 20+ year cardiologist benchmarks (5 published references) |
| Arthritis | Real 70K-record dataset (Cardiovascular Disease, Kaggle) |
| Settings | Updated: Transformer-MoE encoder info, all dataset sources, sample ECG download |

---

## Kaggle Credentials (for real datasets)

Cell 3 handles this. You need a `kaggle.json` API token:

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New API Token**
2. Download `kaggle.json`
3. Upload it when Cell 3 prompts you (or place it at `/root/.kaggle/kaggle.json` manually)

Without credentials the system falls back to the APD dataset (102 records) for arthritis.

---

## ECG Upload Format

The "Upload ECG" button accepts:

- **CSV**: 2500+ rows × 12 columns (one row per sample, one column per lead)
- **JSON**: Array of 12 arrays `[[lead0 samples], [lead1 samples], ...]`

Download a sample file: `https://your-tunnel-url.loca.lt/api/analyze/sample-ecg`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Not Found" at tunnel URL | Append `/dashboard/` to the URL |
| Buttons show "ensure backend is running" | Re-run Cell 8 to restart the server |
| Tunnel link stops working | Re-run Cell 8 to get a new URL |
| GPU out of memory | Runtime → Disconnect → Reconnect; use CPU mode |
| Kaggle download fails | Check `kaggle.json` is valid; or skip — APD fallback is used |
| `ModuleNotFoundError` | Re-run Cell 2 to reinstall dependencies |
