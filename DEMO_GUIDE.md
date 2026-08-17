# Demo Walkthrough Guide — Step-by-Step

> Follow these steps **in order** to demonstrate every feature of the DeepCardio-RAG & Arthritis Analysis System. Each step takes approximately **30–60 seconds**.

---

## Prerequisites

1. Run `run_demo.bat` (double-click) — this starts the backend and opens the dashboard.
2. Wait until the terminal shows `Application startup complete.`
3. The browser should open `http://localhost:8000/dashboard/` automatically.

---

## Step 1: ECG Dashboard (Default Page)

**What you see:** Real-time animated ECG waveform (Lead II), vital stats (HR, PR, QRS, QTc), and two empty cards for the report and RAG context.

**Actions:**
1. ✅ Observe the live ECG animation for 3–5 seconds
2. ✅ Click **"Run DeepCardio-RAG"** button (blue gradient button below the ECG chart)
3. ⏳ Wait ~5–10 seconds for the AI inference to complete
4. ✅ See the **Diagnostic Report** populate on the right panel
5. ✅ See the **Retrieved Clinical Knowledge (RAG)** cards appear below the ECG chart
6. ✅ Note the **metrics**: Inference Time, BLEU-4 Score, Feature Accuracy, Hallucination Delta
7. ✅ Click **"Clinical PDF"** → downloads `ECG_Clinical_Report.pdf`
8. ✅ Click **"Patient PDF"** → downloads `ECG_Patient_Report.pdf`

---

## Step 2: Arthritis EDA Analysis

**Navigate:** Click **"Arthritis Analysis"** (bone icon) in the left sidebar.

**Actions:**
1. ✅ Click **"Load EDA"** button (top right)
2. ✅ See 5 stat cards populate: Total Patients, Features, Male/Female count, Avg Age
3. ✅ See 4 charts render:
   - **Inflammatory Markers** (red bars) — ESR, CRP, RA, ASO
   - **Hematology Markers** (blue bars) — Hb, RBC, PCV, MCV, MCH, MCHC
   - **Biochemistry Markers** (green bars) — Urea, Creatinine, Calcium, Uric Acid
   - **Missing Data %** (orange horizontal bars)
4. ✅ Click **"Train Model"** button
5. ⏳ Wait ~3–5 seconds for training to complete
6. ✅ See training results: Test Accuracy, CV Accuracy, AUC-ROC, Training Time
7. ✅ See **Top Feature Importances** with animated bar chart

---

## Step 3: Patient Predictor

**Navigate:** Click **"Patient Predictor"** (stethoscope icon) in the left sidebar.

**Actions:**
1. ✅ Fill in the blood test form with these sample values:

   | Field          | Value |
   |----------------|-------|
   | Gender         | Female |
   | Age            | 55     |
   | TC (WBC)       | 9000   |
   | ESR (1st hr)   | 45     |
   | ESR (2nd hr)   | 80     |
   | Hemoglobin     | 11.0   |
   | RBC            | 4.2    |
   | PCV            | 35     |
   | MCV            | 83     |
   | MCH            | 26     |
   | MCHC           | 31     |
   | Platelet Count | 280000 |
   | ASO            | 250    |
   | RBS            | 110    |
   | Urea           | 30     |
   | Creatinine     | 0.9    |
   | Calcium        | 9.0    |
   | Uric Acid      | 5.5    |

2. ✅ Click **"Predict Arthritis Risk"** button
3. ✅ See the **Risk Badge** (HIGH or LOW) with color coding
4. ✅ See the **Confidence Bar** and probability breakdown
5. ✅ Click **"Clinical PDF"** → downloads `Arthritis_Risk_Clinical_Report.pdf`
6. ✅ Click **"Patient PDF"** → downloads `Arthritis_Risk_Patient_Report.pdf`

---

## Step 4: Patient Records

**Navigate:** Click **"Patient Records"** (users icon) in the left sidebar.

**Actions:**
1. ✅ Click **"Load Records"** button
2. ✅ See the patient table load with 20 records from the APD dataset
3. ✅ Note the **Status** column showing "Normal" (green) or "High Risk" (red) pills
4. ✅ Scroll through the table to see different patient profiles

---

## Step 5: Vector DB Stats

**Navigate:** Click **"Vector DB Stats"** (database icon) in the left sidebar.

**Actions:**
1. ✅ Click **"Refresh Stats"** button
2. ✅ See 3 information cards:
   - **Milvus Vector Database** — Status (Online), Collection, Records, Embedding Dim, Metric
   - **APD Dataset** — Source, Records, Features, Format, Missing Data tags
   - **ML Models** — ECG Encoder, Report Generator, Arthritis Predictor details

---

## Step 6: Settings

**Navigate:** Click **"Settings"** (gear icon) in the left sidebar.

**Actions:**
1. ✅ Browse the 4 settings cards:
   - **Backend Configuration** — API Server, models list
   - **Display Preferences** — Theme toggle, auto-refresh, font size
   - **Physician Profile** — Name, department, credentials, institution
   - **About System** — Version, build date, dataset, license

---

## Summary of Downloads

After completing all steps, you should have these PDF files downloaded:

| # | File                                 | Generated From      |
|---|--------------------------------------|---------------------|
| 1 | `ECG_Clinical_Report.pdf`            | Step 1 — ECG        |
| 2 | `ECG_Patient_Report.pdf`             | Step 1 — ECG        |
| 3 | `Arthritis_Risk_Clinical_Report.pdf` | Step 3 — Predictor  |
| 4 | `Arthritis_Risk_Patient_Report.pdf`  | Step 3 — Predictor  |

---

## Recording Tips (for Demo Video)

- **Screen resolution:** Use 1920×1080 for best quality
- **Browser:** Chrome or Edge recommended
- **Wait times:** Pause 2–3 seconds after each action for visual clarity
- **Hover effects:** Move your mouse over cards and buttons to show the micro-animations
- **Total time:** Full walkthrough takes approximately **4–6 minutes**
