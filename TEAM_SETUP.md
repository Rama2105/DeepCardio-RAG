# Team setup

How to get access, run the project, and contribute changes.

There are two ways to work. **Colab** for anything involving training (it has the GPU).
**Local** for the dashboard, the API, and Milvus. Most people need both eventually;
start with whichever matches your first task.

---

## Step 1 — Access (everyone, once)

1. Create a free GitHub account at https://github.com/signup if you don't have one.
2. Send your **GitHub username** to the maintainer (Rama2105).
3. Accept the emailed repository invitation. Until you accept, the repo URL returns 404.

Then clone:

```bash
git clone https://github.com/Rama2105/DeepCardio-RAG.git
cd DeepCardio-RAG
```

You will be asked to sign in. On Windows a browser window opens — sign in with your
GitHub account. If you are asked for a password in the terminal instead, that is not
your account password: create a token at https://github.com/settings/tokens
(**Generate new token (classic)** → tick **`repo`**) and paste that.

---

## Step 2 — Get the data and weights

**These are not in the repository** and never will be — the datasets are ~950 MB and
two files exceed GitHub's 100 MB per-file limit, and model weights reach 110 MB.

Ask the maintainer to share the **Google Drive artifacts folder** with you. Then:

1. Open the shared folder in Google Drive
2. Right-click it → **Organise** → **Add shortcut to Drive** → choose **My Drive**
3. Make sure the shortcut is named exactly **`DeepCardio-RAG`**

This matters: the training notebook looks for
`/content/drive/MyDrive/DeepCardio-RAG/data`. A folder sitting only in "Shared with me"
resolves to a different path and the notebook will not find it.

Alternatively you can download the public datasets yourself with your own Kaggle
credentials (slower, but self-service):

```bash
pip install kaggle
python data/download_datasets.py
```

Get a Kaggle key from kaggle.com → Account → Create New API Token, then set
`KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables.

---

## Step 3a — Working in Colab (training)

**All training runs on Colab GPU. Do not train locally, even small modules.**

1. In Colab: **File → Open notebook → GitHub** tab → tick **Include private repos** →
   authorise → open `DeepCardio_Genuine_Training.ipynb`
2. **Runtime → Change runtime type → GPU**
3. Optional but convenient: click the **🔑 key icon** in the left sidebar, add a secret
   named `GITHUB_TOKEN` holding your personal access token, and enable notebook access.
   Without it the notebook prompts you for the token each session.
4. Run the first code cell (runtime detect). It clones the latest `main` from GitHub
   into `/content/DeepCardio-RAG` and symlinks `data/` to Drive. Expect:

```
code rev: <latest commit>
RUNTIME : Colab cloud
cwd     : /content/DeepCardio-RAG
GPU     : Tesla T4
```

5. Run the remaining cells for the modules you are working on.

Every session pulls the latest `main`, so you always run current team code.

**Before committing a notebook: Edit → Clear all outputs.** Notebook outputs create
enormous diffs and conflicts that cannot be merged.

---

## Step 3b — Working locally (dashboard, API, Milvus)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create your environment file:

```bash
# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
```

Open `.env` and set `SECRET_KEY` to your own value:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**The app will not start without a `.env`.**

Start Milvus (Docker Desktop must be running):

```bash
cd C:\milvus
standalone.bat start
python -m database.seed_data
```

Verify from a **separate terminal** — the backend falls back silently, so the startup
log is not proof:

```bash
python -c "from pymilvus import connections, Collection; connections.connect(host='localhost', port='19530'); c=Collection('cardio_knowledge_base'); c.load(); print(c.num_entities)"
```

Expect `67`. If you get a much smaller number or an error, retrieval results are not
trustworthy — see the Milvus notes in `README.md`.

Run it:

```bash
python main.py
```

Dashboard: http://localhost:8000/dashboard/

---

## Step 4 — Making changes

`main` is shared. Never commit directly to it. Work on your own branch:

```bash
git checkout main
git pull                                  # start from the latest team code
git checkout -b feature/your-change

# ... make your changes ...

git add -A
git commit -m "describe what changed"
git push -u origin feature/your-change
```

Then open a pull request on GitHub so someone else sees the change before it lands.

To pick up other people's work:

```bash
git checkout main
git pull
```

---

## Never commit

- API keys or tokens of any kind (GitHub, Kaggle)
- `.env` — it holds `SECRET_KEY`
- Dataset files (`*.csv`, `.wav`, `.dat`, `.hea`)
- Trained weights (`*.pt`, `*.pkl`)
- Notebook outputs — clear them first

`.gitignore` already blocks all of these. If you find yourself adding an exception,
ask first.

---

## Ground rules for results

This is a research prototype tied to a paper under review, so accuracy of reporting
matters more than usual.

- **Report only measured numbers.** Genuine values live in `data/genuine_results.json`
  and `data/vfdb_cv_metrics.json`. Do not quote figures from any manuscript draft as
  if they were measurements.
- **No synthetic or demo-patient numbers** may be reported as results.
- **The echo (EchoNet) module is not trained.** Its outputs are suppressed, not scored.
  Do not re-enable them.
- **No module currently matches its published benchmark.** That is the honest state.

---

## If something breaks

- **Repo looks corrupted** — delete your local folder and re-clone. GitHub has everything.
- **`ModuleNotFoundError` on `import core...`** — your working directory is not the repo
  root. Re-run the runtime-detect cell, or `cd` to the project root.
- **Milvus "ready" but results look wrong** — check the backend name in the log.
  `db_manager.py` falls back silently through Milvus Lite → ChromaDB → FAISS and still
  logs success.
- **Stuck** — ask in the team chat rather than guessing; several of these failure modes
  are silent.
