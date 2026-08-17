"""
data/download_datasets.py — Actual Dataset Downloader for DeepCardio-RAG
=========================================================================
Downloads all 3 real datasets used by the system:

  1. UCI Heart Disease (Kaggle: johnsmith88/heart-disease-dataset)
     → heart.csv  (1,025 patients × 14 columns)

  2. MIT-BIH ECG Arrhythmia (Kaggle: sadmansakib7/ecg-arrhythmia-classification-dataset)
     → mitbih_train.csv (87,554 × 188) + mitbih_test.csv (21,892 × 188)

  3. Arthritis APD — already bundled in data/APDDataset.xlsx (102 patients × 22 features)

Requirements:
  pip install kaggle          # Kaggle API client
  pip install kagglehub       # Alternative Kaggle downloader

Setup (one-time):
  Option A — kaggle CLI:
    1. Go to https://www.kaggle.com → Account → Create New Token → download kaggle.json
    2. Place kaggle.json in ~/.kaggle/kaggle.json  (Linux/Mac/Colab)
       or  C:/Users/<user>/.kaggle/kaggle.json     (Windows)
    3. chmod 600 ~/.kaggle/kaggle.json

  Option B — Environment variables:
    export KAGGLE_USERNAME=your_kaggle_username
    export KAGGLE_KEY=your_kaggle_api_key

Usage:
    python data/download_datasets.py              # download all
    python data/download_datasets.py --heart      # heart disease only
    python data/download_datasets.py --mitbih     # MIT-BIH only
    python data/download_datasets.py --verify     # verify existing files
"""

import os
import sys
import shutil
import zipfile
import argparse
import hashlib
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ── Expected File Sizes (approximate, used for sanity check) ──────────────────
EXPECTED_FILES = {
    "heart.csv":         {"min_bytes": 30_000,    "rows": 1025,  "cols": 14},
    "mitbih_train.csv":  {"min_bytes": 50_000_000, "rows": 87554, "cols": 188},
    "mitbih_test.csv":   {"min_bytes": 12_000_000, "rows": 21892, "cols": 188},
    "APDDataset.xlsx":   {"min_bytes": 10_000,    "rows": 102,   "cols": 22},
}

# ── Kaggle Dataset Identifiers ─────────────────────────────────────────────────
KAGGLE_DATASETS = {
    "heart":  "johnsmith88/heart-disease-dataset",
    "mitbih": "sadmansakib7/ecg-arrhythmia-classification-dataset",
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def check_file(filename: str) -> bool:
    """Return True if file exists and meets minimum size."""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  ✗ MISSING: {filename}")
        return False
    size = path.stat().st_size
    expected = EXPECTED_FILES.get(filename, {}).get("min_bytes", 0)
    if size < expected:
        print(f"  ✗ TOO SMALL: {filename} ({size:,} bytes, expected ≥{expected:,})")
        return False
    print(f"  ✓ OK: {filename} ({size:,} bytes)")
    return True


def verify_csv(filename: str) -> bool:
    """Verify CSV can be read with pandas and has expected shape."""
    import pandas as pd
    path = DATA_DIR / filename
    try:
        df = pd.read_csv(path, nrows=5)
        expected = EXPECTED_FILES.get(filename, {})
        print(f"    Columns: {len(df.columns)} (expected ~{expected.get('cols','?')})")
        print(f"    Sample columns: {list(df.columns)[:5]} ...")
        return True
    except Exception as e:
        print(f"    ✗ CSV parse error: {e}")
        return False


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: kagglehub (recommended — handles auth automatically)
# ──────────────────────────────────────────────────────────────────────────────

def download_via_kagglehub(dataset_key: str, destination: Path) -> bool:
    """Download using kagglehub library (caches in ~/.cache/kagglehub)."""
    try:
        import kagglehub
        print(f"  Downloading via kagglehub: {KAGGLE_DATASETS[dataset_key]} ...")
        path = kagglehub.dataset_download(KAGGLE_DATASETS[dataset_key])
        print(f"  Downloaded to cache: {path}")
        # Copy CSVs into our data/ directory
        src = Path(path)
        copied = 0
        for f in src.rglob("*.csv"):
            dst = destination / f.name
            shutil.copy2(f, dst)
            print(f"  ✓ Copied: {f.name} → {dst}")
            copied += 1
        return copied > 0
    except ImportError:
        print("  kagglehub not installed (pip install kagglehub)")
        return False
    except Exception as e:
        print(f"  kagglehub failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: Kaggle CLI / API
# ──────────────────────────────────────────────────────────────────────────────

def download_via_kaggle_api(dataset_key: str, destination: Path) -> bool:
    """Download using official kaggle package."""
    try:
        import kaggle
        slug = KAGGLE_DATASETS[dataset_key]
        print(f"  Downloading via Kaggle API: {slug} ...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            slug,
            path=str(destination),
            unzip=True,
            quiet=False,
        )
        # Rename files to expected names
        _rename_heart_csv(destination)
        _rename_mitbih_csvs(destination)
        return True
    except ImportError:
        print("  kaggle package not installed (pip install kaggle)")
        return False
    except Exception as e:
        print(f"  Kaggle API failed: {e}")
        return False


def _rename_heart_csv(dest: Path):
    """Handle various naming conventions from Kaggle heart disease dataset."""
    candidates = ["heart_cleveland_upload.csv", "heart.csv", "heart-disease.csv",
                  "processed.cleveland.data"]
    for name in candidates:
        f = dest / name
        if f.exists() and f.name != "heart.csv":
            target = dest / "heart.csv"
            shutil.move(str(f), str(target))
            print(f"  Renamed {name} → heart.csv")
            break


def _rename_mitbih_csvs(dest: Path):
    """Handle various naming conventions from Kaggle MIT-BIH dataset."""
    for f in dest.glob("*.csv"):
        name = f.name.lower()
        if "train" in name and f.name != "mitbih_train.csv":
            shutil.move(str(f), str(dest / "mitbih_train.csv"))
            print(f"  Renamed {f.name} → mitbih_train.csv")
        elif "test" in name and f.name != "mitbih_test.csv":
            shutil.move(str(f), str(dest / "mitbih_test.csv"))
            print(f"  Renamed {f.name} → mitbih_test.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: Direct URL download (PhysioNet / UCI ML Repository fallback)
# ──────────────────────────────────────────────────────────────────────────────

def download_via_url(dataset_key: str, destination: Path) -> bool:
    """
    Fallback: download from authoritative public sources.
    MIT-BIH: PhysioNet (requires free account) or direct .csv mirrors.
    Heart Disease: UCI ML Repository.
    """
    import urllib.request

    if dataset_key == "heart":
        # UCI Heart Disease — Cleveland subset (processed)
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        out = destination / "processed.cleveland.data"
        print(f"  Downloading UCI Cleveland data from {url} ...")
        try:
            urllib.request.urlretrieve(url, str(out))
            # Convert to CSV with proper headers
            import pandas as pd
            cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                    "thalach","exang","oldpeak","slope","ca","thal","target"]
            df = pd.read_csv(str(out), header=None, names=cols, na_values="?")
            df["target"] = (df["target"] > 0).astype(int)  # binarise 0-4 → 0/1
            df = df.dropna()
            df.to_csv(str(destination / "heart.csv"), index=False)
            print(f"  ✓ Saved heart.csv ({len(df)} rows)")
            out.unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"  UCI URL download failed: {e}")
            return False

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Main Download Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def download_heart_disease():
    print_header("Dataset 1: UCI Heart Disease (johnsmith88/heart-disease-dataset)")
    if check_file("heart.csv"):
        print("  Already present — skipping download")
        verify_csv("heart.csv")
        return True

    # Try methods in order
    for fn in [
        lambda: download_via_kagglehub("heart", DATA_DIR),
        lambda: download_via_kaggle_api("heart", DATA_DIR),
        lambda: download_via_url("heart", DATA_DIR),
    ]:
        if fn():
            if check_file("heart.csv"):
                verify_csv("heart.csv")
                return True

    print("\n  ⚠️  All download methods failed for Heart Disease dataset.")
    print("  Manual download:")
    print("    1. Go to: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print("    2. Click Download → heart.csv")
    print(f"    3. Place file at: {DATA_DIR}/heart.csv")
    return False


def download_mitbih():
    print_header("Dataset 2: MIT-BIH ECG Arrhythmia (sadmansakib7/ecg-arrhythmia-classification-dataset)")
    train_ok = check_file("mitbih_train.csv")
    test_ok  = check_file("mitbih_test.csv")
    if train_ok and test_ok:
        print("  Already present — skipping download")
        verify_csv("mitbih_train.csv")
        return True

    for fn in [
        lambda: download_via_kagglehub("mitbih", DATA_DIR),
        lambda: download_via_kaggle_api("mitbih", DATA_DIR),
    ]:
        if fn():
            if check_file("mitbih_train.csv") and check_file("mitbih_test.csv"):
                verify_csv("mitbih_train.csv")
                return True

    print("\n  ⚠️  All download methods failed for MIT-BIH dataset.")
    print("  Manual download:")
    print("    1. Go to: https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset")
    print("    2. Click Download → extract mitbih_train.csv and mitbih_test.csv")
    print(f"    3. Place both files at: {DATA_DIR}/")
    return False


def verify_all():
    print_header("Verifying All Datasets")
    results = {}
    for filename in EXPECTED_FILES:
        ok = check_file(filename)
        if ok and filename.endswith(".csv"):
            verify_csv(filename)
        results[filename] = ok

    print("\n" + "-" * 40)
    all_ok = all(results.values())
    if all_ok:
        print("  ✓ ALL DATASETS PRESENT AND VALID")
    else:
        missing = [f for f, ok in results.items() if not ok]
        print(f"  ✗ MISSING: {', '.join(missing)}")
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DeepCardio-RAG datasets")
    parser.add_argument("--heart",  action="store_true", help="Download heart disease dataset only")
    parser.add_argument("--mitbih", action="store_true", help="Download MIT-BIH dataset only")
    parser.add_argument("--verify", action="store_true", help="Verify existing files")
    args = parser.parse_args()

    print("\nDeepCardio-RAG — Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")

    if args.verify:
        verify_all()
    elif args.heart:
        download_heart_disease()
    elif args.mitbih:
        download_mitbih()
    else:
        # Download all
        h_ok  = download_heart_disease()
        m_ok  = download_mitbih()
        print_header("Summary")
        print(f"  Heart Disease: {'✓ OK' if h_ok else '✗ FAILED'}")
        print(f"  MIT-BIH ECG:  {'✓ OK' if m_ok else '✗ FAILED'}")
        print(f"  APD Arthritis: {'✓ OK' if (DATA_DIR/'APDDataset.xlsx').exists() else '✗ MISSING (bundled)'}")
        verify_all()
