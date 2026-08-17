"""
colab_train_genuine.py — Genuine, reproducible training of every REAL DeepCardio
module on ~3000 records, for execution in Google Colab (GPU).
================================================================================
Addresses the RR review: every number printed here is produced by training on a
REAL dataset and evaluating on a HELD-OUT split — no random-weight inference, no
benchmark values copied from literature, no train-set evaluation.

Modules trained here
--------------------
  1. Flagship ECG encoder  — PTB-XL 12-lead (patient-independent folds)   [GPU]
  2. ECG Arrhythmia CNN    — MIT-BIH beats (pre-split train/test)          [GPU/CPU]
  3. Arthritis ensemble    — real Kaggle dataset, honest repeated-CV       [CPU]

Run in Colab
------------
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/drive/MyDrive/DeepCardio-RAG
    !pip -q install wfdb kagglehub scikit-learn transformers torch
    from colab_train_genuine import run_all
    results = run_all(n_records=3000, ecg_epochs=30, arr_epochs=25)

Kaggle data (PTB-XL, NHANES/BRFSS) needs credentials in Colab:
    import os; os.environ['KAGGLE_USERNAME']='...'; os.environ['KAGGLE_KEY']='...'
(or upload kaggle.json). MIT-BIH CSVs already live in data/ on your Drive.
"""
import os
import json
import traceback
import numpy as np


# ---------------------------------------------------------------------------
# 1. Flagship ECG encoder on PTB-XL (real 12-lead)
# ---------------------------------------------------------------------------
def train_flagship_ecg(n_records: int = 3000, epochs: int = 30, synthetic: bool = False) -> dict:
    from core.train_ptbxl import train_ptbxl
    return train_ptbxl(n_records=n_records, epochs=epochs, synthetic=synthetic)


# ---------------------------------------------------------------------------
# 2. ECG Arrhythmia 1D-CNN on MIT-BIH (real beats already on Drive)
# ---------------------------------------------------------------------------
def train_ecg_arrhythmia(n_records: int = 3000, epochs: int = 25, seed: int = 42) -> dict:
    from core.ecg_arrhythmia_pipeline import ECGArrhythmiaDataLoader, ECGArrhythmiaPredictor
    loader = ECGArrhythmiaDataLoader()
    train_df, test_df = loader.load()

    # Class-stratified subsample of the training set to n_records (honest cap).
    label_col = train_df.columns[-1]
    rng = np.random.RandomState(seed)
    if n_records and len(train_df) > n_records:
        parts = []
        for _, grp in train_df.groupby(label_col):
            k = max(1, int(round(n_records * len(grp) / len(train_df))))
            k = min(k, len(grp))
            parts.append(grp.iloc[rng.choice(len(grp), k, replace=False)])
        train_df = (__import__("pandas").concat(parts)
                    .sample(frac=1.0, random_state=seed).reset_index(drop=True))
        print(f"[ECG-Arrhythmia] Stratified-subsampled train to {len(train_df):,} beats")

    pred = ECGArrhythmiaPredictor()
    m = pred.train(train_df, test_df=test_df, num_epochs=epochs)
    return {
        "dataset": "MIT-BIH Arrhythmia (PhysioNet/Kaggle), 5 AAMI classes",
        "n_train": m["train_samples"], "n_test": m["eval_samples"],
        "eval_set": m["eval_set"],
        "test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
        "test_auc_macro_ovr": m["auc_roc_ovr"],
        "evaluation": "held-out MIT-BIH test split",
    }


# ---------------------------------------------------------------------------
# 3. Arthritis stacking ensemble on a real Kaggle dataset (honest repeated-CV)
# ---------------------------------------------------------------------------
def train_arthritis(n_records: int = 3000, prefer_real: bool = True) -> dict:
    from core.arthritis_pipeline import ArthritisDataLoader, ArthritisPredictor
    loader = ArthritisDataLoader()
    df = loader.load(prefer_real=prefer_real, sample_n=n_records)
    info = loader.dataset_info or {}
    pred = ArthritisPredictor()
    m = pred.train(df)
    cv = m.get("cross_validation") or {}
    return {
        "dataset": info.get("name", "unknown"),
        "is_fallback_APD": info.get("is_fallback", False),
        "n_records": int(len(df)),
        "cv_protocol": cv.get("protocol"),
        "cv_accuracy": cv.get("accuracy"),
        "cv_auc_roc": cv.get("auc_roc"),
        "cv_f1_macro": cv.get("f1_macro"),
        "cv_brier": cv.get("brier"),
        "holdout_accuracy": m.get("accuracy"),
        "holdout_auc_roc": m.get("auc_roc"),
        "holdout_f1": m.get("f1"),
        "total_features": m.get("total_features"),
        "top_features": m.get("top_features"),
        "evaluation": "repeated-CV mean±95% CI (headline) + 20% held-out split",
    }


# ---------------------------------------------------------------------------
# 4. CardioFusion hybrid model (ECG-signal path on real MIT-BIH; learned weights)
# ---------------------------------------------------------------------------
def train_cardiofusion_module(n_records: int = 3000, epochs: int = 20) -> dict:
    from core.train_cardiofusion import train_cardiofusion
    m = train_cardiofusion(n_records=n_records, epochs=epochs)
    return {
        "dataset": m["dataset"], "modality_trained": m["modality_trained"],
        "n_train": m["n_train"], "n_test": m["n_test"],
        "test_accuracy": m["test_accuracy"], "test_f1_macro": m["test_f1_macro"],
        "test_auc_macro_ovr": m["test_auc_macro_ovr"],
        "loss_weighting": m["loss_weighting"],
        "learned_task_weights": m["learned_task_weights"],
        "multimodal_note": m["multimodal_note"],
        "evaluation": m["evaluation"],
    }


# ---------------------------------------------------------------------------
# 5. Echocardiography LVEF (EchoNet-Dynamic 3D-CNN) — was UNtrained (M1/#1)
# ---------------------------------------------------------------------------
def train_echo_module(n_train: int = 3000, n_eval: int = 1000, epochs: int = 25) -> dict:
    from core.train_echonet import train_echonet
    return train_echonet(n_train=n_train, n_eval=n_eval, epochs=epochs)


# ---------------------------------------------------------------------------
# 6. Heart-sound murmur (CirCor DigiScope) — was UNtrained (synthetic-only)
# ---------------------------------------------------------------------------
def train_pcg_module(n_max: int = 3000, epochs: int = 25) -> dict:
    from core.train_circor import train_circor
    return train_circor(n_max=n_max, epochs=epochs)


# ---------------------------------------------------------------------------
# 7. Ventricular arrhythmia (VFDB) — was RANDOM WEIGHTS (the M1 code-blue failure)
# ---------------------------------------------------------------------------
def train_vfdb_module(epochs: int = 30) -> dict:
    from core.train_vfdb import train_vfdb
    return train_vfdb(epochs=epochs)


# ---------------------------------------------------------------------------
# 8. Heart-disease (UCI Cleveland) — was UNtrained; dedup to 303 + baselines (M5)
# ---------------------------------------------------------------------------
def train_heartdisease_module(n_repeats: int = 5, n_splits: int = 3, epochs: int = 80) -> dict:
    from core.train_heartdisease import train_heartdisease
    return train_heartdisease(n_repeats=n_repeats, n_splits=n_splits, epochs=epochs)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_all(n_records: int = 3000, ecg_epochs: int = 30, arr_epochs: int = 25,
            fusion_epochs: int = 20, echo_epochs: int = 25, pcg_epochs: int = 25,
            vfdb_epochs: int = 30, hd_epochs: int = 80,
            synthetic_ptbxl: bool = False, skip: tuple = ()) -> dict:
    """
    Train every genuine module and print one consolidated results table.
    `skip` may contain any of: 'flagship', 'arrhythmia', 'arthritis', 'cardiofusion',
    'echo', 'pcg', 'vfdb', 'heartdisease'.
    Each module is guarded so one failure (e.g. missing dataset) does not abort
    the others. The echo/pcg/vfdb/heartdisease steps train the four modules the
    peer review flagged as never independently trained (echo/pcg/heartdisease) or
    running on random weights (vfdb).
    """
    results = {}
    steps = [
        ("flagship",   "Flagship ECG encoder (PTB-XL 12-lead)",
         lambda: train_flagship_ecg(n_records, ecg_epochs, synthetic_ptbxl)),
        ("arrhythmia", "ECG Arrhythmia CNN (MIT-BIH)",
         lambda: train_ecg_arrhythmia(n_records, arr_epochs)),
        ("arthritis",  "Arthritis ensemble (Kaggle real)",
         lambda: train_arthritis(n_records)),
        ("cardiofusion", "CardioFusion hybrid (ECG-signal path, MIT-BIH, learned weights)",
         lambda: train_cardiofusion_module(n_records, fusion_epochs)),
        ("echo",       "Echocardiography LVEF (EchoNet-Dynamic 3D-CNN)",
         lambda: train_echo_module(n_records, 1000, echo_epochs)),
        ("pcg",        "Heart-sound murmur (CirCor DigiScope)",
         lambda: train_pcg_module(n_records, pcg_epochs)),
        ("vfdb",       "Ventricular arrhythmia (VFDB — replaces random weights)",
         lambda: train_vfdb_module(vfdb_epochs)),
        ("heartdisease", "Heart disease (UCI Cleveland, deduplicated + baselines)",
         lambda: train_heartdisease_module(5, 3, hd_epochs)),
    ]
    for key, title, fn in steps:
        if key in skip:
            continue
        print("\n" + "=" * 72 + f"\n  TRAINING: {title}\n" + "=" * 72)
        try:
            results[key] = fn()
        except Exception as e:
            results[key] = {"error": str(e)}
            print(f"[{key}] FAILED: {e}")
            traceback.print_exc()

    _print_summary(results)
    out = os.path.join(os.path.dirname(__file__), "data", "genuine_results.json")
    try:
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved consolidated genuine results -> {out}")
    except Exception:
        pass
    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 72)
    print("  GENUINE HELD-OUT RESULTS  (real data, no fabricated / literature values)")
    print("=" * 72)
    f = results.get("flagship", {})
    if "error" not in f:
        print(f"  Flagship ECG (PTB-XL) : acc={f.get('test_accuracy')}  "
              f"macro-F1={f.get('test_f1_macro')}  AUC={f.get('test_auc_macro_ovr')}  "
              f"Brier={f.get('test_brier')}  ECE={f.get('test_ece')}"
              + ("   [SYNTHETIC — not valid]" if f.get("synthetic") else ""))
    else:
        print(f"  Flagship ECG (PTB-XL) : SKIPPED/FAILED — {f['error']}")
    a = results.get("arrhythmia", {})
    if "error" not in a:
        print(f"  ECG Arrhythmia (MITBIH): acc={a.get('test_accuracy')}  "
              f"macro-F1={a.get('test_f1_macro')}  AUC={a.get('test_auc_macro_ovr')}")
    else:
        print(f"  ECG Arrhythmia (MITBIH): SKIPPED/FAILED — {a['error']}")
    t = results.get("arthritis", {})
    if "error" not in t:
        cv_a = (t.get("cv_accuracy") or {}); cv_u = (t.get("cv_auc_roc") or {})
        tag = "  [APD fallback — set Kaggle creds for 3000 real records]" if t.get("is_fallback_APD") else ""
        print(f"  Arthritis ensemble     : CV acc={cv_a.get('mean')} CI{cv_a.get('ci95')}  "
              f"CV AUC={cv_u.get('mean')} CI{cv_u.get('ci95')}{tag}")
    else:
        print(f"  Arthritis ensemble     : SKIPPED/FAILED — {t['error']}")
    c = results.get("cardiofusion", {})
    if "error" not in c and c:
        print(f"  CardioFusion (ECG-sig) : acc={c.get('test_accuracy')}  "
              f"macro-F1={c.get('test_f1_macro')}  AUC={c.get('test_auc_macro_ovr')}  "
              f"[learned task weights; single-modality — see multimodal_note]")
    elif "error" in c:
        print(f"  CardioFusion (ECG-sig) : SKIPPED/FAILED — {c['error']}")

    e = results.get("echo", {})
    if "error" not in e and e:
        print(f"  Echo LVEF (EchoNet)    : MAE={e.get('test_mae')}  R2={e.get('test_r2')}  "
              f"cat-F1={e.get('test_cat_f1_macro')}  cat-AUC={e.get('test_cat_auc_macro_ovr')}"
              + ("   [SYNTHETIC — not valid]" if e.get("synthetic") else ""))
    elif "error" in e:
        print(f"  Echo LVEF (EchoNet)    : SKIPPED/FAILED — {e['error']}")
    p = results.get("pcg", {})
    if "error" not in p and p:
        print(f"  PCG murmur (CirCor)    : acc={p.get('test_accuracy')}  "
              f"macro-F1={p.get('test_f1_macro')}  PvA-AUC={p.get('test_present_vs_absent_auc')}")
    elif "error" in p:
        print(f"  PCG murmur (CirCor)    : SKIPPED/FAILED — {p['error']}")
    v = results.get("vfdb", {})
    if "error" not in v and v:
        print(f"  VFDB VF/VT (was random): bin-acc={v.get('test_binary_accuracy')}  "
              f"F1(dang)={v.get('test_binary_f1_dangerous')}  AUC={v.get('test_binary_auc')}")
    elif "error" in v:
        print(f"  VFDB VF/VT             : SKIPPED/FAILED — {v['error']}")
    h = results.get("heartdisease", {})
    if "error" not in h and h:
        hb = ((h.get("cv") or {}).get("bert_moe") or {})
        hl = ((h.get("cv") or {}).get("logistic") or {})
        print(f"  Heart disease (dedup {h.get('n_unique_records')}): "
              f"BERT-MoE CV AUC={(hb.get('auc') or {}).get('mean')}  "
              f"vs logistic CV AUC={(hl.get('auc') or {}).get('mean')}  [R8 baseline check]")
    elif "error" in h:
        print(f"  Heart disease          : SKIPPED/FAILED — {h['error']}")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--ecg-epochs", type=int, default=30)
    ap.add_argument("--arr-epochs", type=int, default=25)
    ap.add_argument("--fusion-epochs", type=int, default=20)
    ap.add_argument("--echo-epochs", type=int, default=25)
    ap.add_argument("--pcg-epochs", type=int, default=25)
    ap.add_argument("--vfdb-epochs", type=int, default=30)
    ap.add_argument("--hd-epochs", type=int, default=80)
    ap.add_argument("--synthetic-ptbxl", action="store_true",
                    help="smoke-test the flagship trainer without the real download")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    run_all(a.n, a.ecg_epochs, a.arr_epochs, a.fusion_epochs,
            a.echo_epochs, a.pcg_epochs, a.vfdb_epochs, a.hd_epochs,
            a.synthetic_ptbxl, tuple(a.skip))
