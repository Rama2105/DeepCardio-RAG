"""
core/cv_vfdb.py — Recording-level cross-validation for the VFDB module
======================================================================
Why this exists
---------------
`core/train_vfdb.py` trains on a SINGLE 15/3/4 recording split of the 22-recording
VFDB. That produced a genuine but fragile number on 2026-07-30:

    val F1(dang) 0.894 / val AUC 0.984   vs   test F1(dang) 0.515 / test AUC 0.805

A val-to-test gap that wide over 4 test recordings is not evidence of leakage —
it is evidence that ONE split of 22 recordings cannot support a point estimate.
Quoting 0.8046 as "the" VFDB AUC invites exactly the reviewer objection the rest
of this audit exists to answer. This module replaces the point estimate with a
distribution: repeated recording-level K-fold, every recording serving as test
data exactly once per repeat, reported as mean with a 95% interval and with every
per-fold score kept visible.

Design decisions worth knowing
------------------------------
- **Folds are whole RECORDINGS.** A recording never appears in train and test of
  the same fold. Windows from one patient are highly correlated, so a window-level
  split would leak and inflate every score.
- **Folds are stratified by each recording's dangerous-window fraction.** With 22
  recordings and rare malignant episodes, unstratified folds can hand you a test
  fold with zero dangerous windows, where F1(dangerous) is 0 and AUC is undefined.
  Recordings are sorted by dangerous fraction and snake-dealt across folds so each
  fold gets a comparable mix.
- **Validation comes out of the TRAIN recordings, never the test fold.** Early
  stopping selects on val F1; the test fold is scored exactly once, after
  training. This is the same discipline applied to CardioFusion on 2026-07-20
  after test-set selection was found there.
- **Segments are extracted ONCE and cached**, then re-indexed per fold. Re-reading
  the .dat files for every fold of every repeat would dominate runtime.
- **A fold whose test set is single-class reports AUC as NaN and is counted**,
  not silently dropped into a mean. Same principle as the single-class guard in
  train_vfdb: refuse to let a degenerate split masquerade as a score.

Honest limits of the interval this reports
------------------------------------------
The 95% interval is a t-interval over fold scores. Folds share training data, so
the scores are NOT independent and the interval is optimistic — it understates
true uncertainty. It is still far more informative than a single split, but it is
a spread-of-folds interval, not a confidence interval over patients. With 22
recordings that limitation is intrinsic to the dataset, not fixable by resampling.
Report it as such; do not upgrade the wording to "95% CI over patients".

Usage (GPU or CPU — dataset is small)
-------------------------------------
    from core.cv_vfdb import cross_validate_vfdb
    cv = cross_validate_vfdb(n_folds=5, n_repeats=2, epochs=30)

    # quick mechanics check (fewer epochs, one repeat)
    cv = cross_validate_vfdb(n_folds=5, n_repeats=1, epochs=3)
"""
import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.vfdb_loader import (
    VFDBDataset, VentricularArrhythmiaDetector, DEFAULT_DATASET_DIR,
)
from core.train_vfdb import (
    _segments_from_record, _evaluate, RHYTHM_CLASSES, IN_CHANNELS,
    SEGMENT_SECONDS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CV_METRICS_SAVE_PATH = os.path.join(DATA_DIR, "vfdb_cv_metrics.json")

# Metrics aggregated across folds. Keep binary_auc first — it is the headline.
_METRIC_KEYS = [
    "binary_auc",
    "binary_f1_dangerous",
    "binary_accuracy",
    "rhythm_f1_macro",
]


def _load_all_segments(ds: VFDBDataset, seg_seconds: int):
    """
    Extract windows for every recording ONCE.

    Returns (per_record, dangerous_fracs) where per_record[i] is
    (X [n,2,L], y_rhythm [n], y_binary [n]) for recording i, and dangerous_fracs[i]
    is that recording's fraction of dangerous windows (used to stratify folds).
    Recordings that yield no windows are dropped and reported.
    """
    per_record, fracs, kept_ids, dropped = [], [], [], []
    for i in range(len(ds)):
        x, yr, yb = _segments_from_record(ds, i, seg_seconds)
        rec_id = ds.records[i]["record_id"]
        if len(x) == 0:
            dropped.append(rec_id)
            continue
        per_record.append((x, yr, yb))
        fracs.append(float(yb.mean()))
        kept_ids.append(rec_id)
    if dropped:
        print(f"[VFDB-CV] {len(dropped)} recording(s) yielded no windows and were "
              f"dropped: {dropped}")
    return per_record, np.array(fracs), kept_ids


def _stratified_recording_folds(dangerous_fracs: np.ndarray, n_folds: int,
                                seed: int):
    """
    Assign whole recordings to folds, balanced on dangerous-window fraction.

    Sort recordings by dangerous fraction, cut the sorted order into consecutive
    strata of n_folds recordings, and randomly assign one member of each stratum
    to each fold. Every fold therefore draws one recording from the top band, one
    from the next, and so on. A plain shuffle can put every event-rich recording
    in one fold and leave another single-class — with 22 recordings that is not a
    remote possibility.

    The within-stratum shuffle is what makes REPEATS meaningful: a deterministic
    deal (e.g. snake-ordering the sorted list) returns the identical partition for
    every seed, so repeated CV would only re-roll the model init and the interval
    would measure init noise rather than partition noise.
    Returns a list of n_folds arrays of recording indices (the test fold members).
    """
    rng = np.random.RandomState(seed)
    order = np.argsort(-np.asarray(dangerous_fracs, dtype=float))

    folds = [[] for _ in range(n_folds)]
    for start in range(0, len(order), n_folds):
        stratum = order[start:start + n_folds]
        # Randomly pair this stratum's members with folds. A short final stratum
        # goes to a random subset of folds, which is what keeps fold sizes within 1.
        targets = rng.permutation(n_folds)[:len(stratum)]
        for rec_idx, f in zip(stratum, targets):
            folds[int(f)].append(int(rec_idx))
    return [np.array(sorted(f), dtype=int) for f in folds]


def _gather(per_record, idxs):
    """Concatenate cached windows for the given recording indices."""
    Xs = [per_record[j][0] for j in idxs]
    Yr = [per_record[j][1] for j in idxs]
    Yb = [per_record[j][2] for j in idxs]
    return np.concatenate(Xs), np.concatenate(Yr), np.concatenate(Yb)


def _fit_one_fold(Xtr, ytr_r, ytr_b, Xva, yva_r, yva_b, Xte, yte_r, yte_b,
                  epochs, batch_size, lr, seed, device, patience=8, verbose=False):
    """
    Train one fold and score its held-out recordings exactly once.

    Mirrors train_vfdb's loop deliberately (class-weighted CE on both heads,
    early stop on val F1(dangerous), cosine LR) so CV numbers are comparable to
    the single-split number rather than measuring a different training recipe.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VentricularArrhythmiaDetector(in_channels=IN_CHANNELS).to(device)

    b_counts = np.bincount(ytr_b, minlength=2).astype(float)
    r_counts = np.bincount(ytr_r, minlength=len(RHYTHM_CLASSES)).astype(float)
    b_w = torch.tensor(b_counts.sum() / (2 * np.maximum(b_counts, 1)),
                       dtype=torch.float32).to(device)
    r_w = torch.tensor(r_counts.sum() / (len(RHYTHM_CLASSES) * np.maximum(r_counts, 1)),
                       dtype=torch.float32).to(device)
    bin_crit = nn.CrossEntropyLoss(weight=b_w)
    rhy_crit = nn.CrossEntropyLoss(weight=r_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr_b, dtype=torch.long),
                      torch.tensor(ytr_r, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)

    best_f1, best_state, bad, stopped_epoch = -1.0, None, 0, epochs
    for ep in range(epochs):
        model.train()
        run = 0.0
        for xb, ybb, ybr in loader:
            xb, ybb, ybr = xb.to(device), ybb.to(device), ybr.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = bin_crit(out["binary_logits"], ybb) + rhy_crit(out["rhythm_logits"], ybr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run += loss.item()
        scheduler.step()

        vm = _evaluate(model, Xva, yva_b, yva_r, device, batch_size)
        if verbose:
            print(f"    epoch {ep+1:02d}/{epochs} | loss {run/len(loader):.4f} | "
                  f"val_F1(dang) {vm['binary_f1_dangerous']:.4f}")
        if vm["binary_f1_dangerous"] > best_f1:
            best_f1 = vm["binary_f1_dangerous"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                stopped_epoch = ep + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tm = _evaluate(model, Xte, yte_b, yte_r, device, batch_size)
    tm["best_val_f1_dangerous"] = float(best_f1)
    tm["stopped_epoch"] = int(stopped_epoch)
    return tm


def _mean_ci(values, confidence=0.95):
    """
    Mean and t-interval over fold scores, ignoring NaN (undefined-AUC folds).

    Returns (mean, lo, hi, n_used). n<2 gives a point estimate with NaN bounds
    rather than a fabricated interval.
    """
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(v.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), n
    sem = float(v.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except Exception:
        crit = 1.96  # normal approximation if scipy is unavailable
    return mean, mean - crit * sem, mean + crit * sem, n


def cross_validate_vfdb(n_folds: int = 5, n_repeats: int = 2, epochs: int = 30,
                        batch_size: int = 64, lr: float = 3e-4,
                        seg_seconds: int = SEGMENT_SECONDS, seed: int = 42,
                        val_fraction: float = 0.15,
                        dataset_dir: str = DEFAULT_DATASET_DIR,
                        save: bool = True, verbose: bool = False) -> dict:
    """
    Repeated recording-level K-fold CV of the VFDB detector.

    Every recording is a test recording exactly once per repeat. Returns a dict
    with per-fold rows and aggregate mean/95% interval, also written to
    data/vfdb_cv_metrics.json. Nothing is saved as model weights — this measures
    the training procedure, it does not produce a shippable checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    ds = VFDBDataset(dataset_dir=dataset_dir)
    if len(ds) == 0:
        raise RuntimeError(
            "VFDB not found. Place PhysioNet VFDB (.dat/.hea/.atr) under data/vfdb/.")

    print(f"[VFDB-CV] extracting {seg_seconds}s windows from {len(ds)} recordings "
          f"(once; cached across all folds)...")
    per_record, fracs, rec_ids = _load_all_segments(ds, seg_seconds)
    n_rec = len(per_record)
    if n_rec < n_folds:
        raise RuntimeError(
            f"Only {n_rec} usable recordings for {n_folds} folds. Lower n_folds.")

    total_windows = int(sum(len(x) for x, _, _ in per_record))
    overall_frac = float(np.average(fracs, weights=[len(x) for x, _, _ in per_record]))
    print(f"[VFDB-CV] {n_rec} recordings | {total_windows} windows | "
          f"dangerous fraction {overall_frac:.3f}")
    print(f"[VFDB-CV] per-recording dangerous fraction: "
          f"min {fracs.min():.3f} / median {np.median(fracs):.3f} / max {fracs.max():.3f}")

    rows = []
    for rep in range(n_repeats):
        folds = _stratified_recording_folds(fracs, n_folds, seed=seed + rep)
        for k, te_idx in enumerate(folds):
            rest = np.array([i for i in range(n_rec) if i not in set(te_idx.tolist())])
            # Validation recordings come out of the training pool, never the test fold.
            rng = np.random.RandomState(seed + 1000 * rep + k)
            rest = rest[rng.permutation(len(rest))]
            n_va = max(1, int(round(val_fraction * len(rest))))
            va_idx, tr_idx = rest[:n_va], rest[n_va:]

            Xtr, ytr_r, ytr_b = _gather(per_record, tr_idx)
            Xva, yva_r, yva_b = _gather(per_record, va_idx)
            Xte, yte_r, yte_b = _gather(per_record, te_idx)

            te_single_class = len(np.unique(yte_b)) < 2
            tr_single_class = len(np.unique(ytr_b)) < 2
            if tr_single_class:
                # Same reasoning as train_vfdb's guard: a single-class TRAIN split
                # means annotations did not parse, which is a bug, not a result.
                raise RuntimeError(
                    f"[repeat {rep+1} fold {k+1}] training split is single-class "
                    f"(dangerous fraction {ytr_b.mean():.3f}) — annotations did not "
                    "parse. Install/upgrade `wfdb` and re-run.")

            tag = f"rep{rep+1}/fold{k+1}"
            print(f"[VFDB-CV] {tag} | train {len(tr_idx)} rec / {len(Xtr)} win | "
                  f"val {len(va_idx)} rec | test {len(te_idx)} rec / {len(Xte)} win | "
                  f"test dangerous frac {yte_b.mean():.3f}"
                  + ("  ⚠ SINGLE-CLASS TEST FOLD" if te_single_class else ""))

            tm = _fit_one_fold(Xtr, ytr_r, ytr_b, Xva, yva_r, yva_b,
                               Xte, yte_r, yte_b,
                               epochs=epochs, batch_size=batch_size, lr=lr,
                               seed=seed + 100 * rep + k, device=device,
                               verbose=verbose)

            row = {
                "repeat": rep + 1,
                "fold": k + 1,
                "test_recordings": [rec_ids[i] for i in te_idx],
                "n_train_recordings": int(len(tr_idx)),
                "n_val_recordings": int(len(va_idx)),
                "n_test_windows": int(len(Xte)),
                "test_dangerous_fraction": round(float(yte_b.mean()), 4),
                "single_class_test_fold": bool(te_single_class),
                "stopped_epoch": tm["stopped_epoch"],
                "best_val_f1_dangerous": round(tm["best_val_f1_dangerous"], 4),
            }
            for m in _METRIC_KEYS:
                row[m] = None if np.isnan(tm[m]) else round(float(tm[m]), 4)
            rows.append(row)
            print(f"[VFDB-CV] {tag} -> AUC {row['binary_auc']} | "
                  f"F1(dang) {row['binary_f1_dangerous']} | "
                  f"rhythm macro-F1 {row['rhythm_f1_macro']}")

    # ---- Aggregate ----
    aggregate = {}
    for m in _METRIC_KEYS:
        mean, lo, hi, n_used = _mean_ci([r[m] for r in rows])
        aggregate[m] = {
            "mean": None if np.isnan(mean) else round(mean, 4),
            "ci95_low": None if np.isnan(lo) else round(lo, 4),
            "ci95_high": None if np.isnan(hi) else round(hi, 4),
            "n_folds_used": n_used,
        }

    n_degenerate = sum(1 for r in rows if r["single_class_test_fold"])
    result = {
        "dataset": (f"MIT-BIH VFDB (PhysioNet), {n_rec} recordings, "
                    f"{seg_seconds}s windows, {total_windows} windows total"),
        "protocol": (f"{n_repeats}x repeated {n_folds}-fold cross-validation, folds are "
                     "WHOLE RECORDINGS stratified by dangerous-window fraction; "
                     "validation recordings drawn from the training pool only"),
        "evaluation": "each test fold scored exactly once, after early stopping on val F1",
        "n_recordings": n_rec,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "overall_dangerous_fraction": round(overall_frac, 4),
        "epochs_max": epochs,
        "device": str(device),
        "runtime_s": round(time.time() - t0, 1),
        "aggregate": aggregate,
        "folds": rows,
        "n_single_class_test_folds": n_degenerate,
        "interval_caveat": (
            "The 95% interval is a t-interval over fold scores. Folds share training "
            "recordings, so fold scores are not independent and this interval is "
            "OPTIMISTIC — it understates true uncertainty. It describes spread across "
            "folds, not a confidence interval over patients. With 22 recordings this "
            "is a limit of the dataset, not of the resampling scheme."),
        "supersedes": (
            "The single 15/3/4 split reported on 2026-07-30 (test F1(dang) 0.5147 / "
            "AUC 0.8046, against val F1 0.894 / AUC 0.984). That point estimate should "
            "be quoted only alongside this distribution."),
    }

    if save:
        with open(CV_METRICS_SAVE_PATH, "w") as f:
            json.dump(result, f, indent=2)

    # ---- Report ----
    print("\n[VFDB-CV] === RECORDING-LEVEL CROSS-VALIDATION ===")
    print(f"  {n_repeats}x {n_folds}-fold over {n_rec} recordings "
          f"({len(rows)} fits, {result['runtime_s']}s)")
    print(f"  {'metric':<24s} {'mean':>8s}   {'95% interval':>18s}   folds")
    for m in _METRIC_KEYS:
        a = aggregate[m]
        if a["mean"] is None:
            print(f"  {m:<24s} {'n/a':>8s}")
            continue
        ci = ("[    n/a    ]" if a["ci95_low"] is None
              else f"[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]")
        print(f"  {m:<24s} {a['mean']:>8.4f}   {ci:>18s}   {a['n_folds_used']}")
    if n_degenerate:
        print(f"  ⚠ {n_degenerate} fold(s) had a single-class test set — AUC undefined "
              f"there and excluded from the mean.")
    print(f"  NOTE: {result['interval_caveat']}")
    if save:
        print(f"  saved -> {CV_METRICS_SAVE_PATH}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cross_validate_vfdb(n_folds=a.folds, n_repeats=a.repeats, epochs=a.epochs,
                        verbose=a.verbose)
