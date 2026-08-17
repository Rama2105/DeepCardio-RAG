"""
core/ablation.py — Ablation studies for DeepCardio-RAG
======================================================
Answers RR review point #5 ("Add ablation studies for each architectural
component"). Each study removes/varies ONE component and reports the change in
genuine held-out / cross-validated performance, so every design choice is
justified by evidence rather than assertion.

Studies
-------
  1. ablate_arthritis()   — stacking ensemble vs each base learner alone vs a
                            logistic baseline (repeated Stratified K-Fold CV).
                            Runs on CPU. Shows the ensemble's contribution.
  2. ablate_ecg_encoder() — flagship ECGTransformerMoE vs a ResNet-34 CNN
                            baseline on PTB-XL (patient-independent split).
                            Needs the PTB-XL download (Colab GPU).
  3. RAG retriever ablation — semantic Sentence-BERT vs lexical TF-IDF; see
                            core.rag_eval.run_rag_evaluation().
"""
import numpy as np
from typing import Dict


# ---------------------------------------------------------------------------
# 1. Arthritis: ensemble vs components (repeated CV)
# ---------------------------------------------------------------------------
def ablate_arthritis(n_records: int = 3000, prefer_real: bool = True,
                     n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> Dict:
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from core.arthritis_pipeline import (ArthritisDataLoader, ArthritisPredictor,
                                         AdvancedFeatureEngineer)

    loader = ArthritisDataLoader()
    df = loader.load(prefer_real=prefer_real, sample_n=n_records)
    pred = ArthritisPredictor()
    y = pred._create_target(df).values
    base_cols = pred._get_feature_cols(df)
    X_eng = AdvancedFeatureEngineer.transform(df[base_cols].copy())
    feat = [c for c in X_eng.columns if c not in {"arthritis_risk", "RA", "CRP"}
            and X_eng[c].dtype in [np.float64, np.int64, float, int]]
    X = X_eng[feat].values.astype(np.float64)
    n = len(y)
    print(f"[Ablation:Arthritis] {n} records, {len(feat)} features, {y.mean():.1%} positive")

    def smote(Xa, ya, s=seed):
        rng = np.random.RandomState(s); cls, cnt = np.unique(ya.astype(int), return_counts=True)
        mx = cnt.max(); Xo, yo = [Xa], [ya]
        for c, k in zip(cls, cnt):
            if k >= mx: continue
            cX = Xa[ya.astype(int) == c]; syn = []
            for _ in range(mx - k):
                i, j = rng.choice(len(cX), 2, True); a = rng.random()
                syn.append(cX[i]*a + cX[j]*(1-a))
            Xo.append(np.array(syn)); yo.append(np.full(mx-k, c, int))
        p = rng.permutation(sum(len(o) for o in yo))
        return np.vstack(Xo)[p], np.concatenate(yo).astype(int)[p]

    def learners():
        from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                                      ExtraTreesClassifier)
        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV
        return {
            "gbm": GradientBoostingClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, min_samples_leaf=2, random_state=seed),
            "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed),
            "et": ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=seed),
            "svc": CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), cv=3),
        }

    configs = ["logistic_baseline", "gbm_only", "rf_only", "et_only", "svc_only", "full_stack"]
    results = {c: {"acc": [], "auc": [], "f1": []} for c in configs}
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    for tr, te in rskf.split(X, y):
        imp = KNNImputer(n_neighbors=5); sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
        Xte = sc.transform(imp.transform(X[te]))
        Xb, yb = smote(Xtr, y[tr]); yte = y[te]
        # individual + baseline
        probs = {}
        probs["logistic_baseline"] = (LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                                      .fit(Xb, yb).predict_proba(Xte)[:, 1])
        clfs = learners()
        for name, clf in clfs.items():
            probs[f"{name}_only"] = clf.fit(Xb, yb).predict_proba(Xte)[:, 1]
        # full stack (refit fresh learners for OOF meta)
        clfs2 = learners()
        oof = [cross_val_predict(c, Xb, yb, cv=3, method="predict_proba", n_jobs=1)[:, 1]
               for c in clfs2.values()]
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(np.column_stack(oof), yb)
        test_cols = [probs[f"{n_}_only"] for n_ in clfs.keys()]
        probs["full_stack"] = meta.predict_proba(np.column_stack(test_cols))[:, 1]

        for c in configs:
            p = probs[c]; pred = (p > 0.5).astype(int)
            results[c]["acc"].append(accuracy_score(yte, pred))
            results[c]["auc"].append(roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan)
            results[c]["f1"].append(f1_score(yte, pred, average="macro", zero_division=0))

    def stat(v):
        v = np.array([x for x in v if not np.isnan(x)])
        return round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)

    table = {}
    for c in configs:
        table[c] = {m: stat(results[c][m]) for m in ["acc", "auc", "f1"]}
    _print_arth(table)
    return {"n_records": n, "protocol": f"{n_splits}x{n_repeats} Repeated Stratified CV", "results": table}


def _print_arth(table: Dict) -> None:
    print("\n" + "=" * 72)
    print("  ABLATION 1 — Arthritis: ensemble vs components  (mean ± std)")
    print("=" * 72)
    print("  {:20s}{:>16s}{:>16s}{:>16s}".format("config", "accuracy", "auc_roc", "f1_macro"))
    for c, m in table.items():
        print("  {:20s}{:>16s}{:>16s}{:>16s}".format(
            c, f"{m['acc'][0]:.3f}±{m['acc'][1]:.3f}",
            f"{m['auc'][0]:.3f}±{m['auc'][1]:.3f}",
            f"{m['f1'][0]:.3f}±{m['f1'][1]:.3f}"))
    print("=" * 72)


# ---------------------------------------------------------------------------
# 2. ECG encoder: Transformer-MoE vs ResNet-34 (PTB-XL)
# ---------------------------------------------------------------------------
def ablate_ecg_encoder(n_records: int = 3000, epochs: int = 30, synthetic: bool = False) -> Dict:
    from core.train_ptbxl import train_ptbxl
    out = {}
    for enc in ["transformer_moe", "resnet"]:
        print(f"\n[Ablation:ECG] encoder = {enc}")
        m = train_ptbxl(n_records=n_records, epochs=epochs, synthetic=synthetic,
                        encoder_type=enc, save=False)   # save=False: don't clobber flagship
        out[enc] = {"test_accuracy": m["test_accuracy"], "test_f1_macro": m["test_f1_macro"],
                    "test_auc_macro_ovr": m["test_auc_macro_ovr"], "model_params": m["model_params"]}
    print("\n" + "=" * 72)
    print("  ABLATION 2 — ECG encoder architecture (PTB-XL, held-out test)")
    print("=" * 72)
    print("  {:18s}{:>12s}{:>12s}{:>12s}{:>14s}".format("encoder", "acc", "f1_macro", "auc", "params"))
    for enc, m in out.items():
        print("  {:18s}{:>12.4f}{:>12.4f}{:>12.4f}{:>14,d}".format(
            enc, m["test_accuracy"], m["test_f1_macro"], m["test_auc_macro_ovr"], m["model_params"]))
    print("=" * 72)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["arthritis", "ecg", "all"], default="arthritis")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.study in ("arthritis", "all"):
        ablate_arthritis(n_records=a.n)
    if a.study in ("ecg", "all"):
        ablate_ecg_encoder(n_records=a.n, epochs=a.epochs, synthetic=a.synthetic)
