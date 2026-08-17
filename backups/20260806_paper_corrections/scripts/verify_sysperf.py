import json

g = json.load(open(r"G:\My Drive\DeepCardio-RAG\data\genuine_results.json"))
cv = json.load(open(r"G:\My Drive\DeepCardio-RAG\data\vfdb_cv_metrics.json"))

# (label, value claimed in tab:sysperf, actual measured value)
checks = [
    ("Flagship AUC",        0.801, g["flagship"]["test_auc_macro_ovr"]),
    ("Flagship F1",         0.487, g["flagship"]["test_f1_macro"]),
    ("Arrhythmia AUC",      0.928, g["arrhythmia"]["test_auc_macro_ovr"]),
    ("Arrhythmia F1",       0.509, g["arrhythmia"]["test_f1_macro"]),
    ("CardioFusion AUC",    0.838, g["cardiofusion"]["test_auc_macro_ovr"]),
    ("CardioFusion F1",     0.540, g["cardiofusion"]["test_f1_macro"]),
    ("Arthritis CV AUC",    0.792, g["arthritis"]["cv_auc_roc"]["mean"]),
    ("Arthritis test AUC",  0.813, g["arthritis"]["holdout_auc_roc"]),
    ("PCG PvA-AUC",         0.761, g["pcg"]["test_present_vs_absent_auc"]),
    ("PCG F1",              0.535, g["pcg"]["test_f1_macro"]),
    ("VFDB CV AUC",         0.919, cv["aggregate"]["binary_auc"]["mean"]),
    ("VFDB CI low",         0.847, cv["aggregate"]["binary_auc"]["ci95_low"]),
    ("VFDB CI high",        0.991, cv["aggregate"]["binary_auc"]["ci95_high"]),
    ("VFDB F1 dangerous",   0.718, cv["aggregate"]["binary_f1_dangerous"]["mean"]),
]

print("tab:sysperf cross-check against committed results")
bad = 0
for label, claimed, actual in checks:
    match = abs(claimed - round(actual, 3)) < 0.0015
    bad += (not match)
    print(f"  {'OK ' if match else 'DIFF'} {label:22} paper={claimed:<7} measured={actual}")
print(f"\nmismatches: {bad} of {len(checks)}")
