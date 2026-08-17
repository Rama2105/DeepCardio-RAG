# Backups — 2026-08-06 correction session

Snapshots of `DeepCardio_CompIntel_Submit.tex` taken *before* each batch of
corrections, plus the scripts that made them. Restore by copying a `.bak-*` file
over the live `.tex` in the project root.

## Restore points, oldest first

| File | State it captures |
|---|---|
| `…tex.bak-heatmap` | Before removing `fig:heatmap` (36 fabricated per-patient confidence values, incl. an Echo row for an untrained module) |
| `…tex.bak-8fixes` | Before the 8-defect number sweep (UCI prevalence swapped, VFDB 142→239, withdrawn demo alert still tabulated, RAG 16→67, phantom SVC, CirCor full-vs-subset, Echo counted as delivered capability, "Risk" column header) |
| `…tex.bak-round2` | Before removing two surviving demo-patient confidence blocks (ECG-image 28.8% from an untrained classifier; PCG "Murmur Detected" at 44.4%) and fixing the duplicated `\label{eq:cf}` |
| `…tex.bak-citations` | Before citing the 20 uncited references (first batch) |
| `…tex.bak-citations2` | Before the second citation batch, which also fixed the conclusion's uncomputed ECG intervals (HR 72 / PR 160 / QRS 90 / QTc 410) and two more stale "16-document corpus" claims |

`index.html.bak-screen3` is the dashboard's Patient Predictor form before it was
rewired from 22 APD blood-test fields to the 9 NHANES inputs the model actually
consumes.

## scripts/

The edit scripts are the audit trail: each replacement asserts its expected
occurrence count, so what changed and how many times is recoverable from the
source. `audit_tex.py` re-runs the citation / cross-reference / percent-claim
audit; `verify_sysperf.py` re-checks every headline metric in `tab:sysperf`
against `data/genuine_results.json` and `data/vfdb_cv_metrics.json`.

Final state after all batches: 60 cited / 60 defined references, 28 pages,
`pdflatex` exit 0, zero undefined and zero multiply-defined labels.
