import io

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
lines = io.open(PATH, encoding="utf-8").read().split("\n")

# 1-indexed 1767..1805 -> 0-indexed 1766..1804
start, end = 1766, 1804
assert lines[start].strip() == r"\begin{figure}[!t]", repr(lines[start])
assert lines[end].strip() == r"\end{figure}", repr(lines[end])
assert r"\label{fig:heatmap}" in "\n".join(lines[start:end + 1])

withdrawal = [
    r"% ---------------------------------------------------------------------",
    r"% WITHDRAWN 2026-08-06: per-module confidence heatmap (was fig:heatmap).",
    r"% The figure hardcoded 36 module-confidence values across six patient",
    r"% cases with no generating script, no data file, and no record of those",
    r"% cases anywhere in the codebase. Its Echo row reported six per-patient",
    r"% confidences for a module that has no trained weights at all --- which",
    r"% the caption of fig:barperf, twenty lines above, correctly states is",
    r"% untrained and unmeasured. Restore only if regenerated from real",
    r"% inference on real cases, which requires EchoNet training first.",
    r"% ---------------------------------------------------------------------",
]

out = lines[:start] + withdrawal + lines[end + 1:]
io.open(PATH, "w", encoding="utf-8").write("\n".join(out))
print(f"removed lines {start+1}..{end+1} ({end-start+1} lines), inserted {len(withdrawal)}-line withdrawal note")
