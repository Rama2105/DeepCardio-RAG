import io

s = io.open(r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex", encoding="utf-8").read()
ok = True
for env in ("figure", "tikzpicture", "axis", "table", "tabular", "document"):
    b = s.count(r"\begin{%s}" % env)
    e = s.count(r"\end{%s}" % env)
    status = "OK" if b == e else "MISMATCH"
    ok &= (b == e)
    print(f"  {env:14} begin={b:3} end={e:3} {status}")
print("\nall environments balanced:", ok)
print("fig:heatmap label present:", r"\label{fig:heatmap}" in s)
print("fig:heatmap \\ref present:", r"\ref{fig:heatmap}" in s)
