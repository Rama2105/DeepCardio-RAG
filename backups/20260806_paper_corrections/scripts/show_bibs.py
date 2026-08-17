import io
import re

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
s = io.open(PATH, encoding="utf-8").read()

uncited = ['acharya2017', 'baloglu2019', 'devlin2019', 'douze2024', 'esteva2019',
           'goldberger2000', 'goodfellow2016', 'he2016', 'johnson2016',
           'krittanawong2017', 'leclerc2019', 'lee2020biobert', 'litjens2017',
           'puri2025', 'radford2019', 'rajpurkar2022', 'shen2017', 'topol2019',
           'vaswani2017', 'wang2021milvus']

# grab each \bibitem{key} ... up to the next \bibitem or \end{thebibliography}
parts = re.split(r"\\bibitem\{", s)
entries = {}
for p in parts[1:]:
    key = p.split("}", 1)[0]
    body = p.split("}", 1)[1]
    body = re.split(r"\\end\{thebibliography\}", body)[0]
    entries[key] = re.sub(r"\s+", " ", body).strip()

for k in uncited:
    txt = entries.get(k, "*** NOT FOUND ***")
    print(f"[{k}]\n    {txt[:250]}\n")
