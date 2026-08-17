import io
import re
from collections import Counter

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
s = io.open(PATH, encoding="utf-8").read()

# ---- citations vs bibitems ------------------------------------------------
cited = Counter()
for m in re.finditer(r"\\cite\{([^}]*)\}", s):
    for k in m.group(1).split(","):
        cited[k.strip()] += 1
defined = set(re.findall(r"\\bibitem\{([^}]*)\}", s))

print("=== CITATIONS ===")
print(f"  distinct cited : {len(cited)}")
print(f"  bibitems       : {len(defined)}")
missing = sorted(k for k in cited if k not in defined)
unused = sorted(k for k in defined if k not in cited)
print(f"  CITED BUT UNDEFINED : {missing or 'none'}")
print(f"  defined but never cited: {unused or 'none'}")

# ---- labels vs refs -------------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]*)\}", s))
refs = set()
for m in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]*)\}", s):
    refs.add(m.group(1))
print("\n=== CROSS-REFERENCES ===")
print(f"  REF WITHOUT LABEL : {sorted(refs - labels) or 'none'}")
print(f"  label never referenced: {sorted(labels - refs) or 'none'}")

# ---- duplicated labels ----------------------------------------------------
lab_counts = Counter(re.findall(r"\\label\{([^}]*)\}", s))
dupes = [k for k, v in lab_counts.items() if v > 1]
print(f"  DUPLICATE labels  : {dupes or 'none'}")

# ---- percentage / metric-looking numbers, for manual eyeball ---------------
print("\n=== ALL PERCENT CLAIMS (context trimmed) ===")
seen = set()
for m in re.finditer(r"([^\n]{0,70}?)(\d{1,3}(?:\.\d+)?)\\%", s):
    ctx = re.sub(r"\s+", " ", m.group(1)).strip()[-62:]
    key = (ctx, m.group(2))
    if key in seen:
        continue
    seen.add(key)
    print(f"  {m.group(2):>7}%  ...{ctx}")
