import sys
sys.path.insert(0, r"G:\My Drive\DeepCardio-RAG")
import warnings
warnings.filterwarnings("ignore")

from core.vfdb_loader import get_vfdb_dataset, DANGEROUS_RHYTHMS

loader = get_vfdb_dataset()
recs = getattr(loader, "records", [])
print("records loaded:", len(recs))
print("loader DANGEROUS_RHYTHMS:", sorted(DANGEROUS_RHYTHMS))

from collections import Counter
tally = Counter()
total_ann = 0
for r in recs:
    for a in r.get("annotations", []):
        total_ann += 1
        tally[a.get("rhythm")] += 1

print("\nannotation counts by rhythm:")
for k, v in tally.most_common():
    print(f"   {str(k):8} {v}")

paper_set = {"VT", "VF", "VFL", "ASYS"}
loader_set = set(DANGEROUS_RHYTHMS)
print("\npaper's 4 types  (VT,VF,VFL,ASYS):", sum(v for k, v in tally.items() if k in paper_set))
print("loader's 7 types                  :", sum(v for k, v in tally.items() if k in loader_set))
