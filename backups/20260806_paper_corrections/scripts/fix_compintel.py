"""Apply the eight verified corrections to DeepCardio_CompIntel_Submit.tex.

Every replacement asserts its expected occurrence count, so a silent no-op or an
over-broad match fails loudly rather than corrupting the paper.
"""
import io
import shutil

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
BAK = r"C:\Users\NSOUMY~1\AppData\Local\Temp\claude\G--My-Drive-DeepCardio-RAG\1719e4ef-9e91-4d1c-a909-1b5999768b09\scratchpad\DeepCardio_CompIntel_Submit.tex.bak-8fixes"
shutil.copy(PATH, BAK)

s = io.open(PATH, encoding="utf-8").read()
log = []


def sub(old, new, count, label):
    global s
    found = s.count(old)
    assert found == count, f"{label}: expected {count} occurrence(s), found {found}\n  {old[:90]!r}"
    s = s.replace(old, new)
    log.append(f"  OK  {label}  (x{count})")


# --- 1. VFDB dangerous-event count: 142 was a hardcoded fallback constant --------
# vfdb_loader.py:390 hardcodes 142 in its demo/fallback summary. The real annotated
# count over the 22 records is 239 under the paper's OWN rhythm set
# (VT 93 + VFL 98 + VF 9 + VFIB 4 + ASYS 13 + HGEA 19 + VER 3).
sub(r"$\dagger$ 142 dangerous events, 30 min/recording.",
    r"$\dagger$ 239 dangerous events, 30 min/recording.", 1, "VFDB count tab:datasets")
sub("containing 142~dangerous arrhythmia events",
    "containing 239~dangerous arrhythmia events", 1, "VFDB count prose 1")
sub(r"(the 142 \emph{dangerous} ventricular events being VT, VF, VFL, ASYS,",
    r"(the 239 \emph{dangerous} ventricular events being VT, VF, VFL, VFIB, ASYS,", 1, "VFDB count prose 2")
sub("VFDB & 22 records & 142 dangerous events; 30~min/rec;",
    "VFDB & 22 records & 239 dangerous events; 30~min/rec;", 1, "VFDB count tab:eda")
sub(r"The 142~\emph{dangerous}", r"The 239~\emph{dangerous}", 1, "VFDB count EDA prose")
sub(r"as annotations but are \emph{not} counted among the 142 dangerous",
    r"as annotations but are \emph{not} counted among the 239 dangerous", 1, "VFDB count EDA prose 2")
sub("VFDB~(22~records, 142~dangerous events, 14~rhythm types).",
    "VFDB~(22~records, 239~dangerous events, 14~rhythm types).", 1, "VFDB count module prose")
sub(r"Total Dangerous Events  & 142\\", r"Total Dangerous Events  & 239\\", 1, "VFDB count tab:vfdb")

# --- 2. Remove the withdrawn demo alert from tab:vfdb ----------------------------
# The paper itself withdraws this exact alert later, calling it "raised by a
# random-weight network on a record its own ECG module concurrently read as
# normal sinus rhythm". Leaving it tabulated contradicts that withdrawal.
sub("Demo Detection          & VF/VFL --- Confidence 32.5\\%\\\\\n"
    "Alert Level             & \\textbf{CRITICAL} (with PCG combined)\\\\\n",
    "% Demo Detection / Alert Level rows WITHDRAWN 2026-08-06: this is the same\n"
    "% random-weight VF/VFL \"CRITICAL\" alert that this paper withdraws in the\n"
    "% Overall System Performance subsection. Do not reinstate without a trained\n"
    "% model and a real record.\n", 1, "tab:vfdb withdrawn demo alert")

# --- 3. CirCor: report the subset actually used, not the full published set ------
sub(r"Total Subjects              & 1\,568\\", r"Total Subjects              & 942\\", 1, "CirCor subjects")
sub(r"Total Recordings            & 5\,272\\", r"Total Recordings            & 3\,163\\", 1, "CirCor recordings")
sub(r"Murmur Absent               & 2\,575~(48.8\%)\\", r"Murmur Absent               & 2\,391~(75.6\%)\\", 1, "CirCor absent")
sub(r"Murmur Present              & 1\,727~(32.8\%)\\", r"Murmur Present              & 616~(19.5\%)\\", 1, "CirCor present")
sub(r"Unknown                     & 970~(18.4\%)\\", r"Unknown                     & 156~(4.9\%)\\", 1, "CirCor unknown")
sub("CirCor & 5\\,272 recordings & 1\\,568 subjects; Absent~2\\,575,\n  Present~1\\,727, Unknown~970\\\\",
    "CirCor & 3\\,163 recordings & 942 subjects; Absent~2\\,391,\n  Present~616, Unknown~156 (public training subset;\n  the full 5\\,272/1\\,568 release includes a hidden test set)\\\\",
    1, "CirCor tab:eda row")
sub(r"PCG & CirCor DigiScope & 5\,272 recs & 3 & PhysioNet\\",
    r"PCG & CirCor DigiScope & 3\,163 recs$^{\ddagger}$ & 3 & PhysioNet\\", 1, "CirCor tab:datasets")
sub(r"\multicolumn{5}{l}{$\dagger$ 239 dangerous events, 30 min/recording.}",
    "\\multicolumn{5}{l}{$\\dagger$ 239 dangerous events, 30 min/recording.}\\\\\n"
    "\\multicolumn{5}{l}{$\\ddagger$ Public training subset; the full release is 5\\,272 recordings.}",
    1, "CirCor footnote")

# --- 4. UCI Cleveland: prevalence was stated with H and D swapped ----------------
# The Kaggle mirror's target is inverted (measured: target=0 is the diseased group),
# so 499 are diseased and 526 healthy -> prevalence 48.7%, not 51.3%.
sub(r"UCI HD & 1\,025 patients & Prevalence~51.3\% (499H/526D);",
    r"UCI HD & 1\,025 rows (302 unique) & Prevalence~48.7\% (499~diseased/526~healthy);", 1, "UCI HD tab:eda")
sub(r"The dataset shape is~(1025, 14) with disease prevalence~51.3\%",
    r"The dataset shape is~(1025, 14) --- only 302 rows are unique --- with disease prevalence~48.7\%",
    1, "UCI HD prose")
sub(r"HD & UCI Cleveland & 1\,025 pts & 2 & Kaggle\\",
    r"HD & UCI Cleveland & 1\,025 rows (302 uniq.) & 2 & Kaggle\\", 1, "UCI HD tab:datasets")

# --- 5. RAG corpus is 67 documents, not 16 --------------------------------------
# The 16 is the pre-rebuild MOCK_KNOWLEDGE_BASE size; tab:sysperf already says 67,
# and the live Milvus collection holds 67. k=5 is ~7.5% of 67, not "a third".
sub("metrics in Section~\\ref{sec:results} are reported over this 16-chunk\ncorpus",
    "metrics in Section~\\ref{sec:results} are reported over this 67-document\ncorpus", 1, "RAG corpus prose 1")
sub("patient. With a 16-chunk corpus and $k=5$, top-$k$ retrieval\n"
    "necessarily returns a third of the corpus regardless of query, which\n"
    "is the mechanism behind this failure;",
    "patient. With a 67-document corpus and $k=5$, top-$k$ retrieval\n"
    "returns a fixed five documents regardless of query relevance, which\n"
    "is the mechanism behind this failure;", 1, "RAG corpus prose 2")
sub("query~(top-$k$, $k=5$, $L_2$ search over the 16-chunk prototype\ncorpus, embedding dim~768). Because $k$ equals nearly a third of the\ncorpus, the",
    "query~(top-$k$, $k=5$, $L_2$ search over the 67-document prototype\ncorpus, embedding dim~768). Because the corpus is small and\nhand-seeded, the", 1, "RAG corpus prose 3")
sub(r"Retrieved Guidelines     & Top-5 of 16-chunk corpus & Top-$k$=5\\",
    r"Retrieved Guidelines     & Top-5 of 67-document corpus & Top-$k$=5\\", 1, "RAG tab:ragperf retrieved")
sub("Guideline Chunks         & 16                & ---\\\\",
    "Guideline Chunks         & 67                & ---\\\\", 1, "RAG tab:ragperf chunks")

# --- 6. The SVC that the saved model does not contain ---------------------------
# data/arthritis_bert_moe_scaler.pkl base_learners = {gbm, rf, et}. No SVC.
sub(r"        + SVC $\rightarrow$ LogReg meta-learner), evaluated on a",
    r"        $\rightarrow$ LogReg meta-learner), evaluated on a", 1, "SVC abstract")
sub(r"ExtraTrees~$+$~SVC $\rightarrow$ LogReg Meta-Learner~(17~engineered",
    r"ExtraTrees $\rightarrow$ LogReg Meta-Learner~(17~engineered", 1, "SVC methods")
sub(r"Model Stack             & BERT-MoE $+$ GBM $+$ RF $+$ ExtraTrees $+$ SVC $\to$ LogReg\\",
    r"Model Stack             & BERT-MoE $+$ GBM $+$ RF $+$ ExtraTrees $\to$ LogReg\\", 1, "SVC tab:arthr")

# --- 7. tab:ptrecords "Risk" column is the ground-truth diagnosis label ----------
sub(r"\textbf{SBP} & \textbf{Inflam.} & \textbf{Risk}\\",
    r"\textbf{SBP} & \textbf{Inflam.} & \textbf{Arthritis (dx)}\\", 1, "tab:ptrecords header")

io.open(PATH, "w", encoding="utf-8").write(s)
print("\n".join(log))
print(f"\nbackup: {BAK}")
