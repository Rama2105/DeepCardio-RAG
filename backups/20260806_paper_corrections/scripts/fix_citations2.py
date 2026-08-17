"""Second citation batch + three defects found while placing them."""
import io
import re
import shutil

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
BAK = r"C:\Users\NSOUMY~1\AppData\Local\Temp\claude\G--My-Drive-DeepCardio-RAG\1719e4ef-9e91-4d1c-a909-1b5999768b09\scratchpad\DeepCardio_CompIntel_Submit.tex.bak-citations2"
shutil.copy(PATH, BAK)

s = io.open(PATH, encoding="utf-8").read()
log = []


def sub(old, new, label, count=1):
    global s
    found = s.count(old)
    assert found == count, f"{label}: expected {count}, found {found}\n  {old[:90]!r}"
    s = s.replace(old, new)
    log.append(f"  OK  {label}")


# ===== DEFECT: two more stale "16" corpus sizes (live Milvus holds 67) ========
sub("A curated corpus of 16~clinical guideline chunks from AHA/ACC",
    "A curated corpus of 67~clinical guideline chunks from AHA/ACC",
    "stale corpus size (methods prose)")
sub("""        fallback) with a prototype clinical guidelines corpus of
        16~chunks, enabling sub-second RAG retrieval in a working
        deployment.""",
    """        fallback) with a prototype clinical guidelines corpus of
        67~chunks, enabling sub-second RAG retrieval in a working
        deployment.""",
    "stale corpus size (contributions bullet)")

# ===== DEFECT: conclusion reports intervals the paper says are NOT computed ===
# tab:ecgparams: "The prototype does not implement delineation, so no interval
# is computed from the signal" -- every row reads "Not computed". The conclusion
# nonetheless printed HR/PR/QRS/QTc and a rhythm call for the demo record.
sub("""\\textbf{Clinical accessibility}: Dual-format PDF generation---confirmed
in the June~15, 2026 deployment~(ECG-20260615-0836, HR~72~bpm,
PR~160~ms, QRS~90~ms, QTc~410~ms, Normal Sinus Rhythm)---democratises""",
    """\\textbf{Clinical accessibility}: Dual-format PDF generation---exercised
end-to-end on PTB-XL record~\\#10586---democratises""",
    "conclusion: withdrew uncomputed ECG intervals")

# ===== Citations =============================================================
# devlin2019 -- TabularBERT tokenisation is a BERT-style embedding scheme.
sub("""\\subsubsection{TabularBERT Feature Tokenisation}

\\begin{equation}""",
    """\\subsubsection{TabularBERT Feature Tokenisation}

Following the BERT encoder formulation~\\cite{devlin2019}, each tabular
feature is projected to its own embedding vector and treated as a token,
so that self-attention operates over features rather than word pieces:

\\begin{equation}""",
    "methods: devlin2019")

# radford2019 -- GPT-2 is the decoder actually used.
sub("""Retrieved chunks are prepended as soft prompts to a GPT-2 decoder
(124M~parameters), which synthesises dual-format PDF reports""",
    """Retrieved chunks are prepended as soft prompts to a GPT-2
decoder~\\cite{radford2019} (124M~parameters), which synthesises
dual-format PDF reports""",
    "methods: radford2019")

# wang2021milvus + douze2024 -- the vector store and its fallback.
sub("""768~dimensions) and stored in Milvus (collection:
\\texttt{cardio\\_knowledge\\_base}, L2 metric).""",
    """768~dimensions) and stored in Milvus~\\cite{wang2021milvus} (collection:
\\texttt{cardio\\_knowledge\\_base}, L2 metric); where no Milvus server is
reachable the implementation falls back to a local FAISS
index~\\cite{douze2024}, and Section~\\ref{sec:limitations} notes that this
fallback is silent.""",
    "methods: wang2021milvus + douze2024")

# goldberger2000 -- PhysioNet, the source of four of the six corpora.
sub("""\\multicolumn{5}{l}{$\\ddagger$ Public training subset; the full release is 5\\,272 recordings.}""",
    """\\multicolumn{5}{l}{$\\ddagger$ Public training subset; the full release is 5\\,272 recordings.}\\\\
\\multicolumn{5}{l}{PhysioNet-hosted corpora are distributed via PhysioBank~\\cite{goldberger2000}.}""",
    "datasets: goldberger2000")

# litjens2017 + shen2017 + goodfellow2016 -- survey/textbook grounding.
sub("""Broad reviews of AI in medicine~\\cite{topol2019,rajpurkar2022} and of
deep learning in healthcare specifically~\\cite{esteva2019} document""",
    """Broad reviews of AI in medicine~\\cite{topol2019,rajpurkar2022} and of
deep learning in healthcare specifically~\\cite{esteva2019}, together with
surveys of deep learning in medical image
analysis~\\cite{litjens2017,shen2017} and the standard treatment of the
underlying methods~\\cite{goodfellow2016}, document""",
    "intro: litjens2017 + shen2017 + goodfellow2016")

# johnson2016 -- MIMIC-III as a future-work direction, not a dataset used here.
sub("""  \\item \\textbf{Federated learning}: Privacy-preserving multi-site
        training via federated averaging.""",
    """  \\item \\textbf{Linkage to richer clinical context}: none of the six
        corpora carry longitudinal notes, medications, or outcomes.
        Coupling the modules to an intensive-care database such as
        MIMIC-III~\\cite{johnson2016} would allow the retrieval corpus to
        be grounded in patient history rather than guideline text alone.
  \\item \\textbf{Federated learning}: Privacy-preserving multi-site
        training via federated averaging.""",
    "future work: johnson2016")

# ===== Remove puri2025: a dataset this work does not use =====================
before = s.count(r"\bibitem{puri2025}")
assert before == 1, f"puri2025 bibitem count = {before}"
s = re.sub(r"\\bibitem\{puri2025\}.*?(?=\\bibitem\{|\\end\{thebibliography\})",
           "", s, flags=re.DOTALL)
assert s.count(r"\bibitem{puri2025}") == 0
log.append("  OK  removed puri2025 (Kaggle 'Heart Disease 3000 Records' -- not used by this work)")

io.open(PATH, "w", encoding="utf-8").write(s)
print("\n".join(log))
print(f"\nbackup: {BAK}")
