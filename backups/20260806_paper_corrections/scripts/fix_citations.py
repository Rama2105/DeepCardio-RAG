"""Cite the previously-uncited references in DeepCardio_CompIntel_Submit.tex.

Each citation is anchored to a claim the reference genuinely supports. One entry
(puri2025) is REMOVED rather than cited: it is a Kaggle "Heart Disease -- 3000
Records" dataset that this work does not use, so any citation of it would be
decorative at best and misleading at worst.
"""
import io
import re
import shutil

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
BAK = r"C:\Users\NSOUMY~1\AppData\Local\Temp\claude\G--My-Drive-DeepCardio-RAG\1719e4ef-9e91-4d1c-a909-1b5999768b09\scratchpad\DeepCardio_CompIntel_Submit.tex.bak-citations"
shutil.copy(PATH, BAK)

s = io.open(PATH, encoding="utf-8").read()
log = []


def sub(old, new, label, count=1):
    global s
    found = s.count(old)
    assert found == count, f"{label}: expected {count}, found {found}\n  {old[:90]!r}"
    s = s.replace(old, new)
    log.append(f"  OK  {label}")


# ---------------- Introduction: AI-in-medicine framing ----------------------
# topol2019, esteva2019, rajpurkar2022, krittanawong2017
sub("""Artificial intelligence~(AI) offers a transformative opportunity. Deep
learning~(DL) models trained on large ECG corpora have demonstrated
cardiologist-level accuracy for arrhythmia detection.""",
    """Artificial intelligence~(AI) offers a transformative opportunity.
Broad reviews of AI in medicine~\\cite{topol2019,rajpurkar2022} and of
deep learning in healthcare specifically~\\cite{esteva2019} document
rapid gains across diagnostic specialties, with cardiovascular medicine
identified early as a leading application area~\\cite{krittanawong2017}.
Deep learning~(DL) models trained on large ECG corpora have demonstrated
cardiologist-level accuracy for arrhythmia detection.""",
    "intro: topol2019 + rajpurkar2022 + esteva2019 + krittanawong2017")

# ---------------- Related work: ECG (baloglu2019) ---------------------------
sub("""Natarajan
\\emph{et~al.}~\\cite{natarajan2020} combined wide CNNs with multi-head
self-attention for 12-lead ECG classification.""",
    """Natarajan
\\emph{et~al.}~\\cite{natarajan2020} combined wide CNNs with multi-head
self-attention for 12-lead ECG classification. Baloglu
\\emph{et~al.}~\\cite{baloglu2019} applied a deep CNN to multi-lead ECG
for myocardial-infarction detection, the diagnostic superclass our
flagship encoder finds hardest~(per-class AUC~0.782).""",
    "related work ECG: baloglu2019")

# ---------------- Related work: echo (leclerc2019) --------------------------
sub("""left-ventricle segmentation, achieving Dice coefficient of~0.95 on the
CAMUS dataset.""",
    """left-ventricle segmentation, achieving Dice coefficient of~0.95 on the
CAMUS dataset~\\cite{leclerc2019}.""",
    "related work echo: leclerc2019")

# ---------------- Related work: heart sound (acharya2017) -------------------
sub("""Potes \\emph{et~al.}~\\cite{potes2016}
combined AdaBoost and a CNN, achieving AUC~0.86.""",
    """Potes \\emph{et~al.}~\\cite{potes2016}
combined AdaBoost and a CNN, achieving AUC~0.86. Acharya
\\emph{et~al.}~\\cite{acharya2017} applied a deep CNN directly to heart
sound recordings, an approach close to the Mel-spectrogram CNN adopted
here.""",
    "related work PCG: acharya2017")

# ---------------- Related work: clinical NLP (lee2020biobert) ---------------
sub("""ChatDoctor~\\cite{chatdoctor2023} demonstrated improved factual accuracy
through RAG augmentation of LLaMA.""",
    """Domain-adapted encoders such as BioBERT~\\cite{lee2020biobert} showed
that pre-training on biomedical corpora improves downstream clinical
text tasks --- a route this prototype does not take, since its decoder
is general-domain GPT-2.
ChatDoctor~\\cite{chatdoctor2023} demonstrated improved factual accuracy
through RAG augmentation of LLaMA.""",
    "related work NLP: lee2020biobert")

# ---------------- Methods: multi-head attention (vaswani2017) ---------------
sub("Multi-head attention concatenates $H=8$ heads:",
    "Multi-head attention~\\cite{vaswani2017} concatenates $H=8$ heads:",
    "methods: vaswani2017")

# ---------------- Methods: residual block -- bare "(He et al.)" -------------
# This was an attribution with NO citation at all, which is a defect in itself.
sub("provides a direct gradient path (He~et~al.), which we use here purely",
    "provides a direct gradient path (He \\emph{et~al.}~\\cite{he2016}), which we use here purely",
    "methods: he2016 (was an uncited attribution)")

io.open(PATH, "w", encoding="utf-8").write(s)
print("\n".join(log))
print(f"\nbackup: {BAK}")
