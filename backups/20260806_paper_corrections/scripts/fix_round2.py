"""Round-2 corrections to DeepCardio_CompIntel_Submit.tex.

Axis swept this round: citations, cross-references, and every surviving
demo-patient confidence. Each replacement asserts its occurrence count.
"""
import io
import shutil

PATH = r"G:\My Drive\DeepCardio-RAG\DeepCardio_CompIntel_Submit.tex"
BAK = r"C:\Users\NSOUMY~1\AppData\Local\Temp\claude\G--My-Drive-DeepCardio-RAG\1719e4ef-9e91-4d1c-a909-1b5999768b09\scratchpad\DeepCardio_CompIntel_Submit.tex.bak-round2"
shutil.copy(PATH, BAK)

s = io.open(PATH, encoding="utf-8").read()
log = []


def sub(old, new, count, label):
    global s
    found = s.count(old)
    assert found == count, f"{label}: expected {count}, found {found}\n  {old[:100]!r}"
    s = s.replace(old, new)
    log.append(f"  OK  {label} (x{count})")


# --- 1. ECG-image demo result: the classifier is UNTRAINED -----------------
# data/ecg_image_classifier.pt does not exist, so this softmax is random —
# the five probabilities (22.9/8.0/23.3/17.1/28.8) are near-uniform, which is
# what a random 5-class network produces. It also auto-triggered a clinical
# directive. Same defect as the withdrawn EchoNet and VFDB demo outputs.
sub("""Demo result: \\textbf{Unclassifiable~(Q)}, Confidence~28.8\\%
(N~22.9\\%, S~8.0\\%, V~23.3\\%, F~17.1\\%, Q~28.8\\%).
Clinical guideline auto-triggered: ``Unclassifiable beat morphology.
Recommend manual review by electrophysiologist.''""",
    """No classification is reported for this pipeline: no trained
weights exist for the image classifier (\\texttt{ecg\\_image\\_classifier.pt}
was never produced), so the network is randomly initialised. An earlier
draft reported a demo result of \\textbf{Unclassifiable~(Q)} at 28.8\\%
confidence, with a guideline auto-triggered recommending
electrophysiologist review; that is withdrawn. Its five class
probabilities were near-uniform~(22.9/8.0/23.3/17.1/28.8\\%), which is
what an untrained five-class softmax returns, and the module is
suppressed at inference by the safety gate.""", 1, "ECG-image demo withdrawn")

# --- 2. tab:pcg demo-detection block ---------------------------------------
# The paper states outright (heart-disease retraction) that "no demo-patient
# confidence is reported anywhere in this paper". These rows contradicted that.
# The PCG module IS trained, so this is not an untrained-output problem — it is
# a single unvalidated demo case reported with a sub-50% "detection" that
# triggers a clinical guideline.
sub("""\\midrule
Demo Detection Result       & Murmur Detected\\\\
Absent Confidence           & 26.9\\%\\\\
Present Confidence          & \\textbf{44.4\\%}\\\\
Unknown Confidence          & 28.6\\%\\\\
\\midrule
AHA/ACC Guideline Triggered & Newly detected murmurs warrant transthoracic echocardiography\\\\
""",
    """% Demo Detection block WITHDRAWN 2026-08-06. The heart-disease retraction
% in this paper states that no demo-patient confidence is reported anywhere;
% these rows contradicted it. They also declared "Murmur Detected" on a 44.4%
% Present confidence -- below the 50% needed to prefer it over the alternatives
% -- and auto-triggered an echocardiography recommendation from it. Held-out
% PCG performance is in Table~\\ref{tab:sysperf}.
""", 1, "tab:pcg demo block withdrawn")

# --- 3. Duplicate \label{eq:cf} --------------------------------------------
# Both equations state the same aggregation; LaTeX warns "Label `eq:cf'
# multiply defined" and \ref resolves to whichever came last. They also used
# two different symbols for the same gated set (\mathcal{A} vs
# \mathcal{M}_{gated}). Keep the Methods equation (which carries the weights
# and defines its set), drop the restatement, and unify on \mathcal{A}.
sub("""\\begin{equation}
R \\;=\\; 100 \\cdot
  \\frac{\\sum_{m\\in\\mathcal{M}_{\\mathrm{gated}}} \\lambda_m\\, p_m^{\\mathrm{risk}}}
       {\\sum_{m\\in\\mathcal{M}_{\\mathrm{gated}}} \\lambda_m},
\\qquad p_m^{\\mathrm{risk}} = f_m(\\hat{y}_m)
\\label{eq:cf}
\\end{equation}
""",
    """\\noindent In that aggregation $p_m^{\\mathrm{risk}} = f_m(\\hat{y}_m)$ maps each
module's raw output to a risk probability.
""", 1, "duplicate eq:cf removed")

sub("First, $\\mathcal{M}_{\\mathrm{gated}}$ contains only modules that pass",
    "First, $\\mathcal{A}$ contains only modules that pass", 1, "unify set symbol 1")
sub("$\\mathcal{M}_{\\mathrm{gated}}$ entirely --- including an echo output",
    "$\\mathcal{A}$ entirely --- including an echo output", 1, "unify set symbol 2")

io.open(PATH, "w", encoding="utf-8").write(s)
print("\n".join(log))
print(f"\nbackup: {BAK}")
