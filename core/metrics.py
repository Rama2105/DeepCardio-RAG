"""
core/metrics.py — Comprehensive Performance Metrics for DeepCardio-RAG
=======================================================================
Implements 10 gold-standard metrics across all pipeline modules:

  Classification (per-class + macro):
    1.  Accuracy          — (TP+TN) / N
    2.  Precision         — TP / (TP+FP)   [macro-avg]
    3.  Recall            — TP / (TP+FN)   [macro-avg] = Sensitivity
    4.  F1-Score          — 2·P·R / (P+R)  [macro-avg]
    5.  AUC-ROC           — Area under ROC curve [macro OvR]
    6.  Specificity       — TN / (TN+FP)   [macro-avg]
    7.  NPV               — TN / (TN+FN)   [macro-avg] Negative Predictive Value

  Natural Language Generation (NLG):
    8.  BLEU-4            — 4-gram corpus BLEU (Papineni et al., 2002)
    9.  ROUGE-L           — Longest Common Subsequence F1 (Lin, 2004)

  Safety / Hallucination:
   10.  Hallucination Rate — fraction of generated tokens absent from retrieved
                             clinical context (lower = safer, target <5%)

Usage:
    from core.metrics import MetricsCalculator
    calc = MetricsCalculator(n_classes=6)
    result = calc.classification_metrics(y_true, y_pred, y_prob)
    nlg = calc.nlg_metrics(generated_reports, reference_reports, contexts)
    full = calc.full_report(y_true, y_pred, y_prob, generated_reports, reference_reports, contexts)
"""

import numpy as np
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from core.logger import get_logger

logger = get_logger(__name__)

# AAMI arrhythmia class names (MIT-BIH standard)
ARRHYTHMIA_CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown/Paced"]

# Heart disease risk class names (UCI)
HEART_DISEASE_CLASS_NAMES = ["No CAD", "CAD Present"]

# Full 6-class names for the combined DeepCardio classifier
DEEPCARDIO_CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown/Paced", "ST-Elevation"]


# ──────────────────────────────────────────────────────────────────────────────
# NLG Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenizer (no external NLTK dependency)."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _bleu_n(hypothesis: List[str], references: List[List[str]], n: int) -> float:
    """Compute n-gram precision with clipping (corpus level)."""
    hyp_ngrams   = _ngrams(hypothesis, n)
    total_hyp    = sum(hyp_ngrams.values())
    if total_hyp == 0:
        return 0.0

    clipped = 0
    for ngram, cnt in hyp_ngrams.items():
        max_ref = max(_ngrams(ref, n)[ngram] for ref in references)
        clipped += min(cnt, max_ref)

    return clipped / total_hyp


def corpus_bleu4(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """
    Compute BLEU-4 score at corpus level.

    Args:
      hypotheses : list of generated report strings
      references : list of reference (gold standard) report strings

    Returns:
      BLEU-4 float in [0, 1]
    """
    hyp_toks = [_tokenize(h) for h in hypotheses]
    ref_toks  = [[_tokenize(r)] for r in references]   # one ref per hypothesis

    # Brevity penalty
    c = sum(len(h) for h in hyp_toks)
    r = sum(len(rr[0]) for rr in ref_toks)
    bp = 1.0 if c >= r else math.exp(1.0 - r / c)

    # 4-gram precision (geometric mean, avoid log(0))
    log_avg = 0.0
    eps = 1e-12
    for n in range(1, 5):
        precs = []
        for hyp, refs in zip(hyp_toks, ref_toks):
            precs.append(_bleu_n(hyp, refs, n))
        avg_p = sum(precs) / max(len(precs), 1)
        log_avg += 0.25 * math.log(max(avg_p, eps))

    return round(bp * math.exp(log_avg), 4)


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Compute LCS length via DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            tmp = dp[j + 1]
            dp[j + 1] = prev + 1 if a[i] == b[j] else max(dp[j + 1], dp[j])
            prev = tmp
    return dp[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 for a single hypothesis-reference pair."""
    h_toks = _tokenize(hypothesis)
    r_toks = _tokenize(reference)
    if not h_toks or not r_toks:
        return 0.0
    lcs = _lcs_len(h_toks, r_toks)
    precision = lcs / len(h_toks)
    recall    = lcs / len(r_toks)
    if precision + recall < 1e-12:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def corpus_rouge_l(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Macro-average ROUGE-L F1 over the corpus."""
    scores = [rouge_l(h, r) for h, r in zip(hypotheses, references)]
    return round(sum(scores) / max(len(scores), 1), 4)


def hallucination_rate(
    generated_reports: List[str],
    retrieved_contexts: List[str],
    threshold: float = 0.5,
) -> float:
    """
    Measures the fraction of generated medical tokens that do NOT appear
    in the corresponding retrieved clinical context.

    A token counts as 'hallucinated' if it is:
      - A content word (noun, number, abbreviation)
      - Absent from the retrieved context vocabulary

    Target: hallucination_rate < 5% (0.05)

    Args:
      generated_reports  : list of generated clinical report strings
      retrieved_contexts : list of concatenated retrieved context strings
      threshold          : not used (kept for API compatibility)

    Returns:
      float in [0, 1] — fraction of content tokens absent from context
    """
    total_tokens = 0
    hallucinated = 0

    # Simple content word heuristic: exclude stopwords
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","to","of","in","for",
        "on","with","as","by","from","at","into","through","and","or",
        "but","not","no","nor","so","yet","both","either","neither",
        "this","that","these","those","which","who","what","when",
        "where","how","if","then","than","such","each","every","all",
    }

    for gen, ctx in zip(generated_reports, retrieved_contexts):
        gen_tokens = [t for t in _tokenize(gen) if t not in STOPWORDS and len(t) > 2]
        ctx_vocab  = set(_tokenize(ctx))
        for tok in gen_tokens:
            total_tokens += 1
            if tok not in ctx_vocab:
                hallucinated += 1

    if total_tokens == 0:
        return 0.0
    rate = hallucinated / total_tokens
    return round(rate, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Utilities (no scikit-learn dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Build n_classes × n_classes confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _per_class_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    From confusion matrix, compute per-class:
      TP, FP, FN, TN, precision, recall, specificity, npv, f1
    """
    K   = cm.shape[0]
    TP  = np.diag(cm).astype(float)
    FP  = cm.sum(axis=0) - TP
    FN  = cm.sum(axis=1) - TP
    TN  = cm.sum() - TP - FP - FN

    eps = 1e-8
    precision   = TP / (TP + FP + eps)
    recall      = TP / (TP + FN + eps)           # sensitivity
    specificity = TN / (TN + FP + eps)
    npv         = TN / (TN + FN + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "npv":         npv,
        "f1":          f1,
    }


def _auc_roc_ovr(
    y_true:    np.ndarray,   # (N,)  int labels
    y_prob:    np.ndarray,   # (N, K) class probabilities
    n_classes: int,
) -> float:
    """
    Macro OvR AUC-ROC (no sklearn).
    Uses trapezoidal rule on sorted probability threshold.
    """
    aucs = []
    for cls in range(n_classes):
        binary_labels = (y_true == cls).astype(int)
        probs = y_prob[:, cls]
        # Sort by probability descending
        sorted_idx = np.argsort(-probs)
        sorted_labs = binary_labels[sorted_idx]
        # Cumulative TP / FP counts
        tp  = np.cumsum(sorted_labs)
        fp  = np.cumsum(1 - sorted_labs)
        n_pos = sorted_labs.sum()
        n_neg = len(sorted_labs) - n_pos
        if n_pos == 0 or n_neg == 0:
            aucs.append(0.5)
            continue
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Prepend origin
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        auc = np.trapz(tpr, fpr)
        aucs.append(abs(auc))   # abs because trapezoidal can give negative

    return round(float(np.mean(aucs)), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Main MetricsCalculator Class
# ──────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Computes all 10 DeepCardio-RAG performance metrics.

    Args:
      n_classes   : number of output classes (default 6 for full DeepCardio)
      class_names : optional list of class name strings for pretty output
    """

    def __init__(
        self,
        n_classes:   int = 6,
        class_names: Optional[List[str]] = None,
    ):
        self.n_classes   = n_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(n_classes)]

    # ── 1-7: Classification Metrics ──────────────────────────────────────────

    def classification_metrics(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None,
    ) -> Dict:
        """
        Compute the 7 classification metrics.

        Args:
          y_true : ground-truth integer class labels, shape (N,)
          y_pred : predicted class labels, shape (N,)
          y_prob : predicted class probabilities, shape (N, K); required for AUC-ROC

        Returns dict with keys:
          accuracy, precision, recall, f1, specificity, npv, auc_roc,
          per_class (dict per class name), confusion_matrix
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        N = len(y_true)

        # 1. Accuracy
        accuracy = float(np.mean(y_true == y_pred))

        # Confusion matrix
        cm = _build_confusion_matrix(y_true, y_pred, self.n_classes)

        # Per-class metrics
        per = _per_class_metrics(cm)

        # Macro averages  (metrics 2-7)
        precision   = float(np.mean(per["precision"]))
        recall      = float(np.mean(per["recall"]))
        f1          = float(np.mean(per["f1"]))
        specificity = float(np.mean(per["specificity"]))
        npv         = float(np.mean(per["npv"]))

        # AUC-ROC (metric 5)
        auc_roc = 0.0
        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob, dtype=np.float32)
            if y_prob_arr.ndim == 1:
                # Binary case: wrap into 2-column array
                y_prob_arr = np.column_stack([1 - y_prob_arr, y_prob_arr])
            auc_roc = _auc_roc_ovr(y_true, y_prob_arr, self.n_classes)

        # Per-class summary dict
        per_class = {}
        for i, name in enumerate(self.class_names):
            per_class[name] = {
                "precision":   round(float(per["precision"][i]),   4),
                "recall":      round(float(per["recall"][i]),      4),
                "specificity": round(float(per["specificity"][i]), 4),
                "npv":         round(float(per["npv"][i]),         4),
                "f1":          round(float(per["f1"][i]),          4),
                "support":     int(per["TP"][i] + per["FN"][i]),
            }

        return {
            # Metrics 1-7
            "accuracy":           round(accuracy, 4),
            "precision":          round(precision, 4),
            "recall":             round(recall, 4),
            "f1":                 round(f1, 4),
            "auc_roc":            auc_roc,
            "specificity":        round(specificity, 4),
            "npv":                round(npv, 4),
            # _macro aliases (used by Colab evaluation cell)
            "precision_macro":    round(precision, 4),
            "recall_macro":       round(recall, 4),
            "f1_macro":           round(f1, 4),
            "auc_roc_macro":      auc_roc,
            "specificity_macro":  round(specificity, 4),
            "npv_macro":          round(npv, 4),
            # Detailed per-class breakdown
            "per_class":          per_class,
            "confusion_matrix":   cm.tolist(),
            "n_samples":          N,
        }

    # ── 8-9: NLG Metrics ─────────────────────────────────────────────────────

    def nlg_metrics(
        self,
        generated_reports:  List[str],
        reference_reports:  List[str],
        retrieved_contexts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute BLEU-4, ROUGE-L, and Hallucination Rate.

        Args:
          generated_reports  : list of generated ECG report strings
          reference_reports  : list of reference/gold-standard report strings
          retrieved_contexts : list of retrieved clinical context strings (for hallucination rate)

        Returns dict with keys: bleu4, rouge_l, hallucination_rate
        """
        assert len(generated_reports) == len(reference_reports), \
            "generated_reports and reference_reports must have equal length"

        # Metric 8: BLEU-4
        bleu4_score = corpus_bleu4(generated_reports, reference_reports)

        # Metric 9: ROUGE-L
        rouge_l_score = corpus_rouge_l(generated_reports, reference_reports)

        # Metric 10: Hallucination Rate
        halluc_rate = 0.0
        if retrieved_contexts is not None:
            assert len(generated_reports) == len(retrieved_contexts), \
                "generated_reports and retrieved_contexts must have equal length"
            halluc_rate = hallucination_rate(generated_reports, retrieved_contexts)

        return {
            "bleu4":             bleu4_score,
            "rouge_l":           rouge_l_score,
            "hallucination_rate": halluc_rate,
        }

    # ── Full Report (all 10 metrics) ─────────────────────────────────────────

    def full_report(
        self,
        y_true:             Union[List[int], np.ndarray],
        y_pred:             Union[List[int], np.ndarray],
        y_prob:             Optional[Union[List, np.ndarray]],
        generated_reports:  List[str],
        reference_reports:  List[str],
        retrieved_contexts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute all 10 metrics and return a consolidated report dict.

        Returns dict with all 10 metric values + detailed breakdowns.
        """
        cls_metrics = self.classification_metrics(y_true, y_pred, y_prob)
        nlg_metrics = self.nlg_metrics(generated_reports, reference_reports, retrieved_contexts)

        summary = {
            # ── Classification (metrics 1-7) ──
            "1_accuracy":      cls_metrics["accuracy"],
            "2_precision":     cls_metrics["precision"],
            "3_recall":        cls_metrics["recall"],
            "4_f1":            cls_metrics["f1"],
            "5_auc_roc":       cls_metrics["auc_roc"],
            "6_specificity":   cls_metrics["specificity"],
            "7_npv":           cls_metrics["npv"],
            # ── NLG (metrics 8-9) ──
            "8_bleu4":         nlg_metrics["bleu4"],
            "9_rouge_l":       nlg_metrics["rouge_l"],
            # ── Safety (metric 10) ──
            "10_hallucination_rate": nlg_metrics["hallucination_rate"],
        }

        full = {**summary, **cls_metrics, **nlg_metrics}

        # Targets check
        full["targets_met"] = {
            "accuracy_gt_95pct":     cls_metrics["accuracy"] >= 0.95,
            "f1_gt_90pct":           cls_metrics["f1"] >= 0.90,
            "auc_roc_gt_90pct":      cls_metrics["auc_roc"] >= 0.90,
            "hallucination_lt_5pct": nlg_metrics["hallucination_rate"] < 0.05,
        }

        self._log_summary(summary)
        return full

    @staticmethod
    def _log_summary(summary: Dict) -> None:
        logger.info("=" * 60)
        logger.info("DeepCardio-RAG — Performance Metrics Summary")
        logger.info("=" * 60)
        labels = {
            "1_accuracy":           "1. Accuracy          ",
            "2_precision":          "2. Precision (macro) ",
            "3_recall":             "3. Recall/Sensitivity",
            "4_f1":                 "4. F1-Score (macro)  ",
            "5_auc_roc":            "5. AUC-ROC (macro)   ",
            "6_specificity":        "6. Specificity (macro)",
            "7_npv":                "7. NPV (macro)       ",
            "8_bleu4":              "8. BLEU-4            ",
            "9_rouge_l":            "9. ROUGE-L           ",
            "10_hallucination_rate": "10. Hallucination Rate",
        }
        for key, label in labels.items():
            val = summary.get(key, "N/A")
            pct = f"{val*100:.2f}%" if isinstance(val, float) else str(val)
            logger.info(f"   {label}: {pct}")
        logger.info("=" * 60)

    # ── Pipeline-specific convenience wrappers ────────────────────────────────

    @classmethod
    def ecg_arrhythmia(cls) -> "MetricsCalculator":
        """Pre-configured for MIT-BIH 5-class arrhythmia classification."""
        return cls(n_classes=5, class_names=ARRHYTHMIA_CLASS_NAMES)

    @classmethod
    def heart_disease(cls) -> "MetricsCalculator":
        """Pre-configured for UCI Heart Disease binary classification."""
        return cls(n_classes=2, class_names=HEART_DISEASE_CLASS_NAMES)

    @classmethod
    def deepcardio(cls) -> "MetricsCalculator":
        """Pre-configured for full 6-class DeepCardio-RAG output."""
        return cls(n_classes=6, class_names=DEEPCARDIO_CLASS_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Metric Functions (for use in training loops)
# ──────────────────────────────────────────────────────────────────────────────

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm  = _build_confusion_matrix(np.asarray(y_true), np.asarray(y_pred), n_classes)
    per = _per_class_metrics(cm)
    return round(float(np.mean(per["f1"])), 4)


def auc_roc_macro(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    return _auc_roc_ovr(np.asarray(y_true), np.asarray(y_prob), n_classes)


def specificity_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm  = _build_confusion_matrix(np.asarray(y_true), np.asarray(y_pred), n_classes)
    per = _per_class_metrics(cm)
    return round(float(np.mean(per["specificity"])), 4)


def npv_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm  = _build_confusion_matrix(np.asarray(y_true), np.asarray(y_pred), n_classes)
    per = _per_class_metrics(cm)
    return round(float(np.mean(per["npv"])), 4)
