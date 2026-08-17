"""
core/rag_eval.py — Retrieval-quality evaluation for the DeepCardio-RAG store
============================================================================
Answers RR review point #6 ("Evaluate retrieval quality using standard RAG
metrics"). Previously the pipeline reported NO retrieval metrics.

Implements the standard IR metrics — Recall@k, Precision@k, MRR, nDCG@k, Hit@k —
against a hand-labelled clinical relevance benchmark built from the actual
knowledge base (database/seed_data.MOCK_KNOWLEDGE_BASE).

The benchmark must be kept in step with the corpus. When the corpus grew from 16
to 67 documents (2026-07-28) the original 12 queries covered only the first 16,
so every new document was an unanswerable distractor and the scores measured a
corpus the benchmark no longer described. Queries covering all five PTB-XL
superclasses were added for that reason. If you add documents, add queries.

Two retrievers are compared honestly:
    * SEMANTIC : Sentence-BERT (all-mpnet-base-v2, 768-dim) — the production model.
    * LEXICAL  : TF-IDF cosine — an offline baseline so the semantic retriever's
                 added value is quantified (a mini retrieval ablation).

The relevance labels are curated from the KB's clinical topic structure and are
transparent (see CLINICAL_QUERIES below) — a small domain benchmark, clearly
labelled as such, not a fabricated score.
"""
import numpy as np
from typing import Dict, List, Callable

from database.seed_data import MOCK_KNOWLEDGE_BASE

# ---------------------------------------------------------------------------
# Labelled clinical retrieval benchmark (query -> relevant KB document ids)
# Relevance is judged by clinical topic match against MOCK_KNOWLEDGE_BASE.
# ---------------------------------------------------------------------------
CLINICAL_QUERIES: List[Dict] = [
    {"query": "ST-segment elevation in anterior precordial leads V2 to V4",
     "relevant": ["gdl_001"]},
    {"query": "irregular R-R intervals with no visible P waves",
     "relevant": ["case_2410"]},
    {"query": "frequent premature ventricular contractions risk of cardiomyopathy",
     "relevant": ["gdl_002", "gdl_arr_001"]},
    {"query": "T-wave inversion in a young athlete right ventricular cardiomyopathy",
     "relevant": ["case_3051"]},
    {"query": "typical angina with ST depression and vessel narrowing on angiography",
     "relevant": ["gdl_hd_001", "case_hd_001"]},
    {"query": "exercise induced angina chronotropic incompetence multivessel disease",
     "relevant": ["gdl_hd_002"]},
    {"query": "high cholesterol and hypertension in older patient statin therapy",
     "relevant": ["gdl_hd_003"]},
    {"query": "reversible thalassemia defect on nuclear stress imaging ischaemia",
     "relevant": ["gdl_hd_004"]},
    {"query": "supraventricular ectopic beats association with atrial fibrillation",
     "relevant": ["gdl_arr_002", "case_arr_002"]},
    {"query": "fusion beats simultaneous supraventricular and ventricular activation",
     "relevant": ["gdl_arr_003"]},
    {"query": "AAMI EC57 five heartbeat classes normal SVEB VEB fusion unknown",
     "relevant": ["gdl_arr_004"]},
    {"query": "multifocal PVCs R-on-T phenomenon ICD recommendation",
     "relevant": ["case_arr_001", "gdl_arr_001"]},

    # ══════════════════════════════════════════════════════════════════════════
    # PTB-XL superclass queries (added 2026-07-28 alongside the corpus
    # expansion). Written as clinical questions rather than paraphrases of the
    # document text: lifting document wording would hand the lexical baseline an
    # artificial win, while pure paraphrase would hand it to the semantic one.
    # ══════════════════════════════════════════════════════════════════════════

    # ── NORM ──
    {"query": "what heart rate and PR interval define normal sinus rhythm in an adult",
     "superclass": "NORM", "relevant": ["gdl_norm_001", "gdl_norm_002"]},
    {"query": "is beat to beat heart rate variation with breathing in a teenager anything to worry about",
     "superclass": "NORM", "relevant": ["gdl_norm_004"]},
    {"query": "healthy young man with concave ST elevation and notching at the J point, is this a heart attack",
     "superclass": "NORM", "relevant": ["gdl_norm_005", "case_norm_002"]},
    {"query": "patient has chest pain but the tracing looks completely normal, is it safe to discharge",
     "superclass": "NORM", "relevant": ["gdl_norm_006"]},

    # ── MI ──
    {"query": "how much ST elevation is required in V2 and V3 to diagnose infarction in a woman",
     "superclass": "MI", "relevant": ["gdl_mi_001"]},
    {"query": "tall R wave with ST depression in the anterior chest leads, what am I missing",
     "superclass": "MI", "relevant": ["gdl_mi_005"]},
    {"query": "pain free now but deep biphasic T waves in the anteroseptal leads, what does it mean",
     "superclass": "MI", "relevant": ["gdl_mi_007"]},
    {"query": "how can I diagnose infarction when the patient already has a left bundle branch block",
     "superclass": "MI", "relevant": ["gdl_mi_008", "gdl_cd_002"]},
    {"query": "inferior infarct with low blood pressure, which extra leads should I record",
     "superclass": "MI", "relevant": ["gdl_mi_006", "case_mi_002"]},

    # ── STTC ──
    {"query": "downsloping ST depression brought on by exertion, how significant is it",
     "superclass": "STTC", "relevant": ["gdl_sttc_001", "case_sttc_001"]},
    {"query": "tall narrow peaked T waves in a dialysis patient who missed a session",
     "superclass": "STTC", "relevant": ["gdl_sttc_006"]},
    {"query": "prominent U waves with flattened T waves, which electrolyte abnormality",
     "superclass": "STTC", "relevant": ["gdl_sttc_005"]},
    {"query": "widespread ST elevation with PR segment depression and no reciprocal change",
     "superclass": "STTC", "relevant": ["gdl_sttc_007"]},
    {"query": "scooped sagging ST segments with a short QT in a patient on cardiac medication",
     "superclass": "STTC", "relevant": ["gdl_sttc_004"]},

    # ── CD ──
    {"query": "wide QRS with an M shaped complex in V1 and a slurred S wave laterally",
     "superclass": "CD", "relevant": ["gdl_cd_001"]},
    {"query": "the PR interval lengthens each beat until one fails to conduct, does this need pacing",
     "superclass": "CD", "relevant": ["gdl_cd_007"]},
    {"query": "constant PR interval with sudden dropped beats, what is the risk of progression",
     "superclass": "CD", "relevant": ["gdl_cd_007", "case_cd_002"]},
    {"query": "atrial rate completely independent of a slow wide escape rhythm",
     "superclass": "CD", "relevant": ["gdl_cd_008"]},
    {"query": "short PR interval with a delta wave, which drugs must be avoided in atrial fibrillation",
     "superclass": "CD", "relevant": ["gdl_cd_011"]},
    {"query": "elderly patient with blackouts, right bundle branch block and marked left axis deviation",
     "superclass": "CD", "relevant": ["gdl_cd_009", "case_cd_001"]},

    # ── HYP ──
    {"query": "voltage threshold for summing the S wave in V1 and the R wave in V5",
     "superclass": "HYP", "relevant": ["gdl_hyp_001"]},
    {"query": "criterion adding the R wave in aVL to the S wave in V3 and its thresholds by sex",
     "superclass": "HYP", "relevant": ["gdl_hyp_002"]},
    {"query": "broad notched P wave in lead II with a deep negative terminal deflection in V1",
     "superclass": "HYP", "relevant": ["gdl_hyp_006"]},
    {"query": "dominant R wave in V1 with right axis deviation in suspected pulmonary hypertension",
     "superclass": "HYP", "relevant": ["gdl_hyp_005", "case_hyp_002"]},
    {"query": "why does the tracing miss ventricular thickening that echocardiography detects",
     "superclass": "HYP", "relevant": ["gdl_hyp_009", "gdl_hyp_001"]},

    # ── Deliberate cross-class query: the same finding is indexed under both
    #    the repolarization and the hypertrophy headings. ──
    {"query": "lateral ST depression and T inversion caused by a thickened ventricle rather than ischaemia",
     "superclass": "HYP", "relevant": ["gdl_sttc_008", "gdl_hyp_004", "case_hyp_001"]},
]


# ---------------------------------------------------------------------------
# Metric primitives (binary relevance)
# ---------------------------------------------------------------------------
def precision_at_k(ranked: List[str], relevant: set, k: int) -> float:
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / max(k, 1)


def recall_at_k(ranked: List[str], relevant: set, k: int) -> float:
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / max(len(relevant), 1)


def hit_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def reciprocal_rank(ranked: List[str], relevant: set) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[str], relevant: set, k: int) -> float:
    dcg = sum((1.0 if d in relevant else 0.0) / np.log2(i + 1)
              for i, d in enumerate(ranked[:k], start=1))
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Retrievers over the KB
# ---------------------------------------------------------------------------
def _rank(sim_row: np.ndarray, doc_ids: List[str]) -> List[str]:
    return [doc_ids[i] for i in np.argsort(-sim_row)]


def semantic_ranker(model_name: str = "all-mpnet-base-v2") -> Callable:
    """Sentence-BERT cosine ranker (the production retriever)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in MOCK_KNOWLEDGE_BASE]
    ids = [d["id"] for d in MOCK_KNOWLEDGE_BASE]
    doc_emb = model.encode(texts, normalize_embeddings=True)

    def rank(query: str) -> List[str]:
        q = model.encode([query], normalize_embeddings=True)[0]
        return _rank(doc_emb @ q, ids)
    return rank


def lexical_ranker() -> Callable:
    """TF-IDF cosine ranker — offline lexical baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [d["text"] for d in MOCK_KNOWLEDGE_BASE]
    ids = [d["id"] for d in MOCK_KNOWLEDGE_BASE]
    vec = TfidfVectorizer(stop_words="english")
    doc_mat = vec.fit_transform(texts)

    def rank(query: str) -> List[str]:
        sim = cosine_similarity(vec.transform([query]), doc_mat)[0]
        return _rank(sim, ids)
    return rank


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_retriever(rank_fn: Callable, ks=(1, 3, 5)) -> Dict:
    agg = {f"precision@{k}": [] for k in ks}
    agg.update({f"recall@{k}": [] for k in ks})
    agg.update({f"ndcg@{k}": [] for k in ks})
    agg.update({f"hit@{k}": [] for k in ks})
    mrr = []
    for item in CLINICAL_QUERIES:
        relevant = set(item["relevant"])
        ranked = rank_fn(item["query"])
        mrr.append(reciprocal_rank(ranked, relevant))
        for k in ks:
            agg[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            agg[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            agg[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
            agg[f"hit@{k}"].append(hit_at_k(ranked, relevant, k))
    out = {m: round(float(np.mean(v)), 4) for m, v in agg.items()}
    out["MRR"] = round(float(np.mean(mrr)), 4)
    out["n_queries"] = len(CLINICAL_QUERIES)
    return out


def hybrid_ranker(model_name: str = "all-mpnet-base-v2") -> Callable:
    """
    Reciprocal-rank fusion of the semantic and lexical rankers — the production
    retrieval strategy (see core/hybrid_retrieval.py for why RRF and not score
    blending). Built from the same in-memory corpus as the other two rankers so
    the three are compared on identical ground.
    """
    from core.hybrid_retrieval import reciprocal_rank_fusion

    sem = semantic_ranker(model_name)
    lex = lexical_ranker()

    def rank(query: str) -> List[str]:
        fused = reciprocal_rank_fusion([sem(query), lex(query)])
        return [doc_id for doc_id, _ in fused]

    return rank


def evaluate_by_superclass(rank_fn: Callable, k: int = 5) -> Dict[str, Dict]:
    """Per-superclass recall@k and MRR, so a class that retrieves badly is
    visible instead of being averaged away by the rest of the benchmark."""
    buckets: Dict[str, List[Dict]] = {}
    for item in CLINICAL_QUERIES:
        buckets.setdefault(item.get("superclass", "legacy"), []).append(item)

    out = {}
    for cls, items in buckets.items():
        rec, mrr = [], []
        for item in items:
            relevant = set(item["relevant"])
            ranked = rank_fn(item["query"])
            rec.append(recall_at_k(ranked, relevant, k))
            mrr.append(reciprocal_rank(ranked, relevant))
        out[cls] = {
            "n":            len(items),
            f"recall@{k}":  round(float(np.mean(rec)), 4),
            "MRR":          round(float(np.mean(mrr)), 4),
        }
    return out


def _validate_benchmark() -> None:
    """
    Every labelled id must exist in the corpus. A typo would make that query
    permanently unanswerable and silently deflate every metric, which would
    look like a retrieval failure rather than a benchmark bug.
    """
    kb_ids = {d["id"] for d in MOCK_KNOWLEDGE_BASE}
    missing = {
        item["query"][:60]: [r for r in item["relevant"] if r not in kb_ids]
        for item in CLINICAL_QUERIES
        if any(r not in kb_ids for r in item["relevant"])
    }
    if missing:
        raise ValueError(f"Benchmark references unknown document ids: {missing}")


def run_rag_evaluation(include_semantic: bool = True) -> Dict:
    _validate_benchmark()
    results = {}
    per_class = {}
    print(f"[RAG-Eval] corpus: {len(MOCK_KNOWLEDGE_BASE)} documents | "
          f"benchmark: {len(CLINICAL_QUERIES)} labelled queries")
    print("[RAG-Eval] Lexical TF-IDF baseline...")
    lex = lexical_ranker()
    results["lexical_tfidf"] = evaluate_retriever(lex)
    per_class["lexical_tfidf"] = evaluate_by_superclass(lex)
    if include_semantic:
        try:
            print("[RAG-Eval] Semantic Sentence-BERT (all-mpnet-base-v2)...")
            sem = semantic_ranker()
            results["semantic_sbert"] = evaluate_retriever(sem)
            per_class["semantic_sbert"] = evaluate_by_superclass(sem)

            print("[RAG-Eval] Hybrid RRF (semantic + lexical)...")
            hyb = hybrid_ranker()
            results["hybrid_rrf"] = evaluate_retriever(hyb)
            per_class["hybrid_rrf"] = evaluate_by_superclass(hyb)
        except Exception as e:
            print(f"[RAG-Eval] Semantic retriever unavailable ({e}) — lexical only.")
    _print(results)
    _print_per_class(per_class)
    results["per_superclass"] = per_class
    return results


def _print_per_class(per_class: Dict[str, Dict]) -> None:
    if not per_class:
        return
    order = ["NORM", "MI", "STTC", "CD", "HYP", "legacy"]
    retrievers = list(per_class)
    print("\n  RECALL@5 BY PTB-XL SUPERCLASS")
    print("  " + "-" * 68)
    print("  {:10s}{:>5s}".format("class", "n") + "".join(f"{r:>20s}" for r in retrievers))
    for cls in order:
        if cls not in next(iter(per_class.values())):
            continue
        n = per_class[retrievers[0]][cls]["n"]
        row = "  {:10s}{:>5d}".format(cls, n)
        for r in retrievers:
            row += f"{per_class[r][cls]['recall@5']:>20.4f}"
        print(row)
    print("  " + "-" * 68)


def _print(results: Dict) -> None:
    metrics = ["MRR", "recall@1", "recall@3", "recall@5",
               "precision@1", "ndcg@3", "ndcg@5", "hit@3"]
    print("\n" + "=" * 78)
    print("  RETRIEVAL QUALITY (labelled clinical benchmark, n="
          f"{next(iter(results.values()))['n_queries']} queries)")
    print("=" * 78)
    print("  {:14s}".format("metric") + "".join(f"{name:>18s}" for name in results))
    for m in metrics:
        row = "  {:14s}".format(m) + "".join(f"{results[r].get(m, 0):>18.4f}" for r in results)
        print(row)
    print("=" * 78)


if __name__ == "__main__":
    run_rag_evaluation()
