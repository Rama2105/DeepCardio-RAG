"""
core/hybrid_retrieval.py — Lexical + semantic fusion for the clinical RAG store
==============================================================================
Why this exists (measured, not assumed). On the 38-query clinical benchmark over
the 67-document corpus, the semantic retriever (Sentence-BERT all-mpnet-base-v2)
was beaten decisively by a plain TF-IDF baseline — MRR 0.689 vs 0.921 — and
collapsed on Conduction Disturbance, recall@5 0.417 vs 1.000. Two diagnosed
causes:

  * Fine-grained intra-class confusability. The AV-block documents are mutually
    near-identical to a dense embedder (within-CD mean pairwise cosine 0.643 vs
    0.451 corpus-wide), so first-degree AV block outranks Wenckebach on a
    Wenckebach query.
  * Notation-heavy queries. "rsR prime", "M-shaped complex in V1" embed poorly,
    while TF-IDF matches those rare tokens exactly.

Crucially neither retriever dominates: semantic wins on MI (0.900 vs 0.800) and
HYP (0.806 vs 0.778), lexical wins on CD and STTC. That is the textbook case for
fusion rather than for picking one.

Method: Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = Σ_i  w_i / (k + rank_i(d))

RRF fuses RANKS, not scores. That is the deciding property here, because the two
retrievers return incomparable quantities — Milvus L2 *distances* (smaller is
better, unbounded) versus TF-IDF cosine *similarity* (larger is better, [0,1]).
Any score-level fusion would need a normalisation whose calibration drifts with
corpus and query distribution; ranks need none.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

# Cormack et al.'s constant. Large k damps the influence of the very top ranks,
# so one retriever being confidently wrong cannot dominate the fused list.
DEFAULT_RRF_K = 60

# Equal weights are the LITERATURE DEFAULT and are deliberately left untuned.
#
# A sensitivity sweep over the 38-query benchmark (semantic:lexical weight, k=60):
#     1:1  MRR 0.818  P@1 0.684  recall@5 0.939   <- default, reported
#     1:2  MRR 0.896  P@1 0.816  recall@5 0.952
#     1:3  MRR 0.941  P@1 0.895  recall@5 0.952
#     0:1  MRR 0.921  P@1 0.868  recall@5 0.899   (pure lexical)
#     1:0  MRR 0.689  P@1 0.553  recall@5 0.811   (pure semantic)
#
# A lexical tilt clearly helps, and the trend is smooth rather than a spike, so
# the effect is probably real. But that benchmark has 38 queries and NO held-out
# split — adopting 1:3 because it topped the table would be fitting to the test
# set, and every number then quoted from it would be inflated. So the shipped
# default stays 1:1, which is what the reported figures come from. Validating a
# lexical tilt needs a held-out query set; the weights are exposed as parameters
# for exactly that experiment.


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Fuse several ranked id lists into one.

    Args:
      rankings : one ranked list of document ids per retriever, best first.
      k        : rank-damping constant.
      weights  : optional per-retriever weight; defaults to 1.0 each.

    Returns:
      [(doc_id, fused_score), ...] best first.

    A document absent from a retriever's list simply contributes nothing from
    that retriever — no imputed rank, no penalty. This keeps the fusion valid
    when the retrievers return lists of different lengths.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights ({len(weights)}) must match rankings ({len(rankings)})")

    scores: Dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda kv: -kv[1])


class LexicalIndex:
    """
    TF-IDF index over a fixed document set.

    Deliberately built from the SAME documents the semantic side can return.
    Indexing a different set (e.g. the seed file while Milvus holds a stale
    corpus) would let fusion rank documents the vector store cannot serve.
    """

    def __init__(self, ids: Sequence[str], texts: Sequence[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        self.ids = list(ids)
        self.texts = list(texts)
        self._by_id = dict(zip(self.ids, self.texts))
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.texts)

    def __len__(self) -> int:
        return len(self.ids)

    def text_for(self, doc_id: str) -> str:
        return self._by_id.get(doc_id, "")

    def rank(self, query: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return [(doc_id, cosine_similarity), ...] best first."""
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        order = np.argsort(-sims)
        if top_n is not None:
            order = order[:top_n]
        return [(self.ids[i], float(sims[i])) for i in order]


def fuse_semantic_lexical(
    semantic_ids: Sequence[str],
    lexical_ids: Sequence[str],
    top_k: int,
    k: int = DEFAULT_RRF_K,
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    """Convenience wrapper: fuse one semantic and one lexical ranking."""
    fused = reciprocal_rank_fusion(
        [list(semantic_ids), list(lexical_ids)],
        k=k,
        weights=[semantic_weight, lexical_weight],
    )
    return fused[:top_k]
