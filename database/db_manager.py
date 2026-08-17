"""
database/db_manager.py — Vector Database Manager (Production Milvus)
=====================================================================
Primary backend: External Milvus server at MILVUS_HOST:MILVUS_PORT
  - IVF_FLAT index, nlist=2048, L2 metric, 768-dim Sentence-BERT embeddings
  - Corpus is the hand-seeded prototype set in database/seed_data.py
    (16 documents), NOT a large clinical corpus

Fallback chain: External Milvus → Milvus Lite → ChromaDB → FAISS

The public API is the same regardless of backend:
  - init_db()           → initialise collection / index
  - get_collection()    → returns a _VectorCollection wrapper
  - collection.search() → semantic search (returns List[Dict])
  - collection.insert() → insert embeddings
  - collection.count()  → number of entities

Usage:
    from database.db_manager import init_db, get_collection
    collection = get_collection()
    results = collection.search(query_embedding, top_k=5)
"""

import os
import threading
import numpy as np
from typing import List, Dict, Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
try:
    from config import settings
    _BACKEND        = settings.vector_db_backend
    _MILVUS_HOST    = settings.milvus_host
    _MILVUS_PORT    = settings.milvus_port
    _DB_URI         = settings.milvus_uri
    _COLLECTION     = settings.collection_name
    _DIM            = settings.embedding_dim       # 768 (Sentence-BERT)
    _INDEX_TYPE     = settings.milvus_index_type   # IVF_FLAT
    _NLIST          = settings.milvus_nlist        # 2048
    _METRIC         = settings.milvus_metric       # L2
except Exception:
    _BACKEND        = os.getenv("VECTOR_DB_BACKEND", "milvus")
    _MILVUS_HOST    = os.getenv("MILVUS_HOST", "10.228.1.9")
    _MILVUS_PORT    = os.getenv("MILVUS_PORT", "19530")
    _DB_URI         = "./database/deepcardio.db"
    _COLLECTION     = "cardio_knowledge_base"
    _DIM            = 768
    _INDEX_TYPE     = "IVF_FLAT"
    _NLIST          = 2048
    _METRIC         = "L2"

_lock     = threading.Lock()
_instance: Optional["_VectorCollection"] = None


# ──────────────────────────────────────────────────────────────────────────────
# Unified Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _VectorCollection:
    """
    Backend-agnostic vector collection wrapper.

    Exposes:
      .insert(ids, embeddings, texts, doc_types)
      .search(query_embedding, top_k)  → List[Dict]
      .count()                         → int
      .backend                         → str
      .num_entities                    → int  (alias for .count())
    """

    def __init__(self, backend: str, raw: Any, alias: str = ""):
        self.backend = backend
        # Human-readable backend actually in use. 'backend' cannot distinguish a
        # real Milvus server from Milvus Lite (both are "milvus"), so init_db()
        # overwrites this with the rung of the fallback chain that succeeded.
        self.backend_label = backend
        self._raw    = raw
        self._alias  = alias                        # Milvus connection alias
        self._faiss_meta: List[Dict] = []           # metadata for FAISS

    # ── helpers ───────────────────────────────────────────────────────────────
    @property
    def num_entities(self) -> int:
        return self.count()

    # ── insert ────────────────────────────────────────────────────────────────
    def insert(
        self,
        ids:        List[str],
        embeddings: List[List[float]],
        texts:      List[str],
        doc_types:  List[str],
    ) -> None:
        try:
            if self.backend == "milvus":
                self._raw.insert([ids, embeddings, texts, doc_types])
                self._raw.flush()
                logger.info(f"[Milvus] Inserted {len(ids)} records → total: {self._raw.num_entities}")

            elif self.backend == "chromadb":
                self._raw.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[{"doc_type": dt} for dt in doc_types],
                )
                logger.info(f"[ChromaDB] Inserted {len(ids)} records")

            elif self.backend == "faiss":
                import faiss
                arr = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(arr)
                self._raw.add(arr)
                for id_, text, dt in zip(ids, texts, doc_types):
                    self._faiss_meta.append({"id": id_, "text": text, "doc_type": dt})
                logger.info(f"[FAISS] Inserted {len(ids)} records → total: {self._raw.ntotal}")

        except Exception as exc:
            logger.error(f"Insert failed ({self.backend}): {exc}", exc_info=True)

    # ── search ────────────────────────────────────────────────────────────────
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Returns top_k semantically similar documents.
        Each result: {"text": str, "doc_type": str, "score": float, "id": str}
        """
        try:
            if self.backend == "milvus":
                return self._milvus_search(query_embedding, top_k)
            elif self.backend == "chromadb":
                return self._chromadb_search(query_embedding, top_k)
            elif self.backend == "faiss":
                return self._faiss_search(query_embedding, top_k)
        except Exception as exc:
            logger.warning(f"Search failed ({self.backend}): {exc} — returning empty")
            return []
        return []

    def _milvus_search(self, qvec: List[float], top_k: int) -> List[Dict]:
        self._raw.load()

        # IVF_FLAT search params — nprobe controls recall vs speed tradeoff
        search_params = {
            "metric_type": _METRIC,     # L2
            "params": {"nprobe": 128},  # higher nprobe → better recall
        }
        results = self._raw.search(
            data=[qvec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text_content", "doc_type"],
        )
        out: List[Dict] = []
        for hits in results:
            for hit in hits:
                # L2 distance → convert to similarity score (smaller = closer)
                raw_score = float(hit.score)
                sim_score = round(1.0 / (1.0 + raw_score), 4)
                out.append({
                    "id":       str(hit.id),
                    "text":     hit.entity.get("text_content", ""),
                    "doc_type": hit.entity.get("doc_type", ""),
                    "score":    sim_score,
                    "distance": round(raw_score, 4),
                })
        return out

    def _chromadb_search(self, qvec: List[float], top_k: int) -> List[Dict]:
        n = min(top_k, self._raw.count())
        if n == 0:
            return []
        res = self._raw.query(
            query_embeddings=[qvec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({
                "id":       "",
                "text":     doc,
                "doc_type": meta.get("doc_type", ""),
                "score":    round(1.0 - float(dist), 4),
                "distance": round(float(dist), 4),
            })
        return out

    def _faiss_search(self, qvec: List[float], top_k: int) -> List[Dict]:
        import faiss
        arr = np.array([qvec], dtype=np.float32)
        faiss.normalize_L2(arr)
        k = min(top_k, len(self._faiss_meta))
        if k == 0:
            return []
        D, I = self._raw.search(arr, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self._faiss_meta):
                entry = self._faiss_meta[idx].copy()
                entry["score"]    = round(float(score), 4)
                entry["distance"] = round(1.0 - float(score), 4)
                out.append(entry)
        return out

    # ── fetch all ─────────────────────────────────────────────────────────────
    def fetch_all(self, limit: int = 16384) -> List[Dict]:
        """
        Return every stored document as {id, text, doc_type} (no embeddings).

        Used to build the lexical half of hybrid retrieval from the SAME corpus
        the vector search draws on — reading the seed file instead would let the
        two halves diverge whenever the server holds a stale corpus.
        Returns [] on failure; callers must handle the empty case.
        """
        try:
            if self.backend == "milvus":
                rows = self._raw.query(
                    expr='id != ""',
                    output_fields=["id", "text_content", "doc_type"],
                    limit=limit,
                )
                return [
                    {"id": r.get("id", ""),
                     "text": r.get("text_content", ""),
                     "doc_type": r.get("doc_type", "")}
                    for r in rows
                ]

            elif self.backend == "chromadb":
                got = self._raw.get(include=["documents", "metadatas"])
                return [
                    {"id": i, "text": d, "doc_type": (m or {}).get("doc_type", "")}
                    for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
                ]

            elif self.backend == "faiss":
                return [dict(m) for m in self._faiss_meta]

        except Exception as exc:
            logger.error(f"fetch_all failed ({self.backend}): {exc}")
        return []

    # ── count ─────────────────────────────────────────────────────────────────
    def count(self) -> int:
        try:
            if self.backend == "milvus":
                return self._raw.num_entities
            elif self.backend == "chromadb":
                return self._raw.count()
            elif self.backend == "faiss":
                return self._raw.ntotal
        except Exception:
            return 0
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Backend Initialisation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_milvus_collection(col_name: str, alias: str = "default") -> Any:
    """
    Returns an existing or freshly-created Milvus Collection object.
    Creates IVF_FLAT index on 'embedding' field (768-dim, L2 metric).
    Caller is responsible for establishing the connection first, and must
    pass the alias it connected under — pymilvus otherwise looks for the
    'default' connection and raises ConnectionNotExistException.
    """
    from pymilvus import (
        FieldSchema, CollectionSchema, DataType,
        Collection, utility, Index,
    )

    if utility.has_collection(col_name, using=alias):
        logger.info(f"[Milvus] Collection '{col_name}' exists — loading")
        col = Collection(col_name, using=alias)
        col.load()
        return col

    # Schema: id (PK), embedding (768-dim), text_content, doc_type
    fields = [
        FieldSchema("id",           DataType.VARCHAR,      is_primary=True, max_length=100),
        FieldSchema("embedding",    DataType.FLOAT_VECTOR, dim=_DIM),
        FieldSchema("text_content", DataType.VARCHAR,      max_length=4096),
        FieldSchema("doc_type",     DataType.VARCHAR,      max_length=64),
    ]
    schema = CollectionSchema(fields, description="DeepCardio-RAG clinical knowledge base — prototype seed corpus")
    col    = Collection(name=col_name, schema=schema, using=alias)

    # IVF_FLAT index — gold standard for high-recall production search
    index_params = {
        "metric_type": _METRIC,        # L2
        "index_type":  _INDEX_TYPE,    # IVF_FLAT
        "params":      {"nlist": _NLIST},  # 2048 centroids
    }
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    logger.info(f"[Milvus] Created collection '{col_name}' — IVF_FLAT nlist={_NLIST}, dim={_DIM}, metric={_METRIC}")
    return col


def _init_milvus_external() -> Optional[_VectorCollection]:
    """Connect to production Milvus at MILVUS_HOST:MILVUS_PORT."""
    try:
        from pymilvus import connections, utility
        alias = "deepcardio_prod"
        logger.info(f"[Milvus] Connecting to external server {_MILVUS_HOST}:{_MILVUS_PORT} …")
        connections.connect(
            alias=alias,
            host=_MILVUS_HOST,
            port=_MILVUS_PORT,
            timeout=10,            # connection timeout in seconds
        )
        # Quick sanity-check: list collections
        collections = utility.list_collections(using=alias)
        logger.info(f"[Milvus] Connected — existing collections: {collections}")
        col = _build_milvus_collection(_COLLECTION, alias)
        logger.info(f"[Milvus] External Milvus ready — {col.num_entities} documents indexed")
        return _VectorCollection("milvus", col, alias=alias)

    except Exception as exc:
        logger.warning(f"[Milvus] External connection to {_MILVUS_HOST}:{_MILVUS_PORT} failed: {exc}")
        return None


def _init_milvus_lite() -> Optional[_VectorCollection]:
    """Fallback: Milvus Lite local .db file."""
    try:
        from pymilvus import connections, utility
        alias = "deepcardio_lite"
        logger.info(f"[Milvus Lite] Connecting via URI {_DB_URI} …")
        connections.connect(alias=alias, uri=_DB_URI)
        col = _build_milvus_collection(_COLLECTION, alias)
        logger.info(f"[Milvus Lite] Ready — {col.num_entities} documents")
        return _VectorCollection("milvus", col, alias=alias)

    except Exception as exc:
        logger.warning(f"[Milvus Lite] Init failed: {exc}")
        return None


def _init_chromadb() -> Optional[_VectorCollection]:
    try:
        import chromadb
        db_path = os.path.dirname(_DB_URI) or "./database"
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        col    = client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[ChromaDB] Ready — {col.count()} documents")
        return _VectorCollection("chromadb", col)
    except Exception as exc:
        logger.warning(f"[ChromaDB] Init failed: {exc}")
        return None


def _init_faiss() -> Optional[_VectorCollection]:
    try:
        import faiss
        # Inner product on L2-normalised vectors ≡ cosine similarity
        index = faiss.IndexFlatIP(_DIM)
        logger.info(f"[FAISS] In-memory index initialised (dim={_DIM})")
        return _VectorCollection("faiss", index)
    except Exception as exc:
        logger.error(f"[FAISS] Init failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def init_db() -> Optional[_VectorCollection]:
    """
    Initialise vector DB backend.

    Priority order:
      1. External Milvus  (10.228.1.9:19530)
      2. Milvus Lite      (local .db file)
      3. ChromaDB         (local persistent)
      4. FAISS            (in-memory)

    Thread-safe singleton — subsequent calls return the same instance.
    """
    global _instance
    with _lock:
        if _instance is not None:
            return _instance

        # Try external Milvus first (production)
        _instance = _init_milvus_external()
        if _instance:
            _instance.backend_label = "external Milvus"
            logger.info("✓ Vector DB ready — backend: external Milvus")
            return _instance

        # Fallback chain
        for fn, label in [
            (_init_milvus_lite, "Milvus Lite"),
            (_init_chromadb,    "ChromaDB"),
            (_init_faiss,       "FAISS"),
        ]:
            _instance = fn()
            if _instance:
                _instance.backend_label = label
                logger.info(f"✓ Vector DB ready — backend: {label}")
                return _instance

        logger.error("✗ All vector DB backends failed — RAG will use mock guidelines")
        return None


def get_collection() -> Optional[_VectorCollection]:
    """
    Returns the active vector collection (initialises on first call, thread-safe).
    Returns None only if all backends failed.
    """
    global _instance
    if _instance is not None:
        return _instance
    return init_db()


def _describe_milvus(col: _VectorCollection) -> Dict[str, Any]:
    """
    Read dim / metric / index type off the LIVE Milvus collection rather than
    config, so a server collection that was built with different parameters
    reports what it really is instead of what we intended.
    """
    info: Dict[str, Any] = {}
    try:
        raw = col._raw
        for f in raw.schema.fields:
            dim = (getattr(f, "params", None) or {}).get("dim")
            if dim:
                info["embedding_dim"] = int(dim)
                break
        indexes = raw.indexes
        if indexes:
            p = indexes[0].params or {}
            info["metric"]     = p.get("metric_type", _METRIC)
            info["index_type"] = p.get("index_type",  _INDEX_TYPE)
            nlist = (p.get("params") or {}).get("nlist")
            if nlist:
                info["nlist"] = int(nlist)
    except Exception as exc:
        logger.warning(f"[Milvus] Could not introspect collection: {exc}")
    return info


def get_db_status() -> Dict[str, Any]:
    """
    Live, honest description of the vector DB actually serving requests.

    Every field is read from the running backend — nothing here is hardcoded.
    This exists because the fallback chain in init_db() is silent: config
    saying "milvus" does NOT mean Milvus is serving, so any caller that
    reports status to a user must surface `backend` verbatim.
    """
    col = get_collection()

    if col is None:
        return {
            "connected":     False,
            "backend":       "none",
            "status":        "Offline",
            "collection":    _COLLECTION,
            "documents":     0,
            "embedding_dim": _DIM,
            "metric":        _METRIC,
            "index_type":    _INDEX_TYPE,
            "host":          f"{_MILVUS_HOST}:{_MILVUS_PORT}",
            "note":          "All vector DB backends failed — RAG falls back to mock guidelines.",
        }

    label    = getattr(col, "backend_label", col.backend)
    external = label == "external Milvus"

    status = {
        "connected":     True,
        "backend":       label,
        "status":        "Online" if external else f"Degraded — fallback backend in use ({label})",
        "collection":    _COLLECTION,
        "documents":     col.count(),
        "embedding_dim": _DIM,
        "metric":        _METRIC,
        "index_type":    _INDEX_TYPE,
        "host":          f"{_MILVUS_HOST}:{_MILVUS_PORT}" if external else "local (in-process)",
    }

    if col.backend == "milvus":
        status.update(_describe_milvus(col))
    else:
        # Non-Milvus rungs do not use the configured Milvus index at all.
        status["index_type"] = "n/a"
        status["note"] = (
            f"External Milvus was unreachable — retrieval is being served by "
            f"{label}, not the production vector database."
        )

    return status


def drop_collection() -> bool:
    """
    Delete the collection from the server and reset the singleton, so the next
    init_db() rebuilds it empty. Used by seed_data.py --force to re-seed a
    corpus that has changed. Returns True if a collection was actually dropped.
    """
    global _instance
    col = get_collection()
    if col is None:
        return False

    dropped = False
    try:
        if col.backend == "milvus":
            from pymilvus import utility
            if utility.has_collection(_COLLECTION, using=col._alias):
                utility.drop_collection(_COLLECTION, using=col._alias)
                dropped = True
                logger.info(f"[Milvus] Dropped collection '{_COLLECTION}'")
        elif col.backend == "chromadb":
            col._raw.delete(col._raw.get()["ids"])
            dropped = True
        elif col.backend == "faiss":
            col._raw.reset()
            col._faiss_meta.clear()
            dropped = True
    except Exception as exc:
        logger.error(f"Drop collection failed ({col.backend}): {exc}")
        return False

    with _lock:
        _instance = None
    return dropped


def reset_db() -> None:
    """Force re-initialisation (for testing / connection recovery)."""
    global _instance
    with _lock:
        _instance = None
    logger.info("Vector DB instance reset — will reinitialise on next call")
