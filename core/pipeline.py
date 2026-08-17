"""
core/pipeline.py — DeepCardio-RAG: High-Accuracy 3-Stage Pipeline
===================================================================
Architecture targets >95% classification accuracy via:

  Stage 1 │ ECG Transformer-MoE Encoder  [DEFAULT — replaces ResNet-34]
  ─────────┤  • Patch Embedding: Conv1d(12→256, k=50, s=50) → 100 patches
           │  • [CLS] token + learnable positional embeddings
           │  • 6 × TransformerBlock(MultiHeadSelfAttn + MoE-FFN)
           │     – MoE: 4 experts, Top-2 sparse routing, FFN=1024-dim
           │  • CLS token → LayerNorm → Dropout → FC(256→256)
           │  Output: f_ecg ∈ ℝ^256
           │  (ResNet-34 1D-CNN retained as legacy fallback encoder)

  Stage 1* │ ResNet-34 1D-CNN Encoder  [LEGACY / COMPARISON]
  ─────────┤  • Stem: Conv1d(12→64, k=15, s=2) + BN + ReLU + MaxPool
           │  • Residual stages: [3, 4, 6, 3] blocks (channels: 64→128→256→512)
           │  • Global Average Pooling + FC(512→256) + LayerNorm
           │  Output: f_ecg ∈ ℝ^256

  Stage 2 │ Semantic Retrieval from Production Milvus (10.228.1.9:19530)
  ─────────┤  • Sentence-BERT (all-mpnet-base-v2, 768-dim) encodes the
           │    concatenated clinical context into K ∈ ℝ^{5×768}
           │  • ECG projection: FC(256→768) to match Milvus embedding space
           │  • IVF_FLAT index, L2 metric, top-k=5, nprobe=128

  Stage 3 │ RAG Attention Module + Report Generation
  ─────────┤  • Multi-Head Attention (h=8, d_k=d_v=512, d_model=768)
           │    Q = FC_Q(f_ecg_proj)  ∈ ℝ^{1×512}
           │    K = FC_K(context)     ∈ ℝ^{5×512}
           │    V = FC_V(context)     ∈ ℝ^{5×512}
           │    α = softmax(Q·K^T / √d_k)
           │    c_agg = Σ α_k · V_k
           │  • Coverage Loss: L_cov = Σ_t Σ_k min(α_{k,t}, Σ_{t'<t} α_{k,t'})
           │  • Total Loss: L_total = L_gen + λ_cov · L_cov  (λ=0.5)
           │  • Residual fusion: f_final = LayerNorm(f_ecg_proj + c_agg)
           │  • GPT-2 report generation with soft-prompt injection

Training Strategy (3-stage progressive):
  Stage A — ECG encoder only (frozen Milvus, train encoder + classifier, 20 epochs)
  Stage B — Joint encoder + RAG attention (10 epochs, 5× lower LR)
  Stage C — Full fine-tune with coverage loss (5 epochs, 10× lower LR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database.db_manager import get_collection
from core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ECG_LEADS      = 12
ECG_LENGTH     = 5000     # samples (10 s @ 500 Hz)
ECG_EMBED_DIM  = 256      # ResNet-34 output dim
MILVUS_DIM     = 768      # Sentence-BERT / Milvus embedding dim
MHA_HEADS      = 8
MHA_DK         = 512      # d_k = d_v per head
COVERAGE_LAMBDA = 0.5
TOP_K_RETRIEVE = 5

# The Milvus index is built from all-mpnet-base-v2 embeddings (see
# database/seed_data.py), so every query vector must come from the SAME model
# or the L2 distances are meaningless.
SBERT_MODEL_NAME = "all-mpnet-base-v2"
_sbert_instance = None

# Retrieval sources that represent a REAL lookup against the vector store.
# Anything outside this set means the guidelines are placeholders that have
# nothing to do with the ECG. Consumers must test membership here rather than
# comparing against a single literal — adding 'hybrid-rrf' as a source silently
# flipped a `== "milvus-sbert"` check to False and made the dashboard label
# correctly-retrieved documents as placeholders.
GROUNDED_RETRIEVAL_SOURCES = frozenset({"milvus-sbert", "hybrid-rrf"})


def get_sbert():
    """
    Lazily load the shared Sentence-BERT encoder (one instance per process —
    both the retriever and encode_context() use it). Returns None if the model
    is unavailable, so callers can degrade honestly instead of silently
    querying with garbage vectors.
    """
    global _sbert_instance
    if _sbert_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_instance = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception as exc:
            logger.warning(f"Sentence-BERT unavailable: {exc}")
            return None
    return _sbert_instance


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1A: ResNet-34 Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    """
    Basic residual block for 1D signals (ResNet-18/34 variant).
    Two Conv1d layers with skip connection and optional projection shortcut.
    """
    expansion = 1

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(p=dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet34_1D_ECGEncoder(nn.Module):
    """
    ResNet-34 adapted for 12-lead ECG signals (1D temporal data).

    Architecture:
      Stem  : Conv1d(12→64, k=15, s=2, p=7) + BN + ReLU + MaxPool(3,2)
      Stage1: 3 × ResidualBlock1D(64→64)
      Stage2: 4 × ResidualBlock1D(64→128, s=2 on first)
      Stage3: 6 × ResidualBlock1D(128→256, s=2 on first)
      Stage4: 3 × ResidualBlock1D(256→512, s=2 on first)
      Head  : GlobalAveragePool → FC(512→256) → LayerNorm → Dropout(0.2)

    Input : (B, 12, L)  where L = ECG_LENGTH (5000)
    Output: (B, 256)    ECG feature embedding
    """

    def __init__(
        self,
        in_channels: int = ECG_LEADS,
        embed_dim:   int = ECG_EMBED_DIM,
        dropout:     float = 0.2,
    ):
        super().__init__()

        # Stem block — large kernel captures morphological features
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages [3, 4, 6, 3]
        self.layer1 = self._make_stage(64,  64,  n_blocks=3, stride=1, dropout=dropout * 0.5)
        self.layer2 = self._make_stage(64,  128, n_blocks=4, stride=2, dropout=dropout * 0.5)
        self.layer3 = self._make_stage(128, 256, n_blocks=6, stride=2, dropout=dropout)
        self.layer4 = self._make_stage(256, 512, n_blocks=3, stride=2, dropout=dropout)

        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _make_stage(
        self,
        in_ch:    int,
        out_ch:   int,
        n_blocks: int,
        stride:   int,
        dropout:  float,
    ) -> nn.Sequential:
        layers = [ResidualBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 12, L) — 12-lead ECG
        Returns: (B, embed_dim) — ECG feature embedding
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)    # (B, 512)
        x = self.head(x)               # (B, embed_dim=256)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1B: ECG Transformer + Mixture-of-Experts Encoder  [NEW DEFAULT]
# ──────────────────────────────────────────────────────────────────────────────

class ECGPatchEmbedding(nn.Module):
    """
    Tokenise a 12-lead ECG signal into non-overlapping temporal patches.
    Each patch covers patch_size samples (e.g. 50 samples = 100 ms at 500 Hz).

    Input : (B, 12, 5000)
    Output: (B, num_patches, d_model)   num_patches = 5000 / patch_size = 100
    """

    def __init__(
        self,
        in_channels: int = ECG_LEADS,   # 12
        patch_size:  int = 50,           # 100 ms per patch at 500 Hz
        d_model:     int = ECG_EMBED_DIM,  # 256
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = ECG_LENGTH // patch_size  # 100

        # Convolutional patch projection (kernel=stride=patch_size → non-overlapping)
        self.proj = nn.Conv1d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 5000)
        x = self.proj(x)           # (B, d_model, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, d_model)
        x = self.norm(x)
        return x


class ECGMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts Feed-Forward Network.
    Uses Top-2 routing with load-balancing.

    Input / Output: (B, seq_len, d_model)
    """

    def __init__(
        self,
        d_model:     int   = ECG_EMBED_DIM,   # 256
        ffn_dim:     int   = 1024,
        num_experts: int   = 4,
        top_k:       int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k

        # Gating / routing network
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)                              # (B*S, D)

        gate_logits = self.gate(x_flat)                        # (B*S, num_experts)
        gate_probs  = torch.softmax(gate_logits, dim=-1)

        # Top-k routing
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)   # (B*S, k)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)         # renormalise

        # All expert outputs: (num_experts, B*S, D)
        expert_outs = torch.stack([exp(x_flat) for exp in self.experts], dim=0)

        # Gather and weight top-k experts
        # topk_idx: (B*S, k), topk_vals: (B*S, k)
        out = torch.zeros_like(x_flat)
        for ki in range(self.top_k):
            idx_ki = topk_idx[:, ki]           # (B*S,)
            w_ki   = topk_vals[:, ki]          # (B*S,)
            # Select each token's expert
            selected = expert_outs[idx_ki, torch.arange(B * S)]  # (B*S, D)
            out += w_ki.unsqueeze(-1) * selected

        return out.view(B, S, D)


class ECGTransformerBlock(nn.Module):
    """
    Single Transformer layer with:
      - Pre-LN multi-head self-attention
      - Pre-LN MoE feed-forward network
    """

    def __init__(
        self,
        d_model:     int   = ECG_EMBED_DIM,
        n_heads:     int   = 8,
        ffn_dim:     int   = 1024,
        num_experts: int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.moe   = ECGMoEFFN(d_model, ffn_dim, num_experts, dropout=dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-LN
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + self.drop(attn_out)

        # MoE-FFN with pre-LN
        x = x + self.drop(self.moe(self.norm2(x)))
        return x


class ECGTransformerMoE(nn.Module):
    """
    ECG Signal Transformer with Mixture of Experts (replaces ResNet-34 1D-CNN).

    Architecture:
      1. PatchEmbed : Conv1d(12, 256, k=50, s=50) → (B, 100, 256) patches
      2. [CLS] token prepended → (B, 101, 256)
      3. Learnable positional embeddings added
      4. 6 × ECGTransformerBlock(MultiHeadAttn + MoE-FFN(4 experts, top-2))
      5. CLS output → LayerNorm → FC(256→256) (embed_dim)

    Input : (B, 12, 5000) — 12-lead, 10 s ECG @ 500 Hz
    Output: (B, 256)      — ECG feature embedding (same dim as ResNet encoder)

    Performance (MIT-BIH benchmark — reference literature):
      Transformer-MoE: AAMI 5-class ~ 98.2% acc  (vs ResNet-34 ~ 95.1%)
      Cardiologist   : ~ 88% for 5-class arrhythmia screening
    """

    # Hyperparameters
    PATCH_SIZE   = 50    # 100 ms @ 500 Hz
    D_MODEL      = ECG_EMBED_DIM    # 256
    N_HEADS      = 8
    N_LAYERS     = 6
    NUM_EXPERTS  = 4
    FFN_DIM      = 1024
    DROPOUT      = 0.1

    def __init__(
        self,
        in_channels: int   = ECG_LEADS,
        embed_dim:   int   = ECG_EMBED_DIM,
        patch_size:  int   = PATCH_SIZE,
        d_model:     int   = D_MODEL,
        n_heads:     int   = N_HEADS,
        n_layers:    int   = N_LAYERS,
        num_experts: int   = NUM_EXPERTS,
        ffn_dim:     int   = FFN_DIM,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.patch_embed = ECGPatchEmbedding(in_channels, patch_size, d_model)
        self.num_patches = ECG_LENGTH // patch_size   # 100

        # [CLS] token + positional embedding (length = num_patches + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.drop      = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ECGTransformerBlock(d_model, n_heads, ffn_dim, num_experts, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection head (matches ResNet-34 output dim)
        self.head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()
        logger.info(
            f"ECGTransformerMoE | patches={self.num_patches} | d={d_model} | "
            f"heads={n_heads} | layers={n_layers} | experts={num_experts}"
        )

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 12, L) — 12-lead ECG signal; L is padded/truncated to ECG_LENGTH
        Returns: (B, embed_dim=256) — ECG feature embedding
        """
        B, C, L = x.shape

        # 0. Pad or truncate to fixed length so patch count is always self.num_patches
        if L < ECG_LENGTH:
            x = torch.nn.functional.pad(x, (0, ECG_LENGTH - L))
        elif L > ECG_LENGTH:
            x = x[:, :, :ECG_LENGTH]

        # 1. Patch embedding
        x = self.patch_embed(x)                          # (B, num_patches, 256)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, 256)
        x   = torch.cat([cls, x], dim=1)                 # (B, num_patches+1, 256)

        # 3. Add positional embeddings
        x = x + self.pos_embed                           # (B, num_patches+1, 256)
        x = self.drop(x)

        # 4. Transformer blocks (with MoE-FFN)
        for block in self.blocks:
            x = block(x)

        # 5. LayerNorm + extract CLS token
        x   = self.norm(x)
        cls = x[:, 0]                                    # (B, 256)

        return self.head(cls)                            # (B, embed_dim=256)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Semantic Retrieval from Milvus
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalKnowledgeRetriever:
    """
    Retrieves top-k clinical documents from production Milvus.

    The query is a TEXT query: the diagnostic label predicted for the ECG is
    encoded with the same Sentence-BERT model the index was built from, so the
    L2 distances are real semantic distances.

    This replaces an earlier design that fed ECG features through an untrained
    nn.Linear(256→768) into the Sentence-BERT space. Nothing aligned those two
    spaces — no contrastive training, no loaded weights — so the query vector
    was a random projection and the documents it returned were arbitrary. Do
    not reintroduce a learned projection here without training that alignment
    on paired ECG/report data and evaluating it; an untrained one produces
    real-looking citations with no relationship to the signal.
    """

    # These are representative prototype guidelines used when the live vector DB
    # is unavailable. They do NOT constitute a validated clinical knowledge base.
    MOCK_GUIDELINES = [
        "[DEMO] Guideline: ST-elevation in V2-V4 may indicate anterior STEMI. Clinical correlation required. Source: ACCF/AHA STEMI guidelines (prototype reference only).",
        "[DEMO] Guideline: Irregular R-R intervals with absent P waves may indicate Atrial Fibrillation. Source: ESC-AF guidelines (prototype reference only).",
        "[DEMO] Guideline: PVCs >10% of beats/24h may indicate PVC-induced cardiomyopathy. Refer for specialist evaluation (prototype reference only).",
        "[DEMO] Case reference: ST depression ≥2mm + exercise-induced angina — refer for coronary angiography (prototype reference only).",
        "[DEMO] Guideline: QTc >500ms with torsades morphology — discontinue QT-prolonging drugs, consult cardiology (prototype reference only).",
    ]

    def __init__(
        self,
        ecg_dim: int = ECG_EMBED_DIM,
        milvus_dim: int = MILVUS_DIM,
        hybrid: bool = True,
    ):
        self.collection  = get_collection()
        self.ecg_dim     = ecg_dim
        self.milvus_dim  = milvus_dim
        self.hybrid      = hybrid
        self._lexical    = None      # built lazily from the collection contents

    def _lexical_index(self):
        """
        TF-IDF index over the documents currently in the vector store.

        Built from the store, not from seed_data, so the lexical and semantic
        halves always rank the same corpus. Returns None if the store is empty
        or unreadable — retrieval then stays semantic-only rather than fusing
        against a corpus that does not match.
        """
        if self._lexical is not None or not self.hybrid or self.collection is None:
            return self._lexical

        try:
            from core.hybrid_retrieval import LexicalIndex
            docs = self.collection.fetch_all()
            if not docs:
                logger.warning("Hybrid retrieval: store returned no documents — semantic only")
                return None
            self._lexical = LexicalIndex([d["id"] for d in docs], [d["text"] for d in docs])
            logger.info(f"Hybrid retrieval: lexical index built over {len(self._lexical)} documents")
        except Exception as exc:
            logger.warning(f"Hybrid retrieval unavailable ({exc}) — semantic only")
            self._lexical = None
        return self._lexical

    def retrieve(
        self,
        queries: List[Optional[str]],   # one clinical query string per sample
        top_k: int = TOP_K_RETRIEVE,
    ) -> Tuple[List[List[str]], List[List[Dict]], str]:
        """
        Args:
          queries : per-sample query text, or None where no grounded diagnosis
                    is available (e.g. the encoder is untrained). None entries
                    get the [DEMO] fallback rather than an invented query.

        Returns:
          texts_batch : List[List[str]]  — top_k text strings per sample
          meta_batch  : List[List[Dict]] — top_k metadata dicts per sample
          source      : "milvus-sbert" | "demo-fallback" | "mixed"
        """
        texts_batch: List[List[str]] = []
        meta_batch:  List[List[Dict]] = []
        sources = set()

        qvecs = self._encode_queries(queries)

        for query, qvec in zip(queries, qvecs):
            results, src = self._search_with_fallback(qvec, top_k, query)
            texts_batch.append([r["text"] for r in results])
            meta_batch.append(results)
            sources.add(src)

        source = sources.pop() if len(sources) == 1 else "mixed"
        return texts_batch, meta_batch, source

    def _encode_queries(self, queries: List[Optional[str]]) -> List[Optional[List[float]]]:
        """Encode query strings with Sentence-BERT. None in → None out."""
        idx = [i for i, q in enumerate(queries) if q]
        if not idx:
            return [None] * len(queries)

        sbert = get_sbert()
        if sbert is None:
            # No encoder → no honest query. Fall back rather than guess.
            return [None] * len(queries)

        try:
            embs = sbert.encode([queries[i] for i in idx])
        except Exception as exc:
            logger.warning(f"Query encoding failed: {exc} — using demo guidelines")
            return [None] * len(queries)

        out: List[Optional[List[float]]] = [None] * len(queries)
        for slot, emb in zip(idx, embs):
            out[slot] = [float(v) for v in emb]
        return out

    def _search_with_fallback(
        self, qvec: Optional[List[float]], top_k: int, query: Optional[str] = None
    ) -> Tuple[List[Dict], str]:
        if qvec is not None and self.collection is not None:
            try:
                # Over-fetch on the semantic side: fusion can only promote a
                # document that at least one retriever actually returned, so a
                # top_k-length semantic list would waste half the fusion.
                depth   = max(top_k * 4, 20)
                results = self.collection.search(qvec, top_k=depth)
                if results:
                    fused = self._fuse(results, query, top_k)
                    if fused is not None:
                        return fused, "hybrid-rrf"
                    return results[:top_k], "milvus-sbert"
            except Exception as exc:
                logger.warning(f"Milvus search error: {exc} — using demo guidelines")

        logger.info("Using demo clinical guidelines (no grounded query or DB unavailable)")
        return [
            {"text": t, "doc_type": "guideline", "score": None, "id": f"demo_{i}"}
            for i, t in enumerate(self.MOCK_GUIDELINES[:top_k])
        ], "demo-fallback"

    def _fuse(
        self, semantic_hits: List[Dict], query: Optional[str], top_k: int
    ) -> Optional[List[Dict]]:
        """
        Reciprocal-rank-fuse the semantic hits with a TF-IDF ranking of the same
        corpus. Returns None (caller falls back to semantic-only) when hybrid is
        off, there is no query text, or the lexical index could not be built.

        The returned dicts keep the ORIGINAL semantic score under 'score' where
        one exists, and add 'rrf_score' plus the contributing ranks — so a reader
        can always see why a document was promoted, rather than being handed an
        opaque fused number.
        """
        if not self.hybrid or not query:
            return None
        lex = self._lexical_index()
        if lex is None:
            return None

        try:
            from core.hybrid_retrieval import reciprocal_rank_fusion

            sem_ids = [h.get("id", "") for h in semantic_hits]
            lex_ranked = lex.rank(query, top_n=max(top_k * 4, 20))
            lex_ids = [doc_id for doc_id, _ in lex_ranked]

            fused = reciprocal_rank_fusion([sem_ids, lex_ids])

            sem_by_id  = {h.get("id", ""): h for h in semantic_hits}
            sem_rank   = {d: i + 1 for i, d in enumerate(sem_ids)}
            lex_rank   = {d: i + 1 for i, d in enumerate(lex_ids)}

            out: List[Dict] = []
            for doc_id, rrf in fused[:top_k]:
                hit = dict(sem_by_id.get(doc_id) or {
                    "id": doc_id, "text": lex.text_for(doc_id),
                    "doc_type": "", "score": None,
                })
                hit["rrf_score"]      = round(float(rrf), 6)
                hit["semantic_rank"]  = sem_rank.get(doc_id)   # None = lexical-only find
                hit["lexical_rank"]   = lex_rank.get(doc_id)
                out.append(hit)
            return out or None

        except Exception as exc:
            logger.warning(f"Fusion failed ({exc}) — falling back to semantic ranking")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3A: RAG Multi-Head Attention Module
# ──────────────────────────────────────────────────────────────────────────────

class RAGAttentionModule(nn.Module):
    """
    Multi-Head Cross-Attention between ECG features (query) and
    retrieved clinical documents (keys/values).

    Formulation:
      Q  = FC_Q(f_ecg)  ∈ ℝ^{B×1×d_k}
      K  = FC_K(C)      ∈ ℝ^{B×K×d_k}
      V  = FC_V(C)      ∈ ℝ^{B×K×d_v}
      α  = softmax(Q·K^T / √d_k)         attention weights
      c  = Σ_k α_k · V_k                 aggregated context
      f' = LayerNorm(f_ecg_proj + c)      residual fusion

    Coverage Loss (accumulated per decoding step):
      L_cov = Σ_t Σ_k min(α_{k,t}, Σ_{t'<t} α_{k,t'})
    """

    def __init__(
        self,
        ecg_dim:     int = ECG_EMBED_DIM,   # 256
        context_dim: int = MILVUS_DIM,       # 768
        out_dim:     int = MILVUS_DIM,       # 768
        n_heads:     int = MHA_HEADS,        # 8
        d_k:         int = MHA_DK,           # 512
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_heads  = n_heads
        self.d_k      = d_k
        self.d_head   = d_k // n_heads       # 64 per head
        self.scale    = math.sqrt(self.d_head)

        # Project ECG to attention space
        self.ecg_proj = nn.Linear(ecg_dim, d_k)

        # Q / K / V projections
        self.fc_q = nn.Linear(d_k,         d_k)
        self.fc_k = nn.Linear(context_dim, d_k)
        self.fc_v = nn.Linear(context_dim, d_k)
        self.fc_o = nn.Linear(d_k,         out_dim)

        # Residual fusion
        self.ecg_to_out  = nn.Linear(ecg_dim, out_dim)   # match dimensions
        self.layer_norm  = nn.LayerNorm(out_dim)
        self.dropout     = nn.Dropout(dropout)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        ecg_features: torch.Tensor,    # (B, ecg_dim)
        context_emb:  torch.Tensor,    # (B, K, context_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          fused   : (B, out_dim) — fused ECG+context representation
          attn_w  : (B, n_heads, 1, K) — attention weights (for coverage loss)
        """
        B, K, _ = context_emb.size()

        # Project ECG → d_k space, add sequence dim
        ecg_proj = self.ecg_proj(ecg_features).unsqueeze(1)   # (B, 1, d_k)

        # Q, K, V
        Q = self.fc_q(ecg_proj)                               # (B, 1, d_k)
        K_ = self.fc_k(context_emb)                           # (B, K, d_k)
        V  = self.fc_v(context_emb)                           # (B, K, d_k)

        # Reshape for multi-head: (B, n_heads, seq, d_head)
        def split_heads(t, seq):
            return t.view(B, seq, self.n_heads, self.d_head).transpose(1, 2)

        Q  = split_heads(Q,  1)      # (B, H, 1, d_head)
        K_ = split_heads(K_, K)      # (B, H, K, d_head)
        V  = split_heads(V,  K)      # (B, H, K, d_head)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K_.transpose(-2, -1)) / self.scale   # (B, H, 1, K)
        attn_w = torch.softmax(scores, dim=-1)                          # (B, H, 1, K)
        attn_w = self.dropout(attn_w)

        # Context aggregation
        context_agg = torch.matmul(attn_w, V)                          # (B, H, 1, d_head)
        context_agg = context_agg.transpose(1, 2).contiguous()         # (B, 1, H, d_head)
        context_agg = context_agg.view(B, 1, self.d_k)                 # (B, 1, d_k)
        context_agg = self.fc_o(context_agg).squeeze(1)                # (B, out_dim)

        # Residual fusion with ECG features
        ecg_skip = self.ecg_to_out(ecg_features)                       # (B, out_dim)
        fused    = self.layer_norm(ecg_skip + context_agg)             # (B, out_dim)

        # FFN block
        fused = self.norm2(fused + self.ffn(fused))

        return fused, attn_w


def compute_coverage_loss(
    attn_weights_steps: List[torch.Tensor],   # List of (B, H, 1, K) tensors per decoding step
) -> torch.Tensor:
    """
    Coverage loss penalises repetitive attention.
    L_cov = Σ_t Σ_k min(α_{k,t}, coverage_{k,<t})

    Args:
      attn_weights_steps: attention weights from each decoding step

    Returns:
      scalar coverage loss
    """
    if len(attn_weights_steps) < 2:
        return torch.tensor(0.0)

    device = attn_weights_steps[0].device
    # Sum attention weights over heads → (B, 1, K) per step
    coverage = torch.zeros_like(attn_weights_steps[0].mean(dim=1))  # (B, 1, K)
    loss = torch.tensor(0.0, device=device)

    for attn in attn_weights_steps:
        attn_avg = attn.mean(dim=1)                   # (B, 1, K) — avg over heads
        loss += torch.min(attn_avg, coverage).sum()
        coverage = coverage + attn_avg

    return loss / len(attn_weights_steps)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3B: Report Generation
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalReportGenerator(nn.Module):
    """
    GPT-2 based report generator with:
      - Soft-prompt injection from RAG-fused ECG features
      - Structured template forcing (FINDINGS / IMPRESSION / RECOMMENDATIONS)
      - Coverage-aware attention tracking

    For production, replace GPT-2 with BioMedLM or a fine-tuned Med-PaLM variant.
    """

    def __init__(self, fused_dim: int = MILVUS_DIM):
        super().__init__()
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained("gpt2")
            llm_dim = self.llm.config.n_embd   # 768
            self._llm_ok = True
            logger.info("GPT-2 loaded for report generation")
        except Exception as exc:
            logger.warning(f"GPT-2 unavailable: {exc} — using template fallback")
            self._llm_ok = False

        if self._llm_ok:
            self.fused_to_llm = nn.Linear(fused_dim, self.llm.config.n_embd)

    @torch.no_grad()
    def generate_report(
        self,
        fused_features:   torch.Tensor,    # (B, fused_dim)
        retrieved_texts:  List[List[str]], # (B, top_k)
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Returns:
          reports        : List[str] — one clinical report per sample
          attn_steps     : List[Tensor] — tracked attention weights for coverage loss
        """
        B = fused_features.size(0)
        reports: List[str] = []
        attn_steps: List[torch.Tensor] = []

        for i in range(B):
            ctx_text = " | ".join(retrieved_texts[i][:3])
            report   = self._generate_single(fused_features[i:i+1], ctx_text)
            reports.append(report)

        return reports, attn_steps

    def _generate_single(self, feature_vec: torch.Tensor, context: str) -> str:
        """Generate one structured clinical ECG report."""
        if not self._llm_ok:
            return self._template_report(context)

        prompt = (
            f"[CLINICAL CONTEXT] {context[:400]}\n\n"
            "--- ECG DIAGNOSTIC REPORT ---\n"
            "RHYTHM: "
        )
        try:
            inputs    = self.tokenizer(prompt, return_tensors="pt")
            text_emb  = self.llm.transformer.wte(inputs.input_ids)
            soft_prompt = self.fused_to_llm(feature_vec).unsqueeze(1)
            combined  = torch.cat([soft_prompt, text_emb], dim=1)
            attn_mask = torch.ones(combined.shape[:2], dtype=torch.long)

            output = self.llm.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=200,
                temperature=0.25,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
            )
            raw = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return self._format_report(raw, context)
        except Exception as exc:
            logger.warning(f"Report generation error: {exc}")
            return self._template_report(context)

    @staticmethod
    def _format_report(raw_text: str, context: str) -> str:
        """Ensure the report follows a structured clinical template."""
        report_lines = [
            "--- ECG DIAGNOSTIC REPORT (AI-GENERATED PROTOTYPE — NOT FOR CLINICAL USE) ---",
            "DISCLAIMER: Narrative generated by GPT-2 (124M), which is NOT fine-tuned on",
            "clinical ECG reports; wording is illustrative. Quantitative ECG classification",
            "is reported separately from the PTB-XL-trained encoder. Do NOT use for clinical",
            "decision-making.\n",
            f"[Context Utilised] {context[:150]}…\n",
        ]
        if "RHYTHM" in raw_text:
            report_lines.append(raw_text[raw_text.find("RHYTHM"):])
        else:
            report_lines.append(raw_text)
        return "\n".join(report_lines)

    @staticmethod
    def _template_report(context: str) -> str:
        return (
            "--- ECG DIAGNOSTIC REPORT (AI-GENERATED PROTOTYPE — NOT FOR CLINICAL USE) ---\n"
            "DISCLAIMER: This narrative is a template fallback (GPT-2 unavailable). The vital\n"
            "values below are PLACEHOLDERS, not measured from the input signal. Genuine ECG\n"
            "classification comes from the PTB-XL-trained encoder and is reported separately.\n"
            "Do NOT use for clinical decisions.\n\n"
            f"[Context Utilised] {context[:200]}\n\n"
            "RHYTHM: [PLACEHOLDER — not computed from signal]\n"
            "FINDINGS:\n"
            "  P Waves:      [PLACEHOLDER — not computed]\n"
            "  PR Interval:  [PLACEHOLDER — not computed]\n"
            "  QRS Duration: [PLACEHOLDER — not computed]\n"
            "  QT/QTc:       [PLACEHOLDER — not computed]\n"
            "  ST Segment:   [PLACEHOLDER — not computed]\n"
            "  T Waves:      [PLACEHOLDER — not computed]\n\n"
            "IMPRESSION: [PLACEHOLDER] Template output only — no values computed from the signal.\n"
            "RECOMMENDATIONS: Requires review by a board-certified cardiologist.\n"
            "Do NOT make any clinical decisions based on this AI-generated draft."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Arrhythmia / Risk Classifier Head (for >95% accuracy objective)
# ──────────────────────────────────────────────────────────────────────────────

class CardioClassifierHead(nn.Module):
    """
    Mixture-of-Experts (MoE) classifier on top of fused ECG+RAG features.
    Provides discrete class probabilities alongside the generative report.

    Classes: Normal | SVEB | VEB | Fusion | Unknown/Paced | ST-Elevation
    Uses dropout + label smoothing to regularise for >95% val accuracy.
    """

    def __init__(
        self,
        in_dim:    int = MILVUS_DIM,
        n_classes: int = 6,
        n_experts: int = 4,
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.n_experts = n_experts
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(in_dim, n_experts),
            nn.Softmax(dim=-1),
        )
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits     : (B, n_classes)
          gate_probs : (B, n_experts) — for load-balancing auxiliary loss
        """
        gate_probs = self.gate(x)                        # (B, n_experts)
        expert_out = torch.stack(
            [e(x) for e in self.experts], dim=1
        )                                                 # (B, n_experts, n_classes)
        logits = (gate_probs.unsqueeze(-1) * expert_out).sum(dim=1)   # (B, n_classes)
        return logits, gate_probs


# ──────────────────────────────────────────────────────────────────────────────
# Full DeepCardio-RAG Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class DeepCardioRAG(nn.Module):
    """
    End-to-end DeepCardio-RAG model.

    Forward pass (inference):
      1. ECG → ResNet-34 encoder → f_ecg (256-dim)
      2. Milvus retrieval → top-5 clinical docs
      3. RAG attention → fused representation (768-dim)
      4. MoE classifier → arrhythmia class probabilities
      5. GPT-2 report generation → structured clinical text

    Loss (training):
      L = L_cls (CrossEntropy + LabelSmoothing) + λ_cov × L_cov
    """

    def __init__(
        self,
        n_classes:       int   = 6,
        n_experts:       int   = 4,
        coverage_lambda: float = COVERAGE_LAMBDA,
        encoder_type:    str   = "transformer",   # "transformer" | "resnet"
    ):
        super().__init__()
        self.coverage_lambda = coverage_lambda
        self.encoder_type    = encoder_type
        # Set True by load_trained_encoder() when genuine PTB-XL weights are loaded.
        self.encoder_is_trained = False

        # Stage 1: ECG Encoder — Transformer-MoE (default) or ResNet-34 (legacy)
        if encoder_type == "transformer":
            self.encoder = ECGTransformerMoE(
                in_channels=ECG_LEADS,
                embed_dim=ECG_EMBED_DIM,
            )
            encoder_desc = "ECGTransformerMoE(6L×8H×4E)"
        else:
            self.encoder = ResNet34_1D_ECGEncoder(
                in_channels=ECG_LEADS,
                embed_dim=ECG_EMBED_DIM,
            )
            encoder_desc = "ResNet-34-1D"

        # Stage 2: Retriever (not a nn.Module — calls Milvus)
        self.retriever = ClinicalKnowledgeRetriever(
            ecg_dim=ECG_EMBED_DIM,
            milvus_dim=MILVUS_DIM,
        )

        # Stage 3: RAG attention
        self.rag_attention = RAGAttentionModule(
            ecg_dim=ECG_EMBED_DIM,
            context_dim=MILVUS_DIM,
            out_dim=MILVUS_DIM,
            n_heads=MHA_HEADS,
            d_k=MHA_DK,
        )

        # Stage 4: MoE classifier
        self.classifier = CardioClassifierHead(
            in_dim=MILVUS_DIM,
            n_classes=n_classes,
            n_experts=n_experts,
        )

        # Stage 5: Report generator
        self.generator = ClinicalReportGenerator(fused_dim=MILVUS_DIM)

        # Genuine PTB-XL 5-class diagnostic head (LayerNorm→Dropout→Linear, the
        # architecture trained in core/train_ptbxl.py). Populated by
        # load_trained_encoder(); stays None when no trained checkpoint exists.
        # This — not the untrained MoE head below — is what drives retrieval,
        # so a query is only issued when it is backed by trained weights.
        self.diagnostic_head: Optional[nn.Module] = None
        self.diagnostic_classes: List[str] = []

        # Loss criterion
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        logger.info(
            f"DeepCardioRAG initialised | Encoder: {encoder_desc} | "
            f"RAG: MHA(h={MHA_HEADS}, dk={MHA_DK}) | "
            f"Classifier: MoE({n_experts} experts, {n_classes} classes)"
        )

    def diagnose(self, ecg_features: torch.Tensor) -> List[Optional[Dict]]:
        """
        Predict the PTB-XL diagnostic superclass per sample using the genuinely
        trained head. Returns None per sample when no trained head is loaded —
        callers must treat that as "no grounded diagnosis", never substitute a
        default class.
        """
        B = ecg_features.size(0)
        if self.diagnostic_head is None:
            return [None] * B

        from core.ptbxl_loader import SUPERCLASS_INFO

        with torch.no_grad():
            logits = self.diagnostic_head(ecg_features)          # (B, 5)
            probs  = torch.softmax(logits, dim=-1)

        out: List[Optional[Dict]] = []
        for i in range(B):
            k    = int(torch.argmax(probs[i]).item())
            code = self.diagnostic_classes[k]
            info = SUPERCLASS_INFO.get(code, {})
            out.append({
                "class":       code,
                "name":        info.get("name", code),
                "description": info.get("description", ""),
                "confidence":  round(float(probs[i][k].item()), 4),
            })
        return out

    @staticmethod
    def _query_text(diagnosis: Optional[Dict]) -> Optional[str]:
        """Build the retrieval query from a predicted diagnosis (None → no query)."""
        if not diagnosis:
            return None
        return f"{diagnosis['name']}. {diagnosis['description']}".strip()

    def encode_context(self, texts: List[List[str]]) -> torch.Tensor:
        """
        Encode retrieved clinical text into dense embeddings using
        Sentence-BERT (all-mpnet-base-v2, 768-dim).

        Falls back to random embeddings if model unavailable.
        """
        try:
            sbert = get_sbert()
            if sbert is None:
                raise RuntimeError("Sentence-BERT not available")
            B, K = len(texts), len(texts[0])
            all_flat  = [t for sample in texts for t in sample]
            embs_flat = sbert.encode(all_flat, convert_to_tensor=True)
            embs      = embs_flat.view(B, K, MILVUS_DIM)
            return embs.to(next(self.encoder.parameters()).device)
        except Exception as exc:
            logger.warning(f"Sentence-BERT unavailable: {exc} — using random context embeddings")
            B, K = len(texts), TOP_K_RETRIEVE
            dev = next(self.encoder.parameters()).device
            return torch.randn(B, K, MILVUS_DIM, device=dev) * 0.1

    def forward(
        self,
        x:      torch.Tensor,                          # (B, 12, L)
        labels: Optional[torch.Tensor] = None,         # (B,) int64
    ) -> Dict:
        """
        Returns dict with keys:
          ecg_features  : (B, 256)
          fused          : (B, 768)
          logits         : (B, n_classes)
          probs          : (B, n_classes)
          attn_weights   : (B, H, 1, K)
          reports        : List[str]
          contexts       : List[List[str]]
          loss           : scalar (only if labels provided)
          coverage_loss  : scalar
        """
        # Stage 1 — ECG encoding
        ecg_features = self.encoder(x)                 # (B, 256)

        # Stage 1b — Diagnostic prediction (genuine PTB-XL head, or None)
        diagnoses = self.diagnose(ecg_features)
        queries   = [self._query_text(d) for d in diagnoses]

        # Stage 2 — Milvus retrieval, queried by the predicted diagnosis text
        texts_batch, meta_batch, retrieval_source = self.retriever.retrieve(
            queries, top_k=TOP_K_RETRIEVE
        )

        # Encode retrieved texts → context embeddings
        context_emb = self.encode_context(texts_batch).clone() # (B, K, 768)

        # Stage 3 — RAG attention fusion
        fused, attn_weights = self.rag_attention(ecg_features, context_emb)  # (B, 768), (B, H, 1, K)

        # Stage 4 — MoE classification
        logits, gate_probs = self.classifier(fused)    # (B, n_classes), (B, n_experts)
        probs = torch.softmax(logits, dim=-1)

        # Stage 5 — Report generation
        reports, attn_steps = self.generator.generate_report(fused, texts_batch)

        # Coverage loss
        cov_loss = compute_coverage_loss(attn_steps)

        # Classification loss (if training)
        total_loss = None
        if labels is not None:
            cls_loss   = self.ce_loss(logits, labels)
            total_loss = cls_loss + self.coverage_lambda * cov_loss

        return {
            "ecg_features":  ecg_features,
            "fused":         fused,
            "logits":        logits,
            "probs":         probs,
            "gate_probs":    gate_probs,
            "attn_weights":  attn_weights,
            "reports":       reports,
            "contexts":      texts_batch,
            "context_meta":  meta_batch,
            "diagnoses":     diagnoses,
            "retrieval_queries": queries,
            "retrieval_source":  retrieval_source,
            "loss":          total_loss,
            "coverage_loss": cov_loss,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3-Stage Training Strategy (utility function)
# ──────────────────────────────────────────────────────────────────────────────

def get_3stage_optimizer(model: DeepCardioRAG, base_lr: float = 1e-3):
    """
    Returns three AdamW optimisers for progressive 3-stage training:

    Stage A (20 ep): encoder + classifier only  LR = base_lr
    Stage B (10 ep): + RAG attention            LR = base_lr / 5
    Stage C ( 5 ep): full model                 LR = base_lr / 10
    """
    import torch.optim as optim

    encoder_params    = list(model.encoder.parameters()) + list(model.classifier.parameters())
    attention_params  = list(model.rag_attention.parameters())
    generator_params  = list(model.generator.parameters()) if hasattr(model.generator, "fused_to_llm") else []

    opt_A = optim.AdamW(encoder_params,   lr=base_lr,       weight_decay=1e-4)
    opt_B = optim.AdamW(attention_params, lr=base_lr / 5,   weight_decay=1e-4)
    opt_C = optim.AdamW(
        encoder_params + attention_params + generator_params,
        lr=base_lr / 10, weight_decay=1e-5,
    )
    return opt_A, opt_B, opt_C


def get_cosine_scheduler(optimizer, T_max: int, eta_min: float = 1e-6):
    import torch.optim.lr_scheduler as sched
    return sched.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


# ──────────────────────────────────────────────────────────────────────────────
# Global Lazy Singleton
# ──────────────────────────────────────────────────────────────────────────────

_model_instance: Optional[DeepCardioRAG] = None

# Genuine PTB-XL-trained encoder weights (produced by core/train_ptbxl.py).
# When present, the flagship encoder is REAL, not random-initialised.
_ECG_ENCODER_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "ecg_ptbxl_encoder.pt")


def load_trained_encoder(model: DeepCardioRAG) -> bool:
    """
    Load genuine PTB-XL-trained weights into the flagship ECG encoder.
    Returns True iff real (non-synthetic) trained weights were loaded, so callers
    can tell whether the encoder output is meaningful or still random.
    """
    if not os.path.isfile(_ECG_ENCODER_WEIGHTS):
        return False
    try:
        ckpt = torch.load(_ECG_ENCODER_WEIGHTS, map_location="cpu", weights_only=False)
        if ckpt.get("metrics", {}).get("synthetic", False):
            logger.warning("ecg_ptbxl_encoder.pt is a SYNTHETIC smoke-test checkpoint — ignoring.")
            return False
        model.encoder.load_state_dict(ckpt["encoder"])
        model.encoder_is_trained = True
        m = ckpt.get("metrics", {})
        model.encoder_metrics = m
        logger.info(f"Loaded genuine PTB-XL encoder weights "
                    f"(test acc={m.get('test_accuracy')}, macro-F1={m.get('test_f1_macro')}).")

        # Load the matching 5-class diagnostic head. Without it the pipeline has
        # no grounded diagnosis, so retrieval falls back to [DEMO] rather than
        # querying Milvus with something unfounded.
        head_sd = ckpt.get("head")
        if head_sd:
            try:
                from core.ptbxl_loader import SUPERCLASSES
                head = nn.Sequential(
                    nn.LayerNorm(ECG_EMBED_DIM),
                    nn.Dropout(0.2),
                    nn.Linear(ECG_EMBED_DIM, len(SUPERCLASSES)),
                )
                head.load_state_dict(head_sd)
                head.eval()
                model.diagnostic_head    = head
                model.diagnostic_classes = list(SUPERCLASSES)
                logger.info("Loaded genuine PTB-XL diagnostic head — retrieval is diagnosis-grounded.")
            except Exception as exc:
                logger.warning(f"Could not load diagnostic head: {exc} — retrieval will use demo guidelines.")
        else:
            logger.warning("Checkpoint has no diagnostic head — retrieval will use demo guidelines.")
        return True
    except Exception as exc:
        logger.warning(f"Could not load trained encoder weights: {exc}")
        return False


_encoder_status_cache: Optional[dict] = None


def get_encoder_status() -> dict:
    """
    Report whether the flagship ECG encoder is genuinely trained, WITHOUT forcing
    the (slow) model singleton into existence.

    Exists because the UI's Vector DB Stats screen hardcoded "random weights (not
    trained)" while the startup log said the opposite — a hardcoded claim about a
    conditional runtime fact is wrong the moment the condition changes, in either
    direction. Callers should render from this, never from a literal.

    If the singleton is already built, its live flags win. Otherwise we read the
    checkpoint header from disk (cached — the file is ~55 MB, too heavy per-request)
    and report what WOULD be loaded, which is the same decision load_trained_encoder()
    will make.
    """
    global _encoder_status_cache

    if _model_instance is not None:
        return {
            "trained":          bool(getattr(_model_instance, "encoder_is_trained", False)),
            "metrics":          getattr(_model_instance, "encoder_metrics", {}) or {},
            "diagnosis_grounded": getattr(_model_instance, "diagnostic_head", None) is not None,
            "source":           "live model",
        }

    if _encoder_status_cache is not None:
        return _encoder_status_cache

    status = {"trained": False, "metrics": {}, "diagnosis_grounded": False,
              "source": "checkpoint on disk"}
    if os.path.isfile(_ECG_ENCODER_WEIGHTS):
        try:
            ckpt = torch.load(_ECG_ENCODER_WEIGHTS, map_location="cpu", weights_only=False)
            m = ckpt.get("metrics", {}) or {}
            if not m.get("synthetic", False):
                status["trained"]            = True
                status["metrics"]            = m
                status["diagnosis_grounded"] = bool(ckpt.get("head"))
            del ckpt
        except Exception as exc:
            logger.warning(f"Could not read encoder checkpoint for status: {exc}")
    else:
        status["source"] = "no checkpoint file"

    _encoder_status_cache = status
    return status


def describe_ecg_encoder(encoder_type: str = "transformer") -> str:
    """One-line, truthful description of the ECG encoder for UI/API display."""
    arch = ("ECGTransformerMoE (6L x 8H x 4E, 256-dim)" if encoder_type == "transformer"
            else "ResNet-34-1D (256-dim)")
    st = get_encoder_status()
    if not st["trained"]:
        return f"{arch} — random weights (not trained)"

    m = st["metrics"]
    acc, f1 = m.get("test_accuracy"), m.get("test_f1_macro")
    bits = []
    if acc is not None:
        bits.append(f"test acc={acc:.4f}" if isinstance(acc, float) else f"test acc={acc}")
    if f1 is not None:
        bits.append(f"macro-F1={f1:.4f}" if isinstance(f1, float) else f"macro-F1={f1}")
    scores = f" ({', '.join(bits)})" if bits else ""
    grounded = "" if st["diagnosis_grounded"] else " — no diagnostic head, retrieval falls back to demo guidelines"
    return f"{arch} — trained on PTB-XL{scores}{grounded}"


def get_model(encoder_type: str = "transformer") -> DeepCardioRAG:
    global _model_instance
    if _model_instance is None:
        logger.info(f"Initialising DeepCardioRAG (ECGTransformerMoE + RAG Attention + MoE) encoder={encoder_type}…")
        _model_instance = DeepCardioRAG(encoder_type=encoder_type)
        loaded = load_trained_encoder(_model_instance)
        _model_instance.eval()
        logger.info(f"DeepCardioRAG ready (encoder={'TRAINED on PTB-XL' if loaded else 'UNTRAINED/random'})")
    return _model_instance
