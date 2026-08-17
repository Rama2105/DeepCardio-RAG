"""
CardioFusion: Unified Multi-Modal Hybrid Cardiac Analysis Model
================================================================
A hybrid deep learning architecture that fuses all 4 cardiac data
modalities into a shared clinical representation space:

  1. Echo Video    (3D-CNN)  → 384-dim → shared 512-dim
  2. ECG Images    (2D-CNN)  → 256-dim → shared 512-dim
  3. Heart Sound   (2D-CNN)  → 256-dim → shared 512-dim
  4. ECG Signals   (1D-CNN)  → 256-dim → shared 512-dim

Key Architecture Features:
─────────────────────────
• Modality-Specific Encoders: Each input type gets its own specialised
  encoder (reusing existing architectures) that produces embeddings.

• Shared Projection Space: All encoders project into a common 512-dim
  space via learnable projection heads with LayerNorm + Dropout.

• Cross-Modal Transformer Fusion: A lightweight 4-layer Transformer
  encoder with learned [MOD] tokens attends across modality embeddings,
  enabling knowledge transfer between e.g. echo videos and heart sounds.

• Multi-Gate Mixture of Experts (MMoE): Task-specific expert networks
  with gating for each clinical output, preventing negative transfer.

• Contrastive Alignment Loss: CLIP-style contrastive loss aligns
  same-patient data across modalities in the shared space.

• 6 Clinical Task Heads:
    ─ Arrhythmia Classification   (5 AAMI classes)
    ─ Ejection Fraction Regression (continuous EF%)
    ─ Heart Murmur Detection      (3 classes)
    ─ Ventricular Arrhythmia Alert (binary dangerous/normal)
    ─ Rhythm Identification        (4 VFDB classes)
    ─ Cardiac Risk Score           (unified 0-100 severity)

Training Strategy:
  Phase 1: Pre-train individual encoders on their own datasets
  Phase 2: Freeze encoders, train fusion + task heads
  Phase 3: End-to-end fine-tuning with lower LR for encoders

Estimated improvement: +8-15% accuracy via cross-modal knowledge
transfer and multi-task regularisation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple


# ==============================================================================
# Configuration
# ==============================================================================
SHARED_DIM = 512            # Unified embedding dimension
NUM_TRANSFORMER_LAYERS = 4
NUM_HEADS = 8
FFN_DIM = 1024
DROPOUT = 0.2
NUM_EXPERTS = 4
TEMPERATURE = 0.07          # Contrastive loss temperature

# Modality identifiers
MODALITIES = ["echo_video", "ecg_image", "heart_sound", "ecg_signal"]


# ==============================================================================
# 1. Modality-Specific Encoders (wrapped from existing modules)
# ==============================================================================
class ModalityEncoder(nn.Module):
    """
    Wraps an existing encoder and adds a projection head
    to map its embeddings into the shared SHARED_DIM space.
    """

    def __init__(self, encoder: nn.Module, source_dim: int,
                 shared_dim: int = SHARED_DIM, freeze: bool = False):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(source_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input and project to shared space. Returns (B, shared_dim)."""
        with torch.set_grad_enabled(self.training):
            raw_emb = self.encoder(x)
        return self.projection(raw_emb)


def build_echo_encoder(embed_dim: int = 384, freeze: bool = False) -> ModalityEncoder:
    from core.video_encoder import EchoVideoEncoder
    enc = EchoVideoEncoder(embed_dim=embed_dim)
    return ModalityEncoder(enc, source_dim=embed_dim, freeze=freeze)


def build_ecg_image_encoder(embed_dim: int = 256, freeze: bool = False) -> ModalityEncoder:
    """Build from the ECGImageClassifier's feature extractor."""
    from core.ecg_image_loader import ECGImageClassifier
    full_model = ECGImageClassifier(embed_dim=embed_dim)
    # Extract just the feature+embedding layers (no classifier head)
    class _Encoder(nn.Module):
        def __init__(self, features, embedding):
            super().__init__()
            self.features = features
            self.embedding = embedding
        def forward(self, x):
            return self.embedding(self.features(x).flatten(1))
    enc = _Encoder(full_model.features, full_model.embedding)
    return ModalityEncoder(enc, source_dim=embed_dim, freeze=freeze)


def build_heart_sound_encoder(embed_dim: int = 256, freeze: bool = False) -> ModalityEncoder:
    from core.heart_sound_loader import HeartSoundClassifier
    full_model = HeartSoundClassifier(embed_dim=embed_dim)
    class _Encoder(nn.Module):
        def __init__(self, features, embedding):
            super().__init__()
            self.features = features
            self.embedding = embedding
        def forward(self, x):
            return self.embedding(self.features(x).flatten(1))
    enc = _Encoder(full_model.features, full_model.embedding)
    return ModalityEncoder(enc, source_dim=embed_dim, freeze=freeze)


def build_ecg_signal_encoder(embed_dim: int = 256, freeze: bool = False) -> ModalityEncoder:
    from core.vfdb_loader import VentricularArrhythmiaDetector
    full_model = VentricularArrhythmiaDetector(embed_dim=embed_dim)
    class _Encoder(nn.Module):
        def __init__(self, encoder_layers, embedding):
            super().__init__()
            self.encoder_layers = encoder_layers
            self.embedding = embedding
        def forward(self, x):
            return self.embedding(self.encoder_layers(x).flatten(1))
    enc = _Encoder(full_model.encoder, full_model.embedding)
    return ModalityEncoder(enc, source_dim=embed_dim, freeze=freeze)


# ==============================================================================
# 2. Learned Modality Tokens + Positional Encoding
# ==============================================================================
class ModalityTokens(nn.Module):
    """
    Learned [MOD] tokens prepended to each modality's embedding.
    These act as 'query' tokens in the Transformer fusion, similar
    to the [CLS] token in BERT or register tokens in ViT.
    """

    def __init__(self, num_modalities: int = 4, dim: int = SHARED_DIM):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_modalities, 1, dim) * 0.02)
        self.modality_embeddings = nn.Embedding(num_modalities, dim)

    def get_token(self, modality_idx: int, batch_size: int) -> torch.Tensor:
        """Returns (B, 1, dim) token for a modality."""
        token = self.tokens[modality_idx].expand(batch_size, -1, -1)
        mod_emb = self.modality_embeddings(
            torch.tensor(modality_idx, device=token.device)
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        return token + mod_emb


# ==============================================================================
# 3. Cross-Modal Transformer Fusion
# ==============================================================================
class CrossModalTransformer(nn.Module):
    """
    Lightweight Transformer encoder that fuses embeddings from
    multiple modalities via self-attention.

    Each modality contributes: [MOD_token, embedding]
    The Transformer attends across all tokens, enabling
    cross-modal feature interaction.
    """

    def __init__(self, d_model: int = SHARED_DIM,
                 nhead: int = NUM_HEADS,
                 num_layers: int = NUM_TRANSFORMER_LAYERS,
                 dim_feedforward: int = FFN_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: (B, N_tokens, d_model) — concatenated modality tokens
            mask: optional attention mask
        Returns:
            (B, N_tokens, d_model) — fused representations
        """
        return self.norm(self.transformer(tokens, src_key_padding_mask=mask))


# ==============================================================================
# 4. Multi-Gate Mixture of Experts (MMoE)
# ==============================================================================
class ExpertNetwork(nn.Module):
    """Single expert: a 2-layer MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMoELayer(nn.Module):
    """
    Multi-Gate Mixture of Experts.
    Each task has its own gating network that blends expert outputs.
    This prevents negative transfer between dissimilar tasks.
    """

    def __init__(self, input_dim: int, expert_dim: int,
                 num_experts: int = NUM_EXPERTS, num_tasks: int = 6):
        super().__init__()
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_dim, expert_dim)
            for _ in range(num_experts)
        ])
        # One gate per task
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — fused embedding
            task_idx: which task's gate to use
        Returns:
            (B, expert_dim) — task-specific expert blend
        """
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, num_experts, expert_dim)
        gate_weights = self.gates[task_idx](x).unsqueeze(-1)  # (B, num_experts, 1)
        blended = (expert_outputs * gate_weights).sum(dim=1)  # (B, expert_dim)
        return blended


# ==============================================================================
# 5. Clinical Task Heads
# ==============================================================================
class ArrhythmiaHead(nn.Module):
    """5-class AAMI arrhythmia classification (N, S, V, F, Q)."""
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 5))
    def forward(self, x): return self.head(x)

class EjectionFractionHead(nn.Module):
    """EF regression + 3-class categorisation."""
    def __init__(self, dim: int):
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, x):
        return {"ef_value": self.regressor(x).squeeze(-1), "ef_class": self.classifier(x)}

class MurmurHead(nn.Module):
    """3-class heart murmur detection (Absent, Present, Unknown)."""
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 3))
    def forward(self, x): return self.head(x)

class VentricularAlertHead(nn.Module):
    """Binary dangerous arrhythmia detection."""
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))
    def forward(self, x): return self.head(x)

class RhythmHead(nn.Module):
    """4-class rhythm identification (Normal, VT, VF/VFL, Other_Dangerous)."""
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 4))
    def forward(self, x): return self.head(x)

class CardiacRiskHead(nn.Module):
    """Unified 0-100 cardiac severity score combining all modality signals."""
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),  # 0-1, scale to 0-100
        )
    def forward(self, x): return self.head(x).squeeze(-1) * 100.0


# ==============================================================================
# 6. Contrastive Alignment Loss
# ==============================================================================
class ContrastiveAlignmentLoss(nn.Module):
    """
    CLIP-style InfoNCE loss that aligns same-patient embeddings
    across different modalities in the shared space.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_a: (B, dim) embeddings from modality A
            emb_b: (B, dim) embeddings from modality B
            Assumes row i in both tensors correspond to the same patient.
        """
        emb_a = F.normalize(emb_a, dim=1)
        emb_b = F.normalize(emb_b, dim=1)
        logits = emb_a @ emb_b.T / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.T, labels)
        return (loss_a2b + loss_b2a) / 2


# ==============================================================================
# 7. CardioFusion: The Full Hybrid Model
# ==============================================================================
class CardioFusion(nn.Module):
    """
    Unified multi-modal cardiac analysis model.

    Supports flexible inference: provide any subset of modalities
    (1 to 4) and the model produces all 6 clinical outputs.
    Missing modalities are masked in the Transformer fusion.
    """

    TASK_NAMES = [
        "arrhythmia",       # 0
        "ejection_fraction", # 1
        "murmur",           # 2
        "ventricular_alert", # 3
        "rhythm",           # 4
        "cardiac_risk",     # 5
    ]

    def __init__(self, shared_dim: int = SHARED_DIM, freeze_encoders: bool = False):
        super().__init__()
        self.shared_dim = shared_dim

        # --- Modality encoders ---
        self.echo_encoder = build_echo_encoder(embed_dim=384, freeze=freeze_encoders)
        self.ecg_img_encoder = build_ecg_image_encoder(embed_dim=256, freeze=freeze_encoders)
        self.heart_sound_encoder = build_heart_sound_encoder(embed_dim=256, freeze=freeze_encoders)
        self.ecg_signal_encoder = build_ecg_signal_encoder(embed_dim=256, freeze=freeze_encoders)

        # --- Modality tokens ---
        self.modality_tokens = ModalityTokens(num_modalities=4, dim=shared_dim)

        # --- Cross-modal Transformer ---
        self.fusion = CrossModalTransformer(d_model=shared_dim)

        # --- Global aggregation ---
        self.global_pool = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.LayerNorm(shared_dim),
        )

        # --- MMoE ---
        self.mmoe = MMoELayer(
            input_dim=shared_dim, expert_dim=shared_dim,
            num_experts=NUM_EXPERTS, num_tasks=6
        )

        # --- Task heads ---
        self.arrhythmia_head = ArrhythmiaHead(shared_dim)
        self.ef_head = EjectionFractionHead(shared_dim)
        self.murmur_head = MurmurHead(shared_dim)
        self.vent_alert_head = VentricularAlertHead(shared_dim)
        self.rhythm_head = RhythmHead(shared_dim)
        self.risk_head = CardiacRiskHead(shared_dim)

        # --- Contrastive loss ---
        self.contrastive_loss = ContrastiveAlignmentLoss()

    def encode_modality(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        """Encode a single modality input to shared space."""
        encoders = {
            "echo_video": self.echo_encoder,
            "ecg_image": self.ecg_img_encoder,
            "heart_sound": self.heart_sound_encoder,
            "ecg_signal": self.ecg_signal_encoder,
        }
        return encoders[modality](x)

    def fuse(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse available modality embeddings via the Transformer.

        Args:
            modality_embeddings: dict of {modality_name: (B, shared_dim)}
        Returns:
            (B, shared_dim) fused global embedding
        """
        batch_size = next(iter(modality_embeddings.values())).size(0)
        device = next(iter(modality_embeddings.values())).device

        tokens_list = []
        mod_indices = {"echo_video": 0, "ecg_image": 1, "heart_sound": 2, "ecg_signal": 3}

        for mod_name, emb in modality_embeddings.items():
            idx = mod_indices[mod_name]
            # [MOD] token + embedding token = 2 tokens per modality
            mod_token = self.modality_tokens.get_token(idx, batch_size)  # (B, 1, dim)
            emb_token = emb.unsqueeze(1)  # (B, 1, dim)
            tokens_list.extend([mod_token, emb_token])

        # Concatenate all tokens: (B, 2*num_present_mods, dim)
        all_tokens = torch.cat(tokens_list, dim=1)

        # Run through Transformer
        fused = self.fusion(all_tokens)

        # Aggregate: mean of [MOD] tokens (every other starting from 0)
        mod_token_indices = list(range(0, all_tokens.size(1), 2))
        mod_tokens_out = fused[:, mod_token_indices, :]  # (B, num_mods, dim)
        global_emb = mod_tokens_out.mean(dim=1)  # (B, dim)

        return self.global_pool(global_emb)

    def predict_all_tasks(self, fused_emb: torch.Tensor) -> Dict[str, Any]:
        """Run all 6 task heads on the fused embedding."""
        return {
            "arrhythmia_logits": self.arrhythmia_head(self.mmoe(fused_emb, 0)),
            "ef_output": self.ef_head(self.mmoe(fused_emb, 1)),
            "murmur_logits": self.murmur_head(self.mmoe(fused_emb, 2)),
            "vent_alert_logits": self.vent_alert_head(self.mmoe(fused_emb, 3)),
            "rhythm_logits": self.rhythm_head(self.mmoe(fused_emb, 4)),
            "cardiac_risk_score": self.risk_head(self.mmoe(fused_emb, 5)),
        }

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Full forward pass.

        Args:
            inputs: dict mapping modality names to input tensors.
                    Any subset of modalities is accepted.
        Returns:
            dict with all task outputs + per-modality embeddings
        """
        # Stage 1: Encode each provided modality
        modality_embeddings = {}
        for mod_name, x in inputs.items():
            if x is not None:
                modality_embeddings[mod_name] = self.encode_modality(mod_name, x)

        if not modality_embeddings:
            raise ValueError("At least one modality input is required.")

        # Stage 2: Cross-modal Transformer fusion
        fused_emb = self.fuse(modality_embeddings)

        # Stage 3: Task prediction via MMoE
        task_outputs = self.predict_all_tasks(fused_emb)

        return {
            "fused_embedding": fused_emb,
            "modality_embeddings": modality_embeddings,
            **task_outputs,
        }

    @torch.no_grad()
    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Clean inference interface returning human-readable results.
        """
        self.eval()
        raw = self.forward(inputs)

        # Arrhythmia
        arr_probs = F.softmax(raw["arrhythmia_logits"], dim=1)
        arr_labels = ["N", "S", "V", "F", "Q"]
        arr_idx = arr_probs.argmax(dim=1)

        # EF
        ef_val = raw["ef_output"]["ef_value"]
        ef_cls_probs = F.softmax(raw["ef_output"]["ef_class"], dim=1)
        ef_cls_labels = ["HFrEF", "HFmrEF", "Normal"]

        # Murmur
        mur_probs = F.softmax(raw["murmur_logits"], dim=1)
        mur_labels = ["Absent", "Present", "Unknown"]

        # Ventricular alert
        vent_probs = F.softmax(raw["vent_alert_logits"], dim=1)

        # Rhythm
        rhy_probs = F.softmax(raw["rhythm_logits"], dim=1)
        rhy_labels = ["Normal", "VT", "VF/VFL", "Other_Dangerous"]

        # Risk
        risk = raw["cardiac_risk_score"]

        B = arr_probs.size(0)
        results = []
        for i in range(B):
            a_idx = arr_idx[i].item()
            ef_c_idx = ef_cls_probs[i].argmax().item()
            m_idx = mur_probs[i].argmax().item()
            r_idx = rhy_probs[i].argmax().item()
            danger = vent_probs[i, 1].item()
            risk_val = risk[i].item()

            # Determine overall alert level
            if danger > 0.8 or risk_val > 80:
                alert = "CRITICAL"
            elif danger > 0.5 or risk_val > 50 or a_idx in [2, 3]:
                alert = "WARNING"
            else:
                alert = "NORMAL"

            results.append({
                "alert_level": alert,
                "cardiac_risk_score": round(risk_val, 1),
                "arrhythmia": {
                    "class": arr_labels[a_idx],
                    "confidence": round(arr_probs[i, a_idx].item(), 4),
                    "probabilities": {arr_labels[j]: round(arr_probs[i, j].item(), 4) for j in range(5)},
                },
                "ejection_fraction": {
                    "value": round(ef_val[i].item(), 1),
                    "category": ef_cls_labels[ef_c_idx],
                    "confidence": round(ef_cls_probs[i, ef_c_idx].item(), 4),
                },
                "murmur": {
                    "class": mur_labels[m_idx],
                    "confidence": round(mur_probs[i, m_idx].item(), 4),
                    "probabilities": {mur_labels[j]: round(mur_probs[i, j].item(), 4) for j in range(3)},
                },
                "ventricular_danger": {
                    "is_dangerous": danger > 0.5,
                    "probability": round(danger, 4),
                },
                "rhythm": {
                    "class": rhy_labels[r_idx],
                    "confidence": round(rhy_probs[i, r_idx].item(), 4),
                    "probabilities": {rhy_labels[j]: round(rhy_probs[i, j].item(), 4) for j in range(4)},
                },
                "modalities_used": list(inputs.keys()),
            })

        return results[0] if B == 1 else results


# ==============================================================================
# 7b. Learned Multi-Task Loss Weighting (answers reviewer Q6)
# ==============================================================================
class HomoscedasticUncertaintyWeighting(nn.Module):
    """
    Learns per-task loss weights via homoscedastic (task) uncertainty instead of
    hand-tuned constants — Kendall, Gal & Cipolla, "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.

    Each task t gets a learnable log-variance s_t = log(sigma_t^2). The combined
    objective is:
        L = sum_t [ (1 / (2 * exp(s_t))) * L_t + 0.5 * s_t ]
    (0.5 * s_t regularises the weights so they can't collapse to zero.) The
    weights 1/(2*sigma^2) are LEARNED jointly with the network — so the answer to
    "how is the CardioFusion weighting determined?" is: by gradient descent on
    task uncertainty, not by fixed hand-set values.
    """

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = list(task_names)
        # log_var initialised to 0 -> initial weight 0.5 for every task
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_names)))

    def combine(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            s = self.log_vars[i]
            weighted = torch.exp(-s) * losses[name] + 0.5 * s
            total = weighted if total is None else total + weighted
        if total is None:
            # No task losses this step — return a differentiable zero on-device
            return self.log_vars.sum() * 0.0
        return total

    def current_weights(self) -> Dict[str, float]:
        """Report the learned 1/(2*sigma^2) weight per task (for logging/paper)."""
        with torch.no_grad():
            return {n: round(float(0.5 * torch.exp(-self.log_vars[i])), 4)
                    for i, n in enumerate(self.task_names)}


# ==============================================================================
# 8. Training Orchestrator
# ==============================================================================
class CardioFusionTrainer:
    """
    Multi-task training orchestrator with phased training strategy.

    Loss = weighted sum of:
        - Arrhythmia CE loss
        - EF MSE + classification CE
        - Murmur CE loss
        - Ventricular alert BCE
        - Rhythm CE loss
        - Risk score MSE
        - Contrastive alignment loss (for paired data)
    """

    # Task list for the learned uncertainty weighting (order defines log_var index).
    TASK_NAMES_FOR_LOSS = [
        "arrhythmia", "ef_regression", "ef_classification",
        "murmur", "ventricular_alert", "rhythm", "contrastive",
    ]

    def __init__(self, model: CardioFusion, lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model
        # Learned homoscedastic uncertainty weighting replaces hand-set weights (Q6).
        self.loss_weighter = HomoscedasticUncertaintyWeighting(self.TASK_NAMES_FOR_LOSS)
        self.loss_weighter.to(next(model.parameters()).device)
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.loss_weighter.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        # Optional class weighting for the arrhythmia head — counters the heavy
        # AAMI class imbalance (Normal dominates) so training does not collapse
        # to the majority class.
        if class_weights is not None:
            class_weights = class_weights.to(next(model.parameters()).device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
        self.contrastive = ContrastiveAlignmentLoss()

    def train_step(self, inputs: Dict[str, torch.Tensor],
                   labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            inputs: modality tensors
            labels: dict with keys matching task names
        Returns:
            dict of per-task losses
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        losses = {}                       # scalar values for logging
        task_losses = {}                  # differentiable tensors for weighting

        # Arrhythmia loss
        if "arrhythmia_labels" in labels:
            loss = self.ce_loss(outputs["arrhythmia_logits"], labels["arrhythmia_labels"])
            task_losses["arrhythmia"] = loss
            losses["arrhythmia"] = loss.item()

        # EF regression loss
        if "ef_values" in labels:
            ef_out = outputs["ef_output"]
            loss_reg = self.mse_loss(ef_out["ef_value"], labels["ef_values"])
            task_losses["ef_regression"] = loss_reg
            losses["ef_regression"] = loss_reg.item()

            if "ef_classes" in labels:
                loss_cls = self.ce_loss(ef_out["ef_class"], labels["ef_classes"])
                task_losses["ef_classification"] = loss_cls
                losses["ef_classification"] = loss_cls.item()

        # Murmur loss
        if "murmur_labels" in labels:
            loss = self.ce_loss(outputs["murmur_logits"], labels["murmur_labels"])
            task_losses["murmur"] = loss
            losses["murmur"] = loss.item()

        # Ventricular alert loss
        if "vent_labels" in labels:
            loss = self.ce_loss(outputs["vent_alert_logits"], labels["vent_labels"])
            task_losses["ventricular_alert"] = loss
            losses["ventricular_alert"] = loss.item()

        # Rhythm loss
        if "rhythm_labels" in labels:
            loss = self.ce_loss(outputs["rhythm_logits"], labels["rhythm_labels"])
            task_losses["rhythm"] = loss
            losses["rhythm"] = loss.item()

        # Contrastive alignment — ONLY valid with same-patient PAIRED modalities.
        # Public benchmarks are single-modality, so this is off unless the caller
        # explicitly provides paired inputs and sets labels["paired_modalities"]=True.
        mod_embs = outputs["modality_embeddings"]
        if len(mod_embs) >= 2 and labels.get("paired_modalities", False):
            emb_list = list(mod_embs.values())
            c_loss = None
            for i in range(len(emb_list)):
                for j in range(i + 1, len(emb_list)):
                    cl = self.contrastive(emb_list[i], emb_list[j])
                    c_loss = cl if c_loss is None else c_loss + cl
            if c_loss is not None:
                task_losses["contrastive"] = c_loss
                losses["contrastive"] = c_loss.item()

        # Learned uncertainty weighting (Q6): weights are trained, not hand-set.
        total_loss = self.loss_weighter.combine(task_losses)
        losses["total"] = total_loss.item()
        losses["learned_task_weights"] = self.loss_weighter.current_weights()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return losses

    def get_training_summary(self) -> Dict[str, Any]:
        """Return model parameter counts and architecture summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "frozen_parameters": total_params - trainable,
            "model_size_mb": round(total_params * 4 / 1024 / 1024, 1),
            "num_experts": NUM_EXPERTS,
            "num_tasks": 6,
            "shared_dim": SHARED_DIM,
            "transformer_layers": NUM_TRANSFORMER_LAYERS,
            "attention_heads": NUM_HEADS,
        }


# ==============================================================================
# 9. Singleton + convenience
# ==============================================================================
_fusion_model: Optional[CardioFusion] = None

def get_cardiofusion_model(freeze_encoders: bool = False) -> CardioFusion:
    global _fusion_model
    if _fusion_model is None:
        print("[CardioFusion] Initialising hybrid multi-modal model...")
        _fusion_model = CardioFusion(freeze_encoders=freeze_encoders)
        _fusion_model.eval()

        # Attempt to load pretrained fusion weights
        import os
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "cardiofusion_weights.pt"
        )
        if os.path.isfile(weights_path):
            _fusion_model.load_state_dict(
                torch.load(weights_path, map_location="cpu"), strict=False
            )
            print("[CardioFusion] Loaded pretrained fusion weights.")
        else:
            print("[CardioFusion] No pretrained weights. Using initialised model (demo mode).")
    return _fusion_model


# ==============================================================================
# 10. Transparent, reproducible risk aggregation (answers reviewer M8 / Q6)
# ==============================================================================
# The peer review (M8/Q6) noted that the paper's Eq.(13) risk score
#   R = 100 * sum_m lambda_m * p_m^risk
# had (a) an UNDEFINED mapping from each module output y_m to a risk probability
# p_m^risk, (b) SEVEN module outputs against SIX weights with no statement of which
# is dropped, and (c) a reported 70/100 that was not reproducible from the equation.
#
# This function fixes all three, transparently and reproducibly:
#   * RISK_WEIGHTS defines exactly six weighted risk contributors (lambda summing
#     to 1.0). The 7th module output — categorical RHYTHM — is deliberately NOT a
#     weighted risk term; it informs the alert level, not the score. That is the
#     output "dropped" from the weighted sum, now stated explicitly.
#   * RISK_MAPPERS defines p_m^risk = f_m(y_m) for each module (the missing map).
#   * Modules that fail the safety gate (untrained, suppressed, or physiologically
#     invalid — see core/safety_gating) are EXCLUDED, and the weights of the
#     surviving modules are renormalised, so an untrained module can never inflate
#     the score and the result is reproducible from the surfaced inputs alone.
RISK_WEIGHTS = {
    "ecg_arrhythmia":  0.25,   # ECG 12-lead / MIT-BIH abnormal-beat probability
    "ventricular":     0.25,   # VFDB dangerous-rhythm probability (if trained)
    "echo_ef":         0.20,   # reduced-EF severity from LVEF
    "murmur":          0.15,   # CirCor P(murmur present)
    "heart_disease":   0.10,   # UCI Cleveland P(disease)
    "arthritis":       0.05,   # NHANES P(high arthritis/CV risk)
}   # sum = 1.00


def _ef_to_risk(ef_pct: float) -> float:
    """Reduced-EF severity in [0,1]: 0 at EF>=50%, rising linearly to 1 at EF<=10%."""
    ef = max(0.0, min(100.0, float(ef_pct)))
    if ef >= 50.0:
        return 0.0
    return min(1.0, (50.0 - ef) / 40.0)


# p_m^risk = f_m(y_m); each returns a probability-like scalar in [0,1].
RISK_MAPPERS = {
    "ecg_arrhythmia": lambda y: 1.0 - float(y.get("p_normal", 1.0)),
    "ventricular":    lambda y: float(y.get("p_dangerous", 0.0)),
    "echo_ef":        lambda y: _ef_to_risk(y.get("ef_pct", 60.0)),
    "murmur":         lambda y: float(y.get("p_present", 0.0)),
    "heart_disease":  lambda y: float(y.get("p_disease", 0.0)),
    "arthritis":      lambda y: float(y.get("p_high_risk", 0.0)),
}


def aggregate_cardiac_risk(module_findings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute the combined cardiac risk score reproducibly, excluding any module that
    fails the safety gate. `module_findings[m]` should carry:
        is_trained (bool), confidence (float, optional),
        outputs (dict consumed by RISK_MAPPERS[m]),
        and for echo_ef an 'ef_pct' measurement that is physiologically validated.
    Returns the score in [0,100], the per-module contributions, and the list of
    excluded modules with reasons — so the number is fully auditable (M8/Q6).
    """
    from core.safety_gating import (gate_module_output, validate_measurement,
                                    check_cross_module_consistency)

    # M1a: a malignant-rhythm alert is withheld when another module concurrently
    # reports an incompatible state. Checked BEFORE scoring so the conflict is
    # visible even when every module passes its own individual gate.
    consistency = check_cross_module_consistency(module_findings)

    contributions, excluded = {}, {}
    active_weight = 0.0
    weighted_sum = 0.0
    for m, lam in RISK_WEIGHTS.items():
        f = module_findings.get(m)
        if not f:
            excluded[m] = ["no finding provided"]
            continue
        gate = gate_module_output(m, f.get("is_trained", False), f.get("confidence"))
        if not gate.used_in_risk_score:
            excluded[m] = gate.reasons
            continue
        y = f.get("outputs", {})
        if m == "echo_ef":  # physiological validity check on EF before use
            vg = validate_measurement("ejection_fraction_pct", y.get("ef_pct"), clamp=False)
            if not vg.used_in_risk_score:
                excluded[m] = vg.reasons
                continue
        p = max(0.0, min(1.0, RISK_MAPPERS[m](y)))
        contributions[m] = {"lambda": lam, "p_risk": round(p, 4), "weighted": round(lam * p, 4)}
        weighted_sum += lam * p
        active_weight += lam

    # Renormalise over surviving modules so excluded ones don't deflate the score.
    score = 100.0 * (weighted_sum / active_weight) if active_weight > 0 else 0.0
    return {
        "cardiac_risk_score": round(score, 1),
        "formula": "R = 100 * sum_m lambda_m * p_m^risk / sum_m lambda_m  (over gated modules)",
        "active_weight": round(active_weight, 4),
        "contributions": contributions,
        "excluded_modules": excluded,
        "cross_module_consistent": consistency.consistent,
        "cross_module_conflicts": consistency.conflicts,
        "suppress_critical_alert": consistency.suppress_alert,
        "note": "Rhythm (categorical) is not a weighted risk term — it informs alert level only. "
                "Untrained/invalid modules are excluded and weights renormalised (M8/Q6). "
                "If suppress_critical_alert is True the caller MUST NOT surface a critical "
                "alert: two modules reported mutually incompatible states (M1a).",
    }
