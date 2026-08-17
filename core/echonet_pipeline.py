"""
EchoNet-Dynamic Full Pipeline
==============================
Combines video encoder, RAG retriever, and report generator into a
single pipeline that mirrors the existing DeepCardio-RAG architecture
but works with echocardiogram *video* data instead of 1D ECG signals.

Pipeline stages:
  1. Video Feature Extraction  (3D-CNN EchoVideoEncoder)
  2. Semantic Retrieval        (Milvus / mock clinical knowledge)
  3. Context-aware Report Gen  (GPT-2 with echo soft-prompt injection)
  4. EF Regression             (Ejection fraction prediction head)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any

from core.video_encoder import EchoVideoEncoder, EFRegressor, EchoNetModel
from core.echonet_loader import (
    EchoNetDataset, get_echonet_loader,
    NUM_FRAMES, FRAME_SIZE, MEAN, STD,
)


# ---------------------------------------------------------------------------
# Cardiac-specific RAG retriever (extends the mock approach for echo data)
# ---------------------------------------------------------------------------
class EchoKnowledgeRetriever:
    """
    Retrieves relevant cardiac guidelines based on predicted EF and
    video embeddings. Falls back to curated echo-specific guidelines
    when Milvus is unavailable.
    """

    # Curated clinical guidelines keyed by EF category
    ECHO_GUIDELINES = {
        "HFrEF": [
            "Guideline (ACC/AHA 2022): LVEF <40% indicates Heart Failure with Reduced Ejection Fraction. "
            "Recommend initiation of GDMT including ACEi/ARB/ARNI, beta-blocker, MRA, and SGLT2i.",
            "Clinical Note: Severe LV systolic dysfunction. Assess for ICD candidacy if LVEF ≤35% "
            "and NYHA Class II-III despite optimal medical therapy for ≥3 months.",
            "Echo Finding: Global hypokinesis with reduced wall thickening. Consider cardiac MRI "
            "for etiology workup (ischemic vs non-ischemic cardiomyopathy).",
        ],
        "HFmrEF": [
            "Guideline (ESC 2021): LVEF 40-49% classified as Heart Failure with Mildly Reduced EF. "
            "Diuretics recommended for congestion. Consider ACEi/ARB/ARNI, beta-blockers, and MRA.",
            "Clinical Note: Borderline LV function. Serial echocardiographic monitoring recommended "
            "every 6-12 months to track trajectory of ventricular function.",
            "Echo Finding: Mild-to-moderate regional wall motion abnormalities. Correlate with "
            "coronary angiography if ischemic etiology suspected.",
        ],
        "Normal": [
            "Guideline: LVEF ≥50% indicates preserved systolic function. If symptomatic, evaluate for "
            "HFpEF with diastolic function assessment (E/e' ratio, LA volume index, TR velocity).",
            "Clinical Note: Normal LV systolic function. If presenting with exertional dyspnea, "
            "consider diastolic stress testing and BNP/NT-proBNP measurement.",
            "Echo Finding: Normal LV cavity size and wall thickness with preserved global "
            "longitudinal strain. Standard follow-up per clinical indication.",
        ],
    }

    def retrieve(self, ef_category: str, top_k: int = 3) -> List[str]:
        guidelines = self.ECHO_GUIDELINES.get(ef_category, self.ECHO_GUIDELINES["Normal"])
        return guidelines[:top_k]


# ---------------------------------------------------------------------------
# Echo Report Generator (adapts the existing GPT-2 approach for echo data)
# ---------------------------------------------------------------------------
class EchoReportGenerator(nn.Module):
    """
    Generates echocardiogram diagnostic reports using video embeddings
    as soft-prompts injected into GPT-2, combined with RAG-retrieved
    cardiac guidelines.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained("gpt2")

        # Project echo embeddings into GPT-2 embedding space
        self.echo_to_llm = nn.Linear(embed_dim, self.llm.config.n_embd)

    @torch.no_grad()
    def generate_report(self, echo_embeddings: torch.Tensor,
                        ef_values: torch.Tensor,
                        ef_categories: List[str],
                        retrieved_contexts: List[List[str]],
                        max_new_tokens: int = 200) -> List[str]:
        """
        Generate clinical echo reports.

        Args:
            echo_embeddings: (B, embed_dim) from the video encoder
            ef_values: (B,) predicted ejection fractions
            ef_categories: list of EF classifications
            retrieved_contexts: list of lists of guideline strings
            max_new_tokens: generation length
        Returns:
            list of generated report strings
        """
        batch_size = echo_embeddings.size(0)
        echo_soft_prompt = self.echo_to_llm(echo_embeddings).unsqueeze(1)

        reports = []
        for i in range(batch_size):
            ef = ef_values[i].item()
            cat = ef_categories[i]
            context = " | ".join(retrieved_contexts[i]) if retrieved_contexts[i] else ""

            prompt = (
                f"Clinical Context: {context}\n\n"
                f"Echocardiogram Analysis — Predicted LVEF: {ef:.1f}% ({cat})\n\n"
                f"Task: Generate a professional echocardiogram diagnostic report as an expert cardiologist. "
                f"Include findings, measurements, impression, and recommendations.\n\n"
                f"--- ECHOCARDIOGRAM REPORT ---\n"
                f"FINDINGS:\n"
                f"- Left Ventricular Ejection Fraction (LVEF): {ef:.1f}%\n"
                f"- Classification: {cat}\n"
                f"- Chamber Assessment: "
            )

            inputs = self.tokenizer(prompt, return_tensors="pt")
            text_embeddings = self.llm.transformer.wte(inputs.input_ids)

            combined = torch.cat([echo_soft_prompt[i:i+1], text_embeddings], dim=1)
            attention_mask = torch.ones(combined.shape[:2], dtype=torch.long)

            outputs = self.llm.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.2,
            )

            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Format clean report
            report = (
                f"ECHOCARDIOGRAM DIAGNOSTIC REPORT\n"
                f"{'='*40}\n"
                f"FINDINGS:\n"
                f"  - LVEF: {ef:.1f}% ({cat})\n"
                f"  - Chamber Assessment: {raw}\n\n"
                f"IMPRESSION:\n"
                f"  {'Reduced' if cat == 'HFrEF' else 'Mildly reduced' if cat == 'HFmrEF' else 'Preserved'} "
                f"left ventricular systolic function with LVEF {ef:.1f}%.\n\n"
                f"RECOMMENDATIONS:\n"
            )

            if cat == "HFrEF":
                report += (
                    "  - Initiate guideline-directed medical therapy (GDMT)\n"
                    "  - Consider ICD evaluation if LVEF ≤35%\n"
                    "  - Cardiac MRI for etiology workup\n"
                    "  - Cardiology referral recommended\n"
                )
            elif cat == "HFmrEF":
                report += (
                    "  - Serial echocardiographic monitoring (6-12 months)\n"
                    "  - Optimise medical therapy per ESC/ACC guidelines\n"
                    "  - Assess for reversible causes of LV dysfunction\n"
                )
            else:
                report += (
                    "  - Routine follow-up per clinical indication\n"
                    "  - If symptomatic, assess diastolic function\n"
                    "  - Repeat echo only if clinical status changes\n"
                )

            reports.append(report)

        return reports


# ---------------------------------------------------------------------------
# Full EchoNet-RAG Pipeline
# ---------------------------------------------------------------------------
class EchoNetRAGPipeline(nn.Module):
    """
    Complete pipeline: Video → 3D-CNN → EF prediction + RAG → Report
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.echo_model = EchoNetModel(embed_dim=embed_dim)
        self.retriever = EchoKnowledgeRetriever()
        self.report_gen = EchoReportGenerator(embed_dim=embed_dim)

    def forward(self, video: torch.Tensor) -> Dict[str, Any]:
        """
        Full inference pipeline.

        Args:
            video: (B, 1, T, H, W) echocardiogram tensor
        Returns:
            dict with: ef_predicted, ef_categories, reports, contexts, embeddings
        """
        # Stage 1: Encode video
        model_out = self.echo_model(video)
        embeddings = model_out["embeddings"]
        ef_pred = model_out["ef_predicted"]
        ef_cats = model_out["ef_categories"]

        # Stage 2: Retrieve guidelines
        contexts = [self.retriever.retrieve(cat) for cat in ef_cats]

        # Stage 3: Generate reports
        reports = self.report_gen.generate_report(
            embeddings, ef_pred, ef_cats, contexts
        )

        return {
            "ef_predicted": [round(ef.item(), 2) for ef in ef_pred],
            "ef_categories": ef_cats,
            "reports": reports,
            "contexts": contexts,
            "embeddings": embeddings,
        }

    def analyze_single(self, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """Convenience method for single-video inference from API."""
        self.eval()
        with torch.no_grad():
            start = time.time()
            result = self.forward(video_tensor)
            elapsed = round(time.time() - start, 2)

        return {
            "inference_time_seconds": elapsed,
            "ef_predicted": result["ef_predicted"][0],
            "ef_category": result["ef_categories"][0],
            "report": result["reports"][0],
            "retrieved_guidelines": result["contexts"][0],
        }


# ---------------------------------------------------------------------------
# Demo / fallback inference (no real video needed)
# ---------------------------------------------------------------------------
def demo_inference() -> Dict[str, Any]:
    """
    Run pipeline with a synthetic echo video tensor.
    Useful when the dataset isn't downloaded yet.
    """
    print("[EchoNet] Running demo inference with synthetic video...")
    dummy_video = torch.randn(1, 1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
    pipeline = get_echonet_pipeline()
    return pipeline.analyze_single(dummy_video)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_pipeline_instance: Optional[EchoNetRAGPipeline] = None

def get_echonet_pipeline() -> EchoNetRAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        print("[EchoNet] Initialising EchoNet-RAG pipeline...")
        _pipeline_instance = EchoNetRAGPipeline(embed_dim=384)
        # Load genuine trained EF weights if train_echonet.py has produced them.
        import os
        weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "data", "echonet_ef_model.pt")
        _pipeline_instance.encoder_is_trained = False
        if os.path.isfile(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location="cpu")
                if ckpt.get("metrics", {}).get("synthetic"):
                    print("[EchoNet] Refusing synthetic checkpoint — staying UNTRAINED.")
                else:
                    _pipeline_instance.echo_model.encoder.load_state_dict(ckpt["encoder"])
                    _pipeline_instance.echo_model.ef_head.load_state_dict(ckpt["ef_head"])
                    _pipeline_instance.encoder_is_trained = True
                    print(f"[EchoNet] Loaded TRAINED EF model "
                          f"(test MAE={ckpt.get('metrics', {}).get('test_mae')}).")
            except Exception as e:
                print(f"[EchoNet] Could not load trained weights ({e}); UNTRAINED.")
        else:
            print("[EchoNet] No trained weights found — UNTRAINED (run core/train_echonet.py).")
        _pipeline_instance.eval()
    return _pipeline_instance
