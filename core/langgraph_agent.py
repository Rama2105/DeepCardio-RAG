"""
core/langgraph_agent.py — LangGraph Multi-Modal Cardiac Diagnostic Agent
=========================================================================
Implements a stateful agentic workflow using LangGraph's StateGraph.

Graph Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                   SUPERVISOR NODE                   │
  │  Routes input to the correct specialist nodes       │
  └──────┬──────┬──────┬──────┬──────┬──────────────────┘
         ↓      ↓      ↓      ↓      ↓
    [ECG]  [ECHO] [SOUND] [ARRH] [ARTHR]   ← Specialist nodes
         ↓      ↓      ↓      ↓      ↓
  ┌─────────────────────────────────────────────────────┐
  │                  AGGREGATOR NODE                    │
  │  Synthesises findings → final clinical report       │
  └─────────────────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────────────────┐
  │                  RAG ENRICHMENT NODE                │
  │  Retrieves relevant guidelines via LangChain RAG    │
  └─────────────────────────────────────────────────────┘

Usage:
    from core.langgraph_agent import run_diagnostic_agent

    state = run_diagnostic_agent({
        "patient_id":   "APD-0001",
        "ecg_tensor":   ecg_tensor,       # optional
        "echo_tensor":  echo_tensor,      # optional
        "mel_tensor":   mel_tensor,       # optional
        "arrh_tensor":  video_tensor,     # optional
        "patient_data": {...},            # optional (arthritis)
    })
    print(state["final_report"])

Install:
    pip install langgraph langchain langchain-community
"""

import time
import threading
from typing import Dict, Any, List, Optional, TypedDict, Annotated

from core.logger import get_logger, LoggedTimer

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Agent State Schema
# ──────────────────────────────────────────────────────────────────────────────

class CardioAgentState(TypedDict, total=False):
    """
    Shared state passed between all nodes in the LangGraph workflow.
    Each node reads relevant fields and writes its findings back.
    """
    # ── Inputs ────────────────────────────────────────────────────────────────
    patient_id:     str
    ecg_tensor:     Any   # torch.Tensor (1, 12, 1250)
    echo_tensor:    Any   # torch.Tensor (1, 1, 32, 112, 112)
    mel_tensor:     Any   # torch.Tensor (1, 1, 64, 156)
    arrh_tensor:    Any   # torch.Tensor (1, 1, 32, 112, 112) — arrhythmia video
    patient_data:   Dict[str, Any]  # arthritis blood test fields

    # ── Routing ───────────────────────────────────────────────────────────────
    active_modalities:  List[str]   # which nodes will run
    completed_nodes:    List[str]

    # ── Node findings ─────────────────────────────────────────────────────────
    ecg_findings:        Dict[str, Any]
    echo_findings:       Dict[str, Any]
    heart_sound_findings:Dict[str, Any]
    arrhythmia_findings: Dict[str, Any]
    arthritis_findings:  Dict[str, Any]

    # ── RAG context ───────────────────────────────────────────────────────────
    retrieved_guidelines: List[str]
    rag_backend:          str

    # ── Final output ──────────────────────────────────────────────────────────
    final_report:        str
    risk_level:          str    # "Low" | "Moderate" | "High" | "Critical"
    recommendations:     List[str]
    inference_time_ms:   float
    error:               Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# Node implementations
# ──────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: CardioAgentState) -> CardioAgentState:
    """
    Supervisor: inspects available inputs and sets active_modalities.
    Decides which specialist nodes should run.
    """
    active = []
    if state.get("ecg_tensor")    is not None: active.append("ecg")
    if state.get("echo_tensor")   is not None: active.append("echo")
    if state.get("mel_tensor")    is not None: active.append("heart_sound")
    if state.get("arrh_tensor")   is not None: active.append("arrhythmia")
    if state.get("patient_data")              : active.append("arthritis")

    if not active:
        active = ["ecg"]   # default to demo ECG mode

    logger.info(
        "Supervisor routing",
        extra={"patient_id": state.get("patient_id", "unknown"), "modalities": active}
    )
    return {**state, "active_modalities": active, "completed_nodes": []}


def ecg_node(state: CardioAgentState) -> CardioAgentState:
    """ECG analysis node — runs 1D-CNN ECG encoder."""
    try:
        import torch
        from core.pipeline import get_model

        tensor = state.get("ecg_tensor")
        if tensor is None:
            tensor = torch.randn(1, 12, 1250)   # synthetic demo

        model   = get_model()
        results = model(tensor)
        findings = {
            "report":      results.get("reports", ["No report generated"])[0],
            "contexts":    results.get("contexts", []),
            "modality":    "ECG Signal",
            "status":      "analysed",
        }
        logger.info("ECG node completed", extra={"patient_id": state.get("patient_id")})

    except Exception as e:
        logger.error(f"ECG node failed: {e}")
        findings = {"status": "failed", "error": str(e), "modality": "ECG Signal"}

    completed = list(state.get("completed_nodes", [])) + ["ecg"]
    return {**state, "ecg_findings": findings, "completed_nodes": completed}


def echo_node(state: CardioAgentState) -> CardioAgentState:
    """Echo video analysis node — runs EchoNet 3D-CNN."""
    try:
        import torch
        from core.echonet_pipeline import get_echonet_pipeline, demo_inference

        tensor = state.get("echo_tensor")
        if tensor is None:
            result = demo_inference()
        else:
            pipeline = get_echonet_pipeline()
            result   = pipeline.analyze_single(tensor)

        findings = {
            "ef_predicted":        result.get("ef_predicted", 0.0),
            "ef_category":         result.get("ef_category", "Unknown"),
            "report":              result.get("report", ""),
            "retrieved_guidelines":result.get("retrieved_guidelines", []),
            "modality":            "Echocardiogram Video",
            "status":              "analysed",
        }
        logger.info("Echo node completed", extra={"ef": findings.get("ef_predicted")})

    except Exception as e:
        logger.error(f"Echo node failed: {e}")
        findings = {"status": "failed", "error": str(e), "modality": "Echocardiogram"}

    completed = list(state.get("completed_nodes", [])) + ["echo"]
    return {**state, "echo_findings": findings, "completed_nodes": completed}


def heart_sound_node(state: CardioAgentState) -> CardioAgentState:
    """Heart sound analysis node — runs mel-spectrogram 2D-CNN."""
    try:
        import torch
        from core.heart_sound_loader import get_heart_sound_classifier, HEART_SOUND_GUIDELINES, MURMUR_CLASSES

        tensor = state.get("mel_tensor")
        if tensor is None:
            tensor = torch.randn(1, 1, 64, 156)

        clf     = get_heart_sound_classifier()
        results = clf.predict(tensor)
        result  = results[0]
        guidelines = HEART_SOUND_GUIDELINES.get(result.get("murmur_class", ""), [])

        findings = {
            **result,
            "retrieved_guidelines": guidelines,
            "modality":             "Heart Sound (PCG)",
            "status":               "analysed",
        }
        logger.info("Heart sound node completed", extra={"murmur": result.get("murmur_class")})

    except Exception as e:
        logger.error(f"Heart sound node failed: {e}")
        findings = {"status": "failed", "error": str(e), "modality": "Heart Sound"}

    completed = list(state.get("completed_nodes", [])) + ["heart_sound"]
    return {**state, "heart_sound_findings": findings, "completed_nodes": completed}


def arrhythmia_node(state: CardioAgentState) -> CardioAgentState:
    """Arrhythmia video classification node — runs 3D-CNN."""
    try:
        import torch
        from core.cardiac_arrhythmia_video_loader import (
            get_arrhythmia_video_classifier, get_arrhythmia_video_dataset
        )

        tensor = state.get("arrh_tensor")
        if tensor is None:
            ds = get_arrhythmia_video_dataset()
            if len(ds) > 0:
                tensor, _ = ds.load_video_tensor(0)
                tensor = tensor.unsqueeze(0).float()
            else:
                tensor = torch.randn(1, 1, 32, 112, 112)

        clf    = get_arrhythmia_video_classifier()
        result = clf.predict(tensor)

        findings = {
            **result,
            "modality": "Arrhythmia Video (Echo)",
            "status":   "analysed",
        }
        logger.info("Arrhythmia node completed", extra={"label": result.get("label")})

    except Exception as e:
        logger.error(f"Arrhythmia node failed: {e}")
        findings = {"status": "failed", "error": str(e), "modality": "Arrhythmia Video"}

    completed = list(state.get("completed_nodes", [])) + ["arrhythmia"]
    return {**state, "arrhythmia_findings": findings, "completed_nodes": completed}


def arthritis_node(state: CardioAgentState) -> CardioAgentState:
    """Arthritis risk prediction node — runs Tabular BERT + MoE."""
    try:
        from core.arthritis_pipeline import get_arthritis_predictor

        patient_data = state.get("patient_data", {})
        predictor    = get_arthritis_predictor()
        result       = predictor.predict(patient_data)

        findings = {
            **result,
            "modality": "Blood Test / Arthritis Risk",
            "status":   "analysed",
        }
        logger.info("Arthritis node completed")

    except Exception as e:
        logger.error(f"Arthritis node failed: {e}")
        findings = {"status": "failed", "error": str(e), "modality": "Arthritis Risk"}

    completed = list(state.get("completed_nodes", [])) + ["arthritis"]
    return {**state, "arthritis_findings": findings, "completed_nodes": completed}


def rag_enrichment_node(state: CardioAgentState) -> CardioAgentState:
    """
    RAG Enrichment: queries LangChain RAG with aggregated findings
    to retrieve the most relevant clinical guidelines.
    """
    from core.langchain_rag import retrieve_guidelines, get_vector_store_stats

    # Build a query from available findings
    query_parts = []
    if state.get("ecg_findings", {}).get("report"):
        query_parts.append("ECG: " + str(state["ecg_findings"]["report"])[:100])
    if state.get("echo_findings", {}).get("ef_category"):
        query_parts.append(f"EF category: {state['echo_findings']['ef_category']}")
    if state.get("arrhythmia_findings", {}).get("label"):
        query_parts.append(f"Arrhythmia: {state['arrhythmia_findings']['label']}")
    if state.get("heart_sound_findings", {}).get("murmur_class"):
        query_parts.append(f"Murmur: {state['heart_sound_findings']['murmur_class']}")
    if not query_parts:
        query_parts.append("cardiac assessment guidelines")

    query    = ". ".join(query_parts)
    stats    = get_vector_store_stats()
    guidelines = retrieve_guidelines(query, top_k=4)

    logger.info(
        "RAG enrichment completed",
        extra={"backend": stats["backend"], "n_guidelines": len(guidelines)}
    )
    return {
        **state,
        "retrieved_guidelines": guidelines,
        "rag_backend": stats["backend"],
    }


def aggregator_node(state: CardioAgentState) -> CardioAgentState:
    """
    Aggregator: synthesises all modality findings into a single
    clinical report with risk stratification and recommendations.
    """
    sections = []
    recommendations = []
    risk_scores = []

    # ── ECG ──────────────────────────────────────────────────────────────────
    ecg = state.get("ecg_findings", {})
    if ecg.get("status") == "analysed":
        sections.append(f"ECG Analysis:\n{ecg.get('report', 'No report')[:300]}")

    # ── Echo ──────────────────────────────────────────────────────────────────
    echo = state.get("echo_findings", {})
    if echo.get("status") == "analysed":
        ef   = echo.get("ef_predicted", 0)
        cat  = echo.get("ef_category", "Unknown")
        sections.append(f"Echocardiogram: EF = {ef:.1f}% — {cat}")
        if ef < 35:
            risk_scores.append(3)
            recommendations.append("Urgent cardiology referral for HFrEF management (EF < 35%)")
            recommendations.append("Start GDMT: ACE inhibitor + beta-blocker + MRA + SGLT2i")
            recommendations.append("Consider ICD evaluation if EF remains < 35%")
        elif ef < 50:
            risk_scores.append(2)
            recommendations.append("Follow-up echocardiogram in 3 months for HFmrEF")
        else:
            risk_scores.append(1)
            recommendations.append("EF within normal range — routine follow-up")

    # ── Arrhythmia video ─────────────────────────────────────────────────────
    arrh = state.get("arrhythmia_findings", {})
    if arrh.get("status") == "analysed":
        label = arrh.get("label", "Unknown")
        conf  = arrh.get("confidence", 0)
        sections.append(f"Arrhythmia Video Analysis: {label} (confidence {conf:.0%})")
        if label == "Arrhythmia":
            risk_scores.append(3)
            recommendations.append("12-lead ECG and 24-hour Holter monitor urgently")
            recommendations.append("Assess for atrial fibrillation — check CHA2DS2-VASc score")
        else:
            risk_scores.append(1)

    # ── Heart Sound ───────────────────────────────────────────────────────────
    hs = state.get("heart_sound_findings", {})
    if hs.get("status") == "analysed":
        murmur = hs.get("murmur_class", "Unknown")
        sections.append(f"Heart Sound Analysis: Murmur — {murmur}")
        if murmur == "Present":
            risk_scores.append(2)
            recommendations.append("Echocardiogram for valvular assessment")

    # ── Arthritis ─────────────────────────────────────────────────────────────
    art = state.get("arthritis_findings", {})
    if art.get("status") == "analysed":
        risk = art.get("risk_label", art.get("prediction", "Unknown"))
        sections.append(f"Arthritis Risk Assessment: {risk}")
        if "high" in str(risk).lower():
            risk_scores.append(2)
            recommendations.append("Rheumatology referral — consider DMARD initiation")

    # ── RAG Guidelines ────────────────────────────────────────────────────────
    guidelines = state.get("retrieved_guidelines", [])
    if guidelines:
        sections.append(
            "Relevant Clinical Guidelines (via LangChain RAG):\n" +
            "\n".join(f"  • {g[:150]}" for g in guidelines[:3])
        )

    # ── Risk stratification ───────────────────────────────────────────────────
    if risk_scores:
        avg = sum(risk_scores) / len(risk_scores)
        if avg >= 2.5:   risk_level = "Critical"
        elif avg >= 2.0: risk_level = "High"
        elif avg >= 1.5: risk_level = "Moderate"
        else:            risk_level = "Low"
    else:
        risk_level = "Unknown"

    if not recommendations:
        recommendations.append("Continue routine cardiac monitoring")
        recommendations.append("Annual follow-up with primary care physician")

    # ── Compose final report ──────────────────────────────────────────────────
    patient_id = state.get("patient_id", "Unknown")
    modalities = ", ".join(state.get("active_modalities", []))
    report = (
        f"DEEPCARDIO-RAG MULTI-MODAL DIAGNOSTIC REPORT\n"
        f"{'='*55}\n"
        f"Patient: {patient_id}\n"
        f"Modalities Analysed: {modalities}\n"
        f"Overall Risk Level: {risk_level}\n"
        f"RAG Backend: {state.get('rag_backend', 'fallback')}\n"
        f"{'='*55}\n\n"
        + "\n\n".join(sections)
        + f"\n\n{'─'*55}\n"
        f"RECOMMENDATIONS:\n"
        + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(recommendations))
        + f"\n\n⚠ RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.\n"
    )

    logger.info(
        "Aggregator completed",
        extra={"risk_level": risk_level, "n_recommendations": len(recommendations)}
    )

    return {
        **state,
        "final_report":    report,
        "risk_level":      risk_level,
        "recommendations": recommendations,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph Workflow Builder
# ──────────────────────────────────────────────────────────────────────────────

_graph_instance = None
_graph_lock     = threading.Lock()


def _build_graph():
    """
    Build and compile the LangGraph StateGraph.
    Falls back to a simple sequential runner if LangGraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END

        g = StateGraph(CardioAgentState)

        # Register nodes
        g.add_node("supervisor",    supervisor_node)
        g.add_node("ecg",           ecg_node)
        g.add_node("echo",          echo_node)
        g.add_node("heart_sound",   heart_sound_node)
        g.add_node("arrhythmia",    arrhythmia_node)
        g.add_node("arthritis",     arthritis_node)
        g.add_node("rag_enrichment",rag_enrichment_node)
        g.add_node("aggregator",    aggregator_node)

        # Supervisor → conditional fan-out
        def route_from_supervisor(state: CardioAgentState):
            """Return list of next nodes based on active_modalities."""
            active = state.get("active_modalities", ["ecg"])
            # Map modality → node name
            node_map = {
                "ecg":         "ecg",
                "echo":        "echo",
                "heart_sound": "heart_sound",
                "arrhythmia":  "arrhythmia",
                "arthritis":   "arthritis",
            }
            return [node_map[m] for m in active if m in node_map] or ["ecg"]

        g.add_conditional_edges("supervisor", route_from_supervisor)

        # All specialist nodes → RAG enrichment
        for node in ("ecg", "echo", "heart_sound", "arrhythmia", "arthritis"):
            g.add_edge(node, "rag_enrichment")

        # RAG → Aggregator → END
        # (LangGraph merges parallel paths automatically before continuing)
        g.add_edge("rag_enrichment", "aggregator")
        g.add_edge("aggregator",     END)

        # Entry point
        g.set_entry_point("supervisor")

        compiled = g.compile()
        logger.info("LangGraph StateGraph compiled successfully")
        return compiled

    except ImportError as e:
        logger.warning(f"LangGraph not installed: {e}. Using sequential fallback runner.")
        return None


def get_graph():
    """Return the compiled LangGraph (singleton, thread-safe). May be None."""
    global _graph_instance
    with _graph_lock:
        if _graph_instance is None:
            _graph_instance = _build_graph()
    return _graph_instance


# ──────────────────────────────────────────────────────────────────────────────
# Fallback sequential runner (when LangGraph not installed)
# ──────────────────────────────────────────────────────────────────────────────

def _run_sequential(initial_state: CardioAgentState) -> CardioAgentState:
    """Execute nodes sequentially without LangGraph — same results, no parallelism."""
    state = supervisor_node(initial_state)
    active = state.get("active_modalities", ["ecg"])

    node_fns = {
        "ecg":         ecg_node,
        "echo":        echo_node,
        "heart_sound": heart_sound_node,
        "arrhythmia":  arrhythmia_node,
        "arthritis":   arthritis_node,
    }
    for mod in active:
        if mod in node_fns:
            state = node_fns[mod](state)

    state = rag_enrichment_node(state)
    state = aggregator_node(state)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_diagnostic_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full multi-modal diagnostic agent.

    Args:
        inputs: Dict with any combination of:
            patient_id   (str)
            ecg_tensor   (torch.Tensor)
            echo_tensor  (torch.Tensor)
            mel_tensor   (torch.Tensor)
            arrh_tensor  (torch.Tensor)
            patient_data (dict)

    Returns:
        Final agent state including:
            final_report, risk_level, recommendations,
            ecg_findings, echo_findings, arrhythmia_findings,
            heart_sound_findings, arthritis_findings,
            retrieved_guidelines, rag_backend,
            inference_time_ms, backend
    """
    t0 = time.perf_counter()

    initial_state: CardioAgentState = {
        "patient_id":             inputs.get("patient_id", "UNKNOWN"),
        "ecg_tensor":             inputs.get("ecg_tensor"),
        "echo_tensor":            inputs.get("echo_tensor"),
        "mel_tensor":             inputs.get("mel_tensor"),
        "arrh_tensor":            inputs.get("arrh_tensor"),
        "patient_data":           inputs.get("patient_data", {}),
        "active_modalities":      [],
        "completed_nodes":        [],
        "ecg_findings":           {},
        "echo_findings":          {},
        "heart_sound_findings":   {},
        "arrhythmia_findings":    {},
        "arthritis_findings":     {},
        "retrieved_guidelines":   [],
        "rag_backend":            "fallback",
        "final_report":           "",
        "risk_level":             "Unknown",
        "recommendations":        [],
        "inference_time_ms":      0.0,
        "error":                  None,
    }

    graph = get_graph()

    try:
        with LoggedTimer(logger, "LangGraph diagnostic agent",
                         extra={"patient_id": initial_state["patient_id"]}):
            if graph is not None:
                final_state = graph.invoke(initial_state)
                backend = "langgraph"
            else:
                final_state = _run_sequential(initial_state)
                backend = "sequential_fallback"

    except Exception as e:
        logger.error(f"Diagnostic agent failed: {e}", exc_info=True)
        final_state = {**initial_state, "final_report": f"Agent error: {e}", "error": str(e)}
        backend = "error"

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {**final_state, "inference_time_ms": elapsed_ms, "backend": backend}


def get_graph_info() -> Dict[str, Any]:
    """Return information about the current graph state."""
    graph = get_graph()
    return {
        "langgraph_available":  graph is not None,
        "backend":              "langgraph" if graph is not None else "sequential_fallback",
        "nodes": [
            "supervisor", "ecg", "echo", "heart_sound",
            "arrhythmia", "arthritis", "rag_enrichment", "aggregator"
        ],
        "supported_modalities": ["ecg", "echo", "heart_sound", "arrhythmia", "arthritis"],
        "description": (
            "Multi-modal cardiac diagnostic agent. "
            "Supervisor routes inputs → specialist nodes run in parallel → "
            "RAG enriches with clinical guidelines → aggregator synthesises final report."
        ),
    }
