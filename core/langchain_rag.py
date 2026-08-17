"""
core/langchain_rag.py — LangChain RAG Pipeline for DeepCardio
==============================================================
Replaces the manual Milvus retrieval + GPT-2 generation with a proper
LangChain retrieval-augmented generation pipeline.

Architecture:
  Clinical Guidelines (text)
      ↓ HuggingFaceEmbeddings (all-MiniLM-L6-v2)
      ↓ FAISS / Chroma Vector Store
      ↓ LangChain Retriever (top-k semantic search)
      ↓ PromptTemplate + LLM (GPT-2 or HuggingFace pipeline)
      ↓ Clinical Report (text)

Usage:
    from core.langchain_rag import get_rag_chain, query_rag

    chain  = get_rag_chain()
    result = query_rag("What does an EF of 28% indicate?")
    print(result["answer"])
    print(result["source_documents"])

Install:
    pip install langchain langchain-community langchain-huggingface faiss-cpu
"""

import os
import threading
from typing import List, Dict, Any, Optional

from core.logger import get_logger, LoggedTimer

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Clinical knowledge base — seeded at startup
# ──────────────────────────────────────────────────────────────────────────────

CLINICAL_DOCUMENTS = [
    # ── ECG / Arrhythmia ─────────────────────────────────────────────────────
    {
        "content": (
            "Normal Sinus Rhythm (NSR): Regular rhythm with rate 60-100 bpm. "
            "P waves precede every QRS complex. PR interval 0.12-0.20s. "
            "QRS duration < 0.12s. No treatment required for NSR."
        ),
        "source": "AHA ECG Guidelines 2023",
        "category": "ecg",
    },
    {
        "content": (
            "Atrial Fibrillation (AF): Irregularly irregular rhythm. Absent P waves replaced "
            "by fibrillatory baseline. Ventricular rate typically 100-160 bpm if uncontrolled. "
            "CHA2DS2-VASc score guides anticoagulation. Rate control target < 110 bpm at rest."
        ),
        "source": "ESC AF Guidelines 2023",
        "category": "ecg",
    },
    {
        "content": (
            "Ventricular Tachycardia (VT): Wide QRS tachycardia > 100 bpm (typically 140-200). "
            "AV dissociation, fusion beats, capture beats are diagnostic. "
            "Sustained VT (>30s) is time-critical and warrants prompt clinician assessment. "
            "Reference (ACC/AHA 2017): antiarrhythmic selection and dosing for haemodynamically "
            "stable VT are clinician decisions and are not recommended by this system."
        ),
        "source": "ACC/AHA Ventricular Arrhythmia Guidelines 2017",
        "category": "ecg",
    },
    {
        "content": (
            "Ventricular Fibrillation (VF): Chaotic irregular waveform, no organised QRS complexes. "
            "This pattern may correspond to a shockable rhythm and to cardiac arrest; clinician "
            "confirmation is required before any resuscitation decision. Reference (AHA ACLS 2020): "
            "resuscitation, shock delivery and medication dosing are clinician-directed and are not "
            "ordered by this system."
        ),
        "source": "AHA ACLS Guidelines 2020",
        "category": "ecg",
    },
    {
        "content": (
            "ST-Elevation Myocardial Infarction (STEMI): ST elevation ≥ 1mm in ≥ 2 contiguous limb leads "
            "or ≥ 2mm in precordial leads. New LBBB equivalent. Reference (ACC/AHA STEMI, updated 2022): "
            "a door-to-balloon time goal under 90 minutes and dual antiplatelet therapy are documented "
            "standards of care; reperfusion suitability and antiplatelet choice are clinician decisions "
            "and are not ordered by this system."
        ),
        "source": "ACC/AHA STEMI Guidelines 2013 (Updated 2022)",
        "category": "ecg",
    },
    # ── Echocardiography / EF ─────────────────────────────────────────────────
    {
        "content": (
            "Heart Failure with Reduced Ejection Fraction (HFrEF): EF < 40%. "
            "Guideline-directed medical therapy (GDMT): ACE inhibitor/ARB/ARNI + beta-blocker + "
            "mineralocorticoid receptor antagonist + SGLT2 inhibitor. "
            "Target resting HR < 70 bpm. Consider ICD if EF < 35% despite optimal therapy."
        ),
        "source": "ESC Heart Failure Guidelines 2021",
        "category": "echo",
    },
    {
        "content": (
            "Heart Failure with Mid-Range EF (HFmrEF): EF 40-49%. "
            "Treatment similar to HFrEF but evidence base is less robust. "
            "Diuretics for fluid overload. Treat underlying cause (ischaemia, hypertension, AF). "
            "Regular echo follow-up every 3-6 months."
        ),
        "source": "ESC Heart Failure Guidelines 2021",
        "category": "echo",
    },
    {
        "content": (
            "Preserved Ejection Fraction (HFpEF): EF ≥ 50% with symptoms of heart failure. "
            "No specific disease-modifying therapy proven until EMPEROR-Preserved (empagliflozin). "
            "Treat comorbidities: hypertension, AF, obesity, diabetes. "
            "Diuretics for congestion. Limit sodium intake < 2g/day."
        ),
        "source": "AHA/ACC HFpEF 2022",
        "category": "echo",
    },
    {
        "content": (
            "Echocardiographic assessment of EF: Biplane Simpson's method is standard. "
            "Normal EF: 52-72% (men), 54-74% (women). Mildly reduced: 41-51%. "
            "Moderately reduced: 30-40%. Severely reduced: < 30%. "
            "3D echocardiography provides most accurate EF with lowest inter-observer variability."
        ),
        "source": "ASE Chamber Quantification Guidelines 2015",
        "category": "echo",
    },
    # ── Heart Sounds / Murmurs ────────────────────────────────────────────────
    {
        "content": (
            "Aortic Stenosis (AS): Crescendo-decrescendo systolic murmur at right upper sternal border. "
            "Radiates to carotids. Severe AS: valve area < 1 cm², mean gradient > 40 mmHg. "
            "Aortic valve replacement (TAVR or SAVR) indicated for symptomatic severe AS. "
            "Classic triad: angina, syncope, dyspnoea."
        ),
        "source": "ACC/AHA Valvular Heart Disease Guidelines 2021",
        "category": "heart_sound",
    },
    {
        "content": (
            "Mitral Regurgitation (MR): Holosystolic murmur at apex radiating to axilla. "
            "Chronic severe MR: surgical repair preferred over replacement when feasible. "
            "Indications: symptoms, EF ≤ 60%, LV end-systolic diameter ≥ 40 mm. "
            "MitraClip (TEER) for high surgical risk patients."
        ),
        "source": "ESC Valvular Heart Disease Guidelines 2021",
        "category": "heart_sound",
    },
    # ── Arrhythmia Video / Cardiac Motion ────────────────────────────────────
    {
        "content": (
            "Cardiac arrhythmia on echocardiography: Irregular wall motion and variable chamber filling "
            "suggest underlying rhythm disorder. Atrial fibrillation causes variable diastolic filling times "
            "and beat-to-beat EF variation > 10%. Coordinate with 12-lead ECG for definitive rhythm diagnosis."
        ),
        "source": "ASE Echo Guidelines 2022",
        "category": "arrhythmia_video",
    },
    {
        "content": (
            "Ventricular arrhythmia on echocardiography: Paradoxical septal motion, focal wall motion "
            "abnormalities, and reduced EF variability may indicate ventricular arrhythmia substrate. "
            "Scar tissue from prior MI is the most common arrhythmia substrate (scar-mediated VT). "
            "Cardiac MRI with LGE for definitive scar characterisation."
        ),
        "source": "HRS Expert Consensus Statement on VT 2019",
        "category": "arrhythmia_video",
    },
    # ── Arthritis / Blood markers ─────────────────────────────────────────────
    {
        "content": (
            "Rheumatoid Arthritis (RA) diagnosis: 2010 ACR/EULAR criteria. "
            "Positive RF and/or anti-CCP antibodies, elevated ESR and CRP, synovitis in ≥ 1 joint. "
            "Early aggressive treatment with DMARDs (methotrexate first-line). "
            "Biologic agents (TNF inhibitors, IL-6 inhibitors) for refractory disease."
        ),
        "source": "ACR/EULAR RA Guidelines 2021",
        "category": "arthritis",
    },
    {
        "content": (
            "Inflammatory markers in arthritis: ESR normal < 20 mm/hr (men), < 30 mm/hr (women). "
            "CRP normal < 10 mg/L. Both elevated in active inflammatory arthritis. "
            "Uric acid > 6.8 mg/dL associated with gout; > 10 mg/dL high risk of acute flares. "
            "Serum calcium elevation may suggest sarcoidosis-related arthropathy."
        ),
        "source": "ACR Laboratory Testing Guidelines 2022",
        "category": "arthritis",
    },
    {
        "content": (
            "Complete blood count in rheumatic disease: Anaemia of chronic disease (normocytic, "
            "low reticulocytes) common in RA and SLE. Hb < 12 g/dL (women), < 13 g/dL (men). "
            "Thrombocytosis may indicate active inflammation. Leucopenia suggests SLE or drug toxicity. "
            "Eosinophilia points to eosinophilic granulomatosis with polyangiitis."
        ),
        "source": "ACR Haematology Guidelines 2022",
        "category": "arthritis",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Vector store builder
# ──────────────────────────────────────────────────────────────────────────────

_rag_chain_instance = None
_rag_lock = threading.Lock()


def _build_vector_store():
    """
    Build a FAISS vector store from CLINICAL_DOCUMENTS using HuggingFace embeddings.
    Falls back to a simple keyword-based retriever if LangChain is not installed.
    """
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.schema import Document

        logger.info("Building LangChain FAISS vector store with HuggingFace embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        docs = [
            Document(
                page_content=d["content"],
                metadata={"source": d["source"], "category": d["category"]},
            )
            for d in CLINICAL_DOCUMENTS
        ]

        store = FAISS.from_documents(docs, embeddings)
        logger.info(f"FAISS store built with {len(docs)} clinical documents")
        return store, embeddings

    except ImportError as e:
        logger.warning(f"LangChain not installed: {e}. Using fallback keyword retriever.")
        return None, None


def _build_llm():
    """
    Build a LangChain-compatible LLM wrapper.
    Tries GPT-2 (local) → fallback to template-based generation.
    """
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline

        logger.info("Loading GPT-2 as LangChain LLM...")
        pipe = hf_pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("GPT-2 LangChain LLM ready")
        return llm

    except Exception as e:
        logger.warning(f"Could not load GPT-2 LLM: {e}. Using template-based fallback.")
        return None


def _build_rag_chain(store, llm):
    """Assemble the LangChain RetrievalQA chain."""
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    prompt_template = """You are a senior cardiologist providing clinical decision support.
Use the following retrieved clinical guidelines to answer the question accurately and concisely.

Retrieved Guidelines:
{context}

Clinical Question: {question}

Clinical Assessment:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    logger.info("LangChain RetrievalQA chain assembled")
    return chain


# ──────────────────────────────────────────────────────────────────────────────
# Fallback keyword retriever (when LangChain not installed)
# ──────────────────────────────────────────────────────────────────────────────

class _FallbackRetriever:
    """Simple keyword-based retriever used when LangChain packages are absent."""

    def retrieve(self, query: str, category: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        query_lower = query.lower()
        scored = []
        for doc in CLINICAL_DOCUMENTS:
            if category and doc["category"] != category:
                continue
            score = sum(1 for w in query_lower.split() if w in doc["content"].lower())
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def generate_report(self, query: str, retrieved: List[Dict]) -> str:
        context = "\n".join(f"- {d['content'][:150]}..." for d in retrieved)
        return (
            f"Clinical Assessment (Keyword RAG):\n\n"
            f"Query: {query}\n\n"
            f"Supporting Guidelines:\n{context}\n\n"
            f"Note: Install langchain + langchain-community for full LLM-augmented reports."
        )


_fallback = _FallbackRetriever()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_rag_chain():
    """
    Return the LangChain RetrievalQA chain (singleton, thread-safe).
    Returns None if LangChain is not installed.
    """
    global _rag_chain_instance
    with _rag_lock:
        if _rag_chain_instance is not None:
            return _rag_chain_instance

        store, embeddings = _build_vector_store()
        if store is None:
            return None

        llm = _build_llm()
        if llm is None:
            return None

        try:
            _rag_chain_instance = _build_rag_chain(store, llm)
        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}", exc_info=True)
            _rag_chain_instance = None

        return _rag_chain_instance


def _gate_rag_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply directive->advisory reframing to everything query_rag() hands back.

    Peer-review M1(c): the system must not issue treatment orders. Two surfaces need
    covering — the retrieved `source_documents`, and the `answer`, which on the
    LangChain path is LLM-generated text and is therefore the one surface no amount
    of corpus curation can make safe in advance. reframe_directives() is idempotent,
    so already-advisory text passes through untouched.
    """
    from core.safety_gating import reframe_directives

    result["answer"] = reframe_directives(result.get("answer", ""))
    for doc in result.get("source_documents", []):
        if isinstance(doc, dict) and "content" in doc:
            doc["content"] = reframe_directives(doc["content"])
    return result


def query_rag(question: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Query the RAG pipeline with a clinical question.

    Args:
        question: Clinical question or patient finding
        category: Optional filter — "ecg" | "echo" | "heart_sound" |
                  "arrhythmia_video" | "arthritis"

    Returns:
        {
          "answer":           str,
          "source_documents": List[str],
          "backend":          "langchain" | "fallback",
        }
    """
    with LoggedTimer(logger, "RAG query", extra={"question": question[:60]}):
        chain = get_rag_chain()

        if chain is not None:
            try:
                result = chain.invoke({"query": question})
                sources = [
                    {
                        "content":  doc.page_content[:200],
                        "source":   doc.metadata.get("source", ""),
                        "category": doc.metadata.get("category", ""),
                    }
                    for doc in result.get("source_documents", [])
                ]
                return _gate_rag_result({
                    "answer":           result.get("result", ""),
                    "source_documents": sources,
                    "backend":          "langchain",
                })
            except Exception as e:
                logger.warning(f"LangChain chain failed: {e} — using fallback")

        # Fallback keyword retrieval
        retrieved = _fallback.retrieve(question, category=category)
        report    = _fallback.generate_report(question, retrieved)
        return _gate_rag_result({
            "answer":           report,
            "source_documents": [{"content": d["content"][:200], "source": d["source"],
                                  "category": d["category"]} for d in retrieved],
            "backend":          "fallback",
        })


def retrieve_guidelines(query: str, category: Optional[str] = None, top_k: int = 3) -> List[str]:
    """
    Lightweight retrieval — returns only guideline text strings.
    Drop-in replacement for the existing mock guidelines used throughout main.py.

    Peer-review M1(c): every string leaving this function passes through
    core/safety_gating.reframe_guideline_list(), so an imperative treatment order can
    never reach a caller. CLINICAL_DOCUMENTS above is already written as advisory
    text; this pass is the net for retrieved content that is NOT curated here (Milvus
    corpus documents, future additions), since reframe_directives() is idempotent and
    leaves already-advisory text untouched.
    """
    from core.safety_gating import reframe_guideline_list

    result = query_rag(query, category=category)
    sources = result.get("source_documents", [])[:top_k]
    if sources:
        return reframe_guideline_list([s["content"] for s in sources])
    return [f"No guidelines retrieved for: {query[:60]}"]


def get_vector_store_stats() -> Dict[str, Any]:
    """Return statistics about the loaded vector store."""
    chain = get_rag_chain()
    return {
        "backend":             "langchain" if chain is not None else "fallback_keyword",
        "num_documents":       len(CLINICAL_DOCUMENTS),
        "categories":          list({d["category"] for d in CLINICAL_DOCUMENTS}),
        "embedding_model":     "sentence-transformers/all-MiniLM-L6-v2",
        "llm":                 "GPT-2 (local)" if chain is not None else "Template (no LLM)",
        "langchain_available": chain is not None,
    }
