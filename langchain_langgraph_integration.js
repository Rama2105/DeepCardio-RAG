const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, ExternalHyperlink,
  TabStopType, TabStopPosition
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}

function para(text, opts = {}) {
  const runs = typeof text === "string"
    ? [new TextRun({ text, ...opts })]
    : text;
  return new Paragraph({ children: runs, spacing: { after: 120 } });
}

function boldPara(label, value) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: label, bold: true }),
      new TextRun({ text: value }),
    ],
  });
}

function codePara(text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Consolas", size: 18, color: "1A1A2E" })],
  });
}

function codeBlock(lines) {
  return lines.map(l => codePara(l));
}

function makeTableRow(cells, isHeader = false) {
  return new TableRow({
    children: cells.map((text, i) => {
      const widths = cells.length === 2 ? [4680, 4680] : cells.length === 3 ? [2400, 3480, 3480] : [2340, 2340, 2340, 2340];
      return new TableCell({
        borders,
        width: { size: widths[i] || 2340, type: WidthType.DXA },
        shading: isHeader ? { fill: "1A365D", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
        margins: cellMargins,
        verticalAlign: "center",
        children: [new Paragraph({
          children: [new TextRun({
            text, bold: isHeader, color: isHeader ? "FFFFFF" : "333333", font: "Arial", size: 20
          })]
        })]
      });
    })
  });
}

function simpleTable(headers, rows) {
  const colCount = headers.length;
  const totalWidth = 9360;
  const colWidth = Math.floor(totalWidth / colCount);
  const widths = headers.map(() => colWidth);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      makeTableRow(headers, true),
      ...rows.map(r => makeTableRow(r, false))
    ]
  });
}

function spacer() {
  return new Paragraph({ spacing: { after: 200 }, children: [] });
}

// ===== BUILD DOCUMENT =====
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1A365D" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2B6CB0" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "3182CE" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
      ]},
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      ]},
    ]
  },
  sections: [
    // ===== COVER PAGE =====
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      children: [
        new Paragraph({ spacing: { before: 3000 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "DeepCardio-RAG", size: 56, bold: true, color: "1A365D", font: "Arial" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
          children: [new TextRun({ text: "LangChain & LangGraph Integration Architecture", size: 36, color: "2B6CB0", font: "Arial" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "Transforming the Multi-Modal Cardiac Analysis Platform", size: 24, color: "718096", italics: true })]
        }),
        new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Prepared for: N Rama Rao", size: 22, color: "4A5568" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Date: March 27, 2026", size: 22, color: "4A5568" })] }),
        new Paragraph({ children: [new PageBreak()] }),
      ]
    },

    // ===== MAIN CONTENT =====
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "DeepCardio-RAG | LangChain/LangGraph Integration", size: 16, color: "999999", italics: true })]
        })] })
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", size: 18, color: "999999" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999" })]
        })] })
      },
      children: [
        // ===== 1. EXECUTIVE SUMMARY =====
        heading("1. Executive Summary"),
        para("This document outlines how integrating LangChain and LangGraph would transform the DeepCardio-RAG multi-modal cardiac analysis platform. Currently, the project uses hand-built RAG pipelines with hardcoded retrieval, GPT-2 with manual soft-prompt injection, and linear pipeline execution. With LangChain/LangGraph, the system gains production-grade RAG with proper vector stores, swappable LLM backends (GPT-4, Claude, Llama), stateful multi-agent workflows with conditional branching, and built-in observability."),
        spacer(),

        // ===== 2. CURRENT vs PROPOSED =====
        heading("2. Current Architecture vs. LangChain/LangGraph Architecture"),
        para("The table below highlights the fundamental differences between the current implementation and what changes with LangChain/LangGraph integration:"),
        spacer(),
        simpleTable(
          ["Component", "Current (Hand-Built)", "With LangChain/LangGraph"],
          [
            ["Vector Store", "Milvus Lite (returns None, always falls back to hardcoded mock guidelines)", "LangChain VectorStore interface: FAISS, Chroma, Milvus, Pinecone, Qdrant \u2014 swappable with 1 line"],
            ["Embeddings", "Custom 384-dim CNN embeddings used directly as DB queries", "LangChain Embeddings: HuggingFace, OpenAI, Cohere \u2014 proper text+signal hybrid embedding"],
            ["LLM / Report Gen", "GPT-2 local with manual soft-prompt injection via torch tensor concatenation", "LangChain ChatModel: GPT-4o, Claude 3.5, Llama 3, Gemini \u2014 structured output, streaming, fallbacks"],
            ["RAG Pipeline", "Linear 3-step: Encode \u2192 Retrieve \u2192 Generate (no retry, no branching)", "LangGraph StateGraph: conditional routing, parallel retrieval, human-in-the-loop checkpoints"],
            ["Prompt Mgmt", "f-string templates hardcoded in Python", "LangChain PromptTemplate / ChatPromptTemplate with variable injection and versioning"],
            ["Memory", "None \u2014 each request is stateless", "LangGraph state persistence: conversation memory, patient history, cross-session context"],
            ["Error Handling", "try/except with traceback.print_exc()", "LangGraph retry nodes, fallback chains, dead-letter queues"],
            ["Observability", "print() statements", "LangSmith tracing: every LLM call, retrieval, and chain step logged with latency and cost"],
            ["Multi-Modal Fusion", "CardioFusion PyTorch model (keeps as-is)", "Wrapped as LangChain Tool \u2014 LangGraph agent decides when to invoke fusion vs single-modality"],
          ]
        ),
        spacer(),

        // ===== 3. WHAT IS LANGCHAIN =====
        heading("3. What LangChain Brings"),
        heading("3.1 Replacing the Broken RAG Pipeline", HeadingLevel.HEADING_2),
        para("The current pipeline in core/pipeline.py has a critical issue: get_collection() always returns None, so the Milvus retriever never works. Every request falls back to two hardcoded mock guidelines. LangChain fixes this fundamentally:"),
        spacer(),
        heading("Current Code (core/pipeline.py):", HeadingLevel.HEADING_3),
        ...codeBlock([
          "class ClinicalKnowledgeRetriever:",
          "    def __init__(self):",
          "        self.collection = get_collection()  # Always returns None!",
          "",
          "    def retrieve_context(self, query_embeddings, top_k=3):",
          "        if self.collection is None:  # Always True",
          '            return ["Hardcoded guideline 1", "Hardcoded guideline 2"]',
        ]),
        spacer(),
        heading("With LangChain:", HeadingLevel.HEADING_3),
        ...codeBlock([
          "from langchain_community.vectorstores import FAISS",
          "from langchain_huggingface import HuggingFaceEmbeddings",
          "from langchain.text_splitter import RecursiveCharacterTextSplitter",
          "",
          "# Load clinical guidelines from multiple sources",
          "splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)",
          "docs = splitter.split_documents(load_clinical_guidelines())",
          "",
          "# Create a REAL working vector store (no Milvus dependency issues)",
          'embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")',
          "vectorstore = FAISS.from_documents(docs, embeddings)",
          'retriever = vectorstore.as_retriever(search_kwargs={"k": 5})',
          "",
          "# One-line swap to any other vector store:",
          "# vectorstore = Chroma.from_documents(docs, embeddings)",
          "# vectorstore = Pinecone.from_documents(docs, embeddings, index_name=...)",
        ]),
        spacer(),

        heading("3.2 Upgrading Report Generation from GPT-2 to Any LLM", HeadingLevel.HEADING_2),
        para("The current system manually injects ECG embeddings into GPT-2 token space using a learned linear projection. This is clever but limits you to GPT-2 (124M parameters, no instruction following). LangChain lets you use any model with structured prompting:"),
        spacer(),
        heading("Current Code (core/pipeline.py):", HeadingLevel.HEADING_3),
        ...codeBlock([
          "class ReportGenerator(nn.Module):",
          '    def __init__(self):',
          '        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")',
          '        self.llm = AutoModelForCausalLM.from_pretrained("gpt2")',
          "        self.ecg_to_llm_space = nn.Linear(384, 768)  # Manual projection",
          "",
          "    def generate_report(self, ecg_embeddings, contexts):",
          "        ecg_soft_prompt = self.ecg_to_llm_space(ecg_embeddings)  # Tensor math",
          '        prompt = f"Context: {contexts[i]}\\nTask: Write clinical report..."',
          "        # Manual embedding concatenation, generate, decode...",
        ]),
        spacer(),
        heading("With LangChain:", HeadingLevel.HEADING_3),
        ...codeBlock([
          "from langchain_openai import ChatOpenAI",
          "from langchain_core.prompts import ChatPromptTemplate",
          "from langchain_core.output_parsers import PydanticOutputParser",
          "",
          "# Structured output with any LLM",
          "class ClinicalReport(BaseModel):",
          '    findings: str = Field(description="ECG/Echo findings")',
          '    impression: str = Field(description="Clinical impression")',
          '    recommendations: List[str] = Field(description="Action items")',
          '    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]',
          "",
          "parser = PydanticOutputParser(pydantic_object=ClinicalReport)",
          "",
          "report_prompt = ChatPromptTemplate.from_messages([",
          '    ("system", "You are an expert cardiologist. Generate a structured clinical report."),',
          '    ("human", """',
          "        Patient cardiac data analysis:",
          "        - Modalities analyzed: {modalities}",
          "        - Key findings: {model_findings}",
          "        - Retrieved guidelines: {retrieved_context}",
          "        ",
          "        {format_instructions}",
          '    """),',
          "])",
          "",
          "# Swap LLMs with one line change:",
          'llm = ChatOpenAI(model="gpt-4o", temperature=0.2)',
          "# llm = ChatAnthropic(model='claude-sonnet-4-20250514')",
          "# llm = ChatOllama(model='llama3')",
          "",
          "chain = report_prompt | llm | parser",
          "report: ClinicalReport = chain.invoke({...})",
        ]),
        spacer(),

        heading("3.3 Clinical Knowledge as a Proper Document Pipeline", HeadingLevel.HEADING_2),
        para("Instead of hardcoded guideline strings scattered across 5 different files (pipeline.py, echonet_pipeline.py, ecg_image_loader.py, heart_sound_loader.py, vfdb_loader.py), LangChain provides a unified document loading and indexing pipeline:"),
        spacer(),
        ...codeBlock([
          "from langchain_community.document_loaders import (",
          "    PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader",
          ")",
          "",
          "# Load guidelines from multiple formats",
          "loaders = [",
          '    PyPDFLoader("data/guidelines/ACC_AHA_2022.pdf"),',
          '    PyPDFLoader("data/guidelines/ESC_2021_HF.pdf"),',
          '    TextLoader("data/guidelines/ecg_interpretation.txt"),',
          '    CSVLoader("data/guidelines/drug_interactions.csv"),',
          "]",
          "",
          "# All guidelines in one searchable vector store",
          "all_docs = []",
          "for loader in loaders:",
          "    all_docs.extend(loader.load_and_split(splitter))",
          "",
          "# Add metadata for filtered retrieval",
          "for doc in all_docs:",
          '    doc.metadata["specialty"] = classify_specialty(doc)',
          '    doc.metadata["evidence_level"] = extract_evidence_level(doc)',
          "",
          "vectorstore = FAISS.from_documents(all_docs, embeddings)",
          "",
          "# Retrieve with metadata filtering",
          "retriever = vectorstore.as_retriever(",
          '    search_type="mmr",  # Maximum Marginal Relevance (diversity)',
          '    search_kwargs={"k": 5, "filter": {"specialty": "cardiology"}}',
          ")",
        ]),
        spacer(),

        // ===== 4. WHAT LANGGRAPH BRINGS =====
        new Paragraph({ children: [new PageBreak()] }),
        heading("4. What LangGraph Brings"),
        para("LangGraph is the game-changer for this project. While LangChain handles the RAG components, LangGraph turns your linear pipelines into intelligent, stateful workflows with conditional branching, parallel execution, and human-in-the-loop checkpoints."),
        spacer(),

        heading("4.1 The Core Concept: StateGraph", HeadingLevel.HEADING_2),
        para("Currently, each pipeline (ECG, Echo, Heart Sound, VFDB, CardioFusion) runs as a fixed linear sequence. LangGraph replaces this with a directed graph where each node is a processing step, and edges can be conditional:"),
        spacer(),
        ...codeBlock([
          "from langgraph.graph import StateGraph, START, END",
          "from typing import TypedDict, Annotated, Literal",
          "",
          "# Define the state that flows through the graph",
          "class CardiacAnalysisState(TypedDict):",
          "    # Inputs",
          "    patient_id: str",
          "    modalities: dict           # Raw input data per modality",
          "    ",
          "    # Processing outputs",
          "    embeddings: dict            # Per-modality embeddings",
          "    model_predictions: dict     # CNN/Fusion model outputs",
          "    retrieved_guidelines: list  # RAG-retrieved context",
          "    ",
          "    # Final outputs",
          "    clinical_report: str",
          "    risk_level: str",
          "    alert_triggered: bool",
          "    requires_review: bool       # Human-in-the-loop flag",
        ]),
        spacer(),

        heading("4.2 The Cardiac Analysis Graph (Replaces All Pipelines)", HeadingLevel.HEADING_2),
        para("Here is the complete LangGraph workflow that replaces the current linear pipelines with an intelligent, branching graph:"),
        spacer(),
        ...codeBlock([
          "def build_cardiac_graph():",
          "    graph = StateGraph(CardiacAnalysisState)",
          "",
          "    # ---- Node 1: Input Router ----",
          "    # Detects which modalities are present and routes accordingly",
          "    def route_input(state):",
          '        mods = [k for k, v in state["modalities"].items() if v is not None]',
          '        state["active_modalities"] = mods',
          "        return state",
          "",
          "    # ---- Node 2: Parallel Encoding ----",
          "    # Runs each modality encoder concurrently (not sequentially!)",
          "    def encode_modalities(state):",
          "        embeddings = {}",
          '        for mod in state["active_modalities"]:',
          '            encoder = get_encoder(mod)  # Your existing CNN encoders',
          '            embeddings[mod] = encoder(state["modalities"][mod])',
          '        state["embeddings"] = embeddings',
          "        return state",
          "",
          "    # ---- Node 3: Fusion Decision ----",
          "    # If multiple modalities, use CardioFusion; otherwise single-modal",
          "    def should_fuse(state) -> Literal['fuse', 'single']:",
          '        return "fuse" if len(state["active_modalities"]) > 1 else "single"',
          "",
          "    # ---- Node 4a: CardioFusion (multi-modal) ----",
          "    def run_fusion(state):",
          "        model = get_cardiofusion_model()",
          '        state["model_predictions"] = model.inference(state["embeddings"])',
          "        return state",
          "",
          "    # ---- Node 4b: Single-Modal Analysis ----",
          "    def run_single_analysis(state):",
          '        mod = state["active_modalities"][0]',
          "        classifier = get_classifier(mod)",
          '        state["model_predictions"] = classifier(state["embeddings"][mod])',
          "        return state",
          "",
          "    # ---- Node 5: RAG Retrieval ----",
          "    def retrieve_guidelines(state):",
          '        query = build_query_from_predictions(state["model_predictions"])',
          "        docs = retriever.invoke(query)",
          '        state["retrieved_guidelines"] = [d.page_content for d in docs]',
          "        return state",
          "",
          "    # ---- Node 6: Risk Assessment ----",
          "    def assess_risk(state) -> Literal['critical', 'normal', 'review']:",
          '        risk = state["model_predictions"].get("cardiac_risk_score", 0)',
          '        danger = state["model_predictions"].get("ventricular_danger", {})',
          '        if risk > 80 or danger.get("is_dangerous", False):',
          '            return "critical"',
          "        elif risk > 50:",
          '            return "review"  # Needs human cardiologist review',
          '        return "normal"',
          "",
          "    # ---- Node 7: Report Generation ----",
          "    def generate_report(state):",
          "        chain = report_prompt | llm | parser",
          '        state["clinical_report"] = chain.invoke({',
          '            "modalities": state["active_modalities"],',
          '            "model_findings": state["model_predictions"],',
          '            "retrieved_context": state["retrieved_guidelines"],',
          "        })",
          "        return state",
          "",
          "    # ---- Node 8: Alert Node (Critical Cases) ----",
          "    def trigger_alert(state):",
          '        state["alert_triggered"] = True',
          '        # Send notification, page on-call cardiologist, etc.',
          "        return state",
          "",
          "    # ---- Node 9: Human Review Checkpoint ----",
          "    def flag_for_review(state):",
          '        state["requires_review"] = True',
          '        # LangGraph can PAUSE here and wait for human input',
          "        return state",
          "",
          "    # ===== WIRE THE GRAPH =====",
          '    graph.add_node("route", route_input)',
          '    graph.add_node("encode", encode_modalities)',
          '    graph.add_node("fuse", run_fusion)',
          '    graph.add_node("single", run_single_analysis)',
          '    graph.add_node("retrieve", retrieve_guidelines)',
          '    graph.add_node("report", generate_report)',
          '    graph.add_node("alert", trigger_alert)',
          '    graph.add_node("review", flag_for_review)',
          "",
          "    # Edges",
          '    graph.add_edge(START, "route")',
          '    graph.add_edge("route", "encode")',
          '    graph.add_conditional_edges("encode", should_fuse,',
          '        {"fuse": "fuse", "single": "single"})',
          '    graph.add_edge("fuse", "retrieve")',
          '    graph.add_edge("single", "retrieve")',
          '    graph.add_edge("retrieve", "report")',
          '    graph.add_conditional_edges("report", assess_risk,',
          '        {"critical": "alert", "review": "review", "normal": END})',
          '    graph.add_edge("alert", END)',
          '    graph.add_edge("review", END)',
          "",
          "    return graph.compile()",
        ]),
        spacer(),

        heading("4.3 Visual Flow Diagram", HeadingLevel.HEADING_2),
        para("The graph above creates this execution flow:"),
        spacer(),
        ...codeBlock([
          "                    [START]",
          "                      |",
          "                 [Route Input]",
          "                      |",
          "               [Encode Modalities]  (parallel CNN encoding)",
          "                   /       \\",
          "            multi-modal   single-modal",
          "                /             \\",
          "        [CardioFusion]    [Single Analysis]",
          "                \\             /",
          "             [RAG Retrieval]  (LangChain vector search)",
          "                      |",
          "           [Generate Report]  (GPT-4o / Claude / Llama)",
          "              /    |     \\",
          "         CRITICAL  REVIEW  NORMAL",
          "            /       |         \\",
          "     [Alert]  [Human Review]  [END]",
          "        |          |",
          "      [END]      [END]",
        ]),
        spacer(),

        heading("4.4 State Persistence and Checkpointing", HeadingLevel.HEADING_2),
        para("LangGraph supports persistent state, meaning the graph can pause at any node (e.g., waiting for a cardiologist to review a borderline case) and resume later. This is impossible with the current linear pipeline:"),
        spacer(),
        ...codeBlock([
          "from langgraph.checkpoint.sqlite import SqliteSaver",
          "",
          "# Persist graph state to SQLite (survives server restarts)",
          'checkpointer = SqliteSaver.from_conn_string("data/cardiac_graph.db")',
          "app = graph.compile(checkpointer=checkpointer)",
          "",
          "# Start analysis (may pause at human review node)",
          'config = {"configurable": {"thread_id": "patient_12345"}}',
          "result = app.invoke(initial_state, config)",
          "",
          "# Later: cardiologist provides review",
          'if result["requires_review"]:',
          "    app.update_state(config, {",
          '        "cardiologist_notes": "Confirmed borderline EF. Recommend follow-up.",',
          '        "approved": True',
          "    })",
          "    final = app.invoke(None, config)  # Resume from checkpoint",
        ]),
        spacer(),

        // ===== 5. MULTI-AGENT ARCHITECTURE =====
        new Paragraph({ children: [new PageBreak()] }),
        heading("5. Multi-Agent Architecture with LangGraph"),
        para("The most powerful upgrade: instead of a single monolithic pipeline, LangGraph enables a multi-agent system where specialized agents collaborate:"),
        spacer(),
        simpleTable(
          ["Agent", "Role", "Tools Available"],
          [
            ["Triage Agent", "Receives patient data, decides which analyses to run, routes to specialists", "Input validation, modality detection, urgency classification"],
            ["ECG Agent", "Analyzes ECG signals and images using 1D-CNN and 2D-CNN encoders", "ECGEncoder1DCNN, ECGImageClassifier, ECG RAG retriever"],
            ["Echo Agent", "Processes echocardiogram videos, predicts EF, assesses LV function", "EchoVideoEncoder, EFRegressor, Echo RAG retriever"],
            ["Auscultation Agent", "Analyzes heart sounds, detects murmurs from phonocardiograms", "HeartSoundClassifier, mel spectrogram processor, Sound RAG"],
            ["Arrhythmia Agent", "Monitors for dangerous ventricular rhythms (VT, VF, asystole)", "VentricularArrhythmiaDetector, VFDB rhythm classifier"],
            ["Fusion Agent", "Combines multi-modal results when multiple data types are available", "CardioFusion model, cross-modal Transformer"],
            ["Report Agent", "Synthesizes all findings into a structured clinical report", "LLM (GPT-4o/Claude), RAG retriever, PDF generator"],
          ]
        ),
        spacer(),
        ...codeBlock([
          "from langgraph.graph import StateGraph, START, END",
          "",
          "# Each agent is a sub-graph",
          "def build_multi_agent_system():",
          "    graph = StateGraph(CardiacAnalysisState)",
          "",
          '    graph.add_node("triage", triage_agent)',
          '    graph.add_node("ecg_agent", ecg_analysis_subgraph)',
          '    graph.add_node("echo_agent", echo_analysis_subgraph)',
          '    graph.add_node("sound_agent", auscultation_subgraph)',
          '    graph.add_node("arrhythmia_agent", arrhythmia_subgraph)',
          '    graph.add_node("fusion_agent", fusion_subgraph)',
          '    graph.add_node("report_agent", report_generation_subgraph)',
          "",
          "    # Triage routes to relevant specialists (parallel!)",
          '    graph.add_edge(START, "triage")',
          '    graph.add_conditional_edges("triage", route_to_specialists, {',
          '        "ecg_only": "ecg_agent",',
          '        "echo_only": "echo_agent",',
          '        "multi_modal": "parallel_agents",  # Fan-out',
          "    })",
          "",
          "    # All specialist agents converge at fusion/report",
          '    graph.add_edge("ecg_agent", "fusion_agent")',
          '    graph.add_edge("echo_agent", "fusion_agent")',
          '    graph.add_edge("sound_agent", "fusion_agent")',
          '    graph.add_edge("arrhythmia_agent", "fusion_agent")',
          '    graph.add_edge("fusion_agent", "report_agent")',
          '    graph.add_edge("report_agent", END)',
          "",
          "    return graph.compile()",
        ]),
        spacer(),

        // ===== 6. WHAT STAYS THE SAME =====
        heading("6. What Stays The Same (Your PyTorch Models)"),
        para("LangChain/LangGraph does NOT replace your deep learning models. All your custom neural networks remain exactly as they are. They get wrapped as LangChain Tools that the graph nodes call:"),
        spacer(),
        simpleTable(
          ["Component", "Status", "How It Integrates"],
          [
            ["ECGEncoder1DCNN", "Kept as-is", "Wrapped as LangChain Tool, called by ECG Agent node"],
            ["EchoVideoEncoder (R2+1D)", "Kept as-is", "Wrapped as LangChain Tool, called by Echo Agent node"],
            ["ECGImageClassifier (2D-CNN)", "Kept as-is", "Wrapped as LangChain Tool, called by ECG Agent node"],
            ["HeartSoundClassifier", "Kept as-is", "Wrapped as LangChain Tool, called by Sound Agent node"],
            ["VentricularArrhythmiaDetector", "Kept as-is", "Wrapped as LangChain Tool, called by Arrhythmia Agent node"],
            ["CardioFusion (Transformer+MMoE)", "Kept as-is", "Wrapped as LangChain Tool, called by Fusion Agent node"],
            ["Tabular BERT+MoE (Arthritis)", "Kept as-is", "Separate LangGraph sub-graph for arthritis workflow"],
          ]
        ),
        spacer(),
        para("The wrapping is simple:"),
        spacer(),
        ...codeBlock([
          "from langchain_core.tools import tool",
          "",
          "@tool",
          "def analyze_ecg_signal(signal_path: str) -> dict:",
          '    """Analyze a 12-lead ECG signal and return arrhythmia classification."""',
          "    signal = load_ecg_signal(signal_path)",
          "    encoder = get_model().encoder",
          "    embedding = encoder(signal)",
          "    # Your existing CNN does the heavy lifting",
          '    return {"embedding": embedding, "classification": classify(embedding)}',
          "",
          "@tool",
          "def analyze_echo_video(video_path: str) -> dict:",
          '    """Analyze echocardiogram video and predict ejection fraction."""',
          "    pipeline = get_echonet_pipeline()",
          "    return pipeline.analyze_single(load_video(video_path))",
          "",
          "@tool",
          "def run_cardiofusion(modality_data: dict) -> dict:",
          '    """Run multi-modal CardioFusion analysis across available modalities."""',
          "    model = get_cardiofusion_model()",
          "    return model.inference(modality_data)",
        ]),
        spacer(),

        // ===== 7. NEW CAPABILITIES =====
        new Paragraph({ children: [new PageBreak()] }),
        heading("7. New Capabilities Unlocked"),

        heading("7.1 Conversational Clinical Interface", HeadingLevel.HEADING_2),
        para("With LangGraph memory, the system can have multi-turn conversations about a patient case. A cardiologist could upload an ECG, ask follow-up questions, request comparison with previous studies, and get progressively refined reports, all within a single session that remembers context:"),
        spacer(),
        ...codeBlock([
          "from langgraph.graph import MessagesState",
          "",
          "class ClinicalChatState(MessagesState):",
          "    patient_id: str",
          "    analysis_history: list  # Previous analyses in this session",
          "    current_findings: dict",
          "",
          "# Doctor: 'Analyze this ECG'",
          "# System: [Runs ECG pipeline, returns findings]",
          "# Doctor: 'Compare with the echo from last week'",
          "# System: [Retrieves previous echo, runs fusion, shows comparison]",
          "# Doctor: 'What does the latest ACC guideline say about this pattern?'",
          "# System: [RAG retrieves specific guideline, presents with context]",
        ]),
        spacer(),

        heading("7.2 Streaming Results", HeadingLevel.HEADING_2),
        para("LangGraph supports streaming, so the frontend can show real-time progress as each node completes, instead of waiting for the entire pipeline:"),
        spacer(),
        ...codeBlock([
          "# FastAPI streaming endpoint",
          "@app.post('/api/analyze/stream')",
          "async def stream_analysis(request):",
          "    async for event in app.astream(state, config):",
          "        # Frontend sees each step as it completes:",
          "        # 'Encoding ECG signal...'",
          "        # 'Retrieved 5 clinical guidelines'",
          "        # 'Generating report with GPT-4o...'",
          "        yield ServerSentEvent(data=json.dumps(event))",
        ]),
        spacer(),

        heading("7.3 Automatic Fallback Chains", HeadingLevel.HEADING_2),
        para("If the primary LLM fails (rate limit, timeout), LangChain automatically falls back:"),
        spacer(),
        ...codeBlock([
          "from langchain_core.runnables import RunnableWithFallbacks",
          "",
          "llm_with_fallback = ChatOpenAI(model='gpt-4o').with_fallbacks([",
          "    ChatAnthropic(model='claude-sonnet-4-20250514'),",
          "    ChatOllama(model='llama3'),    # Local fallback (always available)",
          "])",
        ]),
        spacer(),

        heading("7.4 LangSmith Observability", HeadingLevel.HEADING_2),
        para("Every LLM call, retrieval query, and agent decision is automatically traced. You can see exactly why the system made a particular diagnosis, how long each step took, how much it cost, and which guidelines were retrieved. This is essential for medical AI auditing."),
        spacer(),

        // ===== 8. FILE CHANGES =====
        heading("8. Files That Would Change"),
        spacer(),
        simpleTable(
          ["File", "Change"],
          [
            ["core/pipeline.py", "Replace ClinicalKnowledgeRetriever + ReportGenerator with LangChain chain"],
            ["core/echonet_pipeline.py", "Replace EchoKnowledgeRetriever + EchoReportGenerator with LangChain chain"],
            ["core/langchain_rag.py (NEW)", "Unified RAG setup: vector store, embeddings, retriever, prompt templates"],
            ["core/langgraph_workflow.py (NEW)", "StateGraph definition: all nodes, edges, conditional routing"],
            ["core/agents.py (NEW)", "Multi-agent definitions: Triage, ECG, Echo, Sound, Arrhythmia, Report"],
            ["core/tools.py (NEW)", "LangChain Tool wrappers around existing PyTorch models"],
            ["database/db_manager.py", "Replace Milvus with LangChain VectorStore (FAISS/Chroma)"],
            ["database/seed_data.py", "Use LangChain DocumentLoaders to ingest clinical guidelines"],
            ["main.py", "Simplify: endpoints call graph.invoke() instead of manual pipeline orchestration"],
            ["requirements.txt", "Add: langchain, langgraph, langchain-openai, langchain-community, faiss-cpu"],
            ["core/hybrid_model.py", "NO CHANGE (wrapped as Tool)"],
            ["core/ecg_image_loader.py", "NO CHANGE (wrapped as Tool)"],
            ["core/heart_sound_loader.py", "NO CHANGE (wrapped as Tool)"],
            ["core/vfdb_loader.py", "NO CHANGE (wrapped as Tool)"],
            ["core/video_encoder.py", "NO CHANGE (wrapped as Tool)"],
          ]
        ),
        spacer(),

        // ===== 9. REQUIREMENTS =====
        heading("9. Updated Requirements"),
        spacer(),
        ...codeBlock([
          "# Core LangChain",
          "langchain>=0.3.0",
          "langchain-core>=0.3.0",
          "langchain-community>=0.3.0",
          "",
          "# LangGraph (stateful workflows)",
          "langgraph>=0.2.0",
          "langgraph-checkpoint-sqlite>=1.0.0",
          "",
          "# LLM providers (pick one or more)",
          "langchain-openai>=0.2.0        # GPT-4o",
          "langchain-anthropic>=0.3.0     # Claude 3.5",
          "langchain-ollama>=0.2.0        # Local Llama 3",
          "",
          "# Vector stores (pick one)",
          "faiss-cpu>=1.8.0               # Fast local vector search",
          "# chromadb>=0.5.0              # Alternative",
          "",
          "# Embeddings",
          "langchain-huggingface>=0.1.0",
          "sentence-transformers>=3.0.0",
          "",
          "# Observability",
          "langsmith>=0.1.0",
          "",
          "# Existing deps (unchanged)",
          "torch>=2.0.0",
          "transformers>=4.30.0",
          "fastapi>=0.100.0",
          "uvicorn>=0.23.0",
          "fpdf2",
          "opencv-python-headless",
          "numpy",
          "Pillow",
        ]),
        spacer(),

        // ===== 10. SUMMARY =====
        new Paragraph({ children: [new PageBreak()] }),
        heading("10. Summary: Before vs After"),
        spacer(),
        simpleTable(
          ["Capability", "Before", "After"],
          [
            ["RAG", "Broken (always returns hardcoded mock)", "Production FAISS/Chroma with real guideline retrieval"],
            ["Report Quality", "GPT-2 (124M params, no instruction following)", "GPT-4o / Claude / Llama with structured output"],
            ["Pipeline Logic", "Linear, fixed, no branching", "Conditional graph with parallel execution"],
            ["Error Recovery", "Crash on failure", "Automatic retry + LLM fallback chains"],
            ["State", "Stateless (each request independent)", "Persistent state with conversation memory"],
            ["Human Review", "Not possible", "Graph pauses, waits for cardiologist, resumes"],
            ["Observability", "print() statements", "Full LangSmith tracing with cost/latency"],
            ["Multi-Agent", "Monolithic pipeline", "7 specialized agents collaborating"],
            ["Streaming", "Wait for full result", "Real-time node-by-node progress"],
            ["LLM Flexibility", "Locked to GPT-2", "Any LLM, swappable with 1 line"],
            ["PyTorch Models", "Direct calls", "Same models, wrapped as LangChain Tools"],
          ]
        ),
        spacer(),
        para("The bottom line: LangChain replaces the broken RAG and limited LLM layer, while LangGraph transforms the rigid linear pipelines into intelligent, stateful, multi-agent workflows. Your custom PyTorch models (the real innovation) remain exactly as they are."),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/brave-elegant-volta/mnt/RAG_claude/LangChain_LangGraph_Integration_Architecture.docx", buffer);
  console.log("Document created successfully!");
});
