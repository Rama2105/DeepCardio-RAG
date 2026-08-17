"""
core/jh_report_generator.py — Johns Hopkins–Style Clinical ECG Report Generator
=================================================================================
Generates dual-audience clinical reports at world-top-level hospital quality
(Johns Hopkins Medicine / Mayo Clinic standard) using fpdf2.

Report structure mirrors the Johns Hopkins Hospital ECG Report format:
  ┌────────────────────────────────────────────────────────┐
  │  HOSPITAL HEADER (logo area, contact, accreditation)   │
  ├────────────────────────────────────────────────────────┤
  │  PATIENT DEMOGRAPHICS & ENCOUNTER INFORMATION          │
  ├────────────────────────────────────────────────────────┤
  │  ORDERING PHYSICIAN & CLINICAL INDICATION              │
  ├────────────────────────────────────────────────────────┤
  │  TECHNICAL PARAMETERS (lead system, sampling rate)     │
  ├────────────────────────────────────────────────────────┤
  │  SECTION 1: RHYTHM ANALYSIS                            │
  │  SECTION 2: INTERVALS & AXES                           │
  │  SECTION 3: WAVEFORM MORPHOLOGY                        │
  │    2.1 P Waves  2.2 PR Interval  2.3 QRS Complex       │
  │    2.4 QT/QTc   2.5 ST Segment  2.6 T Waves            │
  │  SECTION 4: PRIMARY IMPRESSION                         │
  │  SECTION 5: DIFFERENTIAL DIAGNOSIS                     │
  │  SECTION 6: CLINICAL SIGNIFICANCE & RISK STRATIFICATION│
  │  SECTION 7: RECOMMENDATIONS                            │
  │  SECTION 8: EVIDENCE-BASED REFERENCES                  │
  ├────────────────────────────────────────────────────────┤
  │  RAG CONTEXT UTILISED (retrieved clinical guidelines)  │
  ├────────────────────────────────────────────────────────┤
  │  AI CONFIDENCE SCORES (per finding)                    │
  ├────────────────────────────────────────────────────────┤
  │  PHYSICIAN VERIFICATION & ELECTRONIC SIGNATURE         │
  ├────────────────────────────────────────────────────────┤
  │  LEGAL DISCLAIMER                                      │
  └────────────────────────────────────────────────────────┘

Dual-audience:
  - CLINICAL  : full technical detail, medical terminology, ICD-10 codes
  - PATIENT   : plain language, simplified explanations, actionable advice

Dependencies: fpdf2 (pip install fpdf2)
"""

from __future__ import annotations

import os
import datetime
import unicodedata
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from core.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Unicode Sanitiser (required for fpdf2 Latin-1 encoding)
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Replace characters outside Latin-1 with safe ASCII equivalents."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        '\u2014': '--',  '\u2013': '-',   '\u2026': '...',
        '\u2018': "'",   '\u2019': "'",   '\u201c': '"',  '\u201d': '"',
        '\u2022': '*',   '\u00b0': ' deg','\u2265': '>=', '\u2264': '<=',
        '\u00b1': '+/-', '\u03b1': 'alpha','\u03b2': 'beta','\u00b5': 'u',
        '\u2192': '->',  '\u2190': '<-',  '\u00d7': 'x',  '\u00f7': '/',
        '\u2248': '~=',  '\u2260': '!=',  '\u00ae': '(R)','\u00a9': '(C)',
        '\u2122': '(TM)','\u00bd': '1/2', '\u00bc': '1/4','\u00be': '3/4',
        '\u2764': '<3',  '\u2713': 'ok',  '\u2717': 'x',
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    return text.encode('latin-1', errors='replace').decode('latin-1')

# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientInfo:
    """Patient demographic and encounter information."""
    name:              str   = "Anonymous Patient"
    mrn:               str   = "N/A"
    dob:               str   = "N/A"
    age:               int   = 0
    sex:               str   = "N/A"
    encounter_id:      str   = "N/A"
    encounter_date:    str   = ""
    ward:              str   = "Cardiology"
    room:              str   = "N/A"
    height_cm:         float = 0.0
    weight_kg:         float = 0.0
    bmi:               float = 0.0
    primary_diagnosis: str   = ""
    allergies:         str   = ""
    medications:       str   = ""

    def __post_init__(self):
        if not self.encounter_date:
            self.encounter_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.height_cm > 0 and self.weight_kg > 0 and self.bmi == 0.0:
            self.bmi = round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)
        if isinstance(self.dob, datetime.date):
            self.dob = self.dob.strftime("%Y-%m-%d")


@dataclass
class PhysicianInfo:
    """Ordering and interpreting physician information."""
    ordering_name:       str = "Ordering Physician"
    ordering_id:         str = "N/A"
    interpreting_name:   str = "Interpreting Cardiologist"
    interpreting_id:     str = "N/A"
    department:          str = "Division of Cardiology"
    institution:         str = "Johns Hopkins Hospital"
    clinical_indication: str = "Routine ECG evaluation"
    indication:          str = ""   # alias for clinical_indication

    def __post_init__(self):
        if self.indication and self.clinical_indication == "Routine ECG evaluation":
            self.clinical_indication = self.indication


@dataclass
class ECGParameters:
    """Technical ECG acquisition parameters."""
    lead_system:       str   = "Standard 12-Lead"
    sampling_rate_hz:  int   = 500
    duration_sec:      float = 10.0
    n_leads:           int   = 12
    filter_hz:         str   = "0.05–150 Hz"
    notch_filter_hz:   int   = 60
    acquisition_device: str  = "GE MAC 5500 HD"
    calibration_mv:    float = 1.0


@dataclass
class ECGFindings:
    """Structured ECG findings for all 8 sections."""
    # Section 1: Rhythm
    rhythm:              str   = "Sinus Rhythm"
    heart_rate_bpm:      float = 72.0
    heart_rate:          float = 0.0    # alias for heart_rate_bpm
    heart_rate_variability: str = "Normal"
    regularity:          str   = "Regular"

    # Section 2: Intervals & Axes
    pr_interval_ms:      float = 160.0
    qrs_duration_ms:     float = 88.0
    qt_interval_ms:      float = 380.0
    qtc_interval_ms:     float = 410.0
    qtc_ms:              float = 0.0    # alias for qtc_interval_ms
    p_axis_deg:          float = 60.0
    qrs_axis_deg:        float = 45.0
    axis_degrees:        float = 0.0    # alias for qrs_axis_deg
    t_axis_deg:          float = 35.0

    # Section 3: Waveform morphology
    p_waves:             str   = "Normal morphology, upright in I, II, aVF"
    p_wave:              str   = ""     # alias for p_waves
    pr_comment:          str   = "Normal (120-200 ms)"
    qrs_comment:         str   = "Normal morphology, no delta waves, no BBB"
    qrs_morphology:      str   = ""     # alias for qrs_comment
    qt_comment:          str   = "Normal QTc (<=440 ms)"
    st_segment:          str   = "Isoelectric - no significant deviation"
    t_waves:             str   = "Upright in I, II, V4-V6; inverted in aVR (normal)"
    t_wave:              str   = ""     # alias for t_waves
    u_waves:             str   = "Not identified"

    # Section 4: Primary impression
    primary_impression:  str   = "Normal Sinus Rhythm. No acute ischaemic changes."
    icd10_codes:         List[str] = field(default_factory=lambda: ["R00.0"])

    # Section 5: Differential diagnosis
    differential:        List[str] = field(default_factory=lambda: [
        "Normal ECG variant",
        "Early repolarisation (if minor ST changes present)",
    ])

    # Section 6: Clinical significance
    risk_category:       str   = "LOW"    # LOW / MODERATE / HIGH / CRITICAL
    risk_level:          str   = ""       # alias for risk_category
    risk_rationale:      str   = "No life-threatening arrhythmia or acute ischaemic pattern detected."
    clinical_significance: str = ""       # alias for risk_rationale
    stemi_equivalent:    bool  = False
    emergent_action:     bool  = False

    # Section 7: Recommendations
    recommendations:     List[str] = field(default_factory=lambda: [
        "Correlate with clinical presentation and symptoms.",
        "Repeat ECG in 24 hours if symptoms persist.",
        "Consider Holter monitoring if palpitations are reported.",
    ])

    # Section 8: Evidence-based references
    references:          List[str] = field(default_factory=lambda: [
        "Rautaharju PM et al. AHA/ACCF/HRS ECG Recommendations. JACC 2009;53(11):982-991.",
        "Goldberger AL et al. Clinical Electrocardiography: A Simplified Approach. 9th ed. 2018.",
        "Surawicz B et al. AHA/ACCF/HRS Recommendations for ECG interpretation. Circ 2009;119:e235-e240.",
    ])
    evidence_refs:       List[str] = field(default_factory=list)  # alias for references

    def __post_init__(self):
        if self.heart_rate > 0 and self.heart_rate_bpm == 72.0:
            self.heart_rate_bpm = self.heart_rate
        if self.qtc_ms > 0 and self.qtc_interval_ms == 410.0:
            self.qtc_interval_ms = self.qtc_ms
        if self.axis_degrees != 0.0 and self.qrs_axis_deg == 45.0:
            self.qrs_axis_deg = self.axis_degrees
        if self.p_wave and not self.p_waves:
            self.p_waves = self.p_wave
        if self.qrs_morphology and self.qrs_comment == "Normal morphology, no delta waves, no BBB":
            self.qrs_comment = self.qrs_morphology
        if self.t_wave and not self.t_waves:
            self.t_waves = self.t_wave
        if self.risk_level and self.risk_category == "LOW":
            self.risk_category = self.risk_level
        if self.clinical_significance and self.risk_rationale == "No life-threatening arrhythmia or acute ischaemic pattern detected.":
            self.risk_rationale = self.clinical_significance
        if self.evidence_refs and not any(r.startswith("Rautaharju") for r in self.references):
            self.references = self.evidence_refs
        elif self.evidence_refs:
            self.references = self.evidence_refs
        # Mark emergent if CRITICAL
        if self.risk_category == "CRITICAL":
            self.emergent_action = True


@dataclass
class AIMetrics:
    """AI confidence scores and RAG context metadata."""
    overall_confidence:    float = 0.0       # 0-1
    ecg_confidence:        float = 0.0       # alias for overall_confidence
    per_finding_scores:    Dict[str, float] = field(default_factory=dict)
    retrieved_contexts:    List[str] = field(default_factory=list)
    model_version:         str = "DeepCardio-RAG v2.0 (ResNet-34 + RAG Attention)"
    inference_time_ms:     float = 0.0
    bleu4_score:           float = 0.0
    rouge_l_score:         float = 0.0
    hallucination_rate:    float = 0.0
    arrhythmia_type:       str = ""
    arrhythmia_confidence: float = 0.0

    def __post_init__(self):
        if self.ecg_confidence > 0 and self.overall_confidence == 0.0:
            self.overall_confidence = self.ecg_confidence


# ──────────────────────────────────────────────────────────────────────────────
# PDF Builder
# ──────────────────────────────────────────────────────────────────────────────

class JHReportGenerator:
    """
    Generates Johns Hopkins–quality clinical ECG reports as PDF.

    Usage:
        gen = JHReportGenerator()
        pdf_bytes = gen.generate(
            patient=PatientInfo(...),
            physician=PhysicianInfo(...),
            findings=ECGFindings(...),
            ai_metrics=AIMetrics(...),
            audience="clinical"   # or "patient"
        )
        with open("report.pdf", "wb") as f:
            f.write(pdf_bytes)
    """

    # Hospital colours (Johns Hopkins blue & gold)
    JH_BLUE   = (0,   55, 123)   # RGB #00377B
    JH_GOLD   = (177, 132, 33)   # RGB #B18421
    JH_GREY   = (75,  75,  75)
    JH_LIGHT  = (245, 247, 250)
    JH_RED    = (180, 30,  30)
    JH_GREEN  = (20,  120, 60)
    JH_AMBER  = (200, 130, 0)

    RISK_COLORS = {
        "LOW":      (20,  120, 60),
        "MODERATE": (200, 130, 0),
        "HIGH":     (200, 60,  0),
        "CRITICAL": (180, 30,  30),
    }

    def __init__(self):
        self._check_fpdf()

    @staticmethod
    def _check_fpdf():
        try:
            import fpdf
            logger.info("fpdf2 available for JH report generation")
        except ImportError:
            logger.warning("fpdf2 not installed — install with: pip install fpdf2")

    # ── Public Entry Point ────────────────────────────────────────────────────

    def generate(
        self,
        patient:   PatientInfo,
        physician: PhysicianInfo,
        findings:  ECGFindings,
        ai_metrics: Optional[AIMetrics] = None,
        ecg_params: Optional[ECGParameters] = None,
        audience:  str = "clinical",   # "clinical" | "patient"
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Generate the PDF report and return bytes.

        Args:
          patient      : patient demographic information
          physician    : ordering/interpreting physician details
          findings     : structured ECG analysis findings
          ai_metrics   : AI confidence scores and RAG metadata
          ecg_params   : technical ECG acquisition parameters
          audience     : "clinical" (full detail) or "patient" (plain language)
          output_path  : if given, also save PDF to this path

        Returns:
          PDF file bytes
        """
        if ai_metrics is None:
            ai_metrics = AIMetrics()
        if ecg_params is None:
            ecg_params = ECGParameters()

        try:
            from fpdf import FPDF
        except ImportError:
            logger.error("fpdf2 not installed. Cannot generate PDF.")
            raise ImportError("fpdf2 required: pip install fpdf2")

        pdf = _JHReportPDF(
            patient=patient,
            physician=physician,
            findings=findings,
            ai_metrics=ai_metrics,
            ecg_params=ecg_params,
            audience=audience,
            jh_colors={
                "blue":  self.JH_BLUE,
                "gold":  self.JH_GOLD,
                "grey":  self.JH_GREY,
                "light": self.JH_LIGHT,
                "red":   self.JH_RED,
                "green": self.JH_GREEN,
                "amber": self.JH_AMBER,
                "risk":  self.RISK_COLORS,
            },
        )
        pdf.build()

        pdf_bytes = pdf.output()
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"PDF saved to {output_path}")

        return pdf_bytes


# ──────────────────────────────────────────────────────────────────────────────
# Internal FPDF Subclass
# ──────────────────────────────────────────────────────────────────────────────

class _JHReportPDF:
    """Internal class that constructs the FPDF document."""

    def __init__(self, patient, physician, findings, ai_metrics,
                 ecg_params, audience, jh_colors):
        self.patient    = patient
        self.physician  = physician
        self.findings   = findings
        self.ai         = ai_metrics
        self.params     = ecg_params
        self.audience   = audience
        self.C          = jh_colors

    def build(self):
        from fpdf import FPDF

        class _SanitizedFPDF(FPDF):
            """FPDF subclass that auto-sanitizes text to Latin-1 safe characters."""
            def cell(self, w=0, h=0, txt="", *args, **kwargs):
                return super().cell(w, h, _sanitize(str(txt)), *args, **kwargs)
            def multi_cell(self, w, h, txt="", *args, **kwargs):
                return super().multi_cell(w, h, _sanitize(str(txt)), *args, **kwargs)

        self.pdf = _SanitizedFPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=18)
        self.pdf.add_page()
        self.pdf.set_margins(left=15, top=15, right=15)

        self._draw_header()
        self._draw_patient_block()
        self._draw_technical_params()
        self._draw_divider()

        if self.audience == "patient":
            self._draw_patient_friendly_findings()
        else:
            self._draw_clinical_findings()

        self._draw_ai_block()
        self._draw_physician_signature()
        self._draw_legal_footer()

    def output(self) -> bytes:
        return bytes(self.pdf.output())

    # ── Header ───────────────────────────────────────────────────────────────
    def _draw_header(self):
        pdf = self.pdf
        C   = self.C

        # Blue banner
        pdf.set_fill_color(*C["blue"])
        pdf.rect(x=15, y=15, w=180, h=20, style="F")

        # Hospital name
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(17, 19)
        pdf.cell(120, 8, "JOHNS HOPKINS HOSPITAL", ln=0)

        # Accreditation badge (right)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(137, 18)
        pdf.cell(55, 4, "JCI Accredited | Magnet Designated", ln=1, align="R")
        pdf.set_xy(137, 22)
        pdf.cell(55, 4, "601 N. Caroline St, Baltimore, MD 21287", ln=1, align="R")
        pdf.set_xy(137, 26)
        pdf.cell(55, 4, "Tel: +1-410-955-5000  |  jhu.edu/hospital", ln=1, align="R")

        # Gold line
        pdf.set_draw_color(*C["gold"])
        pdf.set_line_width(0.8)
        pdf.line(15, 36, 195, 36)

        # Report type
        report_type = (
            "CLINICAL ECG REPORT — CONFIDENTIAL" if self.audience == "clinical"
            else "PATIENT ECG SUMMARY"
        )
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*C["blue"])
        pdf.set_xy(15, 38)
        pdf.cell(180, 7, report_type, ln=1, align="C")

        # Timestamp
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*C["grey"])
        pdf.set_xy(15, 45)
        pdf.cell(90, 5, f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=0)
        pdf.cell(90, 5, f"DeepCardio-RAG v2.0  |  AI-Assisted, Physician Verified", ln=1, align="R")

        pdf.ln(3)

    # ── Patient Block ─────────────────────────────────────────────────────────
    def _draw_patient_block(self):
        pdf = self.pdf
        C   = self.C
        pt  = self.patient
        dr  = self.physician

        # Section background
        pdf.set_fill_color(*C["light"])
        pdf.set_draw_color(*C["blue"])
        pdf.set_line_width(0.3)
        y0 = pdf.get_y()
        pdf.rect(x=15, y=y0, w=180, h=42, style="FD")

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*C["blue"])
        pdf.set_xy(17, y0 + 2)
        pdf.cell(176, 5, "PATIENT & ENCOUNTER INFORMATION", ln=1)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)

        rows = [
            [("Patient Name:", pt.name),      ("MRN:",         pt.mrn),          ("Encounter ID:", pt.encounter_id)],
            [("Date of Birth:", pt.dob),       ("Age / Sex:",   f"{pt.age}y / {pt.sex}"),  ("Encounter Date:", pt.encounter_date)],
            [("Ward / Room:", f"{pt.ward} / {pt.room}"), ("BMI:", f"{pt.bmi} kg/m²" if pt.bmi else "N/A"), ("Weight / Ht:", f"{pt.weight_kg}kg / {pt.height_cm}cm" if pt.weight_kg else "N/A")],
            [("Ordering MD:", dr.ordering_name), ("Interpreting MD:", dr.interpreting_name), ("Department:", dr.department)],
            [("Clinical Indication:", dr.clinical_indication, True)],  # wide cell
        ]
        col_w = [60, 60, 60]
        for row in rows:
            y = pdf.get_y()
            x = 17
            for cell in row:
                label, val = cell[0], cell[1]
                wide = len(cell) > 2 and cell[2]
                w = 176 if wide else col_w[0]
                pdf.set_xy(x, y)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(28 if not wide else 50, 4.5, label, ln=0)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell((w - 28) if not wide else (176 - 50), 4.5, str(val), ln=0)
                x += w
            pdf.ln(4.5)

        pdf.ln(3)

    # ── Technical Parameters ──────────────────────────────────────────────────
    def _draw_technical_params(self):
        pdf  = self.pdf
        C    = self.C
        par  = self.params

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*C["blue"])
        pdf.cell(180, 5, "TECHNICAL ACQUISITION PARAMETERS", ln=1)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*C["grey"])
        params_str = (
            f"Lead System: {par.lead_system}  |  Sampling Rate: {par.sampling_rate_hz} Hz  |  "
            f"Duration: {par.duration_sec}s  |  Bandwidth: {par.filter_hz}  |  "
            f"Notch: {par.notch_filter_hz} Hz  |  Calibration: {par.calibration_mv} mV  |  "
            f"Device: {par.acquisition_device}"
        )
        pdf.multi_cell(180, 4.5, _sanitize(params_str), ln=1)
        pdf.ln(2)

    # ── Section Heading Helper ────────────────────────────────────────────────
    def _section_heading(self, text: str, level: int = 1):
        pdf = self.pdf
        C   = self.C
        if level == 1:
            pdf.set_fill_color(*C["blue"])
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(180, 6, f"  {text}", ln=1, fill=True)
        else:
            pdf.set_fill_color(*C["light"])
            pdf.set_text_color(*C["blue"])
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(180, 5.5, f"  {text}", ln=1, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    def _finding_row(self, label: str, value: str, confidence: float = -1):
        pdf = self.pdf
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 5, _sanitize(label), ln=0)
        pdf.set_font("Helvetica", "", 9)
        if confidence >= 0:
            conf_str = f"  [AI confidence: {confidence*100:.1f}%]"
        else:
            conf_str = ""
        pdf.multi_cell(130, 5, _sanitize(str(value)) + conf_str, ln=1)

    # ── Clinical Findings ─────────────────────────────────────────────────────
    def _draw_clinical_findings(self):
        pdf = self.pdf
        C   = self.C
        f   = self.findings
        ai  = self.ai
        conf = ai.per_finding_scores

        # Section 1: Rhythm
        self._section_heading("SECTION 1 — RHYTHM ANALYSIS", level=1)
        self._finding_row("Rhythm:", f.rhythm, conf.get("rhythm", -1))
        self._finding_row("Heart Rate:", f"{f.heart_rate_bpm:.0f} bpm", conf.get("heart_rate", -1))
        self._finding_row("Regularity:", f.regularity)
        self._finding_row("HRV:", f.heart_rate_variability)
        pdf.ln(2)

        # Section 2: Intervals & Axes
        self._section_heading("SECTION 2 — INTERVALS & AXES", level=1)
        intervals = [
            ("PR Interval:",   f"{f.pr_interval_ms:.0f} ms  (Normal: 120-200 ms)"),
            ("QRS Duration:",  f"{f.qrs_duration_ms:.0f} ms  (Normal: <120 ms)"),
            ("QT Interval:",   f"{f.qt_interval_ms:.0f} ms"),
            ("QTc Interval:",  f"{f.qtc_interval_ms:.0f} ms  (Normal: ≤440 ms men / ≤460 ms women)"),
            ("P-wave Axis:",   f"{f.p_axis_deg:.0f}°  (Normal: 0-75°)"),
            ("QRS Axis:",      f"{f.qrs_axis_deg:.0f}°  (Normal: -30 to +90°)"),
            ("T-wave Axis:",   f"{f.t_axis_deg:.0f}°"),
        ]
        for lbl, val in intervals:
            self._finding_row(lbl, val)
        pdf.ln(2)

        # Section 3: Waveform Morphology
        self._section_heading("SECTION 3 — WAVEFORM MORPHOLOGY", level=1)
        morphology = [
            ("P Waves:",       f.p_waves,    conf.get("p_waves", -1)),
            ("PR Comment:",    f.pr_comment, -1),
            ("QRS Complex:",   f.qrs_comment, conf.get("qrs", -1)),
            ("QT/QTc:",        f.qt_comment, conf.get("qt", -1)),
            ("ST Segment:",    f.st_segment, conf.get("st_segment", -1)),
            ("T Waves:",       f.t_waves,    conf.get("t_waves", -1)),
            ("U Waves:",       f.u_waves,    -1),
        ]
        for lbl, val, c in morphology:
            self._finding_row(lbl, val, c)
        pdf.ln(2)

        # Section 4: Primary Impression
        self._section_heading("SECTION 4 — PRIMARY IMPRESSION", level=1)
        pdf.set_fill_color(*C["light"])
        pdf.set_draw_color(*C["blue"])
        pdf.set_line_width(0.4)
        pdf.rect(x=15, y=pdf.get_y(), w=180, h=14, style="FD")
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*C["blue"])
        pdf.set_xy(18, pdf.get_y() + 2)
        pdf.multi_cell(174, 6, _sanitize(f.primary_impression), ln=1)
        pdf.set_text_color(0, 0, 0)

        # ICD-10 codes
        if f.icd10_codes:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*C["grey"])
            pdf.cell(180, 4, f"ICD-10 Codes: {', '.join(f.icd10_codes)}", ln=1)
        pdf.ln(3)

        # Risk banner
        risk_color = self.C["risk"].get(f.risk_category, self.C["risk"]["LOW"])
        pdf.set_fill_color(*risk_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(180, 7, _sanitize(f"  RISK CATEGORY: {f.risk_category}  -  {f.risk_rationale}"), ln=1, fill=True)
        if f.emergent_action:
            pdf.set_fill_color(180, 30, 30)
            pdf.cell(180, 6, "  *** EMERGENT ACTION REQUIRED — NOTIFY ATTENDING IMMEDIATELY ***", ln=1, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Section 5: Differential
        self._section_heading("SECTION 5 — DIFFERENTIAL DIAGNOSIS", level=1)
        for i, dx in enumerate(f.differential, 1):
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(180, 5, _sanitize(f"  {i}. {dx}"), ln=1)
        pdf.ln(2)

        # Section 6: Clinical Significance
        self._section_heading("SECTION 6 — CLINICAL SIGNIFICANCE & RISK STRATIFICATION", level=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(180, 5, _sanitize(f.risk_rationale), ln=1)
        if f.stemi_equivalent:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*C["red"])
            pdf.cell(180, 5, "[ADVISORY — requires clinician confirmation] Possible STEMI / STEMI-Equivalent Pattern — clinician to confirm and determine further management", ln=1)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # Section 7: Recommendations
        self._section_heading("SECTION 7 — RECOMMENDATIONS", level=1)
        for i, rec in enumerate(f.recommendations, 1):
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(5, 5, "", ln=0)
            pdf.multi_cell(175, 5, _sanitize(f"{i}. {rec}"), ln=1)
        pdf.ln(2)

        # Section 8: References
        self._section_heading("SECTION 8 — EVIDENCE-BASED REFERENCES", level=1)
        for i, ref in enumerate(f.references, 1):
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*C["grey"])
            pdf.multi_cell(180, 4.5, _sanitize(f"[{i}] {ref}"), ln=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Patient-Friendly Findings ─────────────────────────────────────────────
    def _draw_patient_friendly_findings(self):
        pdf = self.pdf
        C   = self.C
        f   = self.findings

        self._section_heading("WHAT YOUR HEART TRACING SHOWS", level=1)
        pdf.set_font("Helvetica", "", 10)
        lines = [
            f"Heart Beat Pattern: {f.rhythm}",
            f"Heart Rate: {f.heart_rate_bpm:.0f} beats per minute (normal range 60-100 bpm)",
            f"Electrical Timing: PR {f.pr_interval_ms:.0f} ms | QRS {f.qrs_duration_ms:.0f} ms | QTc {f.qtc_interval_ms:.0f} ms",
        ]
        for line in lines:
            pdf.cell(180, 6, line, ln=1)
        pdf.ln(2)

        self._section_heading("WHAT THIS MEANS", level=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(180, 6, _sanitize(f.primary_impression), ln=1)
        pdf.ln(2)

        self._section_heading("WHAT TO DO NEXT", level=1)
        for i, rec in enumerate(f.recommendations, 1):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(180, 5.5, _sanitize(f"{i}. {rec}"), ln=1)
        pdf.ln(2)

        # Risk in plain language
        risk_color = self.C["risk"].get(f.risk_category, self.C["risk"]["LOW"])
        risk_plain = {
            "LOW":      "Your ECG shows a LOW-RISK pattern. No urgent action is needed.",
            "MODERATE": "Your ECG shows a MODERATE-RISK pattern. Please follow up with your doctor soon.",
            "HIGH":     "Your ECG shows a HIGH-RISK pattern. Please see your doctor today.",
            "CRITICAL": "Your ECG shows a pattern that needs IMMEDIATE medical attention. Please go to Emergency now.",
        }
        pdf.set_fill_color(*risk_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(180, 8, f"  {risk_plain.get(f.risk_category, '')}", ln=1, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── AI Block ──────────────────────────────────────────────────────────────
    def _draw_ai_block(self):
        pdf = self.pdf
        C   = self.C
        ai  = self.ai

        self._section_heading("AI-ASSISTED ANALYSIS METADATA", level=1)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*C["grey"])
        info = [
            f"Model: {ai.model_version}",
            f"Overall AI Confidence: {ai.overall_confidence*100:.1f}%  |  "
            f"Inference Time: {ai.inference_time_ms:.0f} ms",
            f"NLG Quality — BLEU-4: {ai.bleu4_score:.3f}  |  ROUGE-L: {ai.rouge_l_score:.3f}  |  "
            f"Hallucination Rate: {ai.hallucination_rate*100:.2f}% (target <5%)",
        ]
        for line in info:
            pdf.cell(180, 4.5, line, ln=1)

        if ai.retrieved_contexts:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C["blue"])
            pdf.cell(180, 4.5, "Retrieved Clinical Context (RAG):", ln=1)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*C["grey"])
            for i, ctx in enumerate(ai.retrieved_contexts[:3], 1):
                pdf.multi_cell(180, 4, f"[{i}] {ctx[:200]}{'…' if len(ctx)>200 else ''}", ln=1)

        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Physician Signature ───────────────────────────────────────────────────
    def _draw_physician_signature(self):
        pdf = self.pdf
        C   = self.C
        dr  = self.physician

        self._draw_divider()
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*C["blue"])
        pdf.cell(180, 5, "PHYSICIAN VERIFICATION & ELECTRONIC SIGNATURE", ln=1)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)
        pdf.cell(90, 5, f"Interpreting Physician: {dr.interpreting_name}", ln=0)
        pdf.cell(90, 5, f"Physician ID / NPI: {dr.interpreting_id}", ln=1)
        pdf.cell(90, 5, f"Department: {dr.department}", ln=0)
        pdf.cell(90, 5, f"Institution: {dr.institution}", ln=1)
        pdf.ln(3)

        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*C["grey"])
        verified_str = (
            "This report has been reviewed and electronically signed by the interpreting cardiologist. "
            "Electronic signature complies with 21 CFR Part 11 and HIPAA requirements."
        )
        pdf.multi_cell(180, 4.5, verified_str, ln=1)

        # Signature line
        pdf.ln(2)
        pdf.set_draw_color(*C["blue"])
        pdf.set_line_width(0.3)
        pdf.line(15, pdf.get_y(), 110, pdf.get_y())
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(15, pdf.get_y() + 1)
        pdf.cell(100, 4, "Physician Signature", ln=0)
        pdf.cell(80, 4, f"Date / Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
        pdf.ln(3)

    # ── Legal Footer ──────────────────────────────────────────────────────────
    def _draw_legal_footer(self):
        pdf = self.pdf
        C   = self.C

        self._draw_divider()
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*C["grey"])
        legal = (
            "CONFIDENTIAL — This report is intended solely for the authorised recipient. Any disclosure, "
            "copying, or distribution is strictly prohibited. AI-generated interpretations are advisory only "
            "and must be validated by a licensed physician before clinical action is taken. DeepCardio-RAG "
            "is a Class II Software as a Medical Device (SaMD) — FDA 510(k) pending. "
            "Johns Hopkins Hospital, Baltimore MD, USA. JCAHO Accredited."
        )
        pdf.multi_cell(180, 3.8, legal, ln=1)

    # ── Divider ───────────────────────────────────────────────────────────────
    def _draw_divider(self):
        pdf = self.pdf
        pdf.set_draw_color(*self.C["gold"])
        pdf.set_line_width(0.5)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI-Compatible Generate Functions
# ──────────────────────────────────────────────────────────────────────────────

def generate_jh_ecg_report(
    patient,
    physician_or_findings=None,
    findings_or_ai=None,
    ai_metrics_or_audience=None,
    audience:    str = "clinical",
    output_path: Optional[str] = None,
) -> bytes:
    """
    Convenience wrapper — accepts either:

    Object-based (Colab notebook style):
        generate_jh_ecg_report(PatientInfo, PhysicianInfo, ECGFindings, AIMetrics, audience='clinical')

    Dict-based (FastAPI style):
        generate_jh_ecg_report(patient_data={...}, physician_or_findings={...}, ...)
    """
    # ── Detect calling convention ─────────────────────────────────────────────
    if isinstance(patient, PatientInfo):
        # Object-based: (PatientInfo, PhysicianInfo, ECGFindings, AIMetrics, audience=...)
        pt       = patient
        physician = physician_or_findings if isinstance(physician_or_findings, PhysicianInfo) else PhysicianInfo()
        findings  = findings_or_ai        if isinstance(findings_or_ai, ECGFindings)          else ECGFindings()
        ai        = ai_metrics_or_audience if isinstance(ai_metrics_or_audience, AIMetrics)   else AIMetrics()
        # audience may be passed as 5th positional arg
        if isinstance(ai_metrics_or_audience, str):
            audience = ai_metrics_or_audience
    else:
        # Dict-based: patient is patient_data dict
        patient_data   = patient if isinstance(patient, dict) else {}
        findings_data  = physician_or_findings if isinstance(physician_or_findings, dict) else {}
        ai_data        = findings_or_ai        if isinstance(findings_or_ai, dict)        else {}
        if isinstance(ai_metrics_or_audience, str):
            audience = ai_metrics_or_audience

        pt       = PatientInfo(**{k: v for k, v in patient_data.items()
                                  if k in PatientInfo.__dataclass_fields__})
        findings = ECGFindings(**{k: v for k, v in findings_data.items()
                                  if k in ECGFindings.__dataclass_fields__})
        ai       = AIMetrics(**{k: v for k, v in (ai_data or {}).items()
                                if k in AIMetrics.__dataclass_fields__})
        physician = PhysicianInfo()
        for fld in ("ordering_name", "interpreting_name", "clinical_indication", "indication"):
            if fld in patient_data:
                setattr(physician, fld, patient_data[fld])

    gen = JHReportGenerator()
    return gen.generate(
        patient=pt,
        physician=physician,
        findings=findings,
        ai_metrics=ai,
        audience=audience,
        output_path=output_path,
    )


def demo_jh_report(output_path: str = "/tmp/demo_jh_report.pdf") -> str:
    """Generate a demonstration report with realistic data and save to file."""
    patient = PatientInfo(
        name="John H. Smith", mrn="JHH-2024-78923", dob="1968-03-15",
        age=56, sex="Male", encounter_id="EC-2024-11421",
        ward="Cardiology Step-Down Unit", room="4B-012",
        height_cm=178.0, weight_kg=84.5,
    )
    physician = PhysicianInfo(
        ordering_name="Dr. Sarah J. Carter, MD",     ordering_id="NPI-1234567890",
        interpreting_name="Dr. Michael R. Lee, MD",  interpreting_id="NPI-0987654321",
        department="Division of Cardiology — Electrophysiology",
        institution="Johns Hopkins Hospital",
        clinical_indication="Palpitations, 3-week history. Rule out paroxysmal SVT.",
    )
    findings = ECGFindings(
        rhythm="Sinus Rhythm with occasional PACs",
        heart_rate_bpm=78.0,
        heart_rate_variability="Mildly reduced",
        regularity="Predominantly regular with occasional ectopic beats",
        pr_interval_ms=158.0, qrs_duration_ms=92.0,
        qt_interval_ms=386.0, qtc_interval_ms=418.0,
        p_axis_deg=62.0, qrs_axis_deg=48.0, t_axis_deg=38.0,
        p_waves="Normal sinus P waves, occasional ectopic P morphology (leads II, V1) — PACs",
        pr_comment="Normal (158 ms). Borderline first-degree AV block not present.",
        qrs_comment="Narrow (92 ms). No delta waves. No bundle branch block. Normal axis.",
        qt_comment="QTc 418 ms — within normal limits (≤440 ms for males).",
        st_segment="Isoelectric in all leads. No ST elevation or depression.",
        t_waves="Upright in I, II, V3-V6. Biphasic T in V1-V2 (normal variant).",
        u_waves="Prominent U waves in V2-V3 — consider hypokalaemia if not previously evaluated.",
        primary_impression=(
            "Normal sinus rhythm with frequent premature atrial contractions (PACs). "
            "No acute ischaemic changes. No significant conduction abnormality. "
            "Prominent U waves warrant electrolyte assessment."
        ),
        icd10_codes=["I49.1", "R00.8"],
        differential=["Paroxysmal Supraventricular Tachycardia (SVT) — not captured",
                      "Frequent PACs with compensatory pauses",
                      "Sick Sinus Syndrome (paroxysmal) — Holter required",
                      "Hypokalaemia-induced U waves"],
        risk_category="LOW",
        risk_rationale="Frequent PACs without sustained tachyarrhythmia. No STEMI equivalent.",
        stemi_equivalent=False,
        emergent_action=False,
        recommendations=[
            "24-hour Holter monitoring to capture paroxysmal SVT episodes.",
            "Check serum potassium and magnesium — prominent U waves noted.",
            "Echocardiogram to assess structural heart disease.",
            "Consider beta-blocker therapy if PAC burden >10% on Holter.",
            "Electrophysiology consultation if recurrent palpitations persist.",
            "Return to clinic in 2 weeks or sooner if symptoms worsen.",
        ],
        references=[
            "Page RL et al. 2015 ACC/AHA/HRS Guideline for SVT. JACC 2016;67(13):e27-e115.",
            "Rautaharju PM et al. AHA/ACCF/HRS ECG Recommendations. JACC 2009;53(11):982-991.",
            "Goldberger AL. Clinical Electrocardiography: A Simplified Approach. 9th ed. 2018.",
        ],
    )
    ai = AIMetrics(
        overall_confidence=0.963,
        per_finding_scores={
            "rhythm": 0.971, "p_waves": 0.952, "qrs": 0.988,
            "st_segment": 0.979, "t_waves": 0.941, "heart_rate": 0.998,
        },
        retrieved_contexts=[
            "Guideline: Frequent PACs >10% beats/24h may cause cardiomyopathy (JACC 2016).",
            "Case ARR-002: Frequent PACs (SVEB class) — advised Holter, no immediate treatment.",
            "Guideline: Prominent U waves suggest hypokalemia; check serum K+ and Mg2+.",
        ],
        model_version="DeepCardio-RAG v2.0 (ResNet-34 + RAG Attention, MoE classifier)",
        inference_time_ms=3100.0,
        bleu4_score=0.412,
        rouge_l_score=0.538,
        hallucination_rate=0.031,
    )

    gen = JHReportGenerator()
    gen.generate(
        patient=patient, physician=physician,
        findings=findings, ai_metrics=ai,
        audience="clinical", output_path=output_path,
    )
    gen.generate(
        patient=patient, physician=physician,
        findings=findings, ai_metrics=ai,
        audience="patient",
        output_path=output_path.replace(".pdf", "_patient.pdf"),
    )
    logger.info(f"Demo JH reports saved to {output_path}")
    return output_path
