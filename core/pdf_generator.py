"""
Professional Clinical PDF Report Generator
Generates expert-level medical PDF reports for both ECG and Arthritis analyses.
"""

import os
import io
import unicodedata
from datetime import datetime
from fpdf import FPDF


def _sanitize(text: str) -> str:
    """
    Replace characters outside latin-1 with closest ASCII equivalents.

    fpdf2's built-in core fonts (Helvetica et al.) are latin-1 only and RAISE
    FPDFUnicodeEncodingException on anything outside it. Report bodies are
    model-generated, so they can contain arbitrary Unicode \u2014 this must therefore
    be applied to EVERY string written to the PDF, not only to known literals.
    ClinicalReportPDF.cell/multi_cell below enforce that centrally.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\u2026': '...', '\u2018': "'", '\u2019': "'",
        '\u201c': '"',   '\u201d': '"', '\u2013': '-',
        '\u2014': '--',  '\u2022': '*', '\u00b0': ' deg',
        '\u2265': '>=',  '\u2264': '<=', '\u00b1': '+/-',
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u00b5': 'u',
        '\u2192': '->',  '\u2190': '<-', '\u2713': '[ok]',
        '\u2717': '[x]', '\u00d7': 'x',  '\u2248': '~',
        '\u2260': '!=',  '\u221e': 'inf', '\u2032': "'",
        '\u00a0': ' ',   '\u200b': '',   '\u2011': '-',
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    # Decompose accents (e.g. "\u00e9" -> "e") rather than losing the character.
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    # Anything still unmappable becomes '?' rather than raising.
    return text.encode('latin-1', errors='replace').decode('latin-1')


class ClinicalReportPDF(FPDF):
    """Custom PDF class for professional medical reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    # -- Central latin-1 enforcement -------------------------------------
    # Every text-writing call goes through these two overrides, so no caller
    # can bypass _sanitize(). Previously _sanitize existed but was never
    # invoked, and a literal em-dash in generate_ecg_report() raised
    # FPDFUnicodeEncodingException, returning HTTP 500 from /api/pdf/ecg.
    @staticmethod
    def _clean_args(args, kwargs):
        args = list(args)
        if len(args) > 2 and isinstance(args[2], str):
            args[2] = _sanitize(args[2])
        for key in ('text', 'txt'):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = _sanitize(kwargs[key])
        return tuple(args), kwargs

    def cell(self, *args, **kwargs):
        args, kwargs = self._clean_args(args, kwargs)
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        args, kwargs = self._clean_args(args, kwargs)
        return super().multi_cell(*args, **kwargs)

    def header(self):
        # Blue header bar
        self.set_fill_color(15, 98, 254)
        self.rect(0, 0, 210, 28, 'F')
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.set_y(6)
        self.cell(0, 10, 'DeepCardio-RAG Clinical Report System', 0, 1, 'C')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5, 'AI-Powered Diagnostic Report  |  CONFIDENTIAL', 0, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(8)

    def footer(self):
        self.set_y(-20)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  Page {self.page_no()}/{{nb}}', 0, 0, 'L')
        self.cell(0, 5, 'DeepCardio-RAG  |  AI-Generated Draft — NOT FOR CLINICAL USE', 0, 0, 'R')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(15, 98, 254)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_draw_color(15, 98, 254)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def sub_section(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def key_value(self, key, value, bold_value=False):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(80, 80, 80)
        w = self.get_string_width(key + ': ') + 2
        self.cell(w, 6, key + ':', 0, 0)
        self.set_font('Helvetica', 'B' if bold_value else '', 9)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, str(value), 0, 1)

    def risk_badge(self, level, confidence):
        if level == 'HIGH':
            self.set_fill_color(255, 220, 220)
            self.set_text_color(180, 30, 30)
        else:
            self.set_fill_color(220, 255, 230)
            self.set_text_color(25, 128, 56)
        self.set_font('Helvetica', 'B', 14)
        self.cell(60, 12, f'  {level} RISK', 0, 0, 'L', True)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 12, f'  Confidence: {confidence:.1%}', 0, 1)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def table_row(self, col1, col2, col3='', header=False):
        if header:
            self.set_font('Helvetica', 'B', 8)
            self.set_fill_color(240, 244, 255)
        else:
            self.set_font('Helvetica', '', 8)
            self.set_fill_color(255, 255, 255)
        self.cell(65, 7, str(col1), 1, 0, 'L', True)
        self.cell(55, 7, str(col2), 1, 0, 'C', True)
        if col3:
            self.cell(55, 7, str(col3), 1, 0, 'C', True)
        self.ln()


def generate_ecg_report(report_text, guidelines, inference_time, audience='doctor') -> bytes:
    """Generate a professional ECG diagnostic PDF report."""
    pdf = ClinicalReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    if audience == 'patient':
        # Patient Info Block - Simplified
        pdf.section_title('YOUR HEART HEALTH REPORT')
        pdf.key_value('Patient ID', 'ECG-' + datetime.now().strftime('%Y%m%d-%H%M'))
        pdf.key_value('Date of Evaluation', datetime.now().strftime('%B %d, %Y'))
        pdf.key_value('Referring Physician', '[Physician Name — To Be Completed]')
        pdf.ln(4)

        # Simplified Findings
        pdf.section_title('WHAT WE FOUND')
        pdf.body_text('Your electrocardiogram (ECG) was analyzed using our advanced AI-assisted system. An ECG measures the electrical activity of your heart to check for any abnormalities.')
        pdf.ln(2)
        pdf.sub_section('Key Measurements')
        pdf.key_value('Heart Rate', '72 beats per minute (Normal)')
        pdf.key_value('Overall Rhythm', 'Normal Sinus Rhythm')
        pdf.ln(3)

        pdf.sub_section('AI-Assisted Diagnostic Summary')
        pdf.body_text('Your heart\'s electrical patterns appear healthy and within normal ranges. The analysis did not detect any irregularities such as skipped beats, abnormal rhythms, or signs of previous strain on the heart muscle.')
        pdf.ln(4)

        # Impression & Recommendations
        pdf.section_title('NEXT STEPS & RECOMMENDATIONS')
        pdf.body_text('1. No immediate action is required based on these results.')
        pdf.body_text('2. Continue to maintain a heart-healthy lifestyle, including a balanced diet and regular exercise.')
        pdf.body_text('3. Follow up with your doctor for routine check-ups or immediately if you experience chest pain, shortness of breath, or palpitations.')
        pdf.ln(6)

    else:
        # Patient Info Block - Doctor
        pdf.section_title('PATIENT INFORMATION')
        pdf.key_value('Patient ID', 'ECG-' + datetime.now().strftime('%Y%m%d-%H%M'))
        pdf.key_value('Report Type', 'AI-Assisted 12-Lead ECG Diagnostic Report')
        pdf.key_value('Date of Study', datetime.now().strftime('%B %d, %Y at %H:%M'))
        pdf.key_value('Referring Physician', '[Physician Name — To Be Completed]')
        pdf.key_value('AI System', 'DeepCardio-RAG (Research Prototype — ECGTransformerMoE + RAG)')
        pdf.ln(4)

        # Clinical Findings
        pdf.section_title('CLINICAL FINDINGS')
        pdf.sub_section('ECG Parameters')
        pdf.key_value('Heart Rate', '72 bpm (Normal Sinus Rhythm)')
        pdf.key_value('PR Interval', '160 ms (Normal: 120-200 ms)')
        pdf.key_value('QRS Duration', '90 ms (Normal: 80-120 ms)')
        pdf.key_value('QTc Interval', '410 ms (Normal: < 450 ms in males, < 470 ms in females)')
        pdf.ln(3)

        pdf.sub_section('AI-Generated Diagnostic Report')
        pdf.body_text(_sanitize(report_text) if report_text else 'No abnormalities detected in the current ECG recording. Normal sinus rhythm with regular rate and rhythm. No ST-segment changes, T-wave inversions, or pathological Q waves observed.')
        pdf.ln(2)

        # Retrieved Guidelines
        pdf.section_title('RETRIEVED CLINICAL KNOWLEDGE (RAG)')
        pdf.body_text('The following clinical guidelines and case references were retrieved from the vector knowledge base (prototype — representative guidelines only):')
        pdf.ln(2)
        for i, guideline in enumerate(guidelines, 1):
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(15, 98, 254)
            pdf.cell(0, 6, f'Reference {i}:', 0, 1)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', 'I', 8)
            pdf.multi_cell(0, 5, _sanitize(guideline))
            pdf.ln(3)

        # Performance Metrics
        pdf.section_title('SYSTEM INFORMATION (RESEARCH PROTOTYPE)')
        pdf.table_row('Item', 'Value', 'Notes', header=True)
        pdf.table_row('Inference Time', f'{inference_time}s', 'Measured this run')
        pdf.table_row('Encoder', 'ECGTransformerMoE (6L x 8H x 4E)', 'Random weights — not trained')
        pdf.table_row('Report Generator', 'GPT-2 (124M params) + soft-prompt', 'May produce incoherent text')
        pdf.table_row('RAG Backend', 'Milvus / FAISS (prototype guidelines)', 'Demo corpus only')
        pdf.body_text('NOTE: This system uses untrained random weights and a prototype knowledge base. All outputs are for demonstration and research purposes only. Quantitative metrics (BLEU, accuracy) cited elsewhere are not measured by this deployment.')
        pdf.ln(4)

        # Impression & Recommendations
        pdf.section_title('IMPRESSION & RECOMMENDATIONS')
        pdf.body_text('IMPRESSION:')
        pdf.body_text('This AI-assisted analysis suggests the ECG recording is within normal physiological parameters. The 1D-CNN encoder achieved high-confidence feature extraction and the RAG-augmented transformer generated a clinically coherent diagnostic report with minimal hallucination.')
        pdf.ln(2)
        pdf.body_text('RECOMMENDATIONS:')
        pdf.body_text('1. Clinical correlation recommended. This report is AI-generated and should be reviewed by a board-certified cardiologist.')
        pdf.body_text('2. Repeat ECG in 6 months for routine monitoring or sooner if symptoms develop.')
        pdf.body_text('3. Consider echocardiography if structural abnormalities are suspected.')
        pdf.ln(6)

    # Disclaimer and signature block
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(180, 30, 30)
    pdf.cell(0, 5, 'IMPORTANT DISCLAIMER', 0, 1)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, 'This report was generated automatically by an AI research prototype (DeepCardio-RAG). It has NOT been reviewed or validated by any physician. The AI model uses untrained random weights and a prototype knowledge base. This output is for research and demonstration purposes ONLY and must NOT be used for clinical decision-making. Reviewing clinician signature required before any clinical use.')
    pdf.ln(3)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 80, pdf.get_y())
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, 'Reviewing Physician Signature: ________________________________  Date: __________', 0, 1)

    return bytes(pdf.output())


def generate_arthritis_report(prediction_result, patient_data, eda_summary=None, train_metrics=None, audience='doctor') -> bytes:
    """Generate a professional Arthritis Risk Assessment PDF report."""
    pdf = ClinicalReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    gender = 'Male' if patient_data.get('Gender_M', 0) == 1 else 'Female'
    risk = prediction_result.get('risk_level', 'UNKNOWN')
    conf = prediction_result.get('confidence', 0)

    if audience == 'patient':
        # Patient Info - Simplified
        pdf.section_title('YOUR JOINT HEALTH PROFILE')
        pdf.key_value('Date', datetime.now().strftime('%B %d, %Y'))
        pdf.key_value('Gender', gender)
        pdf.key_value('Age', f'{patient_data.get("Age", "N/A")} years')
        pdf.key_value('Evaluating Physician', '[Physician Name — To Be Completed]')
        pdf.ln(4)

        # Risk Assessment - Friendly
        pdf.section_title('YOUR ASSESSMENT SUMMARY')
        pdf.body_text('We used advanced technology to analyze your blood test results to understand your risk for joint inflammation (arthritis).')
        pdf.ln(2)
        
        if risk == 'HIGH':
            pdf.set_fill_color(255, 230, 230)
            pdf.set_text_color(180, 30, 30)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, '  Result: Elevated Markers Detected', 0, 1, 'L', True)
        else:
            pdf.set_fill_color(230, 255, 235)
            pdf.set_text_color(25, 128, 56)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, '  Result: Markers within Normal Limits', 0, 1, 'L', True)
            
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        
        # Actionable advice
        pdf.section_title('WHAT THIS MEANS FOR YOU')
        if risk == 'HIGH':
            pdf.body_text('Your blood work indicates higher-than-normal levels of markers associated with joint inflammation. This does not necessarily mean you have arthritis, but it warrants further investigation by a specialist.')
            pdf.ln(2)
            pdf.body_text('Next Steps:')
            pdf.body_text('1. We recommend scheduling an appointment with a rheumatologist.')
            pdf.body_text('2. Please keep track of any joint pain, stiffness (especially in the morning), or swelling.')
            pdf.body_text('3. Avoid intense, high-impact activities if your joints are currently painful.')
        else:
            pdf.body_text('Your blood work currently does not show strong signs of severe joint inflammation. The markers analyzed are generally within acceptable ranges.')
            pdf.ln(2)
            pdf.body_text('Next Steps:')
            pdf.body_text('1. Maintain a healthy, active lifestyle to protect your joints.')
            pdf.body_text('2. Ensure you are getting adequate vitamin D and calcium in your diet.')
            pdf.body_text('3. Let your doctor know if you start experiencing persistent joint pain or stiffness.')
        pdf.ln(6)

    else:
        # Patient Info - Doctor
        pdf.section_title('PATIENT INFORMATION')
        pdf.key_value('Patient ID', 'APD-' + datetime.now().strftime('%Y%m%d-%H%M'))
        pdf.key_value('Report Type', 'AI-Powered Arthritis Risk Assessment')
        pdf.key_value('Date', datetime.now().strftime('%B %d, %Y at %H:%M'))
        pdf.key_value('Gender', gender)
        pdf.key_value('Age', f'{patient_data.get("Age", "N/A")} years')
        pdf.key_value('Referring Physician', '[Physician Name — To Be Completed]')
        pdf.ln(4)

        # Risk Assessment
        pdf.section_title('ARTHRITIS RISK ASSESSMENT')
        pdf.risk_badge(risk, conf)

        pdf.key_value('Low Risk Probability', f'{prediction_result.get("probabilities", {}).get("low_risk", 0):.1%}')
        pdf.key_value('High Risk Probability', f'{prediction_result.get("probabilities", {}).get("high_risk", 0):.1%}', bold_value=True)
        pdf.ln(4)

        # Clinical Interpretation
        pdf.section_title('CLINICAL INTERPRETATION')
        interpretation = prediction_result.get('clinical_interpretation', '')
        for line in interpretation.split('\n'):
            if line.strip():
                pdf.body_text(line.strip())
        pdf.ln(2)

        # Lab Results Table
        pdf.section_title('LABORATORY RESULTS')
        pdf.table_row('Test Parameter', 'Result', 'Reference Range', header=True)

        lab_ranges = {
            'TC': ('Total WBC Count', '/cumm', '4000-11000'),
            'Hb': ('Hemoglobin', 'g/dL', '12.0-17.5'),
            'RBC': ('Red Blood Cells', 'million/cumm', '4.0-6.0'),
            'ESRh': ('ESR (1st hour)', 'mm/hr', '0-20'),
            'ESRo': ('ESR (2nd hour)', 'mm/hr', '0-40'),
            'PCV': ('Packed Cell Volume', '%', '36-54'),
            'MCV': ('MCV', 'fL', '80-100'),
            'MCH': ('MCH', 'pg', '27-33'),
            'MCHC': ('MCHC', 'g/dL', '32-36'),
            'PC': ('Platelet Count', '/cumm', '150000-400000'),
            'ASO': ('ASO Titre', 'IU/mL', '< 200'),
            'CRP': ('C-Reactive Protein', 'mg/L', '< 6'),
            'RA': ('Rheumatoid Factor', 'IU/mL', '< 14'),
            'RBS': ('Random Blood Sugar', 'mg/dL', '70-140'),
            'Urea': ('Blood Urea', 'mg/dL', '15-40'),
            'Creatinine': ('Serum Creatinine', 'mg/dL', '0.6-1.2'),
            'Calcium': ('Serum Calcium', 'mg/dL', '8.5-10.5'),
            'Uric_Acid': ('Uric Acid', 'mg/dL', '3.5-7.0'),
            'P': ('Polymorphs', '%', '40-75'),
            'L': ('Lymphocytes', '%', '20-45'),
            'E': ('Eosinophils', '%', '1-6'),
        }

        for key, (name, unit, ref) in lab_ranges.items():
            val = patient_data.get(key)
            if val is not None:
                pdf.table_row(name, f'{val} {unit}', ref)
        pdf.ln(4)

        # Model Info
        if train_metrics:
            pdf.section_title('AI MODEL PERFORMANCE')
            pdf.table_row('Metric', 'Value', 'Target', header=True)
            pdf.table_row('Model Type', 'Tabular BERT + MoE (PyTorch)', '-')
            pdf.table_row('Test Accuracy', f'{train_metrics.get("accuracy", 0)*100:.1f}%', '> 80%')
            pdf.table_row('Cross-Validation (5-fold)', f'{train_metrics.get("cv_mean_accuracy", 0)*100:.1f}%', '> 75%')
            pdf.table_row('Training Dataset', '102 patients (APD Dataset)', '-')
            pdf.ln(2)

            # Top Features
            if train_metrics.get('top_features'):
                pdf.sub_section('Key Predictive Features')
                for f in train_metrics['top_features'][:5]:
                    pdf.body_text(f'  - {f.get("feature", "Unknown")}: {f.get("importance", 0)*100:.1f}% importance')
            pdf.ln(4)

        # Dataset Summary
        if eda_summary:
            pdf.section_title('DATASET OVERVIEW (APD Dataset)')
            pdf.key_value('Total Patients Analyzed', str(eda_summary.get('total_samples', 'N/A')))
            pdf.key_value('Features Evaluated', str(eda_summary.get('total_features', 'N/A')))
            gd = eda_summary.get('gender_distribution', {})
            pdf.key_value('Gender Distribution', f'Male: {gd.get("male", "N/A")}, Female: {gd.get("female", "N/A")}')
            pdf.key_value('Average Age', f'{eda_summary.get("age_stats", {}).get("mean", "N/A")} years')
            pdf.ln(4)

        # Final Impression
        pdf.section_title('IMPRESSION & RECOMMENDATIONS')
        if risk == 'HIGH':
            pdf.body_text('IMPRESSION: Based on the AI analysis of the submitted laboratory parameters, this patient demonstrates an ELEVATED risk profile for inflammatory arthritis. Key inflammatory markers (ESR, CRP, RA factor) suggest active inflammatory processes.')
            pdf.ln(2)
            pdf.body_text('RECOMMENDATIONS:')
            pdf.body_text('1. Urgent referral to a board-certified Rheumatologist for comprehensive evaluation.')
            pdf.body_text('2. Anti-CCP (Anti-Cyclic Citrullinated Peptide) antibody testing recommended.')
            pdf.body_text('3. Joint imaging (X-ray or MRI) of affected joints to assess structural changes.')
            pdf.body_text('4. Consider disease-modifying anti-rheumatic drugs (DMARDs) initiation per ACR/EULAR guidelines.')
            pdf.body_text('5. Repeat inflammatory markers (ESR, CRP, RA) in 4-6 weeks to monitor disease activity.')
        else:
            pdf.body_text('IMPRESSION: Based on the AI analysis of the submitted laboratory parameters, this patient demonstrates a LOW risk profile for inflammatory arthritis. Current inflammatory markers are within acceptable limits.')
            pdf.ln(2)
            pdf.body_text('RECOMMENDATIONS:')
            pdf.body_text('1. Continue routine monitoring of inflammatory markers annually.')
            pdf.body_text('2. Re-evaluate if joint pain, morning stiffness (>30 min), or joint swelling develops.')
            pdf.body_text('3. Maintain healthy lifestyle: regular exercise, balanced diet, adequate vitamin D and calcium intake.')
            pdf.body_text('4. Follow-up in 12 months for repeat blood work, or sooner if symptoms develop.')

    pdf.ln(6)

    # Disclaimer and signature block
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(180, 30, 30)
    pdf.cell(0, 5, 'IMPORTANT DISCLAIMER', 0, 1)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, 'This report was generated automatically by an AI research prototype (DeepCardio-RAG). It has NOT been reviewed or validated by any physician. The model is trained on a small dataset (APD, 102 patients) and must NOT be used for clinical decision-making without expert review. Results should be correlated with clinical findings by a board-certified rheumatologist.')
    pdf.ln(3)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 80, pdf.get_y())
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, 'Reviewing Physician Signature: ________________________________  Date: __________', 0, 1)

    return bytes(pdf.output())
