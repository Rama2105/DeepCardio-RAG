"""
core/safety_gating.py — Clinical input-validity & output-safety gating
======================================================================
Directly addresses peer-review finding M1 (the most serious): the flagship demo
issued "Activate code blue / Immediate defibrillation" from a random-weight
network on a synthetic signal, reported a physically impossible EF = -0.1% that
propagated unchecked into a risk score, and presented untrained-module outputs
with confidences as if they were measurements.

This module provides three gates that the reporting/risk pipeline must pass every
module output through:

  1. PHYSIOLOGICAL PLAUSIBILITY  — reject/flag values outside clinically possible
     ranges (e.g. ejection fraction must lie in [0, 100]%). An implausible value
     is never allowed to reach a downstream risk score.

  2. TRAINED-STATE SUPPRESSION   — a module whose weights are not genuinely trained
     (or whose confidence is below a floor) has its finding SUPPRESSED rather than
     reported with a confidence. Untrained output must not look like a measurement.

  3. DIRECTIVE -> ADVISORY       — treatment orders ("defibrillate", "activate code
     blue", "administer reperfusion", "coronary angiography") are rewritten as
     observations that REQUIRE clinician confirmation. The system never issues an
     order; it surfaces a finding for a clinician to act on.

  4. CROSS-MODULE CONSISTENCY    — a critical alert is withheld when a second module
     concurrently reports a state incompatible with it. The reviewed demo asserted
     VF/VFL on a patient its OWN ECG module simultaneously read as normal sinus
     rhythm at 72 bpm; gates 1-3 each pass that case individually, because the
     contradiction exists only BETWEEN modules. See check_cross_module_consistency.

None of this is a substitute for regulatory clearance. The system is a research
prototype (Software-as-a-Medical-Device would be Class II/III); see SAMD_STATEMENT.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re


# ---------------------------------------------------------------------------
# 1. Physiological plausibility ranges (clinically possible bounds)
# ---------------------------------------------------------------------------
# (hard_min, hard_max) = physically/clinically possible; values outside are
# INVALID and must be rejected. (typ_min, typ_max) = normal-ish range used only
# to flag "outside typical" without rejecting.
PHYSIOLOGICAL_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ejection_fraction_pct": {"hard": (0.0, 100.0), "typ": (15.0, 75.0)},
    "heart_rate_bpm":        {"hard": (10.0, 350.0), "typ": (40.0, 180.0)},
    "qrs_ms":                {"hard": (20.0, 300.0), "typ": (60.0, 120.0)},
    "pr_ms":                 {"hard": (50.0, 600.0), "typ": (120.0, 220.0)},
    "qtc_ms":                {"hard": (200.0, 800.0), "typ": (350.0, 470.0)},
    "systolic_bp_mmhg":      {"hard": (40.0, 320.0), "typ": (90.0, 140.0)},
    "diastolic_bp_mmhg":     {"hard": (20.0, 220.0), "typ": (60.0, 90.0)},
    "probability":           {"hard": (0.0, 1.0),   "typ": (0.0, 1.0)},
}

# Minimum confidence below which a finding is treated as non-informative.
DEFAULT_CONFIDENCE_FLOOR = 0.35


@dataclass
class GateResult:
    """Outcome of gating a single module finding."""
    surfaced: bool                       # True if the finding may be shown/used
    value: Optional[float]               # gated (possibly clamped) value, or None
    status: str                          # VALID | CLAMPED | INVALID | SUPPRESSED | LOW_CONFIDENCE
    reasons: List[str] = field(default_factory=list)
    used_in_risk_score: bool = False     # may this value feed a combined risk score?


# ---------------------------------------------------------------------------
# 1. Physiological plausibility gate
# ---------------------------------------------------------------------------
def validate_measurement(name: str, value: Optional[float],
                         clamp: bool = False) -> GateResult:
    """
    Validate a physiological measurement against its clinically possible range.

    If `clamp` is True and the value is outside the hard range, it is clamped to
    the boundary and marked CLAMPED (still usable). If `clamp` is False, an
    out-of-range value is marked INVALID and is NOT allowed into a risk score.
    """
    if value is None:
        return GateResult(False, None, "INVALID", ["value is None / not computed"], False)
    rng = PHYSIOLOGICAL_RANGES.get(name)
    if rng is None:
        return GateResult(True, float(value), "VALID", [f"no range defined for '{name}'"], True)

    lo, hi = rng["hard"]
    tlo, thi = rng["typ"]
    reasons: List[str] = []
    if value < lo or value > hi:
        if clamp:
            v = max(lo, min(hi, float(value)))
            return GateResult(True, v, "CLAMPED",
                              [f"{name}={value} outside possible [{lo},{hi}] -> clamped to {v}"], True)
        return GateResult(False, None, "INVALID",
                          [f"{name}={value} is physiologically impossible "
                           f"(outside [{lo},{hi}]) — rejected before risk scoring"], False)
    if value < tlo or value > thi:
        reasons.append(f"{name}={value} outside typical [{tlo},{thi}] — flag for review")
    return GateResult(True, float(value), "VALID", reasons, True)


# ---------------------------------------------------------------------------
# 2. Trained-state / confidence suppression gate
# ---------------------------------------------------------------------------
def gate_module_output(module: str, is_trained: bool, confidence: Optional[float],
                       value: Optional[float] = None,
                       confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR) -> GateResult:
    """
    Decide whether a module's finding may be surfaced / used in a risk score.

    - If the module is NOT genuinely trained, the finding is SUPPRESSED (not shown
      with a confidence, not fed to the risk score). This is the M1 fix: an
      untrained module must not produce clinical-looking output.
    - If confidence is below the floor, the finding is LOW_CONFIDENCE (may be shown
      as inconclusive, but excluded from the risk score).
    """
    if not is_trained:
        return GateResult(False, None, "SUPPRESSED",
                          [f"{module} is not independently trained — output withheld "
                           f"(not reported with a confidence, not used in risk score)"], False)
    if confidence is not None and confidence < confidence_floor:
        return GateResult(True, value, "LOW_CONFIDENCE",
                          [f"{module} confidence {confidence:.2f} < floor {confidence_floor} — "
                           f"reported as inconclusive, excluded from risk score"], False)
    return GateResult(True, value, "VALID", [], True)


# ---------------------------------------------------------------------------
# 3. Directive -> advisory reframing
# ---------------------------------------------------------------------------
# Imperative treatment orders that must never be issued by the system, mapped to
# clinician-facing observations. Order matters (longer/more specific first).
_DIRECTIVE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bactivate\s+code\s+blue\b", re.I),
     "finding consistent with a potential arrest rhythm — clinician to assess whether "
     "resuscitation activation is warranted"),
    (re.compile(r"\bimmediate\s+defibrillation\b", re.I),
     "pattern that may correspond to a shockable rhythm — clinician to confirm before "
     "any decision to deliver a shock"),
    (re.compile(r"\bdefibrillat(e|ion)\b", re.I),
     "possible shockable-rhythm finding requiring clinician confirmation"),
    (re.compile(r"\bCPR\s+if\s+pulseless\b", re.I),
     "if the patient is found pulseless, standard clinical protocols apply (clinician-directed)"),
    (re.compile(r"\bepinephrine\s+every\s+[\d\-]+\s*min\b", re.I),
     "medication dosing is a clinician decision and is not recommended by this system"),
    (re.compile(r"\b(administer|initiate)\s+reperfusion(\s+therapy)?\b", re.I),
     "findings that a clinician may wish to evaluate for reperfusion suitability"),
    (re.compile(r"\bcoronary\s+angiography\b", re.I),
     "findings a clinician may consider when deciding on further imaging such as angiography"),
    (re.compile(r"\bsynchronized\s+cardioversion\b", re.I),
     "finding a clinician may consider when evaluating rhythm-control options"),
]

ADVISORY_PREFIX = "[ADVISORY — requires clinician confirmation] "

# Text that already attributes the decision to a clinician is CURATED advisory prose.
# It may legitimately name a therapy ("shockable-rhythm management (defibrillation,
# CPR, medications) is performed under clinician direction") without ordering it, and
# running the substitutions over it rewrites that reference mid-sentence. Detecting
# these markers is what makes reframe_directives() safe to apply to any string.
_CURATED_MARKERS = re.compile(
    r"clinician[- ](?:direct(?:ed|ion)|decision|confirmation)|clinician to |"
    r"not (?:recommended|ordered)(?:\s+or\s+\w+)? by this system|"
    r"are clinician decisions|requires clinician|under clinician",
    re.I,
)


def reframe_directives(text: str) -> str:
    """
    Rewrite imperative treatment orders in `text` as advisory observations.

    IDEMPOTENT: text that already carries ADVISORY_PREFIX is returned unchanged.
    Curated guideline constants (e.g. vfdb_loader.VFDB_GUIDELINES) are already
    written as clinician-facing observations and legitimately mention terms like
    "defibrillation" while attributing the decision to a clinician. Re-running the
    substitutions over them would rewrite those references mid-sentence and stack a
    second prefix, so the guard is what makes it safe to apply this pass broadly.
    """
    if not text:
        return text
    if text.lstrip().startswith(ADVISORY_PREFIX.strip()) or _CURATED_MARKERS.search(text):
        return text
    out = text
    changed = False
    for pat, replacement in _DIRECTIVE_PATTERNS:
        if pat.search(out):
            out = pat.sub(replacement, out)
            changed = True
    return (ADVISORY_PREFIX + out) if changed else out


def reframe_guideline_list(guidelines: List[str]) -> List[str]:
    """Apply directive->advisory reframing to a list of retrieved guideline strings."""
    return [reframe_directives(g) for g in (guidelines or [])]


# ---------------------------------------------------------------------------
# 4. Cross-module consistency
# ---------------------------------------------------------------------------
# Rhythms that are incompatible with an organised sinus rhythm at a normal rate.
# VF/VFL is disorganised depolarisation with no effective cardiac output; VT and
# asystole likewise cannot coexist with an organised sinus rhythm in the same
# recording window. If a rhythm module asserts one of these while the ECG module
# concurrently reports normal sinus rhythm, at least one of the two is wrong and
# the alert must not be issued on the strength of the other.
_MALIGNANT_RHYTHMS = {"VF", "VFL", "VF/VFL", "VT", "ASYS", "ASYSTOLE",
                      "VENTRICULAR FIBRILLATION", "VENTRICULAR FLUTTER",
                      "VENTRICULAR TACHYCARDIA"}
_ORGANISED_RHYTHMS = {"NORMAL", "NSR", "N", "NORMAL SINUS RHYTHM", "SINUS RHYTHM"}

# A perfusing sinus rate: VF/asystole is incompatible with a rate in this band.
_ORGANISED_RATE_BAND = (45.0, 130.0)


@dataclass
class ConsistencyResult:
    consistent: bool
    conflicts: List[str] = field(default_factory=list)
    suppress_alert: bool = False


def _norm(v: Any) -> str:
    return str(v or "").strip().upper()


def check_cross_module_consistency(findings: Dict[str, Dict[str, Any]]) -> ConsistencyResult:
    """
    Detect mutually contradictory findings ACROSS modules (peer-review M1a).

    `findings` maps module name -> dict, of which this reads:
        vfdb  : {'rhythm_class': str}          — asserted rhythm
        ecg   : {'rhythm_class': str,          — concurrently reported rhythm
                 'heart_rate_bpm': float}      — and rate, if measured

    Returns ConsistencyResult; `suppress_alert` is True when a malignant-rhythm
    alert is contradicted, meaning the caller must NOT surface it as critical.

    This is deliberately conservative: it fires only on contradictions that are
    physiologically impossible, not on merely unusual combinations. A disagreement
    it cannot adjudicate is reported as a conflict but does not by itself suppress.
    """
    conflicts: List[str] = []
    suppress = False

    vfdb = findings.get("vfdb") or findings.get("ventricular") or {}
    ecg = findings.get("ecg") or findings.get("ecg_arrhythmia") or {}

    asserted = _norm(vfdb.get("rhythm_class"))
    concurrent = _norm(ecg.get("rhythm_class"))

    if asserted in _MALIGNANT_RHYTHMS:
        if concurrent in _ORGANISED_RHYTHMS:
            conflicts.append(
                f"VFDB asserts '{vfdb.get('rhythm_class')}' while the ECG module "
                f"concurrently reports '{ecg.get('rhythm_class')}' for the same record — "
                f"these rhythms cannot coexist; critical alert withheld pending "
                f"clinician review of the raw signal")
            suppress = True

        hr = ecg.get("heart_rate_bpm")
        if hr is not None:
            try:
                hr = float(hr)
            except (TypeError, ValueError):
                hr = None
        if hr is not None and _ORGANISED_RATE_BAND[0] <= hr <= _ORGANISED_RATE_BAND[1]:
            conflicts.append(
                f"VFDB asserts '{vfdb.get('rhythm_class')}' while a perfusing organised "
                f"rate of {hr:.0f} bpm is reported — incompatible; critical alert withheld")
            suppress = True

    return ConsistencyResult(not conflicts, conflicts, suppress)


# ---------------------------------------------------------------------------
# Combined risk-score guard
# ---------------------------------------------------------------------------
def collect_risk_inputs(module_findings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given per-module findings, return only those that passed BOTH the trained-state
    gate and (where applicable) the physiological gate — i.e. the values that are
    actually allowed to contribute to a combined risk score.

    Each finding dict may contain: is_trained (bool), confidence (float),
    measurement (name,value) tuple, value (float).
    Returns {'usable': {module: value}, 'excluded': {module: [reasons]}}.
    """
    usable, excluded = {}, {}
    for module, f in module_findings.items():
        g = gate_module_output(module, f.get("is_trained", False),
                               f.get("confidence"), f.get("value"))
        if not g.used_in_risk_score:
            excluded[module] = g.reasons
            continue
        meas = f.get("measurement")
        if meas:
            name, val = meas
            mg = validate_measurement(name, val, clamp=False)
            if not mg.used_in_risk_score:
                excluded[module] = mg.reasons
                continue
            usable[module] = mg.value
        else:
            usable[module] = g.value if g.value is not None else f.get("value")
    return {"usable": usable, "excluded": excluded}


SAMD_STATEMENT = (
    "REGULATORY STATUS: DeepCardio-RAG is a research prototype. A system that "
    "produces diagnostic findings or treatment-relevant outputs from patient data "
    "would, if deployed, constitute Software as a Medical Device (SaMD) and require "
    "regulatory clearance (e.g. FDA 510(k)/De Novo, EU MDR CE marking) and "
    "prospective clinical validation. It is NOT cleared for clinical use. Automation "
    "bias is a known risk: clinicians must independently verify every finding, and "
    "the system issues no treatment orders."
)


if __name__ == "__main__":
    # Self-check demonstrating the three gates on the exact M1 failure case.
    print("EF = -0.1% (the impossible value from the review):")
    print("  ", validate_measurement("ejection_fraction_pct", -0.1, clamp=False))
    print("VFDB alert from an UNTRAINED (random-weight) module:")
    print("  ", gate_module_output("VFDB", is_trained=False, confidence=0.325))
    print("Directive reframing:")
    print("  ", reframe_directives(
        "EMERGENCY: Ventricular Fibrillation detected. Activate code blue. "
        "Immediate defibrillation. CPR if pulseless. Epinephrine every 3-5 min."))
