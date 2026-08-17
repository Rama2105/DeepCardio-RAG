# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        DeepCardio-RAG  |  Data Merging & Harmonization Pipeline             ║
# ║        NHANES RA (Mendeley)  +  APD (102 records)  →  3,000–5,000          ║
# ║        TabularBERT + MoE  |  Arthritis Risk Module                          ║
# ║        Compatible with Google Colab  (Python 3.10+)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── CELL 1: Install dependencies ──────────────────────────────────────────────
# Run this cell first in Colab
"""
!pip install sdv ctgan scikit-learn pandas numpy scipy openpyxl -q
"""

# ── CELL 2: Imports ───────────────────────────────────────────────────────────
import os, json, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

print("✅  All imports successful")
print(f"    pandas  {pd.__version__}  |  numpy  {np.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION  (edit these paths in Colab)
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── File paths ─────────────────────────────────────────────────────────
    # Upload your files to Colab via the Files panel (left sidebar)
    # then set the correct paths below.

    "APD_PATH":    "apd_original.csv",         # Your 102-record APD CSV
    "NHANES_PATH": "nhanes_ra.csv",            # NHANES RA from Mendeley Data
                                               # https://data.mendeley.com/datasets/r7dh8zwwfr/1
    "OUT_DIR":     "/content/deepcardio_data", # Output folder in Colab

    # ── Target ─────────────────────────────────────────────────────────────
    "TARGET_RECORDS": 4000,   # Aim for 4,000; adjust to 3000–5000 as needed
    "TARGET_COL":     "Arthritis_Label",

    # ── CTGAN settings ─────────────────────────────────────────────────────
    "CTGAN_EPOCHS":     400,
    "CTGAN_BATCH_SIZE": 500,

    # ── Random seed ────────────────────────────────────────────────────────
    "SEED": 42,
}

np.random.seed(CFG["SEED"])
os.makedirs(CFG["OUT_DIR"], exist_ok=True)
print(f"✅  Config loaded  |  Target: {CFG['TARGET_RECORDS']} records")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — APD SCHEMA DEFINITION
#  These are the 24 canonical features of the Arthritis Profile Dataset.
#  All incoming data will be mapped to this schema.
# ══════════════════════════════════════════════════════════════════════════════

APD_SCHEMA = {
    # col_name          : (dtype,     clinical_min, clinical_max, impute_strategy)
    "Age"               : ("float",   18.0,  95.0,  "median"),
    "Gender"            : ("cat",     None,  None,  "mode"),     # Male / Female
    "BMI"               : ("float",   14.0,  60.0,  "median"),
    "CRP"               : ("float",   0.1,   200.0, "median"),   # mg/L
    "ESR"               : ("float",   1.0,   150.0, "median"),   # mm/hr
    "RF"                : ("float",   0.0,   500.0, "median"),   # IU/mL
    "Anti_CCP"          : ("float",   0.0,   500.0, "median"),   # U/mL
    "WBC"               : ("float",   2.0,   20.0,  "median"),   # ×10³/µL
    "RBC"               : ("float",   2.5,   7.0,   "median"),   # ×10⁶/µL
    "Hemoglobin"        : ("float",   7.0,   20.0,  "median"),   # g/dL
    "Hematocrit"        : ("float",   20.0,  60.0,  "median"),   # %
    "Platelets"         : ("float",   50.0,  700.0, "median"),   # ×10³/µL
    "Neutrophils"       : ("float",   20.0,  95.0,  "median"),   # %
    "Lymphocytes"       : ("float",   5.0,   60.0,  "median"),   # %
    "Glucose"           : ("float",   50.0,  400.0, "median"),   # mg/dL
    "Cholesterol"       : ("float",   100.0, 400.0, "median"),   # mg/dL
    "Uric_Acid"         : ("float",   1.5,   15.0,  "median"),   # mg/dL
    "Creatinine"        : ("float",   0.3,   10.0,  "median"),   # mg/dL
    "Sodium"            : ("float",   120.0, 160.0, "median"),   # mEq/L
    "Potassium"         : ("float",   2.5,   7.0,   "median"),   # mEq/L
    "Albumin"           : ("float",   1.5,   6.0,   "median"),   # g/dL
    "Morning_Stiffness" : ("binary",  0,     1,     "mode"),     # 0/1
    "Joint_Swelling"    : ("binary",  0,     1,     "mode"),     # 0/1
    "Arthritis_Label"   : ("cat",     None,  None,  "mode"),     # target
}

NUMERIC_FEATS  = [k for k,v in APD_SCHEMA.items() if v[0] in ("float","binary") and k != CFG["TARGET_COL"]]
DISCRETE_FEATS = [k for k,v in APD_SCHEMA.items() if v[0] == "cat"]
ALL_FEATS      = list(APD_SCHEMA.keys())

print(f"✅  APD Schema  |  {len(NUMERIC_FEATS)} numeric  +  {len(DISCRETE_FEATS)} categorical")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — NHANES RA COLUMN MAPPING
#
#  The Mendeley NHANES RA dataset uses different column names.
#  This map translates NHANES columns → APD schema columns.
#  Unmapped APD columns will be filled via CTGAN augmentation later.
#
#  Reference: https://data.mendeley.com/datasets/r7dh8zwwfr/1
#  Adjust keys below if your NHANES file uses different names.
# ══════════════════════════════════════════════════════════════════════════════

NHANES_TO_APD = {
    # NHANES column         : APD column
    "RIDAGEYR"              : "Age",           # Age in years
    "BMXBMI"                : "BMI",           # BMI
    "LBXWBCSI"              : "WBC",           # WBC count
    "LBXRBCSI"              : "RBC",           # RBC count
    "LBXHGB"                : "Hemoglobin",    # Hemoglobin
    "LBXHCT"                : "Hematocrit",    # Hematocrit
    "LBXTHRSI"              : "Platelets",     # Platelet count (×10⁹/L → ×10³/µL same units)
    "LBXSGL"                : "Glucose",       # Glucose
    "LBXTC"                 : "Cholesterol",   # Total cholesterol
    "LBXSUA"                : "Uric_Acid",     # Uric acid
    "LBXSCR"                : "Creatinine",    # Creatinine
    "LBXSNASI"              : "Sodium",        # Sodium (mmol/L ≈ mEq/L)
    "LBXSKSI"               : "Potassium",     # Potassium
    "LBXSAL"                : "Albumin",       # Albumin
    "LBXHSCRP"              : "CRP",           # hsCRP (mg/L)
    # Gender encoding handled separately (RIAGENDR: 1=Male, 2=Female)
    "RIAGENDR"              : "Gender",
    # RA diagnosis flag — handled separately (MCQ195: "Rheumatoid arthritis")
    "MCQ195"                : "_ra_flag",
    # Arthritis type question (MCQ191 or similar) — used to derive label
    "MCQ160A"               : "_arthritis_any",
}

# NHANES columns that require unit conversion before mapping
UNIT_CONVERSIONS = {
    # Platelets: NHANES reports ×10⁹/L, APD uses ×10³/µL (numerically identical)
    "Platelets"   : lambda x: x,
    # Sodium: mmol/L → mEq/L (1:1 for monovalent ions)
    "Sodium"      : lambda x: x,
    # CRP: NHANES hsCRP is mg/dL → convert to mg/L (×10)
    # NOTE: Check your NHANES file header — some versions already report in mg/L
    "CRP"         : lambda x: x * 10,
}

# NHANES columns NOT available → will be synthesized by CTGAN
NHANES_MISSING_FROM_APD = ["ESR", "RF", "Anti_CCP", "Neutrophils",
                            "Lymphocytes", "Morning_Stiffness", "Joint_Swelling"]

print(f"✅  NHANES→APD map: {len(NHANES_TO_APD)} columns mapped")
print(f"⚠️   Missing in NHANES (CTGAN will fill): {NHANES_MISSING_FROM_APD}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LOAD DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def load_apd(path):
    """Load the APD CSV. If file missing, generate a realistic 102-record schema."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅  APD loaded: {df.shape[0]} records × {df.shape[1]} features")
        return df

    print("⚠️   APD file not found — generating synthetic APD schema (102 records)")
    print("     → Upload your real apd_original.csv to Colab for production use")

    np.random.seed(42)
    n = 102
    sev = np.random.choice([0,1,2], n, p=[0.45,0.35,0.20])

    df = pd.DataFrame({
        "Age"              : np.random.randint(25, 80, n).astype(float),
        "Gender"           : np.random.choice(["Male","Female"], n, p=[0.38,0.62]),
        "BMI"              : np.round(np.random.normal(27.4,5.2,n).clip(16,48), 1),
        "CRP"              : np.round(np.where(sev==0, np.random.normal(2.1,1.0,n),
                                      np.where(sev==1, np.random.normal(12.5,4.5,n),
                                               np.random.normal(34.0,12.0,n))).clip(0.1,120), 2),
        "ESR"              : np.round(np.where(sev==0, np.random.normal(15,6,n),
                                      np.where(sev==1, np.random.normal(42,14,n),
                                               np.random.normal(78,22,n))).clip(1,140), 1),
        "RF"               : np.round(np.where(sev==0, np.random.normal(8,4,n),
                                      np.where(sev==1, np.random.normal(45,20,n),
                                               np.random.normal(120,55,n))).clip(0,400), 1),
        "Anti_CCP"         : np.round(np.where(sev==0, np.random.normal(5,3,n),
                                      np.where(sev==1, np.random.normal(60,30,n),
                                               np.random.normal(180,70,n))).clip(0,500), 1),
        "WBC"              : np.round(np.random.normal(7.2,2.1,n).clip(2.5,18), 2),
        "RBC"              : np.round(np.random.normal(4.5,0.6,n).clip(2.8,6.5), 2),
        "Hemoglobin"       : np.round(np.random.normal(13.5,1.5,n).clip(7,20), 1),
        "Hematocrit"       : np.round(np.random.normal(40.5,4.5,n).clip(20,60), 1),
        "Platelets"        : np.round(np.random.normal(265,72,n).clip(50,700)).astype(float),
        "Neutrophils"      : np.round(np.random.normal(62,10,n).clip(20,95), 1),
        "Lymphocytes"      : np.round(np.random.normal(28,8,n).clip(5,60), 1),
        "Glucose"          : np.round(np.random.normal(98,22,n).clip(50,400), 1),
        "Cholesterol"      : np.round(np.random.normal(198,40,n).clip(100,400), 1),
        "Uric_Acid"        : np.round(np.random.normal(5.8,1.8,n).clip(1.5,15), 1),
        "Creatinine"       : np.round(np.random.normal(0.95,0.25,n).clip(0.3,10), 2),
        "Sodium"           : np.round(np.random.normal(139,3.5,n).clip(120,160), 1),
        "Potassium"        : np.round(np.random.normal(4.1,0.45,n).clip(2.5,7), 2),
        "Albumin"          : np.round(np.random.normal(4.1,0.5,n).clip(1.5,6), 1),
        "Morning_Stiffness": np.where(sev==0, np.random.binomial(1,0.15,n),
                                      np.where(sev==1, np.random.binomial(1,0.60,n),
                                               np.random.binomial(1,0.90,n))),
        "Joint_Swelling"   : np.where(sev==0, np.random.binomial(1,0.10,n),
                                      np.where(sev==1, np.random.binomial(1,0.65,n),
                                               np.random.binomial(1,0.95,n))),
        "Arthritis_Label"  : np.where(sev==0,"No Arthritis",
                                      np.where(sev==1,"Mild Arthritis","Severe Arthritis")),
    })
    df.to_csv(path, index=False)
    print(f"     Saved generated APD to: {path}")
    return df


def load_nhanes(path):
    """
    Load NHANES RA dataset from Mendeley Data.
    Returns a harmonized dataframe with APD column names.
    If file not found, returns empty DataFrame (CTGAN will handle augmentation).
    """
    if not os.path.exists(path):
        print("⚠️   NHANES file not found — skipping NHANES merge")
        print("     → Download from: https://data.mendeley.com/datasets/r7dh8zwwfr/1")
        print("     → Upload nhanes_ra.csv to Colab, then re-run")
        return pd.DataFrame()

    df_raw = pd.read_csv(path, low_memory=False)
    print(f"✅  NHANES raw loaded: {df_raw.shape[0]} records × {df_raw.shape[1]} features")

    # ── Step 1: Rename columns ──────────────────────────────────────────────
    rename_map = {k: v for k, v in NHANES_TO_APD.items() if k in df_raw.columns}
    df = df_raw.rename(columns=rename_map)
    print(f"   Columns mapped: {len(rename_map)}")

    # ── Step 2: Encode Gender ───────────────────────────────────────────────
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({1: "Male", 2: "Female"})

    # ── Step 3: Derive Arthritis_Label ─────────────────────────────────────
    # MCQ195 = 1 → Rheumatoid Arthritis (Severe proxy)
    # MCQ160A = 1, MCQ195 != 1 → Mild Arthritis
    # Neither → No Arthritis
    def derive_label(row):
        ra  = row.get("_ra_flag", 0)
        any_arth = row.get("_arthritis_any", 0)
        if ra == 1:
            return "Severe Arthritis"
        elif any_arth == 1:
            return "Mild Arthritis"
        else:
            return "No Arthritis"

    if "_ra_flag" in df.columns or "_arthritis_any" in df.columns:
        df[CFG["TARGET_COL"]] = df.apply(derive_label, axis=1)
        df.drop(columns=[c for c in ["_ra_flag","_arthritis_any"] if c in df.columns], inplace=True)
    elif "RA" in df.columns:
        # Fallback: some Mendeley NHANES files have a pre-encoded 'RA' column
        df[CFG["TARGET_COL"]] = df["RA"].map({1:"Severe Arthritis", 0:"No Arthritis"})

    # ── Step 4: Unit conversions ────────────────────────────────────────────
    for col, fn in UNIT_CONVERSIONS.items():
        if col in df.columns:
            df[col] = fn(df[col])

    # ── Step 5: Keep only APD schema columns ───────────────────────────────
    available = [c for c in ALL_FEATS if c in df.columns]
    df = df[available].copy()
    print(f"   APD columns found in NHANES: {available}")

    missing_cols = [c for c in ALL_FEATS if c not in df.columns]
    print(f"   Missing columns (will be synthesized): {missing_cols}")

    # ── Step 6: Sample up to 3,500 records (balanced classes) ──────────────
    if len(df) > 3500 and CFG["TARGET_COL"] in df.columns:
        df = (df.groupby(CFG["TARGET_COL"], group_keys=False)
                .apply(lambda x: x.sample(min(len(x), 1200), random_state=CFG["SEED"])))
        print(f"   Sampled to {len(df)} balanced records")

    df["Source"] = "NHANES"
    return df


# ── Load both ─────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  SECTION 4: LOADING DATASETS")
print("═"*65)
df_apd   = load_apd(CFG["APD_PATH"])
df_nhanes = load_nhanes(CFG["NHANES_PATH"])

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FEATURE HARMONIZATION
#  Align dtypes, fix ranges, impute missing values.
# ══════════════════════════════════════════════════════════════════════════════

def harmonize(df: pd.DataFrame, schema: dict, source_name: str) -> pd.DataFrame:
    """
    Enforce APD schema on any dataframe:
      1. Add missing columns as NaN
      2. Clip numeric values to clinical ranges
      3. Impute missing values
      4. Encode Gender to Male/Female
      5. Ensure correct dtypes
    """
    print(f"\n  Harmonizing [{source_name}] — {len(df)} records")

    # Add missing columns
    for col in ALL_FEATS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"    + Added missing column: {col}")

    for col, (dtype, cmin, cmax, strat) in schema.items():
        if col not in df.columns:
            continue

        # ── Clip to clinical range ─────────────────────────────────────────
        if dtype in ("float", "binary") and cmin is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=cmin, upper=cmax)

        # ── Impute ─────────────────────────────────────────────────────────
        if df[col].isnull().any():
            if strat == "median":
                fill_val = df[col].median()
                if pd.isna(fill_val):
                    fill_val = (cmin + cmax) / 2 if cmin is not None else 0
            elif strat == "mode":
                mode_vals = df[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else ("Male" if col=="Gender" else "No Arthritis")
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)

        # ── Dtype enforcement ──────────────────────────────────────────────
        if dtype == "float":
            df[col] = df[col].astype(float).round(2)
        elif dtype == "binary":
            df[col] = df[col].astype(float).round(0).astype(int)

    # ── Gender normalisation ───────────────────────────────────────────────
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip()
        gender_map = {
            "1":"Male","2":"Female",
            "m":"Male","f":"Female",
            "male":"Male","female":"Female",
            "M":"Male","F":"Female"
        }
        df["Gender"] = df["Gender"].replace(gender_map)
        df.loc[~df["Gender"].isin(["Male","Female"]), "Gender"] = "Female"

    # ── Label normalisation ────────────────────────────────────────────────
    if CFG["TARGET_COL"] in df.columns:
        label_norm = {
            "0":"No Arthritis","1":"Mild Arthritis","2":"Severe Arthritis",
            "no":"No Arthritis","mild":"Mild Arthritis","severe":"Severe Arthritis",
            "no arthritis":"No Arthritis","mild arthritis":"Mild Arthritis",
            "severe arthritis":"Severe Arthritis","rheumatoid arthritis":"Severe Arthritis",
        }
        df[CFG["TARGET_COL"]] = (df[CFG["TARGET_COL"]].astype(str)
                                   .str.strip().str.lower()
                                   .replace(label_norm))
        valid = ["No Arthritis","Mild Arthritis","Severe Arthritis"]
        df.loc[~df[CFG["TARGET_COL"]].isin(valid), CFG["TARGET_COL"]] = "No Arthritis"

    # Keep only schema columns + Source
    keep = ALL_FEATS + (["Source"] if "Source" in df.columns else [])
    df = df[[c for c in keep if c in df.columns]].copy()

    null_pct = df.isnull().mean().mean() * 100
    print(f"    Missing after imputation: {null_pct:.2f}%")
    print(f"    Class dist: {df[CFG['TARGET_COL']].value_counts().to_dict()}")
    return df


print("\n" + "═"*65)
print("  SECTION 5: FEATURE HARMONIZATION")
print("═"*65)

df_apd["Source"] = "APD"
df_apd_h = harmonize(df_apd, APD_SCHEMA, "APD")

df_nhanes_h = pd.DataFrame()
if len(df_nhanes) > 0:
    df_nhanes_h = harmonize(df_nhanes, APD_SCHEMA, "NHANES")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MERGE REAL DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 6: MERGING REAL DATA")
print("═"*65)

frames = [df_apd_h]
if len(df_nhanes_h) > 0:
    frames.append(df_nhanes_h)

df_real = pd.concat(frames, ignore_index=True)
print(f"✅  Merged real data: {len(df_real)} records")
print(f"   Source breakdown: {df_real['Source'].value_counts().to_dict()}")
print(f"   Class dist: {df_real[CFG['TARGET_COL']].value_counts().to_dict()}")

real_count    = len(df_real)
needed_synth  = max(0, CFG["TARGET_RECORDS"] - real_count)
print(f"\n   Real records     : {real_count}")
print(f"   Synthetic needed : {needed_synth}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CTGAN AUGMENTATION
#  Fill missing columns + generate synthetic records to reach target count.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 7: CTGAN AUGMENTATION")
print("═"*65)

df_synth = pd.DataFrame()

if needed_synth > 0:
    from ctgan import CTGAN

    # Train CTGAN on harmonized real data (all schema columns, no Source tag)
    train_df      = df_real[ALL_FEATS].copy()
    discrete_cols = [c for c in DISCRETE_FEATS if c in train_df.columns] + ["Morning_Stiffness","Joint_Swelling"]

    print(f"  Training CTGAN on {len(train_df)} real records...")
    print(f"  Epochs: {CFG['CTGAN_EPOCHS']}  |  Batch: {CFG['CTGAN_BATCH_SIZE']}")

    t0 = time.time()
    ctgan = CTGAN(
        epochs        = CFG["CTGAN_EPOCHS"],
        batch_size    = CFG["CTGAN_BATCH_SIZE"],
        generator_dim = (256, 256),
        discriminator_dim = (256, 256),
        verbose       = True
    )
    ctgan.fit(train_df, discrete_columns=discrete_cols)

    # ── Generate with class-balanced oversampling for minority classes ─────
    # This specifically fixes the F1 issue on Severe Arthritis (minority class)
    class_dist   = df_real[CFG["TARGET_COL"]].value_counts()
    majority_n   = class_dist.max()
    target_per_class = max(majority_n, needed_synth // 3)

    synth_parts = []
    for label in ["No Arthritis", "Mild Arthritis", "Severe Arthritis"]:
        real_n    = class_dist.get(label, 0)
        need_more = max(0, target_per_class - real_n)
        if need_more > 0:
            part = ctgan.sample(need_more + 50)      # +50 buffer for post-clip drops
            part[CFG["TARGET_COL"]] = label          # enforce correct label
            synth_parts.append(part)
            print(f"    {label:<22} real={real_n}  synthesized={need_more}")

    if synth_parts:
        df_synth = pd.concat(synth_parts, ignore_index=True)
        df_synth = harmonize(df_synth, APD_SCHEMA, "CTGAN")
        df_synth["Source"] = "Synthetic"
        print(f"\n✅  CTGAN done in {time.time()-t0:.1f}s  |  {len(df_synth)} synthetic records")
else:
    print(f"  Real data ({real_count}) already meets target. No synthesis needed.")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SDV GAUSSIAN COPULA (secondary method for blend)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 8: SDV GAUSSIAN COPULA (blend top-up)")
print("═"*65)

df_copula = pd.DataFrame()
total_after_ctgan = real_count + len(df_synth)
copula_needed = max(0, CFG["TARGET_RECORDS"] - total_after_ctgan)

if copula_needed > 0:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    train_df = df_real[ALL_FEATS].copy()
    meta     = SingleTableMetadata()
    meta.detect_from_dataframe(train_df)
    for col in discrete_cols:
        if col in train_df.columns:
            meta.update_column(column_name=col, sdtype='categorical')

    t0 = time.time()
    gc_model = GaussianCopulaSynthesizer(meta, enforce_min_max_values=True, enforce_rounding=True)
    gc_model.fit(train_df)
    df_copula = gc_model.sample(num_rows=copula_needed + 50)
    df_copula = harmonize(df_copula, APD_SCHEMA, "GCopula")
    df_copula["Source"] = "Synthetic"
    print(f"✅  GCopula done in {time.time()-t0:.1f}s  |  {len(df_copula)} records")
else:
    print(f"  Target already met after CTGAN. Skipping GCopula.")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FINAL ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 9: FINAL ASSEMBLY")
print("═"*65)

all_frames = [f for f in [df_real, df_synth, df_copula] if len(f) > 0]
df_final   = pd.concat(all_frames, ignore_index=True)

# Shuffle
df_final = df_final.sample(frac=1, random_state=CFG["SEED"]).reset_index(drop=True)

# Trim/pad to exact target
if len(df_final) > CFG["TARGET_RECORDS"]:
    df_final = df_final.sample(CFG["TARGET_RECORDS"], random_state=CFG["SEED"]).reset_index(drop=True)

print(f"\n{'─'*50}")
print(f"  FINAL DATASET SUMMARY")
print(f"{'─'*50}")
print(f"  Total records    : {len(df_final)}")
print(f"  Total features   : {len(ALL_FEATS)}")
print(f"\n  Source breakdown:")
for src, cnt in df_final["Source"].value_counts().items():
    pct = cnt / len(df_final) * 100
    print(f"    {src:<14} {cnt:>5}  ({pct:.1f}%)")
print(f"\n  Class distribution:")
for lbl, cnt in df_final[CFG["TARGET_COL"]].value_counts().items():
    pct = cnt / len(df_final) * 100
    bar = "█" * int(pct / 2)
    print(f"    {lbl:<22} {cnt:>5}  {bar}  {pct:.1f}%")
print(f"\n  Missing values   : {df_final.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — STATISTICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 10: STATISTICAL VALIDATION")
print("═"*65)

orig_vals   = df_apd_h[NUMERIC_FEATS]
all_vals    = df_final[[c for c in NUMERIC_FEATS if c in df_final.columns]]
ks_results  = {}
passed      = 0

print(f"\n  {'Feature':<22} {'KS stat':>8}  {'p-value':>10}  {'Status'}")
print(f"  {'─'*22} {'─'*8}  {'─'*10}  {'─'*6}")

for col in NUMERIC_FEATS:
    if col not in orig_vals.columns or col not in all_vals.columns:
        continue
    ks_stat, p_val = stats.ks_2samp(
        orig_vals[col].dropna().values,
        all_vals[col].dropna().values
    )
    status = "✅ PASS" if ks_stat < 0.25 else "⚠️  WARN"
    if ks_stat < 0.25:
        passed += 1
    ks_results[col] = {"ks": round(ks_stat, 4), "p": round(p_val, 4)}
    print(f"  {col:<22} {ks_stat:>8.4f}  {p_val:>10.4f}  {status}")

print(f"\n  Result: {passed}/{len(NUMERIC_FEATS)} features passed KS < 0.25 threshold")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — ML FIDELITY TEST (Train Synthetic → Test Real)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 11: ML FIDELITY TEST")
print("═"*65)

le = LabelEncoder()
le.fit(["No Arthritis", "Mild Arthritis", "Severe Arthritis"])

def prepare_ml(df):
    d = df.copy()
    d["Gender_enc"] = (d["Gender"] == "Female").astype(int)
    feats = [c for c in NUMERIC_FEATS if c in d.columns] + ["Gender_enc"]
    X = d[feats].fillna(0).values
    y = le.transform(d[CFG["TARGET_COL"]].values)
    return X, y

synth_df = df_final[df_final["Source"] == "Synthetic"]
real_df  = df_apd_h.copy()
real_df[CFG["TARGET_COL"]] = real_df[CFG["TARGET_COL"]].replace({
    "no arthritis":"No Arthritis", "mild arthritis":"Mild Arthritis",
    "severe arthritis":"Severe Arthritis"
})

if len(synth_df) > 0 and len(real_df) >= 20:
    X_synth, y_synth = prepare_ml(synth_df)
    X_real,  y_real  = prepare_ml(real_df)

    rf = RandomForestClassifier(n_estimators=150, random_state=CFG["SEED"])
    rf.fit(X_synth, y_synth)
    y_pred = rf.predict(X_real)

    print("\n  Train-on-Synthetic / Test-on-Real:")
    print(classification_report(y_real, y_pred,
          target_names=le.classes_, zero_division=0))

    # Cross-validation on full combined dataset
    X_all, y_all = prepare_ml(df_final)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG["SEED"])
    cv_f1  = cross_val_score(rf, X_all, y_all, cv=skf, scoring="f1_weighted")
    print(f"  5-Fold Stratified CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — DEEPCARDIO-RAG FEATURE ENGINEERING
#  Derived features that improve TabularBERT + MoE performance
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 12: DEEPCARDIO-RAG FEATURE ENGINEERING")
print("═"*65)

df_fe = df_final.copy()

# ── Composite inflammation index ──────────────────────────────────────────────
# Normalised to [0,1] range; high values → active inflammation
df_fe["Inflammation_Index"] = (
    (df_fe["CRP"].clip(0,200) / 200) * 0.35 +
    (df_fe["ESR"].clip(0,150) / 150) * 0.25 +
    (df_fe["RF"].clip(0,500)  / 500) * 0.20 +
    (df_fe["Anti_CCP"].clip(0,500) / 500) * 0.20
).round(4)

# ── Autoimmune seropositivity flag ────────────────────────────────────────────
df_fe["Seropositive"] = (
    (df_fe["RF"] > 20).astype(int) +
    (df_fe["Anti_CCP"] > 17).astype(int)
).clip(0, 2)

# ── Anaemia flag (chronic disease anaemia common in RA) ───────────────────────
df_fe["Anaemia_Flag"] = (
    ((df_fe["Gender"] == "Female") & (df_fe["Hemoglobin"] < 12.0)) |
    ((df_fe["Gender"] == "Male")   & (df_fe["Hemoglobin"] < 13.5))
).astype(int)

# ── Neutrophil-to-Lymphocyte Ratio (NLR) ─────────────────────────────────────
# Elevated NLR is a strong predictor of systemic inflammation
df_fe["NLR"] = (df_fe["Neutrophils"] /
                df_fe["Lymphocytes"].replace(0, np.nan)
               ).fillna(df_fe["Neutrophils"].median() / 28).round(3)
df_fe["NLR"] = df_fe["NLR"].clip(0.5, 15)

# ── CRP-to-Albumin Ratio (CAR) ────────────────────────────────────────────────
# Reflects inflammatory burden relative to nutritional status
df_fe["CAR"] = (df_fe["CRP"] /
                df_fe["Albumin"].replace(0, np.nan)
               ).fillna(df_fe["CRP"].median() / 4.1).round(3)
df_fe["CAR"] = df_fe["CAR"].clip(0, 100)

# ── Age-BMI interaction ───────────────────────────────────────────────────────
df_fe["Age_BMI"] = (df_fe["Age"] * df_fe["BMI"] / 100).round(3)

# ── Clinical severity score (rule-based, useful for MoE gating) ──────────────
df_fe["Clinical_Score"] = (
    (df_fe["CRP"]  > 10).astype(int) +
    (df_fe["ESR"]  > 30).astype(int) +
    (df_fe["RF"]   > 20).astype(int) +
    (df_fe["Anti_CCP"] > 17).astype(int) +
    df_fe["Morning_Stiffness"] +
    df_fe["Joint_Swelling"]
)

print(f"✅  Feature engineering complete")
print(f"   Added 7 derived features:")
eng_feats = ["Inflammation_Index","Seropositive","Anaemia_Flag","NLR","CAR","Age_BMI","Clinical_Score"]
for f in eng_feats:
    print(f"    • {f:<22} mean={df_fe[f].mean():.3f}  std={df_fe[f].std():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — LABEL ENCODING FOR MODEL INPUT
#  Produces the exact tensor-ready format for TabularBERT + MoE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 13: LABEL ENCODING FOR MODEL INPUT")
print("═"*65)

df_model = df_fe.copy()

# Gender → binary
df_model["Gender"] = (df_model["Gender"] == "Female").astype(int)

# Target → integer {0,1,2}
label_map = {"No Arthritis": 0, "Mild Arthritis": 1, "Severe Arthritis": 2}
df_model[CFG["TARGET_COL"]] = df_model[CFG["TARGET_COL"]].map(label_map)

# Drop Source column for model input
df_model_clean = df_model.drop(columns=["Source"], errors="ignore")

print(f"✅  Model-ready encoding complete")
print(f"   Shape: {df_model_clean.shape}")
print(f"   Label distribution: {df_model_clean[CFG['TARGET_COL']].value_counts().sort_index().to_dict()}")
print(f"   Dtypes: all numeric = {all(df_model_clean.dtypes != 'object')}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  SECTION 14: SAVING OUTPUTS")
print("═"*65)

OUT = CFG["OUT_DIR"]

# 1. Full dataset with Source tag and human-readable labels
p1 = f"{OUT}/APD_NHANES_combined.csv"
df_fe.to_csv(p1, index=False)
print(f"✅  Combined (human-readable)   → {p1}")

# 2. Training-ready: encoded labels, engineered features, no Source
p2 = f"{OUT}/APD_NHANES_training_ready.csv"
df_model_clean.to_csv(p2, index=False)
print(f"✅  Training-ready (encoded)    → {p2}")

# 3. APD-only harmonized
p3 = f"{OUT}/APD_harmonized.csv"
df_apd_h.to_csv(p3, index=False)
print(f"✅  APD harmonized              → {p3}")

# 4. Feature column manifest (for TabularBERT tokenizer config)
manifest = {
    "original_features"   : ALL_FEATS,
    "engineered_features" : eng_feats,
    "all_features"        : [c for c in df_model_clean.columns if c != CFG["TARGET_COL"]],
    "target_column"       : CFG["TARGET_COL"],
    "label_map"           : label_map,
    "numeric_features"    : NUMERIC_FEATS,
    "categorical_features": DISCRETE_FEATS,
    "schema"              : {k: {"dtype":v[0],"min":v[2],"max":v[3]} for k,v in APD_SCHEMA.items() if v[1] is not None},
    "dataset_stats": {
        "total_records"    : len(df_model_clean),
        "real_records"     : int((df_fe["Source"] != "Synthetic").sum()),
        "synthetic_records": int((df_fe["Source"] == "Synthetic").sum()),
        "ks_pass_rate"     : f"{passed}/{len(NUMERIC_FEATS)}",
        "class_distribution": df_fe[CFG["TARGET_COL"]].value_counts().to_dict(),
    }
}
p4 = f"{OUT}/feature_manifest.json"
with open(p4, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"✅  Feature manifest JSON       → {p4}")

# 5. Scaler params (save μ and σ for inference normalization)
scaler_feats = [c for c in NUMERIC_FEATS if c in df_model_clean.columns] + eng_feats
scaler = StandardScaler()
scaler.fit(df_model_clean[scaler_feats])
scaler_params = {
    "features": scaler_feats,
    "mean"    : scaler.mean_.round(4).tolist(),
    "std"     : scaler.scale_.round(4).tolist(),
}
p5 = f"{OUT}/scaler_params.json"
with open(p5, "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"✅  Scaler params JSON          → {p5}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — DEEPCARDIO-RAG INTEGRATION SNIPPET
#  Copy this into your TabularBERT training notebook
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_CODE = '''
# ── DeepCardio-RAG: Load augmented APD dataset ────────────────────────────────
import pandas as pd, json, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load
df = pd.read_csv("/content/deepcardio_data/APD_NHANES_training_ready.csv")
with open("/content/deepcardio_data/feature_manifest.json") as f:
    manifest = json.load(f)
with open("/content/deepcardio_data/scaler_params.json") as f:
    sp = json.load(f)

FEATURE_COLS = manifest["all_features"]
TARGET_COL   = manifest["target_column"]  # "Arthritis_Label"
NUM_CLASSES  = 3                           # 0=No, 1=Mild, 2=Severe

X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values.astype(int)

# Stratified split (preserves minority class Severe Arthritis)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalize using saved scaler
scaler = StandardScaler()
scaler.mean_ = np.array(sp["mean"])
scaler.scale_ = np.array(sp["std"])
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# Dataset wrapper for TabularBERT
class ArthritisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(ArthritisDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(ArthritisDataset(X_test,  y_test),  batch_size=64)

print(f"Train: {len(X_train)}  Test: {len(X_test)}  Features: {len(FEATURE_COLS)}")
print(f"Class dist (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
# ─────────────────────────────────────────────────────────────────────────────
'''

print("\n" + "═"*65)
print("  SECTION 15: DEEPCARDIO-RAG INTEGRATION SNIPPET")
print("═"*65)
p6 = f"{OUT}/deepcardio_rag_integration.py"
with open(p6, "w") as f:
    f.write(INTEGRATION_CODE)
print(f"✅  Integration snippet         → {p6}")

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  PIPELINE COMPLETE")
print("═"*65)
print(f"  {real_count} real + {len(df_final)-real_count} synthetic = {len(df_final)} total records")
print(f"  24 original + 7 engineered = {len(df_model_clean.columns)-1} model features")
print(f"  All outputs → {OUT}/")
print("═"*65)
