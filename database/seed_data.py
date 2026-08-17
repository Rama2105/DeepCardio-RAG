import os
import sys

# Optional dependency check
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Install requirements first.")
    sys.exit(1)

# Ensure the database module can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database.db_manager import init_db, drop_collection

# Hand-written prototype clinical knowledge base — this list IS the entire
# corpus. It is not a sample of a larger one.
#
# Entries tagged with a "superclass" cover the five PTB-XL diagnostic
# superclasses (NORM, MI, STTC, CD, HYP), which is what core/pipeline.py queries
# the store with. Content restates well-established ECG criteria (Sokolow-Lyon,
# Cornell voltage, Romhilt-Estes, Sgarbossa, the Fourth Universal Definition of
# MI, and the AHA/ACCF/HRS ECG standardization recommendations). Criteria are
# attributed by NAME only — no volume/page citations are invented here. These
# are teaching-level summaries for a research prototype and have NOT been
# reviewed by a cardiologist; do not present them as a validated guideline set.
MOCK_KNOWLEDGE_BASE = [
    {"id": "gdl_001", "type": "guideline", "text": "Guideline: In the presence of ST-segment elevation in leads V2-V4, consider anterior myocardial infarction. Administer appropriate reperfusion therapy rapidly."},
    {"id": "case_2410", "type": "case", "text": "Case 2410: Patient exhibited irregular R-R intervals with absence of distinct P waves, indicating Atrial Fibrillation. Rate control and anticoagulation indicated."},
    {"id": "gdl_002", "type": "guideline", "text": "Guideline: Premature ventricular contractions (PVCs) with a frequency > 10/min or multifocal origin warrant further investigation for structural heart disease."},
    {"id": "case_3051", "type": "case", "text": "Case 3051: T-wave inversion in leads V1-V3 in a young athlete. Diagnosed with Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC)."},
    # ── Heart Disease (UCI Cleveland) clinical guidelines ──
    {"id": "gdl_hd_001", "type": "guideline", "text": "Guideline: Patients with chest pain type 0 (typical angina), ST depression >2mm during exercise, and ≥1 major vessel narrowed on fluoroscopy have high probability of obstructive coronary artery disease. Refer for coronary angiography per ACC/AHA guidelines."},
    {"id": "gdl_hd_002", "type": "guideline", "text": "Guideline: Exercise-induced angina combined with chronotropic incompetence (<85% of age-predicted max heart rate) is a strong predictor of multivessel CAD and poor prognosis. Consider early invasive strategy."},
    {"id": "gdl_hd_003", "type": "guideline", "text": "Guideline: Serum cholesterol >240 mg/dL with resting blood pressure >140 mm Hg in patients >55 years significantly increases 10-year ASCVD risk. Initiate high-intensity statin therapy and antihypertensive medication."},
    {"id": "gdl_hd_004", "type": "guideline", "text": "Guideline: Reversible thalassemia defect (thal=3) on nuclear stress imaging indicates inducible ischaemia. Correlate with coronary anatomy. Revascularization may be indicated per ISCHEMIA trial criteria."},
    {"id": "case_hd_001", "type": "case", "text": "Case HD-001: Male, 63, typical angina, cholesterol 310 mg/dL, exercise HR 112 (68% predicted max), ST depression 2.4 mm, 2 vessels narrowed. Diagnosed with 3-vessel CAD. Underwent CABG."},
    {"id": "case_hd_002", "type": "case", "text": "Case HD-002: Female, 52, atypical angina, resting BP 148/92, fasting glucose 135 mg/dL, thalach 162. Normal fluoroscopy. Managed with lifestyle modification and metformin. Follow-up stress test at 6 months."},
    # ── ECG Arrhythmia (MIT-BIH) clinical guidelines ──
    {"id": "gdl_arr_001", "type": "guideline", "text": "Guideline: Isolated premature ventricular complexes (PVCs/VEB) in structurally normal hearts are generally benign. However, frequent PVCs (>10% of heartbeats/24h) may cause cardiomyopathy and warrant ablation consideration."},
    {"id": "gdl_arr_002", "type": "guideline", "text": "Guideline: Supraventricular ectopic beats (SVEB) are common and usually benign. However, frequent SVEBs are associated with increased risk of atrial fibrillation. Consider Holter monitoring and echocardiography."},
    {"id": "gdl_arr_003", "type": "guideline", "text": "Guideline: Fusion beats indicate simultaneous activation from two foci (supraventricular and ventricular). Common in paced rhythms and sustained VT. Evaluate pacemaker function and underlying rhythm."},
    {"id": "gdl_arr_004", "type": "guideline", "text": "Guideline: AAMI EC57 standard classifies heartbeats into 5 classes: Normal (N), SVEB (S), VEB (V), Fusion (F), and Unknown/Paced (Q). VEB class requires highest clinical attention due to sudden cardiac death risk."},
    {"id": "case_arr_001", "type": "case", "text": "Case ARR-001: MIT-BIH Record 200, male 68. Frequent multifocal PVCs (VEB class) with R-on-T phenomenon. Initiated on IV lidocaine. ICD implantation recommended."},
    {"id": "case_arr_002", "type": "case", "text": "Case ARR-002: MIT-BIH Record 108, female 45. Frequent PACs (SVEB class) with compensatory pauses. Asymptomatic. Advised Holter in 3 months. No treatment initiated."},

    # ══════════════════════════════════════════════════════════════════════════
    # PTB-XL superclass coverage — the classes core/pipeline.py queries with.
    # ══════════════════════════════════════════════════════════════════════════

    # ── NORM — Normal ECG ─────────────────────────────────────────────────────
    {"id": "gdl_norm_001", "superclass": "NORM", "type": "guideline", "text": "Normal ECG: normal sinus rhythm is defined by a rate of 60-100 beats per minute, a P wave preceding every QRS complex, P waves upright in leads I, II and aVF and inverted in aVR, a constant PR interval of 120-200 ms, and a QRS duration below 120 ms."},
    {"id": "gdl_norm_002", "superclass": "NORM", "type": "guideline", "text": "Normal ECG: reference intervals for the adult 12-lead electrocardiogram are PR 120-200 ms, QRS duration under 120 ms, and a rate-corrected QT interval (Bazett) at or below 450 ms in men and 460 ms in women. The normal frontal plane QRS axis lies between -30 and +90 degrees."},
    {"id": "gdl_norm_003", "superclass": "NORM", "type": "guideline", "text": "Normal ECG: normal precordial R-wave progression shows R-wave amplitude increasing steadily from V1 to V5 with the transition zone where R equals S falling in V3 or V4. Poor R-wave progression is a non-specific finding that may reflect lead placement, body habitus, prior anterior infarction, or left ventricular hypertrophy."},
    {"id": "gdl_norm_004", "superclass": "NORM", "type": "guideline", "text": "Normal ECG variant: respiratory sinus arrhythmia produces phasic beat-to-beat variation in the R-R interval that increases with inspiration and decreases with expiration. It is a normal finding, particularly in children and young adults, reflects vagal tone, and requires no treatment."},
    {"id": "gdl_norm_005", "superclass": "NORM", "type": "guideline", "text": "Normal ECG variant: benign early repolarization produces concave upward ST-segment elevation with J-point notching or slurring, most often in the precordial leads of young healthy adults. Unlike acute myocardial infarction it lacks reciprocal ST depression, evolves little over serial tracings, and is not accompanied by pathological Q waves."},
    {"id": "gdl_norm_006", "superclass": "NORM", "type": "guideline", "text": "Normal ECG: a normal resting electrocardiogram does not exclude coronary artery disease or acute coronary syndrome. A substantial proportion of patients presenting with acute coronary syndrome have a normal or non-diagnostic initial tracing, so serial ECGs and serial cardiac troponin measurement are required when symptoms suggest ischaemia."},
    {"id": "case_norm_001", "superclass": "NORM", "type": "case", "text": "Case NORM-001: female, 34, routine pre-operative assessment. Rate 72 bpm, sinus rhythm, PR 148 ms, QRS 88 ms, QTc 412 ms, frontal axis +40 degrees, normal R-wave progression, no ST or T-wave abnormality. Reported as a normal ECG; patient cleared for surgery without further cardiac testing."},
    {"id": "case_norm_002", "superclass": "NORM", "type": "case", "text": "Case NORM-002: male, 22, competitive athlete undergoing screening. Sinus bradycardia at 52 bpm with early repolarization in the precordial leads and voltage criteria met in isolation. Asymptomatic, no family history of sudden cardiac death, normal echocardiogram. Interpreted as benign athletic cardiac adaptation rather than pathology."},

    # ── MI — Myocardial Infarction ────────────────────────────────────────────
    {"id": "gdl_mi_001", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: the Fourth Universal Definition of Myocardial Infarction defines diagnostic ST elevation as new elevation at the J point in two contiguous leads of at least 1 mm in all leads other than V2-V3. In leads V2-V3 the threshold is 2 mm in men aged 40 or over, 2.5 mm in men under 40, and 1.5 mm in women."},
    {"id": "gdl_mi_002", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: pathological Q waves indicating prior infarction are defined as any Q wave of at least 0.03 seconds duration and greater than 0.1 mV depth, or a QS complex, present in two contiguous leads. Q waves in leads V2-V3 of at least 0.02 seconds are also considered abnormal."},
    {"id": "gdl_mi_003", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction localisation: ST elevation in V1-V4 indicates anterior or anteroseptal infarction, typically from left anterior descending artery occlusion. Elevation in II, III and aVF indicates inferior infarction from the right coronary or left circumflex artery. Elevation in I, aVL, V5 and V6 indicates lateral infarction."},
    {"id": "gdl_mi_004", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: reciprocal ST-segment depression in the leads opposite the infarct territory strongly supports acute coronary occlusion and helps distinguish true ST-elevation myocardial infarction from acute pericarditis and benign early repolarization, in which reciprocal change is absent."},
    {"id": "gdl_mi_005", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: posterior infarction may present with tall R waves and horizontal ST depression in V1-V3 rather than ST elevation. Posterior leads V7-V9 should be recorded; ST elevation of at least 0.5 mm in these leads confirms posterior involvement and warrants the same reperfusion strategy as anterior ST elevation."},
    {"id": "gdl_mi_006", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: in inferior ST-elevation infarction, right-sided lead V4R should be recorded to detect right ventricular involvement, indicated by ST elevation of at least 1 mm. These patients are preload dependent, so nitrates and other vasodilators should be avoided and volume loading used to maintain output."},
    {"id": "gdl_mi_007", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: Wellens syndrome describes deeply inverted or biphasic T waves in leads V2 and V3 recorded during a pain-free interval, and indicates critical stenosis of the proximal left anterior descending artery. It carries a high risk of extensive anterior infarction; patients require coronary angiography and stress testing should be avoided."},
    {"id": "gdl_mi_008", "superclass": "MI", "type": "guideline", "text": "Myocardial infarction: the Sgarbossa criteria allow infarction to be diagnosed in the presence of left bundle branch block or ventricular pacing. They comprise concordant ST elevation of at least 1 mm, concordant ST depression of at least 1 mm in V1-V3, and excessively discordant ST elevation relative to QRS amplitude."},
    {"id": "case_mi_001", "superclass": "MI", "type": "case", "text": "Case MI-001: male, 58, crushing central chest pain for 40 minutes with diaphoresis. ECG showed 3 mm ST elevation in V2-V4 with reciprocal ST depression in II, III and aVF. Diagnosed as acute anterior ST-elevation myocardial infarction; taken for primary percutaneous coronary intervention with a proximal left anterior descending occlusion stented."},
    {"id": "case_mi_002", "superclass": "MI", "type": "case", "text": "Case MI-002: female, 67, inferior chest discomfort and hypotension. ECG showed ST elevation in II, III and aVF with elevation in III exceeding that in II, and 1.5 mm ST elevation in V4R. Diagnosed as inferior ST-elevation myocardial infarction with right ventricular involvement; managed with fluid loading and avoidance of nitrates."},

    # ── STTC — ST/T-Wave Changes ──────────────────────────────────────────────
    {"id": "gdl_sttc_001", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: horizontal or downsloping ST-segment depression of at least 0.5 mm in two contiguous leads suggests subendocardial ischaemia. Upsloping ST depression is considerably less specific. The magnitude and number of leads involved correlate with the extent of ischaemic myocardium."},
    {"id": "gdl_sttc_002", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: new symmetric T-wave inversion of at least 1 mm in leads with a dominant R wave suggests myocardial ischaemia. Deep symmetric anterior T-wave inversion warrants urgent evaluation for critical proximal left anterior descending stenosis and, when accompanied by prolonged QT, for intracranial pathology."},
    {"id": "gdl_sttc_003", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: non-specific ST-T abnormalities are among the most frequently reported ECG findings and have low diagnostic specificity in isolation. Interpretation requires correlation with symptoms, comparison with prior tracings, and review of electrolytes and medications before ischaemia is inferred."},
    {"id": "gdl_sttc_004", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: the digoxin effect produces downsloping scooped or sagging ST depression with a shortened QT interval, most visible in leads with tall R waves. This pattern reflects therapeutic drug effect rather than toxicity or ischaemia and should not by itself prompt ischaemic workup."},
    {"id": "gdl_sttc_005", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: hypokalaemia produces ST-segment depression, T-wave flattening, and prominent U waves that may merge with the T wave and mimic QT prolongation. Severe hypokalaemia predisposes to torsades de pointes; potassium and magnesium should both be repleted."},
    {"id": "gdl_sttc_006", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: hyperkalaemia produces a characteristic progression beginning with tall peaked narrow-based T waves, followed by PR prolongation and loss of P waves, then progressive QRS widening culminating in a sine-wave pattern. Severe hyperkalaemia is a medical emergency requiring intravenous calcium for membrane stabilisation."},
    {"id": "gdl_sttc_007", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: acute pericarditis produces widespread concave ST elevation with PR-segment depression across most leads and PR elevation in aVR. The absence of reciprocal ST depression and of pathological Q waves distinguishes it from acute ST-elevation myocardial infarction."},
    {"id": "gdl_sttc_008", "superclass": "STTC", "type": "guideline", "text": "ST and T-wave changes: the left ventricular hypertrophy strain pattern produces ST depression with asymmetric T-wave inversion in the lateral leads I, aVL, V5 and V6. These are secondary repolarization changes driven by hypertrophy rather than primary ischaemia, and must be interpreted together with voltage criteria."},
    {"id": "case_sttc_001", "superclass": "STTC", "type": "case", "text": "Case STTC-001: male, 61, exertional chest tightness relieved by rest. ECG during symptoms showed 1.5 mm horizontal ST depression in V4-V6 that resolved when pain settled; troponin mildly elevated. Diagnosed as non-ST-elevation acute coronary syndrome and referred for coronary angiography."},

    # ── CD — Conduction Disturbance ───────────────────────────────────────────
    {"id": "gdl_cd_001", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: right bundle branch block is diagnosed by a QRS duration of at least 120 ms, an rsR prime or M-shaped complex in leads V1 and V2, and a wide slurred S wave in leads I, aVL and V6. Secondary ST depression and T-wave inversion in the right precordial leads are expected and should not be read as ischaemia."},
    {"id": "gdl_cd_002", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: left bundle branch block is diagnosed by a QRS duration of at least 120 ms, a broad notched or slurred R wave in leads I, aVL, V5 and V6, absence of the normal septal Q wave in the lateral leads, and a delayed intrinsicoid deflection. ST segments and T waves are discordant, directed opposite to the terminal QRS."},
    {"id": "gdl_cd_003", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: incomplete right bundle branch block shows an rsR prime pattern in lead V1 with a QRS duration between 110 and 120 ms. It is frequently a normal variant in young adults and does not by itself require investigation in an asymptomatic patient."},
    {"id": "gdl_cd_004", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: left anterior fascicular block is diagnosed by left axis deviation between -45 and -90 degrees, a qR complex in leads I and aVL, an rS complex in leads II, III and aVF, and a QRS duration below 120 ms. It is the most common intraventricular conduction abnormality."},
    {"id": "gdl_cd_005", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: left posterior fascicular block is diagnosed by right axis deviation between +90 and +180 degrees, an rS complex in leads I and aVL, a qR complex in leads II, III and aVF, and a QRS duration below 120 ms. Right ventricular hypertrophy and other causes of right axis deviation must be excluded first."},
    {"id": "gdl_cd_006", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: first-degree atrioventricular block is defined by a PR interval exceeding 200 ms with every P wave conducted. It is usually benign, but marked prolongation beyond 300 ms can impair atrioventricular synchrony and produce symptoms resembling pacemaker syndrome."},
    {"id": "gdl_cd_007", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: second-degree atrioventricular block Mobitz type I, or Wenckebach, shows progressive PR interval lengthening until a P wave fails to conduct, and the block is usually within the atrioventricular node and benign. Mobitz type II shows a constant PR interval with sudden failure of conduction, is usually infranodal, carries a high risk of progression to complete heart block, and generally warrants permanent pacing."},
    {"id": "gdl_cd_008", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: third-degree or complete atrioventricular block shows atrioventricular dissociation in which the atrial rate is faster than and independent of the escape rhythm. A narrow junctional escape at 40-60 beats per minute is more stable than a wide ventricular escape at 20-40. Symptomatic acquired complete heart block is a standard indication for permanent pacing."},
    {"id": "gdl_cd_009", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: bifascicular block describes right bundle branch block combined with either left anterior or left posterior fascicular block. When it occurs with syncope, intermittent complete heart block should be suspected and electrophysiological evaluation or ambulatory monitoring considered."},
    {"id": "gdl_cd_010", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: non-specific intraventricular conduction delay describes a QRS duration of 110 ms or more that meets neither right nor left bundle branch block morphology criteria. It is associated with underlying structural heart disease and merits echocardiographic assessment."},
    {"id": "gdl_cd_011", "superclass": "CD", "type": "guideline", "text": "Conduction disturbance: ventricular pre-excitation in Wolff-Parkinson-White pattern shows a short PR interval below 120 ms, a slurred delta wave at QRS onset, and a widened QRS. In pre-excited atrial fibrillation, atrioventricular nodal blocking agents are contraindicated because they can accelerate conduction down the accessory pathway."},
    {"id": "case_cd_001", "superclass": "CD", "type": "case", "text": "Case CD-001: male, 72, recurrent exertional syncope. ECG showed QRS 148 ms with an rsR prime complex in V1 and a frontal axis of -55 degrees, indicating right bundle branch block with left anterior fascicular block, that is bifascicular block. Electrophysiology study demonstrated infranodal conduction disease and a permanent pacemaker was implanted."},
    {"id": "case_cd_002", "superclass": "CD", "type": "case", "text": "Case CD-002: female, 68, progressive fatigue and presyncope. ECG showed a constant PR interval of 180 ms with intermittent non-conducted P waves in a 2:1 pattern and QRS 130 ms. Diagnosed as Mobitz type II second-degree atrioventricular block; permanent pacemaker implanted given the risk of progression to complete block."},

    # ── HYP — Hypertrophy ─────────────────────────────────────────────────────
    {"id": "gdl_hyp_001", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: the Sokolow-Lyon voltage criterion for left ventricular hypertrophy is satisfied when the sum of the S-wave depth in lead V1 and the tallest R wave in V5 or V6 exceeds 35 mm in adults over 35 years of age. Voltage criteria are specific but relatively insensitive for detecting increased left ventricular mass."},
    {"id": "gdl_hyp_002", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: the Cornell voltage criterion for left ventricular hypertrophy is satisfied when the sum of the R wave in lead aVL and the S wave in lead V3 exceeds 28 mm in men or 20 mm in women. It performs better than Sokolow-Lyon in several validation cohorts."},
    {"id": "gdl_hyp_003", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: the Romhilt-Estes point score diagnoses left ventricular hypertrophy from a weighted combination of QRS voltage, ST-T strain pattern, left atrial enlargement, left axis deviation, QRS duration and delayed intrinsicoid deflection. A total of 5 points is considered diagnostic and 4 points probable."},
    {"id": "gdl_hyp_004", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: left ventricular hypertrophy with a strain pattern shows increased QRS voltage accompanied by ST depression and asymmetric T-wave inversion in the lateral leads. The presence of strain identifies a group with greater left ventricular mass and worse cardiovascular prognosis than voltage criteria alone."},
    {"id": "gdl_hyp_005", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: right ventricular hypertrophy is suggested by an R to S ratio greater than 1 in lead V1, an R wave in V1 of at least 7 mm, right axis deviation beyond +90 degrees, and deep S waves in V5 and V6. Common causes include pulmonary hypertension, chronic lung disease and congenital heart disease."},
    {"id": "gdl_hyp_006", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: left atrial enlargement, historically termed P mitrale, produces a P-wave duration of at least 120 ms in lead II with a notched bifid contour, together with a terminal negative deflection in lead V1 at least 1 mm deep and 40 ms wide. It commonly accompanies mitral valve disease and left ventricular hypertrophy."},
    {"id": "gdl_hyp_007", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: right atrial enlargement, historically termed P pulmonale, produces tall peaked P waves of at least 2.5 mm amplitude in leads II, III and aVF. It is typically seen with chronic obstructive pulmonary disease, pulmonary hypertension and tricuspid valve disease."},
    {"id": "gdl_hyp_008", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: biventricular hypertrophy is difficult to diagnose electrocardiographically because opposing forces cancel. The Katz-Wachtel phenomenon, large equiphasic QRS complexes in the mid-precordial leads, is suggestive, particularly in congenital heart disease with volume overload of both ventricles."},
    {"id": "gdl_hyp_009", "superclass": "HYP", "type": "guideline", "text": "Hypertrophy: electrocardiographic voltage criteria for chamber hypertrophy are affected by body habitus, age, sex, ethnicity, lung disease and pericardial or pleural fluid. Echocardiography or cardiac magnetic resonance imaging remains the reference standard for quantifying chamber size and ventricular mass."},
    {"id": "case_hyp_001", "superclass": "HYP", "type": "case", "text": "Case HYP-001: male, 59, longstanding poorly controlled hypertension. ECG showed an S wave of 22 mm in V1 and an R wave of 26 mm in V5, summing to 48 mm and exceeding the Sokolow-Lyon threshold, with lateral ST depression and T-wave inversion. Diagnosed as left ventricular hypertrophy with strain; echocardiography confirmed an elevated left ventricular mass index."},
    {"id": "case_hyp_002", "superclass": "HYP", "type": "case", "text": "Case HYP-002: female, 44, progressive exertional dyspnoea. ECG showed an R to S ratio above 1 in V1, right axis deviation of +110 degrees, and peaked 3 mm P waves in lead II. Diagnosed as right ventricular hypertrophy with right atrial enlargement; echocardiography confirmed pulmonary arterial hypertension."},
]

def seed_database(force: bool = False):
    print("Initializing Milvus Database...")

    if force:
        # The corpus is versioned in this file, so a changed corpus needs the
        # collection rebuilt — inserting over the old ids would leave stale
        # documents behind and skew retrieval.
        print("--force: dropping existing collection before re-seeding…")
        drop_collection()

    collection = init_db()

    if collection is None:
        print("Failed to initialize Milvus.")
        return

    # Check if we already have data
    if collection.num_entities > 0:
        print(f"Collection already has {collection.num_entities} entities. Skipping seeding.")
        print("Run with --force to drop and re-seed (required after editing MOCK_KNOWLEDGE_BASE).")
        return

    # Use all-mpnet-base-v2 for 768-dim embeddings matching production Milvus schema
    print("Loading text embedding model (all-mpnet-base-v2, 768-dim)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Generating embeddings for clinical knowledge base...")
    texts = [item["text"] for item in MOCK_KNOWLEDGE_BASE]
    embeddings = model.encode(texts)
    
    # Prepare data array for insertion as per Milvus spec
    ids = [item["id"] for item in MOCK_KNOWLEDGE_BASE]
    doc_types = [item["type"] for item in MOCK_KNOWLEDGE_BASE]
    
    print("Inserting data into Milvus Vector DB...")
    # _VectorCollection.insert() takes four parallel lists and flushes internally
    collection.insert(ids, embeddings.tolist(), texts, doc_types)
    print(f"Seeding completed. Total entities in DB: {collection.num_entities}")

    # Per-superclass coverage, so a class silently dropping to zero is visible.
    counts = {}
    for item in MOCK_KNOWLEDGE_BASE:
        counts[item.get("superclass", "unclassified")] = counts.get(item.get("superclass", "unclassified"), 0) + 1
    print("Corpus coverage by PTB-XL superclass:")
    for cls in ("NORM", "MI", "STTC", "CD", "HYP", "unclassified"):
        if cls in counts:
            print(f"  {cls:14s} {counts[cls]:3d} documents")

if __name__ == "__main__":
    seed_database(force="--force" in sys.argv)
