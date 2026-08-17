// Auto-detect API base URL
const API_BASE = window.location.origin;

// ============================================================================
// Navigation — switches pages properly
// ============================================================================
document.querySelectorAll('.nav-item[data-page]').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active-page'));
        const targetPage = item.getAttribute('data-page');
        if (targetPage) document.getElementById(targetPage).classList.add('active-page');
        // Auto-load content when navigating to these pages
        if (targetPage === 'arrhythmia-page') { loadArrhythmiaEDA(); loadArrhythmiaSamples(); }
        if (targetPage === 'echonet-page') { loadEchoEDA(); startEchoAnimation(); loadEchoSampleData(); }
        if (targetPage === 'arthritis-page') loadArthritisSampleData();
    });
});

// ============================================================================
// Seeded RNG (mulberry32) — deterministic speckle noise across frames
// ============================================================================
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ============================================================================
// Echocardiogram Canvas Animation — synthetic pulsing LV + mitral valve
// ============================================================================
let echoAnimId = null;
let echoSpeckleCanvas = null;  // offscreen canvas with fixed speckle

function buildSpeckleCanvas(W, H) {
    const oc = document.createElement('canvas');
    oc.width = W; oc.height = H;
    const octx = oc.getContext('2d');
    const imgData = octx.createImageData(W, H);
    const rng = mulberry32(0xDEADBEEF);
    const cx = W / 2, cy = 8;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const dx = x - cx, dy = y - cy;
            const r = Math.sqrt(dx * dx + dy * dy);
            const ang = Math.abs(Math.atan2(dx, dy));
            const inCone = ang < Math.PI / 3.6 && r > 8 && r < H - 4;
            const idx = (y * W + x) * 4;
            if (inCone) {
                const n = Math.floor(rng() * 22 + 4);
                imgData.data[idx] = n; imgData.data[idx+1] = n;
                imgData.data[idx+2] = Math.floor(n * 0.85); imgData.data[idx+3] = 255;
            } else {
                imgData.data[idx+3] = 255; // all black, alpha=255
            }
        }
    }
    octx.putImageData(imgData, 0, 0);
    return oc;
}

function startEchoAnimation() {
    const canvas = document.getElementById('echoAnimCanvas');
    if (!canvas || echoAnimId) return;   // already running
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = 8;
    const coneAngle = Math.PI / 3.6;
    const coneR = H - 4;

    if (!echoSpeckleCanvas) echoSpeckleCanvas = buildSpeckleCanvas(W, H);

    let frame = 0;
    function draw() {
        const t = (frame % 72) / 72;          // cardiac cycle 0→1
        const phase = t * 2 * Math.PI;
        const systole = (1 - Math.cos(phase)) / 2;  // 0=diastole, 1=systole

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, W, H);

        // --- clip to ultrasound cone ---
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, coneR, Math.PI/2 - coneAngle, Math.PI/2 + coneAngle);
        ctx.closePath();
        ctx.clip();

        // Speckle background (drawImage respects clip)
        ctx.drawImage(echoSpeckleCanvas, 0, 0);

        // LV chamber geometry — shrinks at systole
        const lvCx = cx, lvCy = H * 0.52;
        const lvW = 22 + 14 * (1 - systole);   // wider at diastole
        const lvH = 30 + 18 * (1 - systole);

        // Pericardium (outer wall)
        ctx.beginPath();
        ctx.ellipse(lvCx, lvCy, lvW + 12, lvH + 8, 0, 0, 2 * Math.PI);
        const wallBright = Math.floor(90 + 55 * systole);
        ctx.fillStyle = `rgba(${wallBright}, ${Math.floor(wallBright*0.82)}, ${Math.floor(wallBright*0.42)}, 0.85)`;
        ctx.fill();

        // Myocardium inner wall
        ctx.beginPath();
        ctx.ellipse(lvCx, lvCy, lvW + 5, lvH + 2, 0, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(${Math.floor(wallBright*0.6)}, ${Math.floor(wallBright*0.5)}, ${Math.floor(wallBright*0.25)}, 0.9)`;
        ctx.fill();

        // Blood pool (dark)
        ctx.beginPath();
        ctx.ellipse(lvCx, lvCy, lvW, lvH, 0, 0, 2 * Math.PI);
        ctx.fillStyle = '#04040e';
        ctx.fill();

        // Papillary muscles
        const papB = Math.floor(80 + 50 * systole);
        [[-10, 8, 0.25], [10, 8, -0.25]].forEach(([ox, oy, rot]) => {
            ctx.beginPath();
            ctx.ellipse(lvCx + ox, lvCy + oy, 5, 3.5, rot, 0, 2*Math.PI);
            ctx.fillStyle = `rgba(${papB}, ${Math.floor(papB*0.7)}, 0, 0.85)`;
            ctx.fill();
        });

        // Mitral valve leaflets — open at diastole, close at systole
        const mvOpen = (1 - systole) * 14;
        const mvY = lvCy - lvH + 3;
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = `rgba(230, 210, 140, ${0.7 + 0.3 * systole})`;
        ctx.beginPath();
        ctx.moveTo(lvCx, mvY);
        ctx.quadraticCurveTo(lvCx - 4, mvY + 10, lvCx - mvOpen, mvY + 18);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(lvCx, mvY);
        ctx.quadraticCurveTo(lvCx + 4, mvY + 10, lvCx + mvOpen, mvY + 18);
        ctx.stroke();

        // Aortic root (top-right of LV)
        ctx.beginPath();
        ctx.ellipse(lvCx + lvW*0.5, lvCy - lvH*0.6, 7, 5, -0.4, 0, 2*Math.PI);
        ctx.strokeStyle = 'rgba(200, 180, 120, 0.5)';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.restore();  // release cone clip

        // Cone border lines
        ctx.save();
        ctx.strokeStyle = 'rgba(80, 80, 80, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - Math.sin(coneAngle)*coneR, cy + Math.cos(coneAngle)*coneR);
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.sin(coneAngle)*coneR, cy + Math.cos(coneAngle)*coneR);
        ctx.stroke();
        // Arc border
        ctx.beginPath();
        ctx.arc(cx, cy, coneR, Math.PI/2 - coneAngle, Math.PI/2 + coneAngle);
        ctx.stroke();
        ctx.restore();

        // HUD — EF readout (higher at diastole)
        const ef = Math.round(25 + 40 * (1 - systole));
        const efColor = ef < 40 ? '#fa4d56' : ef < 50 ? '#ff832b' : '#4ade80';
        ctx.fillStyle = efColor;
        ctx.font = 'bold 10px monospace';
        ctx.fillText(`EF: ${ef}%`, 4, 14);
        ctx.fillStyle = '#555';
        ctx.font = '8px monospace';
        ctx.fillText('SYNTH', W - 34, 14);

        frame++;
        echoAnimId = requestAnimationFrame(draw);
    }
    draw();
}

// ============================================================================
// EchoNet Sample Dataset Rows
// ============================================================================
let echoSampleLoaded = false;
async function loadEchoSampleData() {
    if (echoSampleLoaded) return;
    const container = document.getElementById('echoSampleData');
    if (!container) return;
    try {
        const res = await fetch(API_BASE + '/api/echonet/dataset/browse?page=0&page_size=8');
        if (!res.ok) throw new Error('browse failed');
        const data = await res.json();
        if (!data.dataset_loaded || data.total === 0) {
            container.innerHTML = `<div style="padding:1rem;font-size:0.83rem;color:var(--text-light);text-align:center">
                <i class="fa-solid fa-cloud-arrow-down" style="color:var(--primary)"></i>
                EchoNet-Dynamic not downloaded. <a href="https://stanfordaimi.azurewebsites.net/" target="_blank" style="color:var(--primary)">Download from Stanford AIMI</a> and place in <code>data/EchoNet-Dynamic/</code>.
            </div>`;
        } else {
            const rows = data.samples.map(s => `
                <tr>
                    <td><span class="patient-id">${s.filename}</span></td>
                    <td>${s.ef !== null ? s.ef.toFixed(1) + '%' : '—'}</td>
                    <td>${s.esv !== null ? s.esv.toFixed(0) : '—'} mL</td>
                    <td>${s.edv !== null ? s.edv.toFixed(0) : '—'} mL</td>
                    <td>${s.split}</td>
                    <td><button class="analyze-btn" style="padding:3px 8px;font-size:0.72rem" onclick="analyzeDatasetVideo(${s.index})"><i class="fa-solid fa-play"></i></button></td>
                </tr>`).join('');
            container.innerHTML = `
                <div style="padding:0.5rem 1rem;font-size:0.8rem;color:var(--text-light)">Showing 8 of ${data.total.toLocaleString()} videos</div>
                <table class="records-table">
                    <thead><tr><th>File</th><th>EF</th><th>ESV</th><th>EDV</th><th>Split</th><th>Analyze</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>`;
            echoSampleLoaded = true;
        }
    } catch (e) {
        container.innerHTML = `<div style="padding:1rem;font-size:0.83rem;color:var(--text-light);text-align:center">Dataset not available in demo mode.</div>`;
    }
}

// ============================================================================
// Arrhythmia Sample Waveforms (one per AAMI class)
// ============================================================================
let arrSamplesLoaded = false;
async function loadArrhythmiaSamples() {
    if (arrSamplesLoaded) return;
    const container = document.getElementById('arr-samples-container');
    if (!container) return;
    try {
        const res = await fetch(API_BASE + '/api/ecg-arrhythmia/samples?n=1');
        if (!res.ok) throw new Error('samples failed');
        const data = await res.json();  // {className: [[187 values], ...], ...}
        const classColors = {
            'Normal (N)': '#4ade80',
            'Supraventricular Ectopic Beat (SVEB)': '#60a5fa',
            'Ventricular Ectopic Beat (VEB)': '#f87171',
            'Fusion Beat (F)': '#fb923c',
            'Unknown Beat (Q)': '#a78bfa'
        };
        const entries = Object.entries(data);
        if (entries.length === 0) { container.innerHTML = '<div class="empty-state"><p>No sample data.</p></div>'; return; }

        container.innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:0.75rem;padding:1rem">
            ${entries.map(([cls, beats], i) => {
                const color = classColors[cls] || '#94a3b8';
                return `<div style="background:var(--card-bg);border:1px solid var(--border);border-radius:8px;padding:0.75rem">
                    <div style="font-size:0.75rem;font-weight:600;color:${color};margin-bottom:0.4rem">${cls}</div>
                    <canvas id="arr-wave-${i}" width="200" height="60" style="width:100%;display:block;background:#0d0d0d;border-radius:4px"></canvas>
                </div>`;
            }).join('')}
        </div>`;

        // Draw waveforms on canvases
        entries.forEach(([cls, beats], i) => {
            const canvas = document.getElementById(`arr-wave-${i}`);
            if (!canvas || !beats || beats.length === 0) return;
            const signal = beats[0];  // first sample for this class
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const color = classColors[cls] || '#94a3b8';
            const min = Math.min(...signal), max = Math.max(...signal);
            const range = max - min || 1;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            signal.forEach((v, j) => {
                const x = (j / (signal.length - 1)) * W;
                const y = H - ((v - min) / range) * (H - 6) - 3;
                j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
        });
        arrSamplesLoaded = true;
    } catch (e) {
        container.innerHTML = `<div class="empty-state" style="padding:1rem"><p>Could not load sample waveforms: ${e.message}</p></div>`;
    }
}

// ============================================================================
// Arthritis Sample Records (APD patients mini-table)
// ============================================================================
let arthritisSampleLoaded = false;
async function loadArthritisSampleData() {
    if (arthritisSampleLoaded) return;
    const container = document.getElementById('arthritis-sample-data');
    if (!container) return;
    try {
        const res = await fetch(API_BASE + '/api/patients');
        if (!res.ok) throw new Error('patients failed');
        const data = await res.json();
        const isApd = data.dataset_type === 'apd';
        const badge = document.getElementById('arthritis-sample-badge');
        if (badge) {
            badge.textContent = isApd
                ? `${data.dataset_name} · ${data.total} patients`
                : `${data.dataset_name} · ${data.total.toLocaleString()} records`;
        }
        // Train panel previously hardcoded "the APD Dataset" while NHANES was loaded.
        const trainDs = document.getElementById('arthritisTrainDataset');
        if (trainDs) {
            trainDs.textContent = `${data.dataset_name} (${data.total.toLocaleString()} records)`;
        }
        const rows = data.patients.slice(0, 6).map(p => {
            if (isApd) {
                const risk = (parseFloat(p.ra) > 14 || parseFloat(p.crp) > 6);
                return `<tr>
                    <td><span class="patient-id">${p.id}</span></td>
                    <td>${p.gender ?? '—'}</td><td>${p.age ?? '—'}</td>
                    <td>${p.hb ?? '—'}</td><td>${p.esr ?? '—'}</td><td>${p.crp ?? '—'}</td><td>${p.ra ?? '—'}</td><td>${p.uric_acid ?? '—'}</td>
                    <td><span class="status-pill ${risk ? 'status-risk' : 'status-normal'}">${risk ? 'High' : 'Normal'}</span></td>
                </tr>`;
            } else {
                return `<tr>
                    <td><span class="patient-id">${p.id}</span></td>
                    <td>${p.gender ?? '—'}</td><td>${p.age ?? '—'}</td>
                    <td>${p.bmi ?? '—'}</td><td>${p.systolic_bp ?? '—'}</td><td>${p.inflammation ?? '—'}</td><td>${p.smoking ?? '—'}</td>
                    <td><span class="status-pill ${p.arthritis_risk==1?'status-risk':'status-normal'}">${p.arthritis_risk==1?'Yes':'No'}</span></td>
                </tr>`;
            }
        }).join('');
        const thead = isApd
            ? '<tr><th>ID</th><th>Gender</th><th>Age</th><th>Hb</th><th>ESR</th><th>CRP</th><th>RA</th><th>Uric Acid</th><th>Risk</th></tr>'
            // Same correction as the Patient Records screen: this column is the
            // ground-truth NHANES diagnosis label, not a risk assessment.
            : '<tr><th>ID</th><th>Gender</th><th>Age</th><th>BMI</th><th>SBP</th>'
              + '<th title="Derived index = (gout + osteoporosis) / 2. Not a measured inflammatory marker.">Gout/Osteo.</th>'
              + '<th>Smoking</th>'
              + '<th title="NHANES MCQ160A — doctor-diagnosed arthritis. Ground-truth label, not a prediction.">Arthritis (dx)</th></tr>';
        container.innerHTML = `
            <div style="padding:0.5rem 1rem;font-size:0.8rem;color:var(--text-light)">Showing 6 of ${data.total} records — ${data.dataset_name}</div>
            <table class="records-table"><thead>${thead}</thead><tbody>${rows}</tbody></table>`;
        arthritisSampleLoaded = true;
    } catch (e) {
        container.innerHTML = `<div class="empty-state" style="padding:1rem"><p>Could not load sample records.</p></div>`;
    }
}

// Global state for PDF generation
let lastEcgData = null;
let lastPredictData = null;
let lastPatientInput = null;
let lastTrainMetrics = null;
let lastHDPredictData = null;
let lastHDPatientInput = null;
let lastHDTrainMetrics = null;
let lastArrTrainMetrics = null;

// ============================================================================
// PAGE 1: ECG Dashboard
// ============================================================================
const ctx = document.getElementById('ecgChart').getContext('2d');
const ecgChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array.from({ length: 100 }, (_, i) => i),
        datasets: [{
            label: 'Lead II (Simulated)',
            borderColor: '#0f62fe',
            borderWidth: 2,
            data: Array(100).fill(0),
            pointRadius: 0,
            tension: 0.4
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
            x: { display: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { display: false } },
            y: { display: true, min: -2, max: 3, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { display: false } }
        },
        plugins: { legend: { display: true, labels: { color: '#fff' } } }
    }
});

let timeCounter = 0;
function animateEcg() {
    const data = ecgChart.data.datasets[0].data;
    data.shift();
    let val = 0;
    const cycle = timeCounter % 100;
    if (cycle === 10) val = 0.5;
    else if (cycle === 12) val = -0.3;
    else if (cycle === 14) val = 2.5;
    else if (cycle === 16) val = -0.5;
    else if (cycle === 30) val = 0.7;
    else val = (Math.random() * 0.1) - 0.05;
    data.push(val);
    ecgChart.update();
    timeCounter++;
    requestAnimationFrame(animateEcg);
}
animateEcg();

document.getElementById('runAnalysisBtn').addEventListener('click', runDeepCardioAnalysis);

// Upload ECG button — opens file picker; falls back to demo if cancelled
const _ecgFileInput = document.getElementById('ecg-file-input');
document.getElementById('upload-ecg-btn').addEventListener('click', () => {
    _ecgFileInput.value = '';   // reset so same file can be re-selected
    _ecgFileInput.click();
});
_ecgFileInput.addEventListener('change', () => {
    if (_ecgFileInput.files.length > 0) {
        runDeepCardioUpload(_ecgFileInput.files[0]);
    }
});

async function runDeepCardioUpload(file) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const reportBox      = document.getElementById('reportContentBox');
    const metricsDiv     = document.getElementById('reportMetrics');
    const actionsDiv     = document.getElementById('reportActions');

    // Switch to ECG page
    document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active-page'));
    document.getElementById('ecg-page').classList.add('active-page');
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector('[data-page="ecg-page"]').classList.add('active');

    loadingOverlay.classList.remove('hidden');
    reportBox.innerHTML = '';
    metricsDiv.classList.add('hidden');
    actionsDiv.classList.add('hidden');

    const btn = document.getElementById('upload-ecg-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analysing…';

    try {
        const fd = new FormData();
        fd.append('file', file);
        const res = await fetch(API_BASE + '/api/analyze/upload', { method: 'POST', body: fd });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();

        const info = data.upload_info || {};
        reportBox.innerHTML = `
            <div style="padding:10px 0 12px;border-bottom:1px solid var(--border);margin-bottom:12px;font-size:0.82rem;color:var(--text-light);display:flex;gap:16px;flex-wrap:wrap">
                <span><i class="fa-solid fa-file-csv" style="color:var(--primary)"></i> <strong>${info.filename || file.name}</strong></span>
                <span><strong>Leads:</strong> ${info.leads_detected || 12}</span>
                <span><strong>Duration:</strong> ${info.duration_seconds || '?'}s @ ${info.sampling_rate_assumed || 500}Hz</span>
                <span><strong>Samples:</strong> ${(info.samples || 0).toLocaleString()}</span>
            </div>
            <pre class="report-text">${data.report}</pre>`;

        // RAG context
        const ragBox = document.getElementById('ragContextBox');
        if (data.retrieved_guidelines?.length) {
            ragBox.innerHTML = '';
            renderRagProvenance(data.retrieval, ragBox);
            ragBox.insertAdjacentHTML('beforeend', data.retrieved_guidelines.map((g, i) =>
                `<div class="rag-item"><span class="rag-num">${i+1}</span><p>${g}</p></div>`).join(''));
        }

        document.getElementById('valInferenceTime').textContent = data.inference_time_seconds + 's';
        document.getElementById('valBleu').textContent = data.bleu_score_approx?.toFixed(3) ?? '—';
        metricsDiv.classList.remove('hidden');
        actionsDiv.classList.remove('hidden');
        lastEcgData = data;
    } catch (err) {
        console.error(err);
        reportBox.innerHTML = `<div class="empty-state"><i class="fa-solid fa-triangle-exclamation" style="color:var(--secondary)"></i>
            <p>${err.message || 'Upload failed. Check file format (CSV: rows=samples, 12 columns; JSON: array of 12 arrays).'}</p>
            <p style="margin-top:8px;font-size:0.82rem">
                <a href="/api/analyze/sample-ecg" download style="color:var(--primary)"><i class="fa-solid fa-download"></i> Download sample CSV</a>
            </p></div>`;
    } finally {
        loadingOverlay.classList.add('hidden');
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-file-medical"></i> Upload ECG';
    }
}

// Shows where the RAG guidelines came from. A 'demo-fallback' result is canned
// placeholder text that has nothing to do with the ECG on screen, so it must be
// labelled as such — otherwise the panel reads like retrieved clinical evidence.
function renderRagProvenance(retrieval, ragBox) {
    if (!retrieval) return;
    const grounded = !!retrieval.grounded;
    const dx = retrieval.predicted_diagnosis;
    const banner = document.createElement('div');
    banner.className = `rag-provenance ${grounded ? 'grounded' : 'ungrounded'}`;
    banner.innerHTML = `
        <div class="rag-prov-head">
            <i class="fa-solid ${grounded ? 'fa-circle-check' : 'fa-triangle-exclamation'}"></i>
            <strong>${grounded ? 'Retrieved from vector database' : 'Placeholder guidelines — not retrieved'}</strong>
        </div>
        <div class="rag-prov-body">
            ${grounded && dx
                ? `Query: <em>${dx.name}</em> — predicted by the PTB-XL diagnostic head
                   (confidence ${(dx.confidence * 100).toFixed(1)}%), matched by Sentence-BERT similarity.`
                : (retrieval.note || '')}
        </div>`;
    ragBox.appendChild(banner);
}

async function runDeepCardioAnalysis() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const reportBox = document.getElementById('reportContentBox');
    const sourceBanner = document.getElementById('ecgSourceBanner');
    const ragBox = document.getElementById('ragContextBox');
    const metricsDiv = document.getElementById('reportMetrics');
    const actionsDiv = document.getElementById('reportActions');
    loadingOverlay.classList.remove('hidden');
    reportBox.innerHTML = '';
    if (sourceBanner) sourceBanner.innerHTML = '';
    try {
        const response = await fetch(API_BASE + '/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        if (!response.ok) throw new Error("API Request Failed");
        const data = await response.json();
        lastEcgData = data;
        ragBox.innerHTML = '';
        renderRagProvenance(data.retrieval, ragBox);
        const ragScores = data.retrieval?.scores || [];
        data.retrieved_guidelines.forEach((context, i) => {
            const div = document.createElement('div');
            div.className = 'rag-item';
            const score = ragScores[i] != null
                ? `<span class="rag-score">similarity ${ragScores[i].toFixed(4)}</span>` : '';
            div.innerHTML = `<strong>Source Knowledge:</strong>${score}<br/>${context}`;
            ragBox.appendChild(div);
        });

        if (sourceBanner && data.source) {
            const src = data.source;
            const isReal = !!src.real_signal;
            sourceBanner.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 0 14px;margin-bottom:12px;border-bottom:1px solid var(--border);font-size:0.85rem">
                    <span class="risk-badge ${isReal ? 'success' : 'risk-low'}" style="margin:0">
                        <i class="fa-solid ${isReal ? 'fa-heart-pulse' : 'fa-flask'}"></i>
                        <span>${isReal ? 'Real PTB-XL Record' : 'Synthetic Fallback'}${src.ecg_id != null ? ' · ECG #' + src.ecg_id : ''}</span>
                    </span>
                    <span style="color:var(--text-light)"><strong>Ground truth:</strong> ${src.ground_truth_label || src.ground_truth_superclass || '—'}
                        ${(src.all_superclasses || []).length > 1 ? ' (' + src.all_superclasses.join(', ') + ')' : ''}
                    </span>
                </div>`;
        }

        reportBox.innerHTML = `<p>${data.report || "No report generated."}</p>`;
        document.getElementById('valInferenceTime').textContent = `${data.inference_time_seconds}s`;
        document.getElementById('valBleu').textContent = data.bleu_score_approx;
        metricsDiv.classList.remove('hidden');
        actionsDiv.classList.remove('hidden');
    } catch (error) {
        console.error(error);
        reportBox.innerHTML = `<div class="empty-state"><i class="fa-solid fa-triangle-exclamation" style="color:var(--secondary)"></i><p>Error. Ensure backend is running.</p></div>`;
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

// ============================================================================
// PTB-XL Source Dataset Stats (ECG Dashboard)
// ============================================================================
const ptbxlEdaBtn = document.getElementById('loadPtbxlEdaBtn');
if (ptbxlEdaBtn) {
    ptbxlEdaBtn.addEventListener('click', async () => {
        const btn = ptbxlEdaBtn;
        btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
        try {
            const res = await fetch(API_BASE + '/api/ecg/ptbxl-eda');
            const data = await res.json();
            const dist = data.superclass_distribution || {};
            document.getElementById('ptbxlTotal').textContent = (data.total_records || 0).toLocaleString();
            document.getElementById('ptbxlNorm').textContent = dist.NORM ? dist.NORM.count.toLocaleString() : '—';
            document.getElementById('ptbxlMi').textContent = dist.MI ? dist.MI.count.toLocaleString() : '—';
            document.getElementById('ptbxlSttc').textContent = dist.STTC ? dist.STTC.count.toLocaleString() : '—';
            document.getElementById('ptbxlCdHyp').textContent = ((dist.CD?.count || 0) + (dist.HYP?.count || 0)).toLocaleString();
            btn.innerHTML = data.real_signals_available
                ? '<i class="fa-solid fa-check"></i> Real Signals Loaded'
                : '<i class="fa-solid fa-check"></i> Loaded (demo stats)';
        } catch (e) {
            console.error(e);
            btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
        }
        setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-bar"></i> Load Stats'; }, 2500);
    });
}

// ECG PDF export
document.getElementById('ecgDocPdfBtn').addEventListener('click', () => downloadEcgPdf('doctor'));
document.getElementById('ecgPatPdfBtn').addEventListener('click', () => downloadEcgPdf('patient'));

async function downloadEcgPdf(audience) {
    if (!lastEcgData) return alert('Run analysis first.');
    const btnId = audience === 'doctor' ? 'ecgDocPdfBtn' : 'ecgPatPdfBtn';
    const btn = document.getElementById(btnId);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
    try {
        const payload = { ...lastEcgData, audience };
        const res = await fetch(API_BASE + '/api/pdf/ecg', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('PDF generation failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `ECG_${audience === 'doctor' ? 'Clinical' : 'Patient'}_Report.pdf`; a.click();
        URL.revokeObjectURL(url);
    } catch (e) { console.error(e); alert('PDF generation failed.'); }
    btn.disabled = false;
    btn.innerHTML = originalText;
}

// ============================================================================
// PAGE 2: Arthritis EDA
// ============================================================================
let chartInstances = {};
document.getElementById('loadEdaBtn').addEventListener('click', () => loadEDAWithDatasetInfo());

async function loadEDA() {
    const btn = document.getElementById('loadEdaBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/arthritis/eda');
        if (!res.ok) throw new Error('EDA API failed');
        const data = await res.json();
        document.getElementById('totalSamples').textContent = data.total_samples;
        document.getElementById('totalFeatures').textContent = data.total_features;
        document.getElementById('maleCount').textContent = data.gender_distribution.male;
        document.getElementById('femaleCount').textContent = data.gender_distribution.female;
        document.getElementById('avgAge').textContent = data.age_stats.mean || '—';
        renderBarChart('inflammatoryChart', data.inflammatory_markers, '#fa4d56');
        renderBarChart('hematologyChart', data.hematology_markers, '#0f62fe');
        renderBarChart('biochemChart', data.biochemistry_markers, '#198038');
        renderMissingChart(data.missing_percentage);
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (err) {
        console.error(err);
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-pie"></i> Load EDA'; }, 2000);
}

function renderBarChart(canvasId, markers, color) {
    if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
    const labels = Object.keys(markers);
    const values = labels.map(k => markers[k].mean || 0);
    chartInstances[canvasId] = new Chart(document.getElementById(canvasId), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Mean Value', data: values, backgroundColor: color + '99', borderColor: color, borderWidth: 1, borderRadius: 6 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#eee' } }, x: { grid: { display: false } } } }
    });
}

function renderMissingChart(missingPct) {
    if (chartInstances['missingChart']) chartInstances['missingChart'].destroy();
    const labels = Object.keys(missingPct).filter(k => missingPct[k] > 0);
    const values = labels.map(k => missingPct[k]);
    chartInstances['missingChart'] = new Chart(document.getElementById('missingChart'), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Missing %', data: values, backgroundColor: '#ff832b99', borderColor: '#ff832b', borderWidth: 1, borderRadius: 6 }] },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, max: 100, grid: { color: '#eee' } }, y: { grid: { display: false } } } }
    });
}

// Train Model
document.getElementById('trainModelBtn').addEventListener('click', trainModel);

async function trainModel() {
    const btn = document.getElementById('trainModelBtn');
    const resultsDiv = document.getElementById('trainResults');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training MoE & Tabular BERT...';
    resultsDiv.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Training Mixture of Experts (MoE) & Bidirectional Encoder Representations from Transformers (BERT)...</p></div>';
    try {
        const res = await fetch(API_BASE + '/api/arthritis/train', { method: 'POST' });
        if (!res.ok) throw new Error('Train API failed');
        const data = await res.json();
        lastTrainMetrics = data;

        let featuresHtml = '<div class="empty-state" style="padding:1rem;color:var(--text-light)"><p>Feature importances not available for deep learning models.</p></div>';
        if (data.top_features && data.top_features.length > 0) {
            featuresHtml = data.top_features.map(f =>
                `<div class="feature-row"><span class="feat-name">${f.feature}</span><div class="feat-bar-bg"><div class="feat-bar" style="width:${(f.importance * 100 / data.top_features[0].importance).toFixed(0)}%"></div></div><span class="feat-val">${(f.importance * 100).toFixed(1)}%</span></div>`
            ).join('');
        }

        resultsDiv.innerHTML = `
            <div class="train-metrics-grid">
                <div class="train-metric"><span class="tm-value success">${(data.accuracy * 100).toFixed(1)}%</span><span class="tm-label">Test Accuracy</span></div>
                <div class="train-metric"><span class="tm-value">${(data.cv_mean_accuracy * 100).toFixed(1)}%</span><span class="tm-label">CV Accuracy (5-fold)</span></div>
                <div class="train-metric"><span class="tm-value">${data.auc_roc ? (data.auc_roc * 100).toFixed(1) + '%' : '—'}</span><span class="tm-label">AUC-ROC</span></div>
                <div class="train-metric"><span class="tm-value">${data.training_time_seconds}s</span><span class="tm-label">Training Time</span></div>
            </div>
            <div class="model-info-bar">
                <span class="model-badge"><i class="fa-solid fa-layer-group"></i> ${data.model_type}</span>
                <span class="model-badge feat-eng"><i class="fa-solid fa-gears"></i> ${data.total_features} features</span>
            </div>
            <h4 style="margin:1rem 0 0.5rem"><i class="fa-solid fa-ranking-star"></i> Top Feature Importances</h4>
            <div class="feature-importance">${featuresHtml}</div>`;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Trained!';
    } catch (err) {
        console.error(err);
        resultsDiv.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Training failed. Check logs.</p></div>';
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-play"></i> Train Model'; }, 3000);
}

// ============================================================================
// PAGE 3: Patient Predictor
// ============================================================================
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const resultDiv = document.getElementById('predictResult');
    resultDiv.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Predicting...</p></div>';

    const body = {};
    new FormData(form).forEach((val, key) => {
        if (val !== '') body[key] = parseFloat(val);
    });
    lastPatientInput = body;

    try {
        const res = await fetch(API_BASE + '/api/arthritis/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || 'Predict API failed');
        }
        const data = await res.json();
        lastPredictData = data;

        const riskClass = data.risk_level === 'HIGH' ? 'risk-high' : 'risk-low';
        const riskIcon = data.risk_level === 'HIGH' ? 'fa-exclamation-triangle' : 'fa-shield-check';

        resultDiv.innerHTML = `
            <div class="risk-badge ${riskClass}"><i class="fa-solid ${riskIcon}"></i><span>${data.risk_level} RISK</span></div>
            <div class="confidence-bar-container"><span>Confidence</span><div class="confidence-bar-bg"><div class="confidence-bar ${riskClass}" style="width:${(data.confidence * 100).toFixed(0)}%"></div></div><span>${(data.confidence * 100).toFixed(1)}%</span></div>
            <div class="prob-grid">
                <div class="prob-item low"><span class="prob-val">${(data.probabilities.low_risk * 100).toFixed(1)}%</span><span class="prob-label">Low Risk</span></div>
                <div class="prob-item high"><span class="prob-val">${(data.probabilities.high_risk * 100).toFixed(1)}%</span><span class="prob-label">High Risk</span></div>
            </div>
            <div class="predict-actions" style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 15px;">
                <button class="analyze-btn" id="arthritisDocPdfBtn" title="Clinical Report"><i class="fa-solid fa-user-doctor"></i> Clinical PDF</button>
                <button class="analyze-btn" id="arthritisPatPdfBtn" title="Patient Report"><i class="fa-solid fa-user"></i> Patient PDF</button>
            </div>`;
        // Bind PDF button
        document.getElementById('arthritisDocPdfBtn').addEventListener('click', () => downloadArthritisPdf('doctor'));
        document.getElementById('arthritisPatPdfBtn').addEventListener('click', () => downloadArthritisPdf('patient'));
    } catch (err) {
        console.error(err);
        resultDiv.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Prediction failed: ${err.message}. <br><small>Train the model first via the Arthritis Analysis page.</small></p></div>`;
    }
});

async function downloadArthritisPdf(audience) {
    if (!lastPredictData || !lastPatientInput) return alert('Run prediction first.');
    const btnId = audience === 'doctor' ? 'arthritisDocPdfBtn' : 'arthritisPatPdfBtn';
    const btn = document.getElementById(btnId);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF...';
    try {
        const res = await fetch(API_BASE + '/api/pdf/arthritis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_data: lastPatientInput,
                prediction_result: lastPredictData,
                train_metrics: lastTrainMetrics,
                audience: audience
            })
        });
        if (!res.ok) throw new Error('PDF generation failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `Arthritis_Risk_${audience === 'doctor' ? 'Clinical' : 'Patient'}_Report.pdf`; a.click();
        URL.revokeObjectURL(url);
    } catch (e) { console.error(e); alert('PDF generation failed.'); }
    btn.disabled = false;
    btn.innerHTML = originalText;
}

// ============================================================================
// PAGE 4: Patient Records
// ============================================================================
document.getElementById('loadRecordsBtn').addEventListener('click', loadRecords);

async function loadRecords() {
    const btn = document.getElementById('loadRecordsBtn');
    const container = document.getElementById('recordsTable');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/patients');
        if (!res.ok) throw new Error('Patients API failed');
        const data = await res.json();
        const isApd = data.dataset_type === 'apd';
        let rows, thead;
        if (isApd) {
            rows = data.patients.map(p => `
                <tr>
                    <td><span class="patient-id">${p.id}</span></td>
                    <td>${p.gender ?? '—'}</td>
                    <td>${p.age ?? '—'}</td>
                    <td>${p.hb ?? '—'}</td>
                    <td>${p.esr ?? '—'}</td>
                    <td>${p.crp ?? '—'}</td>
                    <td>${p.ra ?? '—'}</td>
                    <td>${p.uric_acid ?? '—'}</td>
                    <td><span class="status-pill ${getStatusClass(p)}"><i class="fa-solid fa-circle"></i> ${getStatus(p)}</span></td>
                </tr>`).join('');
            thead = '<tr><th>ID</th><th>Gender</th><th>Age</th><th>Hb</th><th>ESR</th><th>CRP</th><th>RA</th><th>Uric Acid</th><th>Status</th></tr>';
        } else {
            rows = data.patients.map(p => `
                <tr>
                    <td><span class="patient-id">${p.id}</span></td>
                    <td>${p.gender ?? '—'}</td>
                    <td>${p.age ?? '—'}</td>
                    <td>${p.bmi ?? '—'}</td>
                    <td>${p.systolic_bp ?? '—'}</td>
                    <td>${p.inflammation ?? '—'}</td>
                    <td>${p.smoking ?? '—'}</td>
                    <td><span class="status-pill ${p.arthritis_risk == 1 ? 'status-risk' : 'status-normal'}"><i class="fa-solid fa-circle"></i> ${p.arthritis_risk == 1 ? 'Yes' : 'No'}</span></td>
                </tr>`).join('');
            // "Risk" was wrong: arthritis_risk is NHANES MCQ160A ("has a doctor ever
            // told you that you have arthritis?") — the ground-truth TRAINING LABEL,
            // not a model output. Labelling it Risk/High Risk implied DeepCardio had
            // assessed these patients. It has not; this column is a recorded diagnosis.
            // Likewise InflammationProxy is (HasGout + HasOsteoporosis)/2 — a comorbidity
            // index taking only 0.0/0.5/1.0, NOT a measured marker like CRP or ESR.
            thead = '<tr><th>ID</th><th>Gender</th><th>Age (yrs)</th><th>BMI</th><th>Systolic BP</th>'
                  + '<th title="Derived index = (gout + osteoporosis) / 2. Not a measured inflammatory marker.">Gout/Osteo. index</th>'
                  + '<th>Smoking</th>'
                  + '<th title="NHANES MCQ160A — doctor-diagnosed arthritis. Ground-truth label, not a prediction.">Arthritis (diagnosed)</th></tr>';
        }
        container.innerHTML = `
            <div class="records-header"><span class="records-total"><i class="fa-solid fa-database"></i> Showing ${data.patients.length} of ${data.total.toLocaleString()} records &mdash; ${data.dataset_name}</span></div>
            <table class="records-table">
                <thead>${thead}</thead>
                <tbody>${rows}</tbody>
            </table>`;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (err) {
        console.error(err);
        container.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Failed to load records.</p></div>';
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-download"></i> Load Records'; }, 2000);
}

function getStatus(p) {
    if (p.arthritis_risk !== undefined) return p.arthritis_risk == 1 ? 'High Risk' : 'Normal';
    const ra = parseFloat(p.ra);
    const crp = parseFloat(p.crp);
    if ((ra && ra > 14) || (crp && crp > 6)) return 'High Risk';
    return 'Normal';
}
function getStatusClass(p) {
    return getStatus(p) === 'High Risk' ? 'status-risk' : 'status-normal';
}

// ============================================================================
// PAGE 6: EchoNet-Dynamic Video Analysis
// ============================================================================
let echoFileSelected = null;

// EDA stats loader
document.getElementById('loadEchoEdaBtn').addEventListener('click', loadEchoEDA);

let echoEdaLoaded = false;
async function loadEchoEDA() {
    startEchoAnimation();   // start canvas animation whenever this page loads
    if (echoEdaLoaded) return;
    const btn = document.getElementById('loadEchoEdaBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/echonet/eda');
        if (!res.ok) throw new Error('EchoNet EDA failed');
        const data = await res.json();

        document.getElementById('echoTotalVideos').textContent = data.total_videos?.toLocaleString() || '—';
        document.getElementById('echoMeanEF').textContent = data.ef_stats?.mean ? data.ef_stats.mean + '%' : '—';

        // The loader supplies a `note` saying these are PUBLISHED reference figures for a
        // dataset that isn't downloaded. It was being dropped, so 10,030 videos / 55.6%
        // mean EF / the EF doughnut all read as local data. Surface it.
        if (data.note) {
            const host = document.getElementById('echoTotalVideos').closest('.card') || document.body;
            let noteEl = document.getElementById('echoEdaNote');
            if (!noteEl) {
                noteEl = document.createElement('div');
                noteEl.id = 'echoEdaNote';
                noteEl.style.cssText = 'margin:0.5rem 1rem;padding:0.5rem 0.75rem;border-left:3px solid #ff832b;'
                                     + 'background:rgba(255,131,43,0.08);font-size:0.82rem;line-height:1.4';
                host.appendChild(noteEl);
            }
            noteEl.innerHTML = `<strong>Reference statistics — not local data.</strong> ${data.note}`;
        }

        const cats = data.ef_categories || {};
        document.getElementById('echoHFrEF').textContent = cats['HFrEF (EF<40)'] || '—';
        document.getElementById('echoHFmrEF').textContent = cats['HFmrEF (40-50)'] || '—';
        document.getElementById('echoNormal').textContent = cats['Normal (EF>=50)'] || '—';

        // EF distribution pie chart
        if (chartInstances['efDistChart']) chartInstances['efDistChart'].destroy();
        chartInstances['efDistChart'] = new Chart(document.getElementById('efDistChart'), {
            type: 'doughnut',
            data: {
                labels: Object.keys(cats),
                datasets: [{
                    data: Object.values(cats),
                    backgroundColor: ['#fa4d56', '#ff832b', '#198038'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { padding: 16 } }
                }
            }
        });

        echoEdaLoaded = true;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (err) {
        console.error(err);
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-bar"></i> Load Stats'; }, 2000);
}

// File upload handling
const echoUploadArea = document.getElementById('echoUploadArea');
const echoFileInput = document.getElementById('echoFileInput');
const echoUploadBtn = document.getElementById('echoUploadBtn');

echoUploadArea.addEventListener('click', () => echoFileInput.click());
echoUploadArea.addEventListener('dragover', (e) => { e.preventDefault(); echoUploadArea.style.borderColor = 'var(--primary)'; });
echoUploadArea.addEventListener('dragleave', () => { echoUploadArea.style.borderColor = 'var(--border)'; });
echoUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    echoUploadArea.style.borderColor = 'var(--border)';
    if (e.dataTransfer.files.length > 0) {
        echoFileSelected = e.dataTransfer.files[0];
        echoUploadArea.querySelector('p').textContent = `Selected: ${echoFileSelected.name}`;
        echoUploadBtn.disabled = false;
    }
});
echoFileInput.addEventListener('change', () => {
    if (echoFileInput.files.length > 0) {
        echoFileSelected = echoFileInput.files[0];
        echoUploadArea.querySelector('p').textContent = `Selected: ${echoFileSelected.name}`;
        echoUploadBtn.disabled = false;
    }
});

// Demo analysis
document.getElementById('echoDemoBtn').addEventListener('click', runEchoDemo);

async function runEchoDemo() {
    const overlay = document.getElementById('echoLoadingOverlay');
    const reportBox = document.getElementById('echoReportBox');
    const ragBox = document.getElementById('echoRagBox');
    const metrics = document.getElementById('echoMetrics');
    overlay.classList.remove('hidden');
    reportBox.innerHTML = '';
    ragBox.innerHTML = '';
    try {
        const res = await fetch(API_BASE + '/api/echonet/analyze/demo', { method: 'POST' });
        if (!res.ok) throw new Error('Demo analysis failed');
        const data = await res.json();
        displayEchoResult(data, reportBox, ragBox, metrics);
    } catch (err) {
        console.error(err);
        reportBox.innerHTML = '<div class="empty-state"><i class="fa-solid fa-triangle-exclamation" style="color:var(--secondary)"></i><p>Error. Ensure backend is running.</p></div>';
    } finally {
        overlay.classList.add('hidden');
    }
}

// Upload analysis
echoUploadBtn.addEventListener('click', runEchoUpload);

async function runEchoUpload() {
    if (!echoFileSelected) return;
    const overlay = document.getElementById('echoLoadingOverlay');
    const reportBox = document.getElementById('echoReportBox');
    const ragBox = document.getElementById('echoRagBox');
    const metrics = document.getElementById('echoMetrics');
    overlay.classList.remove('hidden');
    reportBox.innerHTML = '';
    ragBox.innerHTML = '';
    try {
        const formData = new FormData();
        formData.append('video', echoFileSelected);
        const res = await fetch(API_BASE + '/api/echonet/analyze/upload', { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Upload analysis failed');
        const data = await res.json();
        displayEchoResult(data, reportBox, ragBox, metrics);
    } catch (err) {
        console.error(err);
        reportBox.innerHTML = '<div class="empty-state"><i class="fa-solid fa-triangle-exclamation" style="color:var(--secondary)"></i><p>Upload analysis failed. Check video format.</p></div>';
    } finally {
        overlay.classList.add('hidden');
    }
}

function displayEchoResult(data, reportBox, ragBox, metricsDiv) {
    // Report
    const formatted = (data.report || '').replace(/\n/g, '<br>');
    reportBox.innerHTML = `<pre style="white-space:pre-wrap;font-family:'Inter',sans-serif;font-size:0.9rem;line-height:1.6">${data.report || 'No report generated.'}</pre>`;

    // RAG contexts
    ragBox.innerHTML = '';
    (data.retrieved_guidelines || []).forEach(ctx => {
        const div = document.createElement('div');
        div.className = 'rag-item';
        div.innerHTML = `<strong>Clinical Guideline:</strong><br/>${ctx}`;
        ragBox.appendChild(div);
    });

    // Metrics
    document.getElementById('echoInferenceTime').textContent = `${data.inference_time_seconds}s`;

    // An untrained model reports NOTHING. Previously this rendered a deterministic
    // "0.0%" in a red HFrEF tile — an ejection fraction incompatible with life —
    // because the backend emitted a number from weights that were never trained.
    const efEl = document.getElementById('echoEFValue');
    const catEl = document.getElementById('echoEFCategory');
    catEl.className = 'value';
    if (data.model_trained === false || data.ef_predicted === null || data.ef_predicted === undefined) {
        efEl.textContent = 'Not available';
        efEl.title = data.warning || 'Model untrained — no EF can be produced.';
        catEl.textContent = 'Model untrained';
    } else {
        efEl.textContent = `${data.ef_predicted}%`;
        efEl.title = '';
        catEl.textContent = data.ef_category;
        if (data.ef_category === 'HFrEF') catEl.classList.add('danger');
        else if (data.ef_category === 'Normal') catEl.classList.add('success');
    }

    // "Demo (synthetic)" described the INPUT only, so an untrained model read as a
    // real one fed fake video. State the model's status too.
    const modeText = data.mode === 'demo' ? 'Demo (synthetic video)' : 'Real video';
    document.getElementById('echoMode').textContent =
        data.model_trained === false ? `${modeText} · UNTRAINED model` : modeText;
    metricsDiv.classList.remove('hidden');
}

// Dataset browser
document.getElementById('loadEchoBrowseBtn').addEventListener('click', loadEchoBrowse);

async function loadEchoBrowse() {
    const btn = document.getElementById('loadEchoBrowseBtn');
    const container = document.getElementById('echoBrowseContent');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/echonet/dataset/browse?page=0&page_size=20');
        if (!res.ok) throw new Error('Browse API failed');
        const data = await res.json();

        if (!data.dataset_loaded || data.total === 0) {
            container.innerHTML = `<div class="empty-state" style="padding:2rem">
                <i class="fa-solid fa-cloud-arrow-down" style="font-size:2rem;color:var(--primary)"></i>
                <p>EchoNet-Dynamic dataset not downloaded yet.</p>
                <p style="font-size:0.85rem;color:var(--text-light)">Download it from <a href="https://stanfordaimi.azurewebsites.net/" target="_blank">Stanford AIMI</a> and place the files in <code>data/EchoNet-Dynamic/</code></p>
            </div>`;
        } else {
            let rows = data.samples.map(s => `
                <tr>
                    <td>${s.index}</td>
                    <td><span class="patient-id">${s.filename}</span></td>
                    <td>${s.ef !== null ? s.ef.toFixed(1) + '%' : '—'}</td>
                    <td>${s.esv !== null ? s.esv.toFixed(1) : '—'}</td>
                    <td>${s.edv !== null ? s.edv.toFixed(1) : '—'}</td>
                    <td>${s.split}</td>
                    <td><button class="analyze-btn" style="padding:4px 10px;font-size:0.75rem" onclick="analyzeDatasetVideo(${s.index})"><i class="fa-solid fa-play"></i> Analyze</button></td>
                </tr>`).join('');

            container.innerHTML = `
                <div class="records-header"><span class="records-total"><i class="fa-solid fa-film"></i> Total: ${data.total.toLocaleString()} echocardiogram videos</span></div>
                <table class="records-table">
                    <thead><tr><th>#</th><th>Filename</th><th>EF</th><th>ESV</th><th>EDV</th><th>Split</th><th>Action</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
                <div style="padding:0.75rem;font-size:0.85rem;color:var(--text-light)">Showing page 1 of ${data.total_pages} (${data.page_size} per page)</div>`;
        }
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (err) {
        console.error(err);
        container.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Failed to browse dataset. Ensure backend is running.</p></div>';
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-folder-open"></i> Browse Videos'; }, 2000);
}

async function analyzeDatasetVideo(index) {
    const overlay = document.getElementById('echoLoadingOverlay');
    const reportBox = document.getElementById('echoReportBox');
    const ragBox = document.getElementById('echoRagBox');
    const metrics = document.getElementById('echoMetrics');
    overlay.classList.remove('hidden');
    reportBox.innerHTML = '';
    ragBox.innerHTML = '';
    try {
        const res = await fetch(API_BASE + `/api/echonet/analyze/dataset/${index}`, { method: 'POST' });
        if (!res.ok) throw new Error('Dataset analysis failed');
        const data = await res.json();
        displayEchoResult(data, reportBox, ragBox, metrics);
        // Scroll to report
        document.getElementById('echoReportBox').scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        console.error(err);
        reportBox.innerHTML = '<div class="empty-state"><i class="fa-solid fa-triangle-exclamation" style="color:var(--secondary)"></i><p>Dataset analysis failed.</p></div>';
    } finally {
        overlay.classList.add('hidden');
    }
}

// ============================================================================
// PAGE 7: ECG Image Classification
// ============================================================================
let ecgImgFile = null;

document.getElementById('loadEcgImgEdaBtn').addEventListener('click', async () => {
    const btn = document.getElementById('loadEcgImgEdaBtn');
    btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/ecg-images/eda');
        const data = await res.json();
        document.getElementById('ecgImgTotal').textContent = (data.total_images || 0).toLocaleString();
        const dist = data.class_distribution || {};
        document.getElementById('ecgImgN').textContent = dist.N ? dist.N.count.toLocaleString() : '—';
        document.getElementById('ecgImgS').textContent = dist.S ? dist.S.count.toLocaleString() : '—';
        document.getElementById('ecgImgV').textContent = dist.V ? dist.V.count.toLocaleString() : '—';
        document.getElementById('ecgImgFQ').textContent = ((dist.F?.count || 0) + (dist.Q?.count || 0)).toLocaleString();

        if (chartInstances['ecgImgDistChart']) chartInstances['ecgImgDistChart'].destroy();
        const labels = Object.keys(dist);
        chartInstances['ecgImgDistChart'] = new Chart(document.getElementById('ecgImgDistChart'), {
            type: 'doughnut',
            data: { labels: labels.map(l => `${l} (${dist[l].name})`), datasets: [{ data: labels.map(l => dist[l].count), backgroundColor: ['#198038','#0f62fe','#fa4d56','#ff832b','#8a3ffc'], borderWidth: 2, borderColor: '#fff' }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
        });
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (e) { console.error(e); btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error'; }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-bar"></i> Load Stats'; }, 2000);
});

const ecgImgUploadArea = document.getElementById('ecgImgUploadArea');
const ecgImgFileInput = document.getElementById('ecgImgFileInput');
ecgImgUploadArea.addEventListener('click', () => ecgImgFileInput.click());
ecgImgFileInput.addEventListener('change', () => {
    if (ecgImgFileInput.files.length > 0) { ecgImgFile = ecgImgFileInput.files[0]; ecgImgUploadArea.querySelector('p').textContent = `Selected: ${ecgImgFile.name}`; document.getElementById('ecgImgUploadBtn').disabled = false; }
});

document.getElementById('ecgImgDemoBtn').addEventListener('click', async () => {
    const box = document.getElementById('ecgImgResult');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Classifying...</p></div>';
    try {
        const res = await fetch(API_BASE + '/api/ecg-images/classify/demo', { method: 'POST' });
        const data = await res.json();
        displayEcgImgResult(data, box);
    } catch (e) { box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Error.</p></div>'; }
});

document.getElementById('ecgImgUploadBtn').addEventListener('click', async () => {
    if (!ecgImgFile) return;
    const box = document.getElementById('ecgImgResult');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Classifying...</p></div>';
    try {
        const fd = new FormData(); fd.append('image', ecgImgFile);
        const res = await fetch(API_BASE + '/api/ecg-images/classify/upload', { method: 'POST', body: fd });
        const data = await res.json();
        displayEcgImgResult(data, box);
    } catch (e) { box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Classification failed.</p></div>'; }
});

function displayEcgImgResult(data, box) {
    const riskColor = data.class_label === 'N' ? 'success' : data.class_label === 'V' ? 'risk-high' : 'risk-low';
    const probBars = Object.entries(data.probabilities || {}).map(([k,v]) =>
        `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:30px;font-weight:600">${k}</span><div style="flex:1;background:#eee;border-radius:4px;height:20px"><div style="width:${(v*100).toFixed(0)}%;height:100%;background:${k==='N'?'#198038':k==='V'?'#fa4d56':'#0f62fe'};border-radius:4px"></div></div><span style="width:50px;text-align:right">${(v*100).toFixed(1)}%</span></div>`
    ).join('');
    box.innerHTML = `
        <div class="risk-badge ${riskColor}" style="margin-bottom:1rem"><i class="fa-solid fa-${data.class_label === 'N' ? 'check' : 'exclamation-triangle'}"></i><span>${data.class_name} (${data.class_label})</span></div>
        <p><strong>Description:</strong> ${data.description}</p>
        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
        <h4 style="margin:1rem 0 0.5rem">Class Probabilities</h4>
        ${probBars}
        <h4 style="margin:1rem 0 0.5rem"><i class="fa-solid fa-book-open-reader"></i> Clinical Guidelines</h4>
        ${(data.retrieved_guidelines || []).map(g => `<div class="rag-item">${g}</div>`).join('')}`;
}

// ============================================================================
// PAGE 8: Heart Sound Analysis
// ============================================================================
let hsFile = null;

document.getElementById('loadHSEdaBtn').addEventListener('click', async () => {
    const btn = document.getElementById('loadHSEdaBtn');
    btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/heart-sound/eda');
        const data = await res.json();
        document.getElementById('hsSubjects').textContent = (data.total_subjects || 0).toLocaleString();
        document.getElementById('hsRecordings').textContent = (data.total_recordings || 0).toLocaleString();
        const md = data.murmur_distribution || {};
        document.getElementById('hsAbsent').textContent = (md.Absent || 0).toLocaleString();
        document.getElementById('hsPresent').textContent = (md.Present || 0).toLocaleString();
        document.getElementById('hsUnknown').textContent = (md.Unknown || 0).toLocaleString();

        if (chartInstances['hsDistChart']) chartInstances['hsDistChart'].destroy();
        chartInstances['hsDistChart'] = new Chart(document.getElementById('hsDistChart'), {
            type: 'doughnut',
            data: { labels: Object.keys(md), datasets: [{ data: Object.values(md), backgroundColor: ['#198038','#fa4d56','#ff832b'], borderWidth: 2, borderColor: '#fff' }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
        });
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (e) { console.error(e); btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error'; }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-bar"></i> Load Stats'; }, 2000);
});

const hsUploadArea = document.getElementById('hsUploadArea');
const hsFileInput = document.getElementById('hsFileInput');
hsUploadArea.addEventListener('click', () => hsFileInput.click());
hsFileInput.addEventListener('change', () => {
    if (hsFileInput.files.length > 0) { hsFile = hsFileInput.files[0]; hsUploadArea.querySelector('p').textContent = `Selected: ${hsFile.name}`; document.getElementById('hsUploadBtn').disabled = false; }
});

document.getElementById('hsDemoBtn').addEventListener('click', async () => {
    const box = document.getElementById('hsResultBox');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Analyzing heart sound...</p></div>';
    try {
        const res = await fetch(API_BASE + '/api/heart-sound/analyze/demo', { method: 'POST' });
        const data = await res.json();
        displayHSResult(data, box);
    } catch (e) { box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Error.</p></div>'; }
});

document.getElementById('hsUploadBtn').addEventListener('click', async () => {
    if (!hsFile) return;
    const box = document.getElementById('hsResultBox');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Analyzing...</p></div>';
    try {
        const fd = new FormData(); fd.append('audio', hsFile);
        const res = await fetch(API_BASE + '/api/heart-sound/analyze/upload', { method: 'POST', body: fd });
        const data = await res.json();
        displayHSResult(data, box);
    } catch (e) { box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Analysis failed.</p></div>'; }
});

function displayHSResult(data, box) {
    const murmurClass = data.murmur_class || 'Unknown';
    const color = murmurClass === 'Present' ? 'risk-high' : murmurClass === 'Absent' ? 'risk-low' : '';
    const icon = murmurClass === 'Present' ? 'fa-heart-pulse' : murmurClass === 'Absent' ? 'fa-shield-check' : 'fa-question';
    const probBars = Object.entries(data.probabilities || {}).map(([k,v]) =>
        `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:80px;font-weight:600">${k}</span><div style="flex:1;background:#eee;border-radius:4px;height:20px"><div style="width:${(v*100).toFixed(0)}%;height:100%;background:${k==='Present'?'#fa4d56':k==='Absent'?'#198038':'#ff832b'};border-radius:4px"></div></div><span style="width:50px;text-align:right">${(v*100).toFixed(1)}%</span></div>`
    ).join('');
    const pi = data.patient_info || {};
    box.innerHTML = `
        <div class="risk-badge ${color}" style="margin-bottom:1rem"><i class="fa-solid ${icon}"></i><span>Murmur: ${murmurClass}</span></div>
        <div class="report-metrics"><div class="metric"><span class="label">Confidence:</span><span class="value">${(data.confidence * 100).toFixed(1)}%</span></div><div class="metric"><span class="label">Inference:</span><span class="value success">${data.inference_time_seconds}s</span></div><div class="metric"><span class="label">Mode:</span><span class="value">${data.mode}</span></div></div>
        ${pi.patient_id ? `<p style="margin-top:0.5rem"><strong>Patient:</strong> ${pi.patient_id} | <strong>Location:</strong> ${pi.location} | <strong>Age:</strong> ${pi.age} | <strong>Sex:</strong> ${pi.sex}</p>` : ''}
        <h4 style="margin:1rem 0 0.5rem">Murmur Probabilities</h4>
        ${probBars}
        <h4 style="margin:1rem 0 0.5rem"><i class="fa-solid fa-book-open-reader"></i> Clinical Guidelines</h4>
        ${(data.retrieved_guidelines || []).map(g => `<div class="rag-item">${g}</div>`).join('')}`;
}

// ============================================================================
// PAGE 9: VFDB Ventricular Arrhythmia
// ============================================================================
document.getElementById('loadVfdbEdaBtn').addEventListener('click', async () => {
    const btn = document.getElementById('loadVfdbEdaBtn');
    btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/vfdb/eda');
        const data = await res.json();
        document.getElementById('vfdbRecords').textContent = data.total_records || '—';
        document.getElementById('vfdbDangerous').textContent = data.total_dangerous_events || '—';

        if (chartInstances['vfdbDistChart']) chartInstances['vfdbDistChart'].destroy();
        const rd = data.rhythm_distribution || {};
        chartInstances['vfdbDistChart'] = new Chart(document.getElementById('vfdbDistChart'), {
            type: 'bar',
            data: { labels: Object.keys(rd), datasets: [{ label: 'Count', data: Object.values(rd), backgroundColor: Object.keys(rd).map(k => ['VT','VF','VFL','VFIB','ASYS','HGEA','VER'].includes(k) ? '#fa4d56' : '#0f62fe'), borderRadius: 6 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
        });
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (e) { console.error(e); btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error'; }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-bar"></i> Load Stats'; }, 2000);
});

document.getElementById('vfdbDemoBtn').addEventListener('click', async () => {
    const box = document.getElementById('vfdbResultBox');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Detecting arrhythmias...</p></div>';
    try {
        const res = await fetch(API_BASE + '/api/vfdb/analyze/demo', { method: 'POST' });
        const data = await res.json();
        displayVfdbResult(data, box);
    } catch (e) { box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Error.</p></div>'; }
});

document.getElementById('loadVfdbBrowseBtn').addEventListener('click', async () => {
    const btn = document.getElementById('loadVfdbBrowseBtn');
    const container = document.getElementById('vfdbBrowseContent');
    btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/vfdb/browse');
        const data = await res.json();
        if (!data.dataset_loaded) {
            container.innerHTML = '<div class="empty-state" style="padding:2rem"><i class="fa-solid fa-cloud-arrow-down" style="font-size:2rem;color:var(--primary)"></i><p>VFDB dataset not downloaded. Get it from <a href="https://physionet.org/content/vfdb/1.0.0/" target="_blank">PhysioNet</a></p></div>';
        } else {
            let rows = (data.records || []).map(r => `<tr><td>${r.record_id}</td><td>${r.n_signals}</td><td>${(r.leads||[]).join(', ')}</td><td>${r.fs} Hz</td><td>${r.dangerous_events}</td><td>${(r.rhythm_types||[]).join(', ')}</td><td><button class="analyze-btn" style="padding:4px 10px;font-size:0.75rem" onclick="analyzeVfdbRecord(${r.index})"><i class="fa-solid fa-play"></i></button></td></tr>`).join('');
            container.innerHTML = `<table class="records-table"><thead><tr><th>Record</th><th>Signals</th><th>Leads</th><th>Fs</th><th>Dangerous</th><th>Rhythms</th><th>Analyze</th></tr></thead><tbody>${rows}</tbody></table>`;
        }
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (e) { container.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Failed.</p></div>'; btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error'; }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-folder-open"></i> Browse Records'; }, 2000);
});

async function analyzeVfdbRecord(index) {
    const box = document.getElementById('vfdbResultBox');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Analyzing record...</p></div>';
    try {
        const res = await fetch(API_BASE + `/api/vfdb/analyze/${index}?start_sec=0&duration=10`, { method: 'POST' });
        const data = await res.json();
        displayVfdbResult(data, box);
        box.scrollIntoView({ behavior: 'smooth' });
    } catch (e) { box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Analysis failed.</p></div>'; }
}

function displayVfdbResult(data, box) {
    const alert = data.alert_level || 'NORMAL';
    const alertColors = { CRITICAL: '#fa4d56', WARNING: '#ff832b', NORMAL: '#198038' };
    const alertIcons = { CRITICAL: 'fa-skull-crossbones', WARNING: 'fa-exclamation-triangle', NORMAL: 'fa-shield-check' };
    const probBars = Object.entries(data.rhythm_probabilities || {}).map(([k,v]) =>
        `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:120px;font-weight:600">${k}</span><div style="flex:1;background:#eee;border-radius:4px;height:20px"><div style="width:${(v*100).toFixed(0)}%;height:100%;background:${k.includes('VT')||k.includes('VF')?'#fa4d56':'#0f62fe'};border-radius:4px"></div></div><span style="width:50px;text-align:right">${(v*100).toFixed(1)}%</span></div>`
    ).join('');
    const ri = data.record_info || {};
    box.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;padding:1rem;border-radius:12px;background:${alertColors[alert]}15;border:2px solid ${alertColors[alert]}">
            <i class="fa-solid ${alertIcons[alert]}" style="font-size:2rem;color:${alertColors[alert]}"></i>
            <div><strong style="font-size:1.3rem;color:${alertColors[alert]}">${alert}</strong><br><span>Rhythm: ${data.rhythm_class} | Danger: ${(data.danger_probability * 100).toFixed(1)}%</span></div>
        </div>
        <div class="report-metrics"><div class="metric"><span class="label">Inference:</span><span class="value success">${data.inference_time_seconds}s</span></div><div class="metric"><span class="label">Mode:</span><span class="value">${data.mode}</span></div></div>
        ${ri.record_id ? `<p><strong>Record:</strong> ${ri.record_id} | <strong>Leads:</strong> ${(ri.leads||[]).join(', ')}</p>` : ''}
        <h4 style="margin:1rem 0 0.5rem">Rhythm Class Probabilities</h4>
        ${probBars}
        <h4 style="margin:1rem 0 0.5rem"><i class="fa-solid fa-book-open-reader"></i> Clinical Guidelines</h4>
        ${(data.retrieved_guidelines || []).map(g => `<div class="rag-item" style="border-left:3px solid ${alertColors[alert]}">${g}</div>`).join('')}`;
}

// ============================================================================
// PAGE 10: CardioFusion Hybrid Model
// ============================================================================
document.getElementById('fusionInfoBtn').addEventListener('click', async () => {
    const stats = document.getElementById('fusionModelStats');
    try {
        const res = await fetch(API_BASE + '/api/cardiofusion/info');
        const data = await res.json();
        stats.innerHTML = [
            `<span><strong>Total Params:</strong> ${(data.total_parameters/1e6).toFixed(1)}M</span>`,
            `<span><strong>Trainable:</strong> ${(data.trainable_parameters/1e6).toFixed(1)}M</span>`,
            `<span><strong>Model Size:</strong> ${data.model_size_mb}MB</span>`,
            `<span><strong>Shared Dim:</strong> ${data.shared_dim}</span>`,
            `<span><strong>Transformer:</strong> ${data.transformer_layers}L × ${data.attention_heads}H</span>`,
            `<span><strong>Experts:</strong> ${data.num_experts} × ${data.num_tasks} tasks</span>`,
        ].join(' | ');
    } catch (e) { stats.innerHTML = '<span style="color:var(--secondary)">Could not load model info. Ensure backend is running.</span>'; }
});

document.getElementById('fusionDemoBtn').addEventListener('click', async () => {
    const box = document.getElementById('fusionResultBox');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Running CardioFusion multi-modal analysis...</p></div>';

    const allChecked = document.getElementById('fusionModEcho').checked &&
                       document.getElementById('fusionModImg').checked &&
                       document.getElementById('fusionModSound').checked &&
                       document.getElementById('fusionModSignal').checked;

    try {
        let res;
        if (allChecked) {
            res = await fetch(API_BASE + '/api/cardiofusion/demo', { method: 'POST' });
        } else {
            // Use single-modality demo for the first checked
            const mods = [
                ['fusionModEcho', 'echo_video'], ['fusionModImg', 'ecg_image'],
                ['fusionModSound', 'heart_sound'], ['fusionModSignal', 'ecg_signal'],
            ];
            const selected = mods.find(([id]) => document.getElementById(id).checked);
            if (!selected) { box.innerHTML = '<div class="empty-state"><p>Select at least one modality.</p></div>'; return; }
            res = await fetch(API_BASE + `/api/cardiofusion/demo/single/${selected[1]}`, { method: 'POST' });
        }
        if (!res.ok) throw new Error('Analysis failed');
        const data = await res.json();
        displayFusionResult(data, box);
    } catch (e) { console.error(e); box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>CardioFusion analysis failed. Ensure backend is running.</p></div>'; }
});

document.getElementById('fusionUploadBtn').addEventListener('click', async () => {
    const box = document.getElementById('fusionResultBox');
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Processing uploads and running CardioFusion...</p></div>';
    try {
        const fd = new FormData();
        const echoF = document.getElementById('fusionEchoFile').files[0];
        const imgF = document.getElementById('fusionImgFile').files[0];
        const soundF = document.getElementById('fusionSoundFile').files[0];
        if (echoF) fd.append('echo_video', echoF);
        if (imgF) fd.append('ecg_image', imgF);
        if (soundF) fd.append('heart_sound', soundF);
        if (!echoF && !imgF && !soundF) { box.innerHTML = '<div class="empty-state"><p>Upload at least one file.</p></div>'; return; }
        const res = await fetch(API_BASE + '/api/cardiofusion/analyze', { method: 'POST', body: fd });
        if (!res.ok) throw new Error('Upload analysis failed');
        const data = await res.json();
        displayFusionResult(data, box);
    } catch (e) { console.error(e); box.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Analysis failed.</p></div>'; }
});

function displayFusionResult(data, box) {
    const alert = data.alert_level || 'NORMAL';
    const colors = { CRITICAL: '#fa4d56', WARNING: '#ff832b', NORMAL: '#198038' };
    const icons = { CRITICAL: 'fa-skull-crossbones', WARNING: 'fa-exclamation-triangle', NORMAL: 'fa-shield-check' };
    const risk = data.cardiac_risk_score || 0;
    const riskColor = risk > 80 ? '#fa4d56' : risk > 50 ? '#ff832b' : '#198038';
    const modsUsed = (data.modalities_used || []).map(m => `<span class="model-badge" style="font-size:0.75rem">${m.replace('_',' ')}</span>`).join(' ');

    const arr = data.arrhythmia || {};
    const ef = data.ejection_fraction || {};
    const mur = data.murmur || {};
    const vent = data.ventricular_danger || {};
    const rhy = data.rhythm || {};

    const arrBars = Object.entries(arr.probabilities || {}).map(([k,v]) =>
        `<div style="display:flex;align-items:center;gap:6px;margin:2px 0"><span style="width:24px;font-weight:600;font-size:0.8rem">${k}</span><div style="flex:1;background:#eee;border-radius:3px;height:16px"><div style="width:${(v*100).toFixed(0)}%;height:100%;background:${k==='V'||k==='F'?'#fa4d56':'#0f62fe'};border-radius:3px"></div></div><span style="width:40px;text-align:right;font-size:0.8rem">${(v*100).toFixed(1)}%</span></div>`
    ).join('');

    box.innerHTML = `
        <!-- Alert Banner -->
        <div style="display:flex;align-items:center;gap:16px;padding:1.25rem;border-radius:12px;background:${colors[alert]}12;border:2px solid ${colors[alert]};margin-bottom:1.5rem">
            <i class="fa-solid ${icons[alert]}" style="font-size:2.5rem;color:${colors[alert]}"></i>
            <div style="flex:1">
                <div style="font-size:1.5rem;font-weight:700;color:${colors[alert]}">${alert}</div>
                <div style="font-size:0.9rem;color:var(--text-light)">Unified cardiac risk assessment across ${(data.modalities_used||[]).length} modalities</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2.5rem;font-weight:800;color:${riskColor}">${risk.toFixed(0)}</div>
                <div style="font-size:0.75rem;color:var(--text-light)">Risk Score /100</div>
            </div>
        </div>

        <div style="margin-bottom:1rem">${modsUsed} <span style="font-size:0.8rem;color:var(--text-light);margin-left:8px">Inference: ${data.inference_time_seconds}s</span></div>

        <!-- 6 Task Results Grid -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
            <!-- Arrhythmia -->
            <div class="card" style="padding:1rem">
                <h4 style="margin:0 0 8px;font-size:0.9rem"><i class="fa-solid fa-heart-pulse"></i> Arrhythmia</h4>
                <div style="font-size:1.1rem;font-weight:700;color:${arr.class==='N'?'#198038':'#fa4d56'}">${arr.class || '—'}</div>
                <div style="font-size:0.8rem;color:var(--text-light)">Conf: ${((arr.confidence||0)*100).toFixed(1)}%</div>
                <div style="margin-top:6px">${arrBars}</div>
            </div>
            <!-- EF -->
            <div class="card" style="padding:1rem">
                <h4 style="margin:0 0 8px;font-size:0.9rem"><i class="fa-solid fa-gauge-high"></i> Ejection Fraction</h4>
                <div style="font-size:2rem;font-weight:800;color:${(ef.value||55)<40?'#fa4d56':(ef.value||55)<50?'#ff832b':'#198038'}">${(ef.value||0).toFixed(1)}%</div>
                <div style="font-size:0.85rem;font-weight:600">${ef.category || '—'}</div>
                <div style="font-size:0.8rem;color:var(--text-light)">Conf: ${((ef.confidence||0)*100).toFixed(1)}%</div>
            </div>
            <!-- Murmur -->
            <div class="card" style="padding:1rem">
                <h4 style="margin:0 0 8px;font-size:0.9rem"><i class="fa-solid fa-ear-listen"></i> Heart Murmur</h4>
                <div style="font-size:1.1rem;font-weight:700;color:${mur.class==='Present'?'#fa4d56':'#198038'}">${mur.class || '—'}</div>
                <div style="font-size:0.8rem;color:var(--text-light)">Conf: ${((mur.confidence||0)*100).toFixed(1)}%</div>
                ${Object.entries(mur.probabilities||{}).map(([k,v])=>`<div style="font-size:0.8rem">${k}: ${(v*100).toFixed(1)}%</div>`).join('')}
            </div>
            <!-- Ventricular Alert -->
            <div class="card" style="padding:1rem;border:${vent.is_dangerous?'2px solid #fa4d56':''}">
                <h4 style="margin:0 0 8px;font-size:0.9rem"><i class="fa-solid fa-bolt"></i> Ventricular Alert</h4>
                <div style="font-size:1.1rem;font-weight:700;color:${vent.is_dangerous?'#fa4d56':'#198038'}">${vent.is_dangerous ? 'DANGEROUS' : 'NORMAL'}</div>
                <div style="font-size:0.8rem;color:var(--text-light)">Danger prob: ${((vent.probability||0)*100).toFixed(1)}%</div>
            </div>
            <!-- Rhythm -->
            <div class="card" style="padding:1rem">
                <h4 style="margin:0 0 8px;font-size:0.9rem"><i class="fa-solid fa-wave-square"></i> Rhythm</h4>
                <div style="font-size:1.1rem;font-weight:700;color:${rhy.class==='Normal'?'#198038':'#fa4d56'}">${rhy.class || '—'}</div>
                <div style="font-size:0.8rem;color:var(--text-light)">Conf: ${((rhy.confidence||0)*100).toFixed(1)}%</div>
                ${Object.entries(rhy.probabilities||{}).map(([k,v])=>`<div style="font-size:0.8rem">${k}: ${(v*100).toFixed(1)}%</div>`).join('')}
            </div>
            <!-- Risk Gauge -->
            <div class="card" style="padding:1rem;text-align:center">
                <h4 style="margin:0 0 8px;font-size:0.9rem"><i class="fa-solid fa-shield-halved"></i> Cardiac Risk</h4>
                <div style="position:relative;width:100px;height:100px;margin:0 auto;border-radius:50%;background:conic-gradient(${riskColor} ${risk}%, #eee ${risk}%);display:flex;align-items:center;justify-content:center">
                    <div style="width:70px;height:70px;border-radius:50%;background:white;display:flex;align-items:center;justify-content:center;font-size:1.3rem;font-weight:800;color:${riskColor}">${risk.toFixed(0)}</div>
                </div>
                <div style="font-size:0.8rem;margin-top:4px;color:var(--text-light)">Severity Score</div>
            </div>
        </div>`;
}

// ============================================================================
// PAGE 5: Vector DB Stats
// ============================================================================
document.getElementById('loadDbStatsBtn').addEventListener('click', loadDbStats);

async function loadDbStats() {
    const btn = document.getElementById('loadDbStatsBtn');
    const container = document.getElementById('dbStatsContent');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';
    try {
        const res = await fetch(API_BASE + '/api/db/stats');
        if (!res.ok) throw new Error('DB Stats API failed');
        const data = await res.json();
        const apd = data.apd_dataset;
        const models = data.models;
        const missingKeys = Object.keys(apd.missing_data_summary || {});
        const missingHtml = missingKeys.length > 0
            ? missingKeys.map(k => `<span class="missing-tag">${k}: ${apd.missing_data_summary[k].toFixed(1)}%</span>`).join('')
            : '<span class="missing-tag ok">No missing data</span>';

        container.innerHTML = `
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-server" style="color:#8a3ffc"></i> Milvus Vector Database</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Status</span><span class="db-val ${data.milvus_connected ? 'status-active' : 'status-error'}"><i class="fa-solid fa-circle"></i> ${data.milvus_status}</span></div>
                    <div class="db-info-item"><span class="db-label">Backend</span><span class="db-val">${data.milvus_backend}</span></div>
                    <div class="db-info-item"><span class="db-label">Host</span><span class="db-val">${data.milvus_host}</span></div>
                    <div class="db-info-item"><span class="db-label">Collection</span><span class="db-val">${data.milvus_collection}</span></div>
                    <div class="db-info-item"><span class="db-label">Records</span><span class="db-val">${data.milvus_records}</span></div>
                    <div class="db-info-item"><span class="db-label">Embedding Dim</span><span class="db-val">${data.milvus_embedding_dim}</span></div>
                    <div class="db-info-item"><span class="db-label">Metric</span><span class="db-val">${data.milvus_metric}</span></div>
                    <div class="db-info-item"><span class="db-label">Index</span><span class="db-val">${data.milvus_index_type}</span></div>
                </div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-table" style="color:#198038"></i> ${apd.name || 'Arthritis Dataset'}</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Source</span><span class="db-val">${apd.source}</span></div>
                    <div class="db-info-item"><span class="db-label">Records</span><span class="db-val">${apd.total_records}</span></div>
                    <div class="db-info-item"><span class="db-label">Features</span><span class="db-val">${apd.total_features}</span></div>
                    <div class="db-info-item"><span class="db-label">Format</span><span class="db-val">${apd.file_format}</span></div>
                </div>
                <div class="missing-tags"><span class="db-label">Missing Data:</span> ${missingHtml}</div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-brain" style="color:#fa4d56"></i> ML Models</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item full"><span class="db-label">ECG Encoder</span><span class="db-val">${models.ecg_encoder}</span></div>
                    <div class="db-info-item full"><span class="db-label">Report Generator</span><span class="db-val">${models.report_generator}</span></div>
                    <div class="db-info-item full"><span class="db-label">Arthritis Predictor</span><span class="db-val">${models.arthritis_predictor}</span></div>
                </div>
            </div>`;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Loaded';
    } catch (err) {
        console.error(err);
        container.innerHTML = '<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Failed to load stats.</p></div>';
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-sync"></i> Refresh Stats'; }, 2000);
}


// ============================================================================
// PAGE: HEART DISEASE (UCI Cleveland / Kaggle)
// ============================================================================

async function loadHeartDiseaseEDA() {
    const container = document.getElementById('hd-eda-container');
    if (!container) return;
    container.innerHTML = '<p class="loading-text">Loading Heart Disease EDA...</p>';
    try {
        const res = await fetch(API_BASE + '/api/heart-disease/eda');
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to load EDA');

        const classD = data.class_distribution || {};
        container.innerHTML = `
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-heart-pulse" style="color:#ef4444"></i> Dataset Overview</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Total Samples</span><span class="db-val">${data.total_samples}</span></div>
                    <div class="db-info-item"><span class="db-label">Features</span><span class="db-val">${data.total_features}</span></div>
                    <div class="db-info-item"><span class="db-label">Disease Cases</span><span class="db-val">${classD.disease || 'N/A'}</span></div>
                    <div class="db-info-item"><span class="db-label">Healthy Cases</span><span class="db-val">${classD.no_disease || 'N/A'}</span></div>
                    <div class="db-info-item"><span class="db-label">Disease Ratio</span><span class="db-val">${((classD.disease_ratio || 0) * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">Male/Female</span><span class="db-val">${data.gender_distribution?.male || '?'} / ${data.gender_distribution?.female || '?'}</span></div>
                </div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-chart-bar" style="color:#3b82f6"></i> Age Statistics</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Mean Age</span><span class="db-val">${data.age_stats?.mean}</span></div>
                    <div class="db-info-item"><span class="db-label">Min Age</span><span class="db-val">${data.age_stats?.min}</span></div>
                    <div class="db-info-item"><span class="db-label">Max Age</span><span class="db-val">${data.age_stats?.max}</span></div>
                    <div class="db-info-item"><span class="db-label">Std Dev</span><span class="db-val">${data.age_stats?.std}</span></div>
                </div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3>Cardiac Features</h3></div>
                <div style="height:220px"><canvas id="hd-cardiac-chart"></canvas></div>
            </div>`;

        // Cardiac features bar chart
        const cardiac = data.cardiac_features || {};
        const labels = Object.keys(cardiac);
        const means = labels.map(k => cardiac[k]?.mean || 0);
        if (labels.length > 0) {
            const cCtx = document.getElementById('hd-cardiac-chart');
            if (cCtx) new Chart(cCtx, {
                type: 'bar',
                data: { labels, datasets: [{ label: 'Mean Value', data: means, backgroundColor: '#ef4444cc' }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#fff' } } },
                    scales: { x: { ticks: { color: '#ccc' } }, y: { ticks: { color: '#ccc' } } } }
            });
        }
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>${err.message}</p></div>`;
    }
}

async function trainHeartDiseaseModel() {
    const btn = document.getElementById('hd-train-btn');
    const container = document.getElementById('hd-train-results');
    if (!btn || !container) return;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
    container.innerHTML = '<p class="loading-text">Training Tabular BERT + MoE on Heart Disease data...</p>';
    try {
        const res = await fetch(API_BASE + '/api/heart-disease/train', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Training failed');
        lastHDTrainMetrics = data;
        container.innerHTML = `
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-trophy" style="color:#22c55e"></i> Training Results</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Accuracy</span><span class="db-val">${(data.accuracy * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">CV Accuracy</span><span class="db-val">${(data.cv_mean_accuracy * 100).toFixed(1)}% ± ${(data.cv_std * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">AUC-ROC</span><span class="db-val">${data.auc_roc}</span></div>
                    <div class="db-info-item"><span class="db-label">CV AUC</span><span class="db-val">${data.cv_mean_auc || 'N/A'}</span></div>
                    <div class="db-info-item"><span class="db-label">Features</span><span class="db-val">${data.total_features}</span></div>
                    <div class="db-info-item"><span class="db-label">Training Time</span><span class="db-val">${data.training_time_seconds}s</span></div>
                </div>
            </div>`;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Trained';
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>${err.message}</p></div>`;
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-play"></i> Train Model'; }, 3000);
}

async function predictHeartDisease() {
    const btn = document.getElementById('hd-predict-btn');
    const container = document.getElementById('hd-predict-results');
    if (!btn || !container) return;
    btn.disabled = true;

    const fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
    const patient = {};
    for (const f of fields) {
        const el = document.getElementById('hd-' + f);
        if (el && el.value !== '') patient[f] = parseFloat(el.value);
    }

    try {
        const res = await fetch(API_BASE + '/api/heart-disease/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patient)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');
        lastHDPredictData = data;
        lastHDPatientInput = patient;

        const riskColor = data.risk_level === 'HIGH' ? '#ef4444' : '#22c55e';
        container.innerHTML = `
            <div class="card" style="border-left:4px solid ${riskColor}">
                <h3 style="color:${riskColor}">${data.risk_level} RISK</h3>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Confidence</span><span class="db-val">${(data.confidence * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">Disease Prob</span><span class="db-val">${(data.probabilities.disease * 100).toFixed(1)}%</span></div>
                </div>
                <pre style="white-space:pre-wrap;font-size:0.85rem;color:#94a3b8;margin-top:1rem">${data.clinical_interpretation}</pre>
            </div>`;
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>${err.message}</p></div>`;
    }
    btn.disabled = false;
}


// ============================================================================
// PAGE: ECG ARRHYTHMIA CLASSIFICATION (MIT-BIH / Kaggle)
// ============================================================================

let arrEdaLoaded = false;
async function loadArrhythmiaEDA() {
    const container = document.getElementById('arr-eda-container');
    if (!container || arrEdaLoaded) return;
    container.innerHTML = '<p class="loading-text">Loading ECG Arrhythmia EDA...</p>';
    try {
        const res = await fetch(API_BASE + '/api/ecg-arrhythmia/eda');
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to load EDA');

        const classDist = data.class_distribution || {};
        const classImb = data.class_imbalance || {};
        const sigStats = data.signal_statistics || {};

        container.innerHTML = `
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-wave-square" style="color:#06b6d4"></i> Dataset Overview</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Train Samples</span><span class="db-val">${data.total_train_samples?.toLocaleString()}</span></div>
                    <div class="db-info-item"><span class="db-label">Test Samples</span><span class="db-val">${data.total_test_samples?.toLocaleString()}</span></div>
                    <div class="db-info-item"><span class="db-label">Signal Length</span><span class="db-val">${data.signal_length} timesteps</span></div>
                    <div class="db-info-item"><span class="db-label">Classes</span><span class="db-val">${data.num_classes} AAMI</span></div>
                </div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-chart-pie" style="color:#8b5cf6"></i> Class Distribution</h3></div>
                <div style="height:260px"><canvas id="arr-class-chart"></canvas></div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3>Signal Statistics</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Mean Amplitude</span><span class="db-val">${sigStats.mean_amplitude}</span></div>
                    <div class="db-info-item"><span class="db-label">Std Amplitude</span><span class="db-val">${sigStats.std_amplitude}</span></div>
                    <div class="db-info-item"><span class="db-label">Mean Peak</span><span class="db-val">${sigStats.mean_peak}</span></div>
                </div>
            </div>`;

        // Class distribution bar chart
        const labels = Object.keys(classDist);
        const counts = labels.map(k => classDist[k]);
        if (labels.length > 0) {
            const cCtx = document.getElementById('arr-class-chart');
            if (cCtx) new Chart(cCtx, {
                type: 'bar',
                data: {
                    labels: labels.map(l => l.length > 20 ? l.substring(0, 18) + '...' : l),
                    datasets: [{ label: 'Count', data: counts, backgroundColor: ['#22c55e','#eab308','#ef4444','#3b82f6','#8b5cf6'] }]
                },
                options: { responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#fff' } } },
                    scales: { x: { ticks: { color: '#ccc', font: { size: 9 } } }, y: { ticks: { color: '#ccc' } } } }
            });
        }
        arrEdaLoaded = true;
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>${err.message}</p></div>`;
    }
}

async function trainArrhythmiaModel() {
    const btn = document.getElementById('arr-train-btn');
    const container = document.getElementById('arr-train-results');
    if (!btn || !container) return;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training 1D-CNN (may take a few minutes)...';
    container.innerHTML = '<p class="loading-text">Training 1D-CNN on 87,554 heartbeats...</p>';
    try {
        const res = await fetch(API_BASE + '/api/ecg-arrhythmia/train', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Training failed');
        lastArrTrainMetrics = data;

        const perClass = data.per_class_metrics || {};
        let classRows = '';
        for (const [cls, m] of Object.entries(perClass)) {
            const sevColor = m.severity === 'critical' ? '#ef4444' : m.severity === 'moderate' ? '#eab308' : '#22c55e';
            classRows += `<tr><td>${cls}</td><td>${m.total}</td><td>${(m.accuracy * 100).toFixed(1)}%</td><td style="color:${sevColor}">${m.severity}</td></tr>`;
        }

        container.innerHTML = `
            <div class="card db-stat-card">
                <div class="card-header"><h3><i class="fa-solid fa-trophy" style="color:#22c55e"></i> Training Results</h3></div>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Accuracy</span><span class="db-val">${(data.accuracy * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">F1 (Macro)</span><span class="db-val">${(data.f1_macro * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">F1 (Weighted)</span><span class="db-val">${(data.f1_weighted * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">AUC-ROC (OVR)</span><span class="db-val">${data.auc_roc_ovr}</span></div>
                    <div class="db-info-item"><span class="db-label">Model Params</span><span class="db-val">${(data.model_params / 1000).toFixed(0)}K</span></div>
                    <div class="db-info-item"><span class="db-label">Training Time</span><span class="db-val">${data.training_time_seconds}s</span></div>
                </div>
            </div>
            <div class="card db-stat-card">
                <div class="card-header"><h3>Per-Class Performance</h3></div>
                <table class="stats-table"><thead><tr><th>Class</th><th>Samples</th><th>Accuracy</th><th>Severity</th></tr></thead>
                <tbody>${classRows}</tbody></table>
            </div>`;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Trained';
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>${err.message}</p></div>`;
        btn.innerHTML = '<i class="fa-solid fa-xmark"></i> Error';
    }
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-play"></i> Train 1D-CNN'; }, 3000);
}

async function demoArrhythmiaPrediction() {
    const btn = document.getElementById('arr-demo-btn');
    const container = document.getElementById('arr-predict-results');
    if (!btn || !container) return;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Classifying...';
    try {
        const res = await fetch(API_BASE + '/api/ecg-arrhythmia/predict/demo', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');

        const sevColor = data.severity === 'critical' ? '#ef4444' : data.severity === 'moderate' ? '#eab308' : '#22c55e';
        const correct = data.predicted_class === data.true_class;
        container.innerHTML = `
            <div class="card" style="border-left:4px solid ${sevColor}">
                <h3 style="color:${sevColor}">${data.class_name}</h3>
                <div class="db-info-grid">
                    <div class="db-info-item"><span class="db-label">Confidence</span><span class="db-val">${(data.confidence * 100).toFixed(1)}%</span></div>
                    <div class="db-info-item"><span class="db-label">True Class</span><span class="db-val">${data.true_class_name}</span></div>
                    <div class="db-info-item"><span class="db-label">Correct</span><span class="db-val" style="color:${correct ? '#22c55e' : '#ef4444'}">${correct ? 'YES' : 'NO'}</span></div>
                    <div class="db-info-item"><span class="db-label">Severity</span><span class="db-val" style="color:${sevColor}">${data.severity}</span></div>
                </div>
                <pre style="white-space:pre-wrap;font-size:0.85rem;color:#94a3b8;margin-top:1rem">${data.clinical_note}</pre>
            </div>
            <div class="card db-stat-card" style="margin-top:1rem">
                <div class="card-header"><h3>Heartbeat Waveform</h3></div>
                <div style="height:180px"><canvas id="arr-beat-chart"></canvas></div>
            </div>`;

        // Plot the heartbeat waveform
        if (data.signal && data.signal.length > 0) {
            const beatCtx = document.getElementById('arr-beat-chart');
            if (beatCtx) new Chart(beatCtx, {
                type: 'line',
                data: {
                    labels: data.signal.map((_, i) => i),
                    datasets: [{ label: 'Heartbeat', data: data.signal, borderColor: sevColor, borderWidth: 1.5, pointRadius: 0, tension: 0.1 }]
                },
                options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
                    plugins: { legend: { labels: { color: '#fff' } } },
                    scales: { x: { display: false }, y: { ticks: { color: '#ccc' } } } }
            });
        }
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>${err.message}</p></div>`;
    }
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-wave-square"></i> Classify Random Heartbeat';
}

// ============================================================================
// HEART SOUND: Combined Murmur + Malignant Ventricular Arrhythmia Detection
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('hsCombinedBtn');
    if (btn) btn.addEventListener('click', runHSCombinedAnalysis);

    const ecgPipeBtn = document.getElementById('ecgImgPipelineBtn');
    if (ecgPipeBtn) ecgPipeBtn.addEventListener('click', runEcgImagePipeline);

    const unifiedBtn = document.getElementById('fusionUnifiedBtn');
    if (unifiedBtn) unifiedBtn.addEventListener('click', runUnifiedCardioFusion);

    const valBtn = document.getElementById('validationBtn');
    if (valBtn) valBtn.addEventListener('click', loadValidationReport);
});

async function runHSCombinedAnalysis() {
    const btn = document.getElementById('hsCombinedBtn');
    const box = document.getElementById('hsCombinedResult');
    if (!box) return;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...'; }
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Running combined murmur + arrhythmia detection...</p></div>';
    try {
        const res = await fetch(API_BASE + '/api/heart-sound/analyze/combined', { method: 'POST' });
        if (!res.ok) throw new Error('Combined analysis failed');
        const data = await res.json();
        displayHSCombinedResult(data, box);
    } catch (e) {
        console.error(e);
        box.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Analysis error: ${e.message}</p></div>`;
    }
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-stethoscope"></i> Run Combined Analysis'; }
}

function displayHSCombinedResult(data, box) {
    const murmur = data.murmur_detection || {};
    const va     = data.ventricular_arrhythmia_detection || {};
    const summary = data.combined_clinical_summary || {};
    const alertLevel = summary.overall_alert || 'NORMAL';
    const alertColors = { CRITICAL: '#fa4d56', WARNING: '#ff832b', NORMAL: '#198038' };
    const alertIcons  = { CRITICAL: 'fa-skull-crossbones', WARNING: 'fa-exclamation-triangle', NORMAL: 'fa-shield-check' };
    const alertColor  = alertColors[alertLevel] || '#198038';

    // Murmur probability bars
    const murmurBars = Object.entries(murmur.probabilities || {}).map(([k, v]) =>
        `<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
            <span style="width:80px;font-weight:600;font-size:0.85rem">${k}</span>
            <div style="flex:1;background:#e5e7eb;border-radius:4px;height:18px">
                <div style="width:${(v*100).toFixed(0)}%;height:100%;background:${k==='Present'?'#fa4d56':k==='Absent'?'#198038':'#ff832b'};border-radius:4px;transition:width 0.5s"></div>
            </div>
            <span style="width:48px;text-align:right;font-size:0.85rem">${(v*100).toFixed(1)}%</span>
        </div>`
    ).join('');

    box.innerHTML = `
        <!-- Overall Alert -->
        <div style="display:flex;align-items:center;gap:16px;padding:1.25rem;border-radius:12px;background:${alertColor}12;border:2px solid ${alertColor};margin-bottom:1.5rem">
            <i class="fa-solid ${alertIcons[alertLevel]}" style="font-size:2.5rem;color:${alertColor}"></i>
            <div>
                <div style="font-size:1.4rem;font-weight:700;color:${alertColor}">Combined Alert: ${alertLevel}</div>
                <div style="font-size:0.9rem;color:#64748b">${(summary.findings||[]).join(' | ')}</div>
                <div style="font-size:0.8rem;color:#94a3b8">Inference: ${data.inference_time_seconds}s | Mode: ${data.mode}</div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <!-- === MURMUR DETECTION RESULT === -->
            <div class="card" style="padding:1.25rem;border-top:3px solid ${murmur.result==='Present'?'#fa4d56':'#198038'}">
                <h4 style="margin:0 0 12px;display:flex;align-items:center;gap:8px">
                    <i class="fa-solid fa-ear-listen" style="color:${murmur.result==='Present'?'#fa4d56':'#198038'}"></i>
                    Murmur Detection Result
                    <span style="font-size:0.7rem;background:${murmur.result==='Present'?'#fa4d5622':'#19803822'};color:${murmur.result==='Present'?'#fa4d56':'#198038'};padding:2px 8px;border-radius:20px;font-weight:600">${murmur.result}</span>
                </h4>
                <div style="font-size:2rem;font-weight:800;color:${murmur.result==='Present'?'#fa4d56':murmur.result==='Absent'?'#198038':'#ff832b'};margin-bottom:8px">
                    ${murmur.result === 'Present' ? '⚠ MURMUR DETECTED' : murmur.result === 'Absent' ? '✓ NO MURMUR' : '? INCONCLUSIVE'}
                </div>
                <div style="margin-bottom:12px;font-size:0.9rem;color:#64748b">${murmur.interpretation || ''}</div>
                <div style="font-weight:600;margin-bottom:6px">Confidence: ${(murmur.confidence*100).toFixed(1)}%</div>
                ${murmurBars}
                <div style="margin-top:12px;padding:8px;background:#f8fafc;border-radius:6px;font-size:0.8rem">
                    <strong>Dataset:</strong> CirCor DigiScope PhysioNet<br>
                    <strong>Records:</strong> 5,272 heart sound recordings | 1,568 subjects<br>
                    <a href="https://physionet.org/content/circor-heart-sound/1.0.1/" target="_blank" style="color:var(--primary)">View Dataset →</a>
                </div>
                ${(murmur.clinical_guidelines||[]).map(g => `<div class="rag-item" style="margin-top:8px">${g}</div>`).join('')}
            </div>

            <!-- === MALIGNANT VENTRICULAR ARRHYTHMIA DETECTION RESULT === -->
            <div class="card" style="padding:1.25rem;border-top:3px solid ${va.is_malignant?'#fa4d56':'#198038'}">
                <h4 style="margin:0 0 12px;display:flex;align-items:center;gap:8px">
                    <i class="fa-solid fa-bolt" style="color:${va.is_malignant?'#fa4d56':'#198038'}"></i>
                    Malignant Ventricular Arrhythmia Detection
                    <span style="font-size:0.7rem;background:${va.is_malignant?'#fa4d5622':'#19803822'};color:${va.is_malignant?'#fa4d56':'#198038'};padding:2px 8px;border-radius:20px;font-weight:600">${va.is_malignant?'MALIGNANT':'NORMAL'}</span>
                </h4>
                <div style="font-size:${va.is_malignant?'1.1':'2'}rem;font-weight:800;color:${va.is_malignant?'#fa4d56':'#198038'};margin-bottom:8px">
                    ${va.is_malignant
                        ? `🚨 ${va.alert_level}`
                        : `✓ NO MALIGNANT ARRHYTHMIA`}
                </div>
                <div style="font-size:1.1rem;font-weight:600;margin-bottom:8px">Rhythm: <span style="color:${va.is_malignant?'#fa4d56':'#198038'}">${va.result}</span></div>
                <div style="margin-bottom:12px;font-size:0.9rem;color:#64748b">${va.interpretation || ''}</div>
                <div style="font-weight:600;margin-bottom:6px">Confidence: ${(va.confidence*100).toFixed(1)}%</div>
                ${va.is_malignant ? `
                <div style="padding:12px;background:#fa4d5615;border:1px solid #fa4d56;border-radius:8px;margin:8px 0">
                    <strong style="color:#fa4d56">⚠ CRITICAL: ${(va.malignant_types_detected||[]).join(', ')}</strong><br>
                    <span style="font-size:0.85rem">Immediate clinical intervention may be required.</span>
                </div>` : ''}
                <div style="margin-top:12px;padding:8px;background:#f8fafc;border-radius:6px;font-size:0.8rem">
                    <strong>Dataset:</strong> MIT-BIH VFDB (22 recordings, PhysioNet)<br>
                    <strong>Detects:</strong> VT, VF, VFL, Asystole, HGEA<br>
                    <a href="https://physionet.org/content/vfdb/1.0.0/" target="_blank" style="color:var(--primary)">View Dataset →</a>
                </div>
                ${(va.clinical_guidelines||[]).map(g => `<div class="rag-item" style="margin-top:8px;border-left-color:${va.is_malignant?'#fa4d56':'#198038'}">${g}</div>`).join('')}
            </div>
        </div>`;
}

// ============================================================================
// ECG IMAGE: Input → Processing → Output Pipeline Display
// ============================================================================
async function runEcgImagePipeline() {
    const btn = document.getElementById('ecgImgPipelineBtn');
    const box = document.getElementById('ecgImgPipelineResult');
    if (!box) return;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Pipeline...'; }
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Running ECG Image Classification Pipeline...</p></div>';
    try {
        const fd = new FormData();
        if (ecgImgFile) fd.append('image', ecgImgFile);
        const res = await fetch(API_BASE + '/api/ecg-images/classify/pipeline', {
            method: 'POST',
            body: ecgImgFile ? fd : undefined
        });
        if (!res.ok) throw new Error('Pipeline failed');
        const data = await res.json();
        displayEcgImagePipeline(data, box);
    } catch (e) {
        console.error(e);
        box.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Pipeline error: ${e.message}</p></div>`;
    }
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-diagram-project"></i> Run Pipeline'; }
}

function displayEcgImagePipeline(data, box) {
    const stages = data.pipeline_stages || {};
    const s1 = stages.stage1_input || {};
    const s2 = stages.stage2_preprocessing || {};
    const s3 = stages.stage3_feature_extraction || {};
    const s4 = stages.stage4_classification || {};
    const dsInfo = data.dataset_info || {};
    const riskColor = s4.class_label === 'N' ? '#198038' : s4.class_label === 'V' ? '#fa4d56' : '#ff832b';

    const imgTag = (b64) => b64 ? `<img src="data:image/png;base64,${b64}" style="width:100%;max-width:140px;border-radius:8px;border:2px solid var(--border)" alt="ECG Stage">` : '<div style="width:140px;height:140px;background:#f1f5f9;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:0.8rem">No image</div>';

    const probBars = Object.entries(s4.probabilities || {}).map(([k,v]) =>
        `<div style="display:flex;align-items:center;gap:6px;margin:3px 0">
            <span style="width:24px;font-weight:700;font-size:0.8rem">${k}</span>
            <div style="flex:1;background:#e5e7eb;border-radius:3px;height:16px">
                <div style="width:${(v*100).toFixed(0)}%;height:100%;background:${k==='N'?'#198038':k==='V'?'#fa4d56':'#0f62fe'};border-radius:3px"></div>
            </div>
            <span style="width:44px;text-align:right;font-size:0.8rem">${(v*100).toFixed(1)}%</span>
        </div>`
    ).join('');

    box.innerHTML = `
        <h4 style="margin-bottom:1rem"><i class="fa-solid fa-diagram-project"></i> ECG Image Classification Pipeline</h4>

        <!-- Pipeline Flow -->
        <div style="display:flex;align-items:flex-start;gap:8px;overflow-x:auto;padding:1rem 0;margin-bottom:1.5rem">
            <!-- Stage 1 -->
            <div class="card" style="padding:1rem;min-width:160px;text-align:center">
                <div style="font-size:0.75rem;background:#0f62fe22;color:#0f62fe;padding:2px 8px;border-radius:20px;margin-bottom:8px">STAGE 1</div>
                <strong style="font-size:0.9rem">${s1.label}</strong>
                <div style="margin:8px auto">${imgTag(s1.image_base64)}</div>
                <div style="font-size:0.75rem;color:#64748b">${s1.description || ''}</div>
            </div>

            <!-- Arrow -->
            <div style="display:flex;align-items:center;padding-top:60px;font-size:1.5rem;color:#0f62fe;flex-shrink:0">→</div>

            <!-- Stage 2 -->
            <div class="card" style="padding:1rem;min-width:160px;text-align:center">
                <div style="font-size:0.75rem;background:#19803822;color:#198038;padding:2px 8px;border-radius:20px;margin-bottom:8px">STAGE 2</div>
                <strong style="font-size:0.9rem">${s2.label}</strong>
                <div style="margin:8px auto">${imgTag(s2.image_base64)}</div>
                <div style="font-size:0.75rem;color:#64748b">${(s2.steps||[]).join(' → ')}</div>
            </div>

            <!-- Arrow -->
            <div style="display:flex;align-items:center;padding-top:60px;font-size:1.5rem;color:#198038;flex-shrink:0">→</div>

            <!-- Stage 3 -->
            <div class="card" style="padding:1rem;min-width:180px">
                <div style="font-size:0.75rem;background:#8a3ffc22;color:#8a3ffc;padding:2px 8px;border-radius:20px;margin-bottom:8px;text-align:center">STAGE 3</div>
                <strong style="font-size:0.9rem">${s3.label}</strong>
                <div style="margin-top:8px">${(s3.layers||[]).map((l,i) =>
                    `<div style="display:flex;align-items:center;gap:6px;margin:3px 0;font-size:0.8rem">
                        <div style="width:14px;height:14px;border-radius:50%;background:${['#0f62fe','#198038','#ff832b','#fa4d56','#8a3ffc'][i]};flex-shrink:0"></div>
                        ${l}
                    </div>`).join('')}
                </div>
            </div>

            <!-- Arrow -->
            <div style="display:flex;align-items:center;padding-top:60px;font-size:1.5rem;color:#8a3ffc;flex-shrink:0">→</div>

            <!-- Stage 4 -->
            <div class="card" style="padding:1rem;min-width:200px;border-top:3px solid ${riskColor}">
                <div style="font-size:0.75rem;background:${riskColor}22;color:${riskColor};padding:2px 8px;border-radius:20px;margin-bottom:8px">STAGE 4: OUTPUT</div>
                <strong style="font-size:0.9rem">${s4.label}</strong>
                <div style="margin:8px auto">${imgTag(s4.image_base64)}</div>
                <div style="font-size:1.2rem;font-weight:800;color:${riskColor};margin:6px 0">${s4.class_name || s4.class_label} (${s4.class_label})</div>
                <div style="font-size:0.85rem;margin-bottom:8px">${s4.description || ''}</div>
                <div style="font-weight:600;font-size:0.85rem">Confidence: ${((s4.confidence||0)*100).toFixed(1)}%</div>
                <div style="margin-top:8px">${probBars}</div>
            </div>
        </div>

        <!-- Dataset Info -->
        <div style="padding:12px;background:#f8fafc;border-radius:8px;font-size:0.85rem;border:1px solid var(--border)">
            <strong><i class="fa-solid fa-database"></i> Dataset:</strong> ${dsInfo.name || 'ECG Images (Kaggle)'} |
            <strong>Total:</strong> ${(dsInfo.total_images||109445).toLocaleString()} images |
            <strong>Classes:</strong> ${(dsInfo.classes||[]).join(', ')} |
            ${dsInfo.url ? `<a href="${dsInfo.url}" target="_blank" style="color:var(--primary)">View on Kaggle →</a>` : ''}
        </div>

        <!-- Guidelines -->
        ${(s4.retrieved_guidelines||[]).length > 0 ? `
        <h4 style="margin:1rem 0 0.5rem"><i class="fa-solid fa-book-open-reader"></i> Clinical Guidelines</h4>
        ${(s4.retrieved_guidelines||[]).map(g => `<div class="rag-item">${g}</div>`).join('')}` : ''}`;
}

// ============================================================================
// CARDIOFUSION: Unified Multi-Modal Report (All 4 Models)
// ============================================================================
async function runUnifiedCardioFusion() {
    const btn = document.getElementById('fusionUnifiedBtn');
    const box = document.getElementById('fusionUnifiedResult');
    if (!box) return;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running All 4 Models...'; }
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Running ECG Transformer-MoE + EchoNet 3D-CNN + HeartSound CNN + VFDB Arrhythmia Detector...<br><small>This may take 15-30 seconds</small></p></div>';
    try {
        const res = await fetch(API_BASE + '/api/cardiofusion/unified-report', { method: 'POST' });
        if (!res.ok) throw new Error('Unified analysis failed');
        const data = await res.json();
        displayUnifiedFusionResult(data, box);
    } catch (e) {
        console.error(e);
        box.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Unified analysis failed: ${e.message}</p></div>`;
    }
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-atom"></i> Run All 4 Models (Unified Analysis)'; }
}

function displayUnifiedFusionResult(data, box) {
    const risk     = data.overall_risk || 'LOW';
    const riskScore = data.risk_score || 0;
    const riskColors = { HIGH: '#fa4d56', MODERATE: '#ff832b', LOW: '#198038' };
    const riskColor  = riskColors[risk] || '#198038';
    const riskIcons  = { HIGH: 'fa-skull-crossbones', MODERATE: 'fa-exclamation-triangle', LOW: 'fa-shield-check' };
    const results    = data.individual_results || {};
    const riskFactors = data.risk_factors_identified || [];

    const modelCards = Object.entries(results).map(([key, r]) => {
        if (r.error) return `<div class="card" style="padding:1rem;opacity:0.6"><h5>${key}</h5><p style="color:#ef4444">Error: ${r.error}</p></div>`;
        let content = '';
        if (key === 'ecg_signal') {
            content = `<div style="font-size:1.2rem;font-weight:700;color:${r.prediction==='Normal'?'#198038':'#fa4d56'}">${r.prediction}</div><div style="font-size:0.85rem">Conf: ${(r.confidence*100).toFixed(1)}%</div>`;
        } else if (key === 'echo_video') {
            const efColor = r.ef_category==='HFrEF'?'#fa4d56':r.ef_category==='Normal'?'#198038':'#ff832b';
            content = `<div style="font-size:2rem;font-weight:800;color:${efColor}">${r.ef_predicted?.toFixed(1)}%</div><div style="font-weight:600">${r.ef_category}</div>`;
        } else if (key === 'heart_sound') {
            content = `<div style="font-size:1.2rem;font-weight:700;color:${r.murmur_detection==='Present'?'#fa4d56':'#198038'}">${r.murmur_detection==='Present'?'MURMUR DETECTED':'No Murmur'}</div><div style="font-size:0.85rem">Conf: ${(r.confidence*100).toFixed(1)}%</div>`;
        } else if (key === 'ventricular_arrhythmia') {
            content = `<div style="font-size:1.1rem;font-weight:700;color:${r.is_dangerous?'#fa4d56':'#198038'}">${r.is_dangerous?'⚠ MALIGNANT VA':'✓ Normal Rhythm'}</div><div style="font-size:0.85rem">${r.rhythm_class} | Conf: ${(r.confidence*100).toFixed(1)}%</div>`;
        }
        const modelNames = {
            ecg_signal: 'ECG Transformer-MoE', echo_video: 'EchoNet 3D-CNN',
            heart_sound: 'HeartSound CNN (CirCor)', ventricular_arrhythmia: 'VFDB Arrhythmia CNN'
        };
        return `<div class="card" style="padding:1rem;border-top:3px solid ${riskColor}">
            <h5 style="margin:0 0 8px;font-size:0.85rem;color:#64748b">${modelNames[key] || key}</h5>
            ${content}</div>`;
    }).join('');

    box.innerHTML = `
        <!-- Overall Risk Banner -->
        <div style="display:flex;align-items:center;gap:20px;padding:1.5rem;border-radius:12px;background:${riskColor}12;border:2px solid ${riskColor};margin-bottom:1.5rem">
            <i class="fa-solid ${riskIcons[risk]}" style="font-size:3rem;color:${riskColor}"></i>
            <div style="flex:1">
                <div style="font-size:1.8rem;font-weight:800;color:${riskColor}">Overall Cardiac Risk: ${risk}</div>
                <div style="font-size:0.95rem;color:#64748b">4-model unified analysis | Inference: ${data.inference_time_seconds}s</div>
                ${riskFactors.length > 0 ? `<div style="margin-top:8px">${riskFactors.map(f=>
                    `<span style="background:${riskColor}22;color:${riskColor};padding:2px 10px;border-radius:20px;font-size:0.8rem;margin-right:6px">${f}</span>`
                ).join('')}</div>` : ''}
            </div>
            <div style="text-align:center">
                <div style="font-size:3rem;font-weight:900;color:${riskColor}">${riskScore}</div>
                <div style="font-size:0.75rem;color:#64748b">Risk Score /100</div>
            </div>
        </div>

        <!-- 4 Model Results Grid -->
        <h4 style="margin-bottom:12px"><i class="fa-solid fa-layer-group"></i> Individual Model Results</h4>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:1.5rem">
            ${modelCards}
        </div>

        <!-- Doctor Report -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div class="card" style="padding:1rem;border-left:4px solid #0f62fe">
                <h4 style="margin:0 0 12px;color:#0f62fe"><i class="fa-solid fa-user-doctor"></i> Doctor's Clinical Report</h4>
                <pre style="white-space:pre-wrap;font-size:0.82rem;line-height:1.6;color:#334155;max-height:320px;overflow-y:auto">${data.doctor_report || ''}</pre>
                <div style="margin-top:12px;display:flex;gap:8px">
                    <button class="analyze-btn" onclick="downloadUnifiedPdf('doctor', ${JSON.stringify(data).replace(/"/g,'&quot;')})" style="font-size:0.8rem;padding:6px 14px">
                        <i class="fa-solid fa-file-pdf"></i> Doctor PDF
                    </button>
                </div>
            </div>
            <div class="card" style="padding:1rem;border-left:4px solid #198038">
                <h4 style="margin:0 0 12px;color:#198038"><i class="fa-solid fa-user"></i> Patient Summary Report</h4>
                <pre style="white-space:pre-wrap;font-size:0.82rem;line-height:1.6;color:#334155;max-height:320px;overflow-y:auto">${data.patient_report || ''}</pre>
                <div style="margin-top:12px;display:flex;gap:8px">
                    <button class="analyze-btn" onclick="downloadUnifiedPdf('patient', ${JSON.stringify(data).replace(/"/g,'&quot;')})" style="font-size:0.8rem;padding:6px 14px">
                        <i class="fa-solid fa-file-pdf"></i> Patient PDF
                    </button>
                </div>
            </div>
        </div>`;
}

async function downloadUnifiedPdf(audience, data) {
    try {
        const res = await fetch(API_BASE + '/api/pdf/fusion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                doctor_report:      data.doctor_report || '',
                patient_report:     data.patient_report || '',
                overall_risk:       data.overall_risk || 'UNKNOWN',
                risk_score:         data.risk_score || 0,
                models_run:         data.models_run || [],
                individual_results: data.individual_results || {},
                audience
            })
        });
        if (!res.ok) throw new Error('PDF generation failed');
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url;
        a.download = `CardioFusion_${audience === 'doctor' ? 'Doctor' : 'Patient'}_Report.pdf`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) { alert('PDF generation failed: ' + e.message); }
}

// ============================================================================
// VALIDATION: Cardiologist Comparison Report
// ============================================================================
async function loadValidationReport() {
    const btn = document.getElementById('validationBtn');
    const box = document.getElementById('validationResult');
    if (!box) return;
    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...'; }
    box.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Loading cardiologist benchmark comparison...</p></div>';
    try {
        const res = await fetch(API_BASE + '/api/validation/unified-report');
        if (!res.ok) throw new Error('Validation API failed');
        const data = await res.json();
        displayValidationReport(data, box);
    } catch (e) {
        console.error(e);
        box.innerHTML = `<div class="empty-state"><i class="fa-solid fa-xmark" style="color:var(--secondary)"></i><p>Failed: ${e.message}</p></div>`;
    }
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-chart-line"></i> Compare vs Cardiologist'; }
}

// Settings screen: fill the rows that state facts about the running system from the
// live API. They were hardcoded and had drifted — see the comments in index.html.
async function loadSettingsInfo() {
    const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
    try {
        const res = await fetch(API_BASE + '/api/db/stats');
        if (!res.ok) throw new Error('db stats failed');
        const d = await res.json();
        set('settings-vectordb',
            `${d.milvus_backend} (${d.milvus_metric}, ${d.milvus_embedding_dim}-dim, ${d.milvus_index_type}) · ${d.milvus_host}`);
        set('settings-arthritis-model', d.models?.arthritis_predictor || '—');
        const ds = d.apd_dataset || {};
        set('settings-arthritis-dataset',
            `${ds.name || '—'} · ${(ds.total_records ?? 0).toLocaleString()} records, ${ds.total_features ?? '—'} features`);
    } catch (e) {
        ['settings-vectordb', 'settings-arthritis-model', 'settings-arthritis-dataset']
            .forEach(id => set(id, 'unavailable — backend not reachable'));
    }
}
document.addEventListener('DOMContentLoaded', loadSettingsInfo);

// The model column used to be filled from aspirational TARGET metrics. It now
// carries measured results only, so distinguish "never measured" from a bare dash —
// a blank cell reads as a display glitch, not as an untrained module.
function fmtModelMetric(row, value) {
    if (row.model_measured === false) {
        return '<span style="color:#8d8d8d;font-style:italic;font-weight:400">not measured</span>';
    }
    return value ? (value * 100).toFixed(1) + '%' : '—';
}

function displayValidationReport(data, box) {
    const summary = data.summary_table || [];
    const modules = data.modules || {};

    const summaryRows = summary.map(s => `
        <tr>
            <td><strong>${s.module || ''}</strong></td>
            <td style="font-size:0.8rem;color:#64748b">${s.reference || ''}</td>
            <td style="text-align:center">${s.expert_auc ? (s.expert_auc*100).toFixed(1)+'%' : '—'}</td>
            <td style="text-align:center;color:#0f62fe;font-weight:600">${fmtModelMetric(s, s.model_auc)}</td>
            <td style="text-align:center">${s.expert_accuracy ? (s.expert_accuracy*100).toFixed(1)+'%' : '—'}</td>
            <td style="text-align:center;color:#0f62fe;font-weight:600">${fmtModelMetric(s, s.model_accuracy)}</td>
        </tr>`
    ).join('');

    box.innerHTML = `
        <h4 style="margin-bottom:1rem"><i class="fa-solid fa-chart-line"></i> ${data.report_title || 'Clinical Validation Report'}</h4>
        <div style="padding:10px;background:#fff3cd;border:1px solid #ffc107;border-radius:6px;font-size:0.85rem;margin-bottom:1rem">
            <i class="fa-solid fa-triangle-exclamation"></i> ${data.disclaimer || ''}
        </div>

        <div class="card" style="padding:1rem;margin-bottom:1rem;overflow-x:auto">
            <h5 style="margin:0 0 12px">Performance Summary vs Published Cardiologist Benchmarks</h5>
            <table class="records-table" style="font-size:0.85rem">
                <thead>
                    <tr>
                        <th>Module</th><th>Reference Publication</th>
                        <th>Expert AUC</th><th>Model AUC</th>
                        <th>Expert Acc</th><th>Model Acc</th>
                    </tr>
                </thead>
                <tbody>${summaryRows}</tbody>
            </table>
        </div>

        ${Object.entries(modules).map(([key, mod]) => `
        <div class="card" style="padding:1rem;margin-bottom:12px">
            <h5 style="margin:0 0 8px">${mod.task_title || key}</h5>
            <div style="font-size:0.8rem;color:#64748b;margin-bottom:8px">
                <strong>Reference:</strong> ${mod.reference_publication || ''} |
                ${mod.doi ? `<a href="https://doi.org/${mod.doi}" target="_blank" style="color:var(--primary)">DOI: ${mod.doi}</a>` : ''}
            </div>
            <div style="font-size:0.85rem;color:#334155;line-height:1.6">${mod.validation_context || ''}</div>
        </div>`).join('')}`;
}

// ============================================================================
// ARTHRITIS: Show Dataset Source Info in EDA
// ============================================================================
const _origLoadEDA = window.loadEDA || loadEDA;
async function loadEDAWithDatasetInfo() {
    // Call the original loadEDA first
    await (typeof _origLoadEDA === 'function' ? _origLoadEDA() : Promise.resolve());

    // Add dataset source info
    const datasetInfoDiv = document.getElementById('arthritisDatasetInfo');
    if (!datasetInfoDiv) return;

    try {
        const res = await fetch(API_BASE + '/api/arthritis/eda');
        if (!res.ok) return;
        const data = await res.json();

        if (data.dataset_name && data.dataset_url) {
            datasetInfoDiv.innerHTML = `
                <div style="padding:12px;background:${data.is_large_dataset?'#e6f4ea':'#fff3e0'};border:1px solid ${data.is_large_dataset?'#198038':'#ff832b'};border-radius:8px;font-size:0.9rem">
                    <strong><i class="fa-solid fa-database"></i> Dataset Source:</strong>
                    ${data.dataset_name}<br>
                    <strong>Records:</strong> ${data.total_samples?.toLocaleString() || '—'} |
                    <strong>Features:</strong> ${data.total_features || '—'} |
                    <strong>Missing Data:</strong> ${data.low_missing_data ? '<span style="color:#198038">Low (≤20%)</span>' : '<span style="color:#ff832b">Present</span>'}
                    ${data.dataset_url ? `<br><a href="${data.dataset_url}" target="_blank" style="color:var(--primary)"><i class="fa-solid fa-external-link-alt"></i> View Dataset →</a>` : ''}
                    ${data.description ? `<br><small style="color:#64748b">${data.description}</small>` : ''}
                    ${data.is_large_dataset ? `<br><span style="color:#198038;font-size:0.8rem">✓ Large real dataset (${data.total_samples?.toLocaleString()} records) — low missing data</span>` :
                    `<br><span style="color:#ff832b;font-size:0.8rem">⚠ Small dataset (${data.total_samples} records) — kagglehub download for larger dataset</span>`}
                </div>`;
        }
    } catch (e) { console.warn('Dataset info load failed:', e); }
}
