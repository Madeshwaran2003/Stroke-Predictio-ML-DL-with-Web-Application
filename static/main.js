/**
 * Stroke Risk AI — main.js
 * Three.js 3D brain + form submission + LIME/SHAP canvas charts
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ═══════════════════════════════════════════════════════════════
// 1. THREE.JS — 3D Brain Scene
// ═══════════════════════════════════════════════════════════════

const canvas = document.getElementById('brain-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0, 3.5);

// Orbit controls (auto-rotate)
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.8;
controls.enableZoom = false;
controls.enablePan = false;
controls.minPolarAngle = Math.PI * 0.3;
controls.maxPolarAngle = Math.PI * 0.7;

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const rimLight1 = new THREE.DirectionalLight(0x00d2ff, 3.5);
rimLight1.position.set(-3, 2, 2);
scene.add(rimLight1);

const rimLight2 = new THREE.DirectionalLight(0x8a2be2, 2.5);
rimLight2.position.set(3, -1, -2);
scene.add(rimLight2);

const frontLight = new THREE.DirectionalLight(0xffffff, 1.2);
frontLight.position.set(0, 2, 4);
scene.add(frontLight);

// Point lights for inner glow
const pointGlow = new THREE.PointLight(0x00d2ff, 4, 5);
pointGlow.position.set(0, 0, 1);
scene.add(pointGlow);

// Particle system (neural dots around brain)
function createParticles() {
    const geo = new THREE.BufferGeometry();
    const count = 1200;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const r = 1.8 + Math.random() * 1.5;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
        color: 0x00d2ff,
        size: 0.012,
        transparent: true,
        opacity: 0.55,
        sizeAttenuation: true
    });
    return new THREE.Points(geo, mat);
}

const particles = createParticles();
scene.add(particles);

// Load the GLB brain model
let brainModel = null;
const loader = new GLTFLoader();
loader.load(
    '/brain_model',
    (gltf) => {
        brainModel = gltf.scene;

        // Auto-centre and scale
        const box = new THREE.Box3().setFromObject(brainModel);
        const size = box.getSize(new THREE.Vector3()).length();
        const centre = box.getCenter(new THREE.Vector3());
        brainModel.position.sub(centre);
        brainModel.scale.setScalar(2.2 / size);

        // Apply translucent neon material to every mesh
        brainModel.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshPhysicalMaterial({
                    color: 0x88ccff,
                    emissive: 0x0044aa,
                    emissiveIntensity: 0.35,
                    metalness: 0.1,
                    roughness: 0.25,
                    transmission: 0.45,
                    thickness: 0.8,
                    transparent: true,
                    opacity: 0.88,
                    envMapIntensity: 1.0,
                });
                child.castShadow = false;
            }
        });

        scene.add(brainModel);

        // Fade-in
        brainModel.traverse(c => { if (c.isMesh) c.material.opacity = 0; });
        let fadeT = 0;
        const fadeIn = setInterval(() => {
            fadeT += 0.03;
            brainModel.traverse(c => { if (c.isMesh) c.material.opacity = Math.min(fadeT * 0.88, 0.88); });
            if (fadeT >= 1) clearInterval(fadeIn);
        }, 30);
    },
    undefined,
    (err) => console.warn('Brain model could not be loaded:', err)
);

// Resize handler
function resizeRenderer() {
    const hero = document.getElementById('hero');
    const w = hero.clientWidth;
    const h = hero.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
resizeRenderer();
window.addEventListener('resize', resizeRenderer);

// Animation loop
const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Pulse the point light intensity
    pointGlow.intensity = 3.5 + Math.sin(t * 2) * 1.5;

    // Rotate particles slightly
    particles.rotation.y = t * 0.06;
    particles.rotation.x = Math.sin(t * 0.04) * 0.15;

    controls.update();
    renderer.render(scene, camera);
}
animate();

// Animate hero chips with stagger
document.querySelectorAll('.chip').forEach((chip, i) => {
    chip.style.animationDelay = `${i * 0.15 + 0.5}s`;
});

// Smooth scroll for hero CTA
document.getElementById('start-btn').addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('form-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
});


// ═══════════════════════════════════════════════════════════════
// 2. FEATURE CASUAL NAMES (for chart labels)
// ═══════════════════════════════════════════════════════════════

const CASUAL = {
    'gender': 'Gender',
    'age': 'Age',
    'hypertension': 'High Blood Pressure',
    'heart_disease': 'Heart Disease',
    'ever_married': 'Marital Status',
    'work_type': 'Work Type',
    'Residence_type': 'Residence Area',
    'avg_glucose_level': 'Blood Sugar (Glucose)',
    'bmi': 'BMI (Body Weight)',
    'smoking_status': 'Smoking Habit',
    'age*glucose': 'Age × Blood Sugar',
    'age*bmi': 'Age × BMI',
    'risk_score': 'Overall Risk Score',
    'age^2': 'Age² Amplifier',
    'glucose^2': 'Glucose² Amplifier',
    'log1p_age': 'Age (scaled)',
    'log1p_glucose': 'Glucose (scaled)',
    'log1p_bmi': 'BMI (scaled)',
};

function friendlyName(raw) {
    return CASUAL[raw] || raw.replace(/_/g, ' ').replace(/\*/g, '×');
}


// ═══════════════════════════════════════════════════════════════
// 3. CLIENT-SIDE HEALTH TEXT GENERATORS
// ═══════════════════════════════════════════════════════════════

function generateWhyText(data, prob, limeFeatures) {
    const age = parseFloat(data.age || 0);
    const glc = parseFloat(data.avg_glucose_level || 0);
    const bmi = parseFloat(data.bmi || 0);
    const hyp = parseInt(data.hypertension || 0);
    const hrt = parseInt(data.heart_disease || 0);
    const smk = data.smoking_status || '';

    let lines = [];
    const pct = (prob * 100).toFixed(1);
    lines.push(`🧠 <strong>Ensemble probability: ${pct}%</strong>`);
    lines.push('');

    if (age > 70) lines.push('🔴 <strong>Age ' + Math.round(age) + ' yrs</strong> — Stroke risk nearly doubles every decade after 55.');
    else if (age > 55) lines.push('🟡 <strong>Age ' + Math.round(age) + ' yrs</strong> — Risk rises meaningfully after 55.');

    if (glc > 200) lines.push('🔴 <strong>Glucose ' + glc.toFixed(0) + ' mg/dL</strong> — Very high (normal: 70–99). Please see a doctor.');
    else if (glc > 140) lines.push('🟡 <strong>Glucose ' + glc.toFixed(0) + ' mg/dL</strong> — Above normal. Target: under 100.');

    if (bmi >= 30) lines.push('🔴 <strong>BMI ' + bmi.toFixed(1) + '</strong> — Obese range. Raises BP and glucose risk.');
    else if (bmi >= 25) lines.push('🟡 <strong>BMI ' + bmi.toFixed(1) + '</strong> — Slightly above healthy (18.5–24.9).');

    if (hyp) lines.push('🔴 <strong>Hypertension</strong> — The single biggest modifiable stroke risk factor.');
    if (hrt) lines.push('🔴 <strong>Heart disease</strong> — Blood clots can travel to the brain. See your cardiologist.');

    if (smk === 'smokes') lines.push('🔴 <strong>Active smoker</strong> — Doubles stroke risk. Quitting cuts it within a year.');
    else if (smk === 'formerly smoked') lines.push('🟡 <strong>Former smoker</strong> — Risk keeps falling the longer you stay quit!');

    if (limeFeatures && limeFeatures.length > 0) {
        lines.push('');
        lines.push('🤖 <strong>AI focus (LIME top factors):</strong>');
        limeFeatures.slice(0, 5).forEach(f => {
            const dir = f.weight > 0 ? '🔺 raises' : '🔽 lowers';
            const strength = Math.abs(f.weight) > 0.08 ? 'strongly' : Math.abs(f.weight) > 0.03 ? 'noticeably' : 'slightly';
            lines.push(`  • <em>${friendlyName(f.feature)}</em> ${dir} risk ${strength}`);
        });
    }

    if (lines.length <= 2) lines.push('🟢 <strong>No major flags found</strong> — Keep up the healthy lifestyle!');

    return lines.join('<br>');
}

function generateRecsText(data) {
    const age = parseFloat(data.age || 0);
    const glc = parseFloat(data.avg_glucose_level || 0);
    const bmi = parseFloat(data.bmi || 0);
    const hyp = parseInt(data.hypertension || 0);
    const hrt = parseInt(data.heart_disease || 0);
    const smk = data.smoking_status || '';

    let recs = [];

    if (glc > 140) recs.push(`💊 <strong>Blood Sugar:</strong> Check fasting glucose daily · Eat whole grains, veggies, lean protein · 30 min walk daily · Ask about medication.`);
    if (hyp) recs.push(`❤️ <strong>Blood Pressure:</strong> Measure daily, keep a log · DASH diet (<2,300 mg salt/day) · Take BP meds every day without fail.`);
    if (bmi >= 25) recs.push(`🏃 <strong>Weight:</strong> 150 min moderate exercise per week · Losing 5–10% body weight cuts stroke risk significantly.`);
    if (smk === 'smokes') recs.push(`🚭 <strong>Quit Smoking:</strong> Biggest single win you can get · Ask about nicotine patches/gum · Risk drops 50% within one year.`);
    if (hrt) recs.push(`🫀 <strong>Heart Care:</strong> Never skip cardiology follow-ups · Take blood thinners exactly as prescribed.`);
    if (age > 55) recs.push(`👴 <strong>Screening (55+):</strong> Carotid ultrasound · Annual ECG for atrial fibrillation · Remember FAST: Face drooping, Arm weakness, Speech slurred → Call emergency services.`);

    if (!recs.length) recs.push(`🟢 <strong>Great Profile!</strong> Stay active (150+ min/week) · Balanced diet, no smoking, limit alcohol · Annual health check to stay on top of things.`);

    return recs.join('<br><br>');
}


// ═══════════════════════════════════════════════════════════════
// 4. CANVAS CHART RENDERERS
// ═══════════════════════════════════════════════════════════════

function drawGlobalImportanceChart(features) {
    const wrap = document.getElementById('global-imp-wrap');
    const canvas = document.getElementById('global-imp-canvas');
    const ctx = canvas.getContext('2d');

    const items = features.slice(0, 15);
    const BAR_H = 28, PAD = 10, LABEL_W = 210, CHART_W = 360;
    const W = LABEL_W + CHART_W + PAD * 3;
    const H = items.length * (BAR_H + PAD) + PAD * 3;

    const DPR = window.devicePixelRatio || 1;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, W, H);

    const maxVal = Math.max(...items.map(f => f.importance), 0.01);

    // Colour palette: top features get bright cyan, lower ones dim purple
    const palette = [
        '#00D2FF', '#00B8E0', '#009FC2', '#0086A3', '#006D85',
        '#8A5FCF', '#7A52B8', '#6B46A1', '#5B3A8A', '#4C2E73',
        '#6B8FCC', '#5A7AB5', '#49659E', '#385087', '#273B70'
    ];

    items.forEach((f, i) => {
        const y = PAD + i * (BAR_H + PAD);
        const barLen = (f.importance / maxVal) * (CHART_W - PAD * 2);
        const color = palette[Math.min(i, palette.length - 1)];

        // Rank medal for top 3
        const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i + 1}.`;
        ctx.fillStyle = '#8AABCC';
        ctx.font = i < 3 ? 'bold 13px "Outfit", sans-serif' : '600 12px "Outfit", sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(medal, 6, y + BAR_H / 2);

        // Feature label
        ctx.fillStyle = i < 3 ? '#E8F4FF' : '#9BB8D4';
        ctx.textAlign = 'right';
        ctx.fillText(friendlyName(f.feature), LABEL_W - 6, y + BAR_H / 2);

        // Bar
        const barX = LABEL_W + PAD;
        const grad = ctx.createLinearGradient(barX, 0, barX + barLen, 0);
        grad.addColorStop(0, color + 'AA');
        grad.addColorStop(1, color);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(barX, y + 2, Math.max(barLen, 3), BAR_H - 4, 4);
        ctx.fill();

        // Percentage text
        ctx.fillStyle = '#ffffff';
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText((f.importance * 100).toFixed(1) + '%', barX + barLen + 6, y + BAR_H / 2);
    });

    document.getElementById('global-imp-loading').style.display = 'none';
    wrap.style.display = 'block';
}

function drawLimeChart(features) {
    const wrap = document.getElementById('lime-chart-wrap');
    const canvas = document.getElementById('lime-canvas');
    const ctx = canvas.getContext('2d');

    const items = features.slice(0, 12);
    const BAR_H = 32, PAD = 14, LABEL_W = 200, CHART_W = 380;
    const W = LABEL_W + CHART_W + PAD * 2;
    const H = items.length * (BAR_H + PAD) + PAD * 2;

    canvas.width = W;
    canvas.height = H;
    canvas.style.width = '100%';
    canvas.style.height = 'auto';

    const DPR = window.devicePixelRatio || 1;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(DPR, DPR);

    // Background
    ctx.fillStyle = 'rgba(5, 10, 20, 0)';
    ctx.fillRect(0, 0, W, H);

    const maxAbs = Math.max(...items.map(f => Math.abs(f.weight)), 0.01);
    const scale = (CHART_W * 0.46) / maxAbs;

    items.forEach((f, i) => {
        const y = PAD + i * (BAR_H + PAD);
        const barLen = Math.abs(f.weight) * scale;
        const isPos = f.weight > 0;

        // Label
        ctx.fillStyle = '#c0cfe0';
        ctx.font = '600 12px "Outfit", sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(friendlyName(f.feature), LABEL_W - 8, y + BAR_H / 2);

        // Zero line at LABEL_W + CHART_W/2
        const zeroX = LABEL_W + PAD + CHART_W / 2;
        const barX = isPos ? zeroX : zeroX - barLen;

        // Bar gradient
        const grad = ctx.createLinearGradient(barX, 0, barX + barLen, 0);
        if (isPos) {
            grad.addColorStop(0, 'rgba(248,81,73,0.6)');
            grad.addColorStop(1, 'rgba(248,81,73,1)');
        } else {
            grad.addColorStop(0, 'rgba(46,160,67,1)');
            grad.addColorStop(1, 'rgba(46,160,67,0.6)');
        }
        ctx.fillStyle = grad;
        const rr = 4;
        ctx.beginPath();
        ctx.roundRect(barX, y, barLen || 2, BAR_H, rr);
        ctx.fill();

        // Value label
        ctx.fillStyle = '#ffffff';
        ctx.font = '500 11px "JetBrains Mono", monospace';
        ctx.textAlign = isPos ? 'left' : 'right';
        const valX = isPos ? zeroX + barLen + 5 : zeroX - barLen - 5;
        ctx.fillText((f.weight > 0 ? '+' : '') + f.weight.toFixed(3), valX, y + BAR_H / 2);

        // Zero line
        ctx.strokeStyle = 'rgba(200,200,200,0.25)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(zeroX, y - 2);
        ctx.lineTo(zeroX, y + BAR_H + 2);
        ctx.stroke();
    });

    document.getElementById('lime-loading').style.display = 'none';
    wrap.style.display = 'block';
}

function drawShapChart(features, baseVal, probability) {
    const wrap = document.getElementById('shap-chart-wrap');
    const canvas = document.getElementById('shap-canvas');
    const ctx = canvas.getContext('2d');

    const items = features.slice(0, 12);
    const BAR_H = 30, PAD = 12, LABEL_W = 180, CHART_W = 400;
    const W = LABEL_W + CHART_W + PAD * 3;
    const H = (items.length + 2) * (BAR_H + PAD) + PAD * 3;

    const DPR = window.devicePixelRatio || 1;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(DPR, DPR);

    ctx.clearRect(0, 0, W, H);

    // Build waterfall: cumulative from base
    let cumulative = baseVal;
    const bars = items.map(f => {
        const start = cumulative;
        cumulative += f.value;
        return { ...f, start, end: cumulative };
    });
    bars.push({ feature: 'Final prediction', value: 0, start: cumulative, end: probability });

    const allVals = bars.flatMap(b => [b.start, b.end]);
    const minV = Math.min(...allVals, 0);
    const maxV = Math.max(...allVals, 1);
    const range = maxV - minV || 1;
    const toX = (v) => LABEL_W + PAD + ((v - minV) / range) * (CHART_W - PAD * 2);

    bars.forEach((b, i) => {
        const y = PAD + i * (BAR_H + PAD);
        const x1 = toX(Math.min(b.start, b.end));
        const x2 = toX(Math.max(b.start, b.end));
        const barW = Math.max(x2 - x1, 3);

        // Label
        ctx.fillStyle = '#c0cfe0';
        ctx.font = '600 11px "Outfit", sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(i === bars.length - 1 ? 'Final Output' : friendlyName(b.feature), LABEL_W - 6, y + BAR_H / 2);

        // Bar
        const isPos = b.end > b.start;
        const isFinal = i === bars.length - 1;
        const grad = ctx.createLinearGradient(x1, 0, x1 + barW, 0);
        if (isFinal) {
            grad.addColorStop(0, 'rgba(255,210,0,0.7)');
            grad.addColorStop(1, 'rgba(255,180,0,1)');
        } else if (isPos) {
            grad.addColorStop(0, 'rgba(248,81,73,0.6)');
            grad.addColorStop(1, 'rgba(248,81,73,1)');
        } else {
            grad.addColorStop(0, 'rgba(46,160,67,1)');
            grad.addColorStop(1, 'rgba(46,160,67,0.6)');
        }
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x1, y, barW, BAR_H, 4);
        ctx.fill();

        // Value text
        ctx.fillStyle = '#ffffff';
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        const valStr = isFinal ? (probability * 100).toFixed(1) + '%' : (b.value > 0 ? '+' : '') + b.value.toFixed(3);
        ctx.fillText(valStr, x2 + 4, y + BAR_H / 2);
    });

    // Baseline label
    const baseX = toX(baseVal);
    ctx.strokeStyle = 'rgba(0, 210, 255, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(baseX, PAD);
    ctx.lineTo(baseX, H - PAD);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(0, 210, 255, 0.8)';
    ctx.font = '10px "JetBrains Mono"';
    ctx.textAlign = 'center';
    ctx.fillText(`base=${baseVal.toFixed(2)}`, baseX, PAD - 3);

    document.getElementById('shap-loading').style.display = 'none';
    wrap.style.display = 'block';
}


// ═══════════════════════════════════════════════════════════════
// 5. FORM SUBMISSION
// ═══════════════════════════════════════════════════════════════

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = document.getElementById('submit-btn');
    const errorMsg = document.getElementById('error-msg');

    btn.classList.add('loading');
    btn.disabled = true;
    errorMsg.textContent = '';

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        // Call /predict
        const resp = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await resp.json();

        if (!result.success) {
            errorMsg.textContent = 'Error: ' + result.error;
            return;
        }

        // Show results panel, hide form
        document.getElementById('form-section').style.display = 'none';
        const panel = document.getElementById('results-panel');
        panel.style.display = 'block';
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // ── Risk Gauge ─────────────────────────────────────────────────
        const prob = result.probability;
        const pct = Math.round(prob * 100);
        const circumference = 2 * Math.PI * 50; // r=50 → C≈314

        setTimeout(() => {
            const fill = document.getElementById('gauge-fill');
            const gaugePct = document.getElementById('gauge-pct');
            fill.style.transition = 'stroke-dasharray 1.6s cubic-bezier(0.16, 1, 0.3, 1)';
            fill.setAttribute('stroke-dasharray', `${(prob * circumference).toFixed(1)} ${circumference}`);
            gaugePct.textContent = pct + '%';

            if (result.prediction) {
                fill.style.stroke = '#f85149';
                gaugePct.style.fill = '#f85149';
            } else {
                fill.style.stroke = '#2ea043';
                gaugePct.style.fill = '#2ea043';
            }
        }, 100);

        // ── Result Badge ───────────────────────────────────────────────
        const badge = document.getElementById('result-badge');
        const icon = document.getElementById('result-icon');
        const label = document.getElementById('result-label');

        if (result.prediction) {
            badge.classList.add('danger');
            badge.classList.remove('safe');
            icon.textContent = '🔴';
            label.textContent = 'HIGH RISK — Stroke Detected';
        } else {
            badge.classList.add('safe');
            badge.classList.remove('danger');
            icon.textContent = '🟢';
            label.textContent = 'LOW RISK — No Stroke';
        }

        // ── Model Breakdown Tab ────────────────────────────────────────
        const grid = document.getElementById('models-grid');
        grid.innerHTML = '';
        const modelNames = {
            XGBoost: 'XGBoost', RandomForest: 'Random Forest',
            AttentionFNN: 'AttentionFNN', DualPathNet: 'DualPath+Leaf',
            ResGatedNet: 'ResGatedNet', AHIN: 'AHIN Ensemble'
        };
        for (const [key, val] of Object.entries(result.models)) {
            const p = val * 100;
            const col = p >= (result.threshold * 100) ? '#f85149' : '#2ea043';
            grid.innerHTML += `
                <div class="model-card">
                    <span class="model-name">${modelNames[key] || key}</span>
                    <div class="model-bar-track">
                        <div class="model-bar-fill" style="width:${p.toFixed(1)}%; background:${col};"></div>
                    </div>
                    <span class="model-pct" style="color:${col};">${p.toFixed(1)}%</span>
                </div>`;
        }

        // ── Why text (initial, without LIME) ──────────────────────────
        document.getElementById('why-text').innerHTML = generateWhyText(data, prob, []);
        document.getElementById('recs-text').innerHTML = generateRecsText(data);

        // ── /feature_importance (global) — fetch immediately, fast ────────
        fetch('/feature_importance')
            .then(r => r.json())
            .then(imp => {
                if (imp.success && imp.features && imp.features.length > 0) {
                    drawGlobalImportanceChart(imp.features);
                } else {
                    document.getElementById('global-imp-loading').innerHTML =
                        '<p class="chart-na">Global importance unavailable — models may still be loading</p>';
                }
            })
            .catch(() => {
                document.getElementById('global-imp-loading').innerHTML =
                    '<p class="chart-na">Could not load global importance</p>';
            });

        // ── /explain (LIME + SHAP) — async, slow ──────────────────────────
        const xScaled = result._X_scaled;
        if (xScaled) {
            fetch('/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ X_scaled: xScaled })
            })
                .then(r => r.json())
                .then(exp => {
                    if (!exp.success) return;

                    // Update why text with LIME factors
                    document.getElementById('why-text').innerHTML = generateWhyText(data, prob, exp.lime_features);

                    // Draw LIME chart
                    if (exp.lime_features && exp.lime_features.length > 0) {
                        drawLimeChart(exp.lime_features);
                    } else {
                        document.getElementById('lime-loading').innerHTML = '<p class="chart-na">LIME data unavailable — check Flask console for errors</p>';
                    }

                    // Draw SHAP chart
                    if (exp.shap_features && exp.shap_features.length > 0) {
                        drawShapChart(exp.shap_features, exp.shap_base, prob);
                    } else {
                        document.getElementById('shap-loading').innerHTML = '<p class="chart-na">SHAP data unavailable</p>';
                    }
                })
                .catch(() => {
                    document.getElementById('lime-loading').innerHTML = '<p class="chart-na">Could not compute LIME</p>';
                    document.getElementById('shap-loading').innerHTML = '<p class="chart-na">Could not compute SHAP</p>';
                });
        }


    } catch (err) {
        errorMsg.textContent = 'Connection error — ensure Flask is running.';
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});


// ═══════════════════════════════════════════════════════════════
// 6. TABS
// ═══════════════════════════════════════════════════════════════

document.getElementById('tabs-nav').addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn');
    if (!btn) return;
    const tab = btn.dataset.tab;

    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
});


// ═══════════════════════════════════════════════════════════════
// 7. RESET
// ═══════════════════════════════════════════════════════════════

document.getElementById('reset-btn').addEventListener('click', () => {
    document.getElementById('results-panel').style.display = 'none';
    const formSection = document.getElementById('form-section');
    formSection.style.display = 'block';
    formSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Reset charts
    document.getElementById('lime-loading').style.display = 'flex';
    document.getElementById('lime-chart-wrap').style.display = 'none';
    document.getElementById('shap-loading').style.display = 'flex';
    document.getElementById('shap-chart-wrap').style.display = 'none';

    // Reset gauge
    document.getElementById('gauge-fill').setAttribute('stroke-dasharray', '0 314');
    document.getElementById('gauge-pct').textContent = '0%';

    // Reset tabs
    document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === 0));
    document.querySelectorAll('.tab-content').forEach((c, i) => c.classList.toggle('active', i === 0));

    document.getElementById('prediction-form').reset();
    document.getElementById('error-msg').textContent = '';
});


// ═══════════════════════════════════════════════════════════════
// 8. PARTICLES IN HERO BACKGROUND (HTML DOM)
// ═══════════════════════════════════════════════════════════════

(function spawnParticleDots() {
    const container = document.getElementById('particles');
    if (!container) return;
    for (let i = 0; i < 60; i++) {
        const dot = document.createElement('div');
        dot.className = 'particle-dot';
        dot.style.cssText = `
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            width: ${1 + Math.random() * 2}px;
            height: ${1 + Math.random() * 2}px;
            animation-delay: ${Math.random() * 8}s;
            animation-duration: ${5 + Math.random() * 8}s;
        `;
        container.appendChild(dot);
    }
})();
