import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, send_file

# ═══════════════════════════════════════════════════════════════
# 1. ARCHITECTURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

class AttentionFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Softmax(dim=1)
        )
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.GELU(),
            nn.Linear(64, 1),          nn.Sigmoid()
        )
    def forward(self, x):
        return self.fnn(x * self.attention(x)).squeeze(1)

class DualPathNet(nn.Module):
    def __init__(self, n_clinical, n_lifestyle, n_extra):
        super().__init__()
        self.n_c = n_clinical
        self.n_l = n_lifestyle
        self.clinical_path = nn.Sequential(
            nn.Linear(n_clinical, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),          nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32),           nn.GELU()
        )
        self.lifestyle_path = nn.Sequential(
            nn.Linear(n_lifestyle, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),          nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32),           nn.GELU()
        )
        self.extra_path = nn.Sequential(
            nn.Linear(n_extra, 128),     nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.GELU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )
    def forward(self, x_full, x_leaf):
        c = self.clinical_path(x_full[:, :self.n_c])
        l = self.lifestyle_path(x_full[:, self.n_c:self.n_c+self.n_l])
        e = self.extra_path(x_leaf)
        return self.fusion(torch.cat([c, l, e], dim=1)).squeeze(1)


class GatedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU()
        )
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        g = self.gate(x)
        return g * self.transform(x) + (1 - g) * x

class ResGatedNet(nn.Module):
    def __init__(self, input_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.proj   = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU()
        )
        self.blocks = nn.Sequential(*[GatedBlock(hidden) for _ in range(n_blocks)])
        self.out    = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),      nn.Sigmoid()
        )
    def forward(self, x):
        return self.out(self.blocks(self.proj(x))).squeeze(1)


class DynamicFeatureWeighting(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim)
        )
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, x):
        scaled  = self.attn_net(x) / (self.temperature.abs() + 1e-6)
        weights = torch.softmax(scaled, dim=1)
        return x * weights, weights

class HealthInteractionAttention(nn.Module):
    def __init__(self, input_dim, top_k=20):
        super().__init__()
        self.top_k = top_k
        n_pairs    = input_dim * (input_dim - 1) // 2
        self.pair_attention = nn.Sequential(
            nn.Linear(n_pairs, n_pairs // 2), nn.ReLU(),
            nn.Linear(n_pairs // 2, n_pairs), nn.Softmax(dim=1)
        )
        self.compress = nn.Sequential(nn.Linear(n_pairs, top_k), nn.GELU())
        self.last_attn_weights = None
    def forward(self, x):
        B, D = x.shape
        pairs = [x[:, i] * x[:, j]
                 for i in range(D) for j in range(i+1, D)]
        pairs  = torch.stack(pairs, dim=1)
        attn_w = self.pair_attention(pairs)
        self.last_attn_weights = attn_w.detach()
        inter  = self.compress(pairs * attn_w)
        return torch.cat([x, inter], dim=1)

class AHIN(nn.Module):
    def __init__(self, input_dim, top_k=20, hidden=256):
        super().__init__()
        self.dfw = DynamicFeatureWeighting(input_dim)
        self.hia = HealthInteractionAttention(input_dim, top_k)
        combined_dim = input_dim + top_k
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        weighted_x, fw = self.dfw(x)
        interacted_x = self.hia(weighted_x)
        return self.classifier(interacted_x).squeeze(1), fw

# ═══════════════════════════════════════════════════════════════
# 2. FLASK SETUP & MODEL LOADING
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
device = torch.device('cpu')

MODELS_DIR = 'models'
label_encoders = {}
scaler = None
enc_leaf = None
leaf_scaler = None
poly = None

xgb_model = None
rf_model = None
model_A = None
model_B = None
model_C = None
ahin1 = None
ahin2 = None
ahin3 = None
ensemble_weights = None
ensemble_thresh = 0.5

# SHAP / LIME globals
shap_explainer = None
lime_explainer = None
ENG_FEATURE_NAMES = None

FEATURE_NAMES = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
CLINICAL_IDX  = [1, 7, 8, 2, 3]  # age, glucose, bmi, hypertension, heart_disease


def ensemble_predict_fn(X_input):
    """Black-box predict used by SHAP & LIME — returns (N, 2)."""
    X_input = np.array(X_input, dtype=np.float32)
    p_xgb = xgb_model.predict_proba(X_input)[:, 1]
    p_rf  = rf_model.predict_proba(X_input)[:, 1]

    t = torch.tensor(X_input, device=device)

    model_A.eval()
    with torch.no_grad():
        p_A = model_A(t).cpu().numpy()

    leaf_raw = xgb_model.apply(X_input)
    leaf_enc = enc_leaf.transform(leaf_raw).astype(np.float32)
    leaf_sc  = leaf_scaler.transform(leaf_enc).astype(np.float32)
    model_B.eval()
    with torch.no_grad():
        p_B = model_B(t, torch.tensor(leaf_sc, device=device)).cpu().numpy()

    model_C.eval()
    with torch.no_grad():
        p_C = model_C(t).cpu().numpy()

    ahin_preds = []
    for ahin_m in [ahin1, ahin2, ahin3]:
        ahin_m.eval()
        with torch.no_grad():
            p_, _ = ahin_m(t)
            ahin_preds.append(p_.cpu().numpy())
    cD1 = np.abs(ahin_preds[0] - 0.5)
    cD2 = np.abs(ahin_preds[1] - 0.5)
    cD3 = np.abs(ahin_preds[2] - 0.5)
    p_AHIN = (cD1*ahin_preds[0] + cD2*ahin_preds[1] + cD3*ahin_preds[2]) / (cD1+cD2+cD3+1e-8)

    wa, wb, wc, wd, we, wf = ensemble_weights
    ens = wa*p_xgb + wb*p_rf + wc*p_A + wd*p_B + we*p_C + wf*p_AHIN
    return np.column_stack([1 - ens, ens])


def load_models():
    global label_encoders, scaler, enc_leaf, leaf_scaler, poly
    global xgb_model, rf_model, model_A, model_B, model_C, ahin1, ahin2, ahin3
    global ensemble_weights, ensemble_thresh
    global shap_explainer, lime_explainer, ENG_FEATURE_NAMES

    if not os.path.exists(os.path.join(MODELS_DIR, 'ensemble_params.json')):
        print("Models not available yet!")
        return

    print("Loading models and preprocessors from disk...")
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    enc_leaf = joblib.load(os.path.join(MODELS_DIR, 'enc_leaf.pkl'))
    leaf_scaler = joblib.load(os.path.join(MODELS_DIR, 'leaf_scaler.pkl'))

    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))

    with open(os.path.join(MODELS_DIR, 'ensemble_params.json'), 'r') as f:
        params = json.load(f)
        ensemble_weights = params['weights']
        ensemble_thresh = params['threshold']

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly.fit(np.zeros((1, 5)))

    N_ENG = 33
    N_CLINICAL = 5
    N_LIFESTYLE = 5
    N_LEAF = len(enc_leaf.get_feature_names_out())

    model_A = AttentionFNN(N_ENG).to(device)
    model_A.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'model_A.pth'), map_location=device, weights_only=True))
    model_A.eval()

    model_B = DualPathNet(N_CLINICAL, N_LIFESTYLE, N_LEAF).to(device)
    model_B.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'model_B.pth'), map_location=device, weights_only=True))
    model_B.eval()

    model_C = ResGatedNet(N_ENG).to(device)
    model_C.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'model_C.pth'), map_location=device, weights_only=True))
    model_C.eval()

    ahin1 = AHIN(N_ENG, top_k=20, hidden=256).to(device)
    ahin1.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ahin1.pth'), map_location=device, weights_only=True))
    ahin1.eval()

    ahin2 = AHIN(N_ENG, top_k=20, hidden=256).to(device)
    ahin2.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ahin2.pth'), map_location=device, weights_only=True))
    ahin2.eval()

    ahin3 = AHIN(N_ENG, top_k=20, hidden=256).to(device)
    ahin3.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ahin3.pth'), map_location=device, weights_only=True))
    ahin3.eval()

    print("✅ Core models loaded!")

    # ── Load feature names ────────────────────────────────────────────────
    feat_path = os.path.join(MODELS_DIR, 'feature_names.json')
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            ENG_FEATURE_NAMES = json.load(f)
    else:
        ENG_FEATURE_NAMES = (
            FEATURE_NAMES +
            ['age*glucose', 'age*bmi', 'risk_score', 'age^2', 'glucose^2',
             'log1p_age', 'log1p_glucose', 'log1p_bmi'] +
            [f'poly_{i}' for i in range(poly.n_output_features_)]
        )

    # ── Auto-generate background sample if missing ───────────────────────
    bg_path = os.path.join(MODELS_DIR, 'X_train_bg.npy')
    if not os.path.exists(bg_path):
        print("⚙️  X_train_bg.npy not found — auto-generating from scaler statistics...")
        try:
            n_features = len(ENG_FEATURE_NAMES)
            np.random.seed(42)
            # Generate samples in scaled space (mean=0, std=1 is already what scaler produces)
            X_bg_auto = np.random.randn(200, n_features).astype(np.float32)
            np.save(bg_path, X_bg_auto)
            print(f"✅ Auto-generated background sample: {X_bg_auto.shape} → {bg_path}")
        except Exception as e:
            print(f"⚠️  Could not auto-generate background: {e}")

    # ── LIME explainer ────────────────────────────────────────────────────
    if os.path.exists(bg_path):
        try:
            import lime
            import lime.lime_tabular
            X_bg = np.load(bg_path)
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_bg,
                feature_names=ENG_FEATURE_NAMES,
                class_names=['No Stroke', 'Stroke'],
                mode='classification',
                random_state=42,
                discretize_continuous=True
            )
            print("✅ LIME explainer ready")
        except Exception as e:
            print(f"⚠️  LIME setup skipped: {e}")
    else:
        print("⚠️  LIME disabled — background sample unavailable.")

    # ── SHAP KernelExplainer ──────────────────────────────────────────────
    if os.path.exists(bg_path):
        try:
            import shap
            X_bg = np.load(bg_path)
            shap_bg = pd.DataFrame(X_bg[:30], columns=ENG_FEATURE_NAMES)
            shap_explainer = shap.KernelExplainer(ensemble_predict_fn, shap_bg)
            print("✅ SHAP explainer ready")
        except Exception as e:
            print(f"⚠️  SHAP setup skipped: {e}")

    print("✅ All models and explainers loaded successfully!")


@app.before_request
def before_first_request_func():
    if xgb_model is None:
        load_models()


# ═══════════════════════════════════════════════════════════════
# 3. ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/brain_model')
def brain_model():
    """Serves the .glb 3D brain model file."""
    glb_path = os.path.join(os.path.dirname(__file__), 'human_brain_cerebrum__brainstem.glb')
    return send_file(glb_path, mimetype='model/gltf-binary')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        x_raw = []
        for feat in FEATURE_NAMES:
            val = data.get(feat)
            if feat in label_encoders:
                if val is None or str(val) not in label_encoders[feat].classes_:
                    val = "Unknown"
                if val not in label_encoders[feat].classes_:
                    val = label_encoders[feat].classes_[0]
                x_raw.append(label_encoders[feat].transform([str(val)])[0])
            else:
                x_raw.append(float(val) if val is not None else 0.0)

        X_all = np.array([x_raw], dtype=np.float32)

        age_idx, glc_idx, bmi_idx, hyp_idx, hrt_idx = CLINICAL_IDX
        age_v = X_all[:, age_idx]
        glc_v = X_all[:, glc_idx]
        bmi_v = X_all[:, bmi_idx]
        hyp_v = X_all[:, hyp_idx]
        hrt_v = X_all[:, hrt_idx]

        X_poly = poly.transform(X_all[:, CLINICAL_IDX]).astype(np.float32)

        X_eng = np.hstack([
            X_all,
            (age_v * glc_v).reshape(-1, 1),
            (age_v * bmi_v).reshape(-1, 1),
            (age_v + glc_v + bmi_v + hyp_v*10 + hrt_v*10).reshape(-1, 1),
            (age_v**2).reshape(-1, 1),
            (glc_v**2).reshape(-1, 1),
            np.log1p(age_v).reshape(-1, 1),
            np.log1p(glc_v).reshape(-1, 1),
            np.log1p(bmi_v).reshape(-1, 1),
            X_poly
        ]).astype(np.float32)

        X_tr = scaler.transform(X_eng).astype(np.float32)

        xgb_proba = float(xgb_model.predict_proba(X_tr)[:, 1][0])

        leaf_raw = xgb_model.apply(X_tr)
        leaf_enc = enc_leaf.transform(leaf_raw).astype(np.float32)
        leaf_scaled = leaf_scaler.transform(leaf_enc).astype(np.float32)

        rf_proba = float(rf_model.predict_proba(X_tr)[:, 1][0])

        t_X_tr = torch.tensor(X_tr, device=device)
        t_leaf = torch.tensor(leaf_scaled, device=device)

        with torch.no_grad():
            proba_a = float(model_A(t_X_tr).cpu().numpy()[0])
            proba_b = float(model_B(t_X_tr, t_leaf).cpu().numpy()[0])
            proba_c = float(model_C(t_X_tr).cpu().numpy()[0])

            p_d1 = float(ahin1(t_X_tr)[0].cpu().numpy()[0])
            p_d2 = float(ahin2(t_X_tr)[0].cpu().numpy()[0])
            p_d3 = float(ahin3(t_X_tr)[0].cpu().numpy()[0])

            c1 = abs(p_d1 - 0.5)
            c2 = abs(p_d2 - 0.5)
            c3 = abs(p_d3 - 0.5)
            cs = c1 + c2 + c3 + 1e-8
            proba_ahin = (c1*p_d1 + c2*p_d2 + c3*p_d3) / cs

        wa, wb, wc, wd, we, wf = ensemble_weights
        final_proba = (wa*xgb_proba + wb*rf_proba + wc*proba_a + wd*proba_b + we*proba_c + wf*proba_ahin)

        is_stroke = final_proba >= ensemble_thresh

        return jsonify({
            'success': True,
            'prediction': bool(is_stroke),
            'probability': float(final_proba),
            'threshold': float(ensemble_thresh),
            'models': {
                'XGBoost': xgb_proba,
                'RandomForest': rf_proba,
                'AttentionFNN': proba_a,
                'DualPathNet': proba_b,
                'ResGatedNet': proba_c,
                'AHIN': float(proba_ahin)
            },
            # pass X_scaled back for /explain if needed client stores it
            '_X_scaled': X_tr[0].tolist()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/explain', methods=['POST'])
def explain():
    """Computes LIME + SHAP explanations for a single instance.
    Expects JSON: { X_scaled: [...33 floats...] }
    Returns: { lime_features: [...], shap_features: [...], shap_base: float }
    """
    try:
        data = request.json
        X_scaled = np.array(data['X_scaled'], dtype=np.float32).reshape(1, -1)

        response = {'success': True, 'lime_features': [], 'shap_features': [], 'shap_base': 0.5}

        # ── LIME ──────────────────────────────────────────────────────────
        if lime_explainer is not None:
            try:
                exp = lime_explainer.explain_instance(
                    X_scaled[0],
                    ensemble_predict_fn,
                    num_features=12,
                    num_samples=500,
                    labels=(1,)
                )
                lime_list = exp.as_list(label=1)
                import re
                def parse_feat(desc):
                    tokens = re.findall(r'[a-zA-Z_][a-zA-Z_0-9*^()]+', desc)
                    return tokens[0] if tokens else desc.strip()

                response['lime_features'] = [
                    {'feature': parse_feat(fd), 'raw_desc': fd, 'weight': float(w)}
                    for fd, w in lime_list
                ]
            except Exception as e:
                print(f"LIME error: {e}")

        # ── SHAP ──────────────────────────────────────────────────────────
        if shap_explainer is not None:
            try:
                sv = shap_explainer.shap_values(X_scaled, nsamples=80)
                if isinstance(sv, list):
                    sv_stroke = sv[1][0]
                    base_val  = float(shap_explainer.expected_value[1])
                else:
                    sv_stroke = sv[0]
                    base_val  = float(shap_explainer.expected_value)

                # Top 14 by |value|
                idx_sorted = np.argsort(np.abs(sv_stroke))[-14:][::-1]
                response['shap_features'] = [
                    {'feature': ENG_FEATURE_NAMES[i], 'value': float(sv_stroke[i])}
                    for i in idx_sorted
                ]
                response['shap_base'] = base_val
            except Exception as e:
                print(f"SHAP error: {e}")

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """
    Returns global feature importance from XGBoost (gain) + Random Forest.
    This answers: "which features matter most for stroke prediction overall?"
    """
    try:
        if xgb_model is None or ENG_FEATURE_NAMES is None:
            return jsonify({'success': False, 'error': 'Models not loaded'}), 503

        results = []

        # ── XGBoost gain importance ────────────────────────────────────────
        try:
            xgb_imp = xgb_model.get_booster().get_score(importance_type='gain')
            # XGBoost uses f0, f1, ... feature names → map to ENG_FEATURE_NAMES
            xgb_vals = {}
            for k, v in xgb_imp.items():
                # 'f0' → index 0, 'f12' → index 12, etc.
                try:
                    idx = int(k.replace('f', ''))
                    if idx < len(ENG_FEATURE_NAMES):
                        xgb_vals[ENG_FEATURE_NAMES[idx]] = float(v)
                except ValueError:
                    pass
            # Normalise to 0–1
            if xgb_vals:
                max_xgb = max(xgb_vals.values())
                xgb_vals = {k: v / max_xgb for k, v in xgb_vals.items()}
        except Exception as e:
            print(f"XGB importance warning: {e}")
            xgb_vals = {}

        # ── Random Forest importance ───────────────────────────────────────
        try:
            rf_imp_raw = rf_model.feature_importances_
            rf_vals = {}
            for i, v in enumerate(rf_imp_raw):
                if i < len(ENG_FEATURE_NAMES):
                    rf_vals[ENG_FEATURE_NAMES[i]] = float(v)
            # Normalise
            if rf_vals:
                max_rf = max(rf_vals.values())
                rf_vals = {k: v / max_rf for k, v in rf_vals.items()}
        except Exception as e:
            print(f"RF importance warning: {e}")
            rf_vals = {}

        # ── Merge: average of XGB and RF (whichever are available) ────────
        all_features = set(list(xgb_vals.keys()) + list(rf_vals.keys()))
        for feat in all_features:
            xv = xgb_vals.get(feat, 0.0)
            rv = rf_vals.get(feat, 0.0)
            # If only one source, use that; if both, average
            if xv > 0 and rv > 0:
                score = (xv + rv) / 2
            else:
                score = xv or rv
            results.append({'feature': feat, 'importance': round(score, 5)})

        # Sort descending, return top 20
        results.sort(key=lambda x: x['importance'], reverse=True)
        results = results[:20]

        return jsonify({
            'success': True,
            'features': results,
            'sources': {
                'xgb_count': len(xgb_vals),
                'rf_count': len(rf_vals)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)

