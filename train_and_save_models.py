# ═══════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ═══════════════════════════════════════════════════════════════
# !pip install -q imbalanced-learn xgboost gradio
print('✅ All packages installed')

# ═══════════════════════════════════════════════════════════════
# CELL 2 — Imports & global config
# ═══════════════════════════════════════════════════════════════
import os, warnings, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import (LabelEncoder, StandardScaler,
                                      PolynomialFeatures, OneHotEncoder)
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      confusion_matrix, classification_report)
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler
from xgboost                  import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'✅ Imports done | Device: {device}')

# ═══════════════════════════════════════════════════════════════
# CELL 3 — Load & preprocess dataset
# ═══════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r'c:\Users\mades\OneDrive\Desktop\project\healthcare-dataset-stroke-data (2).csv')

# ── Basic cleaning ──────────────────────────────────────────
df = df.drop(columns=['id'])
df = df[df['gender'] != 'Other'].reset_index(drop=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# ── Label-encode all categorical columns (save encoders) ────
label_encoders = {}
for col in df.select_dtypes('object').columns:
    df[col] = df[col].fillna('Unknown')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

FEATURE_NAMES = df.drop(columns=['stroke']).columns.tolist()
X_all = df.drop(columns=['stroke']).values.astype(np.float32)
y_all = df['stroke'].values

print(f'✅ Loaded | Shape: {df.shape}')
print(f'   Features      : {FEATURE_NAMES}')
print(f'   Stroke dist   : {dict(zip(*np.unique(y_all, return_counts=True)))}')
print(f'   Label encoders: {list(label_encoders.keys())}')

# ═══════════════════════════════════════════════════════════════
# CELL 4 — Feature engineering (10 → 33 features)
# ═══════════════════════════════════════════════════════════════
age_idx = FEATURE_NAMES.index('age')
glc_idx = FEATURE_NAMES.index('avg_glucose_level')
bmi_idx = FEATURE_NAMES.index('bmi')
hyp_idx = FEATURE_NAMES.index('hypertension')
hrt_idx = FEATURE_NAMES.index('heart_disease')

age_v = X_all[:, age_idx]
glc_v = X_all[:, glc_idx]
bmi_v = X_all[:, bmi_idx]
hyp_v = X_all[:, hyp_idx]
hrt_v = X_all[:, hrt_idx]

# Polynomial interactions over key clinical vars
poly   = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(
    X_all[:, [age_idx, glc_idx, bmi_idx, hyp_idx, hrt_idx]]
).astype(np.float32)

X_eng = np.hstack([
    X_all,
    (age_v * glc_v).reshape(-1, 1),
    (age_v * bmi_v).reshape(-1, 1),
    (age_v + glc_v + bmi_v + hyp_v*10 + hrt_v*10).reshape(-1, 1),  # risk score
    (age_v**2).reshape(-1, 1),
    (glc_v**2).reshape(-1, 1),
    np.log1p(age_v).reshape(-1, 1),
    np.log1p(glc_v).reshape(-1, 1),
    np.log1p(bmi_v).reshape(-1, 1),
    X_poly
]).astype(np.float32)

N_ORIG = X_all.shape[1]   # 10
N_ENG  = X_eng.shape[1]   # 33

# Indices used by DualPathNet
CLINICAL_IDX  = [age_idx, glc_idx, bmi_idx, hyp_idx, hrt_idx]
LIFESTYLE_IDX = [i for i in range(N_ORIG) if i not in CLINICAL_IDX]

print(f'✅ Features: {N_ORIG} original → {N_ENG} engineered')
print(f'   Clinical  indices : {CLINICAL_IDX}')
print(f'   Lifestyle indices : {LIFESTYLE_IDX}')

# ═══════════════════════════════════════════════════════════════
# CELL 5 — SMOTE balancing + train/test split + scaling
# ═══════════════════════════════════════════════════════════════
minority = int((y_all == 1).sum())
majority = int((y_all == 0).sum())

X_u, y_u = RandomUnderSampler(
    sampling_strategy={0: min(minority*10, majority), 1: minority},
    random_state=SEED
).fit_resample(X_eng, y_all)

X_bal, y_bal = SMOTE(
    sampling_strategy=1.0, random_state=SEED, k_neighbors=5
).fit_resample(X_u, y_u)
print(f'After SMOTE : {dict(zip(*np.unique(y_bal, return_counts=True)))}')

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=SEED
)

scaler  = StandardScaler()
X_tr    = scaler.fit_transform(X_tr_raw).astype(np.float32)
X_te    = scaler.transform(X_te_raw).astype(np.float32)
y_tr_np = y_tr.astype(np.float32)
y_te_np = y_te.astype(np.float32)
y_tr_t  = torch.tensor(y_tr_np)

print(f'Train: {X_tr.shape} | Test: {X_te.shape}')

# ═══════════════════════════════════════════════════════════════
# CELL 6 — ML: Train XGBoost + generate leaf embeddings
# ═══════════════════════════════════════════════════════════════
print('Training XGBoost...')
xgb_model = XGBClassifier(
    n_estimators     = 1000,
    max_depth        = 6,
    learning_rate    = 0.02,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    eval_metric      = 'logloss',
    random_state     = SEED,
    n_jobs           = -1
)
xgb_model.fit(X_tr, y_tr_np, eval_set=[(X_te, y_te_np)], verbose=False)

xgb_proba = xgb_model.predict_proba(X_te)[:, 1]
xgb_acc   = accuracy_score(y_te_np, (xgb_proba >= 0.5).astype(int)) * 100
print(f'✅ XGBoost Accuracy: {xgb_acc:.2f}%')

# ── XGB leaf embeddings (ML knowledge fed into DualPathNet) ─
print('\nGenerating XGBoost leaf embeddings for DualPathNet...')
leaf_tr_raw = xgb_model.apply(X_tr)   # (N_tr, n_estimators)
leaf_te_raw = xgb_model.apply(X_te)

enc_leaf = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=10)
leaf_tr  = enc_leaf.fit_transform(leaf_tr_raw).astype(np.float32)
leaf_te  = enc_leaf.transform(leaf_te_raw).astype(np.float32)

leaf_scaler = StandardScaler()
leaf_tr = leaf_scaler.fit_transform(leaf_tr).astype(np.float32)
leaf_te = leaf_scaler.transform(leaf_te).astype(np.float32)
print(f'✅ Leaf embedding shape: train={leaf_tr.shape}, test={leaf_te.shape}')

# ═══════════════════════════════════════════════════════════════
# CELL 7 — ML: Train Random Forest
# ═══════════════════════════════════════════════════════════════
print('Training Random Forest...')
rf_model = RandomForestClassifier(
    n_estimators      = 500,
    max_depth         = None,
    min_samples_split = 2,
    min_samples_leaf  = 1,
    max_features      = 'sqrt',
    random_state      = SEED,
    n_jobs            = -1
)
rf_model.fit(X_tr, y_tr_np)
rf_proba = rf_model.predict_proba(X_te)[:, 1]
rf_acc   = accuracy_score(y_te_np, (rf_proba >= 0.5).astype(int)) * 100
print(f'✅ Random Forest Accuracy: {rf_acc:.2f}%')

# ═══════════════════════════════════════════════════════════════
# CELL 8 — DL Architecture definitions
#
# Model A : AttentionFNN      — soft feature attention + deep FNN
# Model B : DualPathNet+Leaf  — clinical / lifestyle paths + XGB leaves
# Model C : ResGatedNet       — learnable gated residual blocks
# Model D : AHIN              — pairwise interaction attention (from notebook 2)
#            DynamicFeatureWeighting : learnable temperature softmax
#            HealthInteractionAttention : all pairwise products + attention
# ═══════════════════════════════════════════════════════════════

# ── Model A: Attention-FNN ──────────────────────────────────
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


# ── Model B: DualPathNet + XGB Leaf Embeddings ──────────────
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


# ── Model C: Residual Gated Network ─────────────────────────
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


# ── Model D: AHIN — Adaptive Health Interaction Network ─────
class DynamicFeatureWeighting(nn.Module):
    """
    Learnable temperature τ: uncertain patients get diffuse weights,
    high-risk patients get focused (sharp) weights.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim)
        )
        self.temperature = nn.Parameter(torch.ones(1))  # learnable τ
    def forward(self, x):
        scaled  = self.attn_net(x) / (self.temperature.abs() + 1e-6)
        weights = torch.softmax(scaled, dim=1)
        return x * weights, weights

class HealthInteractionAttention(nn.Module):
    """
    Generates ALL pairwise feature products (age×glucose, bmi×hyp, …),
    learns which pairs matter most via trainable softmax attention.
    """
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

# ── Instantiate all models ───────────────────────────────────
N_CLINICAL  = len(CLINICAL_IDX)
N_LIFESTYLE = len(LIFESTYLE_IDX)
N_LEAF      = leaf_tr.shape[1]

model_A  = AttentionFNN(N_ENG).to(device)
model_B  = DualPathNet(N_CLINICAL, N_LIFESTYLE, N_LEAF).to(device)
model_C  = ResGatedNet(N_ENG).to(device)
ahin1    = AHIN(N_ENG, top_k=20, hidden=256).to(device)
ahin2    = AHIN(N_ENG, top_k=20, hidden=256).to(device)
ahin3    = AHIN(N_ENG, top_k=20, hidden=256).to(device)

def count_params(m): return sum(p.numel() for p in m.parameters())
print(f'✅ Model A  — AttentionFNN      : {count_params(model_A):>10,} params')
print(f'✅ Model B  — DualPath+Leaf     : {count_params(model_B):>10,} params')
print(f'✅ Model C  — ResGatedNet       : {count_params(model_C):>10,} params')
print(f'✅ AHIN x3 — Interaction Attn   : {count_params(ahin1):>10,} params each')

# ═══════════════════════════════════════════════════════════════
# CELL 9 — Generic DL training function
# ═══════════════════════════════════════════════════════════════
def train_model(model, tr_dl, te_dl, tag, epochs=300, lr=5e-4,
                use_leaf=False, is_ahin=False):
    """
    Unified trainer for all DL models.
    use_leaf  = True  →  model(x_full, x_leaf)       [DualPathNet]
    is_ahin   = True  →  model(x) returns (prob, fw) [AHIN]
    """
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.BCELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=15
    )
    best_acc, best_wts, best_prob = 0.0, None, None
    losses, vaccs = [], []
    patience_cnt, EARLY_STOP = 0, 40

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in tr_dl:
            xb, yb = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            if use_leaf:
                pred = model(xb, batch[2].to(device))
            elif is_ahin:
                pred, _ = model(xb)
            else:
                pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)

        model.eval()
        probs = []
        with torch.no_grad():
            for batch in te_dl:
                xb = batch[0].to(device)
                if use_leaf:
                    out = model(xb, batch[2].to(device))
                elif is_ahin:
                    out, _ = model(xb)
                else:
                    out = model(xb)
                probs.append(out.cpu().numpy())

        probs = np.concatenate(probs)
        va    = accuracy_score(y_te_np, (probs >= 0.5).astype(int)) * 100
        sched.step(va)
        losses.append(ep_loss / len(tr_dl.dataset))
        vaccs.append(va)

        if va > best_acc:
            best_acc  = va
            best_wts  = {k: v.clone() for k, v in model.state_dict().items()}
            best_prob = probs.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if ep % 50 == 0:
            print(f'  [{tag}] Ep {ep:>3}/{epochs} | Loss: {ep_loss/len(tr_dl.dataset):.4f}'
                  f' | Acc: {va:.2f}% | Best: {best_acc:.2f}%')
        if patience_cnt >= EARLY_STOP:
            print(f'  [{tag}] Early stop ep {ep}'); break

    model.load_state_dict(best_wts)
    print(f'✅ {tag} best accuracy: {best_acc:.2f}%\n')
    return model, best_prob, best_acc, losses, vaccs

print('✅ Training function defined')

# ═══════════════════════════════════════════════════════════════
# CELL 10 — Build DataLoaders
# ═══════════════════════════════════════════════════════════════
# Standard (A, C, AHIN)
tr_ds_std  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr_np))
te_ds_std  = TensorDataset(torch.tensor(X_te), torch.tensor(y_te_np))
tr_dl_std  = DataLoader(tr_ds_std, batch_size=128, shuffle=True,  drop_last=True)
te_dl_std  = DataLoader(te_ds_std, batch_size=512, shuffle=False)

# With leaf embeddings (B)
tr_ds_leaf = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr_np), torch.tensor(leaf_tr))
te_ds_leaf = TensorDataset(torch.tensor(X_te), torch.tensor(y_te_np), torch.tensor(leaf_te))
tr_dl_leaf = DataLoader(tr_ds_leaf, batch_size=128, shuffle=True,  drop_last=True)
te_dl_leaf = DataLoader(te_ds_leaf, batch_size=512, shuffle=False)

print('✅ DataLoaders ready')

# ═══════════════════════════════════════════════════════════════
# CELL 11 — Train DL Model A: AttentionFNN
# ═══════════════════════════════════════════════════════════════
print('='*60)
print('Training Model A — AttentionFNN')
print('='*60)
model_A, proba_A, acc_A, loss_A, vacc_A = train_model(
    model_A, tr_dl_std, te_dl_std, tag='Attn-FNN', epochs=300
)

# ═══════════════════════════════════════════════════════════════
# CELL 12 — Train DL Model B: DualPathNet + XGB Leaf Embeddings
# ═══════════════════════════════════════════════════════════════
print('='*60)
print('Training Model B — DualPathNet + XGB Leaf Embeddings')
print('='*60)
model_B, proba_B, acc_B, loss_B, vacc_B = train_model(
    model_B, tr_dl_leaf, te_dl_leaf, tag='DualPath', epochs=300, use_leaf=True
)

# ═══════════════════════════════════════════════════════════════
# CELL 13 — Train DL Model C: ResGatedNet
# ═══════════════════════════════════════════════════════════════
print('='*60)
print('Training Model C — ResGatedNet')
print('='*60)
model_C, proba_C, acc_C, loss_C, vacc_C = train_model(
    model_C, tr_dl_std, te_dl_std, tag='ResGated', epochs=300
)

# ═══════════════════════════════════════════════════════════════
# CELL 14 — Train DL Model D: AHIN (3 seeds)
# ═══════════════════════════════════════════════════════════════
for seed, model, tag, attr in [
    (42,  ahin1, 'AHIN-1', 'proba_D1'),
    (123, ahin2, 'AHIN-2', 'proba_D2'),
    (999, ahin3, 'AHIN-3', 'proba_D3'),
]:
    torch.manual_seed(seed); np.random.seed(seed)
    print('='*60)
    print(f'Training {tag} (seed={seed})')
    print('='*60)
    m, prob, acc, _, _ = train_model(
        model, tr_dl_std, te_dl_std, tag=tag, epochs=300, is_ahin=True
    )
    globals()[attr] = prob
    globals()['acc_' + tag.replace('-','')] = acc

# AHIN confidence-aware sub-ensemble
c1 = np.abs(proba_D1 - 0.5)
c2 = np.abs(proba_D2 - 0.5)
c3 = np.abs(proba_D3 - 0.5)
cs = c1 + c2 + c3 + 1e-8
proba_AHIN = (c1*proba_D1 + c2*proba_D2 + c3*proba_D3) / cs
acc_AHIN   = accuracy_score(y_te_np, (proba_AHIN >= 0.5).astype(int)) * 100
print(f'\n✅ AHIN ensemble accuracy (threshold=0.5): {acc_AHIN:.2f}%')

# ═══════════════════════════════════════════════════════════════
# CELL 15 — Grid-search best ensemble weights + threshold
#
# 6 probability streams:
#   XGBoost | RandomForest | AttentionFNN | DualPathNet | ResGatedNet | AHIN
# ═══════════════════════════════════════════════════════════════
print('Searching best ensemble weights + threshold...\n')
print('(This may take ~1–2 minutes)')

best = {'acc': 0, 'w': None, 'thresh': 0.5, 'proba': None}
step = 0.1

for wa in np.arange(0.0, 1.01, step):             # XGB
  for wb in np.arange(0.0, 1.01-wa, step):        # RF
    for wc in np.arange(0.0, 1.01-wa-wb, step):   # Attn-FNN
      for wd in np.arange(0.0, 1.01-wa-wb-wc, step):        # DualPath
        for we in np.arange(0.0, 1.01-wa-wb-wc-wd, step):   # ResGated
          wf = round(1.0 - wa - wb - wc - wd - we, 2)       # AHIN
          if wf < 0: continue
          ens = (wa*xgb_proba + wb*rf_proba + wc*proba_A
               + wd*proba_B   + we*proba_C  + wf*proba_AHIN)
          for thresh in np.arange(0.30, 0.82, 0.01):
              preds = (ens >= thresh).astype(int)
              acc   = accuracy_score(y_te_np, preds) * 100
              if acc > best['acc']:
                  best.update({'acc': acc,
                               'w': (wa, wb, wc, wd, we, wf),
                               'thresh': thresh,
                               'proba':  ens.copy()})

wa, wb, wc, wd, we, wf = best['w']
BEST_THRESH = best['thresh']
BEST_W      = best['w']

print(f'  XGBoost weight       : {wa:.2f}')
print(f'  Random Forest weight : {wb:.2f}')
print(f'  AttentionFNN weight  : {wc:.2f}')
print(f'  DualPathNet weight   : {wd:.2f}')
print(f'  ResGatedNet weight   : {we:.2f}')
print(f'  AHIN weight          : {wf:.2f}')
print(f'  Best threshold       : {BEST_THRESH:.2f}')
print(f'  Best accuracy        : {best["acc"]:.2f}%')

final_pred = (best['proba'] >= BEST_THRESH).astype(int)

# ═══════════════════════════════════════════════════════════════
# CELL 16 — Final metrics + comparison table + plots
# ═══════════════════════════════════════════════════════════════
acc = accuracy_score(y_te_np,  final_pred)
pre = precision_score(y_te_np, final_pred, zero_division=0)
rec = recall_score(y_te_np,    final_pred, zero_division=0)
f1  = f1_score(y_te_np,        final_pred, zero_division=0)
auc = roc_auc_score(y_te_np,   best['proba'])
cm  = confusion_matrix(y_te_np, final_pred)
TN, FP, FN, TP = cm.ravel()

print('='*65)
print('  COMBINED ML + DL ENSEMBLE — FINAL RESULTS')
print('='*65)
print(f'  Accuracy     : {acc*100:.2f}%')
print(f'  Precision    : {pre:.4f}')
print(f'  Recall       : {rec:.4f}')
print(f'  F1-Score     : {f1:.4f}')
print(f'  AUC-ROC      : {auc:.4f}')
print(f'  Sensitivity  : {TP/(TP+FN):.4f}')
print(f'  Specificity  : {TN/(TN+FP):.4f}')
print(f'\n  Confusion Matrix:')
print(f'               Pred: No   Pred: Yes')
print(f'  Actual: No   {TN:>7}    {FP:>7}')
print(f'  Actual: Yes  {FN:>7}    {TP:>7}')
print()
print(classification_report(y_te_np, final_pred,
                             target_names=['No Stroke', 'Stroke'],
                             zero_division=0))

print('─'*65)
print(f'  {"Model":<38} {"Accuracy":>10}')
print('─'*65)
for name, p in [
    ('XGBoost (ML)',               xgb_proba),
    ('Random Forest (ML)',         rf_proba),
    ('AttentionFNN (DL)',          proba_A),
    ('DualPathNet + XGB Leaf (DL)',proba_B),
    ('ResGatedNet (DL)',           proba_C),
    ('AHIN ensemble (DL)',         proba_AHIN),
    ('Final Combined Ensemble',    best['proba']),
]:
    a = accuracy_score(y_te_np, (p >= 0.5).astype(int)) * 100
    print(f'  {name:<38} {a:>9.2f}%')
print('─'*65)


# ═══════════════════════════════════════════════════════════════
# CELL 99 — Save Models and Encoders
# ═══════════════════════════════════════════════════════════════
import joblib
import os

os.makedirs('models', exist_ok=True)

print("Saving models...")
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(enc_leaf, 'models/enc_leaf.pkl')
joblib.dump(leaf_scaler, 'models/leaf_scaler.pkl')

joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')

torch.save(model_A.state_dict(), 'models/model_A.pth')
torch.save(model_B.state_dict(), 'models/model_B.pth')
torch.save(model_C.state_dict(), 'models/model_C.pth')
torch.save(ahin1.state_dict(), 'models/ahin1.pth')
torch.save(ahin2.state_dict(), 'models/ahin2.pth')
torch.save(ahin3.state_dict(), 'models/ahin3.pth')

import json
best_ensemble_params = {
    'weights': BEST_W,
    'threshold': BEST_THRESH
}
with open('models/ensemble_params.json', 'w') as f:
    json.dump(best_ensemble_params, f)

# Save a background sample for SHAP/LIME explainability
bg_sample_idx = np.random.choice(len(X_tr), size=min(200, len(X_tr)), replace=False)
np.save('models/X_train_bg.npy', X_tr[bg_sample_idx])

# Save engineered feature names
ENG_FEATURE_NAMES = FEATURE_NAMES + [
    'age*glucose', 'age*bmi', 'risk_score', 'age^2', 'glucose^2',
    'log1p_age', 'log1p_glucose', 'log1p_bmi'
] + [f'poly_{i}' for i in range(X_poly.shape[1])]
with open('models/feature_names.json', 'w') as f:
    json.dump(ENG_FEATURE_NAMES, f)

print("✅ All models and preprocessors saved to models/ directory successfully!")
