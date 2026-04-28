"""Quick script to save background sample and feature names from existing models."""
import numpy as np
import json
import os
import joblib

os.makedirs('models', exist_ok=True)

# Load scaler to get training data shape info  
scaler = joblib.load('models/scaler.pkl')

# We need the feature names - construct them from the original order
FEATURE_NAMES = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

# The poly features are interaction_only degree=2 on 5 clinical features = C(5,2) = 10 pairs
# Total poly output = 5 original + 10 interactions = 15
# But the original 5 clinical features are already in FEATURE_NAMES, so poly adds:
# Original 10 + 8 manual + 15 poly = 33
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly.fit(np.zeros((1, 5)))
n_poly = poly.n_output_features_

ENG_FEATURE_NAMES = FEATURE_NAMES + [
    'age*glucose', 'age*bmi', 'risk_score', 'age^2', 'glucose^2',
    'log1p_age', 'log1p_glucose', 'log1p_bmi'
] + [f'poly_{i}' for i in range(n_poly)]

with open('models/feature_names.json', 'w') as f:
    json.dump(ENG_FEATURE_NAMES, f)

# Generate a synthetic background sample using scaler's stored mean/var
# This avoids needing the original training data
n_features = len(ENG_FEATURE_NAMES)
mean = scaler.mean_
scale = scaler.scale_
np.random.seed(42)
bg = np.random.randn(200, n_features).astype(np.float32)
# These are already in scaled space, which is what models expect
np.save('models/X_train_bg.npy', bg)

print(f"Saved {len(ENG_FEATURE_NAMES)} feature names: {ENG_FEATURE_NAMES}")
print(f"Saved background sample: {bg.shape}")
print("Done!")
