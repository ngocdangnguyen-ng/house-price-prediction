import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import pandas as pd
import joblib
from src.data.data_loader import load_data
from src.evaluation.metrics import regression_metrics

# Paths
DATA_PATH = os.path.join('data', 'processed', 'engineered_data.csv')
MODEL_DIR = os.path.join('..', 'models')

# 1. Load data
df = load_data(DATA_PATH)
target_col = [col for col in df.columns if 'price' in col.lower()][0]
features = [col for col in df.columns if col != target_col]
X = df[features]
y = df[target_col]

# 2. Load models
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]

for model_file in model_files:
    print(f"\nEvaluating {model_file}...")
    model = joblib.load(os.path.join(MODEL_DIR, model_file))
    y_pred = model.predict(X)
    metrics = regression_metrics(y, y_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

print("\nEvaluation complete.")
