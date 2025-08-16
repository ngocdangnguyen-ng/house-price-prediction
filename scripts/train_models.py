import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import pandas as pd
import joblib
from src.data.data_loader import load_data
from src.data.preprocessor import handle_missing_values, detect_and_treat_outliers, standardize_types, create_basic_features, encode_categorical, scale_numeric_features
from src.features.feature_engineering import create_domain_features, select_features
from src.models.linear_models import LinearModel
from src.models.tree_models import TreeModel
from src.models.neural_networks import NeuralNetModel

# Paths
DATA_PATH = os.path.join('data', 'processed', 'cleaned_data.csv')
MODEL_DIR = os.path.join('..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load and preprocess data
df = load_data(DATA_PATH)
# (Assume data is already preprocessed and feature engineered if using cleaned_data.csv)

# 2. Identify target and features
target_col = [col for col in df.columns if 'price' in col.lower()][0]
features = [col for col in df.columns if col != target_col]

# One-hot encode categorical features to ensure all are numeric
X = df[features]
X = pd.get_dummies(X)
y = df[target_col]

# 3. Train models
models = {
    'linear_regression': LinearModel(),
    'random_forest': TreeModel(model_type='random_forest'),
    'mlp': NeuralNetModel(max_iter=500)
}


model_filename_map = {
    'linear_regression': 'best_model_linear_regression.pkl',
    'random_forest': 'best_model_random_forest.pkl',
    'mlp': 'best_model_mlp.pkl'
}

for name, model in models.items():
    print(f"Training {name}...")
    model.train(X, y)
    model_path = os.path.join(MODEL_DIR, model_filename_map[name])
    model.save(model_path)
    print(f"Saved {name} to {model_path}")

print("All models trained and saved.")
