import argparse
import os
import pandas as pd
import joblib

parser = argparse.ArgumentParser(description='Predict house prices using a trained model.')
parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with features')
parser.add_argument('--model', type=str, required=True, help='Path to trained model .pkl file')
parser.add_argument('--output', type=str, required=True, help='Path to save predictions CSV')
args = parser.parse_args()

# 1. Load input data
df = pd.read_csv(args.input)

# 2. Load model
model = joblib.load(args.model)

# 3. Align features if possible
if hasattr(model, 'feature_names_in_'):
    used_features = list(model.feature_names_in_)
    X = df[used_features]
else:
    X = df

# 4. Predict
y_pred = model.predict(X)
df['prediction'] = y_pred

# 5. Save predictions
df.to_csv(args.output, index=False)
print(f"Predictions saved to {args.output}")
