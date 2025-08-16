import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load data
DATA_PATH = os.path.join('data', 'processed', 'cleaned_data.csv')
df = pd.read_csv(DATA_PATH)

# Define the inputs and targets
features = ['Bedrooms', 'Bathrooms', 'SquareFeet', 'Neighborhood', 'YearBuilt']
target = 'Price'

X = df[features]
y = df[target]

# Preprocessing
numeric_features = ['Bedrooms', 'Bathrooms', 'SquareFeet', 'YearBuilt']
categorical_features = ['Neighborhood']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
pipeline.fit(X, y)

# Save pipeline
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/best_model_pipeline.pkl')
print("Pipeline saved to models/best_model_pipeline.pkl")
