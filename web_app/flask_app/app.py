import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load model and preprocessor (pipeline) ===
pipeline = None
model = None
try:
    # Try different model paths
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/best_model_pipeline.pkl'))
    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        # Try alternative paths
        alt_paths = [
            os.path.join(os.path.dirname(__file__), 'models/best_model_pipeline.pkl'),
            os.path.join(os.path.dirname(__file__), '../models/best_model_pipeline.pkl'),
            'best_model_pipeline.pkl',
            'models/best_model_pipeline.pkl'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                MODEL_PATH = alt_path
                break
        else:
            raise FileNotFoundError(f"Model file not found. Searched paths: {[MODEL_PATH] + alt_paths}")
    logger.info(f"Loading model from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline # for health check
    logger.info("Model loaded successfully")
    logger.info("Loaded pipeline model")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    pipeline = None
    model = None

# === Define form fields (raw input only) ===
FORM_FIELDS = ['Bedrooms', 'Bathrooms', 'SquareFeet', 'Neighborhood', 'YearBuilt']
NEIGHBORHOOD_CATEGORIES = ['Rural', 'Suburb', 'Urban']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    form_data = {field: '' for field in FORM_FIELDS}
    error_message = None

    # Check if model is loaded
    if model is None:
        error_message = "Model not loaded. Please check if the model file exists."

    if request.method == 'POST' and model is not None:
        try:
            # 1. Get form data
            for field in FORM_FIELDS:
                form_data[field] = request.form.get(field, '')

            # 2. Validate required fields
            if not all(form_data.values()):
                error_message = "Please fill in all fields"
            else:
                # 3. Convert numeric fields
                input_data = form_data.copy()
                for field in ['Bedrooms', 'Bathrooms', 'SquareFeet', 'YearBuilt']:
                    try:
                        value = float(input_data[field])
                        if field in ['Bedrooms', 'Bathrooms'] and value < 0:
                            raise ValueError(f"{field} cannot be negative")
                        if field == 'SquareFeet' and value <= 0:
                            raise ValueError("Square feet must be positive")
                        if field == 'YearBuilt' and (value < 1800 or value > 2024):
                            raise ValueError("Year built must be between 1800 and 2024")
                        input_data[field] = value
                    except ValueError as ve:
                        error_message = f"Invalid {field}: {str(ve)}"
                        break
                    except Exception:
                        error_message = f"Invalid {field}: Please enter a valid number"
                        break

                # 4. Validate neighborhood
                if input_data['Neighborhood'] not in NEIGHBORHOOD_CATEGORIES:
                    error_message = "Please select a valid neighborhood"

                # 5. Make prediction if no errors
                if not error_message:
                    # Create DataFrame
                    input_df = pd.DataFrame([input_data])
                    logger.info(f"Input data: {input_data}")
                    # Log: find similar patterns in cleaned_data.csv
                    try:
                        # Try different paths for cleaned_data.csv
                        csv_paths = [
                            os.path.join(os.path.dirname(__file__), '../../data/processed/cleaned_data.csv'),
                            os.path.join(os.path.dirname(__file__), '../data/processed/cleaned_data.csv'),
                            os.path.join(os.path.dirname(__file__), 'data/processed/cleaned_data.csv'),
                            'data/processed/cleaned_data.csv'
                        ]
                        
                        csv_path = None
                        for path in csv_paths:
                            if os.path.exists(path):
                                csv_path = path
                                break
                                
                        if csv_path:
                            df_clean = pd.read_csv(csv_path)
                            mask = (
                                (df_clean['Bedrooms'] == input_data['Bedrooms']) &
                                (df_clean['Bathrooms'] == input_data['Bathrooms']) &
                                (df_clean['SquareFeet'].between(input_data['SquareFeet']-50, input_data['SquareFeet']+50)) &
                                (df_clean['Neighborhood'] == input_data['Neighborhood']) &
                                (df_clean['YearBuilt'].between(input_data['YearBuilt']-5, input_data['YearBuilt']+5))
                            )
                            similar = df_clean[mask]
                            logger.info(f"Found {len(similar)} similar samples in cleaned_data.csv:")
                            logger.info(similar[['SquareFeet','Bedrooms','Bathrooms','Neighborhood','YearBuilt','Price']].to_string())
                            logger.info(f"Mean price of similar: {similar['Price'].mean() if len(similar)>0 else 'N/A'}")
                        else:
                            logger.warning("Could not find cleaned_data.csv file")
                    except Exception as e:
                        logger.warning(f"Could not log similar samples: {e}")
                    # Predict using pipeline
                    y_pred = pipeline.predict(input_df)[0]
                    prediction = round(float(y_pred), 2)
                    logger.info(f"Prediction: {prediction}")

        except Exception as e:
            error_message = f"Prediction error: {str(e)}"
            logger.error(f"Prediction error: {e}")

    return render_template(
        'index.html',
        prediction=prediction,
        form_data=form_data,
        neighborhood_options=NEIGHBORHOOD_CATEGORIES,
        error_message=error_message
    )

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'message': 'Model loaded successfully' if model is not None else 'Model not loaded'
    }
    return status

if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(__file__)}")
    
    # Check templates folder
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        print(f"Warning: Templates directory not found at {templates_dir}")
        print("Please create templates/index.html")
    
    app.run(debug=True, host='0.0.0.0', port=5000)