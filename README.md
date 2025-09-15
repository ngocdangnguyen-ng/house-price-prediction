# House Price Prediction – Personal Machine Learning Project

## Project Overview
This is a personal learning project where I explore machine learning techniques for house price prediction. As a computer science student passionate about data science, I built this project to practice the full ML workflow and share my experience with others. The project covers the main steps of a typical ML pipeline, from data exploration and feature engineering to model development and evaluation.

**Objective:**
To develop a predictive model for residential house prices using real-world data, while gaining hands-on experience with the end-to-end machine learning workflow.

## Key Results
| Model             | RMSE    | R² Score | Notes                       |
|-------------------|---------|----------|-----------------------------|
| Linear Regression | $49,702 | 0.567    | Simple, interpretable       |
| Ridge Regression  | $49,700 | 0.567    | Regularized linear model    |
| Lasso Regression  | $49,703 | 0.566    | Regularized linear model    |
| Decision Tree     | $72,130 | 0.087    | Nonlinear, prone to overfit |
| Random Forest     | $52,942 | 0.508    | Captures non-linear effects |
| Gradient Boosting | $49,754 | 0.566    | Robust, handles nonlinearity|

The best model (Linear Regression/Ridge/Gradient Boosting) achieves an R² of ~0.57 and RMSE of ~$49,700, indicating moderate predictive performance for this dataset. Random Forest performed worse than expected on this split.

Although performance is moderate, the project demonstrates the end-to-end ML workflow effectively.

## Features
* End-to-end ML pipeline: EDA, preprocessing, feature engineering, model training, evaluation, deployment
* Modular, reusable codebase
* Multiple regression models: Linear Regression, Random Forest
* Automated data cleaning, outlier handling, encoding, scaling
* Cross-validation, robust evaluation metrics
* Professional visualizations and reporting
* Dockerized for reproducibility
* Web interface for real-time prediction (Flask, modern UI, error handling)
* Models and pipelines saved with joblib for easy reuse

## Getting Started
**Installation**
```python
git clone https://github.com/ngocdangnguyen-ng/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```
**Quick Example**
```python
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load('models/best_model_pipeline.pkl')

# Example input
house = {
    'Bedrooms': 3,
    'Bathrooms': 2,
    'SquareFeet': 1500,
    'Neighborhood': 'Urban',
    'YearBuilt': 2005
}
X = pd.DataFrame([house])
price = model.predict(X)
print(f"Predicted price: ${price[0]:,.2f}")
```

## Project Structure
```
housing-price-prediction/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── eda/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── models/
│
├── web_app/              
│   ├── flask_app/
│      ├── app.py
│      └── templates/
│          └── index.html
│         
├── scripts/
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predict.py
│   └── train_pipeline.py
│ 
├── tests/
│
├── reports/
│   ├── figures/
│   └── model_performance.md
```

## Process Overview
1. **Data Exploration:** Analyzed 5,000 house sales, visualized distributions, and identified key features
2. **Feature Engineering:** Created new features such as price per square foot and renovation status
3. **Modeling:** Built and compared linear regression, random forest, and gradient boosting models with proper validation

## What I Learned & Challenges
* **Data Preprocessing:** Standardized and scaled features, handled outliers (the dataset used in this project did not contain missing values)
* **Feature Engineering:** Created new features (e.g., price per sqft, total rooms, house age) to improve model performance
* **Model Selection:** Compared multiple algorithms (Linear Regression, Random Forest, Gradient Boosting)
* **Evaluation:** Used cross-validation and multiple metrics for robust assessment
* **Code Organization:** Modular, maintainable Python code and reproducible pipeline

**Key Insights:**
* Location (Neighborhood/ZIP code) and square footage are the most influential factors
* Number of bathrooms is a stronger predictor than bedrooms
* Newer houses (built after 1990) tend to be more valuable

**Challenges:**
* Feature engineering and selecting the most relevant features
* Model interpretability and understanding feature importance
* Tuning model hyperparameters for best performance

## Limitations & Future Work
* The dataset is limited to King County (Seattle area)
* Advanced models (e.g., neural networks) not yet explored
* Further feature engineering and model explainability (e.g., SHAP, LIME) are planned
* Add more domain-specific features
* Integrate hyperparameter tuning (GridSearchCV, Optuna, etc.)
* Expand model selection to include XGBoost, LightGBM, CatBoost
* Add user authentication and logging to the web app
* Deploy as a REST API or on cloud platforms (Heroku, AWS, etc.)
* Enhance reporting with automated PDF/HTML reports

## Acknowledgments
* **Courses:** Andrew Ng's Machine Learning Course
* **Datasets:** Kaggle House Price Prediction Data
* **Tools:** scikit-learn, pandas, matplotlib, seaborn

## Contact
* **LinkedIn:** https://www.linkedin.com/in/ngocnguyen-fr
* **Email:** nndnguyen2016@gmail.com

---
I welcome feedback and suggestions for improvement. Thank you for visiting my project!

## License
This project is licensed under the MIT License. See the LICENSE file for details.
