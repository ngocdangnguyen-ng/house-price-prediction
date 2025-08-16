# House Price Prediction - Model Performance Report

## 1. Introduction
This report presents a formal evaluation and comparison of several machine learning models for house price prediction, including Linear Regression, Random Forest, and Neural Network (MLPRegressor). The analysis aims to identify the most effective model based on rigorous statistical metrics and practical considerations.

## 2. Evaluation Metrics
* **RMSE (Root Mean Squared Error):** Quantifies the average magnitude of prediction errors (lower values indicate better performance).
* **R² Score:** Represents the proportion of variance in the target variable explained by the model (values closer to 1 indicate better fit).
* **Cross-validation:** Assesses the consistency and generalizability of the model by evaluating its performance across multiple data splits.

## 3. Results Summary

| Model             | Test RMSE | Test R² | Notes                        |
|-------------------|-----------|---------|------------------------------|
| Linear Regression | $49,702   | 0.567   | Simple, interpretable         |
| Ridge Regression  | $49,700   | 0.567   | Regularized linear model      |
| Lasso Regression  | $49,703   | 0.566   | Regularized linear model      |
| Decision Tree     | $72,130   | 0.087   | Nonlinear, prone to overfit   |
| Random Forest     | $52,942   | 0.508   | Captures non-linear effects   |
| Gradient Boosting | $49,754   | 0.566   | Robust, handles nonlinearity  |

## 4. Best Model Selection
**Selected model:** best_model_linear_regression.pkl

**Rationale:**
* The predicted values closely align with actual house prices for the majority of cases.

## 6. Observed Limitations
* The model exhibits reduced accuracy for properties with exceptionally high prices (outliers).
* Additional data may be required to further enhance model robustness.
* Some individual predictions still deviate significantly from actual values.

## 7. Recommendations and Next Steps
* Conduct further feature engineering and selection to improve model performance.
* Explore hyperparameter tuning, especially for ensemble and neural network models.
* Consider collecting more data to address rare or extreme cases.
* Deploy the best-performing model in the production Flask web application.

---
*Report generated: 2025-08-12*
* Use the best model in my Flask web app
* Test it with more examples

**To improve later:**
* Get more house data
* Try different ways to clean the data
* Maybe try other models like XGBoost

## 8. Conclusion
I successfully built a model that can predict house prices reasonably well. The best_model_linear_regression.pkl works best for this dataset and is now ready to use in the web application.

---
**Project completed:** August 2025  
**Dataset:** House prices with 5 features  
**Best model accuracy:** 0.567 (Linear Regression/Ridge/Gradient Boosting)
The Linear Regression, Ridge Regression, and Gradient Boosting models achieved the highest R² score (~0.57) and lowest RMSE (~$49,700), indicating moderate predictive performance for this dataset. Random Forest performed worse than expected on this split.