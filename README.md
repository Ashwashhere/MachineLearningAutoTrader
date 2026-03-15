# Autotrader Machine Learning Predictive Modelling

## Overview
This project implements an end-to-end Machine Learning pipeline to predict the selling price of used cars. The target variable is price, a continuous numerical value in GBP, making this a regression task. The primary goal is to accurately model the non-linear depreciation of vehicles and the complex dynamics of the used car market.

## Data Exploration and Processing
Thorough exploratory data analysis and processing were conducted to prepare the data for machine learning:
* **Handling Skewness:** The car prices and mileage displayed heavy positive right-skewness. A log transformation (`np.log1p`) was applied to the target variable to compress the scale and allow the models to focus on percentage errors rather than absolute differences.

* **Feature Engineering:** A `vehicle_age` feature was derived from the registration year to create a more intuitive, ratio-scale metric for vehicle depreciation. Erroneous registration years (e.g., years < 1900 or > 2025) were dropped to prevent the model from learning irrational rules from corrupted data entries.
* **Imputation:** Missing continuous values (like mileage) were filled using the median due to skewness, while missing categorical features utilized the mode.
* **Categorical Encoding:** A hybrid encoding strategy was used to handle varying cardinality. Low-cardinality features utilized One-Hot Encoding. High-cardinality features (like `standard_model`) utilized Target Encoding (mean encoding) to prevent the "curse of dimensionality" and maintain a compact feature space.
* **Scaling:** Standard Scaling (Z-score normalization) was applied to continuous numerical features (mileage, vehicle_age) to ensure algorithms relying on Euclidean distances (like kNN) performed optimally.
* **Data Splitting:** The dataset was split 80/20 into training and test sets using a fixed random seed. All preprocessing was fitted on the training set only and then applied to the test set to prevent data leakage and ensure realistic evaluation.

## Modelling Strategy
All data processing steps were encapsulated within Scikit-Learn Pipeline objects to strictly prevent data leakage. Four different regression models were built, trained, and optimized using GridSearchCV with k-Fold Cross-Validation:
* **Linear Regression:** Served as a simple baseline model to judge whether more complex models add value.
* **K-Nearest Neighbours (kNN):** A non-parametric model tuned for the optimal number of neighbors (k=7) and uniform weight functions.
* **Decision Tree Regressor:** Captured non-linear patterns, optimized with a maximum depth of 15 and a `min_samples_leaf` of 4 to prevent overfitting.
* **Random Forest Regressor:** An ensemble method used to reduce the high variance of individual decision trees and improve generalization. The best parameters were a maximum depth of 20, minimum samples split of 5, and 100 estimators.

## Results and Evaluation
The models were evaluated on the unseen test set using R-Squared (R²), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). 

| Model | R-Squared (R²) | RMSE (£) | MAE (£) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.8997 | 3780.01 | 1963.86 | 15.58% |
| **Decision Tree** | 0.8541 | 4558.10 | 2259.61 | 17.07% |
| **k-Nearest Neighbours** | 0.8198 | 5064.08 | 2884.28 | 23.19% |
| **Linear Regression** | 0.7711 | 5705.80 | 3450.48 | 35.47% |

**Key Findings:**
* The **Random Forest Regressor** is unequivocally the best-performing model, capable of explaining 90% of the price variability. 
* The model's MAPE indicates that, on average, the automated prediction is within 15.58% of the actual list price, which is highly accurate for the unpredictable used car market.
* Residual analysis demonstrated that the ensemble Random Forest approach successfully smoothed out individual tree errors and maintained consistent accuracy across different segments of the market.

## Feature Importance
Analysis of the Random Forest model identified the top three drivers of car prices:
1. **Vehicle Age (Year of Registration):** The dominant predictor that establishes the baseline value tier for a vehicle.
2. **Mileage:** The primary depreciation factor that modifies the baseline value set by the car's age.
3. **Standard Model:** The specific brand and model type, which is critical for distinguishing between market segments (e.g., Luxury vs. Economy).
