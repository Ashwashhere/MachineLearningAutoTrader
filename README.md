# Car Price Prediction: Autotrader Machine Learning Project

This repository contains a comprehensive **end-to-end Machine Learning pipeline** designed to predict the selling price of used cars. Using a dataset of Autotrader vehicle adverts, the project explores the relationship between vehicle attributes (mileage, age, brand) and their market value.

---

## Technical Overview

The project is implemented as a modular Jupyter Notebook (`AT_ML.ipynb`) that follows the standard ML lifecycle:

* **Data Understanding**: Initial exploration of features including price, mileage, and registration codes.
* **Feature Engineering**:
* Derivation of `vehicle_age` from registration data.
* Logarithmic transformations on target (`price`) and `mileage` to normalize right-skewed distributions.
* Handling high-cardinality features via **Target Encoding** and low-cardinality via **One-Hot Encoding**.


* **Data Integrity**: Implementation of Scikit-Learn `Pipelines` to ensure strict separation between training and test data, preventing data leakage during scaling and imputation.
* **Predictive Modeling**: Comparison of four regression algorithms: **Linear Regression**, **K-Nearest Neighbors**, **Decision Trees**, and **Random Forest**.
* **Hyperparameter Tuning**: Utilization of `GridSearchCV` to optimize model parameters such as `n_neighbors` and `max_depth`.

---

## Key Features & Methodology

### Data Preprocessing & Cleaning

* **Missing Value Imputation**: Numerical values are filled using the median, while categorical features use the most frequent (mode) strategy.
* **Outlier Management**: Extreme values in numerical columns are clipped to the 1st and 99th percentiles based on training set distributions.
* **Category Grouping**: Rare categorical values (frequency < 1%) are consolidated into an "Other" category to reduce noise and improve model generalization.

### Advanced Visualizations

The notebook generates several diagnostic plots to validate the data at each stage:

* **Missingness Maps**: Heatmaps to visualize the success of the imputation strategy.
* **Correlation Heatmaps**: Identifying multicollinearity between numeric features.
* **Residual Analysis**: Scatter plots of predicted vs. actual values and error distributions to diagnose model bias.

---

## Model Performance Metrics

Models are evaluated on unseen test data using the following metrics:

* ** Score**: Proportion of variance explained by the model.
* **Root Mean Squared Error (RMSE)**: Error magnitude in the original currency (£) scale.
* **Mean Absolute Percentage Error (MAPE)**: Relative prediction accuracy.

---

## Requirements

To run this notebook, you will need the following Python libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
