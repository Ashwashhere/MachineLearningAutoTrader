<div align="center">

  <h1>🏎️ Predictive Analytics: Autotrader Car Valuation Engine</h1>
  <p><b>An end-to-end Machine Learning regression pipeline built to model non-linear vehicle depreciation and predict used car market prices with high precision.</b></p>

  <!-- Technology Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit Learn" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
    <img src="https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=seaborn&logoColor=white" alt="Seaborn" />
    <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" alt="Matplotlib" />
  </p>

  <br />
  <!-- Replace portfolio_thumbnail.png with your visual image file path -->
  <img src="https://github.com/Ashwashhere/Ashwashhere/blob/3cdcba2cc56c1751e8b0e2bb6f13a499e0c195f8/atml_portfolio.png" alt="Autotrader Car Valuation Project Overview" width="100%" style="border-radius: 8px;" />
  <br />

</div>

<hr />

<h2>📌 Overview</h2>

<p>This project implements an end-to-end Machine Learning pipeline to predict the selling price of used cars based on the Autotrader "Car Sale Adverts" dataset. Predicting vehicle market value is a continuous regression task complicated by steep non-linear depreciation curves, heavy right-skewed pricing distributions, and high-cardinality categorical attributes (e.g., specific vehicle makes and standard models).</p>

<p>The primary objective is to build a robust, leakage-free predictive model that minimizes absolute percentage error across various market segments.</p>

<hr />

<h2>🧹 Data Exploration & Processing Strategy</h2>

<p>To prepare the raw data for modeling while maintaining strict adherence to machine learning best practices, data processing was decoupled into stateless and stateful operations:</p>

<ul>
  <li><b>Target Transformation (Log Scaling):</b> Car prices and mileages exhibited strong positive right-skewness. A logarithmic transformation (<code>np.log1p</code>) was applied to the target variable to compress scale, stabilize variance, and optimize algorithms for percentage-based evaluation.</li>
  <li><b>Feature Engineering:</b> A ratio-scale <code>vehicle_age</code> feature was derived from registration data to capture depreciation directly. Invalid entries (e.g., registration years prior to 1886 or future dates) were cleaned and filtered out.</li>
  <li><b>Leakage-Free Imputation:</b> Preprocessing parameters were derived solely from the training partition ($X_{\text{train}}$). Continuous numerical missing values were imputed using median strategies, while categorical missingness was imputed using the mode.</li>
  <li><b>Hybrid Categorical Encoding:</b>
    <ul>
      <li><b>Low-Cardinality Features (< 100 unique values):</b> Encoded using One-Hot Encoding.</li>
      <li><b>High-Cardinality Features (> 100 unique values, e.g., <code>standard_model</code>):</b> Encoded using Target Encoding (Mean Target Mapping) with smoothed out-of-fold training targets to prevent the curse of dimensionality.</li>
    </ul>
  </li>
  <li><b>Outlier Clipping & Feature Scaling:</b> Extreme numerical values were clipped at the 1st and 99th percentiles of $X_{\text{train}}$ to prevent distortion. Standard Scaling ($Z$-score normalization) was applied within pipelines to optimize distance-based estimators.</li>
  <li><b>Train / Test Partitioning:</b> An 80/20 train-test split was established using a fixed random seed prior to fitting stateful transformers.</li>
</ul>

<hr />

<h2>⚙️ Modeling Pipeline & Hyperparameter Tuning</h2>

<p>All data transformation steps were encapsulated inside Scikit-Learn <code>Pipeline</code> and <code>ColumnTransformer</code> objects to guarantee zero data leakage during cross-validation and testing. Four distinct model families were benchmarked and optimized via <code>GridSearchCV</code>:</p>

<ol>
  <li><b>Linear Regression:</b> Evaluated as a baseline benchmark model to test linear feature relationships.</li>
  <li><b>K-Nearest Neighbors (KNN Regressor):</b> Non-parametric distance estimator tuned for optimal $k$-neighbors ($k=7$) and distance-weighted neighbor scoring.</li>
  <li><b>Decision Tree Regressor:</b> Non-linear tree-based estimator optimized with a maximum depth limit of 15 and <code>min_samples_leaf=4</code> to curb variance and overfitting.</li>
  <li><b>Random Forest Regressor:</b> Ensemble bagging architecture tuned across estimators ($n=100$), tree depth ($\text{max\_depth}=20$), and minimum node splits ($\text{min\_samples\_split}=5$) to minimize variance and generalize complex interactions.</li>
</ol>

<hr />

<h2>📊 Model Evaluation & Results</h2>

<p>Models were evaluated on the unseen test set using R-Squared ($R^2$), Root Mean Squared Error ($\text{RMSE}$), Mean Absolute Error ($\text{MAE}$), and Mean Absolute Percentage Error ($\text{MAPE}$). Metrics were converted back from log-space to actual British Pounds (£) for clinical interpretability:</p>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>$R^2$ Score</th>
      <th>RMSE (£)</th>
      <th>MAE (£)</th>
      <th>MAPE (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Random Forest Regressor</b></td>
      <td><b>0.8997</b></td>
      <td><b>£3,780.01</b></td>
      <td><b>£1,963.86</b></td>
      <td><b>15.58%</b></td>
    </tr>
    <tr>
      <td><b>Decision Tree Regressor</b></td>
      <td>0.8541</td>
      <td>£4,558.10</td>
      <td>£2,259.61</td>
      <td>17.07%</td>
    </tr>
    <tr>
      <td><b>K-Nearest Neighbors (KNN)</b></td>
      <td>0.8198</td>
      <td>£5,064.08</td>
      <td>£2,884.28</td>
      <td>23.19%</td>
    </tr>
    <tr>
      <td><b>Linear Regression</b></td>
      <td>0.7711</td>
      <td>£5,705.80</td>
      <td>£3,450.48</td>
      <td>35.47%</td>
    </tr>
  </tbody>
</table>

<h3>Key Insights:</h3>
<ul>
  <li><b>Predictive Superiority:</b> The <b>Random Forest Regressor</b> achieved the highest accuracy, explaining approximately <b>90% of variance</b> ($R^2 = 0.8997$) in vehicle market price.</li>
  <li><b>Percentage Error Accuracy:</b> The model achieved a <b>MAPE of 15.58%</b>, demonstrating strong valuation precision across both economy and luxury market segments.</li>
  <li><b>Residual Stability:</b> Residual plots confirmed homoscedastic error distributions, demonstrating that the ensemble approach successfully eliminated individual tree overfitting.</li>
</ul>

<hr />

<h2>💡 Feature Importance & Market Drivers</h2>

<p>Feature importance extraction from the champion Random Forest Regressor identified the top structural drivers of used vehicle values:</p>

<ol>
  <li><b>Vehicle Age (Year of Registration):</b> The dominant predictor establishing the baseline pricing tier and overall depreciation bracket.</li>
  <li><b>Mileage:</b> The primary usage metric that adjusts value within a vehicle's specific age group.</li>
  <li><b>Standard Model / Target Encoded Brand:</b> Critical for distinguishing segment pricing tiers (e.g., Premium vs. Mass Market vehicles).</li>
</ol>

<hr />

<h2>🏗️ Directory Structure</h2>

<pre><code>├── atml.py                    # Complete Machine Learning pipeline & notebook source
├── portfolio_thumbnail.png    # High-resolution visual banner for portfolio summary
├── adverts.csv                # Raw Autotrader car sale dataset
└── README.md                  # Repository documentation
</code></pre>

<hr />

<h2>🚀 Getting Started</h2>

<h3>1. Clone the Repository</h3>
<pre><code>git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name</code></pre>

<h3>2. Install Dependencies</h3>
<pre><code>pip install pandas numpy scikit-learn seaborn matplotlib</code></pre>

<h3>3. Run the ML Pipeline</h3>
<pre><code>python3 atml.py</code></pre>

<hr />

<div align="center">
  <p><i>Developed as a machine learning investigation into vehicle price prediction and automated valuation modeling.</i></p>
</div>
