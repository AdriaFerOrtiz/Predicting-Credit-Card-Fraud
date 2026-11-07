# Predicting Credit Card Fraud

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning classification algorithms.

The objective is to develop predictive models capable of distinguishing legitimate transactions from fraudulent ones, addressing one of the main challenges in fraud detection: **severe class imbalance**.

The project follows an end-to-end Machine Learning workflow, covering exploratory data analysis, data preprocessing, model selection, imbalance handling, feature engineering, hyperparameter optimization and final predictions.

The models were developed using the Kaggle competition dataset *Predicting Credit Card Fraud*.

### Final Result

| Model | Kaggle ROC-AUC |
|---|---:|
| LightGBM | 0.94651 |
| XGBoost | 0.94671 |
| **LightGBM + XGBoost Blend** | **0.94689** |

The best submission was obtained by combining the predicted probabilities of LightGBM and XGBoost using an equal-weight average.

---

## Dataset

The dataset contains credit card transactions with numerical, categorical and boolean attributes describing transaction characteristics, customer information and security indicators.

### Dataset Dimensions

| Dataset | Observations | Columns |
|---|---:|---:|
| Training | 699,960 | 26 |
| Test | 299,984 | 25 |
| Sample Submission | 299,984 | 2 |

The training dataset contains 24 predictive features, an identifier (`id`) and the target variable (`is_fraud`).

### Target Variable

The target variable is `is_fraud`:

- `0`: Legitimate transaction.
- `1`: Fraudulent transaction.

### Feature Distribution

The 24 predictive features are divided into three groups:

**Numerical Features (13)**

- `amount_usd`
- `hours_since_last_txn`
- `txn_count_last_24h`
- `distance_from_home_km`
- `card_age_months`
- `customer_age`
- `account_balance_usd`
- `cvv_retry_count`
- `velocity_score`
- `time_of_day_hour`
- `day_of_week`
- `merchant_risk_score`
- `prior_disputes`

**Categorical Features (5)**

- `merchant_category`
- `card_type`
- `auth_method`
- `channel`
- `device_type`

**Boolean Features (6)**

- `is_foreign_transaction`
- `is_new_merchant`
- `used_vpn`
- `ip_country_mismatch`
- `billing_shipping_mismatch`
- `is_ai_generated_scam_attempt`

### Class Imbalance

One of the most important characteristics of the dataset is its severe class imbalance.

| Class | Transactions | Percentage |
|---|---:|---:|
| Legitimate | 688,271 | 98.33% |
| Fraudulent | 11,689 | 1.67% |
| **Total** | **699,960** | **100%** |

The imbalance ratio is approximately **58.88:1**.

This makes accuracy an unsuitable primary evaluation metric, since a model predicting every transaction as legitimate would achieve approximately 98.33% accuracy without detecting any fraud.

---

## Technologies and Tools

The project was developed using the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, classification and evaluation |
| Imbalanced-learn | Class imbalance techniques |
| XGBoost | Gradient Boosting classification |
| LightGBM | Gradient Boosting classification |
| Jupyter Notebook | Experimentation and documentation |
| Git & GitHub | Version control and project management |
| Kaggle | Dataset and competition evaluation |

---

## Project Workflow

The project was divided into seven phases.

### 1. Exploratory Data Analysis (EDA)

The first phase focused on understanding the dataset and identifying potential issues before applying any transformations.

Main activities:

- Analyzed dataset dimensions, feature types and distributions.
- Checked missing values, duplicated observations and ID integrity.
- Investigated numerical distributions, skewness and potential outliers.
- Examined categorical and boolean variables.
- Studied correlations and fraud rates across different feature groups.
- Compared training and test feature distributions.
- Investigated potential data leakage concerns.

**Key findings:**

- No missing values or duplicated observations were detected.
- `cvv_retry_count` showed the strongest numerical correlation with fraud (Pearson correlation: 0.2123).
- `merchant_risk_score` and `velocity_score` also showed positive associations.
- Gift Cards (7.30%), Crypto Exchange (6.53%) and Gaming (5.51%) presented particularly high fraud rates.
- Security-related indicators, especially `ip_country_mismatch`, `billing_shipping_mismatch` and `is_foreign_transaction`, showed strong associations with fraudulent transactions.

### 2. Data Preprocessing

The objective was to prepare the features for Machine Learning algorithms while avoiding data leakage.

Main activities:

- Separated predictive features, target variable and transaction identifiers.
- Excluded `id` from model training.
- Created an 80/20 stratified training and validation split.
- Identified numerical, categorical and boolean features.
- Applied One-Hot Encoding to categorical variables.
- Used RobustScaler for numerical variables when required by the algorithm.
- Built reusable Scikit-learn preprocessing pipelines.

The preprocessing generated 54 features from the original 24 predictive variables.

No missing-value imputation was necessary.

### 3. Baseline Models and Model Selection

Different classification algorithms were trained and evaluated using the same training and validation split.

The following models were compared:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoost
- LightGBM

The initial comparison identified LightGBM and XGBoost as the strongest overall candidates.

**Baseline Model Comparison**

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| LightGBM | **0.94423** | **0.50289** |
| XGBoost | 0.94386 | 0.50234 |
| HistGradientBoosting | 0.94357 | 0.48925 |
| Logistic Regression | 0.94028 | 0.49698 |
| Random Forest | 0.93209 | 0.46195 |
| Extra Trees | 0.92693 | 0.44493 |
| Decision Tree | 0.89131 | 0.38568 |
| Dummy Classifier | 0.50000 | 0.01670 |

ROC-AUC was selected as the primary competition metric, complemented by Average Precision, Precision, Recall and F1-Score.

### 4. Imbalanced Data

Different techniques were investigated to address the imbalance between legitimate and fraudulent transactions.

The experiments included:

- Class Weighting.
- Random Undersampling.
- Random Oversampling.
- SMOTENC.

The main experiments were performed using Logistic Regression, LightGBM and XGBoost.

**Best Results by Strategy**

| Strategy | Best Model | ROC-AUC | Average Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|
| Baseline | LightGBM | 0.94423 | 0.50289 | 0.30582 | 0.43373 |
| Class Weighting | Logistic Regression | 0.94007 | 0.49391 | 0.84303 | 0.19019 |
| Random Undersampling | LightGBM | 0.94406 | 0.49873 | 0.56459 | 0.45502 |
| Random Oversampling | LightGBM | 0.94364 | **0.50449** | 0.55090 | **0.47136** |
| SMOTENC | LightGBM | 0.94212 | 0.49780 | 0.29769 | 0.42726 |

Random Oversampling achieved the best Average Precision and F1-Score among the tested imbalance strategies.

Class Weighting achieved a considerably higher Recall, but at the expense of generating many false positives.

SMOTENC did not provide sufficient improvements to justify its higher computational cost.

### 5. Feature Engineering

New features were created to provide additional information to the classification algorithms.

A total of **16 engineered features** were introduced.

**Ratios and Activity Features**

- `amount_to_abs_balance_ratio`: transaction amount relative to account balance.
- `cvv_retries_per_recent_txn`: CVV retry frequency relative to recent transaction activity.

**Risk Indicator Interactions**

- `vpn_ip_mismatch`
- `foreign_ip_mismatch`
- `foreign_billing_mismatch`
- `security_risk_flags_count`

**Numerical Interactions**

- `merchant_velocity_interaction`
- `cvv_velocity_interaction`

**Logarithmic Transformations**

- `log_amount_usd`
- `log_account_balance_usd`
- `log_cvv_retry_count`
- `log_prior_disputes`

**Cyclical Encoding**

- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`

The engineered dataset contained 40 predictive features.

**Feature Engineering Results**

| Model and Strategy | Original Features AP | Engineered Features AP |
|---|---:|---:|
| LightGBM + Baseline | 0.50289 | 0.49935 |
| LightGBM + Oversampling | 0.50449 | **0.50841** |
| XGBoost + Baseline | 0.50234 | 0.49872 |
| XGBoost + Oversampling | 0.50142 | 0.50396 |

Feature Engineering improved Average Precision when combined with Random Oversampling, particularly for LightGBM.

**Most Relevant Engineered Features**

LightGBM feature importance analysis identified several engineered features among the most important predictors:

| Feature | Split Importance |
|---|---:|
| `merchant_velocity_interaction` | 404 |
| `amount_to_abs_balance_ratio` | 267 |
| `security_risk_flags_count` | 188 |
| `cvv_velocity_interaction` | 170 |
| `hour_sin` | 130 |
| `hour_cos` | 104 |
| `cvv_retries_per_recent_txn` | 80 |
| `log_amount_usd` | 70 |

Feature importance values represent the model's split-based importance and should not be interpreted as causal contributions.

An additional ablation study was performed by removing `is_ai_generated_scam_attempt`.

Removing this feature reduced Average Precision, suggesting that the models relied on information contained in this indicator. Its availability at prediction time remains an important consideration before any real-world deployment.

### 6. Hyperparameter Tuning and Model Evaluation

LightGBM and XGBoost were selected for further optimization.

The optimization process included:

- Stratified 3-Fold Cross Validation.
- RandomizedSearchCV.
- Eight randomized hyperparameter configurations per model.
- Average Precision as the primary optimization metric.
- ROC-AUC as a secondary evaluation metric.

**Cross Validation Results**

| Model | Baseline CV AP | Tuned CV AP | Tuned CV ROC-AUC |
|---|---:|---:|---:|
| LightGBM | 0.49043 | 0.49588 | 0.94517 |
| XGBoost | 0.48336 | **0.49813** | **0.94586** |

XGBoost obtained the best Cross Validation results after hyperparameter optimization.

**Best LightGBM Hyperparameters**

| Parameter | Value |
|---|---:|
| n_estimators | 400 |
| learning_rate | 0.03 |
| num_leaves | 31 |
| min_child_samples | 100 |
| max_depth | -1 |
| subsample | 0.8 |
| colsample_bytree | 0.7 |
| reg_lambda | 1 |

**Best XGBoost Hyperparameters**

| Parameter | Value |
|---|---:|
| n_estimators | 300 |
| learning_rate | 0.07 |
| max_depth | 3 |
| min_child_weight | 1 |
| subsample | 0.7 |
| colsample_bytree | 0.9 |
| reg_lambda | 0.5 |
| gamma | 0.1 |

**Classification Threshold Optimization**

Different classification thresholds were evaluated using an independent threshold-selection subset.

Two optimization criteria were considered:

- Maximum F1-Score: balances Precision and Recall.
- Maximum F2-Score: gives greater importance to Recall.

The selected thresholds were:

| Model | F1 Threshold | F2 Threshold |
|---|---:|---:|
| LightGBM | 0.65282 | 0.37407 |
| XGBoost | 0.68003 | 0.39327 |

Threshold optimization demonstrated the trade-off between detecting more fraudulent transactions and generating additional false positives.

The thresholds were used for classification analysis, not for Kaggle submissions.

### 7. Final Predictions and Kaggle Submission

Both optimized models were retrained using the complete labeled training dataset.

The final training pipeline included:

- The 40 engineered features.
- Random Oversampling with a sampling ratio of 0.1.
- The optimized LightGBM and XGBoost configurations.
- One-Hot Encoding for categorical variables.

Predicted fraud probabilities were generated for all 299,984 test transactions.

Three submission files were created:

- `submission_lightgbm.csv`
- `submission_xgboost.csv`
- `submission_blend.csv`

The blended submission was generated by calculating the equal-weight average of LightGBM and XGBoost predicted probabilities.

**Final Kaggle Results**

| Submission | ROC-AUC |
|---|---:|
| LightGBM | 0.94651 |
| XGBoost | 0.94671 |
| **LightGBM + XGBoost Blend** | **0.94689** |

The blended model achieved the best result among the tested submissions.

The final competition submissions contained predicted probabilities rather than binary classifications, following the ROC-AUC evaluation requirements.

---

## Final Results Summary

| Evaluation | Best Result | Score |
|---|---|---:|
| Baseline model comparison | LightGBM ROC-AUC | 0.94423 |
| Imbalance strategy comparison | LightGBM + Oversampling AP | 0.50449 |
| Feature Engineering | LightGBM + Engineered Features + Oversampling AP | 0.50841 |
| Hyperparameter Tuning | XGBoost CV Average Precision | 0.49813 |
| Tuned model validation | XGBoost Average Precision | 0.50856 |
| **Kaggle submission** | **LightGBM + XGBoost Blend ROC-AUC** | **0.94689** |

---

## Repository Structure

```text
Predicting Credit Card Fraud/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_selection.ipynb
│   ├── 04_imbalanced_data.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_final_predictions.ipynb
│
├── outputs/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Notebook Guide

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_preprocessing.ipynb` | Data preparation and preprocessing pipelines |
| `03_model_selection.ipynb` | Initial classification model comparison |
| `04_imbalanced_data.ipynb` | Class imbalance experiments |
| `05_feature_engineering.ipynb` | Feature creation and feature importance |
| `06_hyperparameter_tuning.ipynb` | Cross Validation, hyperparameter optimization and threshold selection |
| `07_final_predictions.ipynb` | Final model training and Kaggle submissions |

---

## Installation and Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/predicting-credit-card-fraud.git

cd predicting-credit-card-fraud
```

Replace `YOUR_USERNAME` with the GitHub account hosting the repository.

### 2. Create a virtual environment

For Windows CMD:

```cmd
py -m venv .venv

.venv\Scripts\activate.bat
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Download the dataset

Download the competition data from Kaggle:

[Predicting Credit Card Fraud - Kaggle](https://www.kaggle.com/)

Place the original CSV files inside:

```text
data/raw/
```

The expected files are:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The original dataset files and generated submissions are excluded from Git tracking.

### 5. Run the notebooks

Launch Jupyter:

```cmd
jupyter notebook
```

Execute the notebooks in numerical order, starting with `01_eda.ipynb`.

---

## Limitations and Considerations

- The dataset is a Kaggle competition benchmark and should not be considered a substitute for validation using real banking transaction data.
- Class imbalance remains a significant challenge.
- The public leaderboard score does not guarantee equivalent performance on unseen real-world transactions.
- The public leaderboard may evaluate only a subset of the test dataset.
- Probability calibration was not fully investigated.
- The public availability of `is_ai_generated_scam_attempt` at the moment of transaction authorization has not been independently verified. Its potential use in a real-time fraud detection system therefore requires further investigation.
- The feature importance analysis identifies model-level predictive associations, not causal relationships.
- The hyperparameter search was limited to eight randomized configurations per model and three-fold cross-validation.

---

## Final Takeaways

This project provided practical experience with the complete Machine Learning workflow for an imbalanced binary classification problem.

The main lessons were:

- Understanding class imbalance is essential when developing fraud detection models.
- ROC-AUC and Average Precision provide more informative evaluation criteria than Accuracy alone.
- Different resampling strategies can significantly change the Precision-Recall trade-off.
- Feature Engineering can improve predictive performance when new features capture useful relationships.
- LightGBM and XGBoost are effective algorithms for this type of structured dataset.
- Cross Validation and hyperparameter optimization help identify more robust configurations.
- Classification threshold selection must consider the relative costs of false positives and false negatives.
- Combining complementary models can improve competition performance.

**Final Kaggle ROC-AUC: 0.94689**

**Best Submission: LightGBM + XGBoost Blend**