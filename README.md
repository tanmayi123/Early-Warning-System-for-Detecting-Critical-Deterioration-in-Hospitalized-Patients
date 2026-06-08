# Early Warning System for Detecting Critical Deterioration in Hospitalized Patients

A machine learning based clinical decision support tool that predicts the risk of critical deterioration in hospitalized patients using the SUPPORT2 dataset. The system flags high-risk patients early so clinicians can intervene before conditions worsen.

**Live Demo:** https://earlywarning.streamlit.app

---

## Overview

Predicting when a hospitalized patient is about to deteriorate is one of the harder problems in clinical data science. This project tackles it using the SUPPORT2 dataset from the UCI Machine Learning Repository, which contains records for 9,105 patients across 42 clinical variables.

The target variable is `sfdm2`, a measure of functional deterioration mapped to a severity scale of 1 to 5:

| Level | Label | Meaning |
|-------|-------|---------|
| 1 | No Deterioration | Patient is stable with M2 and SIP present |
| 2 | Mild Deterioration | ADL score >= 4 |
| 3 | Moderate Deterioration | SIP score >= 30 |
| 4 | Severe Deterioration | Coma or intubation |
| 5 | Critical | Less than 2 months follow-up |


---

## Live App

The app is deployed on Streamlit Cloud and has three pages:

- **Project Overview** - Summary of the problem, methodology, and model comparison chart
- **Live Prediction** - Enter patient values and get an instant risk level prediction with a probability breakdown and risk gauge
- **Model Performance** - Full metrics table, radar chart comparison, feature importance, and a detailed explanation of why the scores are what they are

https://earlywarning.streamlit.app


<img width="1917" height="859" alt="Screenshot 2026-06-06 at 1 28 21 PM" src="https://github.com/user-attachments/assets/fd9313d8-56df-4f0d-ae68-3946c715f1b3" />

<img width="1918" height="877" alt="Screenshot 2026-06-06 at 1 29 10 PM" src="https://github.com/user-attachments/assets/8eb9cb30-64cc-43b5-aab7-82e6a6d643a7" />

<img width="1920" height="904" alt="Screenshot 2026-06-06 at 1 29 46 PM" src="https://github.com/user-attachments/assets/e72da0de-0026-4eab-93c8-d95518e83e4a" />

<img width="1918" height="955" alt="Screenshot 2026-06-06 at 1 30 14 PM" src="https://github.com/user-attachments/assets/8e50260f-0e06-49e9-bbcb-65f0ef1f8227" />





---

## Dataset

- **Source:** SUPPORT2 dataset, UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/880/support2
- **Size:** 9,105 patient records, 42 features
- **Features include:** demographics, disease group, physiological scores, lab values, ADL scores, survival estimates, hospital costs, and DNR status

---

## Project Structure

```
early-warning-system/
│
├── app.py                          # Streamlit app (3 pages)
├── train_model.py                  # Training script - run this once to generate model files
├── requirements.txt                # Python dependencies
├── README.md
│
├── Early_Warning_System_...ipynb   # Original analysis notebook
│
├── data/
│   └── data.csv                    # SUPPORT2 dataset (downloaded on first run)
│
└── models/                         # Generated after running train_model.py
    ├── best_model.pkl              # Trained XGBoost model
    ├── scaler.pkl                  # StandardScaler (used by models that need scaling)
    ├── feature_names.pkl           # Ordered list of feature names the model expects
    └── model_results.json          # Cross-validation results for all 6 models
```

---

## Methodology

### 1. Data Cleaning and Preprocessing

The raw dataset had significant missing values across both numerical and categorical columns. I handled these separately:

- **Numerical columns:** KNN Imputer with k=5 neighbors
- **Categorical columns (income, race, dnr, sfdm2):** Random Forest based predictive imputation, where each missing column is predicted using the rest of the data as features

After imputation, outliers were removed using the IQR method. Any row with a value outside 1.5x the IQR in any numerical column was dropped, which reduced the dataset from ~9,000 to ~2,500 clean records.

### 2. Feature Selection

Three layers of feature selection were applied:

- **Pearson correlation** with the target variable. Features with correlation below 0.1 were dropped.
- **Cramer's V** for categorical features. Categorical columns with a Cramer's V score below 0.1 against the target were dropped.
- **Variance Threshold** at 0.01. Numerical features with near-zero variance were removed.

This reduced the feature space from 42 down to 28 meaningful features.

### 3. Target Variable Encoding

The `sfdm2` column was mapped to a numeric scale:

```
no(M2 and SIP pres)  -> 1
adl>=4 (>=5 if sur)  -> 2
SIP>=30              -> 3
Coma or Intub        -> 4
<2 mo. follow-up     -> 5
```

### 4. Model Training

Six models were trained and compared. Each handled class imbalance differently:

| Model | Imbalance Strategy |
|-------|--------------------|
| Logistic Regression | class_weight='balanced' |
| Random Forest | class_weight='balanced' |
| XGBoost | compute_class_weight + sample_weight |
| MLP Neural Network | SMOTE |
| Gradient Boosting | SMOTE |
| Naive Bayes | SMOTE |

All models were evaluated using **5-fold cross-validation**.

### 5. Evaluation Metrics

Standard accuracy is not enough for this problem because:
- The classes are imbalanced
- The target is ordinal, meaning predicting Level 5 for a Level 1 patient is much worse than being off by one level

So I used:

- **Accuracy** - overall correctness
- **F1 Macro** - treats all classes equally, important for imbalanced data
- **Quadratic Weighted Kappa (QWK)** - penalizes larger ordinal errors more heavily, the most clinically relevant metric here
- **Custom Score** - average of F1 and QWK, used to select the best model

### 6. Results

| Model | Accuracy | F1 Macro | QWK | Custom Score |
|-------|----------|----------|-----|--------------|
| Logistic Regression | 60.43% | 45.71% | 0.7460 | 0.6016 |
| Random Forest | 75.99% | 42.27% | 0.8748 | 0.6487 |
| **XGBoost** | **73.27%** | **52.02%** | **0.8704** | **0.6953** |
| MLP | 35.09% | 26.73% | 0.5213 | 0.3943 |
| Gradient Boosting | 71.99% | 52.85% | 0.8595 | 0.6940 |
| Naive Bayes | 56.06% | 41.48% | 0.6898 | 0.5523 |

**XGBoost** was selected as the best model with a QWK of 0.8704, which falls in the "almost perfect" agreement range. It struck the best balance between F1 and QWK across all five folds.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/tanmayi123/early-warning-system.git
cd early-warning-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# On Mac/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

The training script will automatically download the data from the UCI repository on first run. If you want to download it manually first:

```bash
mkdir data
python -c "import pandas as pd; pd.read_csv('https://archive.ics.uci.edu/static/public/880/data.csv').to_csv('data/data.csv', index=False); print('Done')"
```

### 5. Train the model

This runs the full pipeline, trains all 6 models with cross-validation, and saves the best one.

```bash
python train_model.py
```

This will take around 10 to 15 minutes. When finished you will see:

```
[DONE] Saved to 'models/' folder:
  - best_model.pkl  (XGBoost)
  - scaler.pkl
  - feature_names.pkl
  - model_results.json
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
scipy
streamlit
plotly
```

Install all at once with:

```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

**Why QWK over accuracy?**
The severity scale is ordinal. A model that predicts Level 4 for a Level 5 patient is much better than one that predicts Level 1. QWK captures this by penalizing larger errors quadratically. Accuracy treats all misclassifications the same, which is not appropriate here.

**Why two imputation strategies?**
Numerical missing values are well-handled by KNN imputation since it uses proximity in feature space. Categorical columns like disease group or DNR status need a different approach because they do not have a meaningful numeric distance. Random Forest imputation predicts the missing category using the rest of the data, which respects the categorical nature of those features.

**Why three layers of feature selection?**
Starting with 42 features, many of which were noisy or redundant, directly training a model would have hurt performance and interpretability. Correlation filtering removes features with no linear relationship to the target. Cramer's V does the same for categorical features. Variance thresholding removes features that barely change across patients and therefore carry no useful signal.

**Why a custom composite score?**
F1 Macro and QWK measure different things. F1 tells you how well the model handles each class. QWK tells you how ordinal the errors are. Using the average of both to select the best model ensures we pick something that is both class-fair and clinically sensible.

---

## Disclaimer

This tool is built for educational and portfolio purposes. It is not validated for clinical use and should not be used to make real medical decisions.

---

## Author

Built by Tanmayi Shurpali
