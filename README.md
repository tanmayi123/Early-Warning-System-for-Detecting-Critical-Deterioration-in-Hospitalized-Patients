# Early-Warning-System-for-Detecting-Critical-Deterioration-in-Hospitalized-Patients

📌 Project Overview

This project develops a machine learning–based early warning system to detect critical deterioration in hospitalized patients using the SUPPORT2 dataset
(UCI ML Repository).
The system is designed to support clinicians by predicting high-risk cases early, enabling proactive interventions and improving patient outcomes.

🎯 Objectives

1.Predict critical deterioration levels (1–5 scale) based on patient demographics, clinical measures, and hospital records.

2.Minimize false negatives to reduce missed detections of critical patients.

3.Balance accuracy with interpretability to ensure real-world clinical adoption.

🛠️ Tech Stack

Languages: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly)

ML Models: Logistic Regression, Random Forest, Gradient Boosting

Data Source: SUPPORT2 dataset (9,105 patient records, 42 features)

Other Tools: Jupyter Notebook, GitHub

📊 Methodology


1.Data Cleaning & Preprocessing

2.Handled missing values using imputation strategies (median, mode)

3.Normalized and engineered features for improved model performance

4.Exploratory Data Analysis (EDA)

5.Visualized trends by demographics, disease categories, and severity levels

6.Identified top clinical and physiological predictors of deterioration

7.Model Training & Evaluation

8.Compared multiple models: Logistic Regression, Random Forest, Gradient Boosting

Key Metrics:

-> Accuracy: 90%+

->Precision: 85%+

->Recall (Sensitivity): 90%+

->False Negatives: <10%

->Interpretability & Deployment Readiness

-> Used feature importance and model interpretation techniques to ensure trust.

->Designed for potential integration with EHR systems for real-time alerts.

🚀 Results

1.Achieved high accuracy (90%+) while maintaining clinical interpretability

2.Reduced false negatives significantly, ensuring critical patients are not overlooked

3.Demonstrated scalability for hospital use with 9,000+ patient records
