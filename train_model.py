"""
train_model.py
==============
Run this script ONCE to:
  1. Load & preprocess the SUPPORT2 dataset
  2. Train all 6 models with 5-fold cross-validation
  3. Compare models and save the best one
  4. Save scaler, feature names, and results JSON

Usage:
    python train_model.py
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.stats import chi2_contingency

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_URL  = "https://archive.ics.uci.edu/static/public/880/data.csv"
DATA_PATH = os.path.join("data", "data.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading data from local file: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"[INFO] Downloading data from UCI repository...")
        try:
            df = pd.read_csv(DATA_URL)
            df.to_csv(DATA_PATH, index=False)
            print(f"[INFO] Saved locally to {DATA_PATH}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download data and no local file found.\n"
                f"Please download the dataset manually from:\n"
                f"  {DATA_URL}\n"
                f"and save it to: {DATA_PATH}\n"
                f"Original error: {e}"
            )
    return df

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("[INFO] Preprocessing...")

    # Drop leakage columns
    df = df.drop(columns=["death", "hospdead"], errors="ignore").copy()

    # ── Map target to numeric ────────────────────────────────────────────────
    sfdm2_map = {
        "no(M2 and SIP pres)":  1,
        "adl>=4 (>=5 if sur)":  2,
        "SIP>=30":               3,
        "Coma or Intub":         4,
        "<2 mo. follow-up":      5,
    }
    df["sfdm2_numeric"] = df["sfdm2"].map(sfdm2_map)

    # ── KNN impute numerical columns ─────────────────────────────────────────
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    knn = KNNImputer(n_neighbors=5)
    df[numerical_cols] = knn.fit_transform(df[numerical_cols])

    # ── Predictive impute categorical columns ─────────────────────────────────
    missing_cats = ["income", "race", "dnr", "sfdm2"]

    # Build a fully numeric copy: encode every column (object or not)
    df_num = pd.DataFrame(index=df.index)
    col_encoders = {}
    for c in df.columns:
        if df[c].dtype == "object" or c in missing_cats:
            le_c = LabelEncoder()
            df_num[c] = le_c.fit_transform(df[c].astype(str).fillna("__missing__"))
            col_encoders[c] = le_c
        else:
            df_num[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for col in missing_cats:
        missing_mask = df[col].isnull()
        if not missing_mask.any():
            continue
        feature_cols = [c for c in df_num.columns if c not in missing_cats + ["id"]]
        X_all = df_num[feature_cols].values.astype(float)
        y_all = df_num[col].values.astype(int)
        X_train_imp = X_all[~missing_mask.values]
        y_train_imp = y_all[~missing_mask.values]
        X_pred_imp  = X_all[missing_mask.values]
        imp_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        imp_model.fit(X_train_imp, y_train_imp)
        predicted = imp_model.predict(X_pred_imp)
        le = col_encoders[col]
        df.loc[missing_mask, col] = le.inverse_transform(predicted.astype(int))

    # ── Outlier removal (IQR) ────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    Q1, Q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[num_cols] < Q1 - 1.5*IQR) | (df[num_cols] > Q3 + 1.5*IQR)).any(axis=1)]
    print(f"[INFO] Shape after outlier removal: {df.shape}")

    # ── Correlation-based feature selection (numerical) ──────────────────────
    num_cols2 = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols2 = [c for c in num_cols2 if c != "sfdm2_numeric"]
    corr = df[num_cols2 + ["sfdm2_numeric"]].corr()
    low_corr = corr["sfdm2_numeric"].abs()
    drop_num = low_corr[low_corr < 0.1].index.tolist()
    df = df.drop(columns=drop_num, errors="ignore")

    # ── Cramér's V for categorical ────────────────────────────────────────────
    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = chi2_contingency(ct)[0]
        n = ct.sum().sum()
        phi2 = chi2 / n
        r, k = ct.shape
        phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rc = r - ((r-1)**2)/(n-1)
        kc = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2c / min(rc-1, kc-1)) if min(rc-1, kc-1) > 0 else 0

    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if "sfdm2" in cat_cols:
        cv_scores = {c: cramers_v(df[c], df["sfdm2"]) for c in cat_cols if c != "sfdm2"}
        drop_cat = [c for c, v in cv_scores.items() if v < 0.1]
        df = df.drop(columns=drop_cat, errors="ignore")

    # ── Variance threshold ───────────────────────────────────────────────────
    cat_cols2 = df.select_dtypes(exclude=["number"]).columns.tolist()
    num_df = df.select_dtypes(include=["number"])
    sel = VarianceThreshold(threshold=0.01)
    reduced = sel.fit_transform(num_df)
    sel_cols = num_df.columns[sel.get_support()]
    df = pd.concat([
        pd.DataFrame(reduced, columns=sel_cols, index=df.index),
        df[cat_cols2]
    ], axis=1)

    # ── Drop original sfdm2 string column ───────────────────────────────────
    df = df.drop(columns=["sfdm2"], errors="ignore")

    # ── One-hot encode remaining categoricals ────────────────────────────────
    ohe_cols = [c for c in ["dzgroup", "dzclass", "ca", "dnr"] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    print(f"[INFO] Final shape after preprocessing: {df.shape}")
    return df


# ── 3. Cross-validate a model ────────────────────────────────────────────────
def cross_validate_model(name, model, X, y, needs_scaling=False, needs_smote=False, xgb_mode=False):
    print(f"[INFO] Cross-validating: {name}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, qwks = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if needs_scaling:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        if needs_smote:
            counts = pd.Series(y_tr).value_counts()
            k = min(5, counts.min() - 1)
            if k > 0 and counts.min() / counts.max() < 0.5:
                X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)

        if xgb_mode:
            y_tr_adj = pd.Series(y_tr).values - 1
            y_te_adj = np.array(y_te) - 1
            cw = compute_class_weight("balanced", classes=np.unique(y_tr_adj), y=y_tr_adj)
            cwd = dict(zip(np.unique(y_tr_adj), cw))
            sw = pd.Series(y_tr_adj).map(cwd).values
            model.fit(X_tr, y_tr_adj, sample_weight=sw)
            y_pred = model.predict(X_te) + 1
            y_te_use = y_te
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_te_use = y_te

        accs.append(accuracy_score(y_te_use, y_pred))
        f1s.append(f1_score(y_te_use, y_pred, average="macro", zero_division=0))
        qwks.append(cohen_kappa_score(y_te_use, y_pred, weights="quadratic"))

    return {
        "accuracy":  round(float(np.mean(accs)),  4),
        "f1_macro":  round(float(np.mean(f1s)),   4),
        "qwk":       round(float(np.mean(qwks)),  4),
        "custom":    round(float((np.mean(f1s) + np.mean(qwks)) / 2), 4),
    }


# ── 4. Train final model on full data ────────────────────────────────────────
def train_final_model(name, model, X_train, y_train, needs_scaling, needs_smote, xgb_mode, scaler):
    if needs_scaling:
        X_train = scaler.fit_transform(X_train)

    if needs_smote:
        counts = pd.Series(y_train).value_counts()
        k = min(5, counts.min() - 1)
        if k > 0 and counts.min() / counts.max() < 0.5:
            X_train, y_train = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_train, y_train)

    if xgb_mode:
        y_adj = np.array(y_train) - 1
        cw = compute_class_weight("balanced", classes=np.unique(y_adj), y=y_adj)
        cwd = dict(zip(np.unique(y_adj), cw))
        sw = pd.Series(y_adj).map(cwd).values
        model.fit(X_train, y_adj, sample_weight=sw)
    else:
        model.fit(X_train, y_train)

    return model


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():
    # Load & preprocess
    df_raw = load_data()
    df = preprocess(df_raw)

    X = df.drop(columns=["sfdm2_numeric"])
    y = df["sfdm2_numeric"].astype(int)
    feature_names = X.columns.tolist()

    # Model registry
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=500),
            "needs_scaling": True, "needs_smote": False, "xgb_mode": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
            "needs_scaling": False, "needs_smote": False, "xgb_mode": False,
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric="mlogloss"),
            "needs_scaling": False, "needs_smote": False, "xgb_mode": True,
        },
        "MLP": {
            "model": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
            "needs_scaling": False, "needs_smote": True, "xgb_mode": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            "needs_scaling": False, "needs_smote": True, "xgb_mode": False,
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "needs_scaling": False, "needs_smote": True, "xgb_mode": False,
        },
    }

    # Cross-validate all
    results = {}
    for name, cfg in models.items():
        results[name] = cross_validate_model(
            name, cfg["model"], X, y,
            cfg["needs_scaling"], cfg["needs_smote"], cfg["xgb_mode"]
        )
        print(f"  → Accuracy: {results[name]['accuracy']}  F1: {results[name]['f1_macro']}  QWK: {results[name]['qwk']}  Custom: {results[name]['custom']}")

    # Pick best by custom score
    best_name = max(results, key=lambda k: results[k]["custom"])
    print(f"\n[INFO] Best model: {best_name} (custom score: {results[best_name]['custom']})")

    # Train best model on full data
    scaler = StandardScaler()
    best_cfg = models[best_name]
    final_model = train_final_model(
        best_name, best_cfg["model"], X.copy(), y.copy(),
        best_cfg["needs_scaling"], best_cfg["needs_smote"], best_cfg["xgb_mode"], scaler
    )

    # Save artifacts
    joblib.dump(final_model,  os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    output = {
        "best_model": best_name,
        "results": results,
        "feature_names": feature_names,
        "xgb_mode": best_cfg["xgb_mode"],
        "needs_scaling": best_cfg["needs_scaling"],
        "label_map": {
            "1": "No Deterioration (M2 & SIP present)",
            "2": "Mild Deterioration (ADL ≥ 4)",
            "3": "Moderate Deterioration (SIP ≥ 30)",
            "4": "Severe Deterioration (Coma/Intubated)",
            "5": "Critical — < 2 Month Follow-up",
        }
    }
    with open(os.path.join(MODEL_DIR, "model_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] Saved to '{MODEL_DIR}/' folder:")
    print(f"  • best_model.pkl  ({best_name})")
    print(f"  • scaler.pkl")
    print(f"  • feature_names.pkl")
    print(f"  • model_results.json")


if __name__ == "__main__":
    main()