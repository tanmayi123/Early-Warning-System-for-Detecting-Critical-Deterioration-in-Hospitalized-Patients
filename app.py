"""
app.py - Early Warning System for Critical Patient Deterioration
Run with: streamlit run app.py
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Early Warning System",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_artifacts():
    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    with open("models/model_results.json") as f:
        results = json.load(f)
    return model, scaler, feature_names, results

model, scaler, feature_names, results = load_artifacts()

LABEL_MAP = {
    1: ("🟢", "Level 1 - No Deterioration",          "Patient shows no signs of critical deterioration. Standard monitoring is recommended."),
    2: ("🟡", "Level 2 - Mild Deterioration",         "Mild functional decline detected. Increased monitoring and early intervention are advised."),
    3: ("🟠", "Level 3 - Moderate Deterioration",     "Moderate deterioration indicated. A clinical review and possible intervention are required."),
    4: ("🔴", "Level 4 - Severe Deterioration",       "Severe deterioration detected. Immediate clinical attention is strongly recommended."),
    5: ("🚨", "Level 5 - Critical / < 2mo Follow-up", "Critical risk level. Urgent intervention is required. Patient may have less than 2 months of follow-up."),
}

RISK_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}

with st.sidebar:
    st.title("EWS Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Project Overview", "Live Prediction", "Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        **Dataset:** SUPPORT2 (UCI ML)  
        **Patients:** 9,105 records  
        **Features:** 42 clinical variables  
        **Best Model:** XGBoost  
        """
    )
    st.markdown("---")
    st.caption("Built for clinical decision support. Not a substitute for medical advice.")


# ==============================================================================
# PAGE 1 - PROJECT OVERVIEW
# ==============================================================================
if page == "Project Overview":
    st.title("Early Warning System for Critical Patient Deterioration")
    st.markdown(
        """
        This project builds a machine learning based early warning system that predicts
        the risk of critical deterioration in hospitalized patients using the SUPPORT2 dataset.
        The goal is to help clinicians identify high-risk patients early so they can intervene
        before conditions get worse
        """
    )

    best = results["results"][results["best_model"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",  results["best_model"])
    c2.metric("Accuracy",    f"{best['accuracy']*100:.1f}%")
    c3.metric("F1 Score",    f"{best['f1_macro']*100:.1f}%")
    c4.metric("QWK Score",   f"{best['qwk']:.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What The System Does")
        st.markdown("""
        The system takes in patient clinical data and predicts their deterioration
        severity on a scale of 1 to 5:

        | Level | Meaning |
        |-------|---------|
        | 🟢 1 | No deterioration - stable patient |
        | 🟡 2 | Mild deterioration - monitor closely |
        | 🟠 3 | Moderate deterioration - review needed |
        | 🔴 4 | Severe - coma or intubation |
        | 🚨 5 | Critical - less than 2 months follow-up |

        By flagging high-risk patients early, clinicians can act before conditions
        worsen, which can reduce mortality and improve overall outcomes.
        """)

    with col2:
        st.subheader("How it is Built")
        st.markdown("""
        **Data Pipeline:**
        - SUPPORT2 dataset: 9,105 patients, 42 features
        - KNN imputation for missing numerical values
        - Random Forest based predictive imputation for categorical columns
        - IQR based outlier removal
        - Multi-layer feature selection using correlation, Cramers V, and variance thresholding

        **Models Trained and Compared:**
        Logistic Regression, Random Forest, XGBoost (best),
        MLP Neural Network, Gradient Boosting, Naive Bayes

        **Evaluation:** 5-fold cross-validation using Accuracy, F1 Macro,
        Quadratic Weighted Kappa (QWK), and a custom composite score.
        """)

    st.markdown("---")

    st.subheader("Model Comparison")
    st.markdown("Here is how all six models performed across the key metrics after cross-validation.")

    model_names = list(results["results"].keys())
    accuracies  = [results["results"][m]["accuracy"]  for m in model_names]
    f1s         = [results["results"][m]["f1_macro"]  for m in model_names]
    qwks        = [results["results"][m]["qwk"]       for m in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=model_names, y=accuracies, marker_color="#3498db"))
    fig.add_trace(go.Bar(name="F1 Macro", x=model_names, y=f1s,        marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="QWK",      x=model_names, y=qwks,       marker_color="#e67e22"))
    fig.update_layout(
        barmode="group",
        title="Cross-Validated Performance Across All Models",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2 - LIVE PREDICTION
# ==============================================================================
elif page == "Live Prediction":
    st.title("Live Patient Risk Prediction")
    st.markdown("Fill in the patient values below and click Predict to get a risk level.")

    st.markdown("---")

    st.markdown("##### Clinical Scores")
    c1, c2, c3 = st.columns(3)
    scoma  = c1.number_input("SCOMA Score (0-100)",  min_value=0.0, max_value=100.0, value=0.0,  step=1.0,
        help="Support-SCOMA measures neurological/coma status. 0 means alert and oriented. Higher values indicate deeper unconsciousness.")
    sps    = c2.number_input("SPS Score",             min_value=0.0, max_value=10.0,  value=5.0,  step=0.1,
        help="SUPPORT Physiology Score. A composite measure of physiological stability based on vitals and lab values. Higher means more unstable.")
    aps    = c3.number_input("APS Score",             min_value=0.0, max_value=200.0, value=30.0, step=1.0,
        help="Acute Physiology Score, part of the APACHE system. Reflects severity of acute illness. Higher scores mean greater severity.")

    st.markdown("##### Time and Hospital Variables")
    c1, c2, c3 = st.columns(3)
    slos   = c1.number_input("Days from Study Entry (slos)",    min_value=0.0, max_value=2000.0, value=10.0,  step=1.0,
        help="Number of days from study entry to hospital discharge or death. Longer stays often indicate more complex cases.")
    dtime  = c2.number_input("Follow-up Time in Days (d.time)", min_value=0.0, max_value=2000.0, value=180.0, step=1.0,
        help="Total follow-up duration in days. Shorter follow-up time often correlates with worse outcomes.")
    hday   = c3.number_input("Day in Hospital (hday)",          min_value=0.0, max_value=200.0,  value=5.0,   step=1.0,
        help="The hospital day on which this assessment was taken. Earlier assessments reflect the acute admission phase.")

    st.markdown("##### Cost Variables")
    c1, c2, c3 = st.columns(3)
    charges = c1.number_input("Total Charges ($)",    min_value=0.0, max_value=500000.0, value=15000.0, step=100.0,
        help="Total hospital charges billed to the patient. Higher values usually reflect longer stays or more intensive care.")
    totcst  = c2.number_input("Total Cost ($)",       min_value=0.0, max_value=500000.0, value=10000.0, step=100.0,
        help="Estimated total cost of hospital care. Correlated with resource utilization and illness severity.")
    totmcst = c3.number_input("Total Micro Cost ($)", min_value=0.0, max_value=100000.0, value=1000.0,  step=100.0,
        help="Cost of microbiology and lab tests. High values may suggest complex infectious workups or diagnostic uncertainty.")

    st.markdown("##### Survival Estimates")
    c1, c2, c3, c4 = st.columns(4)
    surv2m  = c1.number_input("2-Month Survival (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0,
        help="Physician-estimated probability that the patient survives 2 months. Lower values mean higher predicted mortality.")
    surv6m  = c2.number_input("6-Month Survival (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
        help="Physician-estimated probability that the patient survives 6 months. An important long-term prognostic indicator.")
    prg2m   = c3.number_input("2-Month Prognosis",    min_value=0.0, max_value=100.0, value=60.0, step=1.0,
        help="Model-estimated 2-month survival probability from the SUPPORT prognostic model.")
    prg6m   = c4.number_input("6-Month Prognosis",    min_value=0.0, max_value=100.0, value=40.0, step=1.0,
        help="Model-estimated 6-month survival probability from the SUPPORT prognostic model.")

    st.markdown("##### Vitals and Lab Values")
    c1, c2, c3, c4 = st.columns(4)
    hrt     = c1.number_input("Heart Rate (bpm)",         min_value=0.0, max_value=250.0, value=80.0, step=1.0,
        help="Heart rate in beats per minute. Normal range is 60 to 100 bpm. Values outside this range suggest cardiovascular instability.")
    alb     = c2.number_input("Albumin (g/dL)",           min_value=0.0, max_value=10.0,  value=3.5,  step=0.1,
        help="Serum albumin level. Normal is 3.5 to 5.0 g/dL. Low albumin is a strong marker of malnutrition and poor prognosis.")
    avtisst = c3.number_input("AVTISS Score",             min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="Average Therapeutic Intervention Scoring System score. Reflects ICU intervention intensity. Higher means more interventions.")
    adlp    = c4.number_input("ADL Patient Score (0-7)",  min_value=0.0, max_value=7.0,   value=4.0,  step=1.0,
        help="Activities of Daily Living score reported by the patient. 0 means fully dependent, 7 means fully independent.")

    c1, c2 = st.columns(2)
    adls   = c1.number_input("ADL Sick Score (0-7)",      min_value=0.0, max_value=7.0, value=3.0, step=1.0,
        help="ADL score during the illness period. Lower scores indicate greater functional decline.")
    adlsc  = c2.number_input("ADL Caregiver Score (0-7)", min_value=0.0, max_value=7.0, value=4.0, step=1.0,
        help="ADL score as reported by the caregiver. Gives an external view of the patient's functional status.")

    st.markdown("##### Disease Group and DNR Status")
    c1, c2 = st.columns(2)
    dzgroup = c1.selectbox("Disease Group", [
        "ARF/MOSF w/o Malig", "CHF", "COPD", "Cirrhosis",
        "Colon Cancer", "Coma", "Lung Cancer", "MOSF w/Malig"
    ], help="The patient's primary disease category. Each group has a different risk profile and expected trajectory.")
    dnr_status = c2.selectbox("DNR Status", [
        "no dnr", "dnr before sadm", "dnr after sadm"
    ], help="Do Not Resuscitate order status. DNR before study admission is strongly associated with prognosis.")

    submitted = st.button("Predict Risk Level", use_container_width=True, type="primary")

    if submitted:
        user_input = {
            "slos": slos, "d.time": dtime, "scoma": scoma,
            "charges": charges, "totcst": totcst, "totmcst": totmcst,
            "avtisst": avtisst, "sps": sps, "aps": aps,
            "surv2m": surv2m, "surv6m": surv6m, "hday": hday,
            "prg2m": prg2m, "prg6m": prg6m, "hrt": hrt,
            "alb": alb, "adlp": adlp, "adls": adls, "adlsc": adlsc,
            "dzgroup_CHF":          1.0 if dzgroup == "CHF" else 0.0,
            "dzgroup_COPD":         1.0 if dzgroup == "COPD" else 0.0,
            "dzgroup_Cirrhosis":    1.0 if dzgroup == "Cirrhosis" else 0.0,
            "dzgroup_Colon Cancer": 1.0 if dzgroup == "Colon Cancer" else 0.0,
            "dzgroup_Coma":         1.0 if dzgroup == "Coma" else 0.0,
            "dzgroup_Lung Cancer":  1.0 if dzgroup == "Lung Cancer" else 0.0,
            "dzgroup_MOSF w/Malig": 1.0 if dzgroup == "MOSF w/Malig" else 0.0,
            "dnr_dnr before sadm":  1.0 if dnr_status == "dnr before sadm" else 0.0,
            "dnr_no dnr":           1.0 if dnr_status == "no dnr" else 0.0,
        }

        input_vector = [user_input.get(feat, 0.0) for feat in feature_names]
        input_df = pd.DataFrame([input_vector], columns=feature_names)

        raw_pred  = model.predict(input_df)[0]
        pred_label = int(raw_pred) + 1
        pred_label = max(1, min(5, pred_label))
        proba      = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        icon, label_text, explanation = LABEL_MAP[pred_label]
        color = RISK_COLORS[pred_label]

        st.markdown(
            f"""
            <div style="background-color:{color}22; border-left: 6px solid {color};
                        padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="color:{color}; margin:0">{icon} {label_text}</h2>
                <p style="margin:8px 0 0 0; font-size:1.05em">{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Level Probabilities")
            levels       = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
            colors       = [RISK_COLORS[i] for i in range(1, 6)]
            proba_padded = list(proba) + [0] * (5 - len(proba))
            fig_bar = go.Figure(go.Bar(
                x=levels[:len(proba)],
                y=proba_padded[:len(proba)],
                marker_color=colors[:len(proba)],
                text=[f"{p*100:.1f}%" for p in proba_padded[:len(proba)]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                yaxis=dict(range=[0, 1], title="Probability"),
                template="plotly_white",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Overall Risk Gauge")
            risk_score = sum((i + 1) * p for i, p in enumerate(proba_padded[:5]))
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Weighted Risk Score"},
                gauge={
                    "axis": {"range": [1, 5]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [1, 2], "color": "#d5f5e3"},
                        {"range": [2, 3], "color": "#fef9e7"},
                        {"range": [3, 4], "color": "#fdebd0"},
                        {"range": [4, 5], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": risk_score,
                    },
                },
            ))
            fig_gauge.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Input Summary")
        summary_df = pd.DataFrame({
            "Feature": list(user_input.keys()),
            "Value":   list(user_input.values()),
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.info(
            "This prediction comes from a machine learning model trained on the SUPPORT2 dataset. "
            "It is meant to support clinical decision-making, not replace it."
        )


# ==============================================================================
# PAGE 3 - MODEL PERFORMANCE
# ==============================================================================
elif page == "Model Performance":
    st.title("Model Performance and Evaluation")
    st.markdown(
        "Six models are trained and evaluated using 5-fold cross-validation on the preprocessed"
        "SUPPORT2 dataset. Here is a full breakdown of how each one performed."
    )

    if not results or "results" not in results:
        st.error("Could not load model results. Please make sure models/model_results.json exists")
        st.stop()

    st.markdown("---")

    st.subheader("Cross-Validated Metrics Across All Models")
    rows = []
    for name, m in results["results"].items():
        rows.append({
            "Model":        name,
            "Accuracy":     f"{m['accuracy']*100:.2f}%",
            "F1 (Macro)":   f"{m['f1_macro']*100:.2f}%",
            "QWK":          f"{m['qwk']:.4f}",
            "Custom Score": f"{m['custom']:.4f}",
            "Best Model":   "Yes" if name == results["best_model"] else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Radar Chart Comparison")
    st.markdown("This chart shows all four metrics at once so you can visually compare the strengths and weaknesses of each model")
    categories   = ["Accuracy", "F1 Macro", "QWK", "Custom Score"]
    fig_radar    = go.Figure()
    colors_radar = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
    for i, (name, m) in enumerate(results["results"].items()):
        vals = [m["accuracy"], m["f1_macro"], m["qwk"], m["custom"]]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line_color=colors_radar[i % len(colors_radar)],
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What is QWK?")
        st.markdown("""
        **Quadratic Weighted Kappa (QWK)** measures how well the predicted severity levels
        agree with the actual ones, while penalizing bigger mistakes more heavily

        Since the target is ordinal (a scale of 1 to 5), predicting Level 5 for a patient
        who is actually Level 1 is much worse than being off by one level. QWK captures
        that difference, which makes it the most clinically meaningful metric for this problem

        | QWK Range | Interpretation |
        |-----------|----------------|
        | < 0.2     | Slight agreement |
        | 0.2 to 0.4 | Fair |
        | 0.4 to 0.6 | Moderate |
        | 0.6 to 0.8 | Substantial |
        | > 0.8     | Almost perfect |
        """)

    with col2:
        st.subheader("Why XGBoost?")
        best = results["results"]["XGBoost"]
        st.markdown(f"""
        XGBoost came out on top based on the custom score, which averages F1 and QWK
        This balances both per-class accuracy and ordinal prediction quality

        - **Accuracy:** {best['accuracy']*100:.1f}% on a 5-class imbalanced dataset
        - **F1 Macro:** {best['f1_macro']*100:.1f}% across all severity classes
        - **QWK:** {best['qwk']:.3f}, which falls in the "almost perfect" range
        - Gradient boosting handles non-linear clinical relationships better than
          linear or probabilistic models
        - Class weights were applied to avoid bias toward the majority class
        """)

    st.markdown("---")

    st.subheader("Feature Importance")
    st.markdown("These are the features XGBoost relied on most when making predictions.")
    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True).tail(15)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color="#3498db",
        ))
        fig_imp.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance Score",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type.")

    st.markdown("---")

    st.subheader("Understanding the Scores in Context")
    st.markdown(
        "The numbers might not look impressive at first"
        "but the results hold up well once we understand what is being predicted."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Why Accuracy Alone Is Misleading")
        st.markdown("""
        This is a 5-class classification problem with significant class imbalance.
        Some deterioration levels are much rarer than others. A model that just
        predicts the majority class every time could still score high on accuracy
        while being completely useless clinically.

        That is why I focused on:
        - **F1 Macro**, which treats all classes equally regardless of how common they are
        - **QWK**, which penalizes large ordinal errors more than small ones
        - **Class balancing** using SMOTE and class weights depending on the model
        """)

        st.markdown("#### Why This Problem Is Hard")
        st.markdown("""
        Clinical deterioration depends on dozens of interacting factors. The SUPPORT2
        dataset also has a lot of missing values, which I handled using KNN imputation
        for numerical columns and Random Forest based predictive imputation for categorical ones.
        On top of that, the target is ordinal, not just categorical, so standard classification
        metrics only tell part of the story. Getting a QWK above 0.85 on data like this is actually a strong result, indicating that the model is not just getting the right answer, but when it is wrong, it is usually close to the true severity level.
        """)

    with c2:
        st.markdown("#### Score Summary")

        res       = results["results"]
        best_qwk  = max(v["qwk"]      for v in res.values())
        best_acc  = max(v["accuracy"]  for v in res.values())
        best_f1   = max(v["f1_macro"]  for v in res.values())
        worst_qwk = min(v["qwk"]       for v in res.values())

        st.markdown(f"""
        | Metric | Best Result | What It Means |
        |--------|-------------|---------------|
        | Accuracy | **{best_acc*100:.1f}%** | Strong for a 5-class imbalanced dataset |
        | F1 Macro | **{best_f1*100:.1f}%** | Good coverage across all minority classes |
        | QWK | **{best_qwk:.3f}** | Almost perfect ordinal agreement |
        | QWK range | {worst_qwk:.2f} to {best_qwk:.3f} | Consistent across model types |
        """)

        st.markdown("#### What I Did to Make This Rigorous")
        st.markdown(f"""
        - Compared 6 different models, not just one
        - Used 5-fold cross-validation so results are not overfitted
        - Chose QWK as the primary metric because the target is ordinal
        - Handled missing data carefully with two different imputation strategies
        - Applied three layers of feature selection to reduce noise
        - Addressed class imbalance differently per model based on its requirements
        - Selected {results['best_model']} based on a custom score that balances F1 and QWK
        """)

    st.info(
        "The goal of this system is not to achieve perfect accuracy. "
        "It is to reliably flag high-risk patients early so clinicians can act in time. "
        f"A QWK of {best_qwk:.3f} means that when the model is wrong, it is not catastrophically wrong."
    )