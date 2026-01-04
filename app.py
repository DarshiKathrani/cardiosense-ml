import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --------------------------------------------------
# Page Config (IMPORTANT for graph sizing)
# --------------------------------------------------
st.set_page_config(
    page_title="CardioSense",
    layout="centered"
)

# --------------------------------------------------
# Global Styling (Warm + Professional)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #FAFAF9;
    font-family: "Segoe UI", sans-serif;
}

.header {
    text-align: center;
    margin-bottom: 25px;
}

.logo {
    font-size: 34px;
    font-weight: 700;
    color: #7C4A2D;
}

.tagline {
    font-size: 14px;
    color: #6B7280;
}

.card {
    background: white;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

.metric {
    font-size: 26px;
    font-weight: 700;
    color: #7C4A2D;
}

.label {
    font-size: 13px;
    color: #6B7280;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="header">
    <div class="logo">🫀 CardioSense</div>
    <div class="tagline">Clear insights. Calm decisions.</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Navigation
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Dashboard", "Data Source", "Model Details", "Prediction"]
)

# ==================================================
# DASHBOARD
# ==================================================
with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "<div class='card'><div class='metric'>66,489</div>"
            "<div class='label'>Clean Records</div></div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            "<div class='card'><div class='metric'>74%</div>"
            "<div class='label'>Test Accuracy</div></div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            "<div class='card'><div class='metric'>Random Forest</div>"
            "<div class='label'>Final Model</div></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='card'>"
        "CardioSense is a cardiovascular risk assessment system built using a "
        "<b>Random Forest classifier</b>. The model evaluates combined clinical "
        "and lifestyle factors to predict relative cardiovascular risk in a "
        "clear and interpretable manner."
        "</div>",
        unsafe_allow_html=True
    )


# ==================================================
# DATA SOURCE
# ==================================================
with tab2:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Final Rows", "66,489")
    c2.metric("Duplicates Removed", "24")
    c3.metric("Input Features", "11")

    st.subheader("Target Distribution")

    fig1, ax1 = plt.subplots(figsize=(3.8, 2.6))
    ax1.bar(
        ["No Disease", "Disease"],
        [33921, 32568],
        color=["#E5C9A6", "#C97C5D"]
    )
    ax1.set_ylabel("Records", fontsize=9)
    ax1.tick_params(labelsize=9)
    ax1.spines[['top','right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("<div class='card'>✔ Target classes are well balanced, making accuracy (%) a reliable metric.</div>", unsafe_allow_html=True)

    st.subheader("Correlation with Cardiovascular Disease")

    features = ["Systolic BP", "Diastolic BP", "Age", "Cholesterol", "Weight"]
    values = [43, 33, 24, 22, 18]  # percentages

    fig2, ax2 = plt.subplots(figsize=(4, 2.8))
    ax2.barh(features, values, color="#7C4A2D")
    ax2.invert_yaxis()
    ax2.set_xlabel("Correlation (%)", fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.spines[['top','right']].set_visible(False)
    ax2.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig2)

# ==================================================
# MODEL DETAILS
# ==================================================
with tab3:
    st.subheader("Final Model: Random Forest Classifier")

    st.markdown(
        "<div class='card'>"
        "<b>Random Forest Classifier</b> was selected as the final model for "
        "CardioSense due to its stable predictions, ability to handle feature "
        "interactions, and interpretability. Unlike margin-based models, "
        "Random Forest provides consistent and reliable classification behavior "
        "on real-world clinical data."
        "</div>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # Model Configuration
    # -------------------------------
    st.subheader("Model Configuration")

    config_df = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Number of Trees (n_estimators)",
            "Max Depth",
            "Min Samples Split",
            "Train–Test Split",
            "Hyperparameter Tuning"
        ],
        "Value": [
            "Random Forest Classifier",
            "100 / 200 (Grid Search)",
            "None / 10 / 20 (Grid Search)",
            "2 / 5 (Grid Search)",
            "80% Train – 20% Test",
            "GridSearchCV (5-fold cross-validation)"
        ]
    })

    st.dataframe(config_df, use_container_width=True)

    # -------------------------------
    # Performance Metrics
    # -------------------------------
    st.subheader("Performance Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "74%")
    c2.metric("Precision (Avg)", "74%")
    c3.metric("Recall (Avg)", "73%")

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = np.array([
        [5374, 1437],
        [2073, 4414]
    ])

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(cm, cmap="Oranges")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # -------------------------------
    # Classification Report
    # -------------------------------
    st.subheader("Classification Report")

    report_df = pd.DataFrame({
        "Class": ["No Disease (0)", "Disease (1)"],
        "Precision (%)": [72, 75],
        "Recall (%)": [79, 68],
        "F1-score (%)": [75, 72]
    })

    st.dataframe(report_df, use_container_width=True)

    st.caption(
        "Overall Accuracy: 74% | Balanced performance across both classes "
        "with slightly higher recall for non-disease cases."
    )

        # -------- Prediction & Probability --------
       # Get decision score from SVM
with tab4:
    st.subheader("🫀 Cardiovascular Disease Prediction")
    st.caption("Enter complete patient details to predict the presence of heart disease")

    try:
        model = pickle.load(open("random_forest_model.pkl", "rb"))
    except:
        model = None
        st.error("Trained model not found. Please load svm_model.pkl")

    st.markdown("---")

    with st.form("prediction_form_v1"):
        col1, col2 = st.columns(2)

        # -------- Column 1 --------
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
            height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
            ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=200, value=80)

        # -------- Column 2 --------
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            cholesterol = st.selectbox(
                "Cholesterol Level",
                ["Normal", "Above Normal", "Well Above Normal"]
            )
            glucose = st.selectbox(
                "Glucose Level",
                ["Normal", "Above Normal", "Well Above Normal"]
            )
            smoke = st.selectbox("Smoking", ["No", "Yes"])
            alco = st.selectbox("Alcohol Intake", ["No", "Yes"])
            active = st.selectbox("Physical Activity", ["Yes", "No"])

        submitted = st.form_submit_button("🔍 Predict")

if submitted and model is not None:

    # ----- Encode Gender -----
    if gender == "Male":
        gender_val = 1
    else:
        gender_val = 2

    # ----- Encode Cholesterol -----
    if cholesterol == "Normal":
        cholesterol_val = 1
    elif cholesterol == "Above Normal":
        cholesterol_val = 2
    else:
        cholesterol_val = 3

    # ----- Encode Glucose -----
    if glucose == "Normal":
        glucose_val = 1
    elif glucose == "Above Normal":
        glucose_val = 2
    else:
        glucose_val = 3

    # ----- Encode Binary Fields -----
    smoke_val = 1 if smoke == "Yes" else 0
    alco_val = 1 if alco == "Yes" else 0
    active_val = 1 if active == "Yes" else 0

    # ----- Create Input in EXACT Training Order -----
    X = np.array([[
        gender_val,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol_val,
        glucose_val,
        smoke_val,
        alco_val,
        active_val,
        age
    ]])


        # -------- Final Prediction --------
    prediction = model.predict(X)[0]

    st.markdown("### Prediction Result")
        
    if prediction == 1:
        st.markdown("""
        <div style="
            background:#FEF2F2;
            border-left:6px solid #DC2626;
            padding:18px;
            border-radius:10px;">
            <h3 style="color:#7F1D1D;">⚠️ Higher Cardiovascular Risk</h3>
            <p style="color:#7F1D1D; font-size:15px;">
            Based on learned patterns from the dataset, the model predicts a higher
            cardiovascular risk profile.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background:#ECFDF5;
            border-left:6px solid #16A34A;
            padding:18px;
            border-radius:10px;">
            <h3 style="color:#065F46;">🟢 Lower Cardiovascular Risk</h3>
            <p style="color:#064E3B; font-size:15px;">
            Based on learned patterns from the dataset, the model predicts a lower
            cardiovascular risk profile.
            </p>
        </div>
        """, unsafe_allow_html=True)
