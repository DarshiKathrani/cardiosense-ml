import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ✅ ADD THIS IMMEDIATELY AFTER IMPORTS
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="CardioSense - Cardiovascular Risk Assessment",
    page_icon="🫀",
    layout="wide"
)

# --------------------------------------------------
# Theme Toggle State
# --------------------------------------------------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --------------------------------------------------
# Modern Color Scheme (No Gradients)
# --------------------------------------------------
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            'bg_primary': '#0F172A',
            'bg_secondary': '#1E293B',
            'card_bg': '#1E293B',
            'card_hover': '#334155',
            'text_primary': '#F1F5F9',
            'text_secondary': '#94A3B8',
            'text_muted': '#64748B',
            'accent_primary': '#3B82F6',
            'accent_secondary': '#8B5CF6',
            'accent_tertiary': '#EC4899',
            'success': '#10B981',
            'success_bg': '#064E3B',
            'success_border': '#059669',
            'warning': '#F59E0B',
            'warning_bg': '#78350F',
            'warning_border': '#D97706',
            'danger': '#EF4444',
            'danger_bg': '#7F1D1D',
            'danger_border': '#DC2626',
            'border': '#334155',
            'border_light': '#475569',
            'shadow': 'rgba(0, 0, 0, 0.5)',
        }
    else:
        return {
            'bg_primary': '#F8FAFC',
            'bg_secondary': '#FFFFFF',
            'card_bg': '#FFFFFF',
            'card_hover': '#F1F5F9',
            'text_primary': '#0F172A',
            'text_secondary': '#475569',
            'text_muted': '#64748B',
            'accent_primary': '#2563EB',
            'accent_secondary': '#7C3AED',
            'accent_tertiary': '#DB2777',
            'success': '#059669',
            'success_bg': '#D1FAE5',
            'success_border': '#10B981',
            'warning': '#D97706',
            'warning_bg': '#FEF3C7',
            'warning_border': '#F59E0B',
            'danger': '#DC2626',
            'danger_bg': '#FEE2E2',
            'danger_border': '#EF4444',
            'border': '#E2E8F0',
            'border_light': '#CBD5E1',
            'shadow': 'rgba(15, 23, 42, 0.08)',
        }

colors = get_theme_colors()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background: {colors['bg_primary']};
}}

/* Header */
.header-container {{
    background: {colors['accent_primary']};
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px {colors['shadow']}, 0 2px 4px -1px {colors['shadow']};
}}

.logo {{
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}}

.tagline {{
    font-size: 1rem;
    color: rgba(255,255,255,0.95);
    font-weight: 500;
}}

/* Cards */
.metric-card {{
    background: {colors['card_bg']};
    padding: 1.75rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px 0 {colors['shadow']}, 0 1px 2px 0 {colors['shadow']};
    border: 1px solid {colors['border']};
    transition: all 0.2s ease;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px {colors['shadow']}, 0 4px 6px -2px {colors['shadow']};
    border-color: {colors['accent_primary']};
}}

.metric-value {{
    font-size: 2.25rem;
    font-weight: 800;
    color: {colors['accent_primary']};
    margin-bottom: 0.5rem;
    line-height: 1;
}}

.metric-label {{
    font-size: 0.875rem;
    color: {colors['text_secondary']};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Info Card */
.info-card {{
    background: {colors['card_bg']};
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid {colors['accent_primary']};
    color: {colors['text_primary']};
    line-height: 1.7;
    box-shadow: 0 1px 3px 0 {colors['shadow']};
    margin: 1.5rem 0;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {colors['card_bg']};
    padding: 0.5rem;
    border-radius: 12px;
    border: 1px solid {colors['border']};
    box-shadow: 0 1px 2px 0 {colors['shadow']};
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: {colors['text_secondary']};
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.2s ease;
    border: none;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: {colors['card_hover']};
    color: {colors['text_primary']};
}}

.stTabs [aria-selected="true"] {{
    background: {colors['accent_primary']};
    color: white !important;
}}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    background-color: transparent !important;
}}

/* Form */
.stForm {{
    background: {colors['card_bg']};
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid {colors['border']};
    box-shadow: 0 1px 3px 0 {colors['shadow']};
}}

.form-section-header {{
    font-size: 1rem;
    font-weight: 700;
    color: {colors['text_primary']};
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {colors['accent_primary']};
}}

/* Inputs */
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {{
    background: {colors['bg_primary']} !important;
    color: {colors['text_primary']} !important;
    border: 1px solid {colors['border']} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {{
    border-color: {colors['accent_primary']} !important;
    box-shadow: 0 0 0 3px {colors['accent_primary']}20 !important;
}}

/* Labels */
label {{
    color: {colors['text_primary']} !important;
    font-weight: 600 !important;
}}

/* Submit Button */
.stButton > button {{
    width: 100%;
    background: {colors['accent_primary']};
    color: white;
    font-weight: 700;
    padding: 0.875rem 2rem;
    border-radius: 8px;
    border: none;
    font-size: 1rem;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.stButton > button:hover {{
    background: {colors['accent_secondary']};
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px {colors['shadow']};
}}

/* Metrics */
.stMetric {{
    background: {colors['card_bg']};
    padding: 1.25rem;
    border-radius: 12px;
    border: 1px solid {colors['border']};
    box-shadow: 0 1px 3px 0 {colors['shadow']};
}}

.stMetric label {{
    color: {colors['text_secondary']} !important;
}}

.stMetric [data-testid="stMetricValue"] {{
    color: {colors['text_primary']} !important;
}}

/* DataFrame Styling */
.stDataFrame {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {colors['border']};
    box-shadow: 0 1px 3px 0 {colors['shadow']};
}}

.stDataFrame thead tr th {{
    background: {colors['accent_primary']} !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    padding: 1rem !important;
    text-align: left !important;
    border: none !important;
}}

.stDataFrame tbody tr:nth-child(even) {{
    background: {colors['card_hover']} !important;
}}

.stDataFrame tbody tr:nth-child(odd) {{
    background: {colors['card_bg']} !important;
}}

.stDataFrame tbody tr:hover {{
    background: {colors['accent_primary']}15 !important;
}}

.stDataFrame tbody tr td {{
    color: {colors['text_primary']} !important;
    padding: 0.875rem 1rem !important;
    font-size: 0.875rem !important;
    border-bottom: 1px solid {colors['border']} !important;
}}

/* Success/Info boxes */
.stSuccess {{
    background: {colors['success_bg']} !important;
    border: 1px solid {colors['success_border']} !important;
    color: {colors['success']} !important;
}}

.stError {{
    background: {colors['danger_bg']} !important;
    border: 1px solid {colors['danger_border']} !important;
    color: {colors['danger']} !important;
}}

.stInfo {{
    background: {colors['card_bg']} !important;
    border: 1px solid {colors['accent_primary']} !important;
    color: {colors['text_primary']} !important;
}}

/* Divider */
hr {{
    border: none;
    height: 1px;
    background: {colors['border']};
    margin: 2rem 0;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header with Theme Toggle in Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    theme_option = st.radio(
        "Theme",
        ["☀️ Light Mode", "🌙 Dark Mode"],
        index=1 if st.session_state.dark_mode else 0,
        label_visibility="collapsed"
    )
    
    if (theme_option == "🌙 Dark Mode" and not st.session_state.dark_mode) or \
       (theme_option == "☀️ Light Mode" and st.session_state.dark_mode):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f"""
<div class="header-container">
    <div class="logo">🫀 CardioSense</div>
    <div class="tagline">AI-Powered Cardiovascular Risk Assessment Platform</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Navigation
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "📈 Data Insights", "🤖 Model Architecture", "🔮 Risk Prediction"]
)

# ==================================================
# DASHBOARD
# ==================================================
with tab1:
    st.markdown("### 📈 System Overview")
    
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>66,489</div>"
            f"<div class='metric-label'>📋 Clean Records</div></div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>74%</div>"
            f"<div class='metric-label'>🎯 Test Accuracy</div></div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>RF</div>"
            f"<div class='metric-label'>🌲 Algorithm</div></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        f"<div class='info-card'>"
        "<strong>CardioSense</strong> leverages a sophisticated <b>Random Forest classifier</b> "
        "to evaluate cardiovascular risk through comprehensive analysis of clinical biomarkers "
        "and lifestyle indicators. The system provides interpretable, evidence-based risk "
        "assessments to support informed healthcare decisions."
        "</div>",
        unsafe_allow_html=True
    )

# ==================================================
# DATA INSIGHTS
# ==================================================
with tab2:
    st.markdown("### 📊 Dataset Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("📋 Total Records", "66,489", delta="High Quality")
    c2.metric("🔄 Duplicates Removed", "24", delta="-0.04%")
    c3.metric("📈 Input Features", "11", delta="Optimized")

    st.markdown("### 🎯 Target Distribution Analysis")

    # Enhanced chart styling with better colors
    if st.session_state.dark_mode:
        plt.style.use('dark_background')
        chart_bg = colors['card_bg']
    else:
        plt.style.use('default')
        chart_bg = colors['card_bg']
    
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    categories = ["No Disease", "Disease Present"]
    values = [33921, 32568]
    
    # Beautiful gradient-like colors
    bar_colors = ['#10B981', '#EF4444']  # Green to Red
    bars = ax1.bar(categories, values, 
                   color=bar_colors,
                   edgecolor='none',
                   linewidth=0,
                   alpha=0.9,
                   width=0.6)
    
    ax1.set_ylabel("Number of Records", fontsize=11, fontweight='600', 
                   color=colors['text_primary'], labelpad=8)
    ax1.set_xlabel("Diagnosis Category", fontsize=11, fontweight='600', 
                   color=colors['text_primary'], labelpad=8)
    ax1.tick_params(labelsize=10, colors=colors['text_primary'], width=0)
    
    # Remove all spines for cleaner look
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    ax1.set_facecolor(chart_bg)
    fig1.patch.set_facecolor(chart_bg)
    ax1.grid(axis='y', alpha=0.15, linestyle='-', linewidth=1, color=colors['text_muted'])
    ax1.set_axisbelow(True)
    
    # Add value labels with better positioning
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 600,
                f'{int(val):,}',
                ha='center', va='bottom', fontweight='700', 
                color=colors['text_primary'], fontsize=11)
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1500,
                f'({val/sum(values)*100:.1f}%)',
                ha='center', va='bottom', fontweight='600', 
                color=colors['text_secondary'], fontsize=9)
    
    plt.tight_layout()
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig1)
    plt.close()

    st.markdown(f"<div class='info-card'>✅ <strong>Perfectly Balanced Dataset:</strong> The near-equal distribution between classes (51% vs 49%) ensures reliable accuracy metrics and unbiased model training.</div>", unsafe_allow_html=True)

    st.markdown("### 🔬 Feature Importance Analysis")

    features = ["Systolic BP", "Diastolic BP", "Age", "Cholesterol", "Weight"]
    values = [43, 33, 24, 22, 18]
    
    # Beautiful color palette
    feature_colors = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981']

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    bars = ax2.barh(features, values, 
                    color=feature_colors, 
                    edgecolor='none',
                    alpha=0.9,
                    height=0.6)
    
    ax2.invert_yaxis()
    ax2.set_xlabel("Correlation Strength (%)", fontsize=11, fontweight='600', 
                   color=colors['text_primary'], labelpad=8)
    ax2.set_title("Impact of Clinical Features on CVD Risk", fontsize=12, fontweight='700', 
                  color=colors['text_primary'], pad=15)
    ax2.tick_params(labelsize=10, colors=colors['text_primary'], width=0)
    
    # Remove all spines
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax2.set_facecolor(chart_bg)
    fig2.patch.set_facecolor(chart_bg)
    ax2.grid(axis='x', alpha=0.15, linestyle='-', linewidth=1, color=colors['text_muted'])
    ax2.set_axisbelow(True)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax2.text(val + 1.5, i, f'{val}%', va='center', fontweight='700', 
                color=colors['text_primary'], fontsize=10)
    
    plt.tight_layout()
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig2)
    plt.close()

# ==================================================
# MODEL ARCHITECTURE
# ==================================================
with tab3:
    st.markdown("### 🌲 Random Forest Architecture")

    st.markdown(
        f"<div class='info-card'>"
        "<strong>Why Random Forest?</strong><br><br>"
        "Our Random Forest Classifier excels in cardiovascular risk assessment through:<br>"
        "• <strong>Ensemble Learning:</strong> Combines predictions from multiple decision trees<br>"
        "• <strong>Feature Interactions:</strong> Captures complex relationships between clinical markers<br>"
        "• <strong>Robustness:</strong> Resistant to overfitting and outliers<br>"
        "• <strong>Interpretability:</strong> Provides feature importance rankings for clinical insights"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ Hyperparameter Configuration")

    config_df = pd.DataFrame({
        "Parameter": [
            "Algorithm Type",
            "Number of Trees",
            "Maximum Depth",
            "Min Samples Split",
            "Data Split Ratio",
            "Optimization Method"
        ],
        "Configuration": [
            "Random Forest Classifier",
            "100 / 200 (Grid Search)",
            "None / 10 / 20 (Grid Search)",
            "2 / 5 (Grid Search)",
            "80% Training – 20% Testing",
            "GridSearchCV (5-fold CV)"
        ]
    })

    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.markdown("### 📊 Performance Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Accuracy", "74%", delta="Robust")
    c2.metric("🎯 Precision", "74%", delta="Balanced")
    c3.metric("🎯 Recall", "73%", delta="Strong")

    st.markdown("### 🔍 Confusion Matrix Analysis")

    cm = np.array([[5374, 1437], [2073, 4414]])

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    
    from matplotlib.colors import LinearSegmentedColormap
    if st.session_state.dark_mode:
        cmap_colors = [colors['card_bg'], colors['accent_primary']]
    else:
        cmap_colors = ['#FFFFFF', colors['accent_primary']]
    
    cmap = LinearSegmentedColormap.from_list('custom', cmap_colors, N=100)
    
    im = ax3.imshow(cm, cmap=cmap, alpha=0.8)
    
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["No Disease", "Disease Present"], fontweight='600', fontsize=10, color=colors['text_primary'])
    ax3.set_yticklabels(["No Disease", "Disease Present"], fontweight='600', fontsize=10, color=colors['text_primary'])
    ax3.set_xlabel("Predicted Label", fontsize=11, fontweight='700', color=colors['text_primary'], labelpad=10)
    ax3.set_ylabel("True Label", fontsize=11, fontweight='700', color=colors['text_primary'], labelpad=10)
    ax3.set_facecolor(colors['card_bg'])
    fig3.patch.set_facecolor(colors['card_bg'])

    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / cm.sum()) * 100
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else colors['text_primary']
            ax3.text(j, i, f'{cm[i, j]:,}\n({percentage:.1f}%)', 
                    ha="center", va="center", 
                    fontsize=11, fontweight='700', color=text_color)

    plt.tight_layout()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig3)
    plt.close()

    st.markdown("### 📈 Classification Performance Report")

    report_df = pd.DataFrame({
        "Class": ["✅ No Disease (0)", "⚠️ Disease Present (1)"],
        "Precision (%)": [72, 75],
        "Recall (%)": [79, 68],
        "F1-Score (%)": [75, 72]
    })

    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.caption(
        "🎓 **Model Interpretation:** Overall accuracy of 74% with balanced performance. "
        "Higher recall for non-disease cases (79%) ensures fewer false negatives in healthy individuals."
    )

# ==================================================
# PREDICTION
# ==================================================

with tab4:
    st.markdown("### 🫀 Cardiovascular Risk Assessment")
    
    try:
        model = pickle.load(open("random_forest_model.pkl", "rb"))
    #     st.success("✅ Model loaded successfully")
     except:
        model = None
        st.error("⚠️ Model file not found. Please ensure random_forest_model.pkl is in the directory.")

    st.markdown("---")

    form_col, result_col = st.columns([3, 2])

    # ❌ DO NOT initialize prediction_result again

    with form_col:
        with st.form("prediction_form_v1"):
            st.markdown("<div class='form-section-header'>👤 Patient Demographics</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 Age (years)", 1, 120, 45)
                gender = st.selectbox("⚧ Gender", ["Female", "Male"])
                height = st.number_input("📏 Height (cm)", 120, 220, 165)
            with col2:
                weight = st.number_input("⚖️ Weight (kg)", 30, 200, 70)
                active = st.selectbox("🏃 Physical Activity", ["Yes", "No"])
                smoke = st.selectbox("🚬 Smoking Status", ["No", "Yes"])

            st.markdown("---")
            st.markdown("<div class='form-section-header'>🩺 Clinical Measurements</div>", unsafe_allow_html=True)

            col3, col4 = st.columns(2)
            with col3:
                ap_hi = st.number_input("💉 Systolic BP", 80, 250, 120)
                ap_lo = st.number_input("💉 Diastolic BP", 40, 200, 80)
                cholesterol = st.selectbox("🧪 Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
            with col4:
                glucose = st.selectbox("🍬 Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
                alco = st.selectbox("🍷 Alcohol Intake", ["No", "Yes"])

            submitted = st.form_submit_button("🔍 Analyze Risk Profile")

            if submitted and model is not None:
                X = np.array([[
                    1 if gender == "Male" else 2,
                    height, weight, ap_hi, ap_lo,
                    1 if cholesterol == "Normal" else (2 if cholesterol == "Above Normal" else 3),
                    1 if glucose == "Normal" else (2 if glucose == "Above Normal" else 3),
                    1 if smoke == "Yes" else 0,
                    1 if alco == "Yes" else 0,
                    1 if active == "Yes" else 0,
                    age
                ]])

                prediction = model.predict(X)[0]

                st.session_state.prediction_result = {
                    "prediction": prediction,
                    "age": age,
                    "ap_hi": ap_hi,
                    "ap_lo": ap_lo,
                    "cholesterol": cholesterol,
                    "glucose": glucose,
                    "smoke": smoke,
                    "active": active
                }

    # ================= RESULTS =================
    with result_col:
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            prediction = result["prediction"]

            risk_level = "HIGH" if prediction == 1 else "LOW"
            risk_color = "#EF4444" if prediction == 1 else "#10B981"
            needle_rotation = 135 if prediction == 1 else 45

            import streamlit.components.v1 as components

            st.markdown("### 📋 Risk Assessment Results")

            risk_level = "HIGH" if prediction == 1 else "LOW"
            risk_color = "#EF4444" if prediction == 1 else "#10B981"
            needle_rotation = 135 if prediction == 1 else 45
            
            components.html(
            f"""
            <div style="text-align:center;padding:2rem;
            background:{colors['card_bg']};
            border-radius:20px;
            border:2px solid {colors['border']};
            box-shadow:0 10px 40px {colors['shadow']}">
            
            <h3 style="margin-bottom:1rem;">🫀 Risk Meter</h3>
            
            <div style="position:relative;width:250px;height:150px;margin:auto">
            <svg width="250" height="150" viewBox="0 0 250 150" style="transform:scaleX(-1)">
            <path d="M25 125 A100 100 0 0 1 75 25" stroke="#10B981" stroke-width="25" fill="none"/>
            <path d="M75 25 A100 100 0 0 1 175 25" stroke="#F59E0B" stroke-width="25" fill="none"/>
            <path d="M175 25 A100 100 0 0 1 225 125" stroke="#EF4444" stroke-width="25" fill="none"/>
            </svg>
            
            <div style="
            position:absolute;
            bottom:10px;
            left:50%;
            width:4px;
            height:90px;
            background:{colors['text_primary']};
            transform:translateX(-50%) rotate({needle_rotation}deg);
            transform-origin:bottom;
            transition:transform 1s ease-out;">
            </div>
            </div>
            
            <h2 style="color:{risk_color};margin-top:1rem;">{risk_level} RISK</h2>
            <p style="color:{colors['text_secondary']};font-weight:600;">
            Cardiovascular Disease Probability
            </p>
            </div>
            """,
            height=360,
            )
            
            # ------------------ ANALYSIS DESCRIPTION ------------------
              
            # ------------------ ANALYSIS DESCRIPTION ------------------
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card" style="
                    background:{colors['danger_bg']};
                    border-color:{colors['danger_border']};">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title" style="color:{colors['danger']};">
                        Higher Cardiovascular Risk Detected
                    </div>
                    <div class="result-text" style="color:{colors['danger']};">
                        The AI model has identified clinical patterns indicating an
                        <strong>elevated cardiovascular risk profile</strong>.
                        <br><br>
                        <strong>Recommended Actions:</strong><br>
                        • Consult a cardiologist<br>
                        • Lifestyle and dietary changes<br>
                        • Regular BP & cholesterol monitoring<br>
                        • Preventive care planning
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="
                    background:{colors['success_bg']};
                    border-color:{colors['success_border']};">
                    <div class="result-icon">✅</div>
                    <div class="result-title" style="color:{colors['success']};">
                        Lower Cardiovascular Risk Profile
                    </div>
                    <div class="result-text" style="color:{colors['success']};">
                        The AI model indicates a
                        <strong>lower cardiovascular risk</strong> based on current inputs.
                        <br><br>
                        <strong>Maintain Health:</strong><br>
                        • Stay physically active<br>
                        • Balanced diet<br>
                        • Annual health checkups<br>
                        • Monitor vitals regularly
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # ------------------ DISCLAIMER ------------------
            st.info(
                "💡 **Medical Disclaimer:** This AI-powered assessment is a decision support tool "
                "and should NOT replace professional medical advice, diagnosis, or treatment. "
                "Always consult a qualified healthcare provider."
            )


        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem">
                <h2>🫀 Awaiting Analysis</h2>
                <p>Fill the form and click Analyze</p>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Result Card
    #         if prediction == 1:
    #             st.markdown(f"""
    #             <div style="
    #                 background: {colors['danger_bg']};
    #                 border: 2px solid {colors['danger_border']};
    #                 border-radius: 12px;
    #                 padding: 1.5rem;
    #                 box-shadow: 0 1px 3px 0 {colors['shadow']};">
    #                 <div style="color: {'#FCA5A5' if st.session_state.dark_mode else colors['danger']}; font-size: 1rem; font-weight: 700; margin-bottom: 1rem;">
    #                     ⚠️ Higher Risk Detected
    #                 </div>
    #                 <div style="color: {'#FCA5A5' if st.session_state.dark_mode else colors['danger']}; font-size: 0.875rem; line-height: 1.6;">
    #                     <strong>Recommended Actions:</strong><br>
    #                     • Consult a cardiologist<br>
    #                     • Lifestyle modifications<br>
    #                     • Regular monitoring<br>
    #                     • Preventive measures
    #                 </div>
    #             </div>
    #             """, unsafe_allow_html=True)
    #         else:
    #             st.markdown(f"""
    #             <div style="
    #                 background: {colors['success_bg']};
    #                 border: 2px solid {colors['success_border']};
    #                 border-radius: 12px;
    #                 padding: 1.5rem;
    #                 box-shadow: 0 1px 3px 0 {colors['shadow']};">
    #                 <div style="color: {'#6EE7B7' if st.session_state.dark_mode else colors['success']}; font-size: 1rem; font-weight: 700; margin-bottom: 1rem;">
    #                     ✅ Lower Risk Profile
    #                 </div>
    #                 <div style="color: {'#6EE7B7' if st.session_state.dark_mode else colors['success']}; font-size: 0.875rem; line-height: 1.6;">
    #                     <strong>Maintain Health:</strong><br>
    #                     • Continue healthy habits<br>
    #                     • Annual check-ups<br>
    #                     • Monitor vitals<br>
    #                     • Stay active
    #                 </div>
    #             </div>
    #             """, unsafe_allow_html=True)
            
    #         st.markdown("<br>", unsafe_allow_html=True)
            
    #         # Key Metrics Summary
    #         st.markdown(f"""
    #         <div style="
    #             background: {colors['card_bg']};
    #             border: 1px solid {colors['border']};
    #             border-radius: 12px;
    #             padding: 1.5rem;
    #             box-shadow: 0 1px 3px 0 {colors['shadow']};">
    #             <div style="color: {colors['text_primary']}; font-weight: 700; margin-bottom: 1rem;">📊 Key Metrics</div>
    #             <div style="color: {colors['text_primary']}; font-size: 0.875rem; line-height: 2;">
    #                 <strong>Age:</strong> {result['age']} years<br>
    #                 <strong>BP:</strong> {result['ap_hi']}/{result['ap_lo']} mmHg<br>
    #                 <strong>Cholesterol:</strong> {result['cholesterol']}<br>
    #                 <strong>Glucose:</strong> {result['glucose']}<br>
    #                 <strong>Smoking:</strong> {result['smoke']}<br>
    #                 <strong>Active:</strong> {result['active']}
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     else:
    #         st.markdown(f"""
    #         <div style="
    #             text-align: center;
    #             padding: 3rem 2rem;
    #             background: {colors['card_bg']};
    #             border-radius: 12px;
    #             box-shadow: 0 1px 3px 0 {colors['shadow']};
    #             border: 1px solid {colors['border']};">
                
    #             <div style="font-size: 3.5rem; margin-bottom: 1rem;">🫀</div>
    #             <div style="color: {colors['text_secondary']}; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
    #                 Awaiting Analysis
    #             </div>
    #             <div style="color: {colors['text_secondary']}; font-size: 0.875rem; line-height: 1.6;">
    #                 Fill in the patient details and click<br>"Analyze Risk Profile" to see results
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
    
    # st.markdown("---")
    # st.info("💡 **Medical Disclaimer:** This AI-powered assessment is a decision support tool and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions and personalized care plans.")
