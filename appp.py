# ============================================================
#   AI-BASED DIABETES PREDICTION SYSTEM
#   Streamlit Web App
#   Run: streamlit run app.py
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .result-diabetic {
        background-color: #fff5f5;
        border: 2px solid #E74C3C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-non-diabetic {
        background-color: #f0fff4;
        border: 2px solid #2ECC71;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Model not found: {e}\n\nMake sure model.pkl and scaler.pkl are in the models/ folder.")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 About This App")
    st.markdown("""
    <div class='sidebar-info'>
    This AI system predicts diabetes risk using a 
    <b>Random Forest</b> classifier trained on 
    <b>20,000 patient records</b> from the 
    Mendeley Diabetes Dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance")
    metrics = {
        "Accuracy":  "97.15%",
        "Precision": "97%+",
        "Recall":    "97%+",
        "F1-Score":  "97%+",
        "ROC-AUC":   "99%+"
    }
    for metric, value in metrics.items():
        st.markdown(f"**{metric}:** {value}")

    st.divider()
    st.markdown("### 🔬 Features Used")
    features_info = {
        "Age":               "Patient age in years",
        "BMI":               "Body Mass Index",
        "Physical Activity": "Activity level (Low/Med/High)",
        "Blood Pressure":    "Diastolic BP (mmHg)",
        "Cholesterol":       "Total cholesterol (mg/dL)",
        "Glucose":           "Blood glucose level (mg/dL)"
    }
    for feat, desc in features_info.items():
        st.markdown(f"• **{feat}**: {desc}")

    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.caption("This tool is for educational purposes only. "
               "Always consult a qualified healthcare professional "
               "for medical diagnosis.")

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🩺 AI-Based Diabetes Prediction System</h1>
    <p>Enter patient health data below to predict diabetes risk using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Feature Guide", "📋 About"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.subheader("👤 Enter Patient Details")
    st.markdown("Adjust the values below and click **Predict** to get the result.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🧑 Demographics")
        age = st.number_input(
            "Age (years)",
            min_value=1, max_value=120, value=35,
            help="Patient's age in years"
        )
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0, max_value=70.0,
            value=25.0, step=0.1,
            help="Body Mass Index: Underweight<18.5 | Normal 18.5-24.9 | Overweight 25-29.9 | Obese≥30"
        )

    with col2:
        st.markdown("#### 💊 Clinical Measurements")
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)",
            min_value=40, max_value=200, value=80,
            help="Diastolic blood pressure in mmHg. Normal: 60-80"
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=50, max_value=500, value=180,
            help="Total cholesterol. Normal: <200 | Borderline: 200-239 | High: ≥240"
        )
        glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=50, max_value=500, value=100,
            help="Fasting blood glucose. Normal: 70-99 | Prediabetes: 100-125 | Diabetes: ≥126"
        )

    with col3:
        st.markdown("#### 🏃 Lifestyle")
        physical_activity = st.selectbox(
            "Physical Activity Level",
            options=[2, 1, 0],
            format_func=lambda x: {
                2: "🟢 High — Regular intense exercise",
                1: "🟡 Moderate — Light exercise / walking",
                0: "🔴 Low — Sedentary lifestyle"
            }[x],
            help="Select your physical activity level"
        )

        st.markdown("#### 📈 BMI Category")
        if bmi < 18.5:
            st.info("Underweight")
        elif bmi < 25:
            st.success("Normal weight")
        elif bmi < 30:
            st.warning("Overweight")
        else:
            st.error("Obese")

        st.markdown("#### 🩸 Glucose Category")
        if glucose < 100:
            st.success("Normal")
        elif glucose < 126:
            st.warning("Prediabetes range")
        else:
            st.error("Diabetes range")

    st.divider()

    # ── PREDICT BUTTON ──
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button(
            "🔍 Predict Diabetes Risk",
            use_container_width=True,
            type="primary"
        )

    if predict_btn and model_loaded:
        # input_data   = np.array([[age, bmi, physical_activity,
        #                            blood_pressure, cholesterol, glucose]])
        # input_scaled = scaler.transform(input_data)
        # prediction   = model.predict(input_scaled)[0]
        # probability  = model.predict_proba(input_scaled)[0][1]
        FEATURES = ['Age', 'BMI', 'Physical Activity',
            'Blood Pressure', 'Cholesterol', 'Glucose']

        input_df     = pd.DataFrame(
            [[age, bmi, physical_activity,
            blood_pressure, cholesterol, glucose]],
            columns=FEATURES
        )
        input_scaled = scaler.transform(input_df)
        # Handle label encoding
        if prediction in [0, 1]:
            pred_label = "Diabetic" if prediction == 1 else "Non-Diabetic"
        else:
            pred_label = str(prediction)
            probability = model.predict_proba(input_scaled)[0].max()

        is_diabetic = "diabetic" in pred_label.lower() and "non" not in pred_label.lower()

        st.divider()
        st.subheader("🩺 Prediction Result")

        # Risk Level
        if probability >= 0.70:
            risk_level = "HIGH RISK"
            risk_icon  = "🔴"
            risk_color = "red"
        elif probability >= 0.40:
            risk_level = "MODERATE RISK"
            risk_icon  = "🟡"
            risk_color = "orange"
        else:
            risk_level = "LOW RISK"
            risk_icon  = "🟢"
            risk_color = "green"

        # Result Card
        if is_diabetic:
            st.markdown(f"""
            <div class='result-diabetic'>
                <h2>⚠️ {pred_label}</h2>
                <h3>{risk_icon} {risk_level}</h3>
                <h3>Probability: {probability*100:.1f}%</h3>
                <p>This patient shows signs of diabetes risk. Please consult a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-non-diabetic'>
                <h2>✅ {pred_label}</h2>
                <h3>{risk_icon} {risk_level}</h3>
                <h3>Probability: {probability*100:.1f}%</h3>
                <p>This patient shows low diabetes risk. Maintain a healthy lifestyle!</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction",   pred_label)
        m2.metric("Probability",  f"{probability*100:.1f}%")
        m3.metric("Risk Level",   f"{risk_icon} {risk_level}")
        m4.metric("Confidence",   f"{max(probability, 1-probability)*100:.1f}%")

        # Progress Bar
        st.markdown(f"**Diabetes Risk Probability: {probability*100:.1f}%**")
        st.progress(float(probability))

        st.divider()

        # Charts Row
        chart1, chart2 = st.columns(2)

        # ── Gauge Chart ──
        with chart1:
            st.markdown("#### 📊 Risk Gauge")
            fig, ax = plt.subplots(figsize=(7, 3))
            bar_color = '#E74C3C' if probability >= 0.7 else \
                        '#F39C12' if probability >= 0.4 else '#2ECC71'
            ax.barh(['Risk'], [probability*100], color=bar_color, height=0.5)
            ax.barh(['Risk'], [100 - probability*100],
                    left=[probability*100], color='#ECF0F1', height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.axvline(x=40, color='orange', linestyle='--',
                       alpha=0.8, label='Moderate (40%)')
            ax.axvline(x=70, color='red', linestyle='--',
                       alpha=0.8, label='High (70%)')
            ax.legend(fontsize=9)
            ax.set_title(f'Risk Score: {probability*100:.1f}%', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        # ── Feature Comparison Chart ──
        with chart2:
            st.markdown("#### 📈 Patient vs Normal Range")
            features    = ['Age',  'BMI',  'Blood\nPressure', 'Cholesterol', 'Glucose']
            patient_vals = [age,   bmi,    blood_pressure,    cholesterol,   glucose]
            normal_vals  = [35,    22.5,   75,                175,           90]

            x = np.arange(len(features))
            width = 0.35
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.bar(x - width/2, patient_vals, width,
                    label='Patient', color='#E74C3C' if is_diabetic else '#2ECC71',
                    alpha=0.8, edgecolor='white')
            ax2.bar(x + width/2, normal_vals, width,
                    label='Normal Range', color='#3498DB', alpha=0.8, edgecolor='white')
            ax2.set_xticks(x)
            ax2.set_xticklabels(features, fontsize=9)
            ax2.legend(fontsize=9)
            ax2.set_title('Patient vs Normal Values', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            st.pyplot(fig2)
            plt.close()

        # ── Input Summary Table ──
        st.divider()
        st.markdown("#### 📋 Patient Input Summary")
        summary_df = pd.DataFrame({
            'Feature': ['Age', 'BMI', 'Physical Activity',
                        'Blood Pressure', 'Cholesterol', 'Glucose'],
            'Patient Value': [
                f"{age} years",
                f"{bmi:.1f} kg/m²",
                {2: 'High', 1: 'Moderate', 0: 'Low'}[physical_activity],
                f"{blood_pressure} mmHg",
                f"{cholesterol} mg/dL",
                f"{glucose} mg/dL"
            ],
            'Normal Range': [
                "20–60 years",
                "18.5–24.9",
                "Moderate or High",
                "60–80 mmHg",
                "<200 mg/dL",
                "70–99 mg/dL"
            ],
            'Status': [
                "✅" if 20 <= age <= 60    else "⚠️",
                "✅" if bmi < 25           else "⚠️",
                "✅" if physical_activity >= 1 else "⚠️",
                "✅" if blood_pressure <= 80  else "⚠️",
                "✅" if cholesterol < 200     else "⚠️",
                "✅" if glucose < 100         else "⚠️"
            ]
        })
        st.table(summary_df)

        # ── Recommendations ──
        st.divider()
        st.markdown("#### 💡 Health Recommendations")
        recs = []
        if bmi >= 25:
            recs.append("🏃 Maintain a healthy weight through diet and exercise (BMI is elevated)")
        if glucose >= 100:
            recs.append("🍎 Monitor blood sugar — consider reducing sugar and refined carbs intake")
        if cholesterol >= 200:
            recs.append("🥗 Reduce saturated fats — consider heart-healthy diet")
        if blood_pressure > 80:
            recs.append("💊 Monitor blood pressure regularly")
        if physical_activity == 0:
            recs.append("🚶 Increase physical activity — aim for 30 minutes daily")
        if not recs:
            recs.append("✅ Keep up the healthy lifestyle — all indicators look good!")

        for rec in recs:
            st.markdown(f"- {rec}")

# ══════════════════════════════════════════════
# TAB 2 — FEATURE GUIDE
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📊 Feature Reference Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🩸 Glucose Levels")
        glucose_df = pd.DataFrame({
            'Category':    ['Normal', 'Prediabetes', 'Diabetes'],
            'Range':       ['70–99 mg/dL', '100–125 mg/dL', '≥126 mg/dL'],
            'Risk':        ['Low', 'Moderate', 'High']
        })
        st.table(glucose_df)

        st.markdown("#### ⚖️ BMI Categories")
        bmi_df = pd.DataFrame({
            'Category':    ['Underweight', 'Normal', 'Overweight', 'Obese'],
            'BMI Range':   ['< 18.5', '18.5–24.9', '25–29.9', '≥ 30'],
            'Diabetes Risk': ['Low', 'Low', 'Moderate', 'High']
        })
        st.table(bmi_df)

    with col2:
        st.markdown("#### 💓 Blood Pressure")
        bp_df = pd.DataFrame({
            'Category':    ['Normal', 'Elevated', 'High Stage 1', 'High Stage 2'],
            'Range':       ['< 80 mmHg', '80–89 mmHg', '90–99 mmHg', '≥ 100 mmHg'],
            'Action':      ['Monitor', 'Lifestyle change', 'Consult doctor', 'Urgent care']
        })
        st.table(bp_df)

        st.markdown("#### 🏃 Physical Activity Impact")
        pa_df = pd.DataFrame({
            'Level':         ['Low (Sedentary)', 'Moderate', 'High (Active)'],
            'Diabetes Risk': ['High', 'Moderate', 'Low'],
            'Recommendation': ['Start walking daily', 'Add intensity', 'Maintain routine']
        })
        st.table(pa_df)

# ══════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📋 About This Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        The AI-Based Diabetes Prediction System uses supervised
        machine learning to predict diabetes risk from 6 key
        health indicators.

        ### 📦 Dataset
        - **Source:** Mendeley Data
        - **Samples:** 20,000 patient records
        - **Features:** 15 health indicators
        - **Target:** Diabetic / Non-Diabetic

        ### 🤖 Model
        - **Algorithm:** Random Forest Classifier
        - **Trees:** 200 estimators
        - **Validation:** 5-Fold Cross Validation
        """)

    with col2:
        st.markdown("### 📊 Model Performance")
        perf_df = pd.DataFrame({
            'Metric':    ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Score'],
            'Score':     ['97%+',     '97%+',      '97%+',   '97%+',     '99%+',    '97.15%']
        })
        st.table(perf_df)

        st.markdown("""
        ### 🛠️ Tech Stack
        - **Python** 3.12
        - **Scikit-learn** — ML model
        - **Streamlit** — Web interface
        - **Pandas / NumPy** — Data processing
        - **Matplotlib / Seaborn** — Visualization
        - **Joblib** — Model persistence
        """)

    st.divider()
   

# ── Footer ──
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:13px;'>
    🩺 AI-Based Diabetes Prediction System 
</div>
""", unsafe_allow_html=True)