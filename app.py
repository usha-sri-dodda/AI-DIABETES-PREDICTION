# ─────────────────────────────────────────────
# app.py — Streamlit Diabetes Prediction App
# ─────────────────────────────────────────────
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model  = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Header ──
st.title("🩺 AI-Based Diabetes Prediction System")
st.markdown("Enter patient details below to predict diabetes risk.")
st.divider()

# ── Input Form ──
st.subheader("👤 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)",
                          min_value=1, max_value=120, value=35)
    bmi = st.number_input("BMI",
                          min_value=10.0, max_value=70.0,
                          value=25.0, step=0.1)
    physical_activity = st.selectbox(
        "Physical Activity Level",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Low", 1: "Moderate", 2: "High"}[x]
    )

with col2:
    blood_pressure = st.number_input("Blood Pressure (mmHg)",
                                      min_value=40, max_value=200, value=80)
    cholesterol    = st.number_input("Cholesterol (mg/dL)",
                                      min_value=50, max_value=400, value=180)
    glucose        = st.number_input("Glucose Level (mg/dL)",
                                      min_value=50, max_value=400, value=100)

st.divider()

# ── Predict Button ──
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    input_data = np.array([[age, bmi, physical_activity,
                            blood_pressure, cholesterol, glucose]])
    FEATURES = ['Age', 'BMI', 'Physical Activity',
                'Blood Pressure', 'Cholesterol', 'Glucose']
    input_df     = pd.DataFrame(input_data, columns=FEATURES)
    input_scaled = scaler.transform(input_df)
    # input_scaled = scaler.transform(input_data)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("🩺 Prediction Result")

    # Risk level
    if probability >= 0.70:
        risk = "HIGH RISK"
        color = "red"
        icon = "🔴"
    elif probability >= 0.40:
        risk = "MODERATE RISK"
        color = "orange"
        icon = "🟡"
    else:
        risk = "LOW RISK"
        color = "green"
        icon = "🟢"

    # Result display
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction",
                "Diabetic" if prediction == 1 else "Non-Diabetic")
    col2.metric("Probability", f"{probability*100:.1f}%")
    col3.metric("Risk Level", f"{icon} {risk}")

    # Progress bar
    st.markdown(f"**Diabetes Probability: {probability*100:.1f}%**")
    st.progress(float(probability))

    # Result message
    if prediction == 1:
        st.error(f"⚠️ This patient is likely **Diabetic**. "
                 f"Please consult a healthcare professional.")
    else:
        st.success(f"✅ This patient is likely **Non-Diabetic**. "
                   f"Maintain a healthy lifestyle.")

    # ── Gauge Chart ──
    st.divider()
    st.subheader("📊 Risk Gauge")

    fig, ax = plt.subplots(figsize=(8, 3))
    bar_color = '#E74C3C' if probability >= 0.7 else \
                '#F39C12' if probability >= 0.4 else '#2ECC71'
    ax.barh(['Risk'], [probability*100],
            color=bar_color, height=0.4)
    ax.barh(['Risk'], [100 - probability*100],
            left=[probability*100],
            color='#ECF0F1', height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    ax.axvline(x=40, color='orange', linestyle='--',
               alpha=0.7, label='Moderate threshold (40%)')
    ax.axvline(x=70, color='red', linestyle='--',
               alpha=0.7, label='High threshold (70%)')
    ax.legend(fontsize=8)
    ax.set_title(f'Diabetes Risk: {probability*100:.1f}%',
                 fontweight='bold')
    st.pyplot(fig)

    # ── Input Summary ──
    st.divider()
    st.subheader("📋 Patient Input Summary")
    summary_data = [
        ["Age",             f"{age} years",        "20–60 years",    "✅" if 20<=age<=60 else "⚠️"],
        ["BMI",             f"{bmi:.1f} kg/m²",    "18.5–24.9",      "✅" if bmi<25 else "⚠️"],
        ["Physical Act.",   {2:'High',1:'Moderate',0:'Low'}[physical_activity], "Moderate+", "✅" if physical_activity>=1 else "⚠️"],
        ["Blood Pressure",  f"{blood_pressure} mmHg", "60–80 mmHg",  "✅" if blood_pressure<=80 else "⚠️"],
        ["Cholesterol",     f"{cholesterol} mg/dL", "<200 mg/dL",    "✅" if cholesterol<200 else "⚠️"],
        ["Glucose",         f"{glucose} mg/dL",    "70–99 mg/dL",    "✅" if glucose<100 else "⚠️"],
    ]
    summary_df = pd.DataFrame(
        summary_data,
        columns=["Feature", "Patient Value", "Normal Range", "Status"]
    )
    st.table(summary_df)
    # summary = pd.DataFrame({
    #     'Feature': ['Age', 'BMI', 'Physical Activity',
    #                 'Blood Pressure', 'Cholesterol', 'Glucose'],
    #     'Value': [age, bmi,
    #               {0:'Low', 1:'Moderate', 2:'High'}[physical_activity],
    #               blood_pressure, cholesterol, glucose]
    # })
    # st.table(summary)

# ── Footer ──
st.divider()
st.caption("AI-Based Diabetes Prediction System ")