# 🩺 AI-Based Diabetes Prediction System

A machine learning project that predicts diabetes risk 
using Random Forest classification on patient health data.

## 📊 Dataset
- **Source:** Mendeley — Diabetes Risk Prediction Dataset
- **Samples:** 20,000 patient records
- **Features:** Age, BMI, Physical Activity, Blood Pressure,
  Cholesterol, Glucose

## 🤖 Model Performance
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 97.15%  |
| Precision | 97%+    |
| Recall    | 97%+    |
| F1-Score  | 97%+    |
| ROC-AUC   | 99%+    |
| CV Score  | 97.15%  |

## 📁 Project Structure
ai-diabetes-prediction/
├── models/
│   ├── model.pkl        ← trained Random Forest model
│   └── scaler.pkl       ← StandardScaler
├── appp.py               ← Streamlit web app
├── requirements.txt     ← dependencies
└── README.md

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔗 Links
- 📓 Kaggle Notebook: https://www.kaggle.com/code/dushasri/ai-based-diabetes-prediction-system/
- 🌐 Live App: https://ai-diabetes-prediction-b22c6bn7dbhti5hh9wwemb.streamlit.app/

## 🛠️ Tech Stack
- Python 3.12
- Scikit-learn — Random Forest
- Streamlit — Web interface
- Pandas / NumPy — Data processing
- Matplotlib — Visualization
- Joblib — Model persistence

## ⚠️ Disclaimer
For educational purposes only.
Always consult a healthcare professional for medical diagnosis.
