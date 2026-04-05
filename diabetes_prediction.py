# ============================================================
#   AI-BASED DIABETES PREDICTION SYSTEM
#   Dataset: Mendeley - Diabetes Risk Prediction Dataset
#   Model: Random Forest Classifier
# ============================================================

# ─────────────────────────────────────────────
# SECTION 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = {'diabetic': '#E74C3C', 'non_diabetic': '#2ECC71', 'primary': '#2C3E50', 'accent': '#3498DB'}

print("✅ All libraries imported successfully.")


# ─────────────────────────────────────────────
# SECTION 2: LOAD DATASET
# ─────────────────────────────────────────────
import os

# Auto-detect dataset path (Kaggle or local)
possible_paths = [
    '/kaggle/input/',
]

dataset_path = None
for base in possible_paths:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.endswith('.csv'):
                    dataset_path = os.path.join(root, f)
                    break

if dataset_path:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded from: {dataset_path}")
else:
    raise FileNotFoundError("❌ No CSV found. Please upload your Mendeley dataset to Kaggle.")

print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🔍 First 5 Rows:")
display(df.head())


# ─────────────────────────────────────────────
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 3: EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\n📋 Column Names & Data Types:")
print(df.dtypes)

print("\n📈 Statistical Summary:")
display(df.describe())

print("\n🔎 Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else "✅ No missing values found.")

print("\n🎯 Target Variable Distribution:")
print(df['Outcome'].value_counts())
print(f"Class Balance: {df['Outcome'].value_counts(normalize=True).round(3).to_dict()}")

# ── Plot 1: Target Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Target Variable — Diabetes Outcome Distribution', fontsize=14, fontweight='bold')

counts = df['Outcome'].value_counts()
labels = ['Non-Diabetic (0)', 'Diabetic (1)']
colors = [COLORS['non_diabetic'], COLORS['diabetic']]

axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
axes[0].set_title('Count Distribution')
axes[0].set_ylabel('Number of Patients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Percentage Split')

plt.tight_layout()
plt.savefig('target_distribution.png', bbox_inches='tight')
plt.show()
print("📊 Plot saved: target_distribution.png")

# ── Plot 2: Feature Distributions ──
KEY_FEATURES = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Cholesterol', 'PhysicalActivity']
# Map to actual column names (flexible matching)
col_map = {}
for feat in KEY_FEATURES:
    for col in df.columns:
        if feat.lower() in col.lower().replace(' ', '').replace('_', ''):
            col_map[feat] = col
            break

actual_features = list(col_map.values())
print(f"\n✅ Matched Key Features: {actual_features}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Key Health Features', fontsize=15, fontweight='bold')
axes = axes.flatten()

for i, feat in enumerate(actual_features[:6]):
    for outcome, color, label in [(0, COLORS['non_diabetic'], 'Non-Diabetic'),
                                   (1, COLORS['diabetic'], 'Diabetic')]:
        subset = df[df['Outcome'] == outcome][feat].dropna()
        axes[i].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    axes[i].set_title(feat, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', bbox_inches='tight')
plt.show()
print("📊 Plot saved: feature_distributions.png")

# ── Plot 3: Correlation Heatmap ──
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, mask=mask, ax=ax, linewidths=0.5,
            annot_kws={'size': 9}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.show()
print("📊 Plot saved: correlation_heatmap.png")

# ── Plot 4: Boxplots by Outcome ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Spread by Diabetes Outcome (Boxplot)', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, feat in enumerate(actual_features[:6]):
    data_plot = [df[df['Outcome'] == 0][feat].dropna(),
                 df[df['Outcome'] == 1][feat].dropna()]
    bp = axes[i].boxplot(data_plot, patch_artist=True,
                         labels=['Non-Diabetic', 'Diabetic'],
                         medianprops={'color': 'white', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(COLORS['non_diabetic'])
    bp['boxes'][1].set_facecolor(COLORS['diabetic'])
    axes[i].set_title(feat, fontweight='bold')
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('boxplots_by_outcome.png', bbox_inches='tight')
plt.show()
print("📊 Plot saved: boxplots_by_outcome.png")


# ─────────────────────────────────────────────
# SECTION 4: DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 4: DATA PREPROCESSING")
print("="*60)

df_clean = df.copy()

# Step 4.1: Encode Categorical Variables
cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Outcome' in cat_cols:
    cat_cols.remove('Outcome')

le = LabelEncoder()
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    print(f"  ✅ Encoded: {col}")

if not cat_cols:
    print("  ✅ No categorical columns to encode.")

# Step 4.2: Handle Missing Values
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype in ['float64', 'int64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  ✅ Filled missing in '{col}' with median: {median_val:.2f}")
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"  ✅ Filled missing in '{col}' with mode: {mode_val}")

print(f"\n  ✅ Missing values after cleaning: {df_clean.isnull().sum().sum()}")

# Step 4.3: Handle Zero Values in Clinical Columns
# (Zeros in Glucose, BloodPressure, BMI are physiologically invalid)
zero_invalid_cols = [c for c in actual_features if c in df_clean.columns
                     and c not in ['PhysicalActivity', 'Age']]
for col in zero_invalid_cols:
    zeros = (df_clean[col] == 0).sum()
    if zeros > 0:
        median_val = df_clean[df_clean[col] != 0][col].median()
        df_clean[col] = df_clean[col].replace(0, median_val)
        print(f"  ✅ Replaced {zeros} zero(s) in '{col}' with median: {median_val:.2f}")

# Step 4.4: Feature / Target Split
TARGET = 'Outcome'
X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]

print(f"\n  📐 Features (X): {X.shape}")
print(f"  🎯 Target  (y): {y.shape}")
print(f"  📋 Feature columns: {list(X.columns)}")

# Step 4.5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n  ✅ Train size : {X_train.shape[0]} samples")
print(f"  ✅ Test size  : {X_test.shape[0]} samples")

# Step 4.6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n  ✅ Features scaled using StandardScaler.")


# ─────────────────────────────────────────────
# SECTION 5: MODEL TRAINING — RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 5: MODEL TRAINING — RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
print("✅ Random Forest model trained successfully.")

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"\n📊 5-Fold Cross-Validation Accuracy:")
print(f"   Scores : {np.round(cv_scores, 4)}")
print(f"   Mean   : {cv_scores.mean():.4f}")
print(f"   Std Dev: {cv_scores.std():.4f}")


# ─────────────────────────────────────────────
# SECTION 6: MODEL EVALUATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 6: MODEL EVALUATION")
print("="*60)

y_pred       = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print(f"\n  📊 EVALUATION METRICS")
print(f"  {'─'*35}")
print(f"  ✅ Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ✅ Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  ✅ Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  ✅ F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
print(f"  ✅ ROC-AUC   : {roc_auc:.4f}  ({roc_auc*100:.2f}%)")
print(f"\n📋 Full Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# ── Plot 5: Confusion Matrix ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Evaluation — Random Forest', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            linewidths=2, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# ── Plot 6: ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color=COLORS['diabetic'], lw=2.5,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5, label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color=COLORS['diabetic'])
axes[1].set_title('ROC-AUC Curve', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', bbox_inches='tight')
plt.show()
print("📊 Plot saved: model_evaluation.png")

# ── Plot 7: Metrics Bar Chart ──
metrics = {'Accuracy': acc, 'Precision': precision,
           'Recall': recall, 'F1-Score': f1, 'ROC-AUC': roc_auc}
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(metrics.keys(), [v * 100 for v in metrics.values()],
              color=[COLORS['accent'], COLORS['primary'], COLORS['diabetic'],
                     COLORS['non_diabetic'], '#9B59B6'],
              edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Random Forest — Evaluation Metrics Summary', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_summary.png', bbox_inches='tight')
plt.show()
print("📊 Plot saved: metrics_summary.png")


# ─────────────────────────────────────────────
# SECTION 7: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 7: FEATURE IMPORTANCE")
print("="*60)

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(X.columns) * 0.45)))
colors_bar = [COLORS['diabetic'] if v >= importances.quantile(0.75)
              else COLORS['accent'] if v >= importances.median()
              else '#BDC3C7' for v in importances_sorted]
bars = ax.barh(importances_sorted.index, importances_sorted.values,
               color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, importances_sorted.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance Score', fontsize=11)
ax.set_title('Random Forest — Feature Importance', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
patches = [mpatches.Patch(color=COLORS['diabetic'], label='High Importance'),
           mpatches.Patch(color=COLORS['accent'],   label='Medium Importance'),
           mpatches.Patch(color='#BDC3C7',          label='Low Importance')]
ax.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.show()

print("\n🏆 Top 5 Most Important Features:")
print(importances.sort_values(ascending=False).head(5).round(4).to_string())


# ─────────────────────────────────────────────
# SECTION 8: PATIENT PREDICTION INTERFACE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 8: PATIENT PREDICTION INTERFACE")
print("="*60)

def predict_diabetes(patient_data: dict) -> dict:
    """
    Predicts diabetes for a single patient.

    Parameters
    ----------
    patient_data : dict
        Keys must match training feature column names.

    Returns
    -------
    dict with prediction label, probability, and risk level.
    """
    input_df = pd.DataFrame([patient_data])

    # Ensure all training columns are present (fill missing with 0)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]  # align column order
    input_scaled = scaler.transform(input_df)

    prediction  = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]

    if probability >= 0.70:
        risk_level = "🔴 HIGH RISK"
    elif probability >= 0.40:
        risk_level = "🟡 MODERATE RISK"
    else:
        risk_level = "🟢 LOW RISK"

    result = {
        'Prediction'   : 'Diabetic'     if prediction == 1 else 'Non-Diabetic',
        'Probability'  : f'{probability * 100:.2f}%',
        'Risk Level'   : risk_level,
        'Confidence'   : f'{max(probability, 1 - probability) * 100:.2f}%'
    }
    return result

# ── Example Prediction ──
# ⚠️ Replace keys below with your ACTUAL column names from the dataset
sample_patient = {col: float(X[col].median()) for col in X.columns}  # default to medians

# Override with example patient values (adjust keys to match your dataset's column names)
overrides = {
    'Age': 45,
    'BMI': 32.5,
    'Glucose': 148,
    'BloodPressure': 85,
    'Cholesterol': 210,
    'PhysicalActivity': 1,   # 1 = active, 0 = inactive (adjust per dataset encoding)
}
for key, val in overrides.items():
    for col in X.columns:
        if key.lower() in col.lower().replace(' ', '').replace('_', ''):
            sample_patient[col] = val
            break

print("\n👤 Sample Patient Input:")
for k, v in sample_patient.items():
    print(f"   {k}: {v}")

result = predict_diabetes(sample_patient)
print("\n🩺 PREDICTION RESULT:")
print(f"   {'─'*35}")
for k, v in result.items():
    print(f"   {k:15}: {v}")
print(f"   {'─'*35}")


# ─────────────────────────────────────────────
# SECTION 9: FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  SECTION 9: FINAL PROJECT SUMMARY")
print("="*60)

print(f"""
╔══════════════════════════════════════════════════════╗
║       AI-BASED DIABETES PREDICTION SYSTEM           ║
║              FINAL RESULTS SUMMARY                  ║
╠══════════════════════════════════════════════════════╣
║  Dataset       : Mendeley Diabetes Risk Dataset     ║
║  Total Samples : {df.shape[0]:<6} rows                        ║
║  Features Used : {X.shape[1]:<6} columns                     ║
║  Model         : Random Forest (200 estimators)     ║
╠══════════════════════════════════════════════════════╣
║  Accuracy      : {acc*100:>6.2f}%                           ║
║  Precision     : {precision*100:>6.2f}%                           ║
║  Recall        : {recall*100:>6.2f}%                           ║
║  F1-Score      : {f1*100:>6.2f}%                           ║
║  ROC-AUC       : {roc_auc*100:>6.2f}%                           ║
║  CV Accuracy   : {cv_scores.mean()*100:>6.2f}% ± {cv_scores.std()*100:.2f}%                ║
╠══════════════════════════════════════════════════════╣
║  Plots Saved   : 7 visualizations                   ║
║  Status        : ✅ COMPLETE                        ║
╚══════════════════════════════════════════════════════╝
""")

print("📁 Output Files Generated:")
output_files = [
    'target_distribution.png',
    'feature_distributions.png',
    'correlation_heatmap.png',
    'boxplots_by_outcome.png',
    'model_evaluation.png',
    'metrics_summary.png',
    'feature_importance.png',
]
for f in output_files:
    print(f"   📊 {f}")

print("\n✅ AI-Based Diabetes Prediction System — Complete!")
