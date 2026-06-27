import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # prevents chart windows from opening, saves directly
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    roc_curve, ConfusionMatrixDisplay
)
import lightgbm as lgb

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("MODEL TRAINING — LightGBM")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
print(f"✅ Loaded: {df.shape}")

FEATURE_COLS = [
    "Gender_enc", "Married_enc", "Education_enc", "SelfEmployed_enc",
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "total_income",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "loan_to_income_ratio", "income_per_dependent", "loan_to_term_ratio",
    "Area_Rural", "Area_Semiurban", "Area_Urban"
]
TARGET = "Loan_Status"

X = df[FEATURE_COLS]
y = df[TARGET]
gender_col = df["Gender_raw"]
area_col = df["Area_raw"]

X_train, X_test, y_train, y_test, g_train, g_test, a_train, a_test = train_test_split(
    X, y, gender_col, area_col,
    test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)
print("✅ Model trained")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ AUC-ROC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# ── Risk Tiers ──
def assign_risk_tier(p):
    if p < 0.35: return "Low Risk"
    elif p < 0.65: return "Medium Risk"
    else: return "High Risk"

test_results = X_test.copy()
test_results["y_true"] = y_test.values
test_results["y_pred"] = y_pred
test_results["y_proba"] = y_pred_proba
test_results["Gender_raw"] = g_test.values
test_results["Area_raw"] = a_test.values
test_results["risk_tier"] = test_results["y_proba"].apply(assign_risk_tier)
test_results.to_csv(os.path.join(DATA_DIR, "test_results.csv"), index=False)
print("✅ Test results saved")
print(test_results["risk_tier"].value_counts())

# ── ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"LightGBM (AUC={auc:.4f})")
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
ax.set_title("ROC Curve — LightGBM", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_roc_curve.png"), bbox_inches="tight")
plt.close()
print("✅ Saved: 09_roc_curve.png")

# ── Confusion Matrix ──
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Rejected","Approved"],
    cmap="Blues", ax=ax
)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_confusion_matrix.png"), bbox_inches="tight")
plt.close()
print("✅ Saved: 10_confusion_matrix.png")

# ── Feature Importance ──
readable = {
    "Credit_History": "Credit History",
    "loan_to_income_ratio": "Loan to Income Ratio",
    "LoanAmount": "Loan Amount",
    "total_income": "Total Income",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Coapplicant Income",
    "income_per_dependent": "Income per Dependent",
    "Loan_Amount_Term": "Loan Term",
    "loan_to_term_ratio": "Loan to Term Ratio",
    "Dependents": "No. of Dependents",
    "Gender_enc": "Gender",
    "Married_enc": "Married",
    "Education_enc": "Education",
    "SelfEmployed_enc": "Self Employed",
    "Area_Rural": "Area: Rural",
    "Area_Semiurban": "Area: Semiurban",
    "Area_Urban": "Area: Urban"
}
importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)
importance_df["name"] = importance_df["feature"].map(readable)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(importance_df["name"], importance_df["importance"],
        color="#2196F3", edgecolor="white")
ax.set_title("Feature Importance — LightGBM\n(What drives loan approval predictions)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_feature_importance.png"), bbox_inches="tight")
plt.close()
print("✅ Saved: 11_feature_importance.png")

# ── Save model ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)
print("✅ Model saved to models/lgbm_model.pkl")

print("\n" + "=" * 60)
print(f"✅ MODELING COMPLETE — AUC: {auc:.4f}")
print("   Run 06_shap.py next")
print("=" * 60)
