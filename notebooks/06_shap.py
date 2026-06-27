import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shap

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"

print("=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

# ── Load model and data ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))
X_test = df[FEATURE_COLS]

print(f"✅ Model and data loaded")
print(f"   Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")

# ── Compute SHAP values ──
print("\n── Computing SHAP values (may take 30 seconds) ──")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification LightGBM, shap_values may be a list
# We want the positive class (approved = 1)
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print(f"✅ SHAP values computed: {sv.shape}")

# ── Readable feature names ──
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
    "Gender_enc": "Gender (Male=1)",
    "Married_enc": "Married",
    "Education_enc": "Education (Graduate=1)",
    "SelfEmployed_enc": "Self Employed",
    "Area_Rural": "Area: Rural",
    "Area_Semiurban": "Area: Semiurban",
    "Area_Urban": "Area: Urban"
}
readable_names = [readable.get(c, c) for c in FEATURE_COLS]

# ── CHART 1: SHAP Summary Plot (global) ──
print("\n── Chart 1: SHAP Summary Plot ──")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    sv, X_test,
    feature_names=readable_names,
    show=False,
    plot_size=None
)
plt.title("SHAP Summary Plot\n(How each feature pushes predictions toward Approved or Rejected)",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_shap_summary.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 12_shap_summary.png")

# ── CHART 2: SHAP Bar Plot (mean absolute) ──
print("\n── Chart 2: SHAP Bar Plot ──")
mean_shap = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": readable_names,
    "mean_shap": mean_shap
}).sort_values("mean_shap", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(shap_df["feature"], shap_df["mean_shap"],
        color="#FF6B35", edgecolor="white")
ax.set_title("Mean Absolute SHAP Values\n(Average impact of each feature on model output)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_shap_bar.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 13_shap_bar.png")

# ── CHART 3: Waterfall plots for 3 individual applicants ──
print("\n── Chart 3: Individual applicant explanations ──")

# Find one of each risk tier
results = df.copy()
low_idx    = results[results["risk_tier"] == "Low Risk"].index[0]
medium_idx = results[results["risk_tier"] == "Medium Risk"].index[0]
high_idx   = results[results["risk_tier"] == "High Risk"].index[0]

for idx, label, color in [
    (low_idx, "Low Risk — Likely Approved", "#4CAF50"),
    (medium_idx, "Medium Risk — Borderline", "#FF9800"),
    (high_idx, "High Risk — Likely Rejected", "#F44336")
]:
    row_pos = results.index.get_loc(idx)
    shap_explanation = shap.Explanation(
        values=sv[row_pos],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
        data=X_test.iloc[row_pos].values,
        feature_names=readable_names
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, show=False, max_display=10)
    plt.title(f"Individual Explanation: {label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"14_waterfall_{label.split()[0].lower()}_risk.png"
    plt.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {fname}")

# ── Generate reason codes for sample applicants ──
print("\n── Reason Codes for Sample Applicants ──")

def get_reason_codes(row_pos, top_n=3):
    shap_row = sv[row_pos]
    feature_shap = list(zip(readable_names, shap_row))
    sorted_by_abs = sorted(feature_shap, key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_by_abs[:top_n]
    reasons = []
    for feat, val in top_features:
        direction = "increased" if val > 0 else "decreased"
        reasons.append(f"  → {feat}: {direction} approval likelihood (SHAP={val:.3f})")
    return reasons

print("\nSample applicant reason codes:")
for i, (idx, label) in enumerate([(low_idx, "LOW RISK"), (medium_idx, "MEDIUM RISK"), (high_idx, "HIGH RISK")]):
    row_pos = results.index.get_loc(idx)
    actual = "Approved" if df.loc[idx, "y_true"] == 1 else "Rejected"
    predicted = "Approved" if df.loc[idx, "y_pred"] == 1 else "Rejected"
    prob = df.loc[idx, "y_proba"]
    gender = df.loc[idx, "Gender_raw"]
    area = df.loc[idx, "Area_raw"]
    print(f"\nApplicant {i+1} [{label}]")
    print(f"  Gender: {gender} | Area: {area}")
    print(f"  Actual: {actual} | Predicted: {predicted} | Probability: {prob:.1%}")
    print(f"  Top reasons:")
    for r in get_reason_codes(row_pos):
        print(r)

print("\n" + "=" * 60)
print("✅ SHAP COMPLETE — 5 charts saved")
print("   Next: run 07_fairness_audit.py")
print("=" * 60)