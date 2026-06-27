import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"

print("=" * 60)
print("BIAS MITIGATION")
print("=" * 60)

# ── Load data ──
df_full = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))

FEATURE_COLS = [
    "Gender_enc", "Married_enc", "Education_enc", "SelfEmployed_enc",
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "total_income",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "loan_to_income_ratio", "income_per_dependent", "loan_to_term_ratio",
    "Area_Rural", "Area_Semiurban", "Area_Urban"
]

from sklearn.model_selection import train_test_split
X = df_full[FEATURE_COLS]
y = df_full["Loan_Status"]
gender = df_full["Gender_raw"]
area   = df_full["Area_raw"]

X_train, X_test, y_train, y_test, g_train, g_test, a_train, a_test = train_test_split(
    X, y, gender, area, test_size=0.2, random_state=42, stratify=y
)

# Load original model
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    original_model = pickle.load(f)

y_pred_original = original_model.predict(X_test)
y_proba_original = original_model.predict_proba(X_test)[:, 1]

print("✅ Original model loaded")
print(f"   AUC: {roc_auc_score(y_test, y_proba_original):.4f}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_original):.4f}")

# ── Apply bias mitigation ──
print("\n── Applying ExponentiatedGradient with DemographicParity ──")
print("   (This may take 1-2 minutes)")

base_estimator = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

mitigator = ExponentiatedGradient(
    base_estimator,
    constraints=DemographicParity(),
    eps=0.05
)

mitigator.fit(X_train, y_train, sensitive_features=g_train)
print("✅ Mitigated model trained")

y_pred_mitigated = mitigator.predict(X_test)

# Note: mitigated model may not have predict_proba
# Use original proba for AUC comparison, pred for fairness
try:
    y_proba_mitigated = mitigator.predict_proba(X_test)[:, 1]
    auc_mitigated = roc_auc_score(y_test, y_proba_mitigated)
except:
    auc_mitigated = None
    print("   (predict_proba not available for mitigated model)")

acc_mitigated = accuracy_score(y_test, y_pred_mitigated)
print(f"   Accuracy after mitigation: {acc_mitigated:.4f}")

# ── Compare fairness metrics ──
print("\n" + "=" * 60)
print("BEFORE vs AFTER MITIGATION")
print("=" * 60)

def get_fairness_metrics(y_true, y_pred, sensitive):
    mf = MetricFrame(
        metrics={
            "approval_rate": selection_rate,
            "false_rejection_rate": false_negative_rate,
            "accuracy": accuracy_score
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    return mf

mf_before = get_fairness_metrics(y_test, y_pred_original, g_test)
mf_after  = get_fairness_metrics(y_test, y_pred_mitigated, g_test)

print("\nApproval rate by Gender — BEFORE mitigation:")
print(mf_before.by_group["approval_rate"].round(3))

print("\nApproval rate by Gender — AFTER mitigation:")
print(mf_after.by_group["approval_rate"].round(3))

dp_before = mf_before.difference(method='between_groups')['approval_rate']
dp_after  = mf_after.difference(method='between_groups')['approval_rate']

print(f"\nDemographic Parity Difference:")
print(f"  Before: {dp_before:.4f}")
print(f"  After:  {dp_after:.4f}")
print(f"  Reduction: {((dp_before - dp_after)/dp_before*100):.1f}%")

print(f"\nAccuracy:")
print(f"  Before: {accuracy_score(y_test, y_pred_original):.4f}")
print(f"  After:  {acc_mitigated:.4f}")
print(f"  Change: {((acc_mitigated - accuracy_score(y_test, y_pred_original))*100):+.1f}pp")

# ── Chart: Before vs After ──
groups = mf_before.by_group["approval_rate"].index
before_rates = mf_before.by_group["approval_rate"].values * 100
after_rates  = mf_after.by_group["approval_rate"].values * 100

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(groups))
w = 0.35
ax.bar(x - w/2, before_rates, w, label="Before mitigation",
       color="#F44336", edgecolor="white")
ax.bar(x + w/2, after_rates,  w, label="After mitigation",
       color="#4CAF50", edgecolor="white")
for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.1f}%',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel("Model Approval Rate (%)", fontsize=11)
ax.set_title(f"Bias Mitigation: Approval Rate by Gender\nDP Diff: {dp_before:.3f} → {dp_after:.3f}  |  Accuracy: {accuracy_score(y_test, y_pred_original):.3f} → {acc_mitigated:.3f}",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "18_bias_mitigation.png"), bbox_inches="tight")
plt.close()
print("\n✅ Saved: 18_bias_mitigation.png")

# ── Save summary ──
summary = f"""
BIAS MITIGATION SUMMARY
========================
Method: ExponentiatedGradient with DemographicParity constraint

BEFORE mitigation:
  Accuracy: {accuracy_score(y_test, y_pred_original):.4f}
  Demographic Parity Difference (Gender): {dp_before:.4f}

AFTER mitigation:
  Accuracy: {acc_mitigated:.4f}
  Demographic Parity Difference (Gender): {dp_after:.4f}

Fairness improvement: {((dp_before - dp_after)/dp_before*100):.1f}% reduction in DP gap
Accuracy tradeoff: {((acc_mitigated - accuracy_score(y_test, y_pred_original))*100):+.1f} percentage points
"""
with open("../reports/mitigation_summary.md", "w") as f:
    f.write(summary)
print("✅ Saved: reports/mitigation_summary.md")

print("\n" + "=" * 60)
print("✅ BIAS MITIGATION COMPLETE")
print("   Last step: run 09_llm_explanation.py")
print("=" * 60)