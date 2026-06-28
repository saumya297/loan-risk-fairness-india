import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    roc_curve, ConfusionMatrixDisplay, accuracy_score
)
import lightgbm as lgb

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("CIBIL PERSONAL LOAN — FULL MODELING")
print("=" * 60)

# ── Load CIBIL clean data ──
cibil = pd.read_csv(os.path.join(DATA_DIR, "cibil_clean.csv"))
print(f"✅ Loaded CIBIL dataset: {cibil.shape}")
print(f"   Columns: {list(cibil.columns)}")

# ── Check target ──
print(f"\nLoan status distribution:")
print(cibil["loan_status"].value_counts())
print(f"Approval rate: {(cibil['loan_status_encoded']==1).mean()*100:.1f}%")

# ── Feature Engineering ──
print("\n── Feature Engineering ──")

# Derived features
cibil["total_assets"] = (
    cibil["residential_assets_value"] +
    cibil["commercial_assets_value"] +
    cibil["luxury_assets_value"] +
    cibil["bank_asset_value"]
)
cibil["loan_to_asset_ratio"] = cibil["loan_amount"] / (cibil["total_assets"] + 1)
cibil["loan_to_income_ratio"] = cibil["loan_amount"] / (cibil["income_annum"] + 1)
cibil["asset_to_income_ratio"] = cibil["total_assets"] / (cibil["income_annum"] + 1)
cibil["income_per_dependent"] = cibil["income_annum"] / (cibil["no_of_dependents"] + 1)
cibil["emi_estimate"] = cibil["loan_amount"] / (cibil["loan_term"] + 1)
cibil["emi_to_income_ratio"] = cibil["emi_estimate"] / (cibil["income_annum"] / 12 + 1)

print("✅ Created 6 derived features")
print(f"   loan_to_income_ratio mean: {cibil['loan_to_income_ratio'].mean():.4f}")
print(f"   loan_to_asset_ratio mean:  {cibil['loan_to_asset_ratio'].mean():.4f}")

# ── Encode categoricals ──
cibil["education_enc"] = (cibil["education"] == "Graduate").astype(int)
cibil["self_employed_enc"] = (cibil["self_employed"] == "Yes").astype(int)

print("✅ Encoded: education, self_employed")

# ── Define features ──
FEATURE_COLS = [
    "no_of_dependents", "education_enc", "self_employed_enc",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
    "total_assets", "loan_to_asset_ratio", "loan_to_income_ratio",
    "asset_to_income_ratio", "income_per_dependent",
    "emi_estimate", "emi_to_income_ratio"
]

TARGET = "loan_status_encoded"

X = cibil[FEATURE_COLS]
y = cibil[TARGET]

print(f"\n✅ Feature matrix: {X.shape}")
print(f"   {len(FEATURE_COLS)} features")

# ── Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Split: Train {X_train.shape[0]} | Test {X_test.shape[0]}")
print(f"   Train approval rate: {y_train.mean()*100:.1f}%")
print(f"   Test approval rate:  {y_test.mean()*100:.1f}%")

# ── Train LightGBM ──
print("\n── Training LightGBM on CIBIL data ──")
model_cibil = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)
model_cibil.fit(X_train, y_train)
print("✅ Model trained")

y_pred = model_cibil.predict(X_test)
y_proba = model_cibil.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ CIBIL Model Performance:")
print(f"   AUC-ROC:  {auc:.4f}")
print(f"   Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Rejected", "Approved"]))

# ── Save model ──
with open(os.path.join(MODEL_DIR, "lgbm_cibil.pkl"), "wb") as f:
    pickle.dump(model_cibil, f)
with open(os.path.join(MODEL_DIR, "cibil_feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)
print("✅ CIBIL model saved")

# ── CHART 1: ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#E91E8C", linewidth=2,
        label=f"CIBIL Model (AUC={auc:.4f})")
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
ax.set_title("ROC Curve — CIBIL Personal Loan Model",
             fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "19_cibil_roc.png"), bbox_inches="tight")
plt.close()
print("✅ Saved: 19_cibil_roc.png")

# ── CHART 2: Confusion Matrix ──
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Rejected", "Approved"],
    cmap="Purples", ax=ax
)
ax.set_title("Confusion Matrix — CIBIL Model",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "20_cibil_confusion.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 20_cibil_confusion.png")

# ── CHART 3: Feature Importance ──
readable = {
    "no_of_dependents": "No. of Dependents",
    "education_enc": "Education (Graduate)",
    "self_employed_enc": "Self Employed",
    "income_annum": "Annual Income",
    "loan_amount": "Loan Amount",
    "loan_term": "Loan Term",
    "cibil_score": "CIBIL Score",
    "residential_assets_value": "Residential Assets",
    "commercial_assets_value": "Commercial Assets",
    "luxury_assets_value": "Luxury Assets",
    "bank_asset_value": "Bank Assets",
    "total_assets": "Total Assets",
    "loan_to_asset_ratio": "Loan to Asset Ratio",
    "loan_to_income_ratio": "Loan to Income Ratio",
    "asset_to_income_ratio": "Asset to Income Ratio",
    "income_per_dependent": "Income per Dependent",
    "emi_estimate": "EMI Estimate",
    "emi_to_income_ratio": "EMI to Income Ratio"
}

imp_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model_cibil.feature_importances_
}).sort_values("importance", ascending=True)
imp_df["name"] = imp_df["feature"].map(readable)

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(imp_df["name"], imp_df["importance"],
        color="#E91E8C", edgecolor="white")
ax.set_title("Feature Importance — CIBIL Personal Loan Model",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "21_cibil_importance.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 21_cibil_importance.png")

# ── CHART 4: CIBIL Score vs Approval Rate ──
print("\n── CIBIL Score Analysis ──")

cibil_test = cibil.iloc[X_test.index].copy()
cibil_test["y_pred"] = y_pred
cibil_test["y_proba"] = y_proba
cibil_test["y_true"] = y_test.values

# Bin CIBIL scores into ranges
bins = [300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 900]
labels = ["300-400","400-500","500-550","550-600",
          "600-650","650-700","700-750","750-800","800-850","850-900"]
cibil_test["cibil_band"] = pd.cut(
    cibil_test["cibil_score"], bins=bins, labels=labels
)

cibil_band_stats = cibil_test.groupby("cibil_band", observed=True).agg(
    count=("y_true", "count"),
    actual_approval=("y_true", "mean"),
    model_approval=("y_pred", "mean")
).reset_index()
cibil_band_stats["actual_pct"] = cibil_band_stats["actual_approval"] * 100
cibil_band_stats["model_pct"] = cibil_band_stats["model_approval"] * 100

print("\nCIBIL Score Band Analysis:")
print(cibil_band_stats[["cibil_band","count","actual_pct","model_pct"]].to_string())

# Find inflection point
high_approval = cibil_band_stats[cibil_band_stats["actual_pct"] >= 80]
if len(high_approval) > 0:
    inflection = high_approval.iloc[0]["cibil_band"]
    print(f"\n✅ CIBIL inflection point: {inflection}")
    print(f"   Above this band, approval rate exceeds 80%")

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(cibil_band_stats))
w = 0.35
bars1 = ax.bar(x - w/2, cibil_band_stats["actual_pct"], w,
               label="Actual approval rate",
               color="#4CAF50", edgecolor="white")
bars2 = ax.bar(x + w/2, cibil_band_stats["model_pct"], w,
               label="Model predicted rate",
               color="#E91E8C", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(cibil_band_stats["cibil_band"],
                   rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Approval Rate (%)", fontsize=11)
ax.set_title("Approval Rate by CIBIL Score Band\n(Actual vs Model Predicted)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(80, color="red", linestyle="--",
           linewidth=1.5, label="80% threshold")
ax.set_ylim(0, 110)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "22_cibil_score_bands.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 22_cibil_score_bands.png")

# ── CHART 5: SHAP for CIBIL model ──
print("\n── Computing SHAP for CIBIL model ──")
explainer = shap.TreeExplainer(model_cibil)
shap_values = explainer.shap_values(X_test)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

readable_names = [readable.get(c, c) for c in FEATURE_COLS]

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_test,
                  feature_names=readable_names,
                  show=False, plot_size=None)
plt.title("SHAP Summary — CIBIL Personal Loan Model",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "23_cibil_shap.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 23_cibil_shap.png")

# ── COMPARISON: Home Loan vs Personal Loan ──
print("\n" + "=" * 60)
print("MODEL COMPARISON: HOME LOAN vs PERSONAL LOAN")
print("=" * 60)

# Load home loan model results
home_test = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))
home_auc = roc_auc_score(home_test["y_true"], home_test["y_proba"])
home_acc = accuracy_score(home_test["y_true"], home_test["y_pred"])

print(f"\nHome Loan Model (Dream Housing Finance):")
print(f"  Dataset size: 614 rows")
print(f"  AUC-ROC:     {home_auc:.4f}")
print(f"  Accuracy:    {home_acc:.4f}")
print(f"  Top feature: Credit History")

print(f"\nPersonal Loan Model (CIBIL):")
print(f"  Dataset size: 4,269 rows")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Top feature: CIBIL Score")

# Comparison chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AUC comparison
models = ["Home Loan\n(Dream Housing)", "Personal Loan\n(CIBIL)"]
aucs = [home_auc, auc]
accs = [home_acc, acc]
colors = ["#2196F3", "#E91E8C"]

axes[0].bar(models, aucs, color=colors, edgecolor="white", width=0.4)
for i, v in enumerate(aucs):
    axes[0].text(i, v + 0.005, f"{v:.4f}",
                 ha="center", fontsize=12, fontweight="bold")
axes[0].set_title("AUC-ROC Comparison", fontsize=12, fontweight="bold")
axes[0].set_ylabel("AUC-ROC Score")
axes[0].set_ylim(0, 1.1)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Accuracy comparison
axes[1].bar(models, [a*100 for a in accs],
            color=colors, edgecolor="white", width=0.4)
for i, v in enumerate(accs):
    axes[1].text(i, v*100 + 0.5, f"{v*100:.1f}%",
                 ha="center", fontsize=12, fontweight="bold")
axes[1].set_title("Accuracy Comparison", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 110)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Home Loan vs Personal Loan Model Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "24_model_comparison.png"),
            bbox_inches="tight")
plt.close()
print("\n✅ Saved: 24_model_comparison.png")

# ── Save test results ──
cibil_test.to_csv(os.path.join(DATA_DIR, "cibil_test_results.csv"),
                  index=False)
print("✅ Saved: cibil_test_results.csv")

print("\n" + "=" * 60)
print("✅ CIBIL MODELING COMPLETE")
print(f"   Personal Loan AUC: {auc:.4f}")
print(f"   Home Loan AUC:     {home_auc:.4f}")
print(f"   Charts saved:      6 new charts (19-24)")
print("   Run 12_loan_optimizer.py next")
print("=" * 60)