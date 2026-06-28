import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = "../data"
FIG_DIR = "../reports/figures"

print("=" * 60)
print("POPULATION STABILITY INDEX (PSI)")
print("=" * 60)

print("""
What is PSI?
PSI measures whether the distribution of a feature has shifted 
between training and test data. Banks use it to detect model drift.

PSI Interpretation:
  PSI < 0.10  → No significant change (model stable) ✅
  PSI 0.10-0.25 → Some change (monitor) ⚠️
  PSI > 0.25  → Significant shift (model needs retraining) ❌
""")

# ── Load data ──
df_full = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
test_results = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))

from sklearn.model_selection import train_test_split
X = df_full.drop(columns=["Loan_Status", "Gender_raw", "Area_raw"])
y = df_full["Loan_Status"]

_, X_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_idx = df_full.index.difference(X_test.index)
X_train_full = df_full.loc[train_idx]
X_test_full = df_full.loc[X_test.index]

print(f"✅ Train size: {len(X_train_full)} | Test size: {len(X_test_full)}")

# ── PSI Calculation ──
def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(
        actual_pct / expected_pct
    )
    return np.sum(psi_values), psi_values, breakpoints

# ── Compute PSI for key features ──
numeric_features = [
    "ApplicantIncome", "LoanAmount", "total_income",
    "loan_to_income_ratio", "CoapplicantIncome",
    "income_per_dependent", "Loan_Amount_Term"
]

feature_labels = {
    "ApplicantIncome": "Applicant Income",
    "LoanAmount": "Loan Amount",
    "total_income": "Total Income",
    "loan_to_income_ratio": "Loan to Income Ratio",
    "CoapplicantIncome": "Coapplicant Income",
    "income_per_dependent": "Income per Dependent",
    "Loan_Amount_Term": "Loan Term"
}

psi_results = []
print("\n── PSI Results ──")
print(f"{'Feature':<25} {'PSI':>8} {'Status':>15}")
print("-" * 50)

for feat in numeric_features:
    if feat in X_train_full.columns and feat in X_test_full.columns:
        psi_val, _, _ = calculate_psi(
            X_train_full[feat].dropna(),
            X_test_full[feat].dropna()
        )
        if psi_val < 0.10:
            status = "✅ Stable"
        elif psi_val < 0.25:
            status = "⚠️  Monitor"
        else:
            status = "❌ Unstable"

        psi_results.append({
            "feature": feature_labels.get(feat, feat),
            "psi": psi_val,
            "status": status
        })
        print(f"{feature_labels.get(feat,feat):<25} "
              f"{psi_val:>8.4f} {status:>15}")

psi_df = pd.DataFrame(psi_results)

# ── CHART: PSI Bar Chart ──
colors = []
for _, row in psi_df.iterrows():
    if row["psi"] < 0.10:
        colors.append("#4CAF50")
    elif row["psi"] < 0.25:
        colors.append("#FF9800")
    else:
        colors.append("#F44336")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(psi_df["feature"], psi_df["psi"],
               color=colors, edgecolor="white")
ax.axvline(0.10, color="orange", linestyle="--",
           linewidth=1.5, label="Monitor threshold (0.10)")
ax.axvline(0.25, color="red", linestyle="--",
           linewidth=1.5, label="Unstable threshold (0.25)")
ax.set_title("Population Stability Index (PSI)\n"
             "Model Stability Check Across Key Features",
             fontsize=13, fontweight="bold")
ax.set_xlabel("PSI Value")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, val in zip(bars, psi_df["psi"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "27_psi.png"), bbox_inches="tight")
plt.close()
print("\n✅ Saved: 27_psi.png")

stable = psi_df[psi_df["psi"] < 0.10]
monitor = psi_df[(psi_df["psi"] >= 0.10) & (psi_df["psi"] < 0.25)]
unstable = psi_df[psi_df["psi"] >= 0.25]

print(f"\n── PSI Summary ──")
print(f"Stable features (PSI < 0.10):    {len(stable)}/{len(psi_df)}")
print(f"Monitor features (0.10-0.25):    {len(monitor)}/{len(psi_df)}")
print(f"Unstable features (PSI > 0.25):  {len(unstable)}/{len(psi_df)}")
print(f"Overall model stability: "
      f"{'STABLE ✅' if len(unstable)==0 else 'NEEDS REVIEW ⚠️'}")

psi_df.to_csv(
    os.path.join(DATA_DIR, "psi_results.csv"), index=False
)
print("✅ Saved: psi_results.csv")

print("\n" + "=" * 60)
print("✅ PSI COMPLETE")
print("   Run 14_streamlit_app.py to build the demo app")
print("=" * 60)