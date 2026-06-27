# ============================================================
# NOTEBOOK 03 — EXPLORATORY DATA ANALYSIS (EDA)
# Run after 02_cleaning.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

DATA_DIR = "../data"
FIG_DIR = "../reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})
COLORS = {
    "Male": "#2196F3",
    "Female": "#E91E8C",
    "Urban": "#4CAF50",
    "Semiurban": "#FF9800",
    "Rural": "#9C27B0",
    "approved": "#4CAF50",
    "rejected": "#F44336"
}

print("=" * 60)
print("LOADING CLEAN DATA")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "loan_prediction_clean.csv"))
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# CHART 1 — APPROVAL RATE BY GENDER
# ============================================================

print("\n📊 Chart 1: Approval Rate by Gender")

gender_stats = df.groupby("Gender")["Loan_Status"].agg(
    total="count",
    approved="sum"
)
gender_stats["approval_rate"] = (gender_stats["approved"] / gender_stats["total"] * 100).round(1)
gender_stats["rejection_rate"] = 100 - gender_stats["approval_rate"]
print(gender_stats)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    gender_stats.index,
    gender_stats["approval_rate"],
    color=[COLORS.get(g, "#888") for g in gender_stats.index],
    width=0.5, edgecolor="white", linewidth=1.5
)
ax.bar_label(bars, labels=[f"{v}%" for v in gender_stats["approval_rate"]], padding=5, fontsize=12, fontweight="bold")
ax.set_title("Loan Approval Rate by Gender", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Gender", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.axhline(df["Loan_Status"].mean() * 100, color="gray", linestyle="--", linewidth=1, label="Overall average")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_approval_by_gender.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 01_approval_by_gender.png")

# ============================================================
# CHART 2 — APPROVAL RATE BY PROPERTY AREA
# ============================================================

print("\n📊 Chart 2: Approval Rate by Property Area")

area_stats = df.groupby("Property_Area")["Loan_Status"].agg(
    total="count",
    approved="sum"
)
area_stats["approval_rate"] = (area_stats["approved"] / area_stats["total"] * 100).round(1)
print(area_stats)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    area_stats.index,
    area_stats["approval_rate"],
    color=[COLORS.get(a, "#888") for a in area_stats.index],
    width=0.5, edgecolor="white", linewidth=1.5
)
ax.bar_label(bars, labels=[f"{v}%" for v in area_stats["approval_rate"]], padding=5, fontsize=12, fontweight="bold")
ax.set_title("Loan Approval Rate by Property Area (Rural / Semiurban / Urban)", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Property Area", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.axhline(df["Loan_Status"].mean() * 100, color="gray", linestyle="--", linewidth=1, label="Overall average")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_approval_by_area.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 02_approval_by_area.png")

# ============================================================
# CHART 3 — INTERSECTIONAL: GENDER × PROPERTY AREA (KEY CHART)
# ============================================================

print("\n📊 Chart 3: Approval Rate by Gender × Property Area (intersectional)")

intersect = df.groupby(["Gender", "Property_Area"])["Loan_Status"].agg(
    total="count",
    approved="sum"
)
intersect["approval_rate"] = (intersect["approved"] / intersect["total"] * 100).round(1)
print(intersect)

pivot = intersect["approval_rate"].unstack("Property_Area")
print("\nPivot table:")
print(pivot)

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(pivot.index))
width = 0.25
cols = list(pivot.columns)
area_colors = [COLORS.get(c, "#888") for c in cols]

for i, (col, color) in enumerate(zip(cols, area_colors)):
    bars = ax.bar(x + i * width, pivot[col], width, label=col, color=color, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in pivot[col]], padding=3, fontsize=9, fontweight="bold")

ax.set_title("Loan Approval Rate: Gender × Property Area\n(Your Key Fairness Preview Chart)", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Gender", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, fontsize=12)
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Property Area", fontsize=10, title_fontsize=10)
ax.axhline(df["Loan_Status"].mean() * 100, color="gray", linestyle="--", linewidth=1, label="Overall average", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_approval_gender_x_area.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 03_approval_gender_x_area.png — THIS IS YOUR KEY CHART")

# ============================================================
# CHART 4 — LOAN AMOUNT DISTRIBUTION BY GENDER
# ============================================================

print("\n📊 Chart 4: Loan Amount Distribution by Gender")

fig, ax = plt.subplots(figsize=(9, 5))
for gender, color in [("Male", COLORS["Male"]), ("Female", COLORS["Female"])]:
    subset = df[df["Gender"] == gender]["LoanAmount"].dropna()
    ax.hist(subset, bins=30, alpha=0.6, label=f"{gender} (n={len(subset)})", color=color, edgecolor="white")
    ax.axvline(subset.median(), color=color, linestyle="--", linewidth=1.5, label=f"{gender} median: ₹{subset.median():.0f}K")

ax.set_title("Loan Amount Distribution by Gender", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Loan Amount (₹ thousands)", fontsize=12)
ax.set_ylabel("Number of Applicants", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_loanamt_by_gender.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 04_loanamt_by_gender.png")

# ============================================================
# CHART 5 — LOAN AMOUNT DISTRIBUTION BY PROPERTY AREA
# ============================================================

print("\n📊 Chart 5: Loan Amount Distribution by Property Area")

fig, ax = plt.subplots(figsize=(9, 5))
for area, color in [("Rural", COLORS["Rural"]), ("Semiurban", COLORS["Semiurban"]), ("Urban", COLORS["Urban"])]:
    subset = df[df["Property_Area"] == area]["LoanAmount"].dropna()
    ax.hist(subset, bins=25, alpha=0.55, label=f"{area} (n={len(subset)})", color=color, edgecolor="white")
    ax.axvline(subset.median(), color=color, linestyle="--", linewidth=1.5)

ax.set_title("Loan Amount Distribution by Property Area", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Loan Amount (₹ thousands)", fontsize=12)
ax.set_ylabel("Number of Applicants", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_loanamt_by_area.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 05_loanamt_by_area.png")

# ============================================================
# CHART 6 — INCOME DISTRIBUTION BY APPROVAL STATUS
# ============================================================

print("\n📊 Chart 6: Income Distribution by Approval Status")

fig, ax = plt.subplots(figsize=(9, 5))
for status, label, color in [(1, "Approved", COLORS["approved"]), (0, "Rejected", COLORS["rejected"])]:
    subset = df[df["Loan_Status"] == status]["ApplicantIncome"].clip(upper=df["ApplicantIncome"].quantile(0.95))
    ax.hist(subset, bins=30, alpha=0.6, label=f"{label} (n={len(subset)})", color=color, edgecolor="white")

ax.set_title("Applicant Income Distribution by Loan Outcome", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Applicant Income (₹)", fontsize=12)
ax.set_ylabel("Number of Applicants", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_income_by_approval.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 06_income_by_approval.png")

# ============================================================
# CHART 7 — CORRELATION HEATMAP
# ============================================================

print("\n📊 Chart 7: Correlation Heatmap")

# Encode categoricals for correlation
df_enc = df.copy()
df_enc["Gender_enc"] = (df_enc["Gender"] == "Male").astype(int)
df_enc["Married_enc"] = (df_enc["Married"] == "Yes").astype(int)
df_enc["Education_enc"] = (df_enc["Education"] == "Graduate").astype(int)
df_enc["SelfEmployed_enc"] = (df_enc["Self_Employed"] == "Yes").astype(int)
df_enc["Area_enc"] = df_enc["Property_Area"].map({"Rural": 0, "Semiurban": 1, "Urban": 2})

numeric_cols = [
    "Loan_Status", "Gender_enc", "Married_enc", "Education_enc",
    "SelfEmployed_enc", "Dependents", "ApplicantIncome",
    "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Area_enc"
]
readable_names = [
    "Loan Approved", "Male", "Married", "Graduate",
    "Self Employed", "Dependents", "Applicant Income",
    "Coapplicant Income", "Loan Amount", "Loan Term",
    "Credit History", "Area (Rural→Urban)"
]

corr = df_enc[numeric_cols].corr()
corr.index = readable_names
corr.columns = readable_names

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1, ax=ax,
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 9}
)
ax.set_title("Feature Correlation Heatmap\n(Green = positive, Red = negative correlation with Loan Approval)", 
             fontsize=12, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_correlation_heatmap.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 07_correlation_heatmap.png")

# ============================================================
# CHART 8 — CREDIT HISTORY × APPROVAL (IMPORTANT FEATURE)
# ============================================================

print("\n📊 Chart 8: Credit History × Approval Rate")

credit_stats = df.groupby("Credit_History")["Loan_Status"].agg(
    total="count", approved="sum"
)
credit_stats["approval_rate"] = (credit_stats["approved"] / credit_stats["total"] * 100).round(1)
credit_stats.index = ["No Credit History", "Has Credit History"]
print(credit_stats)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(credit_stats.index, credit_stats["approval_rate"],
              color=["#F44336", "#4CAF50"], width=0.4, edgecolor="white", linewidth=1.5)
ax.bar_label(bars, labels=[f"{v}%" for v in credit_stats["approval_rate"]], padding=5, fontsize=13, fontweight="bold")
ax.set_title("Approval Rate by Credit History", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_credit_history_approval.png"), bbox_inches="tight")
plt.show()
print("✅ Saved: 08_credit_history_approval.png")

# ============================================================
# EDA SUMMARY — print to console and save
# ============================================================

print("\n" + "=" * 60)
print("EDA SUMMARY — KEY FINDINGS")
print("=" * 60)

overall_rate = df["Loan_Status"].mean() * 100
male_rate = df[df["Gender"] == "Male"]["Loan_Status"].mean() * 100
female_rate = df[df["Gender"] == "Female"]["Loan_Status"].mean() * 100
rural_rate = df[df["Property_Area"] == "Rural"]["Loan_Status"].mean() * 100
semi_rate = df[df["Property_Area"] == "Semiurban"]["Loan_Status"].mean() * 100
urban_rate = df[df["Property_Area"] == "Urban"]["Loan_Status"].mean() * 100

summary = f"""
EDA KEY FINDINGS
================
Dataset: {df.shape[0]} applicants, {df.shape[1]} features

1. Overall approval rate: {overall_rate:.1f}%

2. By Gender:
   - Male approval rate:   {male_rate:.1f}%
   - Female approval rate: {female_rate:.1f}%
   - Gap: {abs(male_rate - female_rate):.1f} percentage points

3. By Property Area:
   - Rural approval rate:     {rural_rate:.1f}%
   - Semiurban approval rate: {semi_rate:.1f}%
   - Urban approval rate:     {urban_rate:.1f}%

4. Strongest predictor (from correlation): Credit History
   (having a credit history dramatically increases approval chances)

5. Intersectional finding: see Chart 3 for Gender x Area breakdown

INTERPRETATION:
- The raw approval rate gap between genders is {abs(male_rate - female_rate):.1f}pp
- Rural applicants have the {'lowest' if rural_rate == min(rural_rate, semi_rate, urban_rate) else 'highest'} approval rate
- Your Week 3 fairness audit will determine: is this gap due to genuine
  risk differences, or model bias on top of real-world inequality?
"""

print(summary)

os.makedirs("../reports", exist_ok=True)
with open("../reports/eda_summary.md", "w") as f:
    f.write(summary)
print("✅ Saved: reports/eda_summary.md")

print("\n" + "=" * 60)
print(f"✅ EDA COMPLETE — {8} charts saved to reports/figures/")
print("   Next: run 04_feature_engineering.py")
print("=" * 60)