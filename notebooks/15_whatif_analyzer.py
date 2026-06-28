import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"

print("=" * 60)
print("WHAT-IF ANALYZER")
print("=" * 60)

print("""
What does this do?
For each rejected applicant, this analyzer finds the MINIMUM change
needed to flip their decision from Rejected to Approved.

It tests 4 interventions:
1. Increase income
2. Reduce loan amount  
3. Add a co-applicant income
4. Build credit history

This directly answers: "What should I do to get approved?"
Used by real banks in their customer advisory systems.
""")

# ── Load model ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
test_results = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))

THRESHOLD = 0.5

def predict_proba(features_dict):
    X = pd.DataFrame([features_dict])[FEATURE_COLS]
    return model.predict_proba(X)[0][1]

def whatif_income(base_features, target_proba=THRESHOLD,
                  max_increase=5.0, steps=50):
    original_income = base_features["ApplicantIncome"]
    for multiplier in np.linspace(1.0, max_increase, steps):
        f = base_features.copy()
        new_income = original_income * multiplier
        f["ApplicantIncome"] = new_income
        f["total_income"] = new_income + f["CoapplicantIncome"]
        f["loan_to_income_ratio"] = f["LoanAmount"] / (
            f["total_income"] / 1000 + 0.001
        )
        f["income_per_dependent"] = f["total_income"] / (
            f["Dependents"] + 1
        )
        if predict_proba(f) >= target_proba:
            return new_income, multiplier, predict_proba(f)
    return None, None, None

def whatif_reduce_loan(base_features, target_proba=THRESHOLD,
                       steps=50):
    original_loan = base_features["LoanAmount"]
    for loan in np.linspace(original_loan, 10, steps):
        f = base_features.copy()
        f["LoanAmount"] = loan
        f["loan_to_income_ratio"] = loan / (
            f["total_income"] / 1000 + 0.001
        )
        f["loan_to_term_ratio"] = loan / (
            f["Loan_Amount_Term"] + 0.001
        )
        if predict_proba(f) >= target_proba:
            return loan, predict_proba(f)
    return None, None

def whatif_add_coapplicant(base_features, target_proba=THRESHOLD,
                            max_income=50000, steps=50):
    for coapplicant in np.linspace(0, max_income, steps):
        f = base_features.copy()
        f["CoapplicantIncome"] = coapplicant
        f["total_income"] = f["ApplicantIncome"] + coapplicant
        f["loan_to_income_ratio"] = f["LoanAmount"] / (
            f["total_income"] / 1000 + 0.001
        )
        f["income_per_dependent"] = f["total_income"] / (
            f["Dependents"] + 1
        )
        if predict_proba(f) >= target_proba:
            return coapplicant, predict_proba(f)
    return None, None

def whatif_credit_history(base_features):
    if base_features["Credit_History"] == 0:
        f = base_features.copy()
        f["Credit_History"] = 1.0
        new_proba = predict_proba(f)
        flipped = new_proba >= THRESHOLD
        return flipped, new_proba
    return None, None

# ── Run analysis on rejected applicants ──
rejected = test_results[test_results["y_pred"] == 0].copy()
print(f"Analyzing {min(30, len(rejected))} rejected applicants...\n")

whatif_results = []

for idx, row in rejected.head(30).iterrows():
    base = row[FEATURE_COLS].to_dict()
    original_proba = float(row["y_proba"])

    result = {
        "gender": row["Gender_raw"],
        "area": row["Area_raw"],
        "income": row["ApplicantIncome"],
        "loan": row["LoanAmount"],
        "credit_history": row["Credit_History"],
        "original_proba": original_proba
    }

    # Test 1: Increase income
    new_inc, multiplier, new_p = whatif_income(base)
    result["income_needed"] = new_inc
    result["income_multiplier"] = multiplier
    result["income_flip_proba"] = new_p

    # Test 2: Reduce loan
    new_loan, new_p2 = whatif_reduce_loan(base)
    result["loan_needed"] = new_loan
    result["loan_flip_proba"] = new_p2

    # Test 3: Add coapplicant
    coapplicant, new_p3 = whatif_add_coapplicant(base)
    result["coapplicant_needed"] = coapplicant
    result["coapplicant_flip_proba"] = new_p3

    # Test 4: Fix credit history
    flipped, new_p4 = whatif_credit_history(base)
    result["credit_flip"] = flipped
    result["credit_flip_proba"] = new_p4

    # Find easiest intervention
    options = []
    if new_inc:
        options.append(("Increase income", multiplier))
    if new_loan:
        reduction = (row["LoanAmount"] - new_loan) / row["LoanAmount"]
        options.append(("Reduce loan amount", reduction))
    if coapplicant:
        options.append(("Add co-applicant", coapplicant / row["ApplicantIncome"]))
    if flipped:
        options.append(("Build credit history", 0.1))

    if options:
        easiest = min(options, key=lambda x: x[1])
        result["easiest_intervention"] = easiest[0]
    else:
        result["easiest_intervention"] = "Multiple changes needed"

    whatif_results.append(result)

results_df = pd.DataFrame(whatif_results)

# ── Print summary ──
print("WHAT-IF ANALYSIS SUMMARY")
print("=" * 60)

print("\nEasiest intervention distribution:")
print(results_df["easiest_intervention"].value_counts())

print(f"\nIncome intervention:")
income_solvable = results_df["income_needed"].notna().sum()
print(f"  Solvable by income increase: {income_solvable}/{len(results_df)}")
if income_solvable > 0:
    avg_mult = results_df["income_multiplier"].dropna().mean()
    print(f"  Average income multiplier needed: {avg_mult:.1f}x")

print(f"\nLoan reduction intervention:")
loan_solvable = results_df["loan_needed"].notna().sum()
print(f"  Solvable by loan reduction: {loan_solvable}/{len(results_df)}")
if loan_solvable > 0:
    avg_reduction = (
        (results_df["loan"] - results_df["loan_needed"]) /
        results_df["loan"]
    ).dropna().mean() * 100
    print(f"  Average reduction needed: {avg_reduction:.1f}%")

print(f"\nCredit history intervention:")
credit_solvable = results_df["credit_flip"].sum()
print(f"  Solvable by fixing credit history: "
      f"{credit_solvable}/{len(results_df)}")

# ── Sample output ──
print("\nSample What-If Results (first 5):")
for i, row in results_df.head(5).iterrows():
    print(f"\nApplicant {i+1}: {row['gender']} | {row['area']}")
    print(f"  Current probability: {row['original_proba']:.1%} (Rejected)")
    if pd.notna(row.get("credit_flip")) and row["credit_flip"]:
        print(f"  ✅ Fix credit history → "
              f"{row['credit_flip_proba']:.1%} (Approved)")
    if pd.notna(row.get("loan_needed")):
        print(f"  ✅ Reduce loan to ₹{row['loan_needed']:.0f}K → "
              f"{row['loan_flip_proba']:.1%} (Approved)")
    if pd.notna(row.get("income_needed")):
        print(f"  ✅ Increase income to ₹{row['income_needed']:,.0f} → "
              f"{row['income_flip_proba']:.1%} (Approved)")
    if pd.notna(row.get("coapplicant_needed")):
        print(f"  ✅ Add co-applicant earning "
              f"₹{row['coapplicant_needed']:,.0f} → "
              f"{row['coapplicant_flip_proba']:.1%} (Approved)")
    print(f"  💡 Easiest fix: {row['easiest_intervention']}")

# ── CHART 1: Intervention Success Rates ──
interventions = ["Build Credit\nHistory", "Reduce Loan\nAmount",
                 "Add Co-applicant", "Increase Income"]
success_counts = [
    int(results_df["credit_flip"].sum()),
    int(results_df["loan_needed"].notna().sum()),
    int(results_df["coapplicant_needed"].notna().sum()),
    int(results_df["income_needed"].notna().sum())
]
success_pcts = [c / len(results_df) * 100 for c in success_counts]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(interventions, success_pcts,
              color=["#4CAF50","#2196F3","#FF9800","#9C27B0"],
              edgecolor="white", width=0.5)
ax.bar_label(bars,
             labels=[f"{p:.0f}%" for p in success_pcts],
             padding=5, fontsize=12, fontweight="bold")
ax.set_title("What-If Analysis: Which Intervention Flips the Decision?\n"
             "(% of rejected applicants who would be approved)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("% of Rejected Applicants")
ax.set_ylim(0, 120)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "28_whatif_interventions.png"),
            bbox_inches="tight")
plt.close()
print("\n✅ Saved: 28_whatif_interventions.png")

# ── CHART 2: Income multiplier distribution ──
income_data = results_df["income_multiplier"].dropna()
if len(income_data) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(income_data, bins=10, color="#9C27B0",
            edgecolor="white", alpha=0.8)
    ax.axvline(income_data.mean(), color="red", linestyle="--",
               linewidth=2,
               label=f"Mean: {income_data.mean():.1f}x")
    ax.set_title("Income Increase Needed to Flip Rejection\n"
                 "(How many times current income)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Income Multiplier (x times current income)")
    ax.set_ylabel("Number of Applicants")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "29_income_multiplier.png"),
                bbox_inches="tight")
    plt.close()
    print("✅ Saved: 29_income_multiplier.png")

# ── CHART 3: What-If by Gender and Area ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

gender_groups = results_df.groupby("gender").agg(
    credit_fix=("credit_flip", "sum"),
    loan_fix=("loan_needed", "count"),
    total=("gender", "count")
)
gender_groups["credit_pct"] = (
    gender_groups["credit_fix"] / gender_groups["total"] * 100
)
gender_groups["loan_pct"] = (
    gender_groups["loan_fix"] / gender_groups["total"] * 100
)

x = np.arange(len(gender_groups))
w = 0.35
axes[0].bar(x - w/2, gender_groups["credit_pct"], w,
            label="Credit history fix", color="#4CAF50",
            edgecolor="white")
axes[0].bar(x + w/2, gender_groups["loan_pct"], w,
            label="Loan reduction", color="#2196F3",
            edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(gender_groups.index)
axes[0].set_title("Interventions by Gender", fontweight="bold")
axes[0].set_ylabel("% of rejected in group")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

area_groups = results_df.groupby("area").agg(
    credit_fix=("credit_flip", "sum"),
    loan_fix=("loan_needed", "count"),
    total=("area", "count")
)
area_groups["credit_pct"] = (
    area_groups["credit_fix"] / area_groups["total"] * 100
)
area_groups["loan_pct"] = (
    area_groups["loan_fix"] / area_groups["total"] * 100
)

x2 = np.arange(len(area_groups))
axes[1].bar(x2 - w/2, area_groups["credit_pct"], w,
            label="Credit history fix", color="#4CAF50",
            edgecolor="white")
axes[1].bar(x2 + w/2, area_groups["loan_pct"], w,
            label="Loan reduction", color="#2196F3",
            edgecolor="white")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(area_groups.index)
axes[1].set_title("Interventions by Property Area", fontweight="bold")
axes[1].set_ylabel("% of rejected in group")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("What-If Analysis by Demographics",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "30_whatif_demographics.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 30_whatif_demographics.png")

results_df.to_csv(
    os.path.join(DATA_DIR, "whatif_results.csv"), index=False
)
print("✅ Saved: whatif_results.csv")

print("\n" + "=" * 60)
print("✅ WHAT-IF ANALYZER COMPLETE")
print(f"   Analyzed {len(results_df)} rejected applicants")
print(f"   Run 16_calibration.py next")
print("=" * 60)