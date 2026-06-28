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
print("LOAN AMOUNT OPTIMIZER")
print("=" * 60)

# ── Load home loan model ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
test_results = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))

print(f"✅ Model and data loaded")

# ── EMI Affordability Check ──
print("\n── EMI Affordability Check (RBI Rule: EMI ≤ 40% of income) ──")

INTEREST_RATE = 8.5  # typical home loan rate in India %
APPROVAL_THRESHOLD = 0.5

def calculate_emi(loan_amount_thousands, term_months,
                  annual_rate=INTEREST_RATE):
    P = loan_amount_thousands * 1000
    r = annual_rate / (12 * 100)
    n = term_months
    if r == 0:
        return P / n
    emi = P * r * (1 + r)**n / ((1 + r)**n - 1)
    return emi

def check_emi_affordability(monthly_income, emi):
    ratio = emi / monthly_income if monthly_income > 0 else 1
    affordable = ratio <= 0.40
    return affordable, ratio

# ── Loan Amount Optimizer ──
def find_max_loan(applicant_features, feature_cols, model,
                  min_loan=10, max_loan=700, steps=50):
    loan_amounts = np.linspace(min_loan, max_loan, steps)
    last_approved = None
    for loan in loan_amounts:
        features = applicant_features.copy()
        features["LoanAmount"] = loan
        features["loan_to_income_ratio"] = loan / (
            features["total_income"] / 1000 + 0.001
        )
        features["loan_to_term_ratio"] = loan / (
            features["Loan_Amount_Term"] + 0.001
        )
        X = pd.DataFrame([features])[feature_cols]
        proba = model.predict_proba(X)[0][1]
        if proba >= APPROVAL_THRESHOLD:
            last_approved = loan
    return last_approved

print("\n── Analyzing rejected applicants ──")

rejected = test_results[test_results["y_pred"] == 0].copy()
print(f"Total rejected applicants in test set: {len(rejected)}")

results = []
for idx, row in rejected.head(20).iterrows():
    applicant = row[FEATURE_COLS].to_dict()
    monthly_income = row["ApplicantIncome"] / 12
    requested_loan = row["LoanAmount"]
    term = row["Loan_Amount_Term"]

    max_loan = find_max_loan(applicant, FEATURE_COLS, model)

    requested_emi = calculate_emi(requested_loan, term)
    req_affordable, req_ratio = check_emi_affordability(
        monthly_income, requested_emi
    )

    if max_loan:
        max_emi = calculate_emi(max_loan, term)
        max_affordable, max_ratio = check_emi_affordability(
            monthly_income, max_emi
        )
        suggestion = f"₹{max_loan:.0f}K"
        emi_suggestion = f"₹{max_emi:.0f}/month ({max_ratio*100:.0f}% of income)"
    else:
        suggestion = "Not approvable currently"
        emi_suggestion = "N/A"
        max_loan = 0
        max_ratio = 0

    results.append({
        "gender": row["Gender_raw"],
        "area": row["Area_raw"],
        "income": row["ApplicantIncome"],
        "requested_loan": requested_loan,
        "requested_emi": requested_emi,
        "req_emi_ratio": req_ratio * 100,
        "req_affordable": req_affordable,
        "max_approvable_loan": max_loan,
        "suggested_emi": emi_suggestion,
        "max_emi_ratio": max_ratio * 100
    })

results_df = pd.DataFrame(results)
print("\nLoan Optimizer Results (first 10 rejected applicants):")
print(results_df[[
    "gender", "area", "income",
    "requested_loan", "max_approvable_loan",
    "req_emi_ratio", "max_emi_ratio"
]].head(10).to_string())

# ── EMI Affordability Stats ──
print(f"\n── EMI Affordability Summary ──")
affordable_count = results_df["req_affordable"].sum()
print(f"Requested loan affordable (EMI ≤ 40% income): "
      f"{affordable_count}/{len(results_df)} "
      f"({affordable_count/len(results_df)*100:.0f}%)")
print(f"Average requested EMI ratio: "
      f"{results_df['req_emi_ratio'].mean():.1f}% of income")
print(f"Average max approvable loan: "
      f"₹{results_df['max_approvable_loan'].mean():.0f}K")
print(f"Average requested loan: "
      f"₹{results_df['requested_loan'].mean():.0f}K")
reduction = ((results_df['requested_loan'].mean() -
              results_df['max_approvable_loan'].mean()) /
             results_df['requested_loan'].mean() * 100)
print(f"Average reduction needed: {reduction:.1f}%")

# ── CHART 1: Requested vs Max Approvable ──
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
w = 0.35
ax.bar(x - w/2, results_df["requested_loan"], w,
       label="Requested loan", color="#F44336", edgecolor="white")
ax.bar(x + w/2, results_df["max_approvable_loan"], w,
       label="Max approvable", color="#4CAF50", edgecolor="white")
ax.set_title("Requested Loan vs Maximum Approvable Loan\n"
             "(For Rejected Applicants)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Applicant Index")
ax.set_ylabel("Loan Amount (₹ thousands)")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "25_loan_optimizer.png"),
            bbox_inches="tight")
plt.close()
print("\n✅ Saved: 25_loan_optimizer.png")

# ── CHART 2: EMI Ratio Distribution ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(results_df["req_emi_ratio"].clip(upper=150), bins=15,
        color="#FF9800", edgecolor="white", alpha=0.8,
        label="Requested loan EMI ratio")
ax.axvline(40, color="red", linestyle="--",
           linewidth=2, label="RBI limit (40%)")
ax.axvline(results_df["req_emi_ratio"].mean(),
           color="blue", linestyle="--",
           linewidth=2,
           label=f"Mean ({results_df['req_emi_ratio'].mean():.0f}%)")
ax.set_title("EMI to Income Ratio Distribution\n"
             "(RBI guideline: EMI should not exceed 40% of income)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("EMI as % of Monthly Income")
ax.set_ylabel("Number of Applicants")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "26_emi_ratio.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 26_emi_ratio.png")

# ── Save results ──
results_df.to_csv(
    os.path.join(DATA_DIR, "loan_optimizer_results.csv"),
    index=False
)
print("✅ Saved: loan_optimizer_results.csv")

print("\n" + "=" * 60)
print("✅ LOAN OPTIMIZER COMPLETE")
print(f"   Analyzed {len(results_df)} rejected applicants")
print(f"   Average loan reduction needed: {reduction:.1f}%")
print("   Run 13_psi.py next")
print("=" * 60)