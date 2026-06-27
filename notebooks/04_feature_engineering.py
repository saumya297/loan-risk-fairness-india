import pandas as pd
import numpy as np
import os

DATA_DIR = "../data"

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "loan_prediction_clean.csv"))
print(f"✅ Loaded clean data: {df.shape}")

# ── Keep sensitive attributes separately before encoding ──
# These are needed later for fairness audit — do NOT drop them
sensitive = df[["Gender", "Property_Area"]].copy()
print("✅ Sensitive attributes saved separately")

# ── Create new features ──
df["total_income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["loan_to_income_ratio"] = df["LoanAmount"] / (df["total_income"] / 1000)
df["income_per_dependent"] = df["total_income"] / (df["Dependents"] + 1)
df["loan_to_term_ratio"] = df["LoanAmount"] / df["Loan_Amount_Term"]

print("\n✅ New features created:")
print(f"   total_income mean:         ₹{df['total_income'].mean():,.0f}")
print(f"   loan_to_income_ratio mean: {df['loan_to_income_ratio'].mean():.2f}")
print(f"   income_per_dependent mean: ₹{df['income_per_dependent'].mean():,.0f}")

# ── Encode categorical columns ──
df["Gender_enc"]        = (df["Gender"] == "Male").astype(int)
df["Married_enc"]       = (df["Married"] == "Yes").astype(int)
df["Education_enc"]     = (df["Education"] == "Graduate").astype(int)
df["SelfEmployed_enc"]  = (df["Self_Employed"] == "Yes").astype(int)

# Property Area — keep as three separate dummies
area_dummies = pd.get_dummies(df["Property_Area"], prefix="Area").astype(int)
df = pd.concat([df, area_dummies], axis=1)

print("\n✅ Categorical encoding done:")
print(f"   Gender_enc (1=Male):       {df['Gender_enc'].value_counts().to_dict()}")
print(f"   Married_enc (1=Yes):       {df['Married_enc'].value_counts().to_dict()}")
print(f"   Education_enc (1=Grad):    {df['Education_enc'].value_counts().to_dict()}")

# ── Define final feature columns for modeling ──
FEATURE_COLS = [
    "Gender_enc", "Married_enc", "Education_enc", "SelfEmployed_enc",
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "total_income",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "loan_to_income_ratio", "income_per_dependent", "loan_to_term_ratio",
    "Area_Rural", "Area_Semiurban", "Area_Urban"
]

TARGET_COL = "Loan_Status"

print(f"\n✅ Final feature set: {len(FEATURE_COLS)} features")
print(f"   Features: {FEATURE_COLS}")

# ── Save feature matrix ──
df_model = df[FEATURE_COLS + [TARGET_COL]].copy()

# Also save sensitive attributes alongside for fairness audit
df_model["Gender_raw"] = sensitive["Gender"].values
df_model["Area_raw"] = sensitive["Property_Area"].values

out = os.path.join(DATA_DIR, "features_clean.csv")
df_model.to_csv(out, index=False)
print(f"\n✅ Saved: {out}")
print(f"   Shape: {df_model.shape}")

# ── Quick preview ──
print("\nFirst 3 rows of feature matrix:")
print(df_model[FEATURE_COLS[:8]].head(3))

print("\n" + "=" * 60)
print("✅ FEATURE ENGINEERING COMPLETE — run 05_model.py next")
print("=" * 60)