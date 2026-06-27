# ============================================================
# NOTEBOOK 01 — DATA INSPECTION
# Run this AFTER placing your downloaded CSVs in /data folder
# ============================================================

import pandas as pd
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────
DATA_DIR = "../data"
LOAN_FILE = os.path.join(DATA_DIR, "train.csv")          # Loan Prediction dataset
CIBIL_FILE = os.path.join(DATA_DIR, "cibil.csv")         # Indian Bank/CIBIL dataset
                                                           # rename your file to cibil.csv

# ============================================================
# SECTION 1 — LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

loan_df = pd.read_csv(LOAN_FILE)
print(f"\n✅ Loan Prediction dataset loaded: {loan_df.shape[0]} rows, {loan_df.shape[1]} columns")

try:
    cibil_df = pd.read_csv(CIBIL_FILE)
    print(f"✅ CIBIL dataset loaded: {cibil_df.shape[0]} rows, {cibil_df.shape[1]} columns")
    cibil_loaded = True
except FileNotFoundError:
    print("⚠️  CIBIL file not found yet — skipping. Add cibil.csv to /data when ready.")
    cibil_loaded = False

# ============================================================
# SECTION 2 — INSPECT LOAN PREDICTION DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOAN PREDICTION DATASET — FULL INSPECTION")
print("=" * 60)

print("\n── Column Names & Data Types ──")
print(loan_df.dtypes)

print("\n── First 5 Rows ──")
print(loan_df.head())

print("\n── Basic Stats (numeric columns) ──")
print(loan_df.describe())

print("\n── Missing Values per Column ──")
missing = loan_df.isnull().sum()
missing_pct = (missing / len(loan_df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)
print(missing_df)

# ── Target column check ──
print("\n── Target Column: Loan_Status ──")
print(loan_df["Loan_Status"].value_counts())
print(f"Approval rate: {(loan_df['Loan_Status'] == 'Y').mean() * 100:.1f}%")

# ── Key fields for your project ──
print("\n── Gender Distribution ──")
print(loan_df["Gender"].value_counts(dropna=False))

print("\n── Property Area Distribution ──")
print(loan_df["Property_Area"].value_counts(dropna=False))

# ============================================================
# SECTION 3 — CROSS-TAB (Gender × Region × Approval)
# ============================================================

print("\n" + "=" * 60)
print("CROSS-TAB: GENDER × PROPERTY AREA × LOAN STATUS")
print("=" * 60)

# Approval rate by Gender
print("\n── Approval Rate by Gender ──")
gender_approval = loan_df.groupby("Gender")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
).round(1)
print(gender_approval.to_string())

# Approval rate by Property Area
print("\n── Approval Rate by Property Area ──")
area_approval = loan_df.groupby("Property_Area")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
).round(1)
print(area_approval.to_string())

# Intersectional: Gender × Property Area
print("\n── Approval Rate by Gender × Property Area (intersectional) ──")
intersect = loan_df.groupby(["Gender", "Property_Area"])["Loan_Status"].agg(
    total="count",
    approved=lambda x: (x == "Y").sum(),
    approval_rate=lambda x: (x == "Y").mean() * 100
).round(1)
print(intersect)

# ── Flag small subgroups ──
print("\n── ⚠️  Subgroups with fewer than 50 rows (low confidence) ──")
small_groups = intersect[intersect["total"] < 50]
if len(small_groups) > 0:
    print(small_groups)
else:
    print("None — all subgroups have sufficient sample size ✅")

# ============================================================
# SECTION 4 — INSPECT CIBIL DATASET (if loaded)
# ============================================================

if cibil_loaded:
    print("\n" + "=" * 60)
    print("CIBIL DATASET — FULL INSPECTION")
    print("=" * 60)

    print("\n── Column Names & Data Types ──")
    print(cibil_df.dtypes)

    print("\n── First 5 Rows ──")
    print(cibil_df.head())

    print("\n── Missing Values per Column ──")
    missing_c = cibil_df.isnull().sum()
    missing_pct_c = (missing_c / len(cibil_df) * 100).round(2)
    missing_df_c = pd.DataFrame({
        "Missing Count": missing_c,
        "Missing %": missing_pct_c
    }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)
    print(missing_df_c if len(missing_df_c) > 0 else "No missing values ✅")

    print("\n── Target Column ──")
    # Try common target column names
    for col in ["Loan_Status", "Default", "default", "TARGET", "target", "loan_status"]:
        if col in cibil_df.columns:
            print(f"Found target column: {col}")
            print(cibil_df[col].value_counts(dropna=False))
            break

    print("\n── Look for Gender & Location Columns ──")
    for col in cibil_df.columns:
        if any(kw in col.lower() for kw in ["gender", "sex", "region", "area", "location", "urban", "rural", "city"]):
            print(f"\n  → {col}:")
            print(f"    {cibil_df[col].value_counts(dropna=False).to_dict()}")

print("\n" + "=" * 60)
print("✅ INSPECTION COMPLETE — review findings above")
print("   Save this output and note your key observations")
print("=" * 60)
