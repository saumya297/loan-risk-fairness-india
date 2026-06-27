# ============================================================
# NOTEBOOK 02 — DATA CLEANING
# Run after 01_data_inspection.py
# ============================================================

import pandas as pd
import numpy as np
import os

DATA_DIR = "../data"

print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"✅ Loaded: {df.shape}")

# ── Drop Loan_ID ──
df = df.drop(columns=["Loan_ID"])
print("✅ Dropped Loan_ID")

# ── Fix Gender ──
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Gender"] = df["Gender"].str.strip().str.title()
print(f"Gender: {df['Gender'].value_counts().to_dict()}")

# ── Fix Married ──
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])

# ── Fix Dependents ──
df["Dependents"] = df["Dependents"].replace("3+", "3")
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int)

# ── Fix Education ──
df["Education"] = df["Education"].str.strip()

# ── Fix Self_Employed ──
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

# ── Fix LoanAmount ──
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
print(f"LoanAmount missing after fix: {df['LoanAmount'].isnull().sum()}")

# ── Fix Loan_Amount_Term ──
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

# ── Fix Credit_History ──
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
print(f"Credit_History missing after fix: {df['Credit_History'].isnull().sum()}")

# ── Fix Property_Area ──
df["Property_Area"] = df["Property_Area"].str.strip().str.title()
print(f"Property_Area: {df['Property_Area'].value_counts().to_dict()}")

# ── Encode Target ──
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
print(f"Loan_Status: {df['Loan_Status'].value_counts().to_dict()}")
print(f"Approval rate: {df['Loan_Status'].mean()*100:.1f}%")

# ── Verify clean ──
remaining = df.isnull().sum().sum()
print(f"\nRemaining missing values: {remaining}")
if remaining == 0:
    print("✅ Fully clean — no missing values")

# ── Save ──
out = os.path.join(DATA_DIR, "loan_prediction_clean.csv")
df.to_csv(out, index=False)
print(f"✅ Saved: {out}")
print(f"   Final shape: {df.shape}")

# ============================================================
# CLEAN CIBIL DATASET
# ============================================================

print("\n" + "=" * 60)
print("CLEANING CIBIL DATASET")
print("=" * 60)

cibil = pd.read_csv(os.path.join(DATA_DIR, "cibil.csv"))

# Strip spaces from column names (common issue with this dataset)
cibil.columns = cibil.columns.str.strip()
print(f"Columns: {list(cibil.columns)}")
print(f"Shape: {cibil.shape}")

# Fix missing values
for col in cibil.columns:
    if cibil[col].isnull().sum() > 0:
        if cibil[col].dtype == "object":
            cibil[col] = cibil[col].fillna(cibil[col].mode()[0])
        else:
            cibil[col] = cibil[col].fillna(cibil[col].median())
        print(f"  Fixed: {col}")

# Encode loan_status
if "loan_status" in cibil.columns:
    cibil["loan_status"] = cibil["loan_status"].str.strip()
    cibil["loan_status_encoded"] = cibil["loan_status"].map(
        {"Approved": 1, "Rejected": 0}
    )
    print(f"\nloan_status: {cibil['loan_status'].value_counts().to_dict()}")
    approval_rate = cibil["loan_status_encoded"].mean() * 100
    print(f"CIBIL approval rate: {approval_rate:.1f}%")

# Strip spaces from string columns
for col in cibil.select_dtypes(include="object").columns:
    cibil[col] = cibil[col].str.strip()

remaining_c = cibil.isnull().sum().sum()
print(f"\nRemaining missing values: {remaining_c}")
if remaining_c == 0:
    print("✅ CIBIL fully clean")

out_c = os.path.join(DATA_DIR, "cibil_clean.csv")
cibil.to_csv(out_c, index=False)
print(f"✅ Saved: {out_c}")
print(f"   Final shape: {cibil.shape}")

print("\n" + "=" * 60)
print("✅ CLEANING COMPLETE — run 03_eda.py next")
print("=" * 60)