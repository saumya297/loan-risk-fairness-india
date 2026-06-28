import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import lightgbm as lgb

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"

print("=" * 60)
print("INTERSECTIONAL FAIRNESS MITIGATION")
print("=" * 60)

print("""
What makes this different from earlier mitigation?
Earlier (notebook 08) we mitigated bias on GENDER alone.

This notebook mitigates bias on GENDER × AREA combined.
This is called intersectional fairness — addressing compounded
disadvantage (e.g. rural women face BOTH gender AND location bias).

Very few ML projects do this — it's a genuine research contribution.
""")

# ── Load data ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    original_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
X = df[FEATURE_COLS]
y = df["Loan_Status"]
gender = df["Gender_raw"]
area = df["Area_raw"]

# Create intersectional group
intersect = gender + " | " + area

X_train, X_test, y_train, y_test, g_train, g_test, \
a_train, a_test, i_train, i_test = train_test_split(
    X, y, gender, area, intersect,
    test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data loaded: {X.shape}")
print(f"\nIntersectional groups in test set:")
print(i_test.value_counts())

# ── Baseline metrics ──
y_pred_orig = original_model.predict(X_test)
y_proba_orig = original_model.predict_proba(X_test)[:, 1]

mf_orig = MetricFrame(
    metrics={
        "approval_rate": selection_rate,
        "false_rejection_rate": false_negative_rate,
        "accuracy": accuracy_score
    },
    y_true=y_test,
    y_pred=y_pred_orig,
    sensitive_features=i_test
)

print(f"\n── BEFORE Intersectional Mitigation ──")
print(f"Overall accuracy: {accuracy_score(y_test,y_pred_orig):.4f}")
print(f"Overall AUC:      {roc_auc_score(y_test,y_proba_orig):.4f}")
print(f"\nApproval rate by Gender × Area:")
print(mf_orig.by_group["approval_rate"].round(3))
dp_before = mf_orig.difference(
    method="between_groups"
)["approval_rate"]
print(f"\nDemographic Parity Difference: {dp_before:.4f}")

# ── Apply intersectional mitigation ──
print(f"\n── Applying EqualizedOdds on Gender × Area ──")
print("   (This may take 2-3 minutes)")

base = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.05,
    num_leaves=31, random_state=42, verbose=-1
)

mitigator = ExponentiatedGradient(
    base, constraints=EqualizedOdds(), eps=0.05
)

mitigator.fit(X_train, y_train, sensitive_features=i_train)
print("✅ Intersectional mitigation complete")

y_pred_mit = mitigator.predict(X_test)
acc_mit = accuracy_score(y_test, y_pred_mit)

mf_mit = MetricFrame(
    metrics={
        "approval_rate": selection_rate,
        "false_rejection_rate": false_negative_rate,
        "accuracy": accuracy_score
    },
    y_true=y_test,
    y_pred=y_pred_mit,
    sensitive_features=i_test
)

dp_after = mf_mit.difference(
    method="between_groups"
)["approval_rate"]

print(f"\n── AFTER Intersectional Mitigation ──")
print(f"Overall accuracy: {acc_mit:.4f}")
print(f"\nApproval rate by Gender × Area:")
print(mf_mit.by_group["approval_rate"].round(3))
print(f"\nDemographic Parity Difference:")
print(f"  Before: {dp_before:.4f}")
print(f"  After:  {dp_after:.4f}")
reduction = (dp_before - dp_after) / dp_before * 100
print(f"  Change: {reduction:+.1f}%")

# ── Comparison table ──
comparison = pd.DataFrame({
    "before": mf_orig.by_group["approval_rate"] * 100,
    "after": mf_mit.by_group["approval_rate"] * 100
}).round(1)
comparison["change"] = (
    comparison["after"] - comparison["before"]
).round(1)
print(f"\nGroup-level comparison:")
print(comparison.to_string())

# ── CHART 1: Before vs After by intersectional group ──
groups = comparison.index.tolist()
before_vals = comparison["before"].values
after_vals = comparison["after"].values

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(groups))
w = 0.35
bars1 = ax.bar(x - w/2, before_vals, w,
               label="Before mitigation",
               color="#F44336", edgecolor="white", alpha=0.85)
bars2 = ax.bar(x + w/2, after_vals, w,
               label="After mitigation",
               color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=30,
                   ha="right", fontsize=9)
ax.set_ylabel("Model Approval Rate (%)", fontsize=11)
ax.set_title(
    f"Intersectional Fairness Mitigation: Gender × Property Area\n"
    f"DP Difference: {dp_before:.3f} → {dp_after:.3f} | "
    f"Accuracy: {accuracy_score(y_test,y_pred_orig):.3f} → "
    f"{acc_mit:.3f}",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10)
ax.set_ylim(0, 110)
ax.axhline(68.7, color="gray", linestyle="--",
           linewidth=1, alpha=0.7,
           label="Overall average (68.7%)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,
            "37_intersectional_mitigation.png"),
            bbox_inches="tight")
plt.close()
print("\n✅ Saved: 37_intersectional_mitigation.png")

# ── CHART 2: Change heatmap ──
change_pivot = pd.DataFrame(index=["Female","Male"],
                            columns=["Rural","Semiurban","Urban"])
for group in comparison.index:
    parts = group.split(" | ")
    if len(parts) == 2:
        g, a = parts
        if g in change_pivot.index and a in change_pivot.columns:
            change_pivot.loc[g, a] = comparison.loc[group, "change"]

change_pivot = change_pivot.astype(float)
print("\nChange in approval rate by group (pp):")
print(change_pivot)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(change_pivot.values,
               cmap="RdYlGn", vmin=-20, vmax=20)
ax.set_xticks(range(len(change_pivot.columns)))
ax.set_yticks(range(len(change_pivot.index)))
ax.set_xticklabels(change_pivot.columns, fontsize=12)
ax.set_yticklabels(change_pivot.index, fontsize=12)
ax.set_title(
    "Change in Approval Rate After Intersectional Mitigation\n"
    "(Green = improved, Red = reduced)",
    fontsize=12, fontweight="bold"
)
for i in range(len(change_pivot.index)):
    for j in range(len(change_pivot.columns)):
        val = change_pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.1f}pp",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Change in approval rate (pp)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,
            "38_intersectional_change_heatmap.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 38_intersectional_change_heatmap.png")

# ── Save summary ──
summary = f"""
INTERSECTIONAL FAIRNESS MITIGATION SUMMARY
==========================================
Method: ExponentiatedGradient with EqualizedOdds
Sensitive feature: Gender × Property Area (intersectional)

BEFORE mitigation:
  Accuracy: {accuracy_score(y_test,y_pred_orig):.4f}
  DP Difference (intersectional): {dp_before:.4f}

AFTER mitigation:
  Accuracy: {acc_mit:.4f}
  DP Difference (intersectional): {dp_after:.4f}
  Fairness improvement: {reduction:+.1f}%

Group-level changes:
{comparison.to_string()}
"""
with open("../reports/intersectional_mitigation.md", "w") as f:
    f.write(summary)
print("✅ Saved: reports/intersectional_mitigation.md")

print("\n" + "=" * 60)
print("✅ INTERSECTIONAL MITIGATION COMPLETE")
print(f"   DP before: {dp_before:.4f}")
print(f"   DP after:  {dp_after:.4f}")
print(f"   Change:    {reduction:+.1f}%")
print("=" * 60)