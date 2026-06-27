import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from sklearn.metrics import accuracy_score

DATA_DIR = "../data"
FIG_DIR = "../reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("FAIRNESS AUDIT")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))
print(f"✅ Loaded test results: {df.shape}")
print(f"   Columns: {list(df.columns)}")

y_true = df["y_true"]
y_pred = df["y_pred"]
gender  = df["Gender_raw"]
area    = df["Area_raw"]

# ── Create intersectional group ──
df["gender_area"] = df["Gender_raw"] + " | " + df["Area_raw"]
gender_area = df["gender_area"]

print("\n── Group sizes ──")
print("By Gender:")
print(gender.value_counts())
print("\nBy Area:")
print(area.value_counts())
print("\nIntersectional (Gender × Area):")
print(gender_area.value_counts())

# ============================================================
# SECTION 1 — FAIRNESS BY GENDER
# ============================================================
print("\n" + "=" * 60)
print("FAIRNESS BY GENDER")
print("=" * 60)

mf_gender = MetricFrame(
    metrics={
        "approval_rate": selection_rate,
        "false_rejection_rate": false_negative_rate,
        "accuracy": accuracy_score
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=gender
)

print("\nOverall metrics:")
print(mf_gender.overall)
print("\nBy Gender:")
print(mf_gender.by_group)
print(f"\nDemographic Parity Difference (Gender): {mf_gender.difference(method='between_groups')['approval_rate']:.4f}")
print(f"False Rejection Rate Difference (Gender): {mf_gender.difference(method='between_groups')['false_rejection_rate']:.4f}")

# ============================================================
# SECTION 2 — FAIRNESS BY AREA
# ============================================================
print("\n" + "=" * 60)
print("FAIRNESS BY PROPERTY AREA")
print("=" * 60)

mf_area = MetricFrame(
    metrics={
        "approval_rate": selection_rate,
        "false_rejection_rate": false_negative_rate,
        "accuracy": accuracy_score
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=area
)

print("\nBy Area:")
print(mf_area.by_group)
print(f"\nDemographic Parity Difference (Area): {mf_area.difference(method='between_groups')['approval_rate']:.4f}")
print(f"False Rejection Rate Difference (Area): {mf_area.difference(method='between_groups')['false_rejection_rate']:.4f}")

# ============================================================
# SECTION 3 — INTERSECTIONAL GENDER × AREA
# ============================================================
print("\n" + "=" * 60)
print("INTERSECTIONAL FAIRNESS: GENDER × AREA")
print("=" * 60)

mf_intersect = MetricFrame(
    metrics={
        "approval_rate": selection_rate,
        "false_rejection_rate": false_negative_rate,
        "accuracy": accuracy_score
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=gender_area
)

print("\nBy Gender × Area:")
print(mf_intersect.by_group.to_string())

# ============================================================
# SECTION 4 — REAL DATA vs MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("REAL DATA vs MODEL — IS THE MODEL ADDING BIAS?")
print("=" * 60)

comparison = pd.DataFrame({
    "actual_approval_rate": df.groupby("Gender_raw")["y_true"].mean() * 100,
    "model_approval_rate": df.groupby("Gender_raw")["y_pred"].mean() * 100,
}).round(1)
comparison["model_vs_actual_gap"] = (
    comparison["model_approval_rate"] - comparison["actual_approval_rate"]
).round(1)
print("\nGender comparison:")
print(comparison)

comparison_area = pd.DataFrame({
    "actual_approval_rate": df.groupby("Area_raw")["y_true"].mean() * 100,
    "model_approval_rate": df.groupby("Area_raw")["y_pred"].mean() * 100,
}).round(1)
comparison_area["model_vs_actual_gap"] = (
    comparison_area["model_approval_rate"] - comparison_area["actual_approval_rate"]
).round(1)
print("\nArea comparison:")
print(comparison_area)

print("\n── INTERPRETATION ──")
print("If model_vs_actual_gap is close to 0: model reflects reality accurately")
print("If gap is large positive: model is MORE generous than reality for that group")
print("If gap is large negative: model is HARSHER than reality for that group")

# ============================================================
# CHARTS
# ============================================================

# Chart 1: Approval rate by gender — actual vs model
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comparison))
w = 0.35
ax.bar(x - w/2, comparison["actual_approval_rate"], w,
       label="Actual (real data)", color="#4CAF50", edgecolor="white")
ax.bar(x + w/2, comparison["model_approval_rate"], w,
       label="Model prediction", color="#2196F3", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(comparison.index, fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=11)
ax.set_title("Actual vs Model Approval Rate by Gender\n(Are they close? If not — the model adds bias)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.1f}%',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "15_fairness_gender.png"), bbox_inches="tight")
plt.close()
print("\n✅ Saved: 15_fairness_gender.png")

# Chart 2: Approval rate by area — actual vs model
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(comparison_area))
ax.bar(x - w/2, comparison_area["actual_approval_rate"], w,
       label="Actual (real data)", color="#FF9800", edgecolor="white")
ax.bar(x + w/2, comparison_area["model_approval_rate"], w,
       label="Model prediction", color="#9C27B0", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(comparison_area.index, fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=11)
ax.set_title("Actual vs Model Approval Rate by Property Area\n(Are they close? If not — the model adds bias)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.1f}%',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "16_fairness_area.png"), bbox_inches="tight")
plt.close()
print("✅ Saved: 16_fairness_area.png")

# Chart 3: Intersectional heatmap
pivot_actual = df.pivot_table(
    values="y_true", index="Gender_raw",
    columns="Area_raw", aggfunc="mean"
) * 100

pivot_model = df.pivot_table(
    values="y_pred", index="Gender_raw",
    columns="Area_raw", aggfunc="mean"
) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, pivot, title, cmap in zip(
    axes,
    [pivot_actual, pivot_model],
    ["Actual Approval Rate (%)", "Model Predicted Approval Rate (%)"],
    ["Greens", "Blues"]
):
    im = ax.imshow(pivot.values, cmap=cmap, vmin=40, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.1f}%",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="black")
    plt.colorbar(im, ax=ax)

plt.suptitle("Intersectional Fairness: Gender × Property Area\nActual Data vs Model Predictions",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "17_intersectional_heatmap.png"), bbox_inches="tight")
plt.close()
print("✅ Saved: 17_intersectional_heatmap.png")

# ── Save fairness summary ──
summary = f"""
FAIRNESS AUDIT SUMMARY
======================
Gender Demographic Parity Difference: {mf_gender.difference(method='between_groups')['approval_rate']:.4f}
Gender False Rejection Rate Difference: {mf_gender.difference(method='between_groups')['false_rejection_rate']:.4f}
Area Demographic Parity Difference: {mf_area.difference(method='between_groups')['approval_rate']:.4f}
Area False Rejection Rate Difference: {mf_area.difference(method='between_groups')['false_rejection_rate']:.4f}

Gender comparison (actual vs model):
{comparison.to_string()}

Area comparison (actual vs model):
{comparison_area.to_string()}

Intersectional (Gender x Area) approval rates:
{mf_intersect.by_group['approval_rate'].to_string()}
"""
os.makedirs("../reports", exist_ok=True)
with open("../reports/fairness_summary.md", "w") as f:
    f.write(summary)
print("✅ Saved: reports/fairness_summary.md")

print("\n" + "=" * 60)
print("✅ FAIRNESS AUDIT COMPLETE")
print("   Run 08_bias_mitigation.py next")
print("=" * 60)