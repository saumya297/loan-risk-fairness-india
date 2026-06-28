import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../reports/figures"

print("=" * 60)
print("MODEL CALIBRATION CHECK")
print("=" * 60)

print("""
What is calibration?
A well-calibrated model means: when it says 70% approval probability,
roughly 70% of those applicants are actually approved.

Why banks care:
Banks use these probabilities to set interest rates and risk reserves.
If a model says 80% but only 50% are approved — the bank is under-reserving.

Calibration metrics:
- Brier Score: lower is better (0 = perfect, 1 = worst)
- Calibration curve: closer to diagonal = better calibrated
""")

# ── Load data ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, "features_clean.csv"))
X = df[FEATURE_COLS]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Original model probabilities
y_proba_orig = model.predict_proba(X_test)[:, 1]

# ── Calibration curve ──
fraction_pos, mean_pred = calibration_curve(
    y_test, y_proba_orig, n_bins=8, strategy="uniform"
)

# ── Brier Score ──
brier_orig = brier_score_loss(y_test, y_proba_orig)
print(f"Original model Brier Score: {brier_orig:.4f}")
print(f"(Lower is better | 0=perfect | 0.25=uninformative)")

# ── Apply calibration ──
print("\n── Applying Isotonic Calibration ──")
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(
    model, method="isotonic", cv=5
)
calibrated_model.fit(X_train, y_train)
y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

fraction_pos_cal, mean_pred_cal = calibration_curve(
    y_test, y_proba_cal, n_bins=8, strategy="uniform"
)

brier_cal = brier_score_loss(y_test, y_proba_cal)
auc_cal = roc_auc_score(y_test, y_proba_cal)
auc_orig = roc_auc_score(y_test, y_proba_orig)

print(f"Calibrated model Brier Score: {brier_cal:.4f}")
print(f"Improvement: {((brier_orig-brier_cal)/brier_orig*100):.1f}%")
print(f"\nAUC — Original:   {auc_orig:.4f}")
print(f"AUC — Calibrated: {auc_cal:.4f}")

# ── CHART 1: Calibration Curve ──
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0,1],[0,1],"k--",linewidth=1.5,
        label="Perfect calibration")
ax.plot(mean_pred, fraction_pos, "o-",
        color="#F44336", linewidth=2, markersize=8,
        label=f"Original (Brier={brier_orig:.4f})")
ax.plot(mean_pred_cal, fraction_pos_cal, "s-",
        color="#4CAF50", linewidth=2, markersize=8,
        label=f"Calibrated (Brier={brier_cal:.4f})")
ax.fill_between([0,1],[0,1],[0,1],alpha=0.1,color="gray")
ax.set_title("Calibration Curve\n"
             "(Closer to diagonal = more trustworthy probabilities)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Fraction of Positives (Actual)", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "31_calibration_curve.png"),
            bbox_inches="tight")
plt.close()
print("\n✅ Saved: 31_calibration_curve.png")

# ── CHART 2: Probability Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, proba, title, color in zip(
    axes,
    [y_proba_orig, y_proba_cal],
    ["Original Model", "Calibrated Model"],
    ["#F44336", "#4CAF50"]
):
    ax.hist(proba[y_test==1], bins=20, alpha=0.7,
            color=color, label="Approved", edgecolor="white")
    ax.hist(proba[y_test==0], bins=20, alpha=0.7,
            color="gray", label="Rejected", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--",
               linewidth=1.5, label="Decision threshold")
    ax.set_title(f"Probability Distribution\n{title}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Original vs Calibrated Probability Distributions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "32_probability_dist.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 32_probability_dist.png")

# ── CHART 3: Reliability Diagram by Risk Tier ──
test_results = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))
test_results["proba_cal"] = y_proba_cal

tier_stats = test_results.groupby("risk_tier").agg(
    mean_predicted=("y_proba", "mean"),
    actual_approval=("y_true", "mean"),
    count=("y_true", "count"),
    mean_calibrated=("proba_cal", "mean")
).reset_index()

print("\nReliability by Risk Tier:")
print(tier_stats[[
    "risk_tier","count","mean_predicted",
    "actual_approval","mean_calibrated"
]].to_string())

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(tier_stats))
w = 0.25
ax.bar(x - w, tier_stats["mean_predicted"]*100, w,
       label="Model predicted", color="#F44336", edgecolor="white")
ax.bar(x, tier_stats["actual_approval"]*100, w,
       label="Actual approval", color="#4CAF50", edgecolor="white")
ax.bar(x + w, tier_stats["mean_calibrated"]*100, w,
       label="Calibrated prediction", color="#2196F3", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(tier_stats["risk_tier"])
ax.set_ylabel("Approval Rate (%)")
ax.set_title("Reliability by Risk Tier\n"
             "Predicted vs Actual vs Calibrated",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "33_reliability_tiers.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 33_reliability_tiers.png")

# ── Save calibrated model ──
import pickle
with open(os.path.join(MODEL_DIR, "lgbm_calibrated.pkl"), "wb") as f:
    pickle.dump(calibrated_model, f)
print("✅ Saved calibrated model: models/lgbm_calibrated.pkl")

print("\n" + "=" * 60)
print("✅ CALIBRATION COMPLETE")
print(f"   Original Brier Score:   {brier_orig:.4f}")
print(f"   Calibrated Brier Score: {brier_cal:.4f}")
print(f"   Improvement: {((brier_orig-brier_cal)/brier_orig*100):.1f}%")
print("   Run 17_cibil_inflection.py next")
print("=" * 60)