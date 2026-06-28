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
print("CIBIL SCORE INFLECTION POINT ANALYSIS")
print("=" * 60)

print("""
What is the inflection point?
The CIBIL score value above which loan approval probability 
jumps sharply — the "magic number" applicants need to cross.

RBI recommends minimum CIBIL score of 750 for home loans.
We verify: does our data support this? Where does the actual
jump happen?
""")

# ── Load data ──
cibil = pd.read_csv(os.path.join(DATA_DIR, "cibil_clean.csv"))
cibil_test = pd.read_csv(
    os.path.join(DATA_DIR, "cibil_test_results.csv")
)

with open(os.path.join(MODEL_DIR, "lgbm_cibil.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "cibil_feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

print(f"✅ CIBIL dataset: {cibil.shape}")
print(f"   CIBIL score range: "
      f"{cibil['cibil_score'].min()} — {cibil['cibil_score'].max()}")

# ── Score bands analysis ──
bins = [300,400,450,500,550,600,625,650,675,
        700,725,750,775,800,850,900]
labels = ["300-400","400-450","450-500","500-550",
          "550-600","600-625","625-650","650-675",
          "675-700","700-725","725-750","750-775",
          "775-800","800-850","850-900"]

cibil["score_band"] = pd.cut(
    cibil["cibil_score"], bins=bins, labels=labels
)

band_stats = cibil.groupby("score_band", observed=True).agg(
    count=("loan_status_encoded","count"),
    approval_rate=("loan_status_encoded","mean")
).reset_index()
band_stats["approval_pct"] = band_stats["approval_rate"] * 100

print("\nCIBIL Score Band Analysis:")
print(band_stats[["score_band","count","approval_pct"]].to_string())

# ── Find inflection point ──
threshold_50 = band_stats[
    band_stats["approval_pct"] >= 50
]["score_band"].iloc[0] if len(
    band_stats[band_stats["approval_pct"] >= 50]
) > 0 else "N/A"

threshold_80 = band_stats[
    band_stats["approval_pct"] >= 80
]["score_band"].iloc[0] if len(
    band_stats[band_stats["approval_pct"] >= 80]
) > 0 else "N/A"

threshold_90 = band_stats[
    band_stats["approval_pct"] >= 90
]["score_band"].iloc[0] if len(
    band_stats[band_stats["approval_pct"] >= 90]
) > 0 else "N/A"

print(f"\n── Inflection Points ──")
print(f"50% approval threshold: CIBIL band {threshold_50}")
print(f"80% approval threshold: CIBIL band {threshold_80}")
print(f"90% approval threshold: CIBIL band {threshold_90}")
print(f"\nRBI recommended minimum: 750")
print(f"Our data 80% threshold: {threshold_80}")

# ── Smooth probability curve ──
scores = np.arange(300, 901, 5)

# Use median values from cibil_clean which has encoded columns
median_features = {}
for col in FEATURE_COLS:
    if col != "cibil_score":
        if col in cibil.columns:
            median_features[col] = pd.to_numeric(
                cibil[col], errors="coerce"
            ).median()
        else:
            median_features[col] = 0

smooth_probas = []
for score in scores:
    f = median_features.copy()
    f["cibil_score"] = score
    X = pd.DataFrame([f])[FEATURE_COLS]
    p = model.predict_proba(X)[0][1]
    smooth_probas.append(p)
# ── CHART 1: CIBIL Score vs Approval (bar) ──
colors_bar = []
for pct in band_stats["approval_pct"]:
    if pct < 40:
        colors_bar.append("#F44336")
    elif pct < 70:
        colors_bar.append("#FF9800")
    else:
        colors_bar.append("#4CAF50")

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(band_stats)),
              band_stats["approval_pct"],
              color=colors_bar, edgecolor="white")
ax.set_xticks(range(len(band_stats)))
ax.set_xticklabels(band_stats["score_band"],
                   rotation=45, ha="right", fontsize=9)
ax.axhline(80, color="red", linestyle="--",
           linewidth=2, label="80% threshold")
ax.axhline(50, color="orange", linestyle="--",
           linewidth=1.5, label="50% threshold")
ax.set_ylabel("Approval Rate (%)", fontsize=11)
ax.set_title("Loan Approval Rate by CIBIL Score Band\n"
             f"(80% threshold at band: {threshold_80} | "
             f"RBI recommends: 750+)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 110)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, pct in zip(bars, band_stats["approval_pct"]):
    if pct > 5:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{pct:.0f}%",
                ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "34_cibil_bands.png"),
            bbox_inches="tight")
plt.close()
print("\n✅ Saved: 34_cibil_bands.png")

# ── CHART 2: Smooth probability curve ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(scores, [p*100 for p in smooth_probas],
        color="#2196F3", linewidth=2.5,
        label="Model approval probability")
ax.axhline(80, color="red", linestyle="--",
           linewidth=1.5, label="80% approval threshold")
ax.axhline(50, color="orange", linestyle="--",
           linewidth=1.5, label="50% approval threshold")
ax.axvline(750, color="green", linestyle="--",
           linewidth=1.5, label="RBI recommended minimum (750)")

# Find where curve crosses 80%
cross_80 = None
for i, (s, p) in enumerate(zip(scores, smooth_probas)):
    if p >= 0.80:
        cross_80 = s
        break
if cross_80:
    ax.axvline(cross_80, color="purple", linestyle=":",
               linewidth=2,
               label=f"Our data 80% crossing ({cross_80})")

ax.fill_between(scores,
                [p*100 for p in smooth_probas],
                alpha=0.1, color="#2196F3")
ax.set_xlabel("CIBIL Score", fontsize=11)
ax.set_ylabel("Approval Probability (%)", fontsize=11)
ax.set_title("Approval Probability vs CIBIL Score\n"
             "(Smooth curve — all other features held at median)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(300, 900)
ax.set_ylim(0, 110)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "35_cibil_smooth_curve.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 35_cibil_smooth_curve.png")

# ── CHART 3: CIBIL Score Distribution by Outcome ──
fig, ax = plt.subplots(figsize=(9, 5))
approved = cibil[cibil["loan_status_encoded"]==1]["cibil_score"]
rejected = cibil[cibil["loan_status_encoded"]==0]["cibil_score"]
ax.hist(approved, bins=30, alpha=0.6,
        color="#4CAF50", label=f"Approved (n={len(approved)})",
        edgecolor="white")
ax.hist(rejected, bins=30, alpha=0.6,
        color="#F44336", label=f"Rejected (n={len(rejected)})",
        edgecolor="white")
ax.axvline(approved.mean(), color="green", linestyle="--",
           linewidth=2,
           label=f"Approved mean: {approved.mean():.0f}")
ax.axvline(rejected.mean(), color="red", linestyle="--",
           linewidth=2,
           label=f"Rejected mean: {rejected.mean():.0f}")
ax.axvline(750, color="black", linestyle=":",
           linewidth=2, label="RBI minimum (750)")
ax.set_title("CIBIL Score Distribution by Loan Outcome",
             fontsize=12, fontweight="bold")
ax.set_xlabel("CIBIL Score", fontsize=11)
ax.set_ylabel("Number of Applicants", fontsize=11)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "36_cibil_distribution.png"),
            bbox_inches="tight")
plt.close()
print("✅ Saved: 36_cibil_distribution.png")

print(f"\n── Key Statistics ──")
print(f"Approved applicants — mean CIBIL: {approved.mean():.0f}")
print(f"Rejected applicants — mean CIBIL: {rejected.mean():.0f}")
print(f"Gap: {approved.mean()-rejected.mean():.0f} points")
print(f"RBI minimum (750) vs our 80% threshold ({threshold_80}): "
      f"{'aligned' if '750' in str(threshold_80) else 'different'}")

band_stats.to_csv(
    os.path.join(DATA_DIR, "cibil_band_stats.csv"), index=False
)
print("\n✅ Saved: cibil_band_stats.csv")

print("\n" + "=" * 60)
print("✅ CIBIL INFLECTION ANALYSIS COMPLETE")
print(f"   50% approval threshold: band {threshold_50}")
print(f"   80% approval threshold: band {threshold_80}")
print(f"   90% approval threshold: band {threshold_90}")
print("   Run 18_intersectional_mitigation.py next")
print("=" * 60)