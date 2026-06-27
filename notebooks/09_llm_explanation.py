import pandas as pd
import numpy as np
import pickle
import os
import json
import shap
import google.generativeai as genai

DATA_DIR = "../data"
MODEL_DIR = "../models"

print("=" * 60)
print("LLM EXPLANATION LAYER — Google Gemini")
print("=" * 60)

# ── Setup Gemini ──
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("⚠️  GEMINI_API_KEY not set — running in demo mode")
    USE_LLM = False
else:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    USE_LLM = True
    print("✅ Gemini connected")

# ── Load model and data ──
with open(os.path.join(MODEL_DIR, "lgbm_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv"))
X_test = df[FEATURE_COLS]
print(f"✅ Loaded {df.shape[0]} test applicants")

# ── Compute SHAP ──
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

readable = {
    "Credit_History": "Credit History",
    "loan_to_income_ratio": "Loan to Income Ratio",
    "LoanAmount": "Loan Amount",
    "total_income": "Total Income",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Coapplicant Income",
    "income_per_dependent": "Income per Dependent",
    "Loan_Amount_Term": "Loan Term",
    "loan_to_term_ratio": "Loan to Term Ratio",
    "Dependents": "No. of Dependents",
    "Gender_enc": "Gender",
    "Married_enc": "Married",
    "Education_enc": "Education",
    "SelfEmployed_enc": "Self Employed",
    "Area_Rural": "Area: Rural",
    "Area_Semiurban": "Area: Semiurban",
    "Area_Urban": "Area: Urban"
}
readable_names = [readable.get(c, c) for c in FEATURE_COLS]

def get_top_reasons(row_pos, top_n=3):
    pairs = list(zip(readable_names, sv[row_pos]))
    return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]

def build_prompt(data, style="standard"):
    reason_text = "\n".join([
        f"- {n}: {'helped' if v > 0 else 'hurt'} the application (impact: {v:.2f})"
        for n, v in data["reasons"]
    ])
    if style == "standard":
        return f"""You are a helpful loan officer assistant at an Indian bank.

A loan applicant has been assessed by our credit risk model:

- Gender: {data['gender']}
- Property Area: {data['area']}
- Annual Income: ₹{data['income']:,}
- Loan Amount: ₹{data['loan_amount']:,} thousand
- Risk Tier: {data['risk_tier']}
- Decision: {data['prediction']} (confidence: {data['probability']:.1%})

Top 3 factors:
{reason_text}

Write a professional, empathetic 3-4 sentence explanation covering:
1. The decision
2. Main reasons
3. One actionable suggestion if rejected, or encouragement if approved
Do not invent any numbers not given above."""

    else:
        return f"""You are a helpful loan officer at an Indian bank.

Loan applicant details:
- Gender: {data['gender']}, Area: {data['area']}
- Income: ₹{data['income']:,}, Loan: ₹{data['loan_amount']:,}K
- Decision: {data['prediction']}, Risk: {data['risk_tier']}

Factors:
{reason_text}

Write 3-4 very simple sentences in plain language anyone can understand.
No financial jargon. Give one simple tip if rejected.
Do not invent numbers."""

# ── Process sample applicants ──
samples = []
for tier in ["Low Risk", "Medium Risk", "High Risk"]:
    tier_rows = df[df["risk_tier"] == tier]
    if len(tier_rows) > 0:
        samples.append(tier_rows.iloc[0])

results = []

for i, row in enumerate(samples):
    row_pos = df.index.get_loc(row.name)
    reasons = get_top_reasons(row_pos)
    prediction = "Approved" if row["y_pred"] == 1 else "Rejected"

    data = {
        "risk_tier": row["risk_tier"],
        "gender": row["Gender_raw"],
        "area": row["Area_raw"],
        "income": int(row["ApplicantIncome"]),
        "loan_amount": int(row["LoanAmount"]),
        "prediction": prediction,
        "probability": float(row["y_proba"]),
        "reasons": reasons
    }

    print(f"\n{'='*60}")
    print(f"APPLICANT {i+1}: {row['risk_tier']} | {row['Gender_raw']} | {row['Area_raw']}")
    print(f"Decision: {prediction} | Confidence: {row['y_proba']:.1%}")
    print("Top reasons:")
    for name, val in reasons:
        print(f"  {'✅' if val > 0 else '❌'} {name} (SHAP={val:.3f})")

    for style in ["standard", "simplified"]:
        prompt = build_prompt(data, style)

        if USE_LLM:
            try:
                response = gemini_model.generate_content(prompt)
                explanation = response.text
                words = explanation.split()
                avg_word_len = np.mean([len(w) for w in words])

                print(f"\n  [{style.upper()}]")
                print(f"  {explanation}")
                print(f"  Word count: {len(words)} | Avg word length: {avg_word_len:.1f}")

                results.append({
                    "applicant": i+1,
                    "risk_tier": row["risk_tier"],
                    "gender": row["Gender_raw"],
                    "area": row["Area_raw"],
                    "style": style,
                    "explanation": explanation,
                    "word_count": len(words),
                    "avg_word_length": round(avg_word_len, 2)
                })
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
        else:
            print(f"\n  [{style.upper()}] Demo mode — prompt ready")
            results.append({
                "applicant": i+1,
                "style": style,
                "explanation": "demo mode",
                "prompt": prompt
            })

# ── Readability comparison ──
if USE_LLM:
    std = [r for r in results if r["style"] == "standard"]
    simp = [r for r in results if r["style"] == "simplified"]
    if std and simp:
        avg_std = np.mean([r["avg_word_length"] for r in std])
        avg_simp = np.mean([r["avg_word_length"] for r in simp])
        print(f"\n── Readability Comparison ──")
        print(f"  Standard avg word length:   {avg_std:.2f} chars")
        print(f"  Simplified avg word length: {avg_simp:.2f} chars")
        diff = avg_std - avg_simp
        print(f"  Simplified uses {'simpler' if diff > 0 else 'similar'} language (diff: {diff:.2f} chars/word)")

# ── Save ──
os.makedirs("../reports", exist_ok=True)
with open("../reports/llm_explanations.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved: reports/llm_explanations.json")

print("\n" + "=" * 60)
print("✅ LLM EXPLANATION COMPLETE — run 10_final_report.py next")
print("=" * 60)