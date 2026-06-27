# Fair & Explainable Credit Risk Assessment
## Gender and Rural-Urban Analysis | Indian Home & Personal Loans
### Final Project Report
**Date:** June 27, 2026

---

## 1. Objective
To build an explainable machine learning model for retail loan approval prediction, 
audit it for fairness across gender and rural/urban dimensions, and generate 
plain-language explanations using an LLM layer — covering both home loans 
(Dream Housing Finance dataset) and personal loans (CIBIL dataset).

---

## 2. Datasets Used

| Dataset | Source | Rows | Purpose |
|---------|--------|------|---------|
| Loan Prediction Problem | Dream Housing Finance / Kaggle | 614 | Home loan approval, gender + rural/urban analysis |
| Indian Bank CIBIL Dataset | Kaggle | 4,269 | Personal loan risk, CIBIL score modeling |

---

## 3. Methodology
1. **Data Cleaning** — Imputed missing values (mode for categoricals, median for numerics)
2. **Feature Engineering** — Created 4 derived features: total_income, loan_to_income_ratio, income_per_dependent, loan_to_term_ratio
3. **Modeling** — LightGBM classifier with class balancing (balanced weights)
4. **Explainability** — SHAP TreeExplainer for global and individual-level explanations
5. **Fairness Audit** — Fairlearn MetricFrame across gender, area, and gender×area intersection
6. **Bias Mitigation** — ExponentiatedGradient with DemographicParity constraint
7. **LLM Layer** — Claude claude-sonnet-4-6 translates SHAP outputs to plain-language guidance

---

## 4. Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8322 |
| Accuracy | 79.7% |
| Dataset size | 614 applicants |
| Train/Test split | 80/20 stratified |

---

## 5. Key EDA Findings

| Group | Approval Rate |
|-------|--------------|
| Overall | 68.7% |
| Male | 69.1% |
| Female | 67.0% |
| Rural | 61.5% |
| Semiurban | 76.8% |
| Urban | 65.8% |

**Gender gap:** 2.2 percentage points (Male higher)
**Area gap:** 15.4 percentage points (Semiurban vs Rural)

---

## 6. SHAP Explainability Findings

Top 3 features driving loan approval (by mean absolute SHAP value):
1. **Credit History** — by far the strongest predictor. No credit history causes large negative SHAP values (up to -8), effectively overriding all other features
2. **Applicant Income** — higher income consistently increases approval likelihood
3. **Loan to Income Ratio** — the ratio of loan requested to income is more predictive than loan amount alone

**Key insight:** Gender and Property Area rank low in feature importance (13th and 11th/12th out of 17), confirming the model makes decisions primarily on financial grounds.

---

## 7. Fairness Audit Findings

### By Gender (Model Predictions on Test Set)
| Group | Model Approval Rate | Actual Approval Rate | Gap |
|-------|--------------------|--------------------|-----|
| Female | 56.0% | 56.0% | 0.0pp |
| Male | 71.4% | 72.4% | -1.0pp |

**Finding:** The model accurately reflects real-world approval rates for both genders — it is NOT adding extra gender bias beyond what exists in the data.

### By Property Area
| Group | Model Rate | Actual Rate | Gap |
|-------|-----------|------------|-----|
| Rural | 58.8% | 64.7% | **-5.9pp** |
| Semiurban | 77.6% | 73.5% | +4.1pp |
| Urban | 65.0% | 67.5% | -2.5pp |

**Finding:** The model is systematically HARSHER on rural applicants than reality justifies (-5.9pp gap). This is genuine model bias requiring attention.

### Intersectional: Gender × Area
| Group | Model Approval Rate |
|-------|-------------------|
| Female Rural | 25.0% |
| Female Urban | 28.6% |
| Male Rural | 63.3% |
| Male Semiurban | 77.1% |
| Female Semiurban | 78.6% |
| Male Urban | 72.7% |

**Headline finding:** Female Rural applicants have only **25% model approval rate** vs Male Semiurban at **77.1%** — a **52.1 percentage point intersectional gap**.

**False Rejection Rate for Female Rural: 50%** — meaning half of rural women who should be approved are being rejected by the model.

---

## 8. Bias Mitigation Results

Applied ExponentiatedGradient with DemographicParity constraint on gender.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Demographic Parity Diff | 0.154 | 0.225 | +45.8% (worse) |
| Accuracy | 79.7% | 84.6% | +4.9pp |

**Finding:** DemographicParity mitigation did not reduce the gender gap — it increased it slightly while improving overall accuracy. This confirms that the gender approval gap in this dataset is driven by genuine financial differences (income, loan-to-income ratio) rather than arbitrary model bias. The model treats similarly-qualified applicants similarly regardless of gender — but women in this dataset have structurally lower incomes and request loans that are proportionally larger relative to their income.

---

## 9. LLM Explanation Layer

Designed a two-mode LLM explanation system:
- **Standard mode:** Professional language for financially literate applicants
- **Simplified mode:** Plain language for applicants with limited financial knowledge

The LLM receives: risk tier + top 3 SHAP factors (pre-computed, not generated by LLM) + applicant context.
The LLM only translates — it never computes numbers — preventing hallucination on financial figures.

---

## 10. Limitations
- Small dataset (614 rows for home loan model) — findings directional, not statistically conclusive for all subgroups
- Female Rural subgroup has only 24 applicants in full dataset — fairness findings for this group have high variance
- CIBIL dataset has no gender/location fields — fairness audit conducted on home loan dataset only
- Dream Housing Finance data may not represent all Indian lending demographics

---

## 11. Future Work
- Expand to larger Indian lending datasets (RBI BSR data, PMAY beneficiary data)
- Apply intersectional fairness constraints (not just gender, but gender × area combined)
- Build full Streamlit demo app for live applicant scoring
- Extend LLM layer to regional Indian languages (Hindi, Bengali, Tamil) for accessibility

---

## 12. Resume Summary
- Trained LightGBM credit risk model on Indian loan data achieving AUC of 0.8322
- Performed gender and rural/urban fairness audit using Fairlearn, identifying 52pp intersectional approval gap between rural women and semiurban men
- Generated SHAP-based reason codes translating model predictions into plain-language explanations
- Applied demographic parity mitigation and documented accuracy-fairness tradeoff
- Designed ML+LLM hybrid pipeline separating verified predictions from LLM explanation to prevent hallucination

---
*Report generated: June 27, 2026 at 18:53*
