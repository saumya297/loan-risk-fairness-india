# Fair & Explainable Credit Risk Assessment
### Gender and Rural-Urban Analysis | Indian Home & Personal Loans

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.83-green)
![Fairlearn](https://img.shields.io/badge/Fairlearn-Fairness%20Audit-orange)

---

## What This Project Does

Banks use ML models to approve or reject loan applications — but these 
models can treat certain groups unfairly without anyone realising it.

This project builds a complete credit risk pipeline that:
1. **Predicts** loan approval risk for Indian applicants
2. **Explains** every decision using SHAP (which factors drove it)
3. **Audits** whether the model treats men/women and rural/urban 
   applicants fairly
4. **Translates** predictions into plain-language customer guidance 
   using an LLM layer

**Datasets:** Dream Housing Finance (home loans) + Indian Bank CIBIL 
(personal loans)

---

## Key Results

| Metric | Value |
|--------|-------|
| Model AUC-ROC | 0.8322 |
| Model Accuracy | 79.7% |
| Overall approval rate | 68.7% |
| Gender gap (raw data) | 2.2pp (Male 69.3% vs Female 67.0%) |
| Rural vs Semiurban gap | 15.3pp (61.5% vs 76.8%) |

### Headline Fairness Finding
| Group | Approval Rate |
|-------|--------------|
| Female Rural | 25.0% |
| Female Urban | 28.6% |
| Male Rural | 63.3% |
| Male Semiurban | 77.1% |
| Female Semiurban | 78.6% |
| Male Urban | 72.7% |

**Intersectional gap: 52.1 percentage points** between Female Rural (25%) 
and Male Semiurban (77.1%)

**Key finding:** The gender gap is driven by real income differences, 
NOT model bias — confirmed by near-zero model vs actual approval rate 
gap for both genders. However, the model is 5.9pp harsher on rural 
applicants than actual data justifies — genuine model bias.

---

## Key Charts

### Approval Rate: Gender × Property Area
![Gender x Area](reports/figures/03_approval_gender_x_area.png)

### SHAP Summary — What Drives Decisions
![SHAP Summary](reports/figures/12_shap_summary.png)

### Fairness Audit — Actual vs Model by Gender
![Fairness Gender](reports/figures/15_fairness_gender.png)

### Intersectional Heatmap
![Intersectional](reports/figures/17_intersectional_heatmap.png)

---

## Project Structure