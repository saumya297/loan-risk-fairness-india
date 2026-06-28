import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──
st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

# ── Load models ──
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "models", "lgbm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "models", "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(base, "models", "lgbm_cibil.pkl"), "rb") as f:
        cibil_model = pickle.load(f)
    with open(os.path.join(base, "models",
                           "cibil_feature_cols.pkl"), "rb") as f:
        cibil_cols = pickle.load(f)
    return model, feature_cols, cibil_model, cibil_cols

model, FEATURE_COLS, cibil_model, CIBIL_COLS = load_models()

# ── SHAP explainer ──
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

# ── Helper functions ──
READABLE = {
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

def get_risk_tier(proba):
    if proba < 0.35:
        return "🔴 High Risk", "#F44336"
    elif proba < 0.65:
        return "🟡 Medium Risk", "#FF9800"
    else:
        return "🟢 Low Risk", "#4CAF50"

def calculate_emi(loan_k, term, rate=8.5):
    P = loan_k * 1000
    r = rate / (12 * 100)
    n = term
    if r == 0 or n == 0:
        return 0
    return P * r * (1+r)**n / ((1+r)**n - 1)

def find_max_loan(features_dict, min_l=10, max_l=700, steps=50):
    amounts = np.linspace(min_l, max_l, steps)
    last_ok = None
    for amt in amounts:
        f = features_dict.copy()
        f["LoanAmount"] = amt
        f["loan_to_income_ratio"] = amt / (f["total_income"]/1000 + 0.001)
        f["loan_to_term_ratio"] = amt / (f["Loan_Amount_Term"] + 0.001)
        X = pd.DataFrame([f])[FEATURE_COLS]
        p = model.predict_proba(X)[0][1]
        if p >= 0.5:
            last_ok = amt
    return last_ok

# ── App Header ──
st.title("🏦 Fair & Explainable Loan Risk Assessment")
st.markdown(
    "**AI-powered loan risk prediction with fairness analysis** | "
    "Indian Home & Personal Loans"
)
st.markdown("---")

# ── Sidebar ──
st.sidebar.header("📋 Applicant Details")
st.sidebar.markdown("Fill in the applicant information below:")

loan_type = st.sidebar.selectbox(
    "Loan Type",
    ["Home Loan", "Personal Loan (CIBIL)"]
)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox(
    "Education", ["Graduate", "Not Graduate"]
)
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)
applicant_income = st.sidebar.number_input(
    "Applicant Income (₹/month)", 
    min_value=0, max_value=100000,
    value=5000, step=500
)
coapplicant_income = st.sidebar.number_input(
    "Coapplicant Income (₹/month)",
    min_value=0, max_value=50000,
    value=0, step=500
)
loan_amount = st.sidebar.number_input(
    "Loan Amount (₹ thousands)",
    min_value=10, max_value=700,
    value=150, step=10
)
loan_term = st.sidebar.selectbox(
    "Loan Term (months)",
    [60, 120, 180, 240, 300, 360, 480],
    index=5
)
credit_history = st.sidebar.selectbox(
    "Credit History",
    [1.0, 0.0],
    format_func=lambda x: "Has credit history" if x == 1.0
                          else "No credit history"
)

predict_btn = st.sidebar.button(
    "🔍 Assess Risk", type="primary", use_container_width=True
)

# ── Main content ──
if predict_btn:
    # Build features
    total_income = applicant_income + coapplicant_income
    features = {
        "Gender_enc": 1 if gender == "Male" else 0,
        "Married_enc": 1 if married == "Yes" else 0,
        "Education_enc": 1 if education == "Graduate" else 0,
        "SelfEmployed_enc": 1 if self_employed == "Yes" else 0,
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "total_income": total_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "loan_to_income_ratio": loan_amount / (total_income/1000 + 0.001),
        "income_per_dependent": total_income / (dependents + 1),
        "loan_to_term_ratio": loan_amount / (loan_term + 0.001),
        "Area_Rural": 1 if property_area == "Rural" else 0,
        "Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Area_Urban": 1 if property_area == "Urban" else 0
    }

    X_input = pd.DataFrame([features])[FEATURE_COLS]
    proba = model.predict_proba(X_input)[0][1]
    prediction = "Approved" if proba >= 0.5 else "Rejected"
    risk_label, risk_color = get_risk_tier(proba)

    # EMI calculation
    emi = calculate_emi(loan_amount, loan_term)
    monthly_income = total_income
    emi_ratio = emi / monthly_income * 100 if monthly_income > 0 else 0
    emi_ok = emi_ratio <= 40

    # ── Results row ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Decision",
                  "✅ Approved" if prediction == "Approved"
                  else "❌ Rejected")
    with col2:
        st.metric("Approval Probability", f"{proba*100:.1f}%")
    with col3:
        st.metric("Risk Tier", risk_label)
    with col4:
        st.metric("Monthly EMI",
                  f"₹{emi:,.0f}",
                  delta=f"{'✅ Affordable' if emi_ok else '⚠️ Too High'}",
                  delta_color="normal" if emi_ok else "inverse")

    st.markdown("---")

    # ── Two columns layout ──
    left, right = st.columns(2)

    with left:
        st.subheader("📊 SHAP Explanation")
        st.caption("What factors drove this decision?")

        shap_vals = explainer.shap_values(X_input)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_row = sv[0]

        readable_names = [READABLE.get(c, c) for c in FEATURE_COLS]
        pairs = sorted(
            zip(readable_names, sv_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:8]

        names = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        colors_shap = ["#4CAF50" if v > 0 else "#F44336" for v in values]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(names[::-1], values[::-1],
                       color=colors_shap[::-1], edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Feature Impact on Decision\n"
                     "(Green = increases approval, Red = decreases)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Value")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Top 3 reasons:**")
        for name, val in pairs[:3]:
            icon = "✅" if val > 0 else "❌"
            direction = "increased" if val > 0 else "decreased"
            st.markdown(
                f"{icon} **{name}** {direction} approval likelihood"
            )

    with right:
        st.subheader("⚖️ Fairness Context")
        st.caption(
            "How does this applicant compare to their demographic group?"
        )

        group_rates = {
            ("Male", "Urban"): 67.5,
            ("Male", "Semiurban"): 77.3,
            ("Male", "Rural"): 62.3,
            ("Female", "Urban"): 57.6,
            ("Female", "Semiurban"): 76.4,
            ("Female", "Rural"): 58.3
        }
        group_rate = group_rates.get((gender, property_area), 68.7)

        st.info(
            f"📊 Historical approval rate for "
            f"**{gender} | {property_area}** applicants: "
            f"**{group_rate:.1f}%**"
        )

        if gender == "Female" and property_area == "Rural":
            st.warning(
                "⚠️ **Intersectional flag:** Female Rural applicants "
                "have the lowest historical approval rate (58.3%). "
                "Our fairness audit found this gap is driven by "
                "income differences, not model bias."
            )
        elif property_area == "Rural":
            st.warning(
                "⚠️ **Area flag:** Rural applicants receive 5.9% lower "
                "model approval than actual data justifies. "
                "This is a known model bias we are monitoring."
            )

        st.subheader("💡 EMI Affordability")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Monthly EMI", f"₹{emi:,.0f}")
            st.metric("EMI / Income",
                      f"{emi_ratio:.1f}%",
                      delta="RBI limit: 40%")
        with col_b:
            if emi_ok:
                st.success("✅ Within RBI debt-service guideline")
            else:
                st.error("❌ Exceeds RBI debt-service guideline")

        if prediction == "Rejected":
            st.subheader("🔧 Loan Optimizer")
            with st.spinner("Finding maximum approvable amount..."):
                max_loan = find_max_loan(features)
            if max_loan:
                max_emi = calculate_emi(max_loan, loan_term)
                max_ratio = max_emi / monthly_income * 100
                st.success(
                    f"💡 **You could be approved for up to "
                    f"₹{max_loan:.0f}K**\n\n"
                    f"EMI would be ₹{max_emi:,.0f}/month "
                    f"({max_ratio:.0f}% of income)"
                )
            else:
                st.warning(
                    "⚠️ Based on current profile, approval is unlikely "
                    "at any loan amount. Focus on improving credit "
                    "history and income first."
                )

    st.markdown("---")

    # ── Plain language explanation ──
    st.subheader("📝 Plain Language Explanation")
    top3 = pairs[:3]
    reasons_text = ", ".join([
        f"{'strong' if abs(v) > 1 else 'moderate'} "
        f"{'positive' if v > 0 else 'negative'} impact from {n}"
        for n, v in top3
    ])

    if prediction == "Approved":
        explanation = (
            f"Your loan application of ₹{loan_amount}K has been "
            f"**approved** with {proba*100:.0f}% confidence. "
            f"The main factors supporting your application were: "
            f"{reasons_text}. "
            f"Your monthly EMI of ₹{emi:,.0f} is "
            f"{'within' if emi_ok else 'above'} the recommended "
            f"40% of income limit. "
            f"Please maintain timely payments to protect your "
            f"credit score."
        )
    else:
        explanation = (
            f"Your loan application of ₹{loan_amount}K has been "
            f"**rejected** at this time. "
            f"The main factors were: {reasons_text}. "
        )
        if max_loan and max_loan > 0:
            explanation += (
                f"You may be approved for a smaller amount of "
                f"₹{max_loan:.0f}K — consider reapplying with "
                f"this reduced amount. "
            )
        if credit_history == 0:
            explanation += (
                "Building a credit history by repaying a smaller "
                "loan or credit card on time would significantly "
                "improve your future chances."
            )

    st.info(explanation)

    # ── Simplified version ──
    with st.expander("🔤 Simplified Explanation (plain language)"):
        if prediction == "Approved":
            simple = (
                f"Good news! Your loan of ₹{loan_amount}K is approved. "
                f"You will pay ₹{emi:,.0f} every month. "
                f"Please pay on time every month to keep your good record."
            )
        else:
            simple = (
                f"We cannot approve your loan of ₹{loan_amount}K right now. "
            )
            if credit_history == 0:
                simple += (
                    "The biggest reason is that you do not have a credit "
                    "history. Try getting a small loan first and paying "
                    "it back on time. "
                )
            if max_loan:
                simple += (
                    f"You could try asking for a smaller amount of "
                    f"₹{max_loan:.0f}K instead."
                )
        st.write(simple)

else:
    # ── Default landing page ──
    st.info(
        "👈 Fill in the applicant details in the sidebar "
        "and click **Assess Risk** to get started"
    )

    st.markdown("## 📈 Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model AUC", "0.8322")
    col2.metric("Accuracy", "79.7%")
    col3.metric("Intersectional Gap", "52.1pp")
    col4.metric("Charts Generated", "27+")

    st.markdown("---")
    st.markdown("### 🔍 Key Findings")
    st.markdown("""
    | Group | Approval Rate |
    |-------|--------------|
    | 🔴 Female Rural | 25.0% |
    | 🔴 Female Urban | 28.6% |
    | 🟡 Male Rural | 63.3% |
    | 🟢 Male Semiurban | 77.1% |
    | 🟢 Female Semiurban | 78.6% |
    | 🟢 Male Urban | 72.7% |
    """)

    st.markdown("---")
    st.markdown("### 🏗️ Pipeline")
    st.markdown("""
    1. **Data Cleaning** — 2 Indian datasets, missing value imputation
    2. **Feature Engineering** — 17 features including derived ratios
    3. **LightGBM Model** — AUC 0.83, class-balanced training
    4. **SHAP Explainability** — Global + individual explanations
    5. **Fairness Audit** — Gender × Rural/Urban intersectional analysis
    6. **Bias Mitigation** — Demographic parity correction
    7. **Loan Optimizer** — Max approvable amount for rejected applicants
    8. **PSI** — Model stability validation
    9. **LLM Layer** — Plain language customer guidance
    """)