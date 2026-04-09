"""Borrower Assessment Page - Check your loan approval chances."""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import json
import plotly.graph_objects as go
from utils import COLORS, gauge_chart, plotly_layout, section_header, info_card, status_badge


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("artifacts/xgb_credit_risk_model.json")
    return model


@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("artifacts/preprocessing_artifacts.pkl")
    with open("artifacts/model_metadata.json") as f:
        metadata = json.load(f)
    return artifacts, metadata


@st.cache_resource
def load_explainer():
    return joblib.load("artifacts/shap_explainer.pkl")


model = load_model()
artifacts, metadata = load_artifacts()
explainer = load_explainer()

# ── FICO to sub_grade mapping (Lending Club approximate) ──
FICO_TO_GRADE = [
    (750, "A"), (720, "B"), (690, "C"), (660, "D"),
    (640, "E"), (620, "F"), (0, "G"),
]


def fico_to_sub_grade_num(fico):
    grade = "G"
    for threshold, g in FICO_TO_GRADE:
        if fico >= threshold:
            grade = g
            break
    within = min(max((fico % 30) // 6, 0), 4)
    sg = f"{grade}{5 - within}"
    return artifacts["sub_grade_map"].get(sg, 17)


def fico_to_int_rate(fico):
    return max(5.0, 30.0 - (fico - 600) * 0.12)


# ── Advice templates ──
ADVICE = {
    "dti": "Your debt-to-income ratio ({val:.1f}%) is high. Paying down existing debts before applying would improve your chances.",
    "int_rate": "Your estimated interest rate ({val:.1f}%) is above average. A higher FICO score would lower this rate.",
    "loan_amnt": "You're requesting a large loan (${val:,.0f}). Consider applying for a smaller amount to improve approval odds.",
    "annual_inc": "Your income (${val:,.0f}) relative to your loan amount is a factor. Higher income or lower loan amount helps.",
    "revol_util": "Your credit utilization is estimated high. Keeping revolving balances below 30% of limits is recommended.",
    "term": "A 60-month term carries higher risk than 36 months. Choosing a shorter term would improve your profile.",
    "fico_range_low": "Your FICO score ({val:.0f}) is a key factor. Even a 20-point improvement would meaningfully help.",
    "sub_grade": "Your risk tier is below average. Improving your FICO score is the most direct way to move up.",
    "acc_open_past_24mths": "You've opened several accounts recently. Fewer new accounts signals financial stability.",
    "inq_last_6mths": "Multiple recent credit inquiries can signal risk. Avoid applying for new credit before this loan.",
    "home_ownership": "Homeownership is viewed favorably. If you're close to buying, that could help future applications.",
    "credit_history_months": "A longer credit history helps. Keep your oldest accounts open.",
    "emp_length": "Longer employment history signals stability.",
}


# ── Header ──
st.html(section_header("Borrower Assessment",
                        "Check your loan approval chances and get personalized advice"))

# ── Input Form ──
col_form, col_result = st.columns([1, 1.4])

with col_form:
    st.markdown("### Your Information")

    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000,
                                value=10000, step=500)
    term_choice = st.selectbox("Loan Term", ["36 months", "60 months"])
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "medical", "car", "small_business",
        "vacation", "moving", "house", "wedding", "other",
    ], format_func=lambda x: x.replace("_", " ").title())
    annual_inc = st.number_input("Annual Income ($)", min_value=10000,
                                 max_value=500000, value=60000, step=5000)
    emp_length = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
    ], index=5)
    home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    state = st.selectbox("State", sorted(artifacts["state_target_encoding"].keys()))
    fico = st.slider("FICO Score (estimate)", min_value=600, max_value=850, value=700)
    dti = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0,
                     value=15.0, step=0.5)

    predict_btn = st.button("Check My Chances", use_container_width=True)

with col_result:
    if predict_btn:
        # Build feature vector using medians for features not collected
        feature_cols = artifacts["feature_columns"]
        row = {}
        for col in feature_cols:
            row[col] = artifacts["medians"].get(col, 0)

        # Fill in user inputs
        row["loan_amnt"] = loan_amnt
        row["term"] = 36 if term_choice == "36 months" else 60
        row["annual_inc"] = annual_inc
        row["dti"] = dti
        row["fico_range_low"] = fico
        row["int_rate"] = fico_to_int_rate(fico)
        row["sub_grade"] = fico_to_sub_grade_num(fico)
        row["emp_length"] = artifacts["emp_length_map"].get(emp_length, 5)
        row["credit_history_months"] = 180

        # Encode categoricals
        if "home_ownership" in feature_cols:
            row["home_ownership"] = artifacts["label_encodings"].get(
                "home_ownership", {}).get(home, 0)
        if "purpose" in feature_cols:
            row["purpose"] = artifacts["label_encodings"].get(
                "purpose", {}).get(purpose, 0)
        if "addr_state_encoded" in feature_cols:
            row["addr_state_encoded"] = artifacts["state_target_encoding"].get(
                state, artifacts["global_default_rate"])
        if "verification_status" in feature_cols:
            row["verification_status"] = artifacts["label_encodings"].get(
                "verification_status", {}).get("Not Verified", 0)
        if "application_type" in feature_cols:
            row["application_type"] = artifacts["label_encodings"].get(
                "application_type", {}).get("Individual", 0)
        if "initial_list_status" in feature_cols:
            row["initial_list_status"] = artifacts["label_encodings"].get(
                "initial_list_status", {}).get("w", 0)

        X_input = pd.DataFrame([row])[feature_cols]

        # Predict
        prob_default = model.predict_proba(X_input)[0][1]
        prob_approval = (1 - prob_default) * 100

        threshold = metadata["borrower_threshold"]
        is_likely_approved = prob_default < threshold

        # ── Display Results ──
        st.markdown("### Your Results")

        # Gauge
        fig = gauge_chart(
            prob_approval,
            title="Approval Likelihood",
            color_ranges=[
                (0, 40, COLORS["danger"]),
                (40, 70, COLORS["warning"]),
                (70, 100, COLORS["success"]),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Verdict
        if is_likely_approved:
            st.html(
                f'<div style="background:linear-gradient(135deg,#064e3b,#065f46);'
                f'border:1px solid {COLORS["success"]};border-radius:16px;padding:24px;'
                f'text-align:center;box-shadow:0 0 30px rgba(16,185,129,0.1);">'
                f'<div style="margin-bottom:8px;">{status_badge("LIKELY TO BE APPROVED", "success")}</div>'
                f'<span style="color:#a7f3d0;font-size:0.95rem;">'
                f'Your profile shows strong approval indicators</span></div>'
            )
        else:
            st.html(
                f'<div style="background:linear-gradient(135deg,#451a03,#78350f);'
                f'border:1px solid {COLORS["warning"]};border-radius:16px;padding:24px;'
                f'text-align:center;box-shadow:0 0 30px rgba(245,158,11,0.1);">'
                f'<div style="margin-bottom:8px;">{status_badge("APPROVAL UNCERTAIN", "warning")}</div>'
                f'<span style="color:#fde68a;font-size:0.95rem;">'
                f'Review the advice below to improve your chances</span></div>'
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP Advice ──
        st.markdown("### Personalized Advice")
        shap_values = explainer.shap_values(X_input)
        shap_series = pd.Series(shap_values[0], index=feature_cols)

        # Positive SHAP = increases default probability = bad for borrower
        worst_features = shap_series.nlargest(5)

        advice_given = 0
        for feat, shap_val in worst_features.items():
            if shap_val <= 0 or advice_given >= 3:
                break
            template = ADVICE.get(feat)
            if template:
                val = X_input[feat].iloc[0]
                advice_text = template.format(val=val)
                severity = "High" if shap_val > 0.1 else "Medium"
                sev_color = COLORS["danger"] if severity == "High" else COLORS["warning"]
                st.html(info_card(f"{severity} Impact", advice_text, accent_color=sev_color))
                advice_given += 1

        if advice_given == 0:
            st.html(info_card("All Clear",
                              "Your profile looks strong across all factors. "
                              "No major areas for improvement identified.",
                              accent_color=COLORS["success"]))
    else:
        # Placeholder before prediction
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:center;
            height:500px;border:2px dashed {COLORS['border']};border-radius:16px;">
                <div style="text-align:center;">
                    <p style="color:{COLORS['text_muted']};font-size:1.2rem;">
                        Fill in your information and click<br>
                        <strong style="color:{COLORS['accent_light']};">
                        "Check My Chances"</strong>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
