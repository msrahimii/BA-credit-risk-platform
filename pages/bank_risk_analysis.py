"""Bank Risk Analysis Page - Assess borrower default probability."""

import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import json
import plotly.graph_objects as go
from utils import COLORS, gauge_chart, plotly_layout, section_header, status_badge, info_card


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

st.html(section_header("Bank Risk Analysis",
                        "Comprehensive borrower default risk assessment for financial institutions"))

# ── Input Form (more detailed than borrower page) ──
st.markdown("### Borrower Application Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Loan Details**")
    loan_amnt = st.number_input("Loan Amount ($)", 500, 40000, 15000, 500)
    term = st.selectbox("Term", [36, 60])
    int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.0, 0.5)
    purpose = st.selectbox("Purpose", sorted(artifacts["label_encodings"].get("purpose", {}).keys()),
                           format_func=lambda x: x.replace("_", " ").title())
    sub_grade_str = st.selectbox("Sub Grade", sorted(
        artifacts["sub_grade_map"].keys(),
        key=lambda x: artifacts["sub_grade_map"][x]))

with col2:
    st.markdown(f"**Borrower Profile**")
    annual_inc = st.number_input("Annual Income ($)", 10000, 1000000, 65000, 5000)
    emp_length = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
    ], index=5)
    home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    state = st.selectbox("State", sorted(artifacts["state_target_encoding"].keys()))
    app_type = st.selectbox("Application Type", ["Individual", "Joint App"])

with col3:
    st.markdown(f"**Credit Profile**")
    fico = st.number_input("FICO Score", 600, 850, 700, 5)
    dti = st.number_input("DTI (%)", 0.0, 60.0, 18.0, 0.5)
    revol_util = st.number_input("Revolving Utilization (%)", 0.0, 120.0, 45.0, 1.0)
    open_acc = st.number_input("Open Accounts", 0, 50, 10, 1)
    total_acc = st.number_input("Total Accounts", 1, 100, 25, 1)
    delinq_2yrs = st.number_input("Delinquencies (2yr)", 0, 20, 0, 1)

analyze_btn = st.button("Analyze Risk", use_container_width=True)

if analyze_btn:
    # Build feature vector
    feature_cols = artifacts["feature_columns"]
    row = {}
    for col in feature_cols:
        row[col] = artifacts["medians"].get(col, 0)

    row["loan_amnt"] = loan_amnt
    row["term"] = term
    row["int_rate"] = int_rate
    row["annual_inc"] = annual_inc
    row["dti"] = dti
    row["fico_range_low"] = fico
    row["revol_util"] = revol_util
    row["open_acc"] = open_acc
    row["total_acc"] = total_acc
    row["delinq_2yrs"] = delinq_2yrs
    row["sub_grade"] = artifacts["sub_grade_map"].get(sub_grade_str, 17)
    row["emp_length"] = artifacts["emp_length_map"].get(emp_length, 5)
    row["credit_history_months"] = 180

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
            "application_type", {}).get(app_type, 0)
    if "initial_list_status" in feature_cols:
        row["initial_list_status"] = artifacts["label_encodings"].get(
            "initial_list_status", {}).get("w", 0)

    X_input = pd.DataFrame([row])[feature_cols]

    prob_default = model.predict_proba(X_input)[0][1]
    prob_default_pct = prob_default * 100
    threshold = metadata["bank_threshold"]
    is_risky = prob_default >= threshold

    st.divider()

    # ── Results ──
    col_gauge, col_detail = st.columns([1, 1.5])

    with col_gauge:
        st.markdown("### Default Risk Score")
        fig = gauge_chart(
            prob_default_pct,
            title="Probability of Default",
            color_ranges=[
                (0, 20, COLORS["success"]),
                (20, 50, COLORS["warning"]),
                (50, 100, COLORS["danger"]),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk verdict
        if is_risky:
            risk_label = "HIGH RISK"
            risk_color = COLORS["danger"]
            risk_bg = "linear-gradient(135deg, #450a0a, #7f1d1d)"
            risk_msg = "This borrower exceeds the risk threshold. Recommend additional review or denial."
        elif prob_default_pct > 25:
            risk_label = "MODERATE RISK"
            risk_color = COLORS["warning"]
            risk_bg = "linear-gradient(135deg, #451a03, #78350f)"
            risk_msg = "Borrower shows some risk factors. Consider additional collateral or adjusted terms."
        else:
            risk_label = "LOW RISK"
            risk_color = COLORS["success"]
            risk_bg = "linear-gradient(135deg, #064e3b, #065f46)"
            risk_msg = "Borrower profile is within acceptable risk parameters."

        badge_color_key = "danger" if is_risky else ("warning" if prob_default_pct > 25 else "success")
        st.html(
            f'<div style="background:{risk_bg};border:1px solid {risk_color};'
            f'border-radius:16px;padding:24px;text-align:center;'
            f'box-shadow:0 0 30px rgba(0,0,0,0.2);">'
            f'<div style="margin-bottom:8px;">{status_badge(risk_label, badge_color_key)}</div>'
            f'<span style="color:{COLORS["text_muted"]};font-size:0.95rem;">{risk_msg}</span>'
            f'</div>'
        )

    with col_detail:
        st.markdown("### Risk Factor Analysis")

        # SHAP waterfall-style horizontal bar
        shap_values = explainer.shap_values(X_input)
        shap_series = pd.Series(shap_values[0], index=feature_cols)
        top_features = shap_series.abs().nlargest(12)
        top_shap = shap_series[top_features.index].sort_values()

        colors = [COLORS["success"] if v < 0 else COLORS["danger"] for v in top_shap.values]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f.replace("_", " ").title()[:25] for f in top_shap.index],
            x=top_shap.values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
        ))
        fig.update_layout(
            title="SHAP Feature Contributions",
            xaxis_title="Impact on Default Probability",
        )
        fig = plotly_layout(fig, height=450)
        fig.update_layout(
            yaxis=dict(tickfont=dict(size=11)),
            xaxis=dict(zeroline=True, zerolinecolor=COLORS["text_muted"],
                       zerolinewidth=1),
        )
        fig.add_annotation(
            x=max(top_shap.values) * 0.5, y=1.05, xref="x", yref="paper",
            text=f"<span style='color:{COLORS['danger']}'>Increases Risk →</span>",
            showarrow=False, font=dict(size=10),
        )
        fig.add_annotation(
            x=min(top_shap.values) * 0.5, y=1.05, xref="x", yref="paper",
            text=f"<span style='color:{COLORS['success']}'>← Decreases Risk</span>",
            showarrow=False, font=dict(size=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Summary Table ──
    st.markdown("### Application Summary")
    summary = {
        "Loan Amount": f"${loan_amnt:,.0f}",
        "Term": f"{term} months",
        "Interest Rate": f"{int_rate:.2f}%",
        "Sub Grade": sub_grade_str,
        "FICO Score": str(fico),
        "DTI": f"{dti:.1f}%",
        "Annual Income": f"${annual_inc:,.0f}",
        "Default Probability": f"{prob_default_pct:.1f}%",
        "Risk Classification": risk_label,
        "Threshold Used": f"{threshold:.2%}",
    }
    scol1, scol2 = st.columns(2)
    items = list(summary.items())
    for i, (k, v) in enumerate(items):
        col = scol1 if i < len(items) // 2 else scol2
        col.markdown(
            f"""<div style="display:flex;justify-content:space-between;
            padding:8px 16px;border-bottom:1px solid {COLORS['border']};">
            <span style="color:{COLORS['text_muted']}">{k}</span>
            <span style="color:{COLORS['text']};font-weight:600;">{v}</span>
            </div>""",
            unsafe_allow_html=True,
        )
