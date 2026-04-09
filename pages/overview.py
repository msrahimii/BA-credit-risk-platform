"""Credit Risk Platform - Dashboard Overview"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from utils import COLORS, metric_card_html, plotly_layout, section_header

# ── Load KPIs ──
with open("data/viz/kpis.json") as f:
    kpis = json.load(f)

# ── Header ──
st.html(section_header("Dashboard Overview",
                        "Credit risk analytics powered by XGBoost and SHAP"))

# ── KPI Row ──
cols = st.columns(4)
with cols[0]:
    st.html(metric_card_html("Total Loans", f"{kpis['total_loans']:,.0f}"))
with cols[1]:
    st.html(metric_card_html("Default Rate", f"{kpis['default_rate']:.1%}",
                              delta="19.97% of loans", delta_color="danger"))
with cols[2]:
    st.html(metric_card_html("Avg Loan Amount", f"${kpis['avg_loan_amount']:,.0f}"))
with cols[3]:
    st.html(metric_card_html("Avg Interest Rate", f"{kpis['avg_int_rate']:.1f}%"))

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row ──
col1, col2 = st.columns([2, 1])

with col1:
    time_stats = pd.read_csv("data/viz/time_stats.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_stats["year_month"],
        y=time_stats["default_rate"] * 100,
        mode="lines",
        line=dict(color=COLORS["chart_purple"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(129, 140, 248, 0.1)",
        name="Default Rate %",
    ))
    fig.update_layout(title="Default Rate Over Time (Quarterly)")
    fig = plotly_layout(fig, height=380)
    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title="Default Rate (%)",
        xaxis=dict(tickangle=-45, dtick=4),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    grade_stats = pd.read_csv("data/viz/grade_stats.csv")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grade_stats["grade"],
        y=grade_stats["default_rate"] * 100,
        marker=dict(
            color=grade_stats["default_rate"],
            colorscale=[[0, COLORS["chart_green"]], [1, COLORS["chart_red"]]],
            line=dict(width=0),
        ),
        text=[f"{r:.1f}%" for r in grade_stats["default_rate"] * 100],
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"]),
    ))
    fig.update_layout(title="Default Rate by Grade")
    fig = plotly_layout(fig, height=380)
    fig.update_layout(xaxis_title="Grade", yaxis_title="Default Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

# ── Second Charts Row ──
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=time_stats["year_month"],
        y=time_stats["count"],
        marker=dict(color=COLORS["accent_light"], opacity=0.7),
        name="Loan Count",
    ))
    fig.update_layout(title="Loan Volume Over Time")
    fig = plotly_layout(fig, height=350)
    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title="Number of Loans",
        xaxis=dict(tickangle=-45, dtick=4),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    purpose_stats = pd.read_csv("data/viz/purpose_stats.csv").head(8)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=purpose_stats["purpose"],
        x=purpose_stats["count"],
        orientation="h",
        marker=dict(color=COLORS["chart_blue"], opacity=0.8),
        text=[f"{c:,.0f}" for c in purpose_stats["count"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"], size=10),
    ))
    fig.update_layout(title="Top Loan Purposes")
    fig = plotly_layout(fig, height=350)
    fig.update_layout(xaxis_title="Number of Loans", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ──
st.html(
    f'<div style="text-align:center;padding:2rem;color:{COLORS["text_muted"]};font-size:0.8rem;">'
    'Credit Risk Platform | Built with Streamlit, XGBoost, and SHAP | '
    'Data: Lending Club 2007-2018</div>'
)
