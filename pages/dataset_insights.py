"""Dataset Insights Page - EDA visualizations."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import COLORS, plotly_layout, section_header, info_card

st.html(section_header("Dataset Insights",
                        "Exploratory analysis of Lending Club loan data (2007-2018)"))

# ── Load pre-computed data ──
grade_stats = pd.read_csv("data/viz/grade_stats.csv")
purpose_stats = pd.read_csv("data/viz/purpose_stats.csv")
home_stats = pd.read_csv("data/viz/home_ownership_stats.csv")
state_stats = pd.read_csv("data/viz/state_stats.csv")
time_stats = pd.read_csv("data/viz/time_stats.csv")
term_stats = pd.read_csv("data/viz/term_stats.csv")
numeric_dist = pd.read_csv("data/viz/numeric_distributions.csv")
corr_matrix = pd.read_csv("data/viz/correlation_matrix.csv", index_col=0)

# ── Tab Layout ──
tab1, tab2, tab3, tab4 = st.tabs([
    "Distributions", "Default Analysis", "Geographic", "Correlations"
])

with tab1:
    st.markdown("### Feature Distributions")

    col1, col2 = st.columns(2)
    with col1:
        # Loan amount distribution
        fig = go.Figure()
        for target, name, color in [(0, "Non-Default", COLORS["chart_blue"]),
                                     (1, "Default", COLORS["chart_red"])]:
            subset = numeric_dist[numeric_dist["target"] == target]
            fig.add_trace(go.Histogram(
                x=subset["loan_amnt"], name=name, marker_color=color,
                opacity=0.7, nbinsx=40,
            ))
        fig.update_layout(title="Loan Amount Distribution", barmode="overlay")
        fig = plotly_layout(fig, height=380)
        fig.update_layout(xaxis_title="Loan Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Interest rate distribution
        fig = go.Figure()
        for target, name, color in [(0, "Non-Default", COLORS["chart_blue"]),
                                     (1, "Default", COLORS["chart_red"])]:
            subset = numeric_dist[numeric_dist["target"] == target]
            fig.add_trace(go.Histogram(
                x=subset["int_rate"], name=name, marker_color=color,
                opacity=0.7, nbinsx=40,
            ))
        fig.update_layout(title="Interest Rate Distribution", barmode="overlay")
        fig = plotly_layout(fig, height=380)
        fig.update_layout(xaxis_title="Interest Rate (%)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Annual income
        fig = go.Figure()
        for target, name, color in [(0, "Non-Default", COLORS["chart_blue"]),
                                     (1, "Default", COLORS["chart_red"])]:
            subset = numeric_dist[numeric_dist["target"] == target]
            fig.add_trace(go.Histogram(
                x=subset["annual_inc"], name=name, marker_color=color,
                opacity=0.7, nbinsx=50,
            ))
        fig.update_layout(title="Annual Income Distribution", barmode="overlay")
        fig = plotly_layout(fig, height=380)
        fig.update_layout(xaxis_title="Annual Income ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # FICO score
        fig = go.Figure()
        for target, name, color in [(0, "Non-Default", COLORS["chart_blue"]),
                                     (1, "Default", COLORS["chart_red"])]:
            subset = numeric_dist[numeric_dist["target"] == target]
            fig.add_trace(go.Histogram(
                x=subset["fico_range_low"], name=name, marker_color=color,
                opacity=0.7, nbinsx=30,
            ))
        fig.update_layout(title="FICO Score Distribution", barmode="overlay")
        fig = plotly_layout(fig, height=380)
        fig.update_layout(xaxis_title="FICO Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Default Rate Analysis")

    col1, col2 = st.columns(2)
    with col1:
        # By grade
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grade_stats["grade"],
            y=grade_stats["count"],
            name="Loan Count",
            marker_color=COLORS["chart_purple"],
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=grade_stats["grade"],
            y=grade_stats["default_rate"] * 100,
            name="Default Rate %",
            mode="lines+markers",
            marker=dict(color=COLORS["chart_red"], size=10),
            line=dict(color=COLORS["chart_red"], width=2.5),
            yaxis="y2",
        ))
        fig.update_layout(
            title="Grade: Volume vs Default Rate",
            yaxis=dict(title="Loan Count", side="left"),
            yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y",
                        gridcolor="rgba(0,0,0,0)"),
        )
        fig = plotly_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By term
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=term_stats["term"].astype(str) + " months",
            y=term_stats["default_rate"] * 100,
            marker=dict(color=[COLORS["chart_blue"], COLORS["chart_red"]]),
            text=[f"{r:.1f}%" for r in term_stats["default_rate"] * 100],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig.update_layout(title="Default Rate by Loan Term")
        fig = plotly_layout(fig, height=400)
        fig.update_layout(yaxis_title="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # By purpose
        top_purpose = purpose_stats.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_purpose["purpose"].apply(lambda x: x.replace("_", " ").title()),
            x=top_purpose["default_rate"] * 100,
            orientation="h",
            marker=dict(
                color=top_purpose["default_rate"] * 100,
                colorscale=[[0, COLORS["chart_green"]], [1, COLORS["chart_red"]]],
            ),
            text=[f"{r:.1f}%" for r in top_purpose["default_rate"] * 100],
            textposition="outside",
            textfont=dict(color=COLORS["text_muted"], size=10),
        ))
        fig.update_layout(title="Default Rate by Purpose")
        fig = plotly_layout(fig, height=400)
        fig.update_layout(xaxis_title="Default Rate (%)",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By home ownership
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=home_stats["home_ownership"],
            y=home_stats["default_rate"] * 100,
            marker=dict(
                color=home_stats["default_rate"] * 100,
                colorscale=[[0, COLORS["chart_green"]], [1, COLORS["chart_red"]]],
            ),
            text=[f"{r:.1f}%" for r in home_stats["default_rate"] * 100],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig.update_layout(title="Default Rate by Home Ownership")
        fig = plotly_layout(fig, height=400)
        fig.update_layout(yaxis_title="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Geographic Analysis")

    metric = st.radio("Map Metric", ["Default Rate", "Average Loan Amount", "Loan Count"],
                      horizontal=True)

    col_map = {"Default Rate": "default_rate",
               "Average Loan Amount": "avg_loan_amnt",
               "Loan Count": "count"}[metric]
    color_label = metric

    fig = go.Figure(go.Choropleth(
        locations=state_stats["addr_state"],
        z=state_stats[col_map] * (100 if col_map == "default_rate" else 1),
        locationmode="USA-states",
        colorscale="RdYlGn_r" if col_map == "default_rate" else "Purples",
        colorbar=dict(
            title=dict(text=color_label, font=dict(color=COLORS["text_muted"])),
            tickfont=dict(color=COLORS["text_muted"]),
        ),
        marker_line_color=COLORS["border"],
        marker_line_width=0.5,
    ))
    fig.update_layout(
        title=f"{metric} by State",
        geo=dict(
            scope="usa",
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
            landcolor=COLORS["bg_card"],
            showlakes=False,
        ),
    )
    fig = plotly_layout(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Top/bottom states table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Highest Default Rate States**")
        top_default = state_stats.nlargest(10, "default_rate")[
            ["addr_state", "default_rate", "count"]].copy()
        top_default["default_rate"] = (top_default["default_rate"] * 100).round(1).astype(str) + "%"
        top_default.columns = ["State", "Default Rate", "Loans"]
        st.dataframe(top_default, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Lowest Default Rate States**")
        bot_default = state_stats.nsmallest(10, "default_rate")[
            ["addr_state", "default_rate", "count"]].copy()
        bot_default["default_rate"] = (bot_default["default_rate"] * 100).round(1).astype(str) + "%"
        bot_default.columns = ["State", "Default Rate", "Loans"]
        st.dataframe(bot_default, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Feature Correlations")

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=[c.replace("_", " ").title() for c in corr_matrix.columns],
        y=[c.replace("_", " ").title() for c in corr_matrix.index],
        colorscale=[[0, COLORS["chart_blue"]], [0.5, COLORS["bg_dark"]],
                    [1, COLORS["chart_red"]]],
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(tickfont=dict(color=COLORS["text_muted"])),
    ))
    fig.update_layout(title="Correlation Matrix (Key Features)")
    fig = plotly_layout(fig, height=550)
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)

    st.html(info_card("Key Insight",
                      "FICO score and interest rate show the strongest negative correlation (-0.70+), "
                      "confirming that lower credit scores lead to higher rates. The target variable "
                      "(default) correlates most with interest rate, confirming it as the strongest "
                      "default predictor.",
                      accent_color=COLORS["accent_light"]))
