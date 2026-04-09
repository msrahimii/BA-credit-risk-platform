"""Data Processing Steps - Pipeline walkthrough with visualizations."""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from utils import COLORS, section_header, info_card, plotly_layout, hex_to_rgba

st.html(section_header("Data Processing Pipeline",
                        "Step-by-step walkthrough of how we prepared 2.26M loans for modeling"))

# ── Load data for visualizations ──
status_dist = pd.read_csv("data/viz/status_distribution.csv")
corr_matrix = pd.read_csv("data/viz/correlation_matrix.csv", index_col=0)
with open("data/viz/kpis.json") as f:
    kpis = json.load(f)
with open("artifacts/model_metadata.json") as f:
    metadata = json.load(f)

# ── Pipeline overview ──
steps = [
    ("1", "Load Raw Data", "2.26M rows, 151 columns"),
    ("2", "Filter Loan Status", "1.35M rows with definitive outcomes"),
    ("3", "Drop Leakage Columns", "Removed 66 post-origination columns"),
    ("4", "Parse & Clean", "Converted types, engineered credit history"),
    ("5", "Handle Missing Values", "Imputed months-since with -1, median fill"),
    ("6", "Encode Categoricals", "Ordinal, target, and label encoding"),
    ("7", "Feature Selection", "Removed >0.95 correlated pairs"),
    ("8", "Time-Based Split", "Train <2017, Test >=2017"),
]

# Pipeline flow visualization
step_html = '<div style="display:flex;gap:8px;overflow-x:auto;padding:12px 0;">'
for num, title, desc in steps:
    step_html += (
        f'<div style="flex:0 0 auto;min-width:140px;background:linear-gradient(135deg,'
        f'rgba(26,31,78,0.8),rgba(30,36,86,0.6));backdrop-filter:blur(16px);'
        f'border:1px solid rgba(129,140,248,0.15);border-radius:16px;padding:16px;'
        f'text-align:center;position:relative;">'
        f'<div style="background:linear-gradient(135deg,#7c3aed,#6366f1);'
        f'width:32px;height:32px;border-radius:50%;display:flex;align-items:center;'
        f'justify-content:center;margin:0 auto 8px;font-weight:800;font-size:0.85rem;'
        f'color:white;box-shadow:0 0 15px rgba(124,58,237,0.3);">{num}</div>'
        f'<div style="color:{COLORS["text"]};font-weight:700;font-size:0.8rem;'
        f'margin-bottom:4px;">{title}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;line-height:1.3;">'
        f'{desc}</div></div>'
    )
step_html += '</div>'
st.html(step_html)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# STEP 1 & 2: Raw Data & Target Definition
# ════════════════════════════════════════════
st.markdown("### Step 1-2: Raw Data & Target Definition")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Raw Dataset",
        "The Lending Club dataset contains <strong>2.26 million</strong> loan records from "
        "2007 to 2018 Q4 with <strong>151 columns</strong> covering loan details, "
        "borrower profile, credit history, and payment performance. We filtered to loans "
        "with definitive outcomes: <strong>Fully Paid</strong>, <strong>Charged Off</strong>, "
        "or <strong>Default</strong>, yielding <strong>1,345,350 loans</strong>.",
        accent_color=COLORS["chart_blue"],
    ))

    # Funnel chart: raw -> filtered
    fig = go.Figure(go.Funnel(
        y=["Raw Dataset", "Valid Status Only", "After Column Drops", "Final Features"],
        x=[2260668, 1345350, 1345350, 1345350],
        textinfo="value+percent initial",
        textfont=dict(size=13),
        marker=dict(
            color=[COLORS["chart_purple"], COLORS["chart_blue"],
                   COLORS["chart_green"], COLORS["success"]],
            line=dict(width=0),
        ),
        connector=dict(line=dict(color=COLORS["border"], width=1)),
    ))
    fig.update_layout(title="Data Filtering Funnel")
    fig = plotly_layout(fig, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.html(info_card(
        "Target Variable",
        "We created a binary target: <strong>0 = Non-Default</strong> (Fully Paid) and "
        "<strong>1 = Default</strong> (Charged Off + Default). The dataset is imbalanced "
        "with a <strong>~20% default rate</strong> — we handle this via "
        "<code>scale_pos_weight</code> in XGBoost rather than resampling.",
        accent_color=COLORS["chart_red"],
    ))

    # Loan status distribution
    colors_map = {
        "Fully Paid": COLORS["success"],
        "Charged Off": COLORS["danger"],
        "Default": COLORS["warning"],
    }
    fig = go.Figure(go.Pie(
        labels=status_dist["status"],
        values=status_dist["count"],
        hole=0.55,
        marker=dict(
            colors=[colors_map.get(s, COLORS["accent"]) for s in status_dist["status"]],
            line=dict(color=COLORS["bg_dark"], width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=12, color=COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(title="Loan Status Distribution")
    fig = plotly_layout(fig, height=350)
    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            text=f"<b>{kpis['total_loans']:,.0f}</b><br>Loans",
            x=0.5, y=0.5, font=dict(size=16, color=COLORS["text"]),
            showarrow=False,
        )],
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# STEP 3: Data Leakage Prevention
# ════════════════════════════════════════════
st.markdown("### Step 3: Data Leakage Prevention")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.html(info_card(
        "Why This Matters",
        "Data leakage means using information that wouldn't be available at prediction time. "
        "For a loan default model, we can only use data known <strong>at the time the loan was issued</strong>. "
        "Any payment performance, collection, or post-origination data would artificially inflate accuracy "
        "but make the model <strong>useless in production</strong>.",
        accent_color=COLORS["warning"],
    ))

    # Categories of dropped columns
    leakage_categories = {
        "Payment Performance": 9,
        "Payment Dates": 3,
        "Post-Origination Credit": 3,
        "Hardship Program": 15,
        "Settlement Data": 7,
        "Other Post-Origination": 4,
        "ID / Useless Columns": 9,
        "Joint Application": 16,
    }
    cats = list(leakage_categories.keys())
    vals = list(leakage_categories.values())

    sorted_pairs = sorted(zip(cats, vals), key=lambda x: x[1])
    cats = [p[0] for p in sorted_pairs]
    vals = [p[1] for p in sorted_pairs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=vals, orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, COLORS["chart_blue"]], [1, COLORS["danger"]]],
            line=dict(width=0),
        ),
        text=[str(v) for v in vals],
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"], size=11),
    ))
    fig.update_layout(title="Columns Dropped by Category (66 Total)")
    fig = plotly_layout(fig, height=380)
    fig.update_layout(xaxis_title="Number of Columns")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.html(info_card(
        "Column Reduction",
        f"Started with <strong>151 columns</strong> and dropped <strong>66</strong> "
        f"that would cause data leakage or provide no predictive value. "
        f"The final model uses <strong>{len(metadata['feature_columns'])} features</strong>.",
        accent_color=COLORS["accent_light"],
    ))

    # Before/after column count
    col_stages = ["Raw Dataset", "After Column\nDrops (66)", "After Missing\nFilter (>70%)", "Final Features\n(Post Corr Filter)"]
    col_counts = [151, 85, 81, 78]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=col_stages, y=col_counts,
        marker=dict(
            color=[COLORS["chart_red"], COLORS["chart_amber"],
                   COLORS["chart_blue"], COLORS["success"]],
            line=dict(width=0),
        ),
        text=[str(c) for c in col_counts],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=14, family="Inter"),
    ))
    fig.update_layout(title="Column Count at Each Stage")
    fig = plotly_layout(fig, height=380)
    fig.update_layout(yaxis_title="Number of Columns")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# STEP 4-5: Parsing, Cleaning & Missing Values
# ════════════════════════════════════════════
st.markdown("### Steps 4-5: Data Cleaning & Missing Value Strategy")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Feature Engineering",
        "<strong>term:</strong> ' 36 months' &rarr; 36 (numeric)<br>"
        "<strong>emp_length:</strong> '10+ years' &rarr; 10 (ordinal map)<br>"
        "<strong>earliest_cr_line:</strong> Converted to <strong>credit_history_months</strong> "
        "(months since first credit line, using Jan 2019 as reference)<br>"
        "<strong>revol_util / int_rate:</strong> Forced to numeric, handling parse errors",
        accent_color=COLORS["chart_green"],
    ))

with col2:
    st.html(info_card(
        "Missing Value Strategy",
        "<strong>Months-since columns</strong> (e.g., mths_since_last_delinq): "
        "Missing means 'never happened' — imputed with <strong>-1</strong> before "
        "checking missing percentages. This preserved 12 valuable signal columns.<br><br>"
        "<strong>Remaining numerics:</strong> Median imputation<br>"
        "<strong>Categoricals:</strong> Filled with 'Unknown'<br>"
        "<strong>70% threshold:</strong> Columns with >70% missing were dropped",
        accent_color=COLORS["chart_amber"],
    ))

# Missing value handling visualization
missing_strategy = {
    "Months-Since Imputation (-1)": 12,
    "Median Imputation (Numeric)": 57,
    "Unknown Fill (Categorical)": 8,
    "Dropped (>70% Missing)": 4,
}
labels = list(missing_strategy.keys())
values = list(missing_strategy.values())

fig = go.Figure(go.Pie(
    labels=labels, values=values, hole=0.5,
    marker=dict(
        colors=[COLORS["success"], COLORS["chart_blue"],
                COLORS["chart_purple"], COLORS["danger"]],
        line=dict(color=COLORS["bg_dark"], width=3),
    ),
    textinfo="label+value",
    textfont=dict(size=11, color=COLORS["text"]),
    textposition="outside",
))
fig.update_layout(title="Missing Value Handling Strategy (by Column Count)")
fig = plotly_layout(fig, height=380)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# STEP 6: Encoding
# ════════════════════════════════════════════
st.markdown("### Step 6: Categorical Encoding")

col1, col2, col3 = st.columns(3)

with col1:
    st.html(info_card(
        "Ordinal Encoding",
        "<strong>grade:</strong> A&rarr;1, B&rarr;2, ... G&rarr;7<br>"
        "<strong>sub_grade:</strong> A1&rarr;1, A2&rarr;2, ... G5&rarr;35<br><br>"
        "Preserves the natural risk ordering in Lending Club's grading system.",
        accent_color=COLORS["chart_purple"],
    ))

with col2:
    st.html(info_card(
        "Target Encoding",
        "<strong>addr_state:</strong> Each state is replaced by its mean default rate "
        "computed <strong>on the training set only</strong> to prevent leakage.<br><br>"
        "States with no data fall back to the global default rate.",
        accent_color=COLORS["chart_blue"],
    ))

with col3:
    st.html(info_card(
        "Label Encoding",
        "<strong>home_ownership, purpose, verification_status, application_type, "
        "initial_list_status</strong><br><br>"
        "Sorted unique values mapped to integers. Works well with tree-based models.",
        accent_color=COLORS["chart_green"],
    ))

# Sub-grade encoding visualization
grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
grade_nums = list(range(1, 36))
grade_colors = []
for g in "ABCDEFG":
    for _ in range(5):
        if g in "AB":
            grade_colors.append(COLORS["success"])
        elif g in "CD":
            grade_colors.append(COLORS["chart_amber"])
        else:
            grade_colors.append(COLORS["danger"])

fig = go.Figure()
fig.add_trace(go.Bar(
    x=grades, y=grade_nums,
    marker=dict(color=grade_colors, opacity=0.8, line=dict(width=0)),
))
fig.update_layout(title="Sub-Grade Ordinal Encoding (A1=1 to G5=35)")
fig = plotly_layout(fig, height=300)
fig.update_layout(
    xaxis_title="Sub Grade", yaxis_title="Encoded Value",
    xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# STEP 7: Feature Selection
# ════════════════════════════════════════════
st.markdown("### Step 7: Feature Selection — Correlation Filter")

col1, col2 = st.columns([1.5, 1])

with col1:
    # Correlation heatmap
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=[c.replace("_", " ").title() for c in corr_matrix.columns],
        y=[c.replace("_", " ").title() for c in corr_matrix.index],
        colorscale=[[0, COLORS["chart_blue"]], [0.5, COLORS["bg_dark"]],
                    [1, COLORS["chart_red"]]],
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(tickfont=dict(color=COLORS["text_muted"])),
    ))
    fig.update_layout(title="Feature Correlation Matrix (Key Features)")
    fig = plotly_layout(fig, height=450)
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.html(info_card(
        "Correlation Filter (>0.95)",
        "Highly correlated features add redundancy without improving model performance. "
        "We removed feature pairs with <strong>Pearson correlation > 0.95</strong>.<br><br>"
        "<strong>Protected features</strong> (never dropped):<br>"
        "sub_grade, int_rate, fico_range_low, loan_amnt, dti<br><br>"
        "These are critical business features that must remain in the model regardless "
        "of correlation with other variables.",
        accent_color=COLORS["accent_light"],
    ))

    st.html(info_card(
        "Strongest Correlations with Default",
        "<strong>int_rate &rarr; target:</strong> +0.26 (strongest positive)<br>"
        "<strong>fico_range_low &rarr; target:</strong> -0.13 (strongest negative)<br>"
        "<strong>dti &rarr; target:</strong> +0.08<br>"
        "<strong>loan_amnt &rarr; target:</strong> +0.07<br><br>"
        "Interest rate is the single strongest predictor of default, confirming that "
        "Lending Club's pricing already captures significant risk signal.",
        accent_color=COLORS["chart_red"],
    ))

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# STEP 8: Time-Based Split
# ════════════════════════════════════════════
st.markdown("### Step 8: Time-Based Train/Test Split")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Why Time-Based Split?",
        "Random train/test splits would let the model see <strong>future data during training</strong>. "
        "In reality, we'd build the model on historical loans and deploy it on future applications.<br><br>"
        "<strong>Train:</strong> Loans issued before Jan 2017 (<strong>1,119,711 loans</strong>)<br>"
        "<strong>Test:</strong> Loans issued Jan 2017 onwards (<strong>225,639 loans</strong>)<br><br>"
        "This simulates real-world deployment and gives an honest performance estimate.",
        accent_color=COLORS["success"],
    ))

with col2:
    # Train/test split visualization
    split_data = {
        "Set": ["Training Set\n(< 2017)", "Test Set\n(>= 2017)"],
        "Loans": [1119711, 225639],
        "Default Rate": [19.6, 21.3],
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=split_data["Set"],
        y=split_data["Loans"],
        marker=dict(
            color=[COLORS["chart_purple"], COLORS["chart_blue"]],
            line=dict(width=0),
        ),
        text=[f"{v:,.0f} loans<br>{r}% default" for v, r in
              zip(split_data["Loans"], split_data["Default Rate"])],
        textposition="inside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(title="Train / Test Split")
    fig = plotly_layout(fig, height=350)
    fig.update_layout(yaxis_title="Number of Loans")
    st.plotly_chart(fig, use_container_width=True)

# Timeline visualization
time_stats = pd.read_csv("data/viz/time_stats.csv")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_stats["year_month"],
    y=time_stats["count"],
    mode="lines",
    line=dict(color=COLORS["chart_purple"], width=2),
    fill="tozeroy",
    fillcolor=hex_to_rgba(COLORS["chart_purple"], 0.1),
    name="Training Data",
))

# Find the split point index
split_idx = None
for i, ym in enumerate(time_stats["year_month"]):
    if "2017" in str(ym):
        split_idx = i
        break

if split_idx:
    fig.add_trace(go.Scatter(
        x=time_stats["year_month"].iloc[split_idx:],
        y=time_stats["count"].iloc[split_idx:],
        mode="lines",
        line=dict(color=COLORS["chart_blue"], width=2),
        fill="tozeroy",
        fillcolor=hex_to_rgba(COLORS["chart_blue"], 0.15),
        name="Test Data",
    ))
    split_label = time_stats["year_month"].iloc[split_idx]
    fig.add_shape(
        type="line", x0=split_label, x1=split_label, y0=0, y1=1,
        yref="paper",
        line=dict(color=COLORS["warning"], width=2, dash="dash"),
    )
    fig.add_annotation(
        x=split_label, y=1.06, yref="paper",
        text="Split: Jan 2017",
        font=dict(color=COLORS["warning"], size=12),
        showarrow=False,
    )

fig.update_layout(title="Loan Volume Over Time with Train/Test Split")
fig = plotly_layout(fig, height=350)
fig.update_layout(
    xaxis_title="Quarter", yaxis_title="Number of Loans",
    xaxis=dict(tickangle=-45, dtick=4),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════
st.markdown("### Pipeline Summary")

summary_cols = st.columns(4)
summary_items = [
    ("Raw Rows", "2,260,668", COLORS["text_muted"]),
    ("Final Rows", "1,345,350", COLORS["chart_blue"]),
    ("Final Features", str(len(metadata["feature_columns"])), COLORS["chart_purple"]),
    ("Test AUC", f"{metadata['auc_final']:.4f}", COLORS["success"]),
]
for col, (label, value, color) in zip(summary_cols, summary_items):
    col.html(
        f'<div style="background:linear-gradient(135deg,rgba(26,31,78,0.8),rgba(30,36,86,0.6));'
        f'backdrop-filter:blur(16px);border:1px solid rgba(129,140,248,0.12);border-radius:16px;'
        f'padding:20px;text-align:center;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.72rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;font-weight:600;">{label}</div>'
        f'<div style="color:{color};font-size:1.8rem;font-weight:800;margin-top:8px;'
        f'letter-spacing:-0.03em;">{value}</div></div>'
    )

st.markdown("<br>", unsafe_allow_html=True)

st.html(info_card(
    "Key Design Decisions",
    "<strong>1. Time-based split over random split</strong> — prevents temporal data leakage and "
    "simulates real deployment conditions.<br>"
    "<strong>2. Impute months-since with -1 instead of dropping</strong> — 'never delinquent' is "
    "valuable signal, not missing data.<br>"
    "<strong>3. Target encoding for states on train set only</strong> — prevents information from "
    "the test set leaking into features.<br>"
    "<strong>4. Protected features in correlation filter</strong> — business-critical features like "
    "FICO, interest rate, and DTI are never dropped regardless of correlation.<br>"
    "<strong>5. scale_pos_weight over resampling</strong> — handles class imbalance natively in "
    "XGBoost without distorting the data distribution.",
    accent_color=COLORS["accent_light"],
))
