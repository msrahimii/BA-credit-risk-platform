"""Limitations Page - Honest assessment of project constraints and tradeoffs."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from utils import COLORS, section_header, info_card, plotly_layout, hex_to_rgba

st.html(section_header("Limitations & Considerations",
                        "An honest assessment of what this model can and cannot do"))

# ── Load metadata ──
with open("artifacts/model_metadata.json") as f:
    metadata = json.load(f)

# ════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════
st.html(info_card(
    "Why This Page Exists",
    "No model is perfect. Understanding a model's limitations is just as important as "
    "understanding its capabilities. This page documents the real constraints, tradeoffs, "
    "and approximations made during development — so that anyone using this platform knows "
    "exactly what to trust and where to be cautious.",
    accent_color=COLORS["accent_light"],
))

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# DATA LIMITATIONS
# ════════════════════════════════════════════
st.markdown("### Data Limitations")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Dataset Age (2007-2018)",
        "The Lending Club dataset ends in Q4 2018. Since then:<br><br>"
        "- <strong>COVID-19</strong> fundamentally changed borrower behavior and default patterns<br>"
        "- <strong>Interest rate environment</strong> shifted dramatically (near-zero to multi-decade highs)<br>"
        "- <strong>Lending practices</strong> evolved with new regulations and fintech competition<br>"
        "- <strong>Lending Club itself</strong> transitioned from peer-to-peer to a bank model in 2020<br><br>"
        "A model trained on pre-2019 data would need retraining on current data before any real deployment.",
        accent_color=COLORS["danger"],
    ))

    st.html(info_card(
        "Class Imbalance (~80/20)",
        "Only <strong>~20% of loans defaulted</strong>. While we handle this with "
        "<code>scale_pos_weight=4.08</code>, the model still has <strong>4x more non-default "
        "examples</strong> to learn from. This means:<br><br>"
        "- The model is better at recognizing non-defaults than defaults<br>"
        "- Rare default patterns (e.g., defaults from A-grade borrowers) may be underlearned<br>"
        "- Precision and recall for the default class are inherently harder to optimize",
        accent_color=COLORS["warning"],
    ))

with col2:
    st.html(info_card(
        "66 Columns Dropped for Leakage Prevention",
        "We dropped 41 leakage columns, 9 useless/ID columns, and 16 joint-application columns. "
        "While necessary, this removes a large amount of data:<br><br>"
        "- <strong>Payment history</strong> (most predictive of future behavior) is entirely unavailable "
        "at origination time<br>"
        "- <strong>Hardship and settlement data</strong> (15+7 columns) could indicate systemic risk "
        "patterns but are post-origination<br>"
        "- <strong>Joint application features</strong> (16 columns) were too sparse to use, meaning "
        "the model treats joint applicants the same as individual ones",
        accent_color=COLORS["chart_red"],
    ))

    st.html(info_card(
        "Missing Data Impact",
        "Several columns had significant missingness:<br><br>"
        "- <strong>4 columns dropped</strong> with >70% missing values<br>"
        "- <strong>12 months-since columns</strong> imputed with -1 (a reasonable choice for "
        "'never happened,' but -1 is arbitrary and the model must learn this convention)<br>"
        "- <strong>Remaining numerics</strong> filled with median values, which assumes missing "
        "data is missing at random — this may not always hold (e.g., borrowers with no "
        "employment length may differ systematically from those who report it)",
        accent_color=COLORS["chart_amber"],
    ))

# Data age timeline
years = list(range(2007, 2027))
in_dataset = [1 if y <= 2018 else 0 for y in years]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=[str(y) for y in years],
    y=[1] * len(years),
    marker=dict(
        color=[COLORS["chart_purple"] if y <= 2018 else hex_to_rgba(COLORS["danger"], 0.3)
               for y in years],
        line=dict(width=0),
    ),
    text=["In Dataset" if y <= 2018 else "No Data" for y in years],
    textposition="inside",
    textfont=dict(color=COLORS["text"], size=9),
    hovertemplate="Year: %{x}<extra></extra>",
))

fig.add_shape(type="line", x0="2019", x1="2019", y0=0, y1=1.1,
              line=dict(color=COLORS["danger"], width=2, dash="dash"))
fig.add_annotation(x="2020", y=1.05,
                   text="COVID-19 Pandemic",
                   font=dict(color=COLORS["danger"], size=10), showarrow=False)
fig.add_annotation(x="2019", y=1.12,
                   text="Dataset ends here",
                   font=dict(color=COLORS["warning"], size=11, weight=700), showarrow=False)

fig.update_layout(title="Dataset Coverage vs. Time")
fig = plotly_layout(fig, height=280)
fig.update_layout(
    yaxis=dict(visible=False),
    xaxis=dict(tickangle=-45),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# MODEL PERFORMANCE LIMITATIONS
# ════════════════════════════════════════════
st.markdown("### Model Performance Limitations")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        f"AUC of {metadata['auc_final']:.4f} — Good but Not Exceptional",
        "An AUC of 0.73 means the model correctly ranks a random default above a random "
        "non-default <strong>73% of the time</strong>. For context:<br><br>"
        "- <strong>0.5</strong> = random guessing<br>"
        "- <strong>0.7-0.8</strong> = acceptable discrimination<br>"
        "- <strong>0.8-0.9</strong> = excellent (typical for models with bureau data)<br>"
        "- <strong>0.9+</strong> = outstanding (rare in credit risk)<br><br>"
        "Our model sits in the 'acceptable' range. Real-world credit models with full bureau access, "
        "income verification, and behavioral data typically achieve AUC 0.80+.",
        accent_color=COLORS["chart_blue"],
    ))

    # AUC context chart
    auc_benchmarks = {
        "Random\nGuessing": 0.50,
        "Basic\nScorecard": 0.65,
        "This Model": metadata["auc_final"],
        "Full Bureau\nModel": 0.82,
        "Perfect\nModel": 1.00,
    }
    names = list(auc_benchmarks.keys())
    values = list(auc_benchmarks.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=values,
        marker=dict(
            color=[COLORS["text_muted"], COLORS["chart_amber"], COLORS["chart_purple"],
                   COLORS["chart_blue"], COLORS["success"]],
            line=dict(width=0),
        ),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=13),
    ))
    fig.update_layout(title="AUC-ROC in Context")
    fig = plotly_layout(fig, height=320)
    fig.update_layout(yaxis_title="AUC-ROC", yaxis=dict(range=[0, 1.1]))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    improvement = (metadata["auc_final"] - metadata["auc_baseline"]) / metadata["auc_baseline"] * 100
    st.html(info_card(
        f"Marginal Tuning Improvement (+{improvement:.2f}%)",
        f"Optuna tuning improved AUC from <strong>{metadata['auc_baseline']:.4f}</strong> to "
        f"<strong>{metadata['auc_final']:.4f}</strong> — a gain of just "
        f"<strong>{metadata['auc_final'] - metadata['auc_baseline']:.4f}</strong>.<br><br>"
        "This suggests we are near the <strong>performance ceiling</strong> for this feature set. "
        "Further gains would likely require:<br><br>"
        "- Additional data sources (bureau data, bank statements)<br>"
        "- More sophisticated feature engineering<br>"
        "- Ensemble methods combining multiple model types<br>"
        "- A larger hyperparameter search (we only ran 10 trials on 25% of data due to time constraints)",
        accent_color=COLORS["chart_purple"],
    ))

    st.html(info_card(
        "Precision-Recall Tradeoff",
        "Our dual-threshold approach means neither use case gets the ideal balance:<br><br>"
        f"<strong>Borrower threshold ({metadata['borrower_threshold']:.2f}):</strong> "
        f"Precision={metadata['borrower_metrics']['precision']:.1%}, "
        f"Recall={metadata['borrower_metrics']['recall']:.1%}<br>"
        "Only catches 30% of actual defaults — 70% of risky borrowers are told they're fine.<br><br>"
        f"<strong>Bank threshold ({metadata['bank_threshold']:.2f}):</strong> "
        f"Precision={metadata['bank_metrics']['precision']:.1%}, "
        f"Recall={metadata['bank_metrics']['recall']:.1%}<br>"
        "Catches 83% of defaults, but 70% of flagged borrowers would actually repay.<br><br>"
        "There is no threshold that gives both high precision and high recall — "
        "this is a fundamental limitation of the model's discriminative power.",
        accent_color=COLORS["warning"],
    ))

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# FEATURE LIMITATIONS
# ════════════════════════════════════════════
st.markdown("### Feature & Input Limitations")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Circular Features (sub_grade, int_rate)",
        "Our two strongest predictors — <strong>sub_grade</strong> (SHAP: 0.244) and "
        "<strong>int_rate</strong> (SHAP: 0.125) — are assigned <strong>by Lending Club</strong> "
        "using their own proprietary risk model.<br><br>"
        "This creates circularity: we are partly predicting default using features that were "
        "themselves derived from a default risk assessment. In a greenfield deployment (without "
        "a platform like Lending Club pre-scoring borrowers), these features would not exist.<br><br>"
        "Without sub_grade and int_rate, model performance would likely drop significantly.",
        accent_color=COLORS["danger"],
    ))

    st.html(info_card(
        "No Alternative Data",
        "Modern credit risk models increasingly use:<br><br>"
        "- <strong>Bank transaction data</strong> (spending patterns, cash flow)<br>"
        "- <strong>Employment verification</strong> (real-time income confirmation)<br>"
        "- <strong>Behavioral signals</strong> (app usage, application timing)<br>"
        "- <strong>Social/economic indicators</strong> (regional unemployment, housing prices)<br><br>"
        "Our model relies solely on the Lending Club application form and credit bureau snapshot "
        "at origination time. Adding alternative data sources could meaningfully improve performance.",
        accent_color=COLORS["chart_amber"],
    ))

with col2:
    st.html(info_card(
        "FICO Is a Single Snapshot",
        "We use <strong>fico_range_low</strong> — the lower bound of the borrower's FICO score "
        "range at the time of application. Limitations:<br><br>"
        "- It's a <strong>point-in-time snapshot</strong>, not a trend (is the score improving or declining?)<br>"
        "- It's a <strong>range</strong>, not an exact score (we only see the lower bound)<br>"
        "- Real credit decisions use the full bureau report with trade lines, balances, and payment history — "
        "the FICO score is just a summary<br>"
        "- Borrowers can have different scores from different bureaus",
        accent_color=COLORS["chart_blue"],
    ))

    st.html(info_card(
        "Single Time Split (No Walk-Forward CV)",
        "We used a single temporal split: train on loans before Jan 2017, test on 2017-2018. "
        "A more robust approach would be <strong>walk-forward cross-validation</strong>:<br><br>"
        "- Train on 2007-2013, test on 2014<br>"
        "- Train on 2007-2014, test on 2015<br>"
        "- Train on 2007-2015, test on 2016<br>"
        "- ... and so on<br><br>"
        "This would give multiple performance estimates across different time periods, "
        "revealing how stable the model is over time. Our single split may give an optimistic "
        "or pessimistic estimate depending on the specific economic conditions of 2017-2018.",
        accent_color=COLORS["chart_green"],
    ))

# Top features with circularity highlighted
top_features = metadata.get("top_features", {})
if top_features:
    features = list(top_features.keys())[:15]
    values = [top_features[f] for f in features]
    sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
    features = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    circular_features = {"sub_grade", "int_rate"}
    bar_colors = [COLORS["danger"] if f in circular_features else COLORS["chart_purple"]
                  for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f.replace("_", " ").title()[:25] for f in features],
        x=values, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
    ))
    fig.add_annotation(
        x=max(values) * 0.7, y=14, text="Red = assigned by Lending Club (circular)",
        font=dict(color=COLORS["danger"], size=11), showarrow=False,
    )
    fig.update_layout(title="Top 15 Features — Circular Features Highlighted")
    fig = plotly_layout(fig, height=450)
    fig.update_layout(xaxis_title="Mean |SHAP| Value", yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# BORROWER PAGE APPROXIMATIONS
# ════════════════════════════════════════════
st.markdown("### Borrower Page Approximations")

st.html(info_card(
    "Why Approximations Are Needed",
    "The model uses <strong>78 features</strong>, but the Borrower Assessment page only "
    "collects <strong>9 inputs</strong> from the user. The remaining 69 features must be "
    "filled with reasonable defaults. This is a deliberate UX tradeoff — asking for 78 "
    "fields would make the tool unusable.",
    accent_color=COLORS["accent_light"],
))

col1, col2, col3 = st.columns(3)

with col1:
    st.html(info_card(
        "Median Fill for Unknown Features",
        "69 features that the borrower doesn't provide are filled with <strong>training set "
        "medians</strong>. This assumes the borrower is 'average' on every dimension we don't "
        "ask about.<br><br>"
        "In reality, a borrower with a low FICO score likely also has worse-than-median values "
        "for related features (higher delinquencies, lower account age, etc.). The median fill "
        "may make risky borrowers appear safer than they actually are.",
        accent_color=COLORS["chart_red"],
    ))

with col2:
    st.html(info_card(
        "FICO-to-Sub-Grade Mapping",
        "Lending Club assigns sub_grade using a proprietary algorithm that considers many factors "
        "beyond FICO alone. Our approximation:<br><br>"
        "<strong>FICO >= 750:</strong> Grade A<br>"
        "<strong>720-749:</strong> Grade B<br>"
        "<strong>690-719:</strong> Grade C<br>"
        "<strong>660-689:</strong> Grade D<br>"
        "<strong>640-659:</strong> Grade E<br>"
        "<strong>620-639:</strong> Grade F<br>"
        "<strong>&lt; 620:</strong> Grade G<br><br>"
        "Within each grade, the numeric sub-grade (1-5) is derived from the FICO remainder. "
        "This is a rough approximation — real sub-grades depend on DTI, income, and other factors.",
        accent_color=COLORS["chart_amber"],
    ))

with col3:
    st.html(info_card(
        "FICO-to-Interest-Rate Formula",
        "We estimate interest rate from FICO using a simple linear formula:<br><br>"
        "<code>int_rate = max(5.0, 30.0 - (fico - 600) * 0.12)</code><br><br>"
        "This produces rates from ~5% (FICO 850) to ~30% (FICO 600). In reality, interest "
        "rates depend on:<br><br>"
        "- Loan amount and term<br>"
        "- DTI ratio<br>"
        "- Market conditions at the time<br>"
        "- Lending Club's pricing model<br><br>"
        "Our linear approximation may over- or under-estimate by several percentage points.",
        accent_color=COLORS["chart_blue"],
    ))

# Input coverage visualization
fig = go.Figure()
fig.add_trace(go.Bar(
    x=["Collected from User", "Filled with Medians"],
    y=[9, 69],
    marker=dict(
        color=[COLORS["success"], COLORS["text_muted"]],
        line=dict(width=0),
    ),
    text=["9 inputs", "69 defaults"],
    textposition="inside",
    textfont=dict(color="white", size=14),
))
fig.update_layout(title="Borrower Page: Feature Coverage (78 Total)")
fig = plotly_layout(fig, height=280)
fig.update_layout(yaxis_title="Number of Features")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# METHODOLOGICAL LIMITATIONS
# ════════════════════════════════════════════
st.markdown("### Methodological Limitations")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Limited Hyperparameter Search",
        "Due to computational constraints, we ran only <strong>10 Optuna trials</strong> on "
        "<strong>25% of the training data</strong> (~280K rows). Each trial on the full 1.12M "
        "rows was taking ~8 minutes, making a thorough search impractical.<br><br>"
        "A production model would benefit from:<br>"
        "- 50-100+ trials on the full dataset<br>"
        "- Larger search ranges for key parameters<br>"
        "- Multi-objective optimization (AUC + calibration)<br>"
        "- Distributed computing for parallel trial evaluation",
        accent_color=COLORS["chart_purple"],
    ))

    st.html(info_card(
        "No Probability Calibration",
        "AUC measures <strong>ranking ability</strong> (can the model separate defaults from "
        "non-defaults?) but says nothing about whether the predicted probabilities are "
        "<strong>accurate in absolute terms</strong>.<br><br>"
        "If the model predicts 30% default probability, do 30% of those borrowers actually default? "
        "Without calibration (e.g., Platt scaling, isotonic regression), the raw probabilities "
        "should be treated as <strong>relative risk scores, not true probabilities</strong>.",
        accent_color=COLORS["chart_blue"],
    ))

with col2:
    st.html(info_card(
        "Train/Test Distribution Shift",
        f"Our training data (pre-2017) has a default rate of <strong>19.7%</strong>, while "
        f"the test data (2017-2018) has <strong>21.3%</strong>. This 1.6 percentage point "
        f"difference suggests changing economic conditions or lending standards.<br><br>"
        "The model was trained in one regime and tested in a slightly different one. "
        "In real deployment, this gap would likely be much larger (especially post-2020), "
        "and model performance would degrade accordingly.",
        accent_color=COLORS["warning"],
    ))

    st.html(info_card(
        "No Model Monitoring or Retraining Pipeline",
        "This is a static model — once trained, it does not update. A production system needs:<br><br>"
        "- <strong>Drift detection</strong> — monitoring whether input distributions or default "
        "rates are shifting away from training data<br>"
        "- <strong>Performance monitoring</strong> — tracking live AUC, precision, and recall "
        "as ground truth becomes available<br>"
        "- <strong>Automated retraining</strong> — periodic model refresh on new data<br>"
        "- <strong>A/B testing</strong> — comparing model versions before full deployment<br><br>"
        "None of these exist in the current platform.",
        accent_color=COLORS["danger"],
    ))

# Default rate shift visualization
fig = go.Figure()
fig.add_trace(go.Bar(
    x=["Training Set (< 2017)", "Test Set (2017-2018)"],
    y=[19.7, 21.3],
    marker=dict(
        color=[COLORS["chart_purple"], COLORS["chart_blue"]],
        line=dict(width=0),
    ),
    text=["19.7%", "21.3%"],
    textposition="outside",
    textfont=dict(color=COLORS["text"], size=15),
))
fig.add_shape(type="line", x0=-0.5, x1=1.5, y0=19.7, y1=19.7,
              line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
fig.add_annotation(x=1.4, y=20.5, text="+1.6pp shift",
                   font=dict(color=COLORS["warning"], size=12), showarrow=False)
fig.update_layout(title="Default Rate Shift Between Train and Test")
fig = plotly_layout(fig, height=300)
fig.update_layout(yaxis_title="Default Rate (%)", yaxis=dict(range=[18, 23]))
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# WHAT WOULD IMPROVE THE MODEL
# ════════════════════════════════════════════
st.markdown("### What Would Improve This Model")

improve_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
improvements = [
    ("Additional Data Sources", COLORS["success"],
     "Full credit bureau reports, bank transaction data, employment verification, "
     "and macroeconomic indicators would provide richer signal."),
    ("Walk-Forward Validation", COLORS["chart_blue"],
     "Multiple temporal splits would give more robust performance estimates and "
     "reveal how the model degrades over time."),
    ("Probability Calibration", COLORS["chart_purple"],
     "Platt scaling or isotonic regression to ensure predicted probabilities "
     "match observed default rates."),
    ("Larger Tuning Budget", COLORS["chart_amber"],
     "50-100+ Optuna trials on full data with wider search ranges. "
     "Consider additional algorithms (LightGBM, CatBoost) and ensembles."),
    ("Monitoring Pipeline", COLORS["warning"],
     "Drift detection, live performance tracking, automated retraining triggers, "
     "and alerting when model quality degrades."),
    ("Current Data", COLORS["danger"],
     "Retraining on post-2020 data to capture modern lending patterns, "
     "economic conditions, and borrower behaviors."),
]
for title, color, desc in improvements:
    improve_html += (
        f'<div style="flex:1;min-width:200px;background:linear-gradient(135deg,'
        f'rgba(26,31,78,0.7),rgba(30,36,86,0.5));backdrop-filter:blur(16px);'
        f'border:1px solid {hex_to_rgba(color, 0.2)};border-radius:14px;padding:18px;'
        f'border-top:3px solid {color};">'
        f'<div style="color:{color};font-weight:700;font-size:0.82rem;'
        f'margin-bottom:8px;">{title}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.75rem;line-height:1.4;">'
        f'{desc}</div></div>'
    )
improve_html += '</div>'
st.html(improve_html)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# FINAL NOTE
# ════════════════════════════════════════════
st.html(info_card(
    "Bottom Line",
    "This platform demonstrates a <strong>complete end-to-end credit risk modeling workflow</strong>: "
    "data cleaning, leakage prevention, model training, threshold optimization, explainability, "
    "and interactive deployment. The methodology is sound and the results are reasonable.<br><br>"
    "However, the model is <strong>not production-ready</strong> in its current form. The data is "
    "outdated, the tuning budget was limited, calibration was not performed, and there is no "
    "monitoring infrastructure. These are deliberate scope boundaries for an academic/portfolio "
    "project, not oversights.",
    accent_color=COLORS["accent_light"],
))
