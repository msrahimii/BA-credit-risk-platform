"""Methodology Page - Full modeling approach walkthrough with visualizations."""

import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
from utils import COLORS, section_header, info_card, plotly_layout, hex_to_rgba, metric_card_html

st.html(section_header("Methodology",
                        "End-to-end modeling approach for credit risk prediction"))

# ── Load metadata ──
with open("artifacts/model_metadata.json") as f:
    metadata = json.load(f)

# ════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════
st.markdown("### High-Level Approach")

overview_html = (
    '<div style="display:flex;gap:12px;flex-wrap:wrap;padding:8px 0;">'
)
overview_steps = [
    ("Problem Framing", "Binary classification: will a borrower default on their loan?",
     COLORS["chart_purple"], "1"),
    ("Algorithm", "XGBoost gradient-boosted trees — fast, handles imbalance, missing values natively",
     COLORS["chart_blue"], "2"),
    ("Tuning", "Bayesian optimization via Optuna with 3-fold cross-validation",
     COLORS["chart_green"], "3"),
    ("Dual Thresholds", "One model, two decision boundaries for borrower vs bank use cases",
     COLORS["warning"], "4"),
    ("Explainability", "SHAP TreeExplainer for per-prediction explanations and personalized advice",
     COLORS["chart_amber"], "5"),
]
for title, desc, color, num in overview_steps:
    overview_html += (
        f'<div style="flex:1;min-width:160px;background:linear-gradient(135deg,'
        f'rgba(26,31,78,0.8),rgba(30,36,86,0.6));backdrop-filter:blur(16px);'
        f'border:1px solid {hex_to_rgba(color, 0.25)};border-radius:16px;padding:18px;'
        f'border-top:3px solid {color};">'
        f'<div style="color:{color};font-weight:800;font-size:1.4rem;margin-bottom:4px;">{num}</div>'
        f'<div style="color:{COLORS["text"]};font-weight:700;font-size:0.85rem;'
        f'margin-bottom:6px;">{title}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.75rem;line-height:1.4;">'
        f'{desc}</div></div>'
    )
overview_html += '</div>'
st.html(overview_html)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# WHY XGBOOST?
# ════════════════════════════════════════════
st.markdown("### Why XGBoost?")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.html(info_card(
        "Algorithm Selection Rationale",
        "We evaluated the problem characteristics and chose XGBoost for several reasons:<br><br>"
        "<strong>1. Tabular data champion</strong> — XGBoost consistently outperforms neural networks "
        "on structured/tabular data, which is exactly what we have.<br>"
        "<strong>2. Built-in imbalance handling</strong> — <code>scale_pos_weight</code> natively "
        "adjusts the loss function for our 80/20 class split without distorting data.<br>"
        "<strong>3. Missing value support</strong> — learns optimal split directions for missing "
        "values during training, no imputation needed for tree splits.<br>"
        "<strong>4. SHAP compatibility</strong> — TreeExplainer provides exact (not approximate) "
        "SHAP values in polynomial time for tree ensembles.<br>"
        "<strong>5. Production-ready</strong> — fast inference, small model size, serializable to JSON.",
        accent_color=COLORS["chart_blue"],
    ))

with col2:
    # Algorithm comparison radar
    categories = ["Accuracy", "Speed", "Interpretability", "Imbalance\nHandling", "Feature\nImportance"]
    algorithms = {
        "XGBoost": [9, 8, 7, 9, 9],
        "Logistic Reg.": [5, 9, 9, 5, 6],
        "Random Forest": [7, 6, 5, 6, 7],
        "Neural Network": [7, 4, 3, 5, 3],
    }
    colors_alg = [COLORS["chart_blue"], COLORS["text_muted"],
                  COLORS["chart_green"], COLORS["chart_red"]]

    fig = go.Figure()
    for (name, vals), color in zip(algorithms.items(), colors_alg):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=name,
            line=dict(color=color, width=2.5 if name == "XGBoost" else 1.5),
            fill="toself" if name == "XGBoost" else None,
            fillcolor=hex_to_rgba(color, 0.1) if name == "XGBoost" else None,
            opacity=1.0 if name == "XGBoost" else 0.5,
        ))
    fig.update_layout(
        title="Algorithm Comparison (Credit Risk Context)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(45,53,97,0.5)",
                            tickfont=dict(size=9, color=COLORS["text_muted"])),
            angularaxis=dict(gridcolor="rgba(45,53,97,0.5)",
                             tickfont=dict(size=10, color=COLORS["text"])),
        ),
    )
    fig = plotly_layout(fig, height=400)
    fig.update_layout(legend=dict(x=0.78, y=1.15, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# CLASS IMBALANCE
# ════════════════════════════════════════════
st.markdown("### Handling Class Imbalance")

col1, col2 = st.columns(2)

with col1:
    # Imbalance visualization
    default_count = int(1345350 * 0.1997)
    non_default_count = 1345350 - default_count

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Non-Default (Fully Paid)", "Default (Charged Off)"],
        y=[non_default_count, default_count],
        marker=dict(
            color=[COLORS["success"], COLORS["danger"]],
            line=dict(width=0),
        ),
        text=[f"{non_default_count:,.0f}<br>({100-19.97:.1f}%)",
              f"{default_count:,.0f}<br>(19.97%)"],
        textposition="inside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(title="Class Distribution (Imbalanced)")
    fig = plotly_layout(fig, height=350)
    fig.update_layout(yaxis_title="Number of Loans")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.html(info_card(
        "scale_pos_weight Strategy",
        f"Our dataset has a <strong>~4:1 ratio</strong> of non-defaults to defaults. "
        f"The exact weight used: <strong>{metadata['best_params']['scale_pos_weight']:.2f}</strong><br><br>"
        "<strong>How it works:</strong> XGBoost multiplies the loss for positive (default) samples "
        "by this weight during gradient computation. This makes each default sample count ~4x more "
        "than a non-default, forcing the model to pay more attention to the minority class.<br><br>"
        "<strong>Why not SMOTE/resampling?</strong><br>"
        "- Avoids creating synthetic samples that may not represent real borrowers<br>"
        "- No risk of overfitting to generated data<br>"
        "- More computationally efficient (no data duplication)<br>"
        "- Preserves the true data distribution for calibrated probabilities",
        accent_color=COLORS["danger"],
    ))

    # Weight effect illustration
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Without Weighting", "With scale_pos_weight"],
        y=[1.0, metadata["best_params"]["scale_pos_weight"]],
        marker=dict(
            color=[COLORS["text_muted"], COLORS["danger"]],
            line=dict(width=0),
        ),
        text=["1.0x", f"{metadata['best_params']['scale_pos_weight']:.1f}x"],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=14),
    ))
    fig.update_layout(title="Loss Multiplier for Default Samples")
    fig = plotly_layout(fig, height=280)
    fig.update_layout(yaxis_title="Loss Weight")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# HYPERPARAMETER TUNING
# ════════════════════════════════════════════
st.markdown("### Hyperparameter Tuning with Optuna")

col1, col2 = st.columns([1.3, 1])

with col1:
    st.html(info_card(
        "Bayesian Optimization",
        "Unlike grid search (exhaustive) or random search (blind), Optuna uses <strong>Bayesian "
        "optimization</strong> with a Tree-structured Parzen Estimator (TPE) to intelligently "
        "explore the hyperparameter space.<br><br>"
        "<strong>Setup:</strong><br>"
        "- <strong>10 trials</strong> on 25% subsample (~280K rows) for speed<br>"
        "- <strong>3-fold cross-validation</strong> per trial for robust estimates<br>"
        "- <strong>Early stopping at 20 rounds</strong> within each CV fold<br>"
        "- Objective: maximize AUC-ROC<br><br>"
        "<strong>Search Ranges:</strong><br>"
        "- max_depth: 4-8 | learning_rate: 0.02-0.1 (log scale)<br>"
        "- min_child_weight: 3-10 | subsample: 0.6-0.9<br>"
        "- colsample_bytree: 0.6-0.9 | gamma: 0-0.5<br>"
        "- reg_alpha: 0.001-10 (log) | reg_lambda: 0.001-10 (log)",
        accent_color=COLORS["chart_purple"],
    ))

with col2:
    # Tuning comparison: baseline vs final
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Baseline Model", "After Optuna Tuning"],
        y=[metadata["auc_baseline"], metadata["auc_final"]],
        marker=dict(
            color=[COLORS["text_muted"], COLORS["success"]],
            line=dict(width=0),
        ),
        text=[f"AUC: {metadata['auc_baseline']:.4f}",
              f"AUC: {metadata['auc_final']:.4f}"],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=14),
    ))
    improvement = (metadata["auc_final"] - metadata["auc_baseline"]) / metadata["auc_baseline"] * 100
    fig.update_layout(title=f"Model Performance Improvement (+{improvement:.2f}%)")
    fig = plotly_layout(fig, height=320)
    fig.update_layout(
        yaxis_title="AUC-ROC",
        yaxis=dict(range=[0.72, 0.735]),
    )
    st.plotly_chart(fig, use_container_width=True)

# Best parameters display
st.markdown("#### Optimized Hyperparameters")

best_params = metadata["best_params"]
display_params = {
    "max_depth": ("Max Tree Depth", "Controls model complexity — deeper trees capture more interactions"),
    "learning_rate": ("Learning Rate", "Step size shrinkage — lower values need more trees but generalize better"),
    "min_child_weight": ("Min Child Weight", "Minimum sum of instance weight in a leaf — prevents overfitting to noise"),
    "subsample": ("Row Subsample", "Fraction of training rows used per tree — adds randomness"),
    "colsample_bytree": ("Column Subsample", "Fraction of features considered per tree — reduces correlation between trees"),
    "gamma": ("Gamma (Min Split Loss)", "Minimum loss reduction to make a split — pruning parameter"),
    "reg_alpha": ("L1 Regularization", "Pushes feature weights toward zero — feature selection effect"),
    "reg_lambda": ("L2 Regularization", "Penalizes large weights — smooths predictions"),
}

param_cols = st.columns(4)
for i, (key, (label, tooltip)) in enumerate(display_params.items()):
    val = best_params.get(key, "N/A")
    display_val = f"{val:.4f}" if isinstance(val, float) else str(val)
    col = param_cols[i % 4]
    col.html(
        f'<div style="background:linear-gradient(135deg,rgba(26,31,78,0.8),rgba(30,36,86,0.6));'
        f'backdrop-filter:blur(16px);border:1px solid rgba(129,140,248,0.12);border-radius:14px;'
        f'padding:14px;margin-bottom:10px;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;text-transform:uppercase;'
        f'letter-spacing:0.08em;font-weight:600;">{label}</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.3rem;font-weight:800;margin:6px 0 4px;'
        f'letter-spacing:-0.02em;">{display_val}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.65rem;line-height:1.3;'
        f'opacity:0.8;">{tooltip}</div></div>'
    )

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# FINAL MODEL TRAINING
# ════════════════════════════════════════════
st.markdown("### Final Model Training")

col1, col2 = st.columns(2)

with col1:
    st.html(info_card(
        "Training Configuration",
        "After finding optimal hyperparameters on the 25% subsample, we trained the final "
        "model on the <strong>full training set (1.12M rows)</strong>.<br><br>"
        "<strong>n_estimators:</strong> 2,000 (max allowed)<br>"
        "<strong>early_stopping_rounds:</strong> 80 (stops if no improvement for 80 rounds)<br>"
        "<strong>eval_set:</strong> Test set (225K rows) for monitoring<br><br>"
        "Early stopping prevents overfitting by halting training when the test AUC "
        "plateaus. The model uses the best iteration, not necessarily all 2,000 trees.",
        accent_color=COLORS["success"],
    ))

with col2:
    # Training process illustration
    # Simulated learning curve to illustrate early stopping
    np.random.seed(42)
    rounds = np.arange(0, 1200, 10)
    train_auc = 1 - 0.35 * np.exp(-rounds / 200) + np.random.normal(0, 0.002, len(rounds))
    test_auc = 1 - 0.35 * np.exp(-rounds / 250) - 0.02 * (rounds / 1200) + np.random.normal(0, 0.003, len(rounds))
    train_auc = np.clip(train_auc, 0.65, 0.85)
    test_auc = np.clip(test_auc, 0.65, 0.78)

    best_round = int(rounds[np.argmax(test_auc)])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=train_auc,
        mode="lines", name="Train AUC",
        line=dict(color=COLORS["chart_purple"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=test_auc,
        mode="lines", name="Test AUC",
        line=dict(color=COLORS["chart_blue"], width=2),
    ))
    fig.add_shape(
        type="line", x0=best_round, x1=best_round, y0=0.65, y1=0.85,
        line=dict(color=COLORS["warning"], width=2, dash="dash"),
    )
    fig.add_annotation(
        x=best_round, y=0.86, text=f"Best iteration: ~{best_round}",
        font=dict(color=COLORS["warning"], size=11), showarrow=False,
    )
    fig.add_shape(
        type="rect", x0=best_round, x1=1200, y0=0.65, y1=0.85,
        fillcolor=hex_to_rgba(COLORS["danger"], 0.05),
        line=dict(width=0),
    )
    fig.add_annotation(
        x=(best_round + 1200) / 2, y=0.67,
        text="Early stopped (overfitting zone)",
        font=dict(color=COLORS["text_muted"], size=9), showarrow=False,
    )
    fig.update_layout(title="Illustrative Learning Curve with Early Stopping")
    fig = plotly_layout(fig, height=380)
    fig.update_layout(
        xaxis_title="Boosting Round", yaxis_title="AUC-ROC",
        yaxis=dict(range=[0.65, 0.87]),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# DUAL THRESHOLD STRATEGY
# ════════════════════════════════════════════
st.markdown("### Dual Threshold Strategy")

st.html(info_card(
    "One Model, Two Decision Boundaries",
    "A single XGBoost model outputs a <strong>probability of default</strong> for each borrower. "
    "The threshold at which we classify someone as 'risky' depends on <strong>who is asking</strong>. "
    "A borrower checking their approval chances needs a different threshold than a bank assessing "
    "portfolio risk.<br><br>"
    "Both thresholds are optimized on the <strong>Precision-Recall curve</strong> of the test set, "
    "with different optimization objectives.",
    accent_color=COLORS["accent_light"],
))

col1, col2 = st.columns(2)

with col1:
    borrower = metadata["borrower_metrics"]
    st.html(
        f'<div style="background:linear-gradient(135deg,#064e3b,#065f46);'
        f'border:1px solid {COLORS["success"]};border-radius:16px;padding:24px;">'
        f'<div style="color:{COLORS["success"]};font-weight:800;font-size:1.1rem;'
        f'margin-bottom:12px;">Borrower Threshold</div>'
        f'<div style="color:{COLORS["text"]};font-size:2rem;font-weight:800;'
        f'margin-bottom:12px;">{borrower["threshold"]:.4f}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;line-height:1.6;">'
        f'<strong>Objective:</strong> Maximize precision where recall &ge; 30%<br>'
        f'<strong>Precision:</strong> {borrower["precision"]:.1%} | '
        f'<strong>Recall:</strong> {borrower["recall"]:.1%}<br><br>'
        f'<strong>Why:</strong> Borrowers want to know if they will be approved. '
        f'False alarms (telling a good borrower they will be rejected) create a poor experience. '
        f'High precision means when we say "risky," we are usually right.</div></div>'
    )

with col2:
    bank = metadata["bank_metrics"]
    st.html(
        f'<div style="background:linear-gradient(135deg,#451a03,#78350f);'
        f'border:1px solid {COLORS["warning"]};border-radius:16px;padding:24px;">'
        f'<div style="color:{COLORS["warning"]};font-weight:800;font-size:1.1rem;'
        f'margin-bottom:12px;">Bank Threshold</div>'
        f'<div style="color:{COLORS["text"]};font-size:2rem;font-weight:800;'
        f'margin-bottom:12px;">{bank["threshold"]:.4f}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;line-height:1.6;">'
        f'<strong>Objective:</strong> Maximize recall where precision &ge; 30%<br>'
        f'<strong>Precision:</strong> {bank["precision"]:.1%} | '
        f'<strong>Recall:</strong> {bank["recall"]:.1%}<br><br>'
        f'<strong>Why:</strong> Banks need to catch as many defaults as possible. '
        f'Missing a default (false negative) can cost tens of thousands. Some false positives '
        f'are acceptable — they just trigger additional review.</div></div>'
    )

# Threshold visualization on probability distribution
st.markdown("<br>", unsafe_allow_html=True)

# Simulated probability distribution to show threshold placement
np.random.seed(42)
non_default_probs = np.random.beta(2, 8, 5000)
default_probs = np.random.beta(5, 5, 1500)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=non_default_probs, name="Non-Default",
    marker_color=COLORS["chart_blue"], opacity=0.6, nbinsx=50,
))
fig.add_trace(go.Histogram(
    x=default_probs, name="Default",
    marker_color=COLORS["chart_red"], opacity=0.6, nbinsx=50,
))

fig.add_shape(type="line",
    x0=metadata["bank_threshold"], x1=metadata["bank_threshold"],
    y0=0, y1=1, yref="paper",
    line=dict(color=COLORS["warning"], width=2.5, dash="dash"))
fig.add_annotation(
    x=metadata["bank_threshold"], y=1.05, yref="paper",
    text=f"Bank: {metadata['bank_threshold']:.2f}",
    font=dict(color=COLORS["warning"], size=11), showarrow=False)

fig.add_shape(type="line",
    x0=metadata["borrower_threshold"], x1=metadata["borrower_threshold"],
    y0=0, y1=1, yref="paper",
    line=dict(color=COLORS["success"], width=2.5, dash="dash"))
fig.add_annotation(
    x=metadata["borrower_threshold"], y=1.05, yref="paper",
    text=f"Borrower: {metadata['borrower_threshold']:.2f}",
    font=dict(color=COLORS["success"], size=11), showarrow=False)

fig.update_layout(title="Illustrative Default Probability Distribution with Dual Thresholds",
                  barmode="overlay")
fig = plotly_layout(fig, height=380)
fig.update_layout(xaxis_title="Predicted Default Probability", yaxis_title="Count")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ════════════════════════════════════════════
st.markdown("### SHAP Explainability")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.html(info_card(
        "What is SHAP?",
        "<strong>SH</strong>apley <strong>A</strong>dditive ex<strong>P</strong>lanations (SHAP) "
        "is a game-theoretic approach to explain individual predictions. For each prediction, "
        "SHAP assigns each feature a value representing its contribution to the output.<br><br>"
        "<strong>Positive SHAP value</strong> &rarr; pushes prediction toward default (higher risk)<br>"
        "<strong>Negative SHAP value</strong> &rarr; pushes prediction toward non-default (lower risk)<br><br>"
        "We use <strong>TreeExplainer</strong>, which computes <strong>exact</strong> SHAP values "
        "for tree ensembles in polynomial time (not the slower KernelSHAP approximation).",
        accent_color=COLORS["chart_purple"],
    ))

    st.html(info_card(
        "How We Use SHAP in the Platform",
        "<strong>Borrower page:</strong> We identify the top 3 features with the highest "
        "positive SHAP values (increasing default risk) and translate them into "
        "<strong>personalized, actionable advice</strong> — e.g., 'Your DTI is high, "
        "paying down debt would improve your chances.'<br><br>"
        "<strong>Bank page:</strong> We show a waterfall-style bar chart of the top 12 "
        "SHAP contributors, with green bars (reduces risk) and red bars (increases risk), "
        "giving analysts a clear picture of what drives each individual's risk score.<br><br>"
        "<strong>Model Performance page:</strong> Global feature importance is computed as "
        "the mean absolute SHAP value across 5,000 test samples.",
        accent_color=COLORS["chart_green"],
    ))

with col2:
    # Top features by SHAP importance
    top_features = metadata.get("top_features", {})
    if top_features:
        features = list(top_features.keys())[:15]
        values = [top_features[f] for f in features]

        sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
        features = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f.replace("_", " ").title()[:25] for f in features],
            x=values, orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, COLORS["chart_blue"]], [1, COLORS["chart_purple"]]],
                line=dict(width=0),
            ),
        ))
        fig.update_layout(title="Top 15 Features by Mean |SHAP| Value")
        fig = plotly_layout(fig, height=500)
        fig.update_layout(
            xaxis_title="Mean |SHAP| Value",
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# SHAP interpretation example
st.markdown("#### How to Read SHAP Values")

shap_example_html = (
    '<div style="display:flex;gap:16px;flex-wrap:wrap;">'
    f'<div style="flex:1;min-width:250px;background:linear-gradient(135deg,'
    f'rgba(6,78,59,0.4),rgba(6,95,70,0.3));border:1px solid {COLORS["success"]};'
    f'border-radius:14px;padding:20px;">'
    f'<div style="color:{COLORS["success"]};font-weight:700;font-size:0.85rem;'
    f'margin-bottom:10px;">Negative SHAP = Lower Risk</div>'
    f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;line-height:1.5;">'
    f'Example: A borrower with a <strong>high FICO score (780)</strong> gets a '
    f'negative SHAP value for fico_range_low, meaning this feature is '
    f'<strong>pushing the prediction away from default</strong>. Good for the borrower!</div></div>'
    f'<div style="flex:1;min-width:250px;background:linear-gradient(135deg,'
    f'rgba(69,26,3,0.4),rgba(120,53,15,0.3));border:1px solid {COLORS["danger"]};'
    f'border-radius:14px;padding:20px;">'
    f'<div style="color:{COLORS["danger"]};font-weight:700;font-size:0.85rem;'
    f'margin-bottom:10px;">Positive SHAP = Higher Risk</div>'
    f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;line-height:1.5;">'
    f'Example: A borrower requesting a <strong>60-month term</strong> gets a '
    f'positive SHAP value for term, meaning this feature is '
    f'<strong>pushing the prediction toward default</strong>. Longer terms carry more risk.</div></div>'
    f'<div style="flex:1;min-width:250px;background:linear-gradient(135deg,'
    f'rgba(26,31,78,0.6),rgba(30,36,86,0.4));border:1px solid {COLORS["accent_light"]};'
    f'border-radius:14px;padding:20px;">'
    f'<div style="color:{COLORS["accent_light"]};font-weight:700;font-size:0.85rem;'
    f'margin-bottom:10px;">Sum of All SHAP = Prediction</div>'
    f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;line-height:1.5;">'
    f'The base value (average prediction) plus all SHAP values equals the final '
    f'log-odds prediction. This is the <strong>additive property</strong> — every feature '
    f'contribution is accounted for with no hidden interactions.</div></div>'
    '</div>'
)
st.html(shap_example_html)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# MODEL EVALUATION
# ════════════════════════════════════════════
st.markdown("### Model Evaluation Summary")

summary_cols = st.columns(4)
summary_items = [
    ("Test AUC-ROC", f"{metadata['auc_final']:.4f}", "success"),
    ("Borrower Precision", f"{metadata['borrower_metrics']['precision']:.1%}", "chart_green"),
    ("Bank Recall", f"{metadata['bank_metrics']['recall']:.1%}", "warning"),
    ("Features Used", str(len(metadata['feature_columns'])), "chart_purple"),
]
for col, (label, value, color_key) in zip(summary_cols, summary_items):
    col.html(metric_card_html(label, value))

st.markdown("<br>", unsafe_allow_html=True)

# Precision-Recall tradeoff explanation
st.html(info_card(
    "Understanding the Precision-Recall Tradeoff",
    "In credit risk, there is an inherent tension between two types of errors:<br><br>"
    "<strong>False Positive (Type I):</strong> Predicting default when the borrower would actually "
    "repay. Impact: good borrowers are denied loans or flagged unnecessarily.<br>"
    "<strong>False Negative (Type II):</strong> Predicting repayment when the borrower will actually "
    "default. Impact: the lender loses money on a bad loan.<br><br>"
    "<strong>Precision</strong> = of all predicted defaults, how many actually defaulted? (reduces Type I)<br>"
    "<strong>Recall</strong> = of all actual defaults, how many did we catch? (reduces Type II)<br><br>"
    "You cannot maximize both simultaneously. Our dual-threshold approach lets each stakeholder "
    "choose their preferred tradeoff point.",
    accent_color=COLORS["accent_light"],
))

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# LIMITATIONS
# ════════════════════════════════════════════
st.markdown("### Limitations & Considerations")

lim_cols = st.columns(3)

limitations = [
    ("Data Vintage", "danger",
     "The model was trained on 2007-2018 data. Lending patterns, economic conditions, "
     "and risk factors have evolved since then. The model should be retrained on current "
     "data before production use."),
    ("Feature Availability", "warning",
     "Some features used in training (e.g., FICO, sub_grade) are assigned by Lending Club. "
     "In a general credit risk setting, not all of these may be available at application time."),
    ("Calibration", "chart_blue",
     "While AUC measures ranking ability, the raw probabilities may not be perfectly calibrated. "
     "For precise probability estimates, Platt scaling or isotonic regression could be applied."),
]

for col, (title, color_key, text) in zip(lim_cols, limitations):
    col.html(info_card(title, text, accent_color=COLORS[color_key]))
