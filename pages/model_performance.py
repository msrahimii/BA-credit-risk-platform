"""Model Performance Page - Evaluation metrics and feature importance."""

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from utils import COLORS, metric_card_html, plotly_layout, section_header, info_card


def downsample_curve(x_values, y_values, max_points=400):
    """Keep curve payloads compact while preserving the overall shape."""
    if len(x_values) <= max_points:
        return x_values.tolist(), y_values.tolist()

    sample_idx = np.linspace(0, len(x_values) - 1, max_points, dtype=int)
    sample_idx = np.unique(np.concatenate(([0], sample_idx, [len(x_values) - 1])))
    return x_values[sample_idx].tolist(), y_values[sample_idx].tolist()


def build_performance_payload(y_true, y_score, metadata):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    roc_fpr, roc_tpr = downsample_curve(fpr, tpr)
    pr_recall, pr_precision = downsample_curve(recall, precision)

    threshold_points = {}
    for key, threshold in [
        ("borrower", metadata["borrower_threshold"]),
        ("bank", metadata["bank_threshold"]),
    ]:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        threshold_points[key] = {
            "threshold": float(threshold),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
        }

    confusion_matrices = {}
    for key, threshold in [
        ("borrower", metadata["borrower_threshold"]),
        ("bank", metadata["bank_threshold"]),
    ]:
        y_pred = (y_score >= threshold).astype(int)
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        confusion_matrices[key] = {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": float(accuracy),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    return {
        "test_samples": int(len(y_true)),
        "test_default_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, y_score)),
        "roc_curve": {
            "fpr": roc_fpr,
            "tpr": roc_tpr,
        },
        "pr_curve": {
            "recall": pr_recall,
            "precision": pr_precision,
        },
        "threshold_points": threshold_points,
        "confusion_matrices": confusion_matrices,
    }


@st.cache_resource
def load_all():
    loaded_model = xgb.XGBClassifier()
    with open("artifacts/model_metadata.json") as f:
        metadata = json.load(f)

    perf_path = Path("artifacts/model_performance_data.json")
    if perf_path.exists():
        with perf_path.open() as f:
            performance = json.load(f)
        return metadata, performance

    loaded_model.load_model("artifacts/xgb_credit_risk_model.json")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    y_prob = loaded_model.predict_proba(X_test)[:, 1]
    performance = build_performance_payload(y_test, y_prob, metadata)
    return metadata, performance


metadata, performance = load_all()

st.html(section_header("Model Performance",
                        "XGBoost model evaluation on held-out test data (2017-2018)"))

# ── KPI Row ──
cols = st.columns(4)
with cols[0]:
    st.html(metric_card_html("AUC-ROC", f"{metadata['auc_final']:.4f}"))
with cols[1]:
    bm = metadata["borrower_metrics"]
    st.html(metric_card_html("Borrower Precision", f"{bm['precision']:.1%}",
                              delta=f"Recall: {bm['recall']:.1%}"))
with cols[2]:
    bankm = metadata["bank_metrics"]
    st.html(metric_card_html("Bank Recall", f"{bankm['recall']:.1%}",
                              delta=f"Precision: {bankm['precision']:.1%}"))
with cols[3]:
    st.html(metric_card_html("Test Samples", f"{performance['test_samples']:,}",
                              delta=f"{performance['test_default_rate']:.1%} default rate"))

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["Curves", "Confusion Matrices", "Feature Importance"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=performance["roc_curve"]["fpr"], y=performance["roc_curve"]["tpr"],
            mode="lines",
            name=f"XGBoost (AUC = {performance['auc']:.4f})",
            line=dict(color=COLORS["chart_purple"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(129, 140, 248, 0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
        ))
        fig.update_layout(title="ROC Curve")
        fig = plotly_layout(fig, height=420)
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Precision-Recall Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=performance["pr_curve"]["recall"], y=performance["pr_curve"]["precision"],
            mode="lines",
            name="Precision-Recall",
            line=dict(color=COLORS["chart_blue"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.1)",
        ))
        # Mark thresholds
        for name, th, color, symbol in [
            ("Borrower", metadata["borrower_threshold"], COLORS["success"], "circle"),
            ("Bank", metadata["bank_threshold"], COLORS["warning"], "diamond"),
        ]:
            point = performance["threshold_points"][name.lower()]
            fig.add_trace(go.Scatter(
                x=[point["recall"]], y=[point["precision"]],
                mode="markers+text",
                name=f"{name} Threshold ({th:.2f})",
                marker=dict(color=color, size=14, symbol=symbol,
                            line=dict(color="white", width=2)),
                text=[f"{name}"],
                textposition="top right",
                textfont=dict(color=color),
            ))
        fig.update_layout(title="Precision-Recall Curve")
        fig = plotly_layout(fig, height=420)
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(x=0.01, y=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Threshold analysis
    st.markdown("### Threshold Comparison")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.html(info_card(
            f"Borrower Threshold: {metadata['borrower_threshold']:.4f}",
            "Optimized for <strong>precision</strong> — fewer false alarms. "
            "Only flags borrowers as risky when the model is highly confident. "
            "Suitable for the borrower-facing experience.",
            accent_color=COLORS["success"],
        ))
    with tcol2:
        st.html(info_card(
            f"Bank Threshold: {metadata['bank_threshold']:.4f}",
            "Optimized for <strong>recall</strong> — catches more defaults. "
            "Some false positives, but banks prefer not to miss risky borrowers. "
            "Suitable for institutional risk assessment.",
            accent_color=COLORS["warning"],
        ))

with tab2:
    st.markdown("### Confusion Matrices")

    col1, col2 = st.columns(2)

    for col, name, threshold in [
        (col1, "Borrower", metadata["borrower_threshold"]),
        (col2, "Bank", metadata["bank_threshold"]),
    ]:
        with col:
            cm_data = performance["confusion_matrices"][name.lower()]
            tn = cm_data["tn"]
            fp = cm_data["fp"]
            fn = cm_data["fn"]
            tp = cm_data["tp"]

            cm = [[tn, fp], [fn, tp]]
            labels = [["True Neg", "False Pos"], ["False Neg", "True Pos"]]

            fig = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted: No Default", "Predicted: Default"],
                y=["Actual: No Default", "Actual: Default"],
                text=[[f"{labels[i][j]}<br>{cm[i][j]:,}" for j in range(2)] for i in range(2)],
                texttemplate="%{text}",
                textfont=dict(size=13),
                colorscale=[[0, COLORS["bg_card"]], [1, COLORS["accent"]]],
                showscale=False,
            ))
            fig.update_layout(title=f"{name} Page (threshold={threshold:.4f})")
            fig = plotly_layout(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)

            accuracy = cm_data["accuracy"]
            prec = cm_data["precision"]
            rec = cm_data["recall"]
            f1 = cm_data["f1"]

            st.markdown(
                f"""<div style="display:flex;gap:1rem;margin-top:0.5rem;">
                <div style="flex:1;text-align:center;background:{COLORS['bg_card']};
                border-radius:8px;padding:8px;">
                <span style="color:{COLORS['text_muted']};font-size:0.7rem;">ACCURACY</span><br>
                <span style="color:{COLORS['text']};font-weight:700;">{accuracy:.1%}</span>
                </div>
                <div style="flex:1;text-align:center;background:{COLORS['bg_card']};
                border-radius:8px;padding:8px;">
                <span style="color:{COLORS['text_muted']};font-size:0.7rem;">PRECISION</span><br>
                <span style="color:{COLORS['text']};font-weight:700;">{prec:.1%}</span>
                </div>
                <div style="flex:1;text-align:center;background:{COLORS['bg_card']};
                border-radius:8px;padding:8px;">
                <span style="color:{COLORS['text_muted']};font-size:0.7rem;">RECALL</span><br>
                <span style="color:{COLORS['text']};font-weight:700;">{rec:.1%}</span>
                </div>
                <div style="flex:1;text-align:center;background:{COLORS['bg_card']};
                border-radius:8px;padding:8px;">
                <span style="color:{COLORS['text_muted']};font-size:0.7rem;">F1</span><br>
                <span style="color:{COLORS['text']};font-weight:700;">{f1:.1%}</span>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

with tab3:
    st.markdown("### Feature Importance (SHAP)")

    top_features = metadata.get("top_features", {})
    if top_features:
        features = list(top_features.keys())
        values = list(top_features.values())

        # Sort ascending for horizontal bar
        sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
        features = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f.replace("_", " ").title()[:30] for f in features],
            x=values,
            orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, COLORS["chart_blue"]], [1, COLORS["chart_purple"]]],
                line=dict(width=0),
            ),
        ))
        fig.update_layout(title="Top 20 Features by Mean |SHAP| Value")
        fig = plotly_layout(fig, height=600)
        fig.update_layout(
            xaxis_title="Mean |SHAP| Value",
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model parameters
    st.markdown("### Model Configuration")
    best_params = metadata.get("best_params", {})
    param_cols = st.columns(4)
    for i, (k, v) in enumerate(best_params.items()):
        col = param_cols[i % 4]
        display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
        col.markdown(
            f"""<div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
            border-radius:10px;padding:12px;margin-bottom:8px;">
            <span style="color:{COLORS['text_muted']};font-size:0.7rem;text-transform:uppercase;">
            {k}</span><br>
            <span style="color:{COLORS['text']};font-weight:600;">{display_val}</span>
            </div>""",
            unsafe_allow_html=True,
        )
