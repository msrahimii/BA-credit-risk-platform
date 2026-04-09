---
name: Credit Risk Platform Overview
description: Project goals, architecture decisions, and model design for the BA credit risk Streamlit platform
type: project
---

Building a credit risk platform using Lending Club data (2007-2018 Q4, ~2.26M rows).

**Multi-page Streamlit app:**
1. **Borrower page** — check loan approval chances. Precision-focused threshold. SHAP-based personalized advice. Simple UX (few inputs).
2. **Bank page** — check borrower default probability. Recall-focused threshold. More detailed inputs.
3. **Dataset visualizations page** — EDA charts, distributions, correlations from the Lending Club data.
4. **Model performance page** — AUC-ROC curves, confusion matrices, precision-recall curves, feature importance.

**Architecture:**
- One XGBoost model, two thresholds (borrower = high precision, bank = high recall)
- SHAP TreeExplainer for personalized feature-level advice
- Target encoding for `addr_state`, ordinal for `grade`/`sub_grade`
- Time-based train/test split (train <2017, test >=2017)
- Joint application columns dropped (Option A)

**Why:** BA project for a credit risk assessment tool.

**How to apply:** Keep borrower UX simple (few inputs). Bank page can be more detailed. Include visualization pages for both dataset insights and model performance.
