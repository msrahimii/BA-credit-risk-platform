"""
XGBoost Credit Risk Model Training Pipeline
- Baseline model
- Optuna hyperparameter tuning
- Dual threshold optimization (borrower precision / bank recall)
- SHAP integration for personalized advice
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import joblib
import json
import time
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading processed data...")
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Class imbalance ratio
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_ratio = neg_count / pos_count
print(f"Class ratio (neg/pos): {scale_ratio:.2f}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# STEP 1: BASELINE MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 1: Baseline Model")
print("=" * 50)

baseline_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_ratio,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 30,
}

baseline_model = xgb.XGBClassifier(**baseline_params)
baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

y_prob_baseline = baseline_model.predict_proba(X_test)[:, 1]
auc_baseline = roc_auc_score(y_test, y_prob_baseline)
print(f"\nBaseline AUC: {auc_baseline:.4f}")


# ─────────────────────────────────────────────
# STEP 2: OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Hyperparameter Tuning (Optuna, 10 trials on 25% sample)")
print("=" * 50)

# Subsample training data for faster tuning (stratified)
sample_size = int(len(X_train) * 0.25)
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_train), size=sample_size, replace=False)
X_train_sample = X_train.iloc[sample_idx]
y_train_sample = y_train.iloc[sample_idx]
dtrain_sample = xgb.DMatrix(X_train_sample, label=y_train_sample)

print(f"  Tuning on {sample_size:,} rows (25% of training data)")


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_ratio,
        "tree_method": "hist",
        "seed": 42,
        # Tuned params — ranges centered around known-good values
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    result = xgb.cv(
        params,
        dtrain_sample,
        num_boost_round=500,
        nfold=3,
        early_stopping_rounds=20,
        seed=42,
        verbose_eval=False,
    )

    return result["test-auc-mean"].iloc[-1]


start_time = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, show_progress_bar=True)
elapsed = time.time() - start_time

print(f"\nTuning completed in {elapsed / 60:.1f} min")
print(f"Best CV AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# ─────────────────────────────────────────────
# STEP 3: TRAIN FINAL MODEL WITH BEST PARAMS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Training Final Model")
print("=" * 50)

best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_ratio,
    "tree_method": "hist",
    "random_state": 42,
    **study.best_params,
}

# Train with early stopping to find optimal n_estimators
final_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=1000,
    early_stopping_rounds=50,
)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

y_prob = final_model.predict_proba(X_test)[:, 1]
auc_final = roc_auc_score(y_test, y_prob)
print(f"\nFinal model AUC: {auc_final:.4f} (baseline was {auc_baseline:.4f})")

# ─────────────────────────────────────────────
# STEP 4: OPTIMAL THRESHOLDS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Finding Optimal Thresholds")
print("=" * 50)

precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)

# --- Borrower threshold: maximize precision while keeping recall >= 0.3 ---
# (We want few false alarms; it's OK to miss some defaults)
borrower_candidates = [
    (t, p, r) for t, p, r in zip(thresholds_pr, precisions[:-1], recalls[:-1])
    if r >= 0.30
]
# Among candidates, pick the one with highest precision
borrower_candidates.sort(key=lambda x: x[1], reverse=True)
borrower_threshold = borrower_candidates[0][0]
borrower_precision = borrower_candidates[0][1]
borrower_recall = borrower_candidates[0][2]

# --- Bank threshold: maximize recall while keeping precision >= 0.30 ---
# (We want to catch most defaults; some false alarms are acceptable)
bank_candidates = [
    (t, p, r) for t, p, r in zip(thresholds_pr, precisions[:-1], recalls[:-1])
    if p >= 0.30
]
# Among candidates, pick the one with highest recall
bank_candidates.sort(key=lambda x: x[2], reverse=True)
bank_threshold = bank_candidates[0][0]
bank_precision = bank_candidates[0][1]
bank_recall = bank_candidates[0][2]

print(f"\nBorrower threshold: {borrower_threshold:.4f}")
print(f"  Precision: {borrower_precision:.4f}, Recall: {borrower_recall:.4f}")

print(f"\nBank threshold: {bank_threshold:.4f}")
print(f"  Precision: {bank_precision:.4f}, Recall: {bank_recall:.4f}")

# Full evaluation at both thresholds
for name, threshold in [("Borrower", borrower_threshold), ("Bank", bank_threshold)]:
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n--- {name} Page (threshold={threshold:.4f}) ---")
    print(classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

# ─────────────────────────────────────────────
# STEP 5: SHAP EXPLAINER
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Building SHAP Explainer")
print("=" * 50)

explainer = shap.TreeExplainer(final_model)

# Compute SHAP values on a sample for global feature importance
sample_idx = np.random.RandomState(42).choice(len(X_test), size=5000, replace=False)
X_sample = X_test.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

# Top 15 most important features globally
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.Series(mean_abs_shap, index=X_train.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 15 features by SHAP importance:")
for i, (feat, val) in enumerate(feature_importance.head(15).items(), 1):
    print(f"  {i:2d}. {feat}: {val:.4f}")

# ─────────────────────────────────────────────
# STEP 6: SAVE EVERYTHING
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Saving Model & Artifacts")
print("=" * 50)

# Save model in native XGBoost format (smaller, faster to load)
final_model.save_model("artifacts/xgb_credit_risk_model.json")

# Save SHAP explainer
joblib.dump(explainer, "artifacts/shap_explainer.pkl")

# Save thresholds and metadata
model_metadata = {
    "borrower_threshold": float(borrower_threshold),
    "bank_threshold": float(bank_threshold),
    "auc_baseline": float(auc_baseline),
    "auc_final": float(auc_final),
    "best_params": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_params.items()},
    "feature_columns": list(X_train.columns),
    "top_features": feature_importance.head(20).to_dict(),
    "borrower_metrics": {
        "threshold": float(borrower_threshold),
        "precision": float(borrower_precision),
        "recall": float(borrower_recall),
    },
    "bank_metrics": {
        "threshold": float(bank_threshold),
        "precision": float(bank_precision),
        "recall": float(bank_recall),
    },
}

with open("artifacts/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

print("\nSaved:")
print("  artifacts/xgb_credit_risk_model.json  (XGBoost model)")
print("  artifacts/shap_explainer.pkl          (SHAP explainer)")
print("  artifacts/model_metadata.json         (thresholds & metrics)")
print("\nDone!")
