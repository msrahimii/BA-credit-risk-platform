"""
Data Preparation Pipeline for Credit Risk Model
Lending Club Dataset (2007-2018 Q4)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

# ─────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────
print("Step 1: Loading data...")
df = pd.read_csv("accepted_2007_to_2018Q4.csv", low_memory=False)
print(f"  Raw shape: {df.shape}")

# ─────────────────────────────────────────────
# STEP 2: Define Target Variable
# ─────────────────────────────────────────────
print("Step 2: Defining target variable...")

# Keep only loans with definitive outcomes
valid_statuses = ["Fully Paid", "Charged Off", "Default"]
df = df[df["loan_status"].isin(valid_statuses)].copy()

# Binary target: 0 = non-default, 1 = default
df["target"] = (df["loan_status"].isin(["Charged Off", "Default"])).astype(int)
print(f"  After filtering loan_status: {df.shape}")
print(f"  Target distribution:\n{df['target'].value_counts(normalize=True).to_string()}")

# ─────────────────────────────────────────────
# STEP 3: Drop Data Leakage Columns
# ─────────────────────────────────────────────
print("Step 3: Dropping data leakage columns...")

leakage_cols = [
    # Payment performance (post-origination)
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    # Payment dates
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    # Post-origination credit
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # Hardship program (all hardship_* columns)
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # Settlement
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    # Other post-origination
    "pymnt_plan", "funded_amnt", "funded_amnt_inv",
    "disbursement_method",
]

# ─────────────────────────────────────────────
# STEP 4: Drop Useless / ID Columns
# ─────────────────────────────────────────────
print("Step 4: Dropping useless/ID columns...")

useless_cols = [
    "id", "member_id", "url", "desc", "emp_title", "title",
    "zip_code", "policy_code", "loan_status",
]

# ─────────────────────────────────────────────
# STEP 5: Drop Joint Application Columns
# ─────────────────────────────────────────────
print("Step 5: Dropping joint application columns...")

joint_cols = [
    "annual_inc_joint", "dti_joint", "verification_status_joint",
    "revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_earliest_cr_line", "sec_app_inq_last_6mths",
    "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
    "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]

# Drop all at once
all_drop = leakage_cols + useless_cols + joint_cols
# Only drop columns that actually exist
existing_drop = [c for c in all_drop if c in df.columns]
df.drop(columns=existing_drop, inplace=True)
print(f"  Dropped {len(existing_drop)} columns. Remaining: {df.shape[1]}")

# ─────────────────────────────────────────────
# STEP 6: Parse & Clean Columns
# ─────────────────────────────────────────────
print("Step 6: Parsing and cleaning columns...")

# term: " 36 months" -> 36
df["term"] = df["term"].str.strip().str.replace(" months", "", regex=False).astype(float)

# int_rate: "13.99" (already numeric in this dataset, but ensure)
df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

# emp_length: map to numeric
emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 10,
}
df["emp_length"] = df["emp_length"].map(emp_map)

# earliest_cr_line: convert to credit history length in months
# Use issue_d as reference — but we dropped it. Use a fixed reference date instead.
reference_date = pd.Timestamp("2019-01-01")  # just after the last data point
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
df["credit_history_months"] = (
    (reference_date - df["earliest_cr_line"]).dt.days / 30.44
).round().astype("Int64")
df.drop(columns=["earliest_cr_line"], inplace=True)

# revol_util: should be numeric, force conversion
df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

# issue_d: we need it for time-based split before dropping
# We already dropped it in leakage... let's reload just that column
print("  Reloading issue_d for time-based split...")
issue_dates = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols=["issue_d"], low_memory=False)
df["issue_d"] = issue_dates.loc[df.index, "issue_d"]
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

print(f"  Columns after parsing: {df.shape[1]}")

# ─────────────────────────────────────────────
# STEP 7: Handle Missing Values
# ─────────────────────────────────────────────
print("Step 7: Handling missing values...")

# FIRST: impute "months since" columns BEFORE checking missing %.
# Missing here means "never happened" (e.g. never delinquent) — that IS the signal.
mths_since_cols = [c for c in df.columns if "mths_since" in c or "mo_sin" in c]
for col in mths_since_cols:
    df[col] = df[col].fillna(-1)
print(f"  Imputed {len(mths_since_cols)} 'months since' columns with -1")

# Now drop columns with >70% missing (raised from 50% to keep more signal)
missing_pct = df.isnull().mean()
high_missing = missing_pct[missing_pct > 0.70].index.tolist()
# Don't drop issue_d or target
high_missing = [c for c in high_missing if c not in ["issue_d", "target"]]
print(f"  Dropping {len(high_missing)} columns with >70% missing: {high_missing}")
df.drop(columns=high_missing, inplace=True)

# Numeric columns: median imputation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "target"]
medians = df[numeric_cols].median()
df[numeric_cols] = df[numeric_cols].fillna(medians)

# Categorical columns: fill with "Unknown"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "issue_d"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

print(f"  Remaining nulls: {df.isnull().sum().sum()} (excluding issue_d)")

# ─────────────────────────────────────────────
# STEP 8: Encode Categorical Variables
# ─────────────────────────────────────────────
print("Step 8: Encoding categorical variables...")

# Grade: ordinal
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["grade"] = df["grade"].map(grade_map)

# Sub_grade: ordinal
sub_grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
sub_grade_map = {sg: i + 1 for i, sg in enumerate(sub_grades)}
df["sub_grade"] = df["sub_grade"].map(sub_grade_map)

# ── Time-based split FIRST (so target encoding uses only train data) ──
print("  Performing time-based split for encoding...")
split_date = pd.Timestamp("2017-01-01")  # train: <2017, test: >=2017
train_mask = df["issue_d"] < split_date
test_mask = df["issue_d"] >= split_date

# Drop rows where issue_d is null (can't assign to train/test)
valid_date_mask = df["issue_d"].notna()
df = df[valid_date_mask].copy()
train_mask = df["issue_d"] < split_date
test_mask = df["issue_d"] >= split_date

print(f"  Train size: {train_mask.sum()}, Test size: {test_mask.sum()}")

# Target encoding for addr_state (computed on train only)
state_target_mean = df.loc[train_mask].groupby("addr_state")["target"].mean()
global_mean = df.loc[train_mask, "target"].mean()
df["addr_state_encoded"] = df["addr_state"].map(state_target_mean).fillna(global_mean)
df.drop(columns=["addr_state"], inplace=True)

# One-hot / label encode remaining categoricals
remaining_cats = df.select_dtypes(include=["object"]).columns.tolist()
remaining_cats = [c for c in remaining_cats if c != "issue_d"]
print(f"  Label encoding: {remaining_cats}")

label_encodings = {}
for col in remaining_cats:
    categories = sorted(df[col].unique())
    cat_map = {cat: i for i, cat in enumerate(categories)}
    label_encodings[col] = cat_map
    df[col] = df[col].map(cat_map)

# ─────────────────────────────────────────────
# STEP 9: Feature Selection - Remove High Correlation
# ─────────────────────────────────────────────
print("Step 9: Feature selection...")

feature_cols = [c for c in df.columns if c not in ["target", "issue_d"]]

# Protect important features from being dropped by correlation filter
protected_features = {"sub_grade", "int_rate", "fico_range_low", "loan_amnt", "dti"}

# Remove highly correlated pairs (>0.95), but keep protected features
corr_matrix = df[feature_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = set()
for col in upper_tri.columns:
    if col in protected_features:
        continue
    if any(upper_tri[col] > 0.95):
        high_corr_cols.add(col)

high_corr_cols = list(high_corr_cols)
print(f"  Dropping {len(high_corr_cols)} highly correlated columns: {high_corr_cols}")
df.drop(columns=high_corr_cols, inplace=True)

# ─────────────────────────────────────────────
# STEP 10: Final Split & Save
# ─────────────────────────────────────────────
print("Step 10: Final split and save...")

feature_cols = [c for c in df.columns if c not in ["target", "issue_d"]]

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, "target"]
X_test = df.loc[test_mask, feature_cols]
y_test = df.loc[test_mask, "target"]

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"  Train default rate: {y_train.mean():.4f}")
print(f"  Test default rate:  {y_test.mean():.4f}")

# Save processed data
os.makedirs("data", exist_ok=True)
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save encoding mappings and medians for the Streamlit app
artifacts = {
    "feature_columns": feature_cols,
    "medians": medians.to_dict(),
    "label_encodings": label_encodings,
    "state_target_encoding": state_target_mean.to_dict(),
    "global_default_rate": global_mean,
    "grade_map": grade_map,
    "sub_grade_map": sub_grade_map,
    "emp_length_map": emp_map,
}
os.makedirs("artifacts", exist_ok=True)
joblib.dump(artifacts, "artifacts/preprocessing_artifacts.pkl")

print("\nDone! Files saved:")
print("  data/X_train.csv, data/X_test.csv")
print("  data/y_train.csv, data/y_test.csv")
print("  artifacts/preprocessing_artifacts.pkl")
print(f"\nFinal features ({len(feature_cols)}):")
print(f"  {feature_cols}")
