"""
Pre-compute aggregated data for Streamlit visualizations.
This avoids loading the full CSV in the app.
"""

import pandas as pd
import numpy as np
import json
import os

print("Loading raw data...")
df = pd.read_csv("accepted_2007_to_2018Q4.csv", low_memory=False)

# Filter to definitive outcomes only
valid_statuses = ["Fully Paid", "Charged Off", "Default"]
df = df[df["loan_status"].isin(valid_statuses)].copy()
df["target"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

os.makedirs("data/viz", exist_ok=True)

# 1. KPI summary stats
kpis = {
    "total_loans": int(len(df)),
    "total_funded": float(df["funded_amnt"].sum()),
    "avg_loan_amount": float(df["loan_amnt"].mean()),
    "default_rate": float(df["target"].mean()),
    "avg_int_rate": float(pd.to_numeric(df["int_rate"], errors="coerce").mean()),
    "avg_annual_inc": float(df["annual_inc"].median()),
    "avg_dti": float(pd.to_numeric(df["dti"], errors="coerce").mean()),
}
with open("data/viz/kpis.json", "w") as f:
    json.dump(kpis, f, indent=2)
print("  Saved KPIs")

# 2. Default rate by grade
grade_stats = df.groupby("grade").agg(
    count=("target", "size"),
    default_rate=("target", "mean"),
    avg_int_rate=("int_rate", lambda x: pd.to_numeric(x, errors="coerce").mean()),
).reset_index()
grade_stats.to_csv("data/viz/grade_stats.csv", index=False)
print("  Saved grade stats")

# 3. Loan status distribution
status_dist = df["loan_status"].value_counts().reset_index()
status_dist.columns = ["status", "count"]
status_dist.to_csv("data/viz/status_distribution.csv", index=False)
print("  Saved status distribution")

# 4. Default rate over time (by issue month)
df_time = df.dropna(subset=["issue_d"]).copy()
df_time["year_month"] = df_time["issue_d"].dt.to_period("Q").astype(str)
time_stats = df_time.groupby("year_month").agg(
    count=("target", "size"),
    default_rate=("target", "mean"),
    avg_loan_amnt=("loan_amnt", "mean"),
).reset_index()
time_stats.to_csv("data/viz/time_stats.csv", index=False)
print("  Saved time stats")

# 5. Purpose distribution with default rates
purpose_stats = df.groupby("purpose").agg(
    count=("target", "size"),
    default_rate=("target", "mean"),
    avg_loan_amnt=("loan_amnt", "mean"),
).reset_index().sort_values("count", ascending=False)
purpose_stats.to_csv("data/viz/purpose_stats.csv", index=False)
print("  Saved purpose stats")

# 6. Home ownership stats
home_stats = df.groupby("home_ownership").agg(
    count=("target", "size"),
    default_rate=("target", "mean"),
).reset_index()
home_stats = home_stats[home_stats["count"] > 1000]
home_stats.to_csv("data/viz/home_ownership_stats.csv", index=False)
print("  Saved home ownership stats")

# 7. State-level stats
state_stats = df.groupby("addr_state").agg(
    count=("target", "size"),
    default_rate=("target", "mean"),
    avg_loan_amnt=("loan_amnt", "mean"),
).reset_index()
state_stats.to_csv("data/viz/state_stats.csv", index=False)
print("  Saved state stats")

# 8. Numeric distributions (sampled for histograms)
numeric_sample = df[["loan_amnt", "int_rate", "annual_inc", "dti",
                      "fico_range_low", "revol_util", "target"]].copy()
numeric_sample["int_rate"] = pd.to_numeric(numeric_sample["int_rate"], errors="coerce")
numeric_sample["revol_util"] = pd.to_numeric(numeric_sample["revol_util"], errors="coerce")
numeric_sample["dti"] = pd.to_numeric(numeric_sample["dti"], errors="coerce")
# Cap annual_inc at 300k for visualization
numeric_sample["annual_inc"] = numeric_sample["annual_inc"].clip(upper=300000)
numeric_sample = numeric_sample.dropna().sample(n=50000, random_state=42)
numeric_sample.to_csv("data/viz/numeric_distributions.csv", index=False)
print("  Saved numeric distributions")

# 9. Correlation matrix of key features
corr_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low",
             "revol_util", "open_acc", "total_acc", "revol_bal", "target"]
corr_df = df[corr_cols].copy()
for c in corr_cols:
    corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
corr_matrix = corr_df.dropna().corr()
corr_matrix.to_csv("data/viz/correlation_matrix.csv")
print("  Saved correlation matrix")

# 10. Term distribution
term_stats = df.groupby("term").agg(
    count=("target", "size"),
    default_rate=("target", "mean"),
).reset_index()
term_stats.to_csv("data/viz/term_stats.csv", index=False)
print("  Saved term stats")

print("\nAll visualization data saved to data/viz/")
