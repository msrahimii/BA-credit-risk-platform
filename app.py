"""Credit Risk Platform - Navigation Entry Point"""

import streamlit as st
from utils import apply_theme, COLORS

st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# Sidebar branding (below navigation)
st.sidebar.html(
    '<div style="padding:12px 16px;margin-top:12px;'
    'border-top:1px solid rgba(124,77,255,0.1);">'
    '<div style="background:linear-gradient(135deg,#b388ff,#00e5ff);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    'font-size:1.15rem;font-weight:800;letter-spacing:-0.02em;">'
    'Credit Risk Platform</div>'
    '<div style="color:#7a7a90;font-size:0.72rem;margin-top:4px;">'
    'Lending Club Data &middot; XGBoost &middot; SHAP</div></div>'
)

pg = st.navigation([
    st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("pages/borrower_assessment.py", title="Borrower Assessment", icon=":material/person_search:"),
    st.Page("pages/bank_risk_analysis.py", title="Bank Risk Analysis", icon=":material/account_balance:"),
    st.Page("pages/methodology.py", title="Methodology", icon=":material/biotech:"),
    st.Page("pages/dataset_insights.py", title="Dataset Insights", icon=":material/query_stats:"),
    st.Page("pages/data_processing.py", title="Data Processing", icon=":material/build:"),
    st.Page("pages/model_performance.py", title="Model Performance", icon=":material/speed:"),
    st.Page("pages/limitations.py", title="Limitations", icon=":material/report_problem:"),
])
pg.run()
