import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from textblob import TextBlob
 
# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR AI Attrition Predictor", layout="wide")
 
# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('attrition_xgb_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        core_features = joblib.load('feature_columns.pkl')
        return model, scaler, core_features
    except FileNotFoundError:
        st.error("⚠️ Error: Missing .pkl files!")
        return None, None, None
 
model, scaler, core_features = load_assets()
 
# --- 2. THE UI HEADER ---
st.title("🛡️ AI-Driven Employee Retention Dashboard")
st.markdown("Automated risk detection and intervention planning powered by XGBoost.")
 
# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("App Instructions")
    st.info("Upload a CSV to generate the visual health report.")
    template_data = pd.DataFrame({
        'Employee_ID': ['EMP001'], 'Department': ['Sales'], 'Role': ['Manager'], 
        'Work_Location': ['Remote'], 'Base_Salary': [70000], 'Benchmark_Salary': [75000],
        'Job_Satisfaction': [3], 'Engagement_Level': [3], 'Work_Life_Balance': [3], 
        'Management_Support': [3], 'Career_Development': [3], 'Tenure_Years': [2.5], 
        'Employment_Type': ['Full-time'], 'Feedback_Comments': ['Sample feedback here.']
    })
    st.download_button("📥 Download CSV Template", template_data.to_csv(index=False), "hr_template.csv")
 
# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Employee Data (CSV Format)", type="csv")
 
if uploaded_file is not None and model is not None:
    df = pd.read_csv(uploaded_file)
    with st.spinner('AI is analyzing workforce health...'):
        # --- 5. FEATURE ENGINEERING ---
        df_proc = df.copy()
        df_proc['Comp_Ratio'] = df_proc['Base_Salary'] / (df_proc['Benchmark_Salary'] + 1)
        df_proc['Stagnation_Index'] = df_proc['Tenure_Years'] / (df_proc['Career_Development'] + 0.1)
        df_proc['Is_Contractor'] = np.where(df_proc['Employment_Type'] == 'Contract', 1, 0)
        df_proc['Survey_Sentiment'] = df_proc['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
 
        # --- 6. PREDICTION ---
        X_active = df_proc[core_features]
        X_scaled = scaler.transform(X_active)
        risk_probs = model.predict_proba(X_scaled)[:, 1]
        df_proc['Risk_Score_%'] = (risk_probs * 100).round(1)
 
        # --- 7. STRATEGY ENGINE ---
        def assign_strategy(row):
            score = row['Risk_Score_%']
            if score >= 75:
                tier = 'High Risk (Critical)'
                action = 'Urgent Salary Correction' if row['Comp_Ratio'] < 0.9 else 'Immediate Stay Interview'
            elif score >= 40:
                tier = 'Medium Risk (Monitor)'
                action = 'Engagement Project / Flex Work'
            else:
                tier = 'Low Risk (Stable)'
                action = 'Standard Engagement'
            return pd.Series([tier, action])
 
        df_proc[['Risk_Tier', 'Recommended_Action']] = df_proc.apply(assign_strategy, axis=1)
 
    # --- 8. VISUAL ANALYTICS SECTION ---
    st.divider()
    st.subheader("📊 Workforce Intelligence Dashboard")
    # Summary Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Workforce", len(df_proc))
    m2.metric("Critical Risks", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']), delta_color="inverse")
    m3.metric("Avg Retention Health", f"{100 - df_proc['Risk_Score_%'].mean().round(1)}%")
 
    # Charts Row 1
    c1, c2 = st.columns(2)
    with c1:
        # PIE CHART: Overall Attrition Risk
        fig_pie = px.pie(df_proc, names='Risk_Tier', hole=0.5,
                         title="Company-Wide Risk Distribution",
                         color='Risk_Tier',
                         color_discrete_map={'High Risk (Critical)':'#d32f2f', 
                                           'Medium Risk (Monitor)':'#f57c00', 
                                           'Low Risk (Stable)':'#2e7d32'})
        st.plotly_chart(fig_pie, use_container_width=True)
 
    with c2:
        # BAR CHART: Dept-wise Critical Risk
        dept_risk = df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)'].groupby('Department').size().reset_index(name='Count')
        fig_dept = px.bar(dept_risk.sort_values('Count'), x='Count', y='Department', orientation='h',
                         title="Critical Risks by Department",
                         color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig_dept, use_container_width=True)
 
    # Charts Row 2: Deep Dive
    st.markdown("#### 🎯 Identification of Dissatisfaction Factors")
    c3, c4 = st.columns(2)
 
    with c3:
        # SCATTER: Salary vs Growth (The "Why")
        fig_scatter = px.scatter(df_proc, x='Comp_Ratio', y='Stagnation_Index', 
                                 color='Risk_Tier', size='Risk_Score_%',
                                 hover_data=['Employee_ID'],
                                 title="Pay Parity vs. Career Stagnation Matrix",
                                 labels={'Comp_Ratio':'Pay Parity (1.0 = Market)', 'Stagnation_Index':'Career Stagnation'})
        st.plotly_chart(fig_scatter, use_container_width=True)
 
    with c4:
        # BOX PLOT: Sentiment by Risk Tier
        fig_box = px.box(df_proc, x='Risk_Tier', y='Survey_Sentiment', 
                        title="Employee Sentiment Polarity by Risk Group",
                        color='Risk_Tier',
                        color_discrete_map={'High Risk (Critical)':'#d32f2f', 
                                           'Medium Risk (Monitor)':'#f57c00', 
                                           'Low Risk (Stable)':'#2e7d32'})
        st.plotly_chart(fig_box, use_container_width=True)
 
    # --- 9. TABULAR DASHBOARD (RETAINED) ---
    st.divider()
    st.subheader("📋 Targeted Intervention Registry")
    display_cols = ['Employee_ID', 'Department', 'Risk_Score_%', 'Risk_Tier', 'Recommended_Action']
    st.dataframe(
        df_proc[display_cols].sort_values(by='Risk_Score_%', ascending=False),
        use_container_width=True
    )
 
    # --- 10. EXPORT ---
    csv_data = df_proc.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Full Actionable Report", csv_data, "HR_Retention_Strategy.csv", "text/csv")
 
else:
    st.info("System Ready. Please upload employee CSV to begin analysis.")
