import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(layout="wide", page_title="Sales Intelligence Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("sales_performance_data.csv")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Global Filters")
region_list = ['All'] + list(df['Region'].unique())
selected_region = st.sidebar.selectbox("Select Region", region_list, key="reg_filter")

data = df.copy()
if selected_region != 'All':
    data = data[data['Region'] == selected_region]

# --- TABS ---
tabs = st.tabs(["🏠 Home", "📊 Descriptive", "🔍 Diagnostic", "🎯 Perspective", "🔮 Predictive"])

# 1. HOME PAGE
with tabs[0]:
    st.title("Executive Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dialed", data['Calls_Dialed'].sum())
    c2.metric("Qualified", data['Qualified'].sum())
    c3.metric("Avg Talk", f"{data['Call_Time_Mins'].mean():.1f}m")
    c4.metric("Deals", data['Deals_Closed'].sum())
    c5.metric("Revenue", f"₹{data['Total_Revenue'].sum():,.0f}")
    
    col1, col2 = st.columns(2)
    fig_bar = px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], title="Dialed vs Converted")
    col1.plotly_chart(fig_bar, use_container_width=True, key="home_bar_chart")
    fig_hist_home = px.histogram(data, x='Call_Time_Mins', title="Talk Time Distribution")
    col2.plotly_chart(fig_hist_home, use_container_width=True, key="home_hist_chart")

# --- NEW TAB: MANAGER PERFORMANCE ---
with st.expander("Region-wise Manager Performance View"):
    st.header("Manager Performance by Region")
    selected_mgr_region = st.selectbox("Select Region for Manager View", df['Region'].unique())
    mgr_data = df[df['Region'] == selected_mgr_region]
    mgr_perf = mgr_data.groupby('Sales_Manager_Name')[['Total_Revenue', 'Deals_Closed']].sum().reset_index()
    st.dataframe(mgr_perf, use_container_width=True)
    fig_mgr = px.bar(mgr_perf, x='Sales_Manager_Name', y='Total_Revenue', title=f"Revenue by Managers in {selected_mgr_region}")
    st.plotly_chart(fig_mgr, use_container_width=True, key="mgr_perf_chart")

# 2. DESCRIPTIVE
with tabs[1]:
    st.header("Descriptive Analysis")
    col1, col2 = st.columns(2)

    fig_eff = px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], 
                     title="Call Efficiency: Dialed vs Converted Calls", barmode='group')
    col1.plotly_chart(fig_eff, use_container_width=True, key="desc_eff_chart")

    fig_hist = px.histogram(data, x='Call_Time_Mins', nbins=20, 
                            title="Talk Time Distribution (Outlier Detection)",
                            color_discrete_sequence=['#636EFA'])
    col2.plotly_chart(fig_hist, use_container_width=True, key="desc_hist_chart")

    st.subheader("Conversion Funnel")
    funnel_df = pd.DataFrame({
        "Stage": ["Dialed", "Qualified", "Converted", "Deals Closed"],
        "Count": [data['Calls_Dialed'].sum(), data['Qualified'].sum(), 
                  data['Converted'].sum(), data['Deals_Closed'].sum()]
    })
    fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', title="Sales Conversion Funnel")
    st.plotly_chart(fig_funnel, use_container_width=True, key="desc_funnel_chart")

    st.subheader("Summary Metrics Table")
    st.dataframe(data.describe(), use_container_width=True)

# 3. DIAGNOSTIC
with tabs[2]:
    st.header("Diagnostic Analysis")
    numeric_cols = ['Total_Revenue', 'Calls_Dialed', 'Converted', 'Deals_Closed', 'Call_Time_Mins']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    st.write(f"Rows in current view: {len(data)}")
    if len(data) == 0:
        st.error("No data found for this selection. Check your filters.")
    else:
        st.subheader("Pareto: Revenue Contribution")
        pareto_data = data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().sort_values(ascending=False).reset_index()
        fig_pareto = px.bar(pareto_data, x='Sales_Rep_Name', y='Total_Revenue')
        st.plotly_chart(fig_pareto, use_container_width=True, key="desc_pareto_chart")

        st.subheader("Talk Time Outliers")
        fig_box = px.box(data, y='Call_Time_Mins')
        st.plotly_chart(fig_box, use_container_width=True, key="desc_box_chart")

        st.subheader("Call Efficiency")
        df_melted = data.melt(id_vars=['Sales_Rep_Name'], value_vars=['Calls_Dialed', 'Converted'])
        fig_stacked = px.bar(df_melted, x='Sales_Rep_Name', y='value', color='variable', barmode='stack')
        st.plotly_chart(fig_stacked, use_container_width=True, key="home_stacked_chart")

# 4. PERSPECTIVE
with tabs[3]:
    st.header("Perspective & Benchmarks")
    st.subheader("Benchmark: Rep Performance vs Team Average")
    avg_rev = data['Total_Revenue'].mean()
    fig_bench = px.bar(data, x='Sales_Rep_Name', y='Total_Revenue', title="Revenue per Rep vs Team Average")
    fig_bench.add_hline(y=avg_rev, line_dash="dash", line_color="red", annotation_text="Team Average")
    st.plotly_chart(fig_bench, use_container_width=True, key="persp_bench_chart")

    st.subheader("Multi-Metric Radar Analysis")
    rep = st.selectbox("Select Rep for Radar", data['Sales_Rep_Name'].unique(), key="radar_persp_selectbox")
    rep_data = data[data['Sales_Rep_Name'] == rep].iloc[0]
    fig_radar = px.line_polar(r=[rep_data['Calls_Dialed'], rep_data['Call_Time_Mins'], rep_data['Deals_Closed']], 
                              theta=['Calls', 'TalkTime', 'Deals'], line_close=True)
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True, key="persp_radar_chart")

    st.subheader("Revenue Contribution Waterfall")
    fig_water = go.Figure(go.Waterfall(
        name="Revenue", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["New Leads", "Qualified", "Converted", "Total Revenue"],
        y=[0, data['Qualified'].sum(), data['Converted'].sum(), data['Total_Revenue'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    st.plotly_chart(fig_water, use_container_width=True, key="persp_waterfall_chart")

    st.subheader("Cohort Analysis: Revenue by Region")
    cohort_data = data.groupby('Region')['Total_Revenue'].sum().reset_index()
    fig_cohort = px.bar(cohort_data, x='Region', y='Total_Revenue', color='Region', title="Revenue by Region Cohort")
    st.plotly_chart(fig_cohort, use_container_width=True, key="persp_cohort_chart")

    st.subheader("Cumulative Revenue Contribution")
    area_data = data.sort_values('Total_Revenue').reset_index()
    fig_area = px.area(area_data, x=area_data.index, y='Total_Revenue', color='Region', title="Cumulative Revenue by Region")
    st.plotly_chart(fig_area, use_container_width=True, key="persp_area_chart")

# 5. PREDICTIVE
with tabs[4]:
    st.header("Predictive Analysis")
    st.subheader("Benchmark: Rep Performance vs Team Average")
    avg_rev = data['Total_Revenue'].mean()
    fig_bench_pred = px.bar(data, x='Sales_Rep_Name', y='Total_Revenue', title="Revenue per Rep vs Team Average")
    fig_bench_pred.add_hline(y=avg_rev, line_dash="dash", line_color="red", annotation_text="Team Average")
    st.plotly_chart(fig_bench_pred, use_container_width=True, key="pred_bench_chart")

    st.subheader("Multi-Metric Radar Analysis")
    rep = st.selectbox("Select Rep for Radar", data['Sales_Rep_Name'].unique(), key="radar_pred_selectbox")
    rep_data = data[data['Sales_Rep_Name'] == rep].iloc[0
