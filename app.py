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
    col1.plotly_chart(px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], title="Dialed vs Converted"), use_container_width=True)
    col2.plotly_chart(px.histogram(data, x='Call_Time_Mins', title="Talk Time Distribution"), use_container_width=True)

# --- NEW TAB: MANAGER PERFORMANCE ---
with st.expander("Region-wise Manager Performance View"):
    st.header("Manager Performance by Region")
    
    # Filter for specific region
    selected_mgr_region = st.selectbox("Select Region for Manager View", df['Region'].unique(), key="mgr_view_reg")
    mgr_data = df[df['Region'] == selected_mgr_region]
    
    # Group by Manager
    mgr_perf = mgr_data.groupby('Sales_Manager_Name')[['Total_Revenue', 'Deals_Closed']].sum().reset_index()
    
    st.dataframe(mgr_perf, use_container_width=True)
    st.plotly_chart(px.bar(mgr_perf, x='Sales_Manager_Name', y='Total_Revenue', title=f"Revenue by Managers in {selected_mgr_region}"), use_container_width=True)

# 2. DESCRIPTIVE
with tabs[1]:
    st.header("Descriptive Analysis")
    
    # Row 1: Call Efficiency & Talk Time
    col1, col2 = st.columns(2)
    
    # 1. Bar Chart: Dialed vs Connected Calls (Efficiency)
    # We use 'barmode="group"' to compare the two side-by-side
    fig_eff = px.bar(data, x='Sales_Rep_Name', y=['Calls_Dialed', 'Converted'], 
                     title="Call Efficiency: Dialed vs Converted Calls",
                     barmode='group')
    col1.plotly_chart(fig_eff, use_container_width=True)
    
    # 2. Histogram: Talk Time Distribution
    fig_hist = px.histogram(data, x='Call_Time_Mins', nbins=20, 
                            title="Talk Time Distribution (Outlier Detection)",
                            color_discrete_sequence=['#636EFA'])
    col2.plotly_chart(fig_hist, use_container_width=True)
    
    # Row 2: Conversion Funnel
    st.subheader("Conversion Funnel")
    
    # 3. Conversion Funnel: Dialed -> Qualified -> Converted -> Deals Closed
    # We create a temporary dataframe for the funnel stages
    funnel_df = pd.DataFrame({
        "Stage": ["Dialed", "Qualified", "Converted", "Deals Closed"],
        "Count": [data['Calls_Dialed'].sum(), data['Qualified'].sum(), 
                  data['Converted'].sum(), data['Deals_Closed'].sum()]
    })
    
    fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', title="Sales Conversion Funnel")
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # KPI Table
    st.subheader("Summary Metrics Table")
    st.dataframe(data.describe(), use_container_width=True)
# 3. DIAGNOSTIC
with tabs[1]:
    st.header("Descriptive Analysis")
    
    # 1. FORCE DATA TYPES (This fixes the "Charts not showing" issue)
    # This ensures numbers are actually numbers
    numeric_cols = ['Total_Revenue', 'Calls_Dialed', 'Converted', 'Deals_Closed', 'Call_Time_Mins']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # 2. DEBUG: Check if data is actually there
    st.write(f"Rows in current view: {len(data)}")
    if len(data) == 0:
        st.error("No data found for this selection. Check your filters.")
    else:
        # 3. PARETO CHART
        st.subheader("Pareto: Revenue Contribution")
        pareto_data = data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().sort_values(ascending=False).reset_index()
        fig_pareto = px.bar(pareto_data, x='Sales_Rep_Name', y='Total_Revenue')
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # 4. BOX PLOT
        st.subheader("Talk Time Outliers")
        fig_box = px.box(data, y='Call_Time_Mins')
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 5. STACKED BAR (Dialed vs Converted)
        st.subheader("Call Efficiency")
        # We melt the data to make it stackable
        df_melted = data.melt(id_vars=['Sales_Rep_Name'], value_vars=['Calls_Dialed', 'Converted'])
        fig_stacked = px.bar(df_melted, x='Sales_Rep_Name', y='value', color='variable', barmode='stack')
        st.plotly_chart(fig_stacked, use_container_width=True)

# 4. PERSPECTIVE
with tabs[3]:
    st.header("Perspective & Benchmarks")
    rep = st.selectbox("Select Rep for Radar", data['Sales_Rep_Name'].unique(), key="radar_rep")
    rep_data = data[data['Sales_Rep_Name'] == rep].iloc[0]
    fig = px.line_polar(r=[rep_data['Calls_Dialed'], rep_data['Call_Time_Mins'], rep_data['Deals_Closed']], 
                        theta=['Calls', 'TalkTime', 'Deals'], line_close=True)
    st.plotly_chart(fig, use_container_width=True)

# 5. PREDICTIVE
with tabs[4]:
    st.header("Predictive Analysis")
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)
    st.plotly_chart(px.scatter(data, x='Call_Time_Mins', y='Total_Revenue', trendline="ols", title="Revenue vs Talk Time"), use_container_width=True)
    
    # Scenario Simulation
    inc = st.slider("Increase Dialed Calls by %", 0, 50, 10, key="sim_slider")
    st.metric("Projected Revenue Lift", f"₹{(data['Total_Revenue'].sum() * (inc/100)):,.0f}")
