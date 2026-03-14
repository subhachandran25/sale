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

# 2. DESCRIPTIVE
with tabs[1]:
    st.header("Descriptive Analysis")
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.pie(data, names='Sales_Rep_Name', values='Total_Revenue', title="Revenue Share by Rep"), use_container_width=True)
    # Using a bar chart for trend since we don't have a Date column
    col2.plotly_chart(px.bar(data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().reset_index(), x='Sales_Rep_Name', y='Total_Revenue', title="Revenue per Rep"), use_container_width=True)

# 3. DIAGNOSTIC
with tabs[2]:
    st.header("Diagnostic Analysis")
    col1, col2 = st.columns(2)
    pareto = data.groupby('Sales_Rep_Name')['Total_Revenue'].sum().sort_values(ascending=False).reset_index()
    col1.plotly_chart(px.bar(pareto, x='Sales_Rep_Name', y='Total_Revenue', title="Pareto: Revenue Contribution"), use_container_width=True)
    col2.plotly_chart(px.box(data, y='Call_Time_Mins', title="Talk Time Outliers"), use_container_width=True)
    
    # Stacked Bar
    df_stacked = data.groupby(['Sales_Manager_Name', 'Region'])['Deals_Closed'].sum().reset_index()
    st.plotly_chart(px.bar(df_stacked, x='Sales_Manager_Name', y='Deals_Closed', color='Region', title="Deals Closed by Manager & Region", barmode='stack'), use_container_width=True)

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
