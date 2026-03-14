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

# 1. HOME PAGE
# 1. HOME PAGE
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

    # --- MANAGER PERFORMANCE CHART: Connected Calls vs Deals Closed ---
    mgr_perf_home = data.groupby('Sales_Manager_Name')[['Converted','Deals_Closed']].sum().reset_index()
    fig_mgr_bar = px.bar(
        mgr_perf_home, 
        x='Sales_Manager_Name', 
        y=['Converted','Deals_Closed'], 
        barmode='group',
        title="Manager Performance: Connected Calls vs Deals Closed"
    )
    col1.plotly_chart(fig_mgr_bar, use_container_width=True, key="home_mgr_bar_chart")

    # --- LINE CHART: Revenue by Manager with Average Benchmark ---
    mgr_rev = data.groupby('Sales_Manager_Name')['Total_Revenue'].sum().reset_index()
    avg_rev = mgr_rev['Total_Revenue'].mean()
    fig_line = px.line(mgr_rev, x='Sales_Manager_Name', y='Total_Revenue', 
                       markers=True, title="Revenue by Manager with Team Average")
    fig_line.add_hline(y=avg_rev, line_dash="dot", line_color="red", annotation_text="Average Revenue")
    col2.plotly_chart(fig_line, use_container_width=True, key="home_mgr_line_chart")

    # --- PIE CHART: Revenue by Region ---
    st.subheader("Revenue Performance by Region")
    revenue_region = data.groupby('Region')['Total_Revenue'].sum().reset_index()
    fig_pie = px.pie(revenue_region, values='Total_Revenue', names='Region',
                     title="Revenue Distribution Across Regions", hole=0.3)
    st.plotly_chart(fig_pie, use_container_width=True, key="home_pie_chart")



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

    # --- Donut Chart: Call Outcomes (Connected vs Not Connected) ---
    st.subheader("Call Outcomes by Manager")
    call_outcomes = data.groupby('Sales_Manager_Name')[['Calls_Dialed','Converted']].sum().reset_index()
    call_outcomes['Not_Connected'] = call_outcomes['Calls_Dialed'] - call_outcomes['Converted']

    outcomes_long = call_outcomes.melt(
        id_vars=['Sales_Manager_Name'],
        value_vars=['Converted','Not_Connected'],
        var_name='Outcome',
        value_name='Count'
    )

    fig_donut = px.pie(
        outcomes_long,
        values='Count',
        names='Outcome',
        color='Outcome',
        title="Dialed Calls Breakdown (Connected vs Not Connected)",
        hole=0.4
    )
    st.plotly_chart(fig_donut, use_container_width=True, key="desc_donut_chart")

    # --- Grouped Bar Chart: Deals Closed by Region (Manager Level) ---
    st.subheader("Deals Closed by Manager across Regions")
    deals_group = data.groupby(['Sales_Manager_Name','Region'])['Deals_Closed'].sum().reset_index()
    fig_grouped = px.bar(
        deals_group,
        x='Sales_Manager_Name',
        y='Deals_Closed',
        color='Region',
        barmode='group',
        title="Deals Closed by Manager across Regions"
    )
    st.plotly_chart(fig_grouped, use_container_width=True, key="desc_grouped_chart")

    # --- Sunburst Chart: Revenue Contribution (stop at Manager level) ---
    st.subheader("Revenue Contribution Sunburst")
    sunburst_data = data.groupby(['Region','Sales_Manager_Name'])['Total_Revenue'].sum().reset_index()
    fig_sunburst = px.sunburst(
        sunburst_data,
        path=['Region','Sales_Manager_Name'],
        values='Total_Revenue',
        color='Region',
        title="Revenue Contribution by Region → Manager"
    )
    st.plotly_chart(fig_sunburst, use_container_width=True, key="desc_sunburst_chart")

    # --- Line Chart: Talk Time Distribution by Manager ---
    st.subheader("Average Talk Time by Manager")
    talktime_mgr = data.groupby('Sales_Manager_Name')['Call_Time_Mins'].mean().reset_index()
    avg_talk = talktime_mgr['Call_Time_Mins'].mean()
    fig_line = px.line(
        talktime_mgr,
        x='Sales_Manager_Name',
        y='Call_Time_Mins',
        markers=True,
        title="Average Talk Time per Manager"
    )
    fig_line.add_hline(
        y=avg_talk,
        line_dash="dot",
        line_color="red",
        annotation_text="Overall Average"
    )
    st.plotly_chart(fig_line, use_container_width=True, key="desc_line_chart")

    # --- Bubble Chart: Talk Time vs Revenue ---
    st.subheader("Talk Time vs Revenue (Bubble Size = Deals Closed)")
    fig_bubble = px.scatter(
        data,
        x='Call_Time_Mins',
        y='Total_Revenue',
        size='Deals_Closed',
        color='Sales_Manager_Name',
        hover_name='Sales_Manager_Name',
        title="Talk Time vs Revenue by Manager"
    )
    st.plotly_chart(fig_bubble, use_container_width=True, key="desc_bubble_chart")

    # --- Summary Metrics Table ---
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
    rep_data = data[data['Sales_Rep_Name'] == rep].iloc[0]   # ✅ fixed bracket
    fig_radar_pred = px.line_polar(
        r=[rep_data['Calls_Dialed'], rep_data['Call_Time_Mins'], rep_data['Deals_Closed']], 
        theta=['Calls', 'TalkTime', 'Deals'], line_close=True
    )
    fig_radar_pred.update_traces(fill='toself')
    st.plotly_chart(fig_radar_pred, use_container_width=True, key="pred_radar_chart")

    st.subheader("Revenue Contribution Waterfall")
    fig_water_pred = go.Figure(go.Waterfall(
        name="Revenue", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["New Leads", "Qualified", "Converted", "Total Revenue"],
        y=[0, data['Qualified'].sum(), data['Converted'].sum(), data['Total_Revenue'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    st.plotly_chart(fig_water_pred, use_container_width=True, key="pred_waterfall_chart")

    st.subheader("Cohort Analysis: Revenue by Region")
    cohort_data_pred = data.groupby('Region')['Total_Revenue'].sum().reset_index()
    fig_cohort_pred = px.bar(cohort_data_pred, x='Region', y='Total_Revenue', color='Region', title="Revenue by Region Cohort")
    st.plotly_chart(fig_cohort_pred, use_container_width=True, key="pred_cohort_chart")

    st.subheader("Cumulative Revenue Contribution")
    area_data_pred = data.sort_values('Total_Revenue').reset_index()
    fig_area_pred = px.area(area_data_pred, x=area_data_pred.index, y='Total_Revenue', color='Region', title="Cumulative Revenue by Region")
    st.plotly_chart(fig_area_pred, use_container_width=True, key="pred_area_chart")

    # Predictive model
    X = data[['Calls_Dialed', 'Call_Time_Mins', 'Converted']]
    y = data['Total_Revenue']
    model = DecisionTreeRegressor().fit(X, y)

    fig_reg = px.scatter(data, x='Call_Time_Mins', y='Total_Revenue', trendline="ols", title="Revenue vs Talk Time")
    st.plotly_chart(fig_reg, use_container_width=True, key="pred_reg_chart")

    inc = st.slider("Increase Dialed Calls by %", 0, 50, 10, key="sim_slider")
    st.metric("Projected Revenue Lift", f"₹{(data['Total_Revenue'].sum() * (inc/100)):,.0f}")

