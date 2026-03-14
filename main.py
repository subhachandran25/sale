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
    st.plotly_chart(fig_donut, use_container_width=True, key="desc_outcomes_donut")

    # --- Grouped Bar Chart: Deals Closed by Region ---
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
    st.plotly_chart(fig_grouped, use_container_width=True, key="desc_deals_grouped")

    # --- Revenue Contribution: Region → Manager ---
    st.subheader("Revenue Contribution by Region and Manager")
    revenue_group = data.groupby(['Region','Sales_Manager_Name'])['Total_Revenue'].sum().reset_index()
    fig_revenue = px.bar(
        revenue_group,
        x='Sales_Manager_Name',
        y='Total_Revenue',
        color='Region',
        barmode='stack',
        title="Revenue Contribution by Manager across Regions"
    )
    st.plotly_chart(fig_revenue, use_container_width=True, key="desc_revenue_bar")

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
    st.plotly_chart(fig_line, use_container_width=True, key="desc_talktime_line")

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
    st.plotly_chart(fig_bubble, use_container_width=True, key="desc_talktime_bubble")

    # --- Lead Funnel Chart across Regions ---
    st.subheader("Lead Funnel by Region")
    funnel_data = data.groupby('Region')[['New_Leads','Followup_Leads','Qualified','Disqualified']].sum().reset_index()
    funnel_long = funnel_data.melt(
        id_vars=['Region'],
        value_vars=['New_Leads','Followup_Leads','Qualified','Disqualified'],
        var_name='Stage',
        value_name='Count'
    )
    fig_funnel = px.funnel(
        funnel_long,
        x='Count',
        y='Stage',
        color='Region',
        title="Lead Funnel across Regions"
    )
    st.plotly_chart(fig_funnel, use_container_width=True, key="desc_lead_funnel")

    # --- Summary Metrics Table ---
    st.subheader("Summary Metrics Table")
    st.dataframe(data.describe(), use_container_width=True)


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

    # --- Forecast Line Chart: Revenue Projection ---
    st.subheader("Revenue Forecast")
    # If you have a real date column (e.g. 'Month'), replace 'Period' with that
    revenue_ts = data.groupby('Sales_Manager_Name')['Total_Revenue'].sum().reset_index()
    revenue_ts['Period'] = range(1, len(revenue_ts)+1)

    fig_forecast = px.line(
        revenue_ts,
        x='Period',
        y='Total_Revenue',
        markers=True,
        title="Time Series Revenue Projection by Manager"
    )
    st.plotly_chart(fig_forecast, use_container_width=True, key="pred_forecast_line")

    # --- Regression Scatter Plot: Closures vs Talk Time & Connected Calls ---
    st.subheader("Regression Analysis: Closures vs Talk Time & Connected Calls")
    fig_reg = px.scatter(
        data,
        x='Call_Time_Mins',
        y='Deals_Closed',
        size='Converted',
        color='Sales_Manager_Name',
        hover_name='Sales_Manager_Name',
        trendline="ols",
        title="Predicting Closures from Talk Time & Connected Calls"
    )
    st.plotly_chart(fig_reg, use_container_width=True, key="pred_regression_scatter")

    # --- Forecast Funnel Chart: Expected Conversion Rates ---
    st.subheader("Forecast Funnel")
    funnel_df = pd.DataFrame({
        "Stage": ["Dialed", "Qualified", "Converted", "Deals Closed"],
        "Expected": [
            data['Calls_Dialed'].sum() * 1.1,   # assume 10% growth
            data['Qualified'].sum() * 1.1,
            data['Converted'].sum() * 1.1,
            data['Deals_Closed'].sum() * 1.1
        ]
    })
    fig_funnel = px.funnel(funnel_df, x='Expected', y='Stage', title="Expected Conversion Funnel")
    st.plotly_chart(fig_funnel, use_container_width=True, key="pred_forecast_funnel")

    # --- Scenario Simulation Chart: What-if Analysis ---
    st.subheader("Scenario Simulation: Impact of More Calls/Talk Time")
    inc_calls = st.slider("Increase Dialed Calls by %", 0, 50, 10, key="sim_calls_slider")
    inc_talk = st.slider("Increase Talk Time by %", 0, 50, 10, key="sim_talk_slider")

    base_rev = data['Total_Revenue'].sum()
    sim_df = pd.DataFrame({
        "Scenario": ["Base Revenue","With More Calls","With More Talk Time"],
        "Revenue": [
            base_rev,
            base_rev * (1 + inc_calls/100),
            base_rev * (1 + inc_talk/100)
        ]
    })
    fig_sim = px.bar(sim_df, x='Scenario', y='Revenue', title="What-if Revenue Simulation")
    st.plotly_chart(fig_sim, use_container_width=True, key="pred_simulation_chart")

    projected_rev = base_rev * (1 + (inc_calls + inc_talk)/200)
    st.metric("Projected Combined Revenue", f"₹{projected_rev:,.0f}")

