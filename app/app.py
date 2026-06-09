import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Food Demand & Inventory Dashboard",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    orders      = pd.read_csv('data/restaurant_orders.csv')
    weekly      = pd.read_csv('data/weekly_demand_summary.csv')
    waste       = pd.read_csv('data/waste_analysis.csv')
    alerts      = pd.read_csv('data/predictions/crisis_alerts.csv')
    procurement = pd.read_csv('data/predictions/procurement_orders.csv')
    prophet     = pd.read_csv('data/predictions/prophet_forecast.csv')
    xgb         = pd.read_csv('data/predictions/xgboost_predictions.csv')
    return orders, weekly, waste, alerts, procurement, prophet, xgb

orders, weekly, waste, alerts, procurement, prophet, xgb = load_data()
orders_clean = orders.drop_duplicates(subset='order_id')

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("---")

branches = st.sidebar.multiselect(
    "Branch",
    options=orders_clean['restaurant_branch'].unique(),
    default=orders_clean['restaurant_branch'].unique()
)
dish_filter = st.sidebar.selectbox(
    "Dish (for forecast)",
    sorted(orders_clean['dish_name'].unique())
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Run order:**")
st.sidebar.code("01 → 02 → 03 → 04 → 05 → 06")

filtered = orders_clean[orders_clean['restaurant_branch'].isin(branches)]

# ── Title ──────────────────────────────────────────────────────
st.title("Food Demand Prediction & Smart Inventory Dashboard")
st.caption("AI-powered demand forecasting and inventory management system")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Demand Forecast",
    "Inventory Alerts",
    "Procurement",
    "Waste Analysis"
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Business Overview")
    st.caption("Key performance metrics across all selected branches")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders",     f"{len(filtered):,}")
    k2.metric("Total Revenue",    f"${filtered['total_price'].sum():,.2f}")
    k3.metric("Avg Rating",       f"{filtered['customer_rating'].mean():.2f}")
    k4.metric("Cancelled Orders", f"{(filtered['order_status']=='Cancelled').sum()}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orders by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday',
                     'Thursday','Friday','Saturday','Sunday']
        day_data = (filtered['day_of_week']
                    .value_counts()
                    .reindex(day_order)
                    .reset_index())
        day_data.columns = ['Day', 'Orders']
        fig1 = px.bar(day_data, x='Day', y='Orders',
                      color='Orders', color_continuous_scale='Purples',
                      title='Which days get the most orders?')
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Revenue by Branch")
        branch_rev = (filtered.groupby('restaurant_branch')['total_price']
                      .sum().reset_index())
        branch_rev.columns = ['Branch', 'Revenue']
        fig2 = px.pie(branch_rev, values='Revenue', names='Branch',
                      title='Revenue split across branches',
                      color_discrete_sequence=px.colors.sequential.Purples_r)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Orders by Category")
        cat_data = filtered['dish_category'].value_counts().reset_index()
        cat_data.columns = ['Category', 'Orders']
        fig3 = px.bar(cat_data, x='Category', y='Orders',
                      color='Orders', color_continuous_scale='Teal',
                      title='Most popular dish categories')
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Payment Methods")
        pay_data = filtered['payment_method'].value_counts().reset_index()
        pay_data.columns = ['Method', 'Count']
        fig4 = px.pie(pay_data, values='Count', names='Method',
                      title='How customers pay',
                      color_discrete_sequence=px.colors.sequential.Teal_r)
        st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — DEMAND FORECAST
# ──────────────────────────────────────────────────────────────
with tab2:
    st.header("Demand Forecasting")
    st.caption("Predicted vs actual orders using XGBoost + Prophet time series models")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("XGBoost — Predicted vs Actual")
        st.caption("Each dot is one order. Dots close to the dashed line = accurate prediction.")
        fig5 = px.scatter(xgb, x='actual_qty', y='predicted_qty',
                          labels={'actual_qty': 'Actual Quantity',
                                  'predicted_qty': 'Predicted Quantity'},
                          color_discrete_sequence=['#7F77DD'], opacity=0.6)
        fig5.add_shape(type='line', x0=0, y0=0, x1=5, y1=5,
                       line=dict(dash='dash', color='gray', width=1))
        fig5.add_annotation(x=4.5, y=4.2,
                            text="Perfect prediction line",
                            showarrow=False,
                            font=dict(size=10, color='gray'))
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader(f"Prophet Forecast — {dish_filter}")
        st.caption("Teal line = predicted demand. Shaded area = confidence range.")
        dish_data = prophet[prophet['dish_name'] == dish_filter].copy()
        dish_data['ds'] = pd.to_datetime(dish_data['ds'])
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=dish_data['ds'], y=dish_data['yhat'],
            name='Forecast', line=dict(color='#1D9E75', width=2)
        ))
        fig6.add_trace(go.Scatter(
            x=dish_data['ds'], y=dish_data['yhat_upper'],
            fill=None, mode='lines',
            line=dict(width=0), showlegend=False
        ))
        fig6.add_trace(go.Scatter(
            x=dish_data['ds'], y=dish_data['yhat_lower'],
            fill='tonexty', mode='lines', line=dict(width=0),
            fillcolor='rgba(29,158,117,0.15)', name='Confidence range'
        ))
        fig6.update_layout(
            xaxis_title='Date',
            yaxis_title='Predicted Orders',
            legend=dict(orientation='h', y=-0.2)
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.subheader("Weekly Demand Heatmap")
    st.caption("Darker purple = more orders that week. Use this to spot seasonal trends.")
    pivot = weekly.pivot_table(
        index='dish_name', columns='week_number',
        values='total_qty', aggfunc='sum'
    ).fillna(0)
    fig7 = px.imshow(pivot, aspect='auto',
                     color_continuous_scale='Purples',
                     labels=dict(x='Week Number',
                                 y='Dish', color='Orders'))
    fig7.update_layout(height=500)
    st.plotly_chart(fig7, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 3 — INVENTORY ALERTS
# ──────────────────────────────────────────────────────────────
with tab3:
    st.header("Smart Inventory Crisis Alerts")
    st.caption("AI-generated alerts based on predicted weekly demand vs current stock levels")

    red_count    = len(alerts[alerts['alert_status'] == 'RED'])
    yellow_count = len(alerts[alerts['alert_status'] == 'YELLOW'])
    green_count  = len(alerts[alerts['alert_status'] == 'GREEN'])

    m1, m2, m3 = st.columns(3)
    m1.metric("Order Immediately",   red_count)
    m2.metric("Surplus / Spoilage Risk", yellow_count)
    m3.metric("Stock OK",            green_count)

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Alert Distribution")
        counts = alerts['alert_status'].value_counts().reset_index()
        counts.columns = ['Status', 'Count']
        color_map = {
            'RED':    '#E24B4A',
            'YELLOW': '#EF9F27',
            'GREEN':  '#639922'
        }
        fig8 = px.pie(counts, values='Count', names='Status',
                      color='Status', color_discrete_map=color_map,
                      hole=0.4)
        fig8.update_traces(textposition='inside',
                           textinfo='percent+label')
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.subheader("Ingredient Alert Details")
        st.caption("Sorted by days of stock left — most urgent at top")

        def highlight_alert(val):
            colors = {
                'RED':    'background-color: #E24B4A; color: white; font-weight: bold',
                'YELLOW': 'background-color: #EF9F27; color: white; font-weight: bold',
                'GREEN':  'background-color: #639922; color: white; font-weight: bold'
            }
            return colors.get(val, '')

        display_cols = ['ingredient_name', 'alert_status', 'alert_reason',
                        'closing_stock_kg', 'adjusted_burn_kg',
                        'days_of_stock_left']
        display_cols = [c for c in display_cols if c in alerts.columns]

        styled = (alerts[display_cols]
                  .sort_values('days_of_stock_left')
                  .style.applymap(highlight_alert,
                                  subset=['alert_status']))
        st.dataframe(styled, use_container_width=True, height=350)

# ──────────────────────────────────────────────────────────────
# TAB 4 — PROCUREMENT
# ──────────────────────────────────────────────────────────────
with tab4:
    st.header("Auto-Generated Procurement Orders")
    st.caption("Draft purchase orders created automatically based on crisis alerts")

    total_cost   = procurement['estimated_cost_usd'].sum()
    urgent_count = len(procurement[procurement['priority'] == '1 - Urgent'])
    normal_count = len(procurement[procurement['priority'] == '2 - Normal'])

    p1, p2, p3 = st.columns(3)
    p1.metric("Total Estimated Cost", f"${total_cost:,.2f}")
    p2.metric("Urgent Orders",        urgent_count)
    p3.metric("Normal Orders",        normal_count)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost by Supplier")
        if 'supplier' in procurement.columns:
            sup_cost = (procurement.groupby('supplier')['estimated_cost_usd']
                        .sum().reset_index())
            sup_cost.columns = ['Supplier', 'Cost']
            fig9 = px.bar(
                sup_cost.sort_values('Cost', ascending=False),
                x='Supplier', y='Cost',
                color='Cost', color_continuous_scale='Reds',
                title='How much to spend per supplier'
            )
            fig9.update_layout(showlegend=False)
            st.plotly_chart(fig9, use_container_width=True)

    with col2:
        st.subheader("Urgent vs Normal Orders")
        pri_data = procurement['priority'].value_counts().reset_index()
        pri_data.columns = ['Priority', 'Count']
        fig10 = px.pie(pri_data, values='Count', names='Priority',
                       color_discrete_sequence=['#E24B4A', '#EF9F27'],
                       title='Order urgency breakdown')
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Procurement Order List")
    st.caption("These are draft orders — review before placing with suppliers")

    display_proc = ['ingredient_name', 'supplier', 'priority',
                    'order_qty_kg', 'estimated_cost_usd',
                    'order_date', 'expected_delivery', 'order_status']
    display_proc = [c for c in display_proc if c in procurement.columns]
    st.dataframe(
        procurement[display_proc].sort_values('priority'),
        use_container_width=True, height=350
    )

# ──────────────────────────────────────────────────────────────
# TAB 5 — WASTE ANALYSIS
# ──────────────────────────────────────────────────────────────
with tab5:
    st.header("Food Waste Analysis")
    st.caption("Identify which dishes generate the most waste to reduce food costs")

    w1, w2, w3 = st.columns(3)
    w1.metric("Total Waste (kg)",  f"{waste['total_waste_kg'].sum():.2f} kg")
    w2.metric("High Risk Dishes",  len(waste[waste['waste_reduction_opportunity'] == 'High']))
    w3.metric("Low Risk Dishes",   len(waste[waste['waste_reduction_opportunity'] == 'Low']))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Dishes by Waste")
        st.caption("Red = high waste risk, Orange = medium, Green = low")
        color_map2 = {
            'High':   '#E24B4A',
            'Medium': '#EF9F27',
            'Low':    '#639922'
        }
        fig11 = px.bar(
            waste.sort_values('total_waste_kg', ascending=False).head(10),
            x='dish_name', y='total_waste_kg',
            color='waste_reduction_opportunity',
            color_discrete_map=color_map2,
            labels={'dish_name': 'Dish',
                    'total_waste_kg': 'Total Waste (kg)',
                    'waste_reduction_opportunity': 'Risk Level'}
        )
        fig11.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        st.subheader("Waste vs Orders")
        st.caption("High orders + high waste = best opportunity to reduce costs")
        fig12 = px.scatter(
            waste, x='total_orders', y='total_waste_kg',
            color='waste_reduction_opportunity',
            color_discrete_map=color_map2,
            hover_name='dish_name',
            size='total_waste_kg',
            labels={'total_orders': 'Total Orders',
                    'total_waste_kg': 'Total Waste (kg)'}
        )
        st.plotly_chart(fig12, use_container_width=True)

    st.markdown("---")
    st.subheader("Waste from Cancelled Orders")
    st.caption("Cancellations contribute directly to food waste — reducing them saves cost")
    fig13 = px.bar(
        waste.sort_values('waste_from_cancellations_kg',
                          ascending=False).head(10),
        x='dish_name', y='waste_from_cancellations_kg',
        color_discrete_sequence=['#E24B4A'],
        labels={'dish_name': 'Dish',
                'waste_from_cancellations_kg': 'Waste from Cancellations (kg)'}
    )
    fig13.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig13, use_container_width=True)

st.markdown("---")
st.caption("Food Demand Prediction & Smart Inventory Dashboard | Built with Streamlit + Prophet + XGBoost")