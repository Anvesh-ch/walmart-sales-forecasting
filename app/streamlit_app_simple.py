import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏪 Walmart Sales Forecasting Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Dashboard Controls")

# Sample data for demonstration
@st.cache_data
def create_sample_data():
    """Create sample data for demonstration purposes."""
    np.random.seed(42)
    
    # Sample stores and departments
    stores = [f"Store_{i:03d}" for i in range(1, 21)]
    depts = [f"Dept_{i:02d}" for i in range(1, 11)]
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='W')
    
    # Generate sample sales data
    data = []
    for store in stores:
        for dept in depts:
            for date in dates:
                base_sales = np.random.normal(1000, 200)
                holiday_boost = 1.2 if date.month in [11, 12] else 1.0  # Holiday season boost
                sales = max(0, base_sales * holiday_boost + np.random.normal(0, 50))
                
                data.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date,
                    'Sales': sales,
                    'IsHoliday': 1 if date.month in [11, 12] else 0
                })
    
    return pd.DataFrame(data)

# Load sample data
df = create_sample_data()

# Sidebar filters
st.sidebar.subheader("🔍 Filters")

# Store filter
selected_store = st.sidebar.selectbox(
    "Select Store:",
    options=['All Stores'] + sorted(df['Store'].unique().tolist()),
    index=0
)

# Department filter
if selected_store == 'All Stores':
    dept_options = sorted(df['Dept'].unique().tolist())
else:
    dept_options = sorted(df[df['Store'] == selected_store]['Dept'].unique().tolist())

selected_dept = st.sidebar.selectbox(
    "Select Department:",
    options=['All Departments'] + dept_options,
    index=0
)

# Date range filter
st.sidebar.subheader("📅 Date Range")
date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Filter data based on selections
filtered_df = df.copy()

if selected_store != 'All Stores':
    filtered_df = filtered_df[filtered_df['Store'] == selected_store]

if selected_dept != 'All Departments':
    filtered_df = filtered_df[filtered_df['Dept'] == selected_dept]

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['Date'] >= pd.Timestamp(start_date)) & 
        (filtered_df['Date'] <= pd.Timestamp(end_date))
    ]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Sales Overview")
    
    # Sales trend chart
    if not filtered_df.empty:
        fig = px.line(
            filtered_df.groupby('Date')['Sales'].sum().reset_index(),
            x='Date',
            y='Sales',
            title=f"Sales Trend - {selected_store} - {selected_dept}",
            labels={'Sales': 'Total Sales ($)', 'Date': 'Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

with col2:
    st.subheader("📊 Summary Statistics")
    
    if not filtered_df.empty:
        total_sales = filtered_df['Sales'].sum()
        avg_sales = filtered_df['Sales'].mean()
        max_sales = filtered_df['Sales'].max()
        min_sales = filtered_df['Sales'].min()
        
        st.metric("Total Sales", f"${total_sales:,.0f}")
        st.metric("Average Sales", f"${avg_sales:,.0f}")
        st.metric("Max Sales", f"${max_sales:,.0f}")
        st.metric("Min Sales", f"${min_sales:,.0f}")
    else:
        st.info("Select filters to view statistics.")

# Store performance comparison
st.subheader("🏪 Store Performance Comparison")
if not filtered_df.empty:
    store_performance = filtered_df.groupby('Store')['Sales'].sum().sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=store_performance.values,
        y=store_performance.index,
        orientation='h',
        title="Top 10 Stores by Total Sales",
        labels={'x': 'Total Sales ($)', 'y': 'Store'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Department analysis
st.subheader("📦 Department Analysis")
if not filtered_df.empty:
    dept_performance = filtered_df.groupby('Dept')['Sales'].sum().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=dept_performance.values,
            names=dept_performance.index,
            title="Sales Distribution by Department"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=dept_performance.index,
            y=dept_performance.values,
            title="Sales by Department",
            labels={'x': 'Department', 'y': 'Total Sales ($)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Holiday impact analysis
st.subheader("🎉 Holiday Impact Analysis")
if not filtered_df.empty:
    holiday_analysis = filtered_df.groupby('IsHoliday')['Sales'].agg(['mean', 'count']).round(2)
    holiday_analysis.index = ['Non-Holiday', 'Holiday']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sales Comparison:**")
        st.dataframe(holiday_analysis)
    
    with col2:
        fig = px.bar(
            x=['Non-Holiday', 'Holiday'],
            y=holiday_analysis['mean'],
            title="Average Sales: Holiday vs Non-Holiday",
            labels={'x': 'Period', 'y': 'Average Sales ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 <strong>Walmart Sales Forecasting Dashboard</strong> | Built with Streamlit</p>
        <p>📊 This is a demonstration dashboard. Add your actual data to see real forecasts.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Information about the full app
st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ **About This Demo**")
st.sidebar.markdown("""
This is a simplified version of the Walmart Sales Forecasting dashboard.

**To use the full version:**
1. Add your Walmart CSV files to `data/raw/`
2. Run the complete pipeline: `make all`
3. Use the main `streamlit_app.py`

**Current Status:** Demo mode with sample data
""")

if __name__ == "__main__":
    pass
