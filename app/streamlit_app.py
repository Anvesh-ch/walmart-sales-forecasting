import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Walmart Sales Forecasting Dashboard")
st.markdown("View sales forecasts, store risk analysis, and markdown ROI insights")

@st.cache_data
def load_data():
    """Load dashboard data from exports directory."""
    exports_dir = Path("exports")
    
    data = {}
    
    # Load forecasts data
    forecasts_path = exports_dir / "forecasts_store_dept.csv"
    if forecasts_path.exists():
        data['forecasts'] = pd.read_csv(forecasts_path)
        data['forecasts']['Date'] = pd.to_datetime(data['forecasts']['Date'])
    else:
        st.warning("Forecasts data not found. Please run the export_dashboard pipeline first.")
        data['forecasts'] = pd.DataFrame()
    
    # Load store risk data
    risk_path = exports_dir / "store_risk.csv"
    if risk_path.exists():
        data['risk'] = pd.read_csv(risk_path)
    else:
        data['risk'] = pd.DataFrame()
    
    # Load markdown ROI data
    roi_path = exports_dir / "markdown_roi.csv"
    if roi_path.exists():
        data['roi'] = pd.read_csv(roi_path)
    else:
        data['roi'] = pd.DataFrame()
    
    # Load dashboard summary
    summary_path = exports_dir / "dashboard_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data['summary'] = json.load(f)
    else:
        data['summary'] = {}
    
    return data

def main():
    """Main dashboard application."""
    
    # Load data
    data = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Model filter
    if not data['forecasts'].empty:
        available_models = data['forecasts']['model'].unique()
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=available_models,
            index=0 if len(available_models) > 0 else None
        )
        
        # Store filter
        available_stores = sorted(data['forecasts']['Store'].unique())
        selected_store = st.sidebar.selectbox(
            "Select Store",
            options=available_stores,
            index=0 if len(available_stores) > 0 else None
        )
        
        # Department filter
        if selected_store:
            store_depts = data['forecasts'][data['forecasts']['Store'] == selected_store]['Dept'].unique()
            available_depts = sorted(store_depts)
            selected_dept = st.sidebar.selectbox(
                "Select Department",
                options=available_depts,
                index=0 if len(available_depts) > 0 else None
            )
        else:
            selected_dept = None
        
        # Date range filter
        if not data['forecasts'].empty:
            min_date = data['forecasts']['Date'].min()
            max_date = data['forecasts']['Date'].max()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None
    else:
        selected_model = None
        selected_store = None
        selected_dept = None
        date_range = None
    
    # Main content
    if data['forecasts'].empty:
        st.error("No data available. Please ensure the export_dashboard pipeline has been run.")
        return
    
    # Dashboard summary
    if data['summary']:
        st.header("Dashboard Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stores", data['summary'].get('total_stores', 'N/A'))
        with col2:
            st.metric("Total Departments", data['summary'].get('total_departments', 'N/A'))
        with col3:
            st.metric("Total Observations", f"{data['summary'].get('total_observations', 'N/A'):,}")
        with col4:
            st.metric("Total Revenue", f"${data['summary'].get('total_revenue', 'N/A'):,.0f}")
    
    # Forecasts vs Actuals
    st.header("Forecasts vs Actuals")
    
    if selected_store and selected_dept and selected_model:
        # Filter data for selected store, department, and model
        filtered_data = data['forecasts'][
            (data['forecasts']['Store'] == selected_store) &
            (data['forecasts']['Dept'] == selected_dept) &
            (data['forecasts']['model'] == selected_model)
        ].copy()
        
        if not filtered_data.empty:
            # Create forecast vs actual plot
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['y_true'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['y_pred'],
                mode='lines+markers',
                name='Predicted Sales',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Confidence intervals if available
            if 'lower' in filtered_data.columns and 'upper' in filtered_data.columns:
                fig.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data['upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data['lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Confidence Interval',
                    line=dict(width=0),
                    showlegend=False
                ))
            
            # Holiday markers
            if 'IsHoliday' in filtered_data.columns:
                holiday_data = filtered_data[filtered_data['IsHoliday'] == True]
                if not holiday_data.empty:
                    fig.add_trace(go.Scatter(
                        x=holiday_data['Date'],
                        y=holiday_data['y_true'],
                        mode='markers',
                        name='Holiday Weeks',
                        marker=dict(color='orange', size=10, symbol='star'),
                        showlegend=True
                    ))
            
            fig.update_layout(
                title=f"Sales Forecasts vs Actuals - Store {selected_store}, Dept {selected_dept} ({selected_model})",
                xaxis_title="Date",
                yaxis_title="Weekly Sales ($)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics for selected series
            col1, col2, col3 = st.columns(3)
            with col1:
                mae = np.mean(np.abs(filtered_data['y_true'] - filtered_data['y_pred']))
                st.metric("MAE", f"${mae:,.0f}")
            with col2:
                rmse = np.sqrt(np.mean((filtered_data['y_true'] - filtered_data['y_pred'])**2))
                st.metric("RMSE", f"${rmse:,.0f}")
            with col3:
                mape = np.mean(np.abs((filtered_data['y_true'] - filtered_data['y_pred']) / filtered_data['y_true'])) * 100
                st.metric("MAPE", f"{mape:.1f}%")
        else:
            st.warning(f"No data available for Store {selected_store}, Dept {selected_dept}, Model {selected_model}")
    else:
        st.info("Please select a Store, Department, and Model to view forecasts")
    
    # Store Risk Analysis
    if not data['risk'].empty:
        st.header("Store Risk Analysis")
        
        # Top 10 stores by risk
        top_risk = data['risk'].head(10)
        
        fig = px.bar(
            top_risk,
            x='Store',
            y='abs_delta',
            color='pct_delta',
            title="Top 10 Stores by Risk (Absolute Sales Delta)",
            labels={'abs_delta': 'Absolute Sales Delta ($)', 'pct_delta': 'Percentage Delta (%)'},
            color_continuous_scale='RdYlGn_r'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk table
        st.subheader("Risk Details")
        st.dataframe(
            top_risk[['Store', 'Dept', 'last_sales', 'trailing_4_avg', 'abs_delta', 'pct_delta']],
            use_container_width=True
        )
    
    # Markdown ROI Analysis
    if not data['roi'].empty:
        st.header("Markdown ROI Analysis")
        
        # Top 10 departments by uplift
        top_roi = data['roi'].head(10)
        
        fig = px.bar(
            top_roi,
            x='Dept',
            y='uplift_percentage',
            color='markdown_sales_correlation',
            title="Top 10 Departments by Markdown Uplift",
            labels={'uplift_percentage': 'Uplift Percentage (%)', 'markdown_sales_correlation': 'Markdown-Sales Correlation'},
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI table
        st.subheader("ROI Details")
        st.dataframe(
            top_roi[['Store', 'Dept', 'uplift_percentage', 'markdown_sales_correlation', 'total_markdown_spend', 'total_sales']],
            use_container_width=True
        )
    
    # Model Comparison
    if not data['forecasts'].empty:
        st.header("Model Performance Comparison")
        
        # Calculate metrics by model
        model_metrics = []
        for model in data['forecasts']['model'].unique():
            model_data = data['forecasts'][data['forecasts']['model'] == model]
            if 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                mae = np.mean(np.abs(model_data['y_true'] - model_data['y_pred']))
                rmse = np.sqrt(np.mean((model_data['y_true'] - model_data['y_pred'])**2))
                mape = np.mean(np.abs((model_data['y_true'] - model_data['y_pred']) / model_data['y_true'])) * 100
                
                model_metrics.append({
                    'Model': model,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })
        
        if model_metrics:
            metrics_df = pd.DataFrame(model_metrics)
            
            # Create comparison chart
            fig = go.Figure()
            
            for metric in ['MAE', 'RMSE']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df['Model'],
                    y=metrics_df[metric],
                    text=[f"${val:,.0f}" for val in metrics_df[metric]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Error Metric ($)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics table
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Dashboard powered by Walmart Sales Forecasting Pipeline")

if __name__ == "__main__":
    main()
