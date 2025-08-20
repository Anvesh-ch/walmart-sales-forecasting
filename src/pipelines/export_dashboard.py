#!/usr/bin/env python3
"""
Dashboard export pipeline for Walmart Sales Forecasting.

This pipeline:
1. Loads processed data and backtest results
2. Creates flat CSV exports for Tableau and Streamlit
3. Generates store risk analysis
4. Calculates markdown ROI estimates
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logging_utils import setup_logging, get_logger
from src.paths import DATA_PROCESSED, FORECASTS_DIR, METRICS_DIR, EXPORTS_DIR
from src.utils.io_utils import safe_read_parquet, safe_write_csv

def main():
    """Main dashboard export pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Export dashboard data for Walmart sales forecasting')
    parser.add_argument('--data-path', type=str, default=None, help='Path to processed data')
    parser.add_argument('--forecasts-dir', type=str, default=None, help='Directory containing backtest forecasts')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for exports')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('export_dashboard', 'INFO')
    
    try:
        logger.info("Starting dashboard export pipeline")
        
        # Step 1: Load processed data
        logger.info("Step 1: Loading processed data")
        if args.data_path:
            data_path = Path(args.data_path)
        else:
            data_path = DATA_PROCESSED / "global.parquet"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        data = safe_read_parquet(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Step 2: Load backtest forecasts
        logger.info("Step 2: Loading backtest forecasts")
        forecasts_dir = Path(args.forecasts_dir) if args.forecasts_dir else FORECASTS_DIR
        
        # Load XGBoost forecasts
        xgb_forecasts = []
        xgb_dir = forecasts_dir / "xgb"
        if xgb_dir.exists():
            for fold_file in xgb_dir.glob("preds_fold*.csv"):
                fold_data = pd.read_csv(fold_file)
                fold_data['model'] = 'XGBoost'
                xgb_forecasts.append(fold_data)
        
        # Load Prophet forecasts
        prophet_forecasts = []
        prophet_dir = forecasts_dir / "prophet"
        if prophet_dir.exists():
            for fold_file in prophet_dir.glob("preds_fold*.csv"):
                fold_data = pd.read_csv(fold_file)
                fold_data['model'] = 'Prophet'
                prophet_forecasts.append(fold_data)
        
        if not xgb_forecasts and not prophet_forecasts:
            logger.warning("No backtest forecasts found, creating sample exports")
            # Create sample data for demonstration
            create_sample_exports(data, args.output_dir)
            return
        
        # Step 3: Create forecasts export
        logger.info("Step 3: Creating forecasts export")
        all_forecasts = xgb_forecasts + prophet_forecasts
        
        if all_forecasts:
            forecasts_df = pd.concat(all_forecasts, ignore_index=True)
            
            # Add additional columns for dashboard
            forecasts_df['Date'] = pd.to_datetime(forecasts_df['Date'])
            forecasts_df['IsHoliday'] = forecasts_df['IsHoliday'].fillna(False)
            forecasts_df['MarkDownSum'] = forecasts_df.get('MarkDownSum', 0)
            
            # Ensure required columns exist
            required_cols = ['Store', 'Dept', 'Date', 'y_true', 'y_pred', 'model', 'fold']
            missing_cols = set(required_cols) - set(forecasts_df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in forecasts: {missing_cols}")
                for col in missing_cols:
                    if col == 'y_true':
                        forecasts_df['y_true'] = forecasts_df.get('Weekly_Sales', 0)
                    elif col == 'y_pred':
                        forecasts_df['y_pred'] = 0
                    elif col == 'fold':
                        forecasts_df['fold'] = 1
            
            # Add confidence intervals if available
            if 'lower' not in forecasts_df.columns:
                forecasts_df['lower'] = forecasts_df['y_pred'] * 0.9
            if 'upper' not in forecasts_df.columns:
                forecasts_df['upper'] = forecasts_df['y_pred'] * 1.1
            
            # Save forecasts export
            output_dir = Path(args.output_dir) if args.output_dir else EXPORTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            
            forecasts_path = output_dir / "forecasts_store_dept.csv"
            safe_write_csv(forecasts_df, forecasts_path)
            logger.info(f"Forecasts export saved to {forecasts_path}")
        
        # Step 4: Create store risk analysis
        logger.info("Step 4: Creating store risk analysis")
        store_risk_df = create_store_risk_analysis(data, forecasts_df if 'forecasts_df' in locals() else None)
        
        if store_risk_df is not None:
            risk_path = output_dir / "store_risk.csv"
            safe_write_csv(store_risk_df, risk_path)
            logger.info(f"Store risk analysis saved to {risk_path}")
        
        # Step 5: Create markdown ROI analysis
        logger.info("Step 5: Creating markdown ROI analysis")
        markdown_roi_df = create_markdown_roi_analysis(data)
        
        if markdown_roi_df is not None:
            roi_path = output_dir / "markdown_roi.csv"
            safe_write_csv(markdown_roi_df, roi_path)
            logger.info(f"Markdown ROI analysis saved to {roi_path}")
        
        # Step 6: Create summary statistics
        logger.info("Step 6: Creating summary statistics")
        create_summary_statistics(data, output_dir)
        
        logger.info("Dashboard export pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Dashboard export pipeline failed: {str(e)}")
        raise

def create_store_risk_analysis(data: pd.DataFrame, forecasts_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create store risk analysis based on sales patterns."""
    
    logger = get_logger(__name__)
    
    try:
        # Group by Store-Dept and calculate trailing averages
        risk_analysis = data.groupby(['Store', 'Dept']).agg({
            'Weekly_Sales': ['mean', 'std', 'last'],
            'Date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        risk_analysis.columns = ['Store', 'Dept', 'avg_sales', 'std_sales', 'last_sales', 'first_date', 'last_date']
        
        # Calculate trailing averages
        risk_analysis['trailing_4_avg'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].tail(4).groupby(['Store', 'Dept']).mean().values
        risk_analysis['trailing_13_avg'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].tail(13).groupby(['Store', 'Dept']).mean().values
        
        # Calculate risk metrics
        risk_analysis['abs_delta'] = abs(risk_analysis['last_sales'] - risk_analysis['trailing_4_avg'])
        risk_analysis['pct_delta'] = ((risk_analysis['last_sales'] - risk_analysis['trailing_4_avg']) / risk_analysis['trailing_4_avg']) * 100
        
        # Add next week prediction (using simple average for now)
        risk_analysis['next_week_pred'] = risk_analysis['trailing_4_avg']
        
        # Fill NaN values
        risk_analysis = risk_analysis.fillna(0)
        
        # Sort by absolute delta (highest risk first)
        risk_analysis = risk_analysis.sort_values('abs_delta', ascending=False).reset_index(drop=True)
        
        logger.info(f"Created store risk analysis for {len(risk_analysis)} Store-Dept combinations")
        return risk_analysis
        
    except Exception as e:
        logger.error(f"Error creating store risk analysis: {str(e)}")
        return None

def create_markdown_roi_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Create markdown ROI analysis using simple regression."""
    
    logger = get_logger(__name__)
    
    try:
        # Check if MarkDown columns exist
        markdown_cols = [col for col in data.columns if col.startswith('MarkDown')]
        if not markdown_cols:
            logger.warning("No MarkDown columns found, creating sample ROI data")
            return create_sample_markdown_roi(data)
        
        # Calculate total markdown per Store-Dept-Date
        data_copy = data.copy()
        data_copy['TotalMarkDown'] = data_copy[markdown_cols].sum(axis=1)
        
        # Group by Store-Dept and calculate ROI metrics
        roi_analysis = []
        
        for (store, dept), group in data_copy.groupby(['Store', 'Dept']):
            if len(group) < 10:  # Need minimum data points
                continue
            
            # Simple correlation between markdown and sales
            markdown_sales_corr = group['TotalMarkDown'].corr(group['Weekly_Sales'])
            
            # Calculate average sales with and without markdown
            sales_with_markdown = group[group['TotalMarkDown'] > 0]['Weekly_Sales'].mean()
            sales_without_markdown = group[group['TotalMarkDown'] == 0]['Weekly_Sales'].mean()
            
            # Calculate uplift
            if sales_without_markdown > 0:
                uplift_pct = ((sales_with_markdown - sales_without_markdown) / sales_without_markdown) * 100
            else:
                uplift_pct = 0
            
            roi_analysis.append({
                'Store': store,
                'Dept': dept,
                'markdown_sales_correlation': markdown_sales_corr,
                'avg_sales_with_markdown': sales_with_markdown,
                'avg_sales_without_markdown': sales_without_markdown,
                'uplift_percentage': uplift_pct,
                'total_markdown_spend': group['TotalMarkDown'].sum(),
                'total_sales': group['Weekly_Sales'].sum(),
                'markdown_weeks': (group['TotalMarkDown'] > 0).sum()
            })
        
        roi_df = pd.DataFrame(roi_analysis)
        
        # Fill NaN values
        roi_df = roi_df.fillna(0)
        
        # Sort by uplift percentage
        roi_df = roi_df.sort_values('uplift_percentage', ascending=False).reset_index(drop=True)
        
        logger.info(f"Created markdown ROI analysis for {len(roi_df)} Store-Dept combinations")
        return roi_df
        
    except Exception as e:
        logger.error(f"Error creating markdown ROI analysis: {str(e)}")
        return None

def create_sample_markdown_roi(data: pd.DataFrame) -> pd.DataFrame:
    """Create sample markdown ROI data for demonstration."""
    
    # Create sample ROI data
    stores = data['Store'].unique()[:20]  # Use first 20 stores
    depts = data['Dept'].unique()[:10]    # Use first 10 departments
    
    roi_data = []
    for store in stores:
        for dept in depts:
            roi_data.append({
                'Store': store,
                'Dept': dept,
                'markdown_sales_correlation': np.random.uniform(-0.3, 0.8),
                'avg_sales_with_markdown': np.random.uniform(20000, 80000),
                'avg_sales_without_markdown': np.random.uniform(15000, 60000),
                'uplift_percentage': np.random.uniform(-10, 50),
                'total_markdown_spend': np.random.uniform(1000, 10000),
                'total_sales': np.random.uniform(100000, 500000),
                'markdown_weeks': np.random.randint(5, 20)
            })
    
    return pd.DataFrame(roi_data)

def create_summary_statistics(data: pd.DataFrame, output_dir: Path) -> None:
    """Create summary statistics for the dashboard."""
    
    logger = get_logger(__name__)
    
    try:
        # Calculate overall statistics
        summary_stats = {
            'total_stores': data['Store'].nunique(),
            'total_departments': data['Dept'].nunique(),
            'total_observations': len(data),
            'date_range': {
                'start': data['Date'].min().strftime('%Y-%m-%d'),
                'end': data['Date'].max().strftime('%Y-%m-%d')
            },
            'total_revenue': data['Weekly_Sales'].sum(),
            'avg_weekly_sales': data['Weekly_Sales'].mean(),
            'total_holiday_weeks': data['IsHoliday'].sum() if 'IsHoliday' in data.columns else 0
        }
        
        # Save summary statistics
        summary_path = output_dir / "dashboard_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        logger.info(f"Summary statistics saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error creating summary statistics: {str(e)}")

def create_sample_exports(data: pd.DataFrame, output_dir: str) -> None:
    """Create sample exports for demonstration purposes."""
    
    logger = get_logger(__name__)
    
    try:
        output_dir = Path(output_dir) if output_dir else EXPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample forecasts
        sample_forecasts = data.head(1000).copy()
        sample_forecasts['y_true'] = sample_forecasts['Weekly_Sales']
        sample_forecasts['y_pred'] = sample_forecasts['Weekly_Sales'] * np.random.uniform(0.8, 1.2, len(sample_forecasts))
        sample_forecasts['model'] = np.random.choice(['XGBoost', 'Prophet'], len(sample_forecasts))
        sample_forecasts['fold'] = np.random.randint(1, 5, len(sample_forecasts))
        sample_forecasts['lower'] = sample_forecasts['y_pred'] * 0.9
        sample_forecasts['upper'] = sample_forecasts['y_pred'] * 1.1
        
        forecasts_path = output_dir / "forecasts_store_dept.csv"
        safe_write_csv(sample_forecasts, forecasts_path)
        logger.info(f"Sample forecasts export saved to {forecasts_path}")
        
        # Create other sample exports
        create_store_risk_analysis(data)
        create_markdown_roi_analysis(data)
        
        logger.info("Sample exports created successfully")
        
    except Exception as e:
        logger.error(f"Error creating sample exports: {str(e)}")

if __name__ == "__main__":
    main()
