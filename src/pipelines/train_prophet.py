#!/usr/bin/env python3
"""
Prophet training pipeline for Walmart Sales Forecasting.

This pipeline:
1. Loads processed data
2. Trains Prophet models for top N Store-Dept series
3. Saves trained models
4. Generates series information report
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logging_utils import setup_logging, get_logger
from src.paths import DATA_PROCESSED, MODELS_DIR
from src.utils.io_utils import safe_read_parquet
from src.models.prophet_model import ProphetModel

def main():
    """Main Prophet training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Prophet models for Walmart sales forecasting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data-path', type=str, default=None, help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for models')
    parser.add_argument('--top-n', type=int, default=50, help='Number of top series to model')
    parser.add_argument('--changepoint-prior-scale', type=float, default=0.05, help='Prophet changepoint prior scale')
    parser.add_argument('--seasonality-prior-scale', type=float, default=10.0, help='Prophet seasonality prior scale')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('train_prophet', 'INFO')
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    try:
        logger.info("Starting Prophet training pipeline")
        
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
        
        # Step 2: Initialize Prophet model
        logger.info("Step 2: Initializing Prophet model")
        model_params = {
            'changepoint_prior_scale': args.changepoint_prior_scale,
            'seasonality_prior_scale': args.seasonality_prior_scale,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'changepoint_range': 0.8,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays': True  # Enable US holidays
        }
        
        prophet_model = ProphetModel(**model_params)
        logger.info(f"Prophet model initialized with parameters: {model_params}")
        
        # Step 3: Train models for top N series
        logger.info(f"Step 3: Training Prophet models for top {args.top_n} series")
        prophet_model.fit(data, top_n=args.top_n)
        
        # Step 4: Generate series information report
        logger.info("Step 4: Generating series information report")
        series_info = prophet_model.series_info
        
        logger.info(f"Successfully trained models for {len(series_info)} series")
        
        # Print top series information
        logger.info("Top series by revenue:")
        for i, (series_key, info) in enumerate(list(series_info.items())[:10]):
            logger.info(f"  {i+1}. Store {info['store']}, Dept {info['dept']}: "
                       f"${info['total_revenue']:,.0f} total revenue")
        
        # Step 5: Save models and artifacts
        logger.info("Step 5: Saving models and artifacts")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = MODELS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Prophet models
        model_path = output_dir / "prophet_models"
        prophet_model.save(model_path)
        logger.info(f"Prophet models saved to {model_path}")
        
        # Save series information
        series_info_path = output_dir / "prophet_series_info.csv"
        series_df = pd.DataFrame.from_dict(series_info, orient='index')
        series_df.to_csv(series_info_path, index=False)
        logger.info(f"Series information saved to {series_info_path}")
        
        # Save training summary
        summary = {
            'n_series_trained': len(series_info),
            'top_n': args.top_n,
            'model_params': model_params,
            'total_revenue_covered': sum(info['total_revenue'] for info in series_info.values()),
            'avg_data_points': np.mean([info['data_points'] for info in series_info.values()]),
            'date_range': {
                'min': min(info['date_range'][0] for info in series_info.values()),
                'max': max(info['date_range'][1] for info in series_info.values())
            }
        }
        
        summary_path = output_dir / "prophet_training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Training summary saved to {summary_path}")
        
        # Step 6: Test predictions on sample data
        logger.info("Step 6: Testing predictions")
        try:
            # Get sample data for testing
            sample_data = data.head(1000)  # Use first 1000 rows for testing
            test_predictions = prophet_model.predict(sample_data)
            
            logger.info(f"Test predictions completed for {len(test_predictions)} samples")
            logger.info(f"Prediction range: {test_predictions.min():.2f} to {test_predictions.max():.2f}")
            
        except Exception as e:
            logger.warning(f"Test predictions failed: {str(e)}")
        
        logger.info("Prophet training pipeline completed successfully")
        
        # Print summary
        logger.info("Training Summary:")
        logger.info(f"  Model: Prophet")
        logger.info(f"  Series trained: {len(series_info)}")
        logger.info(f"  Top N requested: {args.top_n}")
        logger.info(f"  Total revenue covered: ${summary['total_revenue_covered']:,.0f}")
        logger.info(f"  Average data points per series: {summary['avg_data_points']:.1f}")
        
        return prophet_model
        
    except Exception as e:
        logger.error(f"Prophet training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
