#!/usr/bin/env python3
"""
Backtesting pipeline for Walmart Sales Forecasting.

This pipeline:
1. Loads processed data and trained models
2. Runs rolling-origin backtesting for all models
3. Saves backtest results and metrics
4. Generates comparison reports
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logging_utils import setup_logging, get_logger
from src.paths import DATA_PROCESSED, FORECASTS_DIR, METRICS_DIR
from src.utils.io_utils import safe_read_parquet, safe_save_joblib
from src.utils.backtest import RollingOriginBacktest
from src.models.xgb_model import XGBoostModel
from src.models.prophet_model import ProphetModel
from src.utils.eval_utils import calculate_all_metrics

def main():
    """Main backtesting pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run backtesting for Walmart sales forecasting models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data-path', type=str, default=None, help='Path to processed data')
    parser.add_argument('--folds', type=int, default=4, help='Number of backtesting folds')
    parser.add_argument('--horizon', type=int, default=13, help='Forecast horizon in weeks')
    parser.add_argument('--models-dir', type=str, default=None, help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('backtest_all', 'INFO')
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    try:
        logger.info("Starting backtesting pipeline")
        
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
        
        # Step 2: Load trained models
        logger.info("Step 2: Loading trained models")
        models_dir = Path(args.models_dir) if args.models_dir else Path("data/artifacts/models")
        
        # Load XGBoost model
        xgb_model = None
        xgb_path = models_dir / "xgb_model.joblib"
        if xgb_path.exists():
            xgb_model = XGBoostModel()
            xgb_model.load(xgb_path)
            logger.info("XGBoost model loaded successfully")
        else:
            logger.warning("XGBoost model not found, skipping XGBoost backtesting")
        
        # Load Prophet models
        prophet_model = None
        prophet_path = models_dir / "prophet_models"
        if prophet_path.exists():
            prophet_model = ProphetModel()
            prophet_model.load(prophet_path)
            logger.info("Prophet models loaded successfully")
        else:
            logger.warning("Prophet models not found, skipping Prophet backtesting")
        
        if not xgb_model and not prophet_model:
            raise ValueError("No trained models found for backtesting")
        
        # Step 3: Initialize backtesting
        logger.info(f"Step 3: Initializing backtesting with {args.folds} folds and {args.horizon} week horizon")
        backtest = RollingOriginBacktest(
            df=data,
            date_col='Date',
            target_col='Weekly_Sales',
            group_cols=['Store', 'Dept'],
            folds=args.folds,
            horizon=args.horizon
        )
        
        # Step 4: Run backtesting for each model
        all_results = {}
        
        # XGBoost backtesting
        if xgb_model:
            logger.info("Step 4a: Running XGBoost backtesting")
            try:
                xgb_results = backtest.run_backtest(
                    model_fit_func=lambda train_data, **kwargs: xgb_model.fit(train_data),
                    model_predict_func=lambda model, test_data: model.predict(test_data)
                )
                
                if xgb_results:
                    # Evaluate XGBoost results
                    xgb_metrics = backtest.evaluate_all_folds(xgb_results)
                    all_results['XGBoost'] = {
                        'results': xgb_results,
                        'metrics': xgb_metrics
                    }
                    
                    # Save XGBoost predictions
                    xgb_preds_dir = FORECASTS_DIR / "xgb"
                    xgb_preds_dir.mkdir(parents=True, exist_ok=True)
                    backtest.save_predictions(xgb_results, str(xgb_preds_dir))
                    
                    logger.info("XGBoost backtesting completed successfully")
                else:
                    logger.warning("XGBoost backtesting produced no results")
                    
            except Exception as e:
                logger.error(f"XGBoost backtesting failed: {str(e)}")
        
        # Prophet backtesting
        if prophet_model:
            logger.info("Step 4b: Running Prophet backtesting")
            try:
                prophet_results = backtest.run_backtest(
                    model_fit_func=lambda train_data, **kwargs: prophet_model.fit(train_data, top_n=50),
                    model_predict_func=lambda model, test_data: model.predict(test_data)
                )
                
                if prophet_results:
                    # Evaluate Prophet results
                    prophet_metrics = backtest.evaluate_all_folds(prophet_results)
                    all_results['Prophet'] = {
                        'results': prophet_results,
                        'metrics': prophet_metrics
                    }
                    
                    # Save Prophet predictions
                    prophet_preds_dir = FORECASTS_DIR / "prophet"
                    prophet_preds_dir.mkdir(parents=True, exist_ok=True)
                    backtest.save_predictions(prophet_results, str(prophet_preds_dir))
                    
                    logger.info("Prophet backtesting completed successfully")
                else:
                    logger.warning("Prophet backtesting produced no results")
                    
            except Exception as e:
                logger.error(f"Prophet backtesting failed: {str(e)}")
        
        # Step 5: Generate comparison report
        logger.info("Step 5: Generating comparison report")
        
        comparison_data = []
        for model_name, results in all_results.items():
            metrics = results['metrics']
            for _, row in metrics.iterrows():
                comparison_data.append({
                    'model': model_name,
                    'fold': row['fold'],
                    'mae': row['mae'],
                    'rmse': row['rmse'],
                    'mape': row['mape'],
                    'smape': row['smape'],
                    'wape': row['wape']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison results
        comparison_path = METRICS_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")
        
        # Step 6: Generate summary statistics
        logger.info("Step 6: Generating summary statistics")
        
        summary_stats = {}
        for model_name, results in all_results.items():
            metrics = results['metrics']
            summary_stats[model_name] = {
                'mean_mae': metrics['mae'].mean(),
                'std_mae': metrics['mae'].std(),
                'mean_rmse': metrics['rmse'].mean(),
                'std_rmse': metrics['rmse'].std(),
                'mean_mape': metrics['mape'].mean(),
                'std_mape': metrics['mape'].std(),
                'mean_smape': metrics['smape'].mean(),
                'std_smape': metrics['smape'].std(),
                'mean_wape': metrics['wape'].mean(),
                'std_wape': metrics['wape'].std()
            }
        
        # Save summary statistics
        summary_path = METRICS_DIR / "backtest_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Backtest summary saved to {summary_path}")
        
        # Step 7: Print results summary
        logger.info("Backtesting Results Summary:")
        for model_name, stats in summary_stats.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    MAE: {stats['mean_mae']:.2f} ± {stats['std_mae']:.2f}")
            logger.info(f"    RMSE: {stats['mean_rmse']:.2f} ± {stats['std_rmse']:.2f}")
            logger.info(f"    MAPE: {stats['mean_mape']:.2f}% ± {stats['std_mape']:.2f}%")
        
        logger.info("Backtesting pipeline completed successfully")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Backtesting pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
