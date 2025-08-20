#!/usr/bin/env python3
"""
XGBoost training pipeline for Walmart Sales Forecasting.

This pipeline:
1. Loads processed data
2. Trains XGBoost model
3. Saves trained model
4. Generates feature importance report
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
from src.utils.io_utils import safe_read_parquet, safe_save_joblib
from src.models.xgb_model import XGBoostModel
from src.utils.eval_utils import calculate_all_metrics

def main():
    """Main XGBoost training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train XGBoost model for Walmart sales forecasting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data-path', type=str, default=None, help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for model')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators')
    parser.add_argument('--max-depth', type=int, default=6, help='Maximum tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('train_xgb', 'INFO')
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    try:
        logger.info("Starting XGBoost training pipeline")
        
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
        
        # Step 2: Initialize XGBoost model
        logger.info("Step 2: Initializing XGBoost model")
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'random_state': args.seed
        }
        
        xgb_model = XGBoostModel(**model_params)
        logger.info(f"XGBoost model initialized with parameters: {model_params}")
        
        # Step 3: Train model
        logger.info("Step 3: Training XGBoost model")
        xgb_model.fit(data)
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model")
        
        # Make predictions on training data for evaluation
        train_predictions = xgb_model.predict(data)
        
        # Calculate metrics
        metrics = calculate_all_metrics(data['Weekly_Sales'], train_predictions)
        logger.info(f"Training metrics: {metrics}")
        
        # Step 5: Get feature importance
        logger.info("Step 5: Generating feature importance report")
        feature_importance = xgb_model.get_feature_importance()
        
        if feature_importance is not None:
            logger.info(f"Top 10 features by importance:")
            logger.info(feature_importance.head(10))
        
        # Step 6: Save model and artifacts
        logger.info("Step 6: Saving model and artifacts")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = MODELS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "xgb_model.joblib"
        xgb_model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        if feature_importance is not None:
            importance_path = output_dir / "xgb_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Save training metrics
        metrics_path = output_dir / "xgb_training_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")
        
        # Save model info
        model_info = xgb_model.get_model_info()
        info_path = output_dir / "xgb_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        logger.info(f"Model info saved to {info_path}")
        
        # Step 7: Cross-validation (optional)
        logger.info("Step 7: Performing cross-validation")
        try:
            cv_scores = xgb_model.cross_validate(data, n_splits=5)
            cv_path = output_dir / "xgb_cv_scores.json"
            with open(cv_path, 'w') as f:
                json.dump(cv_scores, f, indent=2, default=str)
            logger.info(f"Cross-validation scores saved to {cv_path}")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        
        logger.info("XGBoost training pipeline completed successfully")
        
        # Print summary
        logger.info("Training Summary:")
        logger.info(f"  Model: XGBoost")
        logger.info(f"  Parameters: {model_params}")
        logger.info(f"  Training samples: {len(data)}")
        logger.info(f"  Features: {len(xgb_model.feature_columns) if xgb_model.feature_columns else 'Unknown'}")
        logger.info(f"  Training MAE: {metrics.get('mae', 'N/A'):.2f}")
        logger.info(f"  Training RMSE: {metrics.get('rmse', 'N/A'):.2f}")
        
        return xgb_model
        
    except Exception as e:
        logger.error(f"XGBoost training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
