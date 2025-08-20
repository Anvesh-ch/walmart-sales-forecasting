import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
from datetime import datetime, timedelta
from src.logging_utils import get_logger
from src.utils.ts_utils import get_train_test_split_dates

logger = get_logger(__name__)

class RollingOriginBacktest:
    """Rolling origin backtesting for time series models."""
    
    def __init__(self, df: pd.DataFrame, date_col: str = "Date", 
                 target_col: str = "Weekly_Sales", group_cols: List[str] = None,
                 folds: int = 4, horizon: int = 13):
        
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.group_cols = group_cols or ['Store', 'Dept']
        self.folds = folds
        self.horizon = horizon
        
        # Ensure data is sorted by date
        self.df = self.df.sort_values(date_col).reset_index(drop=True)
        
        # Get unique dates
        self.unique_dates = sorted(self.df[date_col].unique())
        
        logger.info(f"Initialized RollingOriginBacktest with {folds} folds and {horizon} week horizon")
    
    def create_folds(self) -> List[Dict[str, Any]]:
        """Create train-test splits for rolling origin backtesting."""
        
        logger.info("Creating rolling origin folds")
        
        folds_data = []
        total_dates = len(self.unique_dates)
        
        # Calculate fold boundaries
        fold_size = total_dates // (self.folds + 1)
        
        for fold in range(self.folds):
            # Calculate split points
            train_end_idx = (fold + 1) * fold_size
            test_end_idx = min(train_end_idx + self.horizon, total_dates)
            
            if test_end_idx <= train_end_idx:
                break
            
            train_end_date = self.unique_dates[train_end_idx - 1]
            test_start_date = self.unique_dates[train_end_idx]
            test_end_date = self.unique_dates[test_end_idx - 1]
            
            # Create train and test masks
            train_mask = self.df[self.date_col] <= train_end_date
            test_mask = (self.df[self.date_col] >= test_start_date) & (self.df[self.date_col] <= test_end_date)
            
            fold_data = {
                'fold': fold + 1,
                'train_start': self.unique_dates[0],
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'train_data': self.df[train_mask].copy(),
                'test_data': self.df[test_mask].copy(),
                'train_mask': train_mask,
                'test_mask': test_mask
            }
            
            folds_data.append(fold_data)
            
            logger.info(f"Fold {fold + 1}: Train {fold_data['train_start']} to {fold_data['train_end']}, "
                       f"Test {fold_data['test_start']} to {fold_data['test_end']}")
        
        logger.info(f"Created {len(folds_data)} folds for backtesting")
        return folds_data
    
    def run_backtest(self, model_fit_func: Callable, model_predict_func: Callable,
                    model_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Run backtesting for a given model."""
        
        logger.info("Starting rolling origin backtesting")
        
        if model_params is None:
            model_params = {}
        
        folds_data = self.create_folds()
        results = []
        
        for fold_data in folds_data:
            logger.info(f"Processing fold {fold_data['fold']}")
            
            try:
                # Fit model on training data
                logger.info(f"Fitting model on fold {fold_data['fold']} training data")
                model = model_fit_func(fold_data['train_data'], **model_params)
                
                # Make predictions on test data
                logger.info(f"Making predictions on fold {fold_data['fold']} test data")
                predictions = model_predict_func(model, fold_data['test_data'])
                
                # Store results
                fold_result = {
                    'fold': fold_data['fold'],
                    'train_start': fold_data['train_start'],
                    'train_end': fold_data['train_end'],
                    'test_start': fold_data['test_start'],
                    'test_end': fold_data['test_end'],
                    'model': model,
                    'predictions': predictions,
                    'actuals': fold_data['test_data'][self.target_col].values,
                    'test_data': fold_data['test_data']
                }
                
                results.append(fold_result)
                logger.info(f"Completed fold {fold_data['fold']}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold_data['fold']}: {str(e)}")
                continue
        
        logger.info(f"Backtesting completed for {len(results)} folds")
        return results
    
    def evaluate_fold(self, fold_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single fold's predictions."""
        
        from src.utils.eval_utils import calculate_all_metrics
        
        actuals = fold_result['actuals']
        predictions = fold_result['predictions']
        
        # Ensure predictions is array-like
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        elif hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
        
        # Calculate metrics
        metrics = calculate_all_metrics(actuals, predictions)
        metrics['fold'] = fold_result['fold']
        
        return metrics
    
    def evaluate_all_folds(self, backtest_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Evaluate all folds and return aggregated metrics."""
        
        logger.info("Evaluating all backtest folds")
        
        fold_metrics = []
        
        for fold_result in backtest_results:
            try:
                metrics = self.evaluate_fold(fold_result)
                fold_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating fold {fold_result['fold']}: {str(e)}")
                continue
        
        if not fold_metrics:
            logger.error("No valid fold metrics to evaluate")
            return pd.DataFrame()
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(fold_metrics)
        
        # Calculate aggregate statistics
        agg_metrics = metrics_df.drop('fold', axis=1).agg(['mean', 'std', 'min', 'max'])
        
        logger.info("Fold evaluation completed")
        logger.info(f"Aggregate metrics:\n{agg_metrics}")
        
        return metrics_df
    
    def save_predictions(self, backtest_results: List[Dict[str, Any]], 
                        output_dir: str) -> None:
        """Save predictions from all folds to CSV files."""
        
        logger.info(f"Saving backtest predictions to {output_dir}")
        
        for fold_result in backtest_results:
            fold_num = fold_result['fold']
            
            # Create predictions dataframe
            pred_df = fold_result['test_data'].copy()
            pred_df['y_true'] = fold_result['actuals']
            pred_df['y_pred'] = fold_result['predictions']
            pred_df['fold'] = fold_num
            
            # Save to CSV
            output_file = f"{output_dir}/preds_fold{fold_num}.csv"
            pred_df.to_csv(output_file, index=False)
            
            logger.info(f"Saved fold {fold_num} predictions to {output_file}")
        
        logger.info("All fold predictions saved successfully")

def create_time_series_split(df: pd.DataFrame, date_col: str = "Date",
                            test_size: float = 0.2, 
                            group_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a simple train-test split for time series data."""
    
    logger.info(f"Creating time series split with test_size: {test_size}")
    
    if group_cols is None:
        group_cols = ['Store', 'Dept']
    
    # Get unique dates
    unique_dates = sorted(df[date_col].unique())
    
    # Calculate split point
    split_idx = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates[split_idx]
    
    # Create masks
    train_mask = df[date_col] <= split_date
    test_mask = df[date_col] > split_date
    
    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    
    logger.info(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
    logger.info(f"Train period: {train_data[date_col].min()} to {train_data[date_col].max()}")
    logger.info(f"Test period: {test_data[date_col].min()} to {test_data[date_col].max()}")
    
    return train_data, test_data
