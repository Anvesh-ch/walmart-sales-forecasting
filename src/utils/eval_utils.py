import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple
from src.logging_utils import get_logger

logger = get_logger(__name__)

def mae(y_true: Union[np.ndarray, pd.Series], 
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Mean Absolute Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Root Mean Square Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true: Union[np.ndarray, pd.Series], 
          y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not np.any(mask):
        return np.nan
    
    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100

def wape(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Weighted Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    if np.sum(y_true) == 0:
        return np.nan
    
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

def calculate_all_metrics(y_true: Union[np.ndarray, pd.Series], 
                         y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculate all evaluation metrics."""
    
    logger.info("Calculating evaluation metrics")
    
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred)
    }
    
    logger.info(f"Metrics calculated: {metrics}")
    return metrics

def calculate_metrics_by_group(df: pd.DataFrame, 
                              y_true_col: str, 
                              y_pred_col: str, 
                              group_cols: list) -> pd.DataFrame:
    """Calculate metrics grouped by specified columns."""
    
    logger.info(f"Calculating metrics by groups: {group_cols}")
    
    def group_metrics(group):
        return pd.Series(calculate_all_metrics(
            group[y_true_col], group[y_pred_col]
        ))
    
    metrics_df = df.groupby(group_cols).apply(group_metrics).reset_index()
    
    logger.info(f"Group metrics calculated for {len(metrics_df)} groups")
    return metrics_df

def validate_predictions(y_true: Union[np.ndarray, pd.Series], 
                        y_pred: Union[np.ndarray, pd.Series]) -> bool:
    """Validate that predictions and actuals are valid for evaluation."""
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check shapes
    if y_true.shape != y_pred.shape:
        logger.error(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
        return False
    
    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        logger.warning("NaN values found in y_true or y_pred")
    
    # Check for infinite values
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        logger.warning("Infinite values found in y_true or y_pred")
    
    # Check for negative values in sales (if applicable)
    if np.any(y_true < 0) or np.any(y_pred < 0):
        logger.warning("Negative values found in y_true or y_pred")
    
    return True

def calculate_rolling_metrics(y_true: Union[np.ndarray, pd.Series], 
                             y_pred: Union[np.ndarray, pd.Series], 
                             window: int = 13) -> pd.DataFrame:
    """Calculate rolling metrics over specified window."""
    
    logger.info(f"Calculating rolling metrics with window {window}")
    
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    
    # Calculate rolling metrics
    rolling_mae = y_true.rolling(window=window).apply(
        lambda x: mae(x, y_pred.loc[x.index])
    )
    
    rolling_rmse = y_true.rolling(window=window).apply(
        lambda x: rmse(x, y_pred.loc[x.index])
    )
    
    rolling_mape = y_true.rolling(window=window).apply(
        lambda x: mape(x, y_pred.loc[x.index])
    )
    
    rolling_df = pd.DataFrame({
        'rolling_mae': rolling_mae,
        'rolling_rmse': rolling_rmse,
        'rolling_mape': rolling_mape
    })
    
    logger.info(f"Rolling metrics calculated for {len(rolling_df)} periods")
    return rolling_df
