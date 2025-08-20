import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.base import BaseModel
from src.logging_utils import get_logger
from src.utils.ts_utils import create_time_series_split

logger = get_logger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost model for global sales forecasting."""
    
    def __init__(self, **kwargs):
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 10
        }
        
        # Update with provided parameters
        default_params.update(kwargs)
        
        super().__init__("XGBoost", **default_params)
        
        # Initialize XGBoost model
        self.model = xgb.XGBRegressor(**default_params)
        
        logger.info(f"XGBoost model initialized with parameters: {default_params}")
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'XGBoostModel':
        """Fit the XGBoost model to the training data."""
        
        logger.info("Starting XGBoost model training")
        
        # Validate input data
        if not self.validate_data(data):
            raise ValueError("Input data validation failed")
        
        # Prepare features
        feature_cols = self._get_feature_columns(data)
        self.feature_columns = feature_cols
        
        # Split data into train and validation
        train_data, val_data = self._create_train_val_split(data, **kwargs)
        
        # Prepare training features
        X_train = train_data[feature_cols]
        y_train = train_data[self.target_column]
        
        # Prepare validation features
        X_val = val_data[feature_cols]
        y_val = val_data[self.target_column]
        
        # Ensure all features are numeric
        X_train = self._ensure_numeric_features(X_train)
        X_val = self._ensure_numeric_features(X_val)
        
        # Train the model
        logger.info(f"Training XGBoost model with {len(feature_cols)} features")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        logger.info(f"Validation MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
        
        self.is_fitted = True
        logger.info("XGBoost model training completed successfully")
        
        return self
    
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions using the fitted XGBoost model."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate input data
        if not self.validate_data(data, require_target=False):
            raise ValueError("Input data validation failed")
        
        # Prepare features
        feature_data = self.prepare_features(data)
        
        # Make predictions
        predictions = self.model.predict(feature_data)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def _get_feature_columns(self, data: pd.DataFrame) -> list:
        """Get feature columns for modeling."""
        
        # Exclude non-feature columns
        exclude_cols = ['Store', 'Dept', 'Date', self.target_column, 'y_true', 'y_pred', 'fold']
        
        # Get all columns that are not excluded
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Filter to numeric columns only
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Identified {len(numeric_cols)} numeric feature columns")
        return numeric_cols
    
    def _create_train_val_split(self, data: pd.DataFrame, 
                               test_size: float = 0.2, **kwargs) -> tuple:
        """Create train-validation split for time series data."""
        
        # Use time series split to maintain temporal order
        train_data, val_data = create_time_series_split(
            data, date_col='Date', test_size=test_size
        )
        
        return train_data, val_data
    
    def _ensure_numeric_features(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all features are numeric."""
        
        # Convert to numeric where possible
        for col in feature_data.columns:
            if not pd.api.types.is_numeric_dtype(feature_data[col]):
                try:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                except:
                    feature_data[col] = 0
        
        # Fill any NaN values
        feature_data = feature_data.fillna(0)
        
        return feature_data
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from XGBoost model."""
        
        if not self.is_fitted:
            logger.warning("Cannot get feature importance from unfitted model")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if self.feature_columns:
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                logger.info("Feature importance retrieved successfully")
                return importance_df
        
        logger.info("Feature importance not available")
        return None
    
    def cross_validate(self, data: pd.DataFrame, n_splits: int = 5) -> Dict[str, list]:
        """Perform time series cross-validation."""
        
        logger.info(f"Starting {n_splits}-fold time series cross-validation")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        # Prepare features
        feature_data = self.prepare_features(data)
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'mae': [],
            'rmse': [],
            'mape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(feature_data)):
            logger.info(f"Processing fold {fold + 1}")
            
            # Split data
            X_train, X_val = feature_data.iloc[train_idx], feature_data.iloc[val_idx]
            y_train, y_val = data.iloc[train_idx][self.target_column], data.iloc[val_idx][self.target_column]
            
            # Train model on this fold
            fold_model = xgb.XGBRegressor(**self.model_params)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['mape'].append(mape)
            
            logger.info(f"Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        # Calculate average scores
        avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        logger.info(f"Cross-validation completed. Average scores: {avg_scores}")
        
        return cv_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the XGBoost model."""
        
        info = super().get_model_info()
        
        if self.is_fitted and self.model is not None:
            info.update({
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'subsample': self.model.subsample,
                'colsample_bytree': self.model.colsample_bytree
            })
        
        return info
