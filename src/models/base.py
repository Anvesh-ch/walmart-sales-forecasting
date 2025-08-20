from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from src.logging_utils import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """Base class for all forecasting models."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.model_params = kwargs
        self.feature_columns = None
        self.target_column = 'Weekly_Sales'
        
        logger.info(f"Initialized {model_name} model with parameters: {kwargs}")
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseModel':
        """Fit the model to the training data."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the fitted model to disk."""
        from src.utils.io_utils import safe_save_joblib
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model object
        safe_save_joblib(self.model, file_path)
        
        # Save metadata
        metadata_path = file_path.with_suffix('.metadata.pkl')
        metadata = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        from src.utils.io_utils import safe_save_pickle
        safe_save_pickle(metadata, metadata_path)
        
        logger.info(f"Model saved to {file_path}")
    
    def load(self, file_path: Union[str, Path]) -> 'BaseModel':
        """Load a fitted model from disk."""
        from src.utils.io_utils import safe_load_joblib, safe_load_pickle
        
        file_path = Path(file_path)
        
        # Load model object
        self.model = safe_load_joblib(file_path)
        
        # Load metadata
        metadata_path = file_path.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            metadata = safe_load_pickle(metadata_path)
            self.model_name = metadata.get('model_name', self.model_name)
            self.is_fitted = metadata.get('is_fitted', True)
            self.model_params = metadata.get('model_params', {})
            self.feature_columns = metadata.get('feature_columns', self.feature_columns)
            self.target_column = metadata.get('target_column', self.target_column)
        
        self.is_fitted = True
        logger.info(f"Model loaded from {file_path}")
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if not self.is_fitted:
            logger.warning("Cannot get feature importance from unfitted model")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if self.feature_columns:
                return pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': importance
                }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance not available for this model")
        return None
    
    def validate_data(self, data: pd.DataFrame, require_target: bool = True) -> bool:
        """Validate input data for the model."""
        
        if data.empty:
            logger.error("Input data is empty")
            return False
        
        # Check for required columns
        required_cols = ['Store', 'Dept', 'Date']
        if require_target:
            required_cols.append(self.target_column)
        
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for missing values in key columns
        key_missing = data[required_cols].isnull().sum().sum()
        if key_missing > 0:
            logger.warning(f"Found {key_missing} missing values in key columns")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            logger.error("Date column must be datetime type")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        
        if self.feature_columns is None:
            logger.error("Model not fitted - feature columns unknown")
            return data
        
        # Select only feature columns that exist in the data
        available_features = [col for col in self.feature_columns if col in data.columns]
        missing_features = set(self.feature_columns) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with 0
            for col in missing_features:
                data[col] = 0
        
        # Ensure all features are numeric
        feature_data = data[available_features].copy()
        feature_data = feature_data.fillna(0)
        
        # Convert to numeric where possible
        for col in feature_data.columns:
            if not pd.api.types.is_numeric_dtype(feature_data[col]):
                try:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                except:
                    feature_data[col] = 0
        
        feature_data = feature_data.fillna(0)
        
        logger.info(f"Prepared {len(available_features)} features for prediction")
        return feature_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        
        info = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'target_column': self.target_column
        }
        
        if self.is_fitted and self.model is not None:
            info['model_type'] = type(self.model).__name__
            
            # Add model-specific info
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info['max_depth'] = self.model.max_depth
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return f"{self.__class__.__name__}({info})"
