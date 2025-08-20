import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from prophet import Prophet
from src.models.base import BaseModel
from src.logging_utils import get_logger
from src.paths import LOCAL_DIR
from pathlib import Path

logger = get_logger(__name__)

class ProphetModel(BaseModel):
    """Prophet model for local per-series sales forecasting."""
    
    def __init__(self, **kwargs):
        # Set default parameters for Prophet
        default_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'changepoint_range': 0.8,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays': None
        }
        
        # Update with provided parameters
        default_params.update(kwargs)
        
        super().__init__("Prophet", **default_params)
        
        # Prophet model will be created per series
        self.models = {}  # Store models for each Store-Dept combination
        self.series_info = {}  # Store information about each series
        
        logger.info(f"Prophet model initialized with parameters: {default_params}")
    
    def fit(self, data: pd.DataFrame, top_n: int = 50, **kwargs) -> 'ProphetModel':
        """Fit Prophet models for top N Store-Dept series by total revenue."""
        
        logger.info(f"Starting Prophet model training for top {top_n} series")
        
        # Validate input data
        if not self.validate_data(data):
            raise ValueError("Input data validation failed")
        
        # Get top N series by total revenue
        top_series = self._get_top_series(data, top_n)
        
        # Fit Prophet model for each series
        for i, (store, dept) in enumerate(top_series):
            logger.info(f"Training Prophet model for Store {store}, Dept {dept} ({i+1}/{len(top_series)})")
            
            try:
                # Get series data
                series_data = data[(data['Store'] == store) & (data['Dept'] == dept)].copy()
                
                if len(series_data) < 10:  # Need minimum data points
                    logger.warning(f"Insufficient data for Store {store}, Dept {dept}: {len(series_data)} points")
                    continue
                
                # Prepare data for Prophet
                prophet_data = self._prepare_prophet_data(series_data)
                
                # Create and fit Prophet model
                model = self._create_prophet_model()
                model.fit(prophet_data)
                
                # Store model and series info
                series_key = f"{store}_{dept}"
                self.models[series_key] = model
                self.series_info[series_key] = {
                    'store': store,
                    'dept': dept,
                    'data_points': len(series_data),
                    'date_range': (series_data['Date'].min(), series_data['Date'].max()),
                    'total_revenue': series_data['Weekly_Sales'].sum()
                }
                
                logger.info(f"Successfully trained Prophet model for {series_key}")
                
            except Exception as e:
                logger.error(f"Error training Prophet model for Store {store}, Dept {dept}: {str(e)}")
                continue
        
        self.is_fitted = True
        logger.info(f"Prophet model training completed for {len(self.models)} series")
        
        return self
    
    def predict(self, data: pd.DataFrame, horizon_weeks: int = 13, **kwargs) -> np.ndarray:
        """Make predictions using the fitted Prophet models."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate input data
        if not self.validate_data(data, require_target=False):
            raise ValueError("Input data validation failed")
        
        # Initialize predictions array
        predictions = np.zeros(len(data))
        
        # Make predictions for each series that has a model
        for series_key, model in self.models.items():
            store, dept = series_key.split('_')
            store, dept = int(store), int(dept)
            
            # Get data for this series
            series_mask = (data['Store'] == store) & (data['Dept'] == dept)
            series_data = data[series_mask]
            
            if len(series_data) == 0:
                continue
            
            try:
                # Prepare data for Prophet
                prophet_data = self._prepare_prophet_data(series_data)
                
                # Make predictions
                future = self._create_future_dataframe(prophet_data, horizon_weeks)
                forecast = model.predict(future)
                
                # Map predictions back to original data
                for idx, row in series_data.iterrows():
                    # Find corresponding forecast row
                    forecast_row = forecast[forecast['ds'] == row['Date']]
                    if not forecast_row.empty:
                        predictions[data.index.get_loc(idx)] = forecast_row.iloc[0]['yhat']
                
            except Exception as e:
                logger.error(f"Error making predictions for {series_key}: {str(e)}")
                continue
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        logger.info(f"Made Prophet predictions for {len(predictions)} samples")
        return predictions
    
    def predict_with_intervals(self, data: pd.DataFrame, horizon_weeks: int = 13) -> Dict[str, np.ndarray]:
        """Make predictions with confidence intervals."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Initialize arrays
        predictions = np.zeros(len(data))
        lower_bounds = np.zeros(len(data))
        upper_bounds = np.zeros(len(data))
        
        # Make predictions for each series
        for series_key, model in self.models.items():
            store, dept = series_key.split('_')
            store, dept = int(store), int(dept)
            
            # Get data for this series
            series_mask = (data['Store'] == store) & (data['Dept'] == dept)
            series_data = data[series_mask]
            
            if len(series_data) == 0:
                continue
            
            try:
                # Prepare data for Prophet
                prophet_data = self._prepare_prophet_data(series_data)
                
                # Make predictions
                future = self._create_future_dataframe(prophet_data, horizon_weeks)
                forecast = model.predict(future)
                
                # Map predictions back to original data
                for idx, row in series_data.iterrows():
                    forecast_row = forecast[forecast['ds'] == row['Date']]
                    if not forecast_row.empty:
                        data_idx = data.index.get_loc(idx)
                        predictions[data_idx] = forecast_row.iloc[0]['yhat']
                        lower_bounds[data_idx] = forecast_row.iloc[0]['yhat_lower']
                        upper_bounds[data_idx] = forecast_row.iloc[0]['yhat_upper']
                
            except Exception as e:
                logger.error(f"Error making predictions for {series_key}: {str(e)}")
                continue
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.maximum(upper_bounds, 0)
        
        return {
            'predictions': predictions,
            'lower': lower_bounds,
            'upper': upper_bounds
        }
    
    def _get_top_series(self, data: pd.DataFrame, top_n: int) -> List[tuple]:
        """Get top N Store-Dept combinations by total revenue."""
        
        # Calculate total revenue per Store-Dept
        series_revenue = data.groupby(['Store', 'Dept'])['Weekly_Sales'].sum().sort_values(ascending=False)
        
        # Get top N series
        top_series = series_revenue.head(top_n).index.tolist()
        
        logger.info(f"Selected top {len(top_series)} series by revenue")
        return top_series
    
    def _prepare_prophet_data(self, series_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model."""
        
        # Prophet expects columns 'ds' (date) and 'y' (target)
        prophet_data = pd.DataFrame({
            'ds': series_data['Date'],
            'y': series_data['Weekly_Sales']
        })
        
        # Sort by date
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        return prophet_data
    
    def _create_prophet_model(self) -> Prophet:
        """Create a Prophet model with specified parameters."""
        
        model = Prophet(
            changepoint_prior_scale=self.model_params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.model_params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=self.model_params.get('holidays_prior_scale', 10.0),
            seasonality_mode=self.model_params.get('seasonality_mode', 'additive'),
            changepoint_range=self.model_params.get('changepoint_range', 0.8),
            yearly_seasonality=self.model_params.get('yearly_seasonality', True),
            weekly_seasonality=self.model_params.get('weekly_seasonality', True),
            daily_seasonality=self.model_params.get('daily_seasonality', False)
        )
        
        # Add US holidays if specified
        if self.model_params.get('holidays') is not None:
            model.add_country_holidays(country_name='US')
        
        return model
    
    def _create_future_dataframe(self, prophet_data: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
        """Create future dataframe for Prophet forecasting."""
        
        # Get the latest date
        latest_date = prophet_data['ds'].max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=latest_date + pd.Timedelta(days=1),
            periods=horizon_weeks * 7,
            freq='D'
        )
        
        # Convert to weekly (end of week)
        future_weeks = future_dates.to_period('W').unique().to_timestamp()
        
        # Create future dataframe
        future = pd.DataFrame({'ds': future_weeks})
        
        return future
    
    def save(self, file_path: str) -> None:
        """Save all Prophet models to disk."""
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted models")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for series_key, model in self.models.items():
            model_file = file_path.parent / f"prophet_{series_key}.pkl"
            model.save(str(model_file))
        
        # Save metadata
        metadata_path = file_path.with_suffix('.metadata.pkl')
        metadata = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params,
            'series_info': self.series_info,
            'n_models': len(self.models)
        }
        
        from src.utils.io_utils import safe_save_pickle
        safe_save_pickle(metadata, metadata_path)
        
        logger.info(f"Saved {len(self.models)} Prophet models to {file_path.parent}")
    
    def load(self, file_path: str) -> 'ProphetModel':
        """Load Prophet models from disk."""
        
        file_path = Path(file_path)
        
        # Load metadata
        metadata_path = file_path.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            from src.utils.io_utils import safe_load_pickle
            metadata = safe_load_pickle(metadata_path)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.is_fitted = metadata.get('is_fitted', True)
            self.model_params = metadata.get('model_params', {})
            self.series_info = metadata.get('series_info', {})
        
        # Load models
        for series_key in self.series_info.keys():
            model_file = file_path.parent / f"prophet_{series_key}.pkl"
            if model_file.exists():
                model = Prophet()
                model.load(str(model_file))
                self.models[series_key] = model
        
        self.is_fitted = True
        logger.info(f"Loaded {len(self.models)} Prophet models from {file_path.parent}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Prophet models."""
        
        info = super().get_model_info()
        info.update({
            'n_series': len(self.models),
            'series_info': self.series_info
        })
        
        return info
