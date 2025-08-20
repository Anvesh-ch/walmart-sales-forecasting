import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Any, Union, Optional
from src.logging_utils import get_logger

logger = get_logger(__name__)

def safe_read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Safely read CSV file with error handling."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully read CSV: {file_path}, shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {str(e)}")
        raise

def safe_write_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Safely write DataFrame to CSV with error handling."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing CSV file: {file_path}")
        df.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Successfully wrote CSV: {file_path}")
        
    except Exception as e:
        logger.error(f"Error writing CSV file {file_path}: {str(e)}")
        raise

def safe_read_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """Safely read parquet file with error handling."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Reading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully read parquet: {file_path}, shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading parquet file {file_path}: {str(e)}")
        raise

def safe_write_parquet(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Safely write DataFrame to parquet with error handling."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing parquet file: {file_path}")
        df.to_parquet(file_path, index=False, **kwargs)
        logger.info(f"Successfully wrote parquet: {file_path}")
        
    except Exception as e:
        logger.error(f"Error writing parquet file {file_path}: {str(e)}")
        raise

def safe_save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Safely save object using pickle with error handling."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving pickle file: {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved pickle: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {str(e)}")
        raise

def safe_load_pickle(file_path: Union[str, Path]) -> Any:
    """Safely load object using pickle with error handling."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading pickle file: {file_path}")
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Successfully loaded pickle: {file_path}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {str(e)}")
        raise

def safe_save_joblib(obj: Any, file_path: Union[str, Path]) -> None:
    """Safely save object using joblib with error handling."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving joblib file: {file_path}")
        joblib.dump(obj, file_path)
        logger.info(f"Successfully saved joblib: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving joblib file {file_path}: {str(e)}")
        raise

def safe_load_joblib(file_path: Union[str, Path]) -> Any:
    """Safely load object using joblib with error handling."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading joblib file: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Successfully loaded joblib: {file_path}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading joblib file {file_path}: {str(e)}")
        raise

def check_file_exists(file_path: Union[str, Path]) -> bool:
    """Check if file exists and log status."""
    file_path = Path(file_path)
    exists = file_path.exists()
    if exists:
        logger.info(f"File exists: {file_path}")
    else:
        logger.warning(f"File not found: {file_path}")
    return exists
