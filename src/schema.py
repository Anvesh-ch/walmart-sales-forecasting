from typing import Dict, List, Any
import pandas as pd
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Expected schemas for each CSV file
TRAIN_SCHEMA = {
    "Store": "int64",
    "Dept": "int64", 
    "Date": "datetime64[ns]",
    "Weekly_Sales": "float64",
    "IsHoliday": "bool"
}

FEATURES_SCHEMA = {
    "Store": "int64",
    "Date": "datetime64[ns]",
    "Temperature": "float64",
    "Fuel_Price": "float64",
    "MarkDown1": "float64",
    "MarkDown2": "float64", 
    "MarkDown3": "float64",
    "MarkDown4": "float64",
    "MarkDown5": "float64",
    "CPI": "float64",
    "Unemployment": "float64",
    "IsHoliday": "bool"
}

STORES_SCHEMA = {
    "Store": "int64",
    "Type": "object",
    "Size": "int64"
}

# Valid value ranges
VALID_RANGES = {
    "Store": (1, 45),
    "Dept": (1, 99),
    "Weekly_Sales": (0, float('inf')),
    "Temperature": (-50, 150),
    "Fuel_Price": (0, 10),
    "MarkDown1": (0, float('inf')),
    "MarkDown2": (0, float('inf')),
    "MarkDown3": (0, float('inf')),
    "MarkDown4": (0, float('inf')),
    "MarkDown5": (0, float('inf')),
    "CPI": (100, 300),
    "Unemployment": (0, 20),
    "Size": (10000, 200000)
}

# Valid store types
VALID_STORE_TYPES = ["A", "B", "C"]

def validate_dataframe(df: pd.DataFrame, expected_schema: Dict[str, str], 
                      name: str) -> bool:
    """Validate dataframe against expected schema."""
    
    logger.info(f"Validating {name} dataframe")
    
    # Check columns
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns in {name}: {missing_cols}")
        return False
    
    # Check data types
    for col, expected_dtype in expected_schema.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype:
                logger.warning(f"Column {col} in {name}: expected {expected_dtype}, got {actual_dtype}")
    
    # Check value ranges
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if invalid_count > 0:
                    logger.warning(f"Column {col} in {name}: {invalid_count} values outside valid range [{min_val}, {max_val}]")
    
    # Check store types
    if "Type" in df.columns:
        invalid_types = df[~df["Type"].isin(VALID_STORE_TYPES)]["Type"].unique()
        if len(invalid_types) > 0:
            logger.warning(f"Invalid store types in {name}: {invalid_types}")
    
    logger.info(f"{name} validation completed")
    return True

def validate_merged_data(df: pd.DataFrame) -> bool:
    """Validate merged dataset for modeling."""
    
    logger.info("Validating merged dataset")
    
    # Check for required columns
    required_cols = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns in merged data: {missing_cols}")
        return False
    
    # Check for duplicates
    duplicates = df.duplicated(subset=["Store", "Dept", "Date"]).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate Store-Dept-Date combinations")
    
    # Check for missing values in key columns
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Missing values in merged data:\n{missing_counts}")
    
    # Check date range
    date_range = df["Date"].max() - df["Date"].min()
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()} ({date_range.days} days)")
    
    logger.info("Merged data validation completed")
    return True
