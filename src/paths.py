from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Artifacts directories
ARTIFACTS_ROOT = PROJECT_ROOT / "data" / "artifacts"
MODELS_DIR = ARTIFACTS_ROOT / "models"
FORECASTS_DIR = ARTIFACTS_ROOT / "forecasts"
METRICS_DIR = ARTIFACTS_ROOT / "metrics"
FIGURES_DIR = ARTIFACTS_ROOT / "figures"

# Exports directory
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Input files
TRAIN_CSV = DATA_RAW / "train.csv"
FEATURES_CSV = DATA_RAW / "features.csv"
STORES_CSV = DATA_RAW / "stores.csv"

# Output files
GLOBAL_PARQUET = DATA_PROCESSED / "global.parquet"
LOCAL_DIR = DATA_PROCESSED / "local"

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, ARTIFACTS_ROOT, 
                  MODELS_DIR, FORECASTS_DIR, METRICS_DIR, FIGURES_DIR, 
                  EXPORTS_DIR, LOCAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
