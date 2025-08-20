# Walmart Sales Forecasting

A production-style repository for sales forecasting using the Walmart Store Sales dataset. This project implements a two-model approach with Prophet (local per-series) and XGBoost (global tabular) models, featuring rolling-origin backtesting and comprehensive evaluation metrics.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Walmart CSV files in `data/raw/`:**
   - `train.csv` - Sales data
   - `features.csv` - Store features and economic indicators
   - `stores.csv` - Store metadata

3. **Run the complete pipeline:**
   ```bash
   make all
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Repository Structure

```
project_root/
├── data/
│   ├── raw/           # User drops Walmart CSVs here
│   ├── interim/       # Intermediate processed data
│   ├── processed/     # Final processed datasets
│   └── artifacts/     # Models, forecasts, metrics
├── src/
│   ├── data/          # Data ingestion, merging, cleaning
│   ├── features/      # Feature engineering
│   ├── models/        # Model implementations
│   ├── pipelines/     # End-to-end pipelines
│   └── utils/         # Utilities and helpers
├── app/               # Streamlit dashboard
├── exports/           # Flat CSVs for Tableau
└── tests/             # Unit tests
```

## Two-Model Approach

- **Prophet (Local)**: Trains individual models for top-N Store-Dept series by revenue. Captures local seasonality and trends.
- **XGBoost (Global)**: Single model trained on the entire panel dataset with engineered features. Handles cross-series patterns and interactions.

## Backtesting Design

- **Rolling Origin**: 4 folds with 13-week forecast horizon
- **Time-Aware Splits**: Maintains temporal order to prevent data leakage
- **Comprehensive Metrics**: MAE, RMSE, MAPE, sMAPE, WAPE
- **Fold Stability**: Coefficient of variation analysis across folds

## Data Preparation

- **MarkDown Handling**: Forward-fill promo weeks up to 8 weeks, remaining NaNs to 0
- **Economic Features**: Standardize CPI and Unemployment within each store over time
- **Time Features**: Lags [1,2,3,4,6,8,13], rolling windows [3,4,6,8,13], holiday flags
- **Encodings**: One-hot store types, department categories, interaction features

## CLI Commands

```bash
# Data preparation
python -m src.pipelines.prepare_data --seed 42

# Model training
python -m src.pipelines.train_prophet --top_n 50 --seed 42
python -m src.pipelines.train_xgb --seed 42

# Backtesting and evaluation
python -m src.pipelines.backtest_all --folds 4 --horizon 13 --seed 42
python -m src.pipelines.compare_all

# Dashboard export
python -m src.pipelines.export_dashboard
```

## Dashboard Exports

The `export_dashboard` pipeline creates flat CSVs for Tableau and Streamlit:

- **`forecasts_store_dept.csv`**: Store-Dept level forecasts with confidence intervals
- **`store_risk.csv`**: Risk analysis based on sales volatility and trends
- **`markdown_roi.csv`**: ROI estimates from markdown promotions

## Tableau Integration

Connect Tableau to the `exports/*.csv` files for:
- Interactive sales forecasting dashboards
- Store performance monitoring
- Promotional effectiveness analysis
- Risk assessment visualizations

## Key Features

- **Deterministic**: Global seed=42 for reproducible results
- **Production-Ready**: Clean logging, error handling, validation
- **Lightweight**: Optimized for laptop execution
- **Extensible**: Modular design for easy model additions
- **No External Dependencies**: Local CSV processing only

## Performance

- **Prophet**: Top-50 series by revenue (~2-5 minutes)
- **XGBoost**: Global model with engineered features (~1-3 minutes)
- **Backtesting**: 4 folds with 13-week horizon (~10-20 minutes)
- **Total Pipeline**: Complete end-to-end execution in reasonable time

## Requirements

- Python 3.8+
- 8GB+ RAM recommended
- No cloud services or external APIs required
- Deterministic results with seed=42
