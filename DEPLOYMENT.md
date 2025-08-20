# Deployment Guide

## GitHub Repository
✅ **Repository Created**: [https://github.com/Anvesh-ch/walmart-sales-forecasting](https://github.com/Anvesh-ch/walmart-sales-forecasting)
✅ **Code Pushed**: All source code and configuration files are now on GitHub

## Streamlit Cloud Deployment

### Step 1: Access Streamlit Cloud
1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"

### Step 2: Configure Your App
- **Repository**: `Anvesh-ch/walmart-sales-forecasting`
- **Branch**: `main`
- **Main file path**: `app/streamlit_app.py`
- **App URL**: Will be auto-generated (e.g., `https://walmart-sales-forecasting-xxxxx.streamlit.app`)

### Step 3: Deploy
1. Click "Deploy!"
2. Wait for the build to complete (usually 2-5 minutes)
3. Your app will be live at the provided URL

## Local Development

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py

# Or use the Makefile
make all  # Run complete pipeline first
```

### Test the Pipeline
```bash
# Run complete pipeline
make all

# Or individual stages
make prepare_data
make train_prophet
make train_xgb
make backtest_all
make compare_all
make export_dashboard
```

## App Features

The Streamlit app provides:
- **Forecast vs Actuals**: Interactive charts with confidence intervals
- **Store Risk Analysis**: Sales volatility and risk assessment
- **Markdown ROI Analysis**: Promotional effectiveness insights
- **Model Performance Comparison**: Metrics across all models
- **Interactive Filters**: Store, Department, Date Range, Model selection

## Data Requirements

Before running the app, ensure you have:
1. **Raw Data**: Place `train.csv`, `features.csv`, `stores.csv` in `data/raw/`
2. **Run Pipeline**: Execute `make all` to generate all required data files
3. **Exports**: The app reads from `exports/` directory

## Troubleshooting

### Common Issues
1. **Missing Data**: Run `make all` to generate required data files
2. **Dependencies**: Ensure all packages in `requirements.txt` are installed
3. **Port Conflicts**: Change port in `.streamlit/config.toml` if needed

### Streamlit Cloud Issues
1. **Build Failures**: Check `requirements.txt` and `packages.txt`
2. **Import Errors**: Ensure all dependencies are in `requirements.txt`
3. **Data Access**: App reads from local files, not external APIs

## Repository Structure
```
walmart-sales-forecasting/
├── app/streamlit_app.py          # Main Streamlit application
├── src/                          # Core pipeline code
├── data/                         # Data directories
├── exports/                      # Dashboard exports
├── .streamlit/config.toml        # Streamlit configuration
├── requirements.txt              # Python dependencies
├── packages.txt                  # System dependencies
└── Makefile                     # Pipeline orchestration
```

## Next Steps
1. **Deploy to Streamlit Cloud** using the steps above
2. **Add Data**: Place your Walmart CSV files in `data/raw/`
3. **Run Pipeline**: Execute `make all` to generate forecasts
4. **Access App**: Use the Streamlit Cloud URL for your deployed app

## Support
- **Documentation**: See `README.md` for detailed project information
- **Issues**: Report problems on the GitHub repository
- **Testing**: Run `python tests/run_tests.py` to verify functionality
