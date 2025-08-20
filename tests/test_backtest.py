import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils.backtest import RollingOriginBacktest, create_time_series_split

class TestBacktest(unittest.TestCase):
    """Test backtesting functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample time series data
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='W')
        stores = [1, 2, 3]
        depts = [1, 2]
        
        data = []
        for date in dates:
            for store in stores:
                for dept in depts:
                    data.append({
                        'Store': store,
                        'Dept': dept,
                        'Date': date,
                        'Weekly_Sales': np.random.uniform(10000, 50000)
                    })
        
        self.test_data = pd.DataFrame(data)
        self.test_data = self.test_data.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    def test_rolling_origin_backtest_initialization(self):
        """Test RollingOriginBacktest initialization."""
        backtest = RollingOriginBacktest(
            df=self.test_data,
            date_col='Date',
            target_col='Weekly_Sales',
            group_cols=['Store', 'Dept'],
            folds=4,
            horizon=13
        )
        
        self.assertEqual(backtest.folds, 4)
        self.assertEqual(backtest.horizon, 13)
        self.assertEqual(backtest.target_col, 'Weekly_Sales')
        self.assertEqual(backtest.group_cols, ['Store', 'Dept'])
        self.assertTrue(len(backtest.unique_dates) > 0)
    
    def test_create_folds(self):
        """Test fold creation."""
        backtest = RollingOriginBacktest(
            df=self.test_data,
            date_col='Date',
            target_col='Weekly_Sales',
            folds=3,
            horizon=13
        )
        
        folds = backtest.create_folds()
        
        # Check number of folds
        self.assertEqual(len(folds), 3)
        
        # Check fold structure
        for fold in folds:
            self.assertIn('fold', fold)
            self.assertIn('train_start', fold)
            self.assertIn('train_end', fold)
            self.assertIn('test_start', fold)
            self.assertIn('test_end', fold)
            self.assertIn('train_data', fold)
            self.assertIn('test_data', fold)
            
            # Check temporal order
            self.assertLess(fold['train_end'], fold['test_start'])
            self.assertLess(fold['test_start'], fold['test_end'])
            
            # Check data shapes
            self.assertTrue(len(fold['train_data']) > 0)
            self.assertTrue(len(fold['test_data']) > 0)
    
    def test_time_series_split(self):
        """Test time series split functionality."""
        train_data, test_data = create_time_series_split(
            self.test_data, 
            date_col='Date', 
            test_size=0.2
        )
        
        # Check split proportions
        total_rows = len(self.test_data)
        expected_train = int(total_rows * 0.8)
        expected_test = total_rows - expected_train
        
        self.assertAlmostEqual(len(train_data), expected_train, delta=1)
        self.assertAlmostEqual(len(test_data), expected_test, delta=1)
        
        # Check temporal order
        max_train_date = train_data['Date'].max()
        min_test_date = test_data['Date'].min()
        self.assertLess(max_train_date, min_test_date)
        
        # Check no overlap
        train_stores = set(train_data['Store'].unique())
        test_stores = set(test_data['Store'].unique())
        self.assertEqual(train_stores, test_stores)
        
        train_depts = set(train_data['Dept'].unique())
        test_depts = set(test_data['Dept'].unique())
        self.assertEqual(train_depts, test_depts)
    
    def test_backtest_with_mock_model(self):
        """Test backtesting with a mock model."""
        backtest = RollingOriginBacktest(
            df=self.test_data,
            date_col='Date',
            target_col='Weekly_Sales',
            folds=2,
            horizon=13
        )
        
        # Mock model functions
        def mock_fit(train_data, **kwargs):
            return "mock_model"
        
        def mock_predict(model, test_data):
            return np.random.uniform(10000, 50000, len(test_data))
        
        # Run backtesting
        results = backtest.run_backtest(
            model_fit_func=mock_fit,
            model_predict_func=mock_predict
        )
        
        # Check results
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn('fold', result)
            self.assertIn('model', result)
            self.assertIn('predictions', result)
            self.assertIn('actuals', result)
            self.assertIn('test_data', result)
            
            # Check predictions shape
            self.assertEqual(len(result['predictions']), len(result['actuals']))
            self.assertEqual(len(result['predictions']), len(result['test_data']))
    
    def test_evaluate_fold(self):
        """Test fold evaluation."""
        backtest = RollingOriginBacktest(
            df=self.test_data,
            date_col='Date',
            target_col='Weekly_Sales',
            folds=2,
            horizon=13
        )
        
        # Create mock fold result
        mock_result = {
            'fold': 1,
            'actuals': np.array([10000, 20000, 30000]),
            'predictions': np.array([11000, 19000, 31000])
        }
        
        metrics = backtest.evaluate_fold(mock_result)
        
        # Check metrics structure
        expected_metrics = ['fold', 'mae', 'rmse', 'mape', 'smape', 'wape']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check fold number
        self.assertEqual(metrics['fold'], 1)
        
        # Check metric values are finite
        for metric in ['mae', 'rmse', 'mape', 'smape', 'wape']:
            self.assertTrue(np.isfinite(metrics[metric]))
    
    def test_evaluate_all_folds(self):
        """Test evaluation of all folds."""
        backtest = RollingOriginBacktest(
            df=self.test_data,
            date_col='Date',
            target_col='Weekly_Sales',
            folds=2,
            horizon=13
        )
        
        # Create mock backtest results
        mock_results = []
        for fold in range(1, 3):
            mock_result = {
                'fold': fold,
                'actuals': np.array([10000, 20000, 30000]),
                'predictions': np.array([11000, 19000, 31000])
            }
            mock_results.append(mock_result)
        
        # Evaluate all folds
        metrics_df = backtest.evaluate_all_folds(mock_results)
        
        # Check dataframe structure
        self.assertEqual(len(metrics_df), 2)
        expected_columns = ['fold', 'mae', 'rmse', 'mape', 'smape', 'wape']
        for col in expected_columns:
            self.assertIn(col, metrics_df.columns)
        
        # Check fold numbers
        self.assertEqual(list(metrics_df['fold']), [1, 2])
    
    def test_save_predictions(self):
        """Test saving predictions functionality."""
        import tempfile
        import os
        
        backtest = RollingOriginBacktest(
            df=self.test_data,
            date_col='Date',
            target_col='Weekly_Sales',
            folds=2,
            horizon=13
        )
        
        # Create mock results
        mock_results = []
        for fold in range(1, 3):
            mock_result = {
                'fold': fold,
                'test_data': self.test_data.head(10).copy(),
                'actuals': np.random.uniform(10000, 50000, 10),
                'predictions': np.random.uniform(10000, 50000, 10)
            }
            mock_results.append(mock_result)
        
        # Save predictions to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            backtest.save_predictions(mock_results, temp_dir)
            
            # Check files were created
            expected_files = ['preds_fold1.csv', 'preds_fold2.csv']
            for file_name in expected_files:
                file_path = os.path.join(temp_dir, file_name)
                self.assertTrue(os.path.exists(file_path))
                
                # Check file content
                saved_data = pd.read_csv(file_path)
                self.assertIn('y_true', saved_data.columns)
                self.assertIn('y_pred', saved_data.columns)
                self.assertIn('fold', saved_data.columns)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with insufficient data
        small_data = self.test_data.head(50)
        
        with self.assertRaises(Exception):
            backtest = RollingOriginBacktest(
                df=small_data,
                date_col='Date',
                target_col='Weekly_Sales',
                folds=10,  # Too many folds for small data
                horizon=13
            )
            backtest.create_folds()
        
        # Test with single date
        single_date_data = self.test_data[self.test_data['Date'] == self.test_data['Date'].min()]
        
        with self.assertRaises(Exception):
            backtest = RollingOriginBacktest(
                df=single_date_data,
                date_col='Date',
                target_col='Weekly_Sales',
                folds=1,
                horizon=13
            )
            backtest.create_folds()

if __name__ == '__main__':
    unittest.main()
