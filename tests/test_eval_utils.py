import unittest
import numpy as np
import pandas as pd
from src.utils.eval_utils import (
    mae, rmse, mape, smape, wape, 
    calculate_all_metrics, validate_predictions
)

class TestEvalUtils(unittest.TestCase):
    """Test evaluation utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.array([100, 200, 300, 400, 500])
        self.y_pred = np.array([110, 190, 310, 390, 510])
        
        # Edge cases
        self.y_true_zero = np.array([0, 100, 200, 0, 300])
        self.y_pred_zero = np.array([10, 110, 190, 20, 310])
        
        self.y_true_negative = np.array([-100, 100, 200, -50, 300])
        self.y_pred_negative = np.array([-90, 110, 190, -40, 310])
    
    def test_mae(self):
        """Test Mean Absolute Error calculation."""
        expected_mae = np.mean(np.abs(self.y_true - self.y_pred))
        calculated_mae = mae(self.y_true, self.y_pred)
        self.assertAlmostEqual(expected_mae, calculated_mae, places=10)
        
        # Test with pandas Series
        y_true_series = pd.Series(self.y_true)
        y_pred_series = pd.Series(self.y_pred)
        mae_series = mae(y_true_series, y_pred_series)
        self.assertAlmostEqual(expected_mae, mae_series, places=10)
    
    def test_rmse(self):
        """Test Root Mean Square Error calculation."""
        expected_rmse = np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
        calculated_rmse = rmse(self.y_true, self.y_pred)
        self.assertAlmostEqual(expected_rmse, calculated_rmse, places=10)
    
    def test_mape(self):
        """Test Mean Absolute Percentage Error calculation."""
        # Calculate expected MAPE manually
        pct_errors = np.abs((self.y_true - self.y_pred) / self.y_true)
        expected_mape = np.mean(pct_errors) * 100
        
        calculated_mape = mape(self.y_true, self.y_pred)
        self.assertAlmostEqual(expected_mape, calculated_mape, places=10)
        
        # Test with zero values (should handle gracefully)
        mape_zero = mape(self.y_true_zero, self.y_pred_zero)
        self.assertTrue(np.isfinite(mape_zero))
    
    def test_smape(self):
        """Test Symmetric Mean Absolute Percentage Error calculation."""
        # Calculate expected sMAPE manually
        denominator = np.abs(self.y_true) + np.abs(self.y_pred)
        smape_errors = 2 * np.abs(self.y_pred - self.y_true) / denominator
        expected_smape = np.mean(smape_errors) * 100
        
        calculated_smape = smape(self.y_true, self.y_pred)
        self.assertAlmostEqual(expected_smape, calculated_smape, places=10)
    
    def test_wape(self):
        """Test Weighted Absolute Percentage Error calculation."""
        # Calculate expected WAPE manually
        abs_errors = np.abs(self.y_true - self.y_pred)
        expected_wape = np.sum(abs_errors) / np.sum(self.y_true) * 100
        
        calculated_wape = wape(self.y_true, self.y_pred)
        self.assertAlmostEqual(expected_wape, calculated_wape, places=10)
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        metrics = calculate_all_metrics(self.y_true, self.y_pred)
        
        expected_keys = ['mae', 'rmse', 'mape', 'smape', 'wape']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))
        
        # Verify values match individual calculations
        self.assertAlmostEqual(mae(self.y_true, self.y_pred), metrics['mae'], places=10)
        self.assertAlmostEqual(rmse(self.y_true, self.y_pred), metrics['rmse'], places=10)
        self.assertAlmostEqual(mape(self.y_true, self.y_pred), metrics['mape'], places=10)
    
    def test_validate_predictions(self):
        """Test prediction validation."""
        # Valid predictions
        self.assertTrue(validate_predictions(self.y_true, self.y_pred))
        
        # Shape mismatch
        y_pred_wrong_shape = np.array([100, 200, 300])
        self.assertFalse(validate_predictions(self.y_true, y_pred_wrong_shape))
        
        # NaN values
        y_true_nan = np.array([100, np.nan, 300, 400, 500])
        self.assertTrue(validate_predictions(y_true_nan, self.y_pred))
        
        # Infinite values
        y_true_inf = np.array([100, np.inf, 300, 400, 500])
        self.assertTrue(validate_predictions(y_true_inf, self.y_pred))
        
        # Negative values
        self.assertTrue(validate_predictions(self.y_true_negative, self.y_pred_negative))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty arrays
        with self.assertRaises(ValueError):
            mae(np.array([]), np.array([]))
        
        # Single value
        single_mae = mae(np.array([100]), np.array([110]))
        self.assertEqual(single_mae, 10.0)
        
        # All zeros
        y_true_zeros = np.zeros(5)
        y_pred_zeros = np.zeros(5)
        mae_zeros = mae(y_true_zeros, y_pred_zeros)
        self.assertEqual(mae_zeros, 0.0)
        
        # All same values
        y_true_same = np.ones(5) * 100
        y_pred_same = np.ones(5) * 100
        mae_same = mae(y_true_same, y_pred_same)
        self.assertEqual(mae_same, 0.0)

if __name__ == '__main__':
    unittest.main()
