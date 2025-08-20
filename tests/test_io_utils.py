import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from src.utils.io_utils import (
    safe_read_csv, safe_write_csv, safe_read_parquet, safe_write_parquet,
    safe_save_pickle, safe_load_pickle, safe_save_joblib, safe_load_joblib,
    check_file_exists
)

class TestIOUtils(unittest.TestCase):
    """Test IO utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample dataframe
        self.test_df = pd.DataFrame({
            'A': np.random.randint(1, 100, 10),
            'B': np.random.randn(10),
            'C': ['test' + str(i) for i in range(10)]
        })
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_safe_read_write_csv(self):
        """Test CSV read/write operations."""
        csv_path = self.temp_path / "test.csv"
        
        # Test writing
        safe_write_csv(self.test_df, csv_path)
        self.assertTrue(csv_path.exists())
        
        # Test reading
        loaded_df = safe_read_csv(csv_path)
        pd.testing.assert_frame_equal(self.test_df, loaded_df)
    
    def test_safe_read_write_parquet(self):
        """Test parquet read/write operations."""
        parquet_path = self.temp_path / "test.parquet"
        
        # Test writing
        safe_write_parquet(self.test_df, parquet_path)
        self.assertTrue(parquet_path.exists())
        
        # Test reading
        loaded_df = safe_read_parquet(parquet_path)
        pd.testing.assert_frame_equal(self.test_df, loaded_df)
    
    def test_safe_save_load_pickle(self):
        """Test pickle save/load operations."""
        pickle_path = self.temp_path / "test.pkl"
        
        # Test saving
        test_object = {'data': self.test_df, 'metadata': {'version': '1.0'}}
        safe_save_pickle(test_object, pickle_path)
        self.assertTrue(pickle_path.exists())
        
        # Test loading
        loaded_object = safe_load_pickle(pickle_path)
        self.assertEqual(loaded_object['metadata']['version'], '1.0')
        pd.testing.assert_frame_equal(loaded_object['data'], self.test_df)
    
    def test_safe_save_load_joblib(self):
        """Test joblib save/load operations."""
        joblib_path = self.temp_path / "test.joblib"
        
        # Test saving
        test_object = {'data': self.test_df, 'metadata': {'version': '1.0'}}
        safe_save_joblib(test_object, joblib_path)
        self.assertTrue(joblib_path.exists())
        
        # Test loading
        loaded_object = safe_load_joblib(joblib_path)
        self.assertEqual(loaded_object['metadata']['version'], '1.0')
        pd.testing.assert_frame_equal(loaded_object['data'], self.test_df)
    
    def test_check_file_exists(self):
        """Test file existence checking."""
        # Test existing file
        csv_path = self.temp_path / "test.csv"
        safe_write_csv(self.test_df, csv_path)
        self.assertTrue(check_file_exists(csv_path))
        
        # Test non-existing file
        non_existent_path = self.temp_path / "nonexistent.csv"
        self.assertFalse(check_file_exists(non_existent_path))
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test reading non-existent CSV
        non_existent_csv = self.temp_path / "nonexistent.csv"
        with self.assertRaises(FileNotFoundError):
            safe_read_csv(non_existent_csv)
        
        # Test reading non-existent parquet
        non_existent_parquet = self.temp_path / "nonexistent.parquet"
        with self.assertRaises(FileNotFoundError):
            safe_read_parquet(non_existent_parquet)
        
        # Test loading non-existent pickle
        non_existent_pickle = self.temp_path / "nonexistent.pkl"
        with self.assertRaises(FileNotFoundError):
            safe_load_pickle(non_existent_pickle)
        
        # Test loading non-existent joblib
        non_existent_joblib = self.temp_path / "nonexistent.joblib"
        with self.assertRaises(FileNotFoundError):
            safe_load_joblib(non_existent_joblib)
    
    def test_directory_creation(self):
        """Test automatic directory creation."""
        nested_path = self.temp_path / "nested" / "deep" / "test.csv"
        
        # Should create directories automatically
        safe_write_csv(self.test_df, nested_path)
        self.assertTrue(nested_path.exists())
        self.assertTrue(nested_path.parent.exists())
    
    def test_data_integrity(self):
        """Test data integrity across different formats."""
        # Test CSV integrity
        csv_path = self.temp_path / "test.csv"
        safe_write_csv(self.test_df, csv_path)
        csv_loaded = safe_read_csv(csv_path)
        pd.testing.assert_frame_equal(self.test_df, csv_loaded)
        
        # Test parquet integrity
        parquet_path = self.temp_path / "test.parquet"
        safe_write_parquet(self.test_df, parquet_path)
        parquet_loaded = safe_read_parquet(parquet_path)
        pd.testing.assert_frame_equal(self.test_df, parquet_loaded)
        
        # Test pickle integrity
        pickle_path = self.temp_path / "test.pkl"
        safe_save_pickle(self.test_df, pickle_path)
        pickle_loaded = safe_load_pickle(pickle_path)
        pd.testing.assert_frame_equal(self.test_df, pickle_loaded)
        
        # Test joblib integrity
        joblib_path = self.temp_path / "test.joblib"
        safe_save_joblib(self.test_df, joblib_path)
        joblib_loaded = safe_load_joblib(joblib_path)
        pd.testing.assert_frame_equal(self.test_df, joblib_loaded)

if __name__ == '__main__':
    unittest.main()
