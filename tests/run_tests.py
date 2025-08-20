#!/usr/bin/env python3
"""
Test runner for Walmart Sales Forecasting project.

Run all tests or specific test modules.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all tests in the project."""
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_tests(test_modules):
    """Run specific test modules."""
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(f'tests.{module}')
            suite.addTests(tests)
            print(f"Loaded tests from {module}")
        except Exception as e:
            print(f"Error loading {module}: {e}")
    
    if suite.countTestCases() > 0:
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    else:
        print("No tests to run")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for Walmart Sales Forecasting')
    parser.add_argument('--modules', nargs='+', help='Specific test modules to run')
    
    args = parser.parse_args()
    
    if args.modules:
        success = run_specific_tests(args.modules)
    else:
        print("Running all tests...")
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
