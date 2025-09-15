import unittest
import pandas as pd
import numpy as np
from src.data_handler.validator import DataValidator

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        # Create sample dataframe for testing
        self.df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, np.nan],
            'string_col': ['a', 'b', 'c', 'd', 'e', 'f'],
            'mixed_col': ['1', '2', '3', '4', '5', '6']
        })
    
    def test_validate_and_clean(self):
        df_cleaned, issues = DataValidator.validate_and_clean(self.df)
        
        # Check if missing values are detected
        self.assertIn('numeric_col', issues['missing_values'])
        self.assertEqual(issues['missing_values']['numeric_col'], 1)
        
        # Check if data types are inferred correctly
        self.assertEqual(df_cleaned['mixed_col'].dtype, np.dtype('int64'))

if __name__ == '__main__':
    unittest.main()