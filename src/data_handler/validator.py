import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataValidator:
    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate and clean dataframe, return cleaned df and issues found."""
        issues = {
            'missing_values': {},
            'duplicates': 0,
            'data_types': {},
            'outliers': {}
        }
        
        # Check for missing values
        missing = df.isnull().sum()
        issues['missing_values'] = missing[missing > 0].to_dict()
        
        # Check for duplicates
        issues['duplicates'] = df.duplicated().sum()
        
        # Infer better data types
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            # Try to convert to numeric
            if df_cleaned[col].dtype == 'object':
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
                except:
                    pass
            
            # Try to convert to datetime
            if df_cleaned[col].dtype == 'object':
                try:
                    temp = pd.to_datetime(df_cleaned[col], errors='coerce')
                    if temp.notna().sum() > len(df_cleaned) * 0.5:
                        df_cleaned[col] = temp
                except:
                    pass
        
        # Detect outliers using IQR method for numeric columns
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df_cleaned[col] < (Q1 - 1.5 * IQR)) | 
                       (df_cleaned[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                issues['outliers'][col] = outliers
        
        return df_cleaned, issues