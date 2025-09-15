import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the dataset."""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # in MB
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'datetime_columns': self.datetime_cols
        }
    
    def get_numeric_analysis(self) -> Dict:
        """Analyze numeric columns."""
        if not self.numeric_cols:
            return {}
        
        analysis = {}
        for col in self.numeric_cols:
            analysis[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'q25': self.df[col].quantile(0.25),
                'q75': self.df[col].quantile(0.75),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'zeros': (self.df[col] == 0).sum(),
                'unique': self.df[col].nunique()
            }
        return analysis
    
    def get_categorical_analysis(self) -> Dict:
        """Analyze categorical columns."""
        if not self.categorical_cols:
            return {}
        
        analysis = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            analysis[col] = {
                'unique_values': self.df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequency': value_counts.values[0] if len(value_counts) > 0 else 0,
                'top_5': value_counts.head(5).to_dict()
            }
        return analysis
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        if len(self.numeric_cols) > 1:
            return self.df[self.numeric_cols].corr()
        return pd.DataFrame()
    
    def get_feature_importance(self, target_col: str = None) -> Dict:
        """Calculate feature importance if a target column is identified."""
        if not target_col or target_col not in self.df.columns:
            return {}
        
        try:
            X = self.df.drop(columns=[target_col])
            y = self.df[target_col]
            
            # Encode categorical variables
            le_dict = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
                le_dict[col] = le
            
            # Fill missing values
            X = X.fillna(X.mean())
            
            # Determine if regression or classification
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance.to_dict('records')
        except:
            return {}
    
    def detect_patterns(self) -> Dict:
        """Detect patterns and anomalies in the data."""
        patterns = {
            'missing_patterns': {},
            'distribution_types': {},
            'potential_id_columns': [],
            'constant_columns': [],
            'high_cardinality': []
        }
        
        # Missing value patterns
        missing_df = self.df.isnull()
        if missing_df.any().any():
            patterns['missing_patterns'] = {
                'columns_with_missing': missing_df.sum()[missing_df.sum() > 0].to_dict(),
                'rows_with_missing': missing_df.any(axis=1).sum()
            }
        
        # Distribution types for numeric columns
        for col in self.numeric_cols:
            if self.df[col].nunique() > 10:
                # Perform normality test
                if len(self.df[col].dropna()) > 20:
                    _, p_value = stats.normaltest(self.df[col].dropna())
                    patterns['distribution_types'][col] = 'normal' if p_value > 0.05 else 'non-normal'
        
        # Identify potential ID columns
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df):
                patterns['potential_id_columns'].append(col)
        
        # Identify constant columns
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                patterns['constant_columns'].append(col)
        
        # High cardinality categorical columns
        for col in self.categorical_cols:
            if self.df[col].nunique() > len(self.df) * 0.5:
                patterns['high_cardinality'].append(col)
        
        return patterns