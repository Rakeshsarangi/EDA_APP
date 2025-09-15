import hashlib
import os
from typing import Any
import pandas as pd

class SecurityManager:
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        for char in dangerous_chars:
            text = text.replace(char, '')
        return text.strip()
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate dataframe for security issues."""
        # Check for suspicious column names
        suspicious_patterns = ['exec', 'eval', '__', 'import', 'os.', 'sys.']
        for col in df.columns:
            for pattern in suspicious_patterns:
                if pattern in str(col).lower():
                    return False
        return True
    
    @staticmethod
    def create_data_hash(df: pd.DataFrame) -> str:
        """Create hash of dataframe for integrity checking."""
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()