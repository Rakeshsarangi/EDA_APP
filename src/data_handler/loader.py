import pandas as pd
import json
from typing import Optional, Union
import streamlit as st
from config.settings import settings

class DataLoader:
    @staticmethod
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file with security checks."""
        try:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                st.error(f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
                return None
            
            # Get file extension
            file_name = uploaded_file.name
            file_ext = '.' + file_name.split('.')[-1].lower()
            
            if file_ext not in settings.ALLOWED_EXTENSIONS:
                st.error(f"File type {file_ext} not supported")
                return None
            
            # Load based on file type
            if file_ext == '.csv':
                encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
                last_error = None
                df = None

                for enc in encodings_to_try:
                    try:
                        # Reset file pointer to beginning for each attempt
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        break  # Success! Exit the loop
                    except UnicodeDecodeError as e:
                        last_error = e
                        continue
                    except Exception as e:
                        # Handle other potential errors (empty file, etc.)
                        last_error = e
                        continue

                # Only raise error if we couldn't load with any encoding
                if df is None:
                    if last_error:
                        raise ValueError(
                            f"CSV file could not be loaded. Tried encodings: {encodings_to_try}. "
                            f"Last error: {last_error}"
                        )
                    else:
                        raise ValueError("CSV file is empty or has no valid data")
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == '.json':
                data = json.load(uploaded_file)
                df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
            else:
                return None
            
            # Limit rows if necessary
            if len(df) > settings.MAX_ROWS_FOR_ANALYSIS:
                st.warning(f"Dataset truncated to {settings.MAX_ROWS_FOR_ANALYSIS} rows for analysis")
                df = df.head(settings.MAX_ROWS_FOR_ANALYSIS)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None