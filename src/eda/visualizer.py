import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def create_distribution_plots(self, numeric_cols: List[str]) -> Dict[str, Tuple[go.Figure, Dict[str, str]]]:
        """Create distribution plots for numeric columns with descriptions."""
        plots = {}
        
        for col in numeric_cols[:10]:  # Limit to 10 columns
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'{col} - Histogram', f'{col} - Box Plot')
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.df[col], name='Histogram', showlegend=False),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=self.df[col], name='Box Plot', showlegend=False),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=f'Distribution Analysis: {col}',
                height=400
            )
            
            # Add description and use case
            info = {
                'description': f"""
                **Distribution Analysis for {col}:**
                - **Histogram (Left):** Shows the frequency distribution of values, revealing patterns like normal distribution, skewness, or multimodality
                - **Box Plot (Right):** Displays the five-number summary (min, Q1, median, Q3, max) and identifies outliers as individual points
                """,
                'use_case': """
                **When to use this visualization:**
                • Understand the spread and central tendency of numerical data
                • Identify outliers and extreme values
                • Check for data quality issues (unexpected distributions)
                • Compare distributions before and after transformations
                • Assess normality assumptions for statistical tests
                """
            }
            
            plots[col] = (fig, info)
            
        return plots
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> Tuple[go.Figure, Dict[str, str]]:
        """Create correlation heatmap with description."""
        if correlation_matrix.empty:
            return None, None
            
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            height=600,
            width=800
        )
        
        info = {
            'description': """
            **Correlation Matrix:**
            Shows the linear relationship strength between all pairs of numeric variables.
            - **Red colors:** Positive correlation (variables increase together)
            - **Blue colors:** Negative correlation (one increases as other decreases)
            - **White:** No linear correlation
            - **Values range:** -1 (perfect negative) to +1 (perfect positive)
            """,
            'use_case': """
            **When to use this visualization:**
            • Identify multicollinearity before regression analysis
            • Find redundant features for dimensionality reduction
            • Discover relationships between variables
            • Feature selection for machine learning models
            • Validate hypotheses about variable relationships
            """
        }
        
        return fig, info
    
    def create_categorical_plots(self, categorical_cols: List[str]) -> Dict[str, Tuple[go.Figure, Dict[str, str]]]:
        """Create plots for categorical columns with descriptions."""
        plots = {}
        
        for col in categorical_cols[:10]:  # Limit to 10 columns
            value_counts = self.df[col].value_counts().head(15)
            
            fig = go.Figure(data=[
                go.Bar(x=value_counts.index, y=value_counts.values)
            ])
            
            fig.update_layout(
                title=f'Value Counts: {col}',
                xaxis_title=col,
                yaxis_title='Count',
                height=400
            )
            
            # Calculate some stats for the description
            total_unique = self.df[col].nunique()
            total_values = len(self.df[col])
            most_common = value_counts.index[0] if len(value_counts) > 0 else 'N/A'
            
            info = {
                'description': f"""
                **Category Distribution for {col}:**
                - **Total unique values:** {total_unique}
                - **Most common value:** "{most_common}" ({value_counts.values[0] if len(value_counts) > 0 else 0} occurrences)
                - **Coverage:** Top 15 categories shown (if applicable)
                - Bar heights represent frequency of each category
                """,
                'use_case': """
                **When to use this visualization:**
                • Understand categorical variable distribution
                • Identify data imbalance issues
                • Find rare or dominant categories
                • Data quality checks (unexpected categories)
                • Inform encoding strategies for ML models
                """
            }
            
            plots[col] = (fig, info)
            
        return plots
    
    def create_time_series_plots(self, datetime_cols: List[str], numeric_cols: List[str]) -> Dict[str, Tuple[go.Figure, Dict[str, str]]]:
        """Create time series plots with descriptions."""
        plots = {}
        
        for date_col in datetime_cols[:3]:  # Limit to 3 date columns
            for num_col in numeric_cols[:3]:  # Limit to 3 numeric columns
                fig = go.Figure()
                
                # Sort by date
                sorted_df = self.df.sort_values(date_col)
                
                fig.add_trace(go.Scatter(
                    x=sorted_df[date_col],
                    y=sorted_df[num_col],
                    mode='lines+markers',
                    name=num_col
                ))
                
                fig.update_layout(
                    title=f'{num_col} over {date_col}',
                    xaxis_title=date_col,
                    yaxis_title=num_col,
                    height=400
                )
                
                info = {
                    'description': f"""
                    **Time Series Analysis: {num_col} over {date_col}:**
                    - Shows how {num_col} changes over time
                    - Each point represents a measurement at a specific time
                    - Lines connect consecutive time points
                    - Look for trends, seasonality, and anomalies
                    """,
                    'use_case': """
                    **When to use this visualization:**
                    • Track metrics over time (sales, temperature, stock prices)
                    • Identify trends and seasonal patterns
                    • Detect anomalies or change points
                    • Forecast future values
                    • Monitor system performance or KPIs
                    """
                }
                
                plots[f'{date_col}_{num_col}'] = (fig, info)
                
        return plots
    
    def create_scatter_matrix(self, numeric_cols: List[str], max_cols: int = 5) -> Tuple[go.Figure, Dict[str, str]]:
        """Create scatter matrix with description."""
        if len(numeric_cols) < 2:
            return None, None
            
        cols_to_use = numeric_cols[:min(max_cols, len(numeric_cols))]
        
        fig = px.scatter_matrix(
            self.df[cols_to_use],
            dimensions=cols_to_use,
            height=800
        )
        
        fig.update_layout(title='Scatter Matrix')
        
        info = {
            'description': f"""
            **Scatter Matrix (Pair Plot):**
            - Displays pairwise relationships between {len(cols_to_use)} numeric variables
            - **Diagonal:** Distribution of each variable (histogram)
            - **Off-diagonal:** Scatter plots showing relationships between pairs
            - Useful for spotting linear/non-linear relationships and clusters
            """,
            'use_case': """
            **When to use this visualization:**
            • Explore relationships between multiple variables simultaneously
            • Identify variable pairs with strong relationships
            • Detect non-linear patterns
            • Find clusters or groups in multivariate data
            • Quick exploratory analysis of numeric features
            """
        }
        
        return fig, info
    
    def create_missing_data_plot(self) -> Tuple[go.Figure, Dict[str, str]]:
        """Create missing data visualization with description."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None, None
            
        # Calculate percentages
        missing_pct = (missing / len(self.df) * 100).round(1)
        
        fig = go.Figure(data=[
            go.Bar(
                x=missing.index,
                y=missing.values,
                text=[f'{val}<br>({pct}%)' for val, pct in zip(missing.values, missing_pct.values)],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Missing Data by Column',
            xaxis_title='Columns',
            yaxis_title='Missing Count',
            height=400
        )
        
        total_missing = missing.sum()
        total_cells = len(self.df) * len(self.df.columns)
        overall_pct = (total_missing / total_cells * 100)
        
        info = {
            'description': f"""
            **Missing Data Analysis:**
            - **Total missing values:** {total_missing:,} ({overall_pct:.2f}% of all data)
            - **Columns with missing data:** {len(missing)} out of {len(self.df.columns)}
            - **Most missing:** {missing.index[0]} ({missing.values[0]:,} values, {missing_pct.values[0]}%)
            - Bar heights show absolute count, labels show count and percentage
            """,
            'use_case': """
            **When to use this visualization:**
            • Assess data quality and completeness
            • Decide on imputation strategies
            • Identify patterns in missing data (MCAR, MAR, MNAR)
            • Determine if columns should be dropped
            • Plan data cleaning workflow
            """
        }
        
        return fig, info