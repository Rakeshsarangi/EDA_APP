import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def create_distribution_plots(self, numeric_cols: List[str]) -> Dict[str, go.Figure]:
        """Create distribution plots for numeric columns."""
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
            
            plots[col] = fig
            
        return plots
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        if correlation_matrix.empty:
            return None
            
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
        
        return fig
    
    def create_categorical_plots(self, categorical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create plots for categorical columns."""
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
            
            plots[col] = fig
            
        return plots
    
    def create_time_series_plots(self, datetime_cols: List[str], numeric_cols: List[str]) -> Dict[str, go.Figure]:
        """Create time series plots."""
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
                
                plots[f'{date_col}_{num_col}'] = fig
                
        return plots
    
    def create_scatter_matrix(self, numeric_cols: List[str], max_cols: int = 5) -> go.Figure:
        """Create scatter matrix for numeric columns."""
        if len(numeric_cols) < 2:
            return None
            
        cols_to_use = numeric_cols[:min(max_cols, len(numeric_cols))]
        
        fig = px.scatter_matrix(
            self.df[cols_to_use],
            dimensions=cols_to_use,
            height=800
        )
        
        fig.update_layout(title='Scatter Matrix')
        
        return fig
    
    def create_missing_data_plot(self) -> go.Figure:
        """Create missing data visualization."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None
            
        fig = go.Figure(data=[
            go.Bar(
                x=missing.index,
                y=missing.values,
                text=missing.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Missing Data by Column',
            xaxis_title='Columns',
            yaxis_title='Missing Count',
            height=400
        )
        
        return fig