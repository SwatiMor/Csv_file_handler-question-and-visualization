# visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple, List
import io
import base64

class Visualizer:
    def __init__(self):
        """Initialize the visualizer"""
        pass
    
    def create_visualization(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Create a visualization based on the specified configuration
        
        Args:
            df: Pandas DataFrame containing the data
            config: Visualization configuration dictionary
            
        Returns:
            Tuple containing:
            - Plotly figure object (or None if error)
            - Error message (or None if successful)
        """
        try:
            vis_type = config.get('type', 'bar')
            title = config.get('title', 'Data Visualization')
            columns = config.get('columns', [])
            
            # If no columns specified, try to determine appropriate ones
            if not columns:
                cat_cols = [col for col, dtype in df.dtypes.items() if dtype == 'object']
                num_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(df[col])]
                
                if vis_type == 'bar' and cat_cols and num_cols:
                    columns = [cat_cols[0], num_cols[0]]
                elif vis_type in ['line', 'scatter'] and len(num_cols) >= 2:
                    columns = [num_cols[0], num_cols[1]]
                elif vis_type == 'histogram' and num_cols:
                    columns = [num_cols[0]]
            
            # Create visualization based on type
            if vis_type == 'bar':
                if len(columns) >= 2 and columns[0] in df.columns and columns[1] in df.columns:
                    # Use the first column as categorical and the second as numerical
                    x_col = columns[0]
                    y_col = columns[1]
                    
                    # Use aggregation if x has few unique values
                    if df[x_col].nunique() < 50:
                        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                        fig = px.bar(agg_df, x=x_col, y=y_col, title=title)
                    else:
                        # Sample data if there are too many unique values
                        sample_df = df.sample(min(len(df), 100))
                        fig = px.bar(sample_df, x=x_col, y=y_col, title=title)
                    
                    return fig, None
                else:
                    # Try to find appropriate columns
                    cat_cols = [col for col, dtype in df.dtypes.items() if dtype == 'object']
                    num_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(df[col])]
                    
                    if cat_cols and num_cols:
                        x_col = cat_cols[0]
                        y_col = num_cols[0]
                        
                        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                        fig = px.bar(agg_df, x=x_col, y=y_col, title=title)
                        return fig, None
                    else:
                        return None, "Could not find appropriate categorical and numerical columns for bar chart"
                    
            elif vis_type == 'line':
                if len(columns) >= 2 and all(col in df.columns for col in columns[:2]):
                    x_col = columns[0]
                    y_col = columns[1]
                    
                    # Sort by x if it's numeric
                    if pd.api.types.is_numeric_dtype(df[x_col]):
                        plot_df = df.sort_values(by=x_col)
                    else:
                        plot_df = df
                        
                    fig = px.line(plot_df, x=x_col, y=y_col, title=title)
                    return fig, None
                else:
                    # Try to find appropriate columns
                    num_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(df[col])]
                    
                    if len(num_cols) >= 2:
                        fig = px.line(df, x=num_cols[0], y=num_cols[1], title=title)
                        return fig, None
                    else:
                        return None, "Could not find enough numerical columns for line chart"
                    
            elif vis_type == 'scatter':
                if len(columns) >= 2 and all(col in df.columns for col in columns[:2]):
                    x_col = columns[0]
                    y_col = columns[1]
                    
                    # Add color if third column is specified
                    if len(columns) >= 3 and columns[2] in df.columns:
                        color_col = columns[2]
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, title=title)
                        
                    return fig, None
                else:
                    # Try to find appropriate columns
                    num_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(df[col])]
                    
                    if len(num_cols) >= 2:
                        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=title)
                        return fig, None
                    else:
                        return None, "Could not find enough numerical columns for scatter plot"
                    
            elif vis_type == 'histogram':
                if columns and columns[0] in df.columns:
                    x_col = columns[0]
                    fig = px.histogram(df, x=x_col, title=title)
                    return fig, None
                else:
                    # Try to find appropriate column
                    num_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(df[col])]
                    
                    if num_cols:
                        fig = px.histogram(df, x=num_cols[0], title=title)
                        return fig, None
                    else:
                        return None, "Could not find a numerical column for histogram"
                    
            else:
                return None, f"Unsupported visualization type: {vis_type}"
                
        except Exception as e:
            return None, f"Error creating visualization: {str(e)}"