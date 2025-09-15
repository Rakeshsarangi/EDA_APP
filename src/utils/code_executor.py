import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import traceback
import ast
import sys
from io import StringIO
from contextlib import contextmanager
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

@contextmanager
def capture_output():
    """Capture stdout and stderr."""
    old_out, old_err = sys.stdout, sys.stderr
    try:
        out = StringIO()
        err = StringIO()
        sys.stdout, sys.stderr = out, err
        yield out, err
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class SafeCodeExecutor:
    """Safely execute generated visualization code."""
    
    def __init__(self):
        self.safe_namespace = self._create_safe_namespace()
    
    def _create_safe_namespace(self) -> Dict:
        """Create a safe namespace for code execution."""
        import math
        import statistics
        from datetime import datetime, timedelta
        
        namespace = {
            # Plotly imports
            'go': go,
            'px': px,
            'Figure': go.Figure,
            'make_subplots': make_subplots,
            
            # Data manipulation
            'pd': pd,
            'np': np,
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            
            # Math and statistics
            'math': math,
            'statistics': statistics,
            
            # Datetime
            'datetime': datetime,
            'timedelta': timedelta,
            
            # Built-in functions
            'len': len,
            'range': range,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'any': any,
            'all': all,
            'print': print,
            'isinstance': isinstance,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'repr': repr,
            
            # Exception handling
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'ImportError': ImportError,
            'NameError': NameError,
            'ZeroDivisionError': ZeroDivisionError,
        }
        
        return namespace
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to handle common patterns."""
        lines = code.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip import statements as they're already available in namespace
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                # Check if it's trying to import allowed modules
                if any(module in line for module in ['plotly', 'pandas', 'numpy', 'math', 'statistics', 'datetime']):
                    continue  # Skip the import line
                else:
                    # Replace with a comment
                    processed_lines.append(f"    # Skipped: {line.strip()}")
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code for safety before execution."""
        try:
            # Parse the code to check for syntax errors
            tree = ast.parse(code)
            
            # Check for forbidden operations
            forbidden_names = {
                'eval', 'exec', 'compile', '__import__', 'open', 'file', 
                'input', 'raw_input', 'execfile', 'reload', 'vars', 'globals', 
                'locals', 'dir', 'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests'
            }
            
            for node in ast.walk(tree):
                # Check for forbidden function calls
                if isinstance(node, ast.Name):
                    if node.id in forbidden_names:
                        return False, f"Forbidden name used: {node.id}"
                
                # Check for file operations
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['open', 'file']:
                            return False, "File operations not allowed"
                
                # Check for attribute access to forbidden modules
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in ['os', 'sys', 'subprocess']:
                            return False, f"Access to {node.value.id} not allowed"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def execute_visualization_code(self, code: str, df: pd.DataFrame) -> Tuple[Optional[go.Figure], str]:
        """Execute visualization code safely and return the figure."""
        
        # Preprocess the code
        processed_code = self._preprocess_code(code)
        
        # Validate code first
        is_valid, validation_msg = self.validate_code(processed_code)
        if not is_valid:
            return None, f"Code validation failed: {validation_msg}"
        
        # Prepare the namespace with the dataframe
        local_namespace = self.safe_namespace.copy()
        local_namespace['df'] = df
        
        try:
            # Capture any output
            with capture_output() as (out, err):
                # Execute the code with our safe namespace as both global and local
                exec(processed_code, local_namespace, local_namespace)
                
                # Look for the function (usually create_viz or similar)
                viz_func = None
                for name, obj in local_namespace.items():
                    if callable(obj) and name not in self.safe_namespace and not name.startswith('_'):
                        viz_func = obj
                        break
                
                if viz_func is None:
                    return None, "No visualization function found in the code"
                
                # Call the function with the dataframe
                fig = viz_func(df)
                
                # Validate the output
                if not isinstance(fig, (go.Figure, type(None))):
                    return None, f"Function must return a plotly Figure, got {type(fig)}"
                
                # Get any printed output
                output = out.getvalue()
                error_output = err.getvalue()
                
                if error_output:
                    return fig, f"Warnings: {error_output}"
                
                return fig, "Success"
                
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
    
    def execute_multiple_visualizations(self, visualizations: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Execute multiple visualization codes and return results."""
        results = []
        
        for viz in visualizations:
            try:
                # Clean up the code before execution
                code = viz['code']
                
                # If the code has import statements at the beginning, move them inside the function
                if 'import' in code:
                    lines = code.split('\n')
                    new_lines = []
                    for line in lines:
                        if line.strip().startswith('def '):
                            new_lines.append(line)
                            # Add a comment about imports being available
                            new_lines.append("    # All necessary imports are already available")
                        elif not (line.strip().startswith('import ') or line.strip().startswith('from ')):
                            new_lines.append(line)
                    code = '\n'.join(new_lines)
                
                fig, message = self.execute_visualization_code(code, df)
                
                results.append({
                    'title': viz.get('title', 'Untitled'),
                    'description': viz.get('description', ''),
                    'figure': fig,
                    'success': fig is not None,
                    'message': message,
                    'code': viz['code']
                })
                
            except Exception as e:
                results.append({
                    'title': viz.get('title', 'Untitled'),
                    'description': viz.get('description', ''),
                    'figure': None,
                    'success': False,
                    'message': str(e),
                    'code': viz['code']
                })
        
        return results

class VisualizationOptimizer:
    """Optimize and enhance generated visualizations."""
    
    @staticmethod
    def enhance_figure(fig: go.Figure, title: str = None) -> go.Figure:
        """Enhance a plotly figure with better styling."""
        if fig is None:
            return None
        
        # Update layout for better appearance
        fig.update_layout(
            title=title if title else fig.layout.title.text,
            template='plotly_white',
            hovermode='closest',
            showlegend=True,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(size=12),
            title_font=dict(size=16, color='#2c3e50'),
        )
        
        # Update axes if they exist
        try:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#cccccc'
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                showline=True,
                linewidth=1,
                linecolor='#cccccc'
            )
        except:
            pass  # Some figures might not have traditional axes
        
        return fig