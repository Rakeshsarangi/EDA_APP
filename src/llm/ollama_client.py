import ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, List, Optional, Tuple
import json
import pandas as pd
import numpy as np
from config.settings import settings
import traceback
import re

class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1
        )
    
    def generate_analysis_code_for_question(self, question: str, data_context: Dict) -> str:
        """Generate Python code to answer the data question."""
        
        prompt_template = """
        You are a data analysis expert. Generate Python code to answer the following question about a dataset.
        
        Dataset Information:
        - Columns: {columns}
        - Data Types: {dtypes}
        - Shape: {shape} 
        - Numeric Columns: {numeric_columns}
        - Categorical Columns: {categorical_columns}
        
        Question: {question}
        
        Generate Python code that:
        1. Takes a DataFrame 'df' as input
        2. Performs the necessary analysis to answer the question
        3. Returns a dictionary with 'answer' (string) and optionally 'data' (for tables/lists)
        4. Handles potential errors gracefully
        5. Provides clear, formatted output
        
        The code should follow this structure:
        ```python
        def analyze(df):
            try:
                # Your analysis code here
                # Compute the answer
                
                result = {{
                    'answer': 'Clear text answer to the question',
                    'data': None  # Optional: DataFrame, list, or dict for tabular results
                }}
                return result
            except Exception as e:
                return {{'answer': f'Error: {{str(e)}}', 'data': None}}
        ```
        
        Examples of good answers:
        - For "which customer has highest sales": {{'answer': 'Customer ABC has the highest sales with $123,456', 'data': top_5_customers_df}}
        - For "what is the average": {{'answer': 'The average value is 45.2', 'data': None}}
        - For "show me top 5": {{'answer': 'Here are the top 5 items:', 'data': top_5_df}}
        
        Return ONLY the function code, no explanations.
        """
        
        prompt = PromptTemplate(
            input_variables=["columns", "dtypes", "shape", "numeric_columns", 
                           "categorical_columns", "question"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            columns = data_context.get('columns', [])
            dtypes = data_context.get('dtypes', {})
            shape = data_context.get('shape', (0, 0))
            numeric_cols = data_context.get('numeric_columns', [])
            categorical_cols = data_context.get('categorical_columns', [])
            
            code = chain.run(
                columns=", ".join(columns) if columns else "No columns",
                dtypes=json.dumps(dtypes, indent=2) if dtypes else "{}",
                shape=f"{shape[0]} rows × {shape[1]} columns",
                numeric_columns=", ".join(numeric_cols) if numeric_cols else "None",
                categorical_columns=", ".join(categorical_cols) if categorical_cols else "None",
                question=question
            )
            
            return code.strip()
            
        except Exception as e:
            print(f"Error generating analysis code: {e}")
            # Return a fallback function
            return """
def analyze(df):
    return {
        'answer': 'Unable to generate analysis code. Please try rephrasing your question.',
        'data': None
    }
"""
    
    def execute_analysis_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Safely execute the generated analysis code."""
        
        # Create a safe namespace for execution
        namespace = {
            'pd': pd,
            'np': np,
            'df': df,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'round': round,
            'sum': sum,
            'min': min,
            'max': max,
            'list': list,
            'dict': dict,
            'set': set,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'range': range,
            'abs': abs,
            'Exception': Exception,
            'ValueError': ValueError,
            'KeyError': KeyError,
            'TypeError': TypeError,
        }
        
        try:
            # Clean the code
            code = code.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            # Execute the code
            exec(code, namespace)
            
            # Check if the analyze function exists
            if 'analyze' not in namespace:
                return {
                    'answer': 'Error: Analysis function not found in generated code.',
                    'data': None
                }
            
            # Call the analyze function
            result = namespace['analyze'](df)
            
            # Validate result format
            if not isinstance(result, dict):
                return {
                    'answer': 'Error: Analysis did not return expected format.',
                    'data': None
                }
            
            # Ensure required keys exist
            if 'answer' not in result:
                result['answer'] = 'Analysis completed but no answer text was provided.'
            if 'data' not in result:
                result['data'] = None
                
            return result
            
        except Exception as e:
            return {
                'answer': f'Error executing analysis: {str(e)}\n{traceback.format_exc()}',
                'data': None
            }
    
    def classify_question(self, question: str, data_context: Dict) -> Dict[str, Any]:
        """
        Classify whether a question needs code execution or can be answered from context.
        
        Returns:
            Dict with 'needs_execution' (bool), 'reasoning' (str), and 'direct_answer' (str, optional)
        """
        
        prompt_template = """
        Analyze this question about a dataset and determine if it requires code execution or can be answered directly from the dataset metadata.
        
        Dataset Context:
        - Columns: {columns}
        - Shape: {shape}
        - Data Types: {dtypes}
        - Numeric Columns: {numeric_columns}
        - Categorical Columns: {categorical_columns}
        - DateTime Columns: {date_columns}
        
        Question: {question}
        
        Classify the question into one of these categories:
        
        1. OBSERVATIONAL - Can be answered directly from metadata without code execution
        Examples: "How many rows?", "What columns exist?", "What's the data type of X?"
        
        2. COMPUTATIONAL - Requires code execution on the actual data
        Examples: "What's the average of X?", "Which customer has highest sales?", "Show top 10"
        
        Return a JSON object:
        {{
            "needs_execution": true/false,
            "category": "OBSERVATIONAL" or "COMPUTATIONAL",
            "reasoning": "Brief explanation of why",
            "direct_answer": "If OBSERVATIONAL, provide the answer here using the context. If COMPUTATIONAL, set to null"
        }}
        
        Return ONLY valid JSON, no other text.
        """
        
        prompt = PromptTemplate(
            input_variables=["columns", "shape", "dtypes", "numeric_columns", 
                        "categorical_columns", "date_columns", "question"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            columns = data_context.get('columns', [])
            shape = data_context.get('shape', (0, 0))
            dtypes = data_context.get('dtypes', {})
            numeric_cols = data_context.get('numeric_columns', [])
            categorical_cols = data_context.get('categorical_columns', [])
            date_cols = data_context.get('datetime_columns', [])
            
            response = chain.run(
                columns=", ".join(columns) if columns else "No columns",
                shape=f"{shape[0]} rows × {shape[1]} columns",
                dtypes=json.dumps(dtypes, indent=2) if dtypes else "{}",
                numeric_columns=", ".join(numeric_cols) if numeric_cols else "None",
                categorical_columns=", ".join(categorical_cols) if categorical_cols else "None",
                date_columns=", ".join(date_cols) if date_cols else "None",
                question=question
            )
            
            # Clean and parse response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            
            result = json.loads(response)
            return result
            
        except Exception as e:
            print(f"Error classifying question: {e}")
            # Default to execution if classification fails
            return {
                'needs_execution': True,
                'category': 'COMPUTATIONAL',
                'reasoning': 'Classification failed, defaulting to code execution',
                'direct_answer': None
            }

    def answer_data_question_with_execution(self, question: str, df: pd.DataFrame, 
                                         data_context: Dict) -> Tuple[str, Any, str]:
        """
        Answer question - either directly from context or by executing code.
        
        Returns:
            Tuple of (answer_text, data_result, code_used)
        """
        
        # First, classify the question
        classification = self.classify_question(question, data_context)
        
        # If it's observational and we have a direct answer, return it
        if not classification['needs_execution'] and classification['direct_answer']:
            answer_text = f"**{classification['direct_answer']}**\n\n"
            answer_text += f"_{classification['reasoning']}_"
            
            return answer_text, None, "# No code execution needed - answered from dataset metadata"
        
        # Otherwise, proceed with code generation and execution
        # Generate the analysis code
        code = self.generate_analysis_code_for_question(question, data_context)
        
        # Execute the code
        result = self.execute_analysis_code(code, df)
        
        # Format the answer
        answer_text = result.get('answer', 'Unable to compute answer.')
        data_result = result.get('data', None)
        
        # Add classification reasoning
        answer_text = f"**Analysis Type:** {classification['category']}\n\n{answer_text}"
        
        # If there's tabular data, format it nicely
        if data_result is not None:
            if isinstance(data_result, pd.DataFrame):
                if len(data_result) > 0:
                    answer_text += f"\n\n📊 **Results Table** ({len(data_result)} rows):"
            elif isinstance(data_result, dict):
                answer_text += "\n\n📊 **Results:**"
            elif isinstance(data_result, list):
                answer_text += f"\n\n📊 **Results** ({len(data_result)} items):"
        
        return answer_text, data_result, code
    
    def refine_user_question(self, question: str, data_context: Dict) -> str:
        """Refine user question to be more specific and contextual."""
        
        prompt_template = """
        You are a data analysis assistant. A user has asked a question about their dataset. 
        Your task is to rewrite their question to be more specific and clear in the context of the actual data.
        
        Dataset Information:
        - Columns: {columns}
        - Data Types: {dtypes}
        - Shape: {shape}
        - Numeric Columns: {numeric_columns}
        - Categorical Columns: {categorical_columns}
        - Date Columns: {date_columns}
        
        Original User Question: {question}
        
        Rewrite the question to be:
        1. More specific to the actual columns and data available
        2. Clearer in intent and expected output
        3. Technically precise while maintaining the user's original intent
        4. Include relevant column names where applicable
        
        Rules:
        - If the user mentions generic terms like "sales", "revenue", "customers", map them to actual column names if identifiable
        - If asking for analysis, specify what type (statistical, trend, comparison, etc.)
        - If the question is vague, make reasonable assumptions based on available data
        - Keep the refined question concise but complete
        - Don't change the fundamental intent of the question
        
        Return ONLY the refined question, no explanations or additional text.
        """
        
        prompt = PromptTemplate(
            input_variables=["columns", "dtypes", "shape", "numeric_columns", 
                           "categorical_columns", "date_columns", "question"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            # Prepare context
            columns = data_context.get('columns', [])
            dtypes = data_context.get('dtypes', {})
            shape = data_context.get('shape', (0, 0))
            numeric_cols = data_context.get('numeric_columns', [])
            categorical_cols = data_context.get('categorical_columns', [])
            date_cols = data_context.get('datetime_columns', [])
            
            refined_question = chain.run(
                columns=", ".join(columns) if columns else "No columns",
                dtypes=json.dumps(dtypes, indent=2) if dtypes else "{}",
                shape=f"{shape[0]} rows × {shape[1]} columns",
                numeric_columns=", ".join(numeric_cols) if numeric_cols else "None",
                categorical_columns=", ".join(categorical_cols) if categorical_cols else "None",
                date_columns=", ".join(date_cols) if date_cols else "None",
                question=question
            )
            
            return refined_question.strip()
            
        except Exception as e:
            print(f"Error refining question: {e}")
            # Return original question if refinement fails
            return question
    
    def suggest_related_questions(self, question: str, data_context: Dict) -> List[str]:
        """Suggest related questions based on the user's input and data context."""
        
        prompt_template = """
        Based on a user's question about their dataset, suggest 3 related follow-up questions they might want to ask.
        
        Dataset Information:
        - Columns: {columns}
        - Numeric Columns: {numeric_columns}
        - Categorical Columns: {categorical_columns}
        
        User's Question: {question}
        
        Generate 3 specific, relevant follow-up questions that:
        1. Explore different aspects of the same topic
        2. Use actual column names from the dataset
        3. Would provide valuable insights
        
        Return as a JSON array of strings. Example:
        ["Question 1", "Question 2", "Question 3"]
        
        Return ONLY the JSON array, no additional text.
        """
        
        prompt = PromptTemplate(
            input_variables=["columns", "numeric_columns", "categorical_columns", "question"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            columns = data_context.get('columns', [])
            numeric_cols = data_context.get('numeric_columns', [])
            categorical_cols = data_context.get('categorical_columns', [])
            
            response = chain.run(
                columns=", ".join(columns[:20]) if columns else "No columns",
                numeric_columns=", ".join(numeric_cols[:10]) if numeric_cols else "None",
                categorical_columns=", ".join(categorical_cols[:10]) if categorical_cols else "None",
                question=question
            )
            
            # Clean and parse response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            
            suggestions = json.loads(response)
            return suggestions[:3]  # Ensure we return exactly 3
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return []
    
    # Keep all other existing methods (analyze_and_generate_viz_code, etc.)
    def analyze_and_generate_viz_code(self, data_info: Dict, sample_data: str) -> List[Dict]:
        """Analyze data and generate multiple visualization codes."""
        
        prompt_template = """
        You are an expert data visualization specialist. Analyze the following dataset and generate Python code for creating insightful visualizations.
        
        Dataset Information:
        {data_info}
        
        Sample Data (first 5 rows):
        {sample_data}
        
        Based on this data, generate Python code for 8-10 different insightful visualizations that would best represent this data.
        
        IMPORTANT REQUIREMENTS:
        1. Use only plotly.graph_objects (go) or plotly.express (px) for visualizations
        2. Each visualization must be a complete, standalone function
        3. Function must accept a pandas DataFrame 'df' as parameter
        4. Function must return a plotly figure object
        5. Include error handling
        6. Make visualizations interactive and insightful
        7. Choose visualization types based on the actual data characteristics
        
        Return your response as a JSON array with the following structure:
        [
            {{
                "title": "Descriptive title for the visualization",
                "description": "Brief explanation of what this shows",
                "code": "def create_viz(df):\\n    # Complete function code here\\n    return fig",
                "priority": 1
            }},
            ...
        ]
        
        RESPOND ONLY WITH VALID JSON. No additional text or markdown.
        """
        
        prompt = PromptTemplate(
            input_variables=["data_info", "sample_data"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.run(
                data_info=json.dumps(data_info, indent=2, default=str),
                sample_data=sample_data
            )
            
            # Clean the response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            
            visualizations = json.loads(response)
            
            valid_visualizations = []
            for viz in visualizations:
                if all(key in viz for key in ["title", "description", "code"]):
                    if not viz["code"].startswith("def "):
                        viz["code"] = f"def create_viz(df):\n{viz['code']}"
                    valid_visualizations.append(viz)
            
            return valid_visualizations
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return self._get_fallback_visualizations()
    
    def _get_fallback_visualizations(self) -> List[Dict]:
        """Return fallback visualizations if AI generation fails."""
        return [
            {
                "title": "Data Distribution Overview",
                "description": "Overview of numeric column distributions",
                "code": """def create_viz(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4]
    if len(numeric_cols) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns found", x=0.5, y=0.5)
        return fig
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(numeric_cols))
    
    for idx, col in enumerate(numeric_cols):
        row = idx // 2 + 1
        col_idx = idx % 2 + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(title="Distribution Overview", height=600)
    return fig""",
                "priority": 1
            }
        ]
    
    def refine_visualization_code(self, code: str, error_message: str) -> str:
        """Refine visualization code if it encounters an error."""
        
        prompt_template = """
        Fix the following visualization code error:
        
        Code: {code}
        Error: {error_message}
        
        Return ONLY the corrected function code.
        """
        
        prompt = PromptTemplate(
            input_variables=["code", "error_message"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = chain.run(code=code, error_message=error_message)
            return response.strip()
        except Exception as e:
            print(f"Error refining code: {e}")
            return code
    
    def generate_eda_insights(self, data_summary: Dict) -> str:
        """Generate insights from EDA results."""
        prompt_template = """
        Provide key insights based on this data analysis:
        
        {data_summary}
        
        Include:
        1. Key findings
        2. Data quality issues
        3. Patterns or correlations
        4. Recommendations
        
        Be concise and actionable.
        """
        
        prompt = PromptTemplate(
            input_variables=["data_summary"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            summary_str = json.dumps(data_summary, indent=2, default=str)
            response = chain.run(data_summary=summary_str)
            return response
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def suggest_analysis_code(self, data_info: Dict, analysis_type: str) -> str:
        """Generate Python code for specific analysis."""
        prompt_template = """
        Generate Python code for {analysis_type} analysis:
        
        Data Info: {data_info}
        
        Use pandas, numpy, and plotly. Include error handling and comments.
        
        Return only Python code:
        """
        
        prompt = PromptTemplate(
            input_variables=["data_info", "analysis_type"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            info_str = json.dumps(data_info, indent=2, default=str)
            response = chain.run(data_info=info_str, analysis_type=analysis_type)
            return response
        except Exception as e:
            return f"# Error generating code: {str(e)}"