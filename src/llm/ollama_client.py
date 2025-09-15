import ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, List
import json
from config.settings import settings

class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1
        )
    
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
        
        Focus on creating diverse, meaningful visualizations such as:
        - Distribution plots for numeric data
        - Correlation matrices or networks
        - Time series if datetime columns exist
        - Geographic plots if location data exists
        - Categorical comparisons
        - Outlier detection plots
        - Trend analysis
        - Statistical summaries
        - Interactive 3D plots if appropriate
        
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
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            
            # Parse JSON
            visualizations = json.loads(response)
            
            # Validate and clean the visualizations
            valid_visualizations = []
            for viz in visualizations:
                if all(key in viz for key in ["title", "description", "code"]):
                    # Ensure the code is properly formatted
                    if not viz["code"].startswith("def "):
                        viz["code"] = f"def create_viz(df):\n{viz['code']}"
                    valid_visualizations.append(viz)
            
            return valid_visualizations
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response was: {response[:500] if 'response' in locals() else 'No response'}...")
            return self._get_fallback_visualizations()
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
            },
            {
                "title": "Correlation Heatmap",
                "description": "Correlation matrix of numeric columns",
                "code": """def create_viz(df):
    import plotly.graph_objects as go
    import pandas as pd
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation", x=0.5, y=0.5)
        return fig
    
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(title="Correlation Matrix", height=600)
    return fig""",
                "priority": 2
            },
            {
                "title": "Top Categories Bar Chart",
                "description": "Most frequent values in categorical columns",
                "code": """def create_viz(df):
    import plotly.graph_objects as go
    import pandas as pd
    
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No categorical columns found", x=0.5, y=0.5)
        return fig
    
    # Use first categorical column
    col = cat_cols[0]
    value_counts = df[col].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(x=value_counts.index, y=value_counts.values)
    ])
    
    fig.update_layout(
        title=f"Top 10 Categories in {col}",
        xaxis_title=col,
        yaxis_title="Count",
        height=500
    )
    return fig""",
                "priority": 3
            }
        ]
    
    def refine_visualization_code(self, code: str, error_message: str) -> str:
        """Refine visualization code if it encounters an error."""
        
        prompt_template = """
        The following visualization code encountered an error. Please fix it.
        
        Original Code:
        {code}
        
        Error Message:
        {error_message}
        
        Please provide the corrected code that:
        1. Fixes the error
        2. Uses only plotly (go or px) for visualization
        3. Accepts a DataFrame 'df' as parameter
        4. Returns a plotly figure
        5. Includes proper error handling
        
        Return ONLY the corrected function code, no explanations.
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
        You are a data analyst expert. Based on the following data analysis results, 
        provide key insights and recommendations.
        
        Data Summary:
        {data_summary}
        
        Please provide:
        1. Key findings from the data
        2. Potential data quality issues
        3. Interesting patterns or correlations
        4. Recommendations for further analysis
        
        Keep your response concise and actionable.
        
        Insights:
        """
        
        prompt = PromptTemplate(
            input_variables=["data_summary"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            # Convert data summary to string
            summary_str = json.dumps(data_summary, indent=2, default=str)
            response = chain.run(data_summary=summary_str)
            return response
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def answer_data_question(self, question: str, data_context: Dict) -> str:
        """Answer questions about the data."""
        prompt_template = """
        You are a data analyst assistant. Answer the following question based on the data context provided.
        
        Data Context:
        {data_context}
        
        Question: {question}
        
        Provide a clear, concise answer based on the data. If the answer cannot be determined from 
        the given context, explain what additional analysis would be needed.
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["data_context", "question"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            context_str = json.dumps(data_context, indent=2, default=str)
            response = chain.run(data_context=context_str, question=question)
            return response
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def suggest_analysis_code(self, data_info: Dict, analysis_type: str) -> str:
        """Generate Python code for specific analysis."""
        prompt_template = """
        Generate Python code for {analysis_type} analysis based on the following data information:
        
        Data Info:
        {data_info}
        
        Requirements:
        1. Use pandas, numpy, and plotly for analysis and visualization
        2. Include proper error handling
        3. Add comments explaining each step
        4. Make the code production-ready
        
        Generate only the Python code without any explanations:
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