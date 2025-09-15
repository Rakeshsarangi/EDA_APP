import streamlit as st
import pandas as pd
import numpy as np
from src.data_handler.loader import DataLoader
from src.data_handler.validator import DataValidator
from src.eda.analyzer import EDAAnalyzer
from src.eda.visualizer import Visualizer
from src.llm.ollama_client import OllamaClient
from src.utils.code_executor import SafeCodeExecutor, VisualizationOptimizer
import plotly.graph_objects as go
from typing import Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Local EDA Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = OllamaClient()

def main():
    st.title("🔒 Secure Local Data Analysis Assistant")
    st.markdown("Upload your data and get comprehensive EDA with AI-powered insights - all running locally!")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files up to 100MB"
        )
        
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading data..."):
                df = DataLoader.load_file(uploaded_file)
                
            if df is not None:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
                
                # Data validation
                with st.spinner("Validating data..."):
                    df_cleaned, issues = DataValidator.validate_and_clean(df)
                    st.session_state.df = df_cleaned
                
                # Display validation results
                if issues['missing_values']:
                    st.warning(f"⚠️ Missing values detected in {len(issues['missing_values'])} columns")
                if issues['duplicates'] > 0:
                    st.warning(f"⚠️ {issues['duplicates']} duplicate rows found")
                if issues['outliers']:
                    st.info(f"📊 Outliers detected in {len(issues['outliers'])} columns")
    
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Visualizations", 
            "🤖 AI Insights", 
            "💬 Q&A",
            "🔧 Generated Code"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.header("Dataset Overview")
            
            # Initialize analyzer
            analyzer = EDAAnalyzer(df)
            basic_stats = analyzer.get_basic_stats()
            
            # Display basic information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", basic_stats['shape'][0])
            with col2:
                st.metric("Columns", basic_stats['shape'][1])
            with col3:
                st.metric("Memory Usage", f"{basic_stats['memory_usage']:.2f} MB")
            with col4:
                st.metric("Numeric Columns", len(basic_stats['numeric_columns']))
            
            # Display first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info)
            
            # Numeric statistics
            if basic_stats['numeric_columns']:
                st.subheader("Numeric Column Statistics")
                st.dataframe(df[basic_stats['numeric_columns']].describe())
            
            # Categorical statistics
            if basic_stats['categorical_columns']:
                st.subheader("Categorical Column Summary")
                cat_analysis = analyzer.get_categorical_analysis()
                for col, stats in cat_analysis.items():
                    with st.expander(f"📝 {col}"):
                        st.write(f"Unique values: {stats['unique_values']}")
                        st.write(f"Most frequent: {stats['most_frequent']}")
                        st.write("Top 5 values:")
                        st.write(stats['top_5'])
        
        # Tab 2: Visualizations
        with tab2:
            st.header("AI-Powered Data Visualizations")
            
            # Create sub-tabs
            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "🤖 AI-Generated Visualizations", 
                "📊 Standard Analysis", 
                "📝 Generated Code"
            ])
            
            # AI-Generated Visualizations Tab
            with viz_tab1:
                st.subheader("AI-Generated Custom Visualizations")
                st.info("Click 'Generate Visualizations' to let AI analyze your data and create custom visualizations")
                
                # Generate visualizations button
                if st.button("🎨 Generate Visualizations", type="primary", key="gen_viz"):
                    with st.spinner("AI is analyzing your data and generating visualization code..."):
                        
                        # Prepare data info for AI
                        data_info = {
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'dtypes': df.dtypes.astype(str).to_dict(),
                            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
                            'missing_values': df.isnull().sum().to_dict(),
                            'unique_values': {col: df[col].nunique() for col in df.columns},
                            'basic_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
                        }
                        
                        # Get sample data
                        sample_data = df.head(5).to_string()
                        
                        # Generate visualization codes using AI
                        visualizations = st.session_state.ollama_client.analyze_and_generate_viz_code(
                            data_info, 
                            sample_data
                        )
                        
                        # Store in session state
                        st.session_state.ai_visualizations = visualizations
                        
                    # Execute the generated codes
                    if 'ai_visualizations' in st.session_state and st.session_state.ai_visualizations:
                        with st.spinner("Executing visualization code..."):
                            executor = SafeCodeExecutor()
                            optimizer = VisualizationOptimizer()
                            
                            # Execute all visualizations
                            results = executor.execute_multiple_visualizations(
                                st.session_state.ai_visualizations, 
                                df
                            )
                            
                            # Store results
                            st.session_state.viz_results = results
                
                # Display generated visualizations
                if 'viz_results' in st.session_state:
                    results = st.session_state.viz_results
                    
                    # Show success metrics
                    successful = sum(1 for r in results if r['success'])
                    total = len(results)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Visualizations", total)
                    with col2:
                        st.metric("Successfully Generated", successful)
                    with col3:
                        st.metric("Success Rate", f"{(successful/total*100):.1f}%")
                    
                    # Display each visualization
                    for idx, result in enumerate(results):
                        st.markdown(f"### {idx+1}. {result['title']}")
                        
                        if result['description']:
                            st.markdown(f"*{result['description']}*")
                        
                        if result['success'] and result['figure']:
                            # Enhance and display the figure
                            enhanced_fig = VisualizationOptimizer.enhance_figure(
                                result['figure'], 
                                result['title']
                            )
                            st.plotly_chart(enhanced_fig, use_container_width=True)
                            
                            # Show code in expander
                            with st.expander("View Generated Code"):
                                st.code(result['code'], language='python')
                        else:
                            st.error(f"Failed to generate: {result['message']}")
                            
                            # Offer to retry with refined code
                            if st.button(f"🔄 Retry with AI Fix", key=f"retry_{idx}"):
                                with st.spinner("AI is fixing the code..."):
                                    # Get refined code
                                    refined_code = st.session_state.ollama_client.refine_visualization_code(
                                        result['code'],
                                        result['message']
                                    )
                                    
                                    # Try executing refined code
                                    executor = SafeCodeExecutor()
                                    fig, message = executor.execute_visualization_code(refined_code, df)
                                    
                                    if fig:
                                        st.success("Fixed successfully!")
                                        enhanced_fig = VisualizationOptimizer.enhance_figure(fig, result['title'])
                                        st.plotly_chart(enhanced_fig, use_container_width=True)
                                        
                                        with st.expander("View Fixed Code"):
                                            st.code(refined_code, language='python')
                                    else:
                                        st.error(f"Still failed: {message}")
                        
                        st.divider()
                
                # Regenerate button
                if 'viz_results' in st.session_state:
                    if st.button("🔄 Generate New Visualizations", key="regen_viz"):
                        # Clear previous results
                        if 'ai_visualizations' in st.session_state:
                            del st.session_state.ai_visualizations
                        if 'viz_results' in st.session_state:
                            del st.session_state.viz_results
                        st.rerun()
            
            # Standard Analysis Tab (keep existing code)
            with viz_tab2:
                st.subheader("Standard Statistical Visualizations")
                
                visualizer = Visualizer(df)
                
                # Missing data plot
                st.subheader("Missing Data Analysis")
                missing_plot = visualizer.create_missing_data_plot()
                if missing_plot:
                    st.plotly_chart(missing_plot, use_container_width=True)
                else:
                    st.info("No missing data found!")
                
                # Distribution plots
                if basic_stats['numeric_columns']:
                    st.subheader("Distribution Analysis")
                    dist_plots = visualizer.create_distribution_plots(basic_stats['numeric_columns'])
                    
                    # Create columns for plots
                    cols = st.columns(2)
                    for idx, (col_name, fig) in enumerate(dist_plots.items()):
                        with cols[idx % 2]:
                            st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                if len(basic_stats['numeric_columns']) > 1:
                    st.subheader("Correlation Analysis")
                    corr_matrix = analyzer.get_correlation_matrix()
                    corr_plot = visualizer.create_correlation_heatmap(corr_matrix)
                    if corr_plot:
                        st.plotly_chart(corr_plot, use_container_width=True)
            
            # Generated Code Tab
            with viz_tab3:
                st.subheader("All Generated Visualization Codes")
                
                if 'viz_results' in st.session_state:
                    st.info("Here are all the visualization codes generated by AI for your dataset")
                    
                    # Create a combined code file
                    all_codes = []
                    for idx, result in enumerate(st.session_state.viz_results):
                        code_block = f"""
        # Visualization {idx+1}: {result['title']}
        # {result['description']}
        {result['code']}

        """
                        all_codes.append(code_block)
                    
                    combined_code = "import plotly.graph_objects as go\nimport plotly.express as px\nimport pandas as pd\nimport numpy as np\n\n" + "\n".join(all_codes)
                    
                    # Display code
                    st.code(combined_code, language='python')
                    
                    # Download button
                    st.download_button(
                        label="📥 Download All Visualization Code",
                        data=combined_code,
                        file_name="ai_generated_visualizations.py",
                        mime="text/plain"
                    )
                    
                    # Individual code blocks
                    st.markdown("### Individual Visualization Codes")
                    for idx, result in enumerate(st.session_state.viz_results):
                        with st.expander(f"{idx+1}. {result['title']}"):
                            st.markdown(f"**Description:** {result['description']}")
                            st.markdown(f"**Status:** {'✅ Success' if result['success'] else '❌ Failed'}")
                            st.code(result['code'], language='python')
                            
                            # Individual download button
                            st.download_button(
                                label=f"Download {result['title']}.py",
                                data=result['code'],
                                file_name=f"viz_{idx+1}_{result['title'].lower().replace(' ', '_')}.py",
                                mime="text/plain",
                                key=f"download_{idx}"
                            )
                else:
                    st.warning("No visualizations generated yet. Go to 'AI-Generated Visualizations' tab and click 'Generate Visualizations'")
        
        # Tab 3: AI Insights
        with tab3:
            st.header("AI-Powered Insights")
            
            if st.button("🔍 Generate Insights", type="primary"):
                with st.spinner("Analyzing data with AI..."):
                    # Prepare data summary for LLM
                    data_summary = {
                        'basic_stats': basic_stats,
                        'numeric_analysis': analyzer.get_numeric_analysis() if basic_stats['numeric_columns'] else {},
                        'categorical_analysis': analyzer.get_categorical_analysis() if basic_stats['categorical_columns'] else {},
                        'patterns': analyzer.detect_patterns(),
                        'sample_data': df.head(10).to_dict()
                    }
                    
                    # Generate insights
                    insights = st.session_state.ollama_client.generate_eda_insights(data_summary)
                    
                    st.markdown("### 💡 Key Insights")
                    st.write(insights)
                    
                    # Store results
                    st.session_state.eda_results = data_summary
        
        # Tab 4: Q&A
        with tab4:
            st.header("Ask Questions About Your Data")
            
            question = st.text_input("Enter your question:")
            
            if st.button("Get Answer") and question:
                with st.spinner("Thinking..."):
                    # Prepare context
                    if st.session_state.eda_results:
                        context = st.session_state.eda_results
                    else:
                        context = {
                            'columns': list(df.columns),
                            'shape': df.shape,
                            'dtypes': df.dtypes.astype(str).to_dict(),
                            'sample': df.head(10).to_dict(),
                            'describe': df.describe().to_dict() if basic_stats['numeric_columns'] else {}
                        }
                    
                    answer = st.session_state.ollama_client.answer_data_question(question, context)
                    
                    st.markdown("### 📝 Answer")
                    st.write(answer)
        
        # Tab 5: Generated Code
        with tab5:
            st.header("Generate Analysis Code")
            
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Basic EDA", "Statistical Tests", "Feature Engineering", 
                 "Outlier Detection", "Time Series Analysis", "Custom"]
            )
            
            if analysis_type == "Custom":
                custom_request = st.text_area("Describe the analysis you want:")
            
            if st.button("Generate Code"):
                with st.spinner("Generating code..."):
                    data_info = {
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.astype(str).to_dict(),
                        'shape': df.shape,
                        'numeric_columns': basic_stats['numeric_columns'],
                        'categorical_columns': basic_stats['categorical_columns']
                    }
                    
                    if analysis_type == "Custom":
                        analysis_request = custom_request if 'custom_request' in locals() else analysis_type
                    else:
                        analysis_request = analysis_type
                    
                    code = st.session_state.ollama_client.suggest_analysis_code(
                        data_info, 
                        analysis_request
                    )
                    
                    st.markdown("### 📝 Generated Python Code")
                    st.code(code, language='python')
                    
                    # Add download button for code
                    st.download_button(
                        label="Download Code",
                        data=code,
                        file_name=f"{analysis_type.lower().replace(' ', '_')}_analysis.py",
                        mime="text/plain"
                    )
    
    else:
        # Landing page when no data is loaded
        st.info("👈 Please upload a data file from the sidebar to begin analysis")
        
        # Features section
        st.markdown("## Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔒 Completely Local
            - All processing happens on your machine
            - No data leaves your computer
            - Uses Ollama for local LLM inference
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Comprehensive EDA
            - Automatic statistical analysis
            - Interactive visualizations
            - Pattern detection
            """)
        
        with col3:
            st.markdown("""
            ### 🤖 AI-Powered
            - Natural language Q&A
            - Automated insights generation
            - Custom analysis code generation
            """)

if __name__ == "__main__":
    main()