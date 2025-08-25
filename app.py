import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import traceback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)

# Try to import custom agents and utilities with fallbacks
try:
    from agents.ingest_agent import IngestAgent
    from agents.profiling_agent import ProfilingAgent
    from agents.relations_agent import RelationsAgent
    from agents.target_task_agent import TargetTaskAgent
    from graph.pipeline import pipeline
    from utils.io_utils import safe_read_file, get_sample_datasets, DatasetContext
    AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Custom agents not available: {e}")
    AGENTS_AVAILABLE = False
    
    # Fallback implementations
    class DatasetContext:
        def __init__(self, df, filename):
            self.df = df
            self.filename = filename
            self.warnings = []
            self.target_column = None
            self.task_type = None
    
    def safe_read_file(filepath, max_rows=None):
        """Fallback file reader"""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, nrows=max_rows)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath, nrows=max_rows)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
                if max_rows:
                    df = df.head(max_rows)
            else:
                return None, "Unsupported file format"
            return df, None
        except Exception as e:
            return None, str(e)
    
    def get_sample_datasets():
        """Fallback sample datasets"""
        # Create some sample datasets
        np.random.seed(42)
        
        # Iris-like dataset
        iris_data = {
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'sepal_width': np.random.normal(3.0, 0.4, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'petal_width': np.random.normal(1.2, 0.8, 150),
            'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
        }
        
        # Housing-like dataset
        housing_data = {
            'bedrooms': np.random.randint(1, 6, 500),
            'bathrooms': np.random.randint(1, 4, 500),
            'sqft': np.random.normal(2000, 800, 500),
            'age': np.random.randint(0, 50, 500),
            'price': np.random.normal(300000, 150000, 500)
        }
        
        return {
            'Iris Dataset': pd.DataFrame(iris_data),
            'Housing Dataset': pd.DataFrame(housing_data)
        }

# Page config
st.set_page_config(
    page_title="Auto-EDA & ML Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def basic_profiling(df):
    """Basic profiling fallback when agents aren't available"""
    results = {
        'status': 'success',
        'basic_stats': {},
        'visualizations': {}
    }
    
    # Basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    results['basic_stats'] = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(categorical_cols),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    # Missing values visualization
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        results['visualizations']['missing'] = fig
    
    # Distribution plots for numeric columns
    if len(numeric_cols) > 0:
        # Create subplots for distributions
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=list(numeric_cols[:n_rows*n_cols]),
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(numeric_cols[:9]):  # Max 9 plots
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title_text="Distribution of Numeric Variables",
            showlegend=False,
            height=300 * n_rows
        )
        results['visualizations']['distributions'] = fig
    
    return results

def basic_relations_analysis(df):
    """Basic relationships analysis fallback"""
    results = {
        'status': 'success',
        'relationship_summary': {},
        'correlations': {},
        'visualizations': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    results['relationship_summary'] = {
        'total_features': len(df.columns),
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols),
        'key_findings': []
    }
    
    # Correlation analysis for numeric columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    strong_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        results['correlations']['pearson'] = {
            'matrix': corr_matrix,
            'strong_pairs': strong_pairs
        }
        
        # Correlation heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig.update_layout(width=700, height=600)
        results['visualizations']['correlation_heatmaps'] = {
            'pearson_heatmap': fig
        }
        
        if strong_pairs:
            results['relationship_summary']['key_findings'].append(
                f"Found {len(strong_pairs)} strong correlations (|r| ≥ 0.7)"
            )
    
    return results

def basic_target_detection(df):
    """Basic target detection fallback"""
    results = {
        'status': 'success',
        'target_candidates': [],
        'recommended_target': None,
        'confidence_score': 0,
        'reasoning': []
    }
    
    # Simple heuristics for target detection
    candidates = []
    
    for col in df.columns:
        score = 0
        reasons = []
        
        # Check if column name suggests it's a target
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'prediction', 
                          'price', 'salary', 'income', 'survived', 'diagnosis', 'species']
        
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                score += 30
                reasons.append(f"Column name contains '{keyword}'")
                break
        
        # Check data characteristics
        unique_ratio = df[col].nunique() / len(df)
        
        if df[col].dtype in ['object', 'category']:
            if unique_ratio < 0.1:  # Low cardinality categorical
                score += 20
                reasons.append("Low cardinality categorical (good for classification)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            if unique_ratio > 0.8:  # High cardinality numeric
                score += 15
                reasons.append("High cardinality numeric (good for regression)")
        
        # Position bias - often targets are last columns
        if df.columns.get_loc(col) >= len(df.columns) - 3:
            score += 10
            reasons.append("Located near end of dataset")
        
        candidates.append({
            'column': col,
            'score': score,
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique(),
            'unique_ratio': unique_ratio,
            'missing_count': df[col].isnull().sum()
        })
    
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    results['target_candidates'] = candidates
    
    if candidates[0]['score'] > 20:
        results['recommended_target'] = candidates[0]['column']
        results['confidence_score'] = min(candidates[0]['score'] * 2, 100)
        
        # Infer task type
        target_col = candidates[0]['column']
        if df[target_col].dtype in ['object', 'category'] or candidates[0]['unique_ratio'] < 0.1:
            results['task_type'] = 'classification'
        else:
            results['task_type'] = 'regression'
        
        # Add reasoning
        results['reasoning'] = [
            f"Column '{target_col}' scored highest ({candidates[0]['score']} points)",
            f"Detected as {results['task_type']} task",
            f"Has {candidates[0]['unique_values']} unique values"
        ]
        
        # Target analysis
        results['target_analysis'] = analyze_target_column(df, target_col, results['task_type'])
    
    return results

def analyze_target_column(df, target_col, task_type):
    """Analyze the detected target column"""
    analysis = {
        'target_stats': {},
        'recommendations': []
    }
    
    target_series = df[target_col]
    
    # Basic stats
    analysis['target_stats'] = {
        'unique_values': target_series.nunique(),
        'missing_values': target_series.isnull().sum(),
        'missing_percentage': (target_series.isnull().sum() / len(target_series)) * 100,
        'data_type': str(target_series.dtype)
    }
    
    if task_type == 'classification':
        # Class distribution
        class_dist = target_series.value_counts().to_dict()
        analysis['target_stats']['class_distribution'] = class_dist
        
        # Class balance
        class_counts = list(class_dist.values())
        balance_ratio = min(class_counts) / max(class_counts) if class_counts else 1.0
        analysis['target_stats']['class_balance_ratio'] = balance_ratio
        
        if balance_ratio < 0.3:
            analysis['recommendations'].append(
                "⚠️ Significant class imbalance detected. Consider using stratified sampling, "
                "SMOTE, or class weights in your models."
            )
        
        if len(class_dist) > 10:
            analysis['recommendations'].append(
                "🔍 Many classes detected. Consider grouping rare classes or using "
                "hierarchical classification approaches."
            )
    
    elif task_type == 'regression':
        # Distribution stats
        analysis['target_stats'].update({
            'mean': target_series.mean(),
            'std': target_series.std(),
            'min': target_series.min(),
            'max': target_series.max(),
            'skewness': target_series.skew()
        })
        
        if abs(target_series.skew()) > 1:
            analysis['recommendations'].append(
                "📊 Target variable is skewed. Consider log transformation or "
                "other normalization techniques."
            )
        
        if target_series.std() / target_series.mean() > 1:
            analysis['recommendations'].append(
                "📈 High variance in target. Consider scaling or robust regression methods."
            )
    
    # Missing values
    if analysis['target_stats']['missing_percentage'] > 5:
        analysis['recommendations'].append(
            "❌ Significant missing values in target. Consider imputation strategies "
            "or removing incomplete records."
        )
    
    return analysis

# Initialize agents if available
if AGENTS_AVAILABLE and 'agents_initialized' not in st.session_state:
    try:
        pipeline.add_agent('ingest', IngestAgent())
        pipeline.add_agent('profiling', ProfilingAgent())
        pipeline.add_agent('relations', RelationsAgent())
        pipeline.add_agent('target_task', TargetTaskAgent())
        st.session_state.agents_initialized = True
    except Exception as e:
        st.warning(f"Could not initialize agents: {e}")
        AGENTS_AVAILABLE = False

def display_target_analysis(target_results, context):
    """Display target detection and task inference results"""
    
    st.subheader("🎯 Target Detection & Task Inference")
    
    if target_results.get('recommended_target'):
        target_col = target_results['recommended_target']
        confidence = target_results['confidence_score']
        task_type = target_results.get('task_type', 'Unknown')
        
        # Main results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Detected Target", target_col)
        with col2:
            st.metric("📊 Task Type", task_type.title())
        with col3:
            st.metric("🔍 Confidence", f"{confidence:.1f}/100")
        
        # Target analysis details
        target_analysis = target_results.get('target_analysis', {})
        if target_analysis:
            st.subheader("📈 Target Analysis")
            
            # Basic stats
            stats = target_analysis.get('target_stats', {})
            if stats:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Basic Statistics:**")
                    st.write(f"• Unique values: {stats.get('unique_values', 'N/A')}")
                    st.write(f"• Missing values: {stats.get('missing_values', 'N/A')} ({stats.get('missing_percentage', 0):.1f}%)")
                    st.write(f"• Data type: {stats.get('data_type', 'N/A')}")
                
                with col2:
                    if task_type == 'classification':
                        class_dist = stats.get('class_distribution', {})
                        if class_dist:
                            st.write("**Class Distribution:**")
                            for class_val, count in list(class_dist.items())[:5]:  # Show top 5
                                st.write(f"• {class_val}: {count}")
                                
                            balance_ratio = stats.get('class_balance_ratio', 1.0)
                            if balance_ratio < 0.3:
                                st.warning(f"⚠️ Class imbalance detected (ratio: {balance_ratio:.2f})")
                    
                    elif task_type == 'regression':
                        st.write("**Distribution Statistics:**")
                        st.write(f"• Mean: {stats.get('mean', 0):.2f}")
                        st.write(f"• Std Dev: {stats.get('std', 0):.2f}")
                        st.write(f"• Range: [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]")
                        st.write(f"• Skewness: {stats.get('skewness', 0):.2f}")
            
            # Reasoning
            reasoning = target_results.get('reasoning', [])
            if reasoning:
                st.write("**Detection Reasoning:**")
                for reason in reasoning[:5]:  # Show top 5 reasons
                    st.write(f"• {reason}")
            
            # Recommendations
            recommendations = target_analysis.get('recommendations', [])
            if recommendations:
                st.write("**📋 Recommendations:**")
                for rec in recommendations:
                    st.info(rec)
        
        # Manual target override option
        st.subheader("🔧 Manual Target Override")
        
        current_target_idx = list(context.df.columns).index(target_col) if target_col in context.df.columns else 0
        
        selected_target = st.selectbox(
            "Choose target column:",
            options=list(context.df.columns),
            index=current_target_idx,
            format_func=lambda x: f"{x} ({context.df[x].dtype}, {context.df[x].nunique()} unique)"
        )
        
        if selected_target != target_col:
            st.info(f"Target changed to: {selected_target}")
            # In a full implementation, you'd re-run the analysis here
            st.rerun()
    
    else:
        st.warning("No target column could be automatically detected")
        
        # Show all candidates
        candidates = target_results.get('target_candidates', [])
        if candidates:
            st.write("**Top Target Candidates:**")
            candidates_df = pd.DataFrame(candidates)
            st.dataframe(candidates_df.head(10), use_container_width=True)

def display_relations_analysis(relations_results):
    """Display relationship analysis results"""
    
    st.subheader("🔗 Feature Relationships")
    
    # Summary
    summary = relations_results.get('relationship_summary', {})
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", summary.get('total_features', 0))
        with col2:
            st.metric("Numeric Features", summary.get('numeric_features', 0))
        with col3:
            st.metric("Categorical Features", summary.get('categorical_features', 0))
        with col4:
            st.metric("Key Findings", len(summary.get('key_findings', [])))
    
    # Key findings
    key_findings = summary.get('key_findings', [])
    if key_findings:
        st.write("**🔍 Key Findings:**")
        for finding in key_findings:
            st.info(finding)
    
    # Correlations section
    correlations = relations_results.get('correlations', {})
    if correlations:
        st.subheader("🔢 Correlations")
        
        for corr_type, corr_data in correlations.items():
            if 'matrix' in corr_data:
                st.write(f"**{corr_type.title()} Correlation:**")
                
                # Show visualization
                viz = relations_results.get('visualizations', {})
                heatmap_key = f'{corr_type}_heatmap'
                if 'correlation_heatmaps' in viz and heatmap_key in viz['correlation_heatmaps']:
                    st.plotly_chart(viz['correlation_heatmaps'][heatmap_key], use_container_width=True)
                
                # Strong correlations table
                strong_pairs = corr_data.get('strong_pairs', [])
                if strong_pairs:
                    st.write(f"**Strong {corr_type.title()} Correlations (|r| ≥ 0.7):**")
                    pairs_df = pd.DataFrame(strong_pairs)
                    st.dataframe(pairs_df, use_container_width=True)
                else:
                    st.info(f"No strong {corr_type} correlations found")

def main():
    st.title("🤖 Auto-EDA & ML Model Advisor")
    st.markdown("*AI-Agent powered Exploratory Data Analysis and Model Suggestions*")
    
    # Show agent status
    if not AGENTS_AVAILABLE:
        st.warning("⚠️ Custom agents not available. Using fallback analysis methods.")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # File upload or sample data selection
    data_source = st.sidebar.radio("Data Source:", ["Upload File", "Sample Datasets"])
    
    df = None
    filename = ""
    
    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'parquet'],
            help="Upload CSV, Excel, or Parquet file"
        )
        
        if uploaded_file is not None:
            filename = uploaded_file.name
            
            # Save uploaded file temporarily
            temp_path = Path("temp") / filename
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read file
            max_rows = st.sidebar.number_input("Max rows to load", value=50000, min_value=100)
            df, error = safe_read_file(str(temp_path), max_rows=max_rows)
            
            if error:
                st.error(f"Error reading file: {error}")
                return
                
    else:
        # Sample datasets
        sample_datasets = get_sample_datasets()
        dataset_name = st.sidebar.selectbox("Choose Sample Dataset:", list(sample_datasets.keys()))
        
        if dataset_name:
            df = sample_datasets[dataset_name]
            filename = f"{dataset_name.lower().replace(' ', '_')}.csv"
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    run_profiling = st.sidebar.checkbox("Generate Profiling Reports", value=True)
    run_relations = st.sidebar.checkbox("Analyze Relationships", value=True)
    run_target_detection = st.sidebar.checkbox("Auto-detect Target", value=True)
    enable_llm = st.sidebar.checkbox("Enable LLM Advisor", value=False, 
                                   help="Requires Ollama setup")
    
    # Main content
    if df is not None and not df.empty:
        st.success(f"📊 Dataset loaded: **{filename}** | Shape: {df.shape}")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Dataset Overview", 
            "📊 Profiling", 
            "🔗 Relationships",
            "🎯 Target & Task",
            "📝 Reports"
        ])
        
        with tab1:
            st.subheader("Dataset Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
                st.metric("Columns", df.shape[1])
            with col2:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory", f"{memory_mb:.1f} MB")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Basic info
            st.subheader("Column Information")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing': df.isnull().sum().values,
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values,
                'Unique': [df[col].nunique() for col in df.columns],
                'Unique %': [(df[col].nunique() / len(df) * 100) for col in df.columns]
            })
            st.dataframe(info_df, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Data Profiling")
            
            if st.button("🚀 Run Analysis", type="primary", key="run_analysis"):
                with st.spinner("Running analysis..."):
                    try:
                        # Create initial context
                        context = DatasetContext(df, filename)
                        
                        if AGENTS_AVAILABLE:
                            # Use full agent pipeline if available
                            agents_to_run = ['ingest']
                            if run_profiling:
                                agents_to_run.append('profiling')
                            if run_relations:
                                agents_to_run.append('relations')
                            if run_target_detection:
                                agents_to_run.append('target_task')
                            
                            results, updated_context = pipeline.run(context, agents_to_run)
                        else:
                            # Use fallback methods
                            results = {}
                            if run_profiling:
                                results['profiling'] = basic_profiling(df)
                            if run_relations:
                                results['relations'] = basic_relations_analysis(df)
                            if run_target_detection:
                                results['target_task'] = basic_target_detection(df)
                            updated_context = context
                        
                        # Store in session state
                        st.session_state.results = results
                        st.session_state.context = updated_context
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.code(traceback.format_exc())
                        return
                
                st.success("✅ Analysis complete!")
                st.rerun()
            
            # Display results if available
            if 'results' in st.session_state and 'context' in st.session_state:
                results = st.session_state.results
                context = st.session_state.context
                
                # Show warnings
                if hasattr(context, 'warnings') and context.warnings:
                    st.subheader("⚠️ Data Quality Warnings")
                    for warning in context.warnings:
                        st.warning(warning)
                
                # Show profiling results
                if 'profiling' in results and results['profiling'].get('status') == 'success':
                    profiling_results = results['profiling']
                    
                    # Basic statistics
                    if 'basic_stats' in profiling_results:
                        stats = profiling_results['basic_stats']
                        st.subheader("📊 Dataset Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Rows", f"{stats.get('total_rows', 0):,}")
                        with col2:
                            st.metric("Total Columns", stats.get('total_cols', 0))
                        with col3:
                            st.metric("Numeric Columns", stats.get('numeric_cols', 0))
                        with col4:
                            st.metric("Categorical Columns", stats.get('categorical_cols', 0))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Missing Values", f"{stats.get('missing_values', 0):,}")
                        with col2:
                            st.metric("Missing %", f"{stats.get('missing_percentage', 0):.1f}%")
                    
                    # Visualizations
                    if 'visualizations' in profiling_results:
                        viz = profiling_results['visualizations']
                        
                        if 'missing' in viz:
                            st.plotly_chart(viz['missing'], use_container_width=True)
                        
                        if 'distributions' in viz:
                            st.subheader("Distribution Plots")
                            st.plotly_chart(viz['distributions'], use_container_width=True)
                    
                    # HTML reports (if available)
                    if profiling_results.get('html_reports'):
                        st.subheader("📄 Generated Reports")
                        html_reports = profiling_results['html_reports']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'ydata' in html_reports:
                                st.success("✅ YData Profiling Report Generated")
                                try:
                                    with open(html_reports['ydata'], 'rb') as f:
                                        st.download_button(
                                            "📊 Download YData Report",
                                            f,
                                            file_name="ydata_profile.html",
                                            mime="text/html"
                                        )
                                except:
                                    st.error("Error loading YData report file")
                        
                        with col2:
                            if 'sweetviz' in html_reports:
                                st.success("✅ Sweetviz Report Generated")
                                try:
                                    with open(html_reports['sweetviz'], 'rb') as f:
                                        st.download_button(
                                            "📈 Download Sweetviz Report",
                                            f,
                                            file_name="sweetviz_report.html",
                                            mime="text/html"
                                        )
                                except:
                                    st.error("Error loading Sweetviz report file")
        
        with tab3:
            st.subheader("🔗 Feature Relationships")
            
            # Display results if available
            if 'results' in st.session_state and 'relations' in st.session_state.results:
                relations_results = st.session_state.results['relations']
                if relations_results.get('status') == 'success':
                    display_relations_analysis(relations_results)
                else:
                    st.error(f"Relations analysis failed: {relations_results.get('error', 'Unknown error')}")
            else:
                st.info("🚧 Run analysis first to see relationship insights")
        
        with tab4:
            st.subheader("🎯 Target Detection & Task Inference")
            
            # Display results if available
            if 'results' in st.session_state and 'target_task' in st.session_state.results:
                target_results = st.session_state.results['target_task']
                context = st.session_state.context
                
                if target_results.get('status') == 'success':
                    display_target_analysis(target_results, context)
                else:
                    st.error(f"Target detection failed: {target_results.get('error', 'Unknown error')}")
            else:
                st.info("🚧 Run analysis first to detect target column and task type")
        
        with tab5:
            st.subheader("📝 Analysis Report")
            
            if 'results' in st.session_state and 'context' in st.session_state:
                # Generate comprehensive report
                results = st.session_state.results
                context = st.session_state.context
                
                st.write("## Dataset Analysis Summary")
                
                # Dataset overview
                st.write(f"**Dataset:** {context.filename}")
                st.write(f"**Shape:** {context.df.shape[0]:,} rows × {context.df.shape[1]} columns")
                st.write(f"**Memory Usage:** {context.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                # Target and task summary
                if 'target_task' in results and results['target_task'].get('status') == 'success':
                    target_info = results['target_task']
                    if target_info.get('recommended_target'):
                        st.write(f"**Target Column:** {target_info['recommended_target']}")
                        st.write(f"**Task Type:** {target_info.get('task_type', 'Unknown').title()}")
                        st.write(f"**Confidence:** {target_info.get('confidence_score', 0):.1f}/100")
                
                # Key findings
                st.subheader("Key Findings")
                findings = []
                
                # Data quality findings
                missing_pct = (context.df.isnull().sum().sum() / (context.df.shape[0] * context.df.shape[1])) * 100
                if missing_pct > 10:
                    findings.append(f"⚠️ Dataset has {missing_pct:.1f}% missing values")
                elif missing_pct == 0:
                    findings.append("✅ No missing values detected")
                
                # Column type findings
                numeric_cols = len(context.df.select_dtypes(include=['number']).columns)
                categorical_cols = len(context.df.select_dtypes(include=['object', 'category']).columns)
                findings.append(f"📊 {numeric_cols} numeric and {categorical_cols} categorical features")
                
                # Relationship findings
                if 'relations' in results and results['relations'].get('status') == 'success':
                    rel_summary = results['relations'].get('relationship_summary', {})
                    key_findings = rel_summary.get('key_findings', [])
                    findings.extend(key_findings)
                
                # Display findings
                for finding in findings:
                    st.write(f"• {finding}")
                
                # Warnings
                if hasattr(context, 'warnings') and context.warnings:
                    st.subheader("⚠️ Warnings")
                    for warning in context.warnings:
                        st.warning(warning)
                
                # Recommendations
                st.subheader("📋 Recommendations")
                recommendations = [
                    "Consider feature engineering for categorical variables with high cardinality",
                    "Check for outliers in numeric features before modeling",
                    "Validate data quality and handle missing values appropriately"
                ]
                
                # Add task-specific recommendations
                if 'target_task' in results and results['target_task'].get('status') == 'success':
                    target_analysis = results['target_task'].get('target_analysis', {})
                    target_recommendations = target_analysis.get('recommendations', [])
                    recommendations.extend(target_recommendations)
                
                for rec in recommendations:
                    st.write(f"• {rec}")
                
                # Export options
                st.subheader("📁 Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generate CSV report
                    if st.button("📊 Export Summary CSV"):
                        summary_data = {
                            'Metric': ['Rows', 'Columns', 'Missing %', 'Numeric Columns', 'Categorical Columns'],
                            'Value': [
                                context.df.shape[0],
                                context.df.shape[1],
                                f"{missing_pct:.1f}%",
                                numeric_cols,
                                categorical_cols
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            "Download Summary CSV",
                            csv,
                            file_name=f"{context.filename.split('.')[0]}_summary.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    # Generate full report
                    if st.button("📄 Export Full Report"):
                        # Create detailed report content
                        report_content = f"""# Dataset Analysis Report
                        
## Dataset Overview
- **Filename:** {context.filename}
- **Shape:** {context.df.shape[0]:,} rows × {context.df.shape[1]} columns
- **Memory Usage:** {context.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
- **Missing Data:** {missing_pct:.1f}%

## Column Information
"""
                        # Add column details
                        for col in context.df.columns:
                            dtype = context.df[col].dtype
                            missing = context.df[col].isnull().sum()
                            unique = context.df[col].nunique()
                            report_content += f"- **{col}:** {dtype}, {missing} missing, {unique} unique\n"
                        
                        # Add findings
                        report_content += "\n## Key Findings\n"
                        for finding in findings:
                            report_content += f"- {finding}\n"
                        
                        # Add recommendations
                        report_content += "\n## Recommendations\n"
                        for rec in recommendations:
                            report_content += f"- {rec}\n"
                        
                        st.download_button(
                            "Download Full Report",
                            report_content,
                            file_name=f"{context.filename.split('.')[0]}_analysis_report.md",
                            mime="text/markdown"
                        )
            else:
                st.info("🚧 Run analysis first to generate comprehensive report")
    
    else:
        st.info("👆 Please upload a dataset or select a sample dataset to begin analysis")
        
        # Show sample datasets preview
        st.subheader("Available Sample Datasets")
        sample_datasets = get_sample_datasets()
        
        for name, sample_df in sample_datasets.items():
            with st.expander(f"📊 {name} Dataset"):
                st.write(f"**Shape:** {sample_df.shape}")
                st.write(f"**Columns:** {list(sample_df.columns)}")
                st.dataframe(sample_df.head(3))

if __name__ == "__main__":
    main()