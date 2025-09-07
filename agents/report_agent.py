import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class ReportAgent:
    """Agent for generating comprehensive analysis reports"""
    
    def __init__(self):
        self.name = "Report Generation Agent"
    
    def process(self, context, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive HTML/Markdown report
        
        Args:
            context: DatasetContext
            all_results: Combined results from all agents
            
        Returns:
            Dict with report paths and metadata
        """
        try:
            # Create reports directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = Path("reports") / timestamp
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate reports
            html_report = self._generate_html_report(context, all_results, reports_dir)
            markdown_report = self._generate_markdown_report(context, all_results, reports_dir)
            json_summary = self._generate_json_summary(context, all_results, reports_dir)
            
            # Create index file linking to all resources
            index_file = self._create_index_file(context, all_results, reports_dir)
            
            return {
                'status': 'success',
                'reports_directory': str(reports_dir),
                'html_report': html_report,
                'markdown_report': markdown_report,
                'json_summary': json_summary,
                'index_file': index_file,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'reports_directory': None
            }
    
    def _generate_html_report(self, context, all_results: Dict, reports_dir: Path) -> str:
        """Generate comprehensive HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auto-EDA Analysis Report - {context.filename}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metric {{ display: inline-block; background: #ecf0f1; padding: 15px; margin: 5px; border-radius: 8px; min-width: 120px; text-align: center; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
        .success {{ background: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }}
        .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .section {{ margin: 30px 0; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
        .recommendation {{ background: #e8f4fd; border: 1px solid #bee5eb; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Auto-EDA Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Dataset:</strong> {context.filename}</p>
        
        {self._generate_html_executive_summary(context, all_results)}
        {self._generate_html_dataset_overview(context)}
        {self._generate_html_data_quality_section(context, all_results)}
        {self._generate_html_modeling_results(all_results)}
        {self._generate_html_recommendations(all_results)}
        {self._generate_html_technical_details(all_results)}
        
        <div class="section">
            <h2>📁 Generated Artifacts</h2>
            {self._generate_html_artifacts_section(all_results)}
        </div>
    </div>
</body>
</html>
"""
        
        html_file = reports_dir / "analysis_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_file)
    
    def _generate_html_executive_summary(self, context, all_results: Dict) -> str:
        """Generate executive summary section"""
        # Extract key metrics
        shape = context.df.shape
        missing_pct = (context.df.isnull().sum().sum() / (shape[0] * shape[1])) * 100
        
        # Model performance
        best_model = "Not available"
        best_score = "N/A"
        task_type = context.task_type or "Not determined"
        
        if all_results.get('baseline', {}).get('status') == 'success':
            baseline_results = all_results['baseline']
            best_model = baseline_results.get('best_model', 'Unknown')
            models = baseline_results.get('models', {})
            if best_model in models:
                best_score = f"{models[best_model].get('primary_score', 0):.3f}"
        
        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
                <div class="metric">
                    <div class="metric-label">Rows</div>
                    <div class="metric-value">{shape[0]:,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Features</div>
                    <div class="metric-value">{shape[1]}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Missing %</div>
                    <div class="metric-value">{missing_pct:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Task Type</div>
                    <div class="metric-value" style="font-size: 18px;">{task_type.title()}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Model</div>
                    <div class="metric-value" style="font-size: 16px;">{best_model}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Score</div>
                    <div class="metric-value">{best_score}</div>
                </div>
            </div>
            
            <div class="info">
                <strong>Key Findings:</strong>
                <ul>
                    <li>Dataset contains {shape[0]:,} samples with {shape[1]} features</li>
                    <li>Task identified as: <strong>{task_type}</strong></li>
                    <li>Target column: <strong>{context.target_column or 'Not specified'}</strong></li>
                    <li>Best performing model: <strong>{best_model}</strong></li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_html_dataset_overview(self, context) -> str:
        """Generate dataset overview section"""
        df = context.df
        
        # Column types summary
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = df.select_dtypes(include=['object']).shape[1]
        
        # Missing data summary
        missing_summary = df.isnull().sum().sort_values(ascending=False)
        high_missing = missing_summary[missing_summary > len(df) * 0.1]  # >10% missing
        
        overview_html = f"""
        <div class="section">
            <h2>📋 Dataset Overview</h2>
            
            <h3>Column Types</h3>
            <div style="display: flex; gap: 20px;">
                <div class="metric">
                    <div class="metric-label">Numeric</div>
                    <div class="metric-value">{numeric_cols}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Categorical</div>
                    <div class="metric-value">{categorical_cols}</div>
                </div>
            </div>
        """
        
        if len(high_missing) > 0:
            overview_html += f"""
            <h3>⚠️ Columns with High Missing Data (>10%)</h3>
            <table>
                <tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>
            """
            for col, missing_count in high_missing.head(10).items():
                missing_pct = (missing_count / len(df)) * 100
                overview_html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{missing_count:,}</td>
                    <td>{missing_pct:.1f}%</td>
                </tr>
                """
            overview_html += "</table>"
        
        overview_html += "</div>"
        return overview_html
    
    def _generate_html_data_quality_section(self, context, all_results: Dict) -> str:
        """Generate data quality analysis section"""
        quality_html = """
        <div class="section">
            <h2>🔍 Data Quality Analysis</h2>
        """
        
        # Warnings from context
        if context.warnings:
            quality_html += "<h3>⚠️ Quality Warnings</h3>"
            for warning in context.warnings:
                quality_html += f'<div class="warning">{warning}</div>'
        
        # Relations analysis results
        if all_results.get('relations', {}).get('status') == 'success':
            relations = all_results['relations']
            summary = relations.get('relationship_summary', {})
            
            if summary.get('key_findings'):
                quality_html += "<h3>🔗 Key Relationship Findings</h3>"
                for finding in summary['key_findings']:
                    quality_html += f'<div class="info">{finding}</div>'
            
            if summary.get('recommendations'):
                quality_html += "<h3>📋 Data Quality Recommendations</h3>"
                for rec in summary['recommendations']:
                    quality_html += f'<div class="warning">{rec}</div>'
        
        quality_html += "</div>"
        return quality_html
    
    def _generate_html_modeling_results(self, all_results: Dict) -> str:
        """Generate modeling results section"""
        modeling_html = """
        <div class="section">
            <h2>🤖 Modeling Results</h2>
        """
        
        baseline_results = all_results.get('baseline', {})
        if baseline_results.get('status') == 'success':
            models = baseline_results.get('models', {})
            best_model = baseline_results.get('best_model')
            
            # Model comparison table
            modeling_html += """
            <h3>📊 Model Performance Comparison</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Primary Score</th>
                    <th>Training Time (s)</th>
                    <th>Status</th>
                </tr>
            """
            
            for model_name, metrics in models.items():
                if 'error' not in metrics:
                    primary_score = metrics.get('primary_score', 0)
                    training_time = metrics.get('training_time', 0)
                    status = "✅ Success"
                    row_class = "style='background-color: #d4edda;'" if model_name == best_model else ""
                else:
                    primary_score = "Error"
                    training_time = "-"
                    status = "❌ Failed"
                    row_class = "style='background-color: #f8d7da;'"
                
                modeling_html += f"""
                <tr {row_class}>
                    <td><strong>{model_name}</strong></td>
                    <td>{primary_score if isinstance(primary_score, str) else f'{primary_score:.4f}'}</td>
                    <td>{f"{training_time:.2f}" if isinstance(training_time, (int, float)) else training_time}</td>
                    <td>{status}</td>
                </tr>
                """
            
            modeling_html += "</table>"
            
            # Best model details
            if best_model and best_model in models:
                best_metrics = models[best_model]
                modeling_html += f"""
                <div class="success">
                    <h4>🏆 Best Model: {best_model}</h4>
                    <p><strong>Primary Score:</strong> {best_metrics.get('primary_score', 0):.4f}</p>
                    <p><strong>Cross-validation Score:</strong> {best_metrics.get('cv_score', 0):.4f}</p>
                </div>
                """
        else:
            modeling_html += '<div class="warning">Modeling analysis was not successful or not performed.</div>'
        
        # Feature importance
        explain_results = all_results.get('explain', {})
        if explain_results.get('status') == 'success':
            importance = explain_results.get('feature_importance', {}).get('permutation', {})
            if importance and 'importances' in importance:
                top_features = importance['importances'][:10]
                
                modeling_html += """
                <h3>⭐ Top Important Features</h3>
                <table>
                    <tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Std Dev</th></tr>
                """
                
                for i, feature_info in enumerate(top_features, 1):
                    modeling_html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{feature_info['feature']}</td>
                        <td>{feature_info['importance_mean']:.4f}</td>
                        <td>±{feature_info['importance_std']:.4f}</td>
                    </tr>
                    """
                
                modeling_html += "</table>"
        
        modeling_html += "</div>"
        return modeling_html
    
    def _generate_html_recommendations(self, all_results: Dict) -> str:
        """Generate recommendations section"""
        rec_html = """
        <div class="section">
            <h2>💡 Recommendations & Next Steps</h2>
        """
        
        advisor_results = all_results.get('advisor', {})
        if advisor_results.get('status') == 'success':
            recommendations = advisor_results.get('final_recommendations', {})
            
            for section, recs in recommendations.items():
                if recs and section != 'ai_insights':  # Handle AI insights separately
                    section_title = section.replace('_', ' ').title()
                    rec_html += f"<h3>📌 {section_title}</h3>"
                    
                    for rec in recs:
                        rec_html += f'<div class="recommendation">{rec}</div>'
            
            # AI insights (if available)
            if recommendations.get('ai_insights'):
                rec_html += "<h3>🤖 AI-Generated Insights</h3>"
                insights = recommendations['ai_insights'][0]  # First (and likely only) insight
                rec_html += f'<div class="info" style="white-space: pre-line;">{insights}</div>'
        
        else:
            rec_html += '<div class="info">Detailed recommendations are not available.</div>'
        
        rec_html += "</div>"
        return rec_html
    
    def _generate_html_technical_details(self, all_results: Dict) -> str:
        """Generate technical details section"""
        tech_html = """
        <div class="section">
            <h2>🔧 Technical Details</h2>
        """
        
        # Processing summary
        processing_summary = []
        for agent_name, results in all_results.items():
            if isinstance(results, dict) and 'status' in results:
                status_icon = "✅" if results['status'] == 'success' else "❌"
                duration = results.get('duration', 0)
                processing_summary.append({
                    'agent': agent_name.title(),
                    'status': results['status'],
                    'icon': status_icon,
                    'duration': f"{duration:.2f}s" if duration else "N/A"
                })
        
        if processing_summary:
            tech_html += """
            <h3>🔄 Processing Summary</h3>
            <table>
                <tr><th>Agent</th><th>Status</th><th>Duration</th></tr>
            """
            for item in processing_summary:
                tech_html += f"""
                <tr>
                    <td>{item['icon']} {item['agent']}</td>
                    <td>{item['status'].title()}</td>
                    <td>{item['duration']}</td>
                </tr>
                """
            tech_html += "</table>"
        
        # Preprocessing info
        baseline_results = all_results.get('baseline', {})
        if baseline_results.get('preprocessing_info'):
            prep_info = baseline_results['preprocessing_info']
            tech_html += """
            <h3>⚙️ Data Preprocessing</h3>
            <ul>
            """
            for transformer in prep_info.get('transformers', []):
                tech_html += f"<li><strong>{transformer['name']}:</strong> {transformer['columns']} columns ({transformer['type']})</li>"
            tech_html += "</ul>"
        
        tech_html += "</div>"
        return tech_html
    
    def _generate_html_artifacts_section(self, all_results: Dict) -> str:
        """Generate artifacts section with download links"""
        artifacts_html = "<ul>"
        
        # Profiling reports
        profiling_results = all_results.get('profiling', {})
        if profiling_results.get('html_reports'):
            reports = profiling_results['html_reports']
            if 'ydata' in reports:
                artifacts_html += f'<li>📊 <a href="{reports["ydata"]}">YData Profiling Report</a></li>'
            if 'sweetviz' in reports:
                artifacts_html += f'<li>📈 <a href="{reports["sweetviz"]}">Sweetviz Report</a></li>'
        
        # Add links to current report files
        artifacts_html += '<li>📋 <a href="analysis_report.html">HTML Analysis Report</a> (this file)</li>'
        artifacts_html += '<li>📝 <a href="analysis_report.md">Markdown Report</a></li>'
        artifacts_html += '<li>🔧 <a href="analysis_summary.json">JSON Summary</a></li>'
        
        artifacts_html += "</ul>"
        return artifacts_html
    
    def _generate_markdown_report(self, context, all_results: Dict, reports_dir: Path) -> str:
        """Generate Markdown report"""
        md_content = f"""# 🤖 Auto-EDA Analysis Report

**Dataset:** {context.filename}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Executive Summary

- **Rows:** {context.df.shape[0]:,}
- **Features:** {context.df.shape[1]}
- **Target Column:** {context.target_column or 'Not specified'}
- **Task Type:** {context.task_type or 'Not determined'}

## 🔍 Key Findings

{self._generate_markdown_findings(context, all_results)}

## 🤖 Modeling Results

{self._generate_markdown_modeling_results(all_results)}

## ⭐ Feature Importance

{self._generate_markdown_feature_importance(all_results)}

## 💡 Recommendations

{self._generate_markdown_recommendations(all_results)}

## 📁 Generated Files

- `analysis_report.html` - Interactive HTML report
- `analysis_report.md` - This markdown report
- `analysis_summary.json` - Machine-readable summary

---
*Report generated by Auto-EDA Agent System*
"""
        
        md_file = reports_dir / "analysis_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(md_file)
    
    def _generate_markdown_findings(self, context, all_results: Dict) -> str:
        """Generate key findings for markdown"""
        findings = []
        
        # Data quality findings
        missing_pct = (context.df.isnull().sum().sum() / (context.df.shape[0] * context.df.shape[1])) * 100
        if missing_pct > 10:
            findings.append(f"- ⚠️ **High missing data:** {missing_pct:.1f}% of values are missing")
        
        if context.warnings:
            findings.append(f"- 🔍 **Data quality issues:** {len(context.warnings)} warnings detected")
        
        # Modeling findings
        baseline_results = all_results.get('baseline', {})
        if baseline_results.get('status') == 'success':
            best_model = baseline_results.get('best_model')
            if best_model:
                findings.append(f"- 🏆 **Best model:** {best_model}")
        
        return '\n'.join(findings) if findings else "- ✅ No major issues detected"
    
    def _generate_markdown_modeling_results(self, all_results: Dict) -> str:
        """Generate modeling results for markdown"""
        baseline_results = all_results.get('baseline', {})
        if baseline_results.get('status') != 'success':
            return "Modeling analysis was not successful."
        
        models = baseline_results.get('models', {})
        best_model = baseline_results.get('best_model')
        
        results_md = f"**Best Model:** {best_model}\n\n"
        results_md += "| Model | Score | Training Time |\n"
        results_md += "|-------|-------|---------------|\n"
        
        for model_name, metrics in models.items():
            if 'error' not in metrics:
                score = metrics.get('primary_score', 0)
                time = metrics.get('training_time', 0)
                marker = " 🏆" if model_name == best_model else ""
                results_md += f"| {model_name}{marker} | {score:.4f} | {time:.2f}s |\n"
        
        return results_md
    
    def _generate_markdown_feature_importance(self, all_results: Dict) -> str:
        """Generate feature importance for markdown"""
        explain_results = all_results.get('explain', {})
        if explain_results.get('status') != 'success':
            return "Feature importance analysis not available."
        
        importance = explain_results.get('feature_importance', {}).get('permutation', {})
        if not importance or 'importances' not in importance:
            return "Feature importance results not found."
        
        top_features = importance['importances'][:10]
        
        importance_md = "| Rank | Feature | Importance |\n"
        importance_md += "|------|---------|------------|\n"
        
        for i, feature_info in enumerate(top_features, 1):
            importance_md += f"| {i} | {feature_info['feature']} | {feature_info['importance_mean']:.4f} |\n"
        
        return importance_md
    
    def _generate_markdown_recommendations(self, all_results: Dict) -> str:
        """Generate recommendations for markdown"""
        advisor_results = all_results.get('advisor', {})
        if advisor_results.get('status') != 'success':
            return "Recommendations not available."
        
        recommendations = advisor_results.get('final_recommendations', {})
        rec_md = ""
        
        for section, recs in recommendations.items():
            if recs and section != 'ai_insights':
                section_title = section.replace('_', ' ').title()
                rec_md += f"\n### {section_title}\n\n"
                for rec in recs:
                    rec_md += f"- {rec}\n"
        
        return rec_md
    
    def _generate_json_summary(self, context, all_results: Dict, reports_dir: Path) -> str:
        """Generate JSON summary for programmatic access"""
        summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_filename': context.filename,
                'dataset_shape': context.df.shape,
                'target_column': context.target_column,
                'task_type': context.task_type
            },
            'data_quality': {
                'missing_data_pct': (context.df.isnull().sum().sum() / (context.df.shape[0] * context.df.shape[1])) * 100,
                'warnings_count': len(context.warnings),
                'warnings': context.warnings
            },
            'modeling': {},
            'feature_importance': {},
            'recommendations': {}
        }
        
        # Add modeling results
        baseline_results = all_results.get('baseline', {})
        if baseline_results.get('status') == 'success':
            summary['modeling'] = {
                'best_model': baseline_results.get('best_model'),
                'models_trained': len([m for m in baseline_results.get('models', {}).values() if 'error' not in m]),
                'models_failed': len([m for m in baseline_results.get('models', {}).values() if 'error' in m])
            }
        
        # Add feature importance
        explain_results = all_results.get('explain', {})
        if explain_results.get('status') == 'success':
            importance = explain_results.get('feature_importance', {}).get('permutation', {})
            if importance and 'importances' in importance:
                top_5 = importance['importances'][:5]
                summary['feature_importance'] = {
                    'method': 'permutation_importance',
                    'top_features': [f['feature'] for f in top_5],
                    'top_scores': [f['importance_mean'] for f in top_5]
                }
        
        # Add recommendations count
        advisor_results = all_results.get('advisor', {})
        if advisor_results.get('status') == 'success':
            recommendations = advisor_results.get('final_recommendations', {})
            summary['recommendations'] = {
                'sections': list(recommendations.keys()),
                'total_recommendations': sum(len(recs) for recs in recommendations.values())
            }
        
        json_file = reports_dir / "analysis_summary.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return str(json_file)
    
    def _create_index_file(self, context, all_results: Dict, reports_dir: Path) -> str:
        """Create index file linking to all generated reports"""
        index_content = f"""# Auto-EDA Analysis Results

**Dataset:** {context.filename}  
**Analysis Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📋 Available Reports

### Main Reports
- [📊 Interactive HTML Report](analysis_report.html) - Comprehensive analysis with visualizations
- [📝 Markdown Report](analysis_report.md) - Text-based summary
- [🔧 JSON Summary](analysis_summary.json) - Machine-readable results

### Detailed Profiling Reports
"""
        
        # Add profiling report links if available
        profiling_results = all_results.get('profiling', {})
        if profiling_results.get('html_reports'):
            reports = profiling_results['html_reports']
            if 'ydata' in reports:
                ydata_name = Path(reports['ydata']).name
                index_content += f"- [📊 YData Profiling Report]({ydata_name}) - Comprehensive data profiling\n"
            if 'sweetviz' in reports:
                sweetviz_name = Path(reports['sweetviz']).name
                index_content += f"- [📈 Sweetviz Report]({sweetviz_name}) - Visual data analysis\n"
        
        index_content += f"""
## 📊 Quick Summary

- **Rows:** {context.df.shape[0]:,}
- **Features:** {context.df.shape[1]}
- **Task:** {context.task_type or 'Not determined'}
- **Target:** {context.target_column or 'Not specified'}
"""
        
        # Add modeling summary if available
        baseline_results = all_results.get('baseline', {})
        if baseline_results.get('status') == 'success':
            best_model = baseline_results.get('best_model')
            if best_model:
                index_content += f"- **Best Model:** {best_model}\n"
        
        index_file = reports_dir / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        return str(index_file)
