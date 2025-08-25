import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging

# Optional LLM support (Ollama via LangChain)
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.schema import HumanMessage
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ChatOllama = None

logger = logging.getLogger(__name__)

class AdvisorAgent:
    """Agent that provides ML advice based on analysis results"""
    
    def __init__(self, use_llm: bool = False, llm_model: str = "llama3.1:8b"):
        self.name = "AI Advisor Agent"
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_model = llm_model
        self.llm = None
        
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize local LLM if available"""
        try:
            self.llm = ChatOllama(model=self.llm_model, temperature=0.1)
            logger.info(f"✅ LLM initialized: {self.llm_model}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {str(e)}")
            self.use_llm = False
            self.llm = None
    
    def process(self, context, baseline_results: Dict, explain_results: Dict, 
                relations_results: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive ML advice based on all analysis results
        
        Args:
            context: DatasetContext
            baseline_results: Results from BaselineAgent
            explain_results: Results from ExplainAgent  
            relations_results: Results from RelationsAgent
            
        Returns:
            Dict with advice, recommendations, and insights
        """
        try:
            # Collect analysis findings
            findings = self._collect_findings(
                context, baseline_results, explain_results, relations_results
            )
            
            # Generate rule-based advice
            rule_based_advice = self._generate_rule_based_advice(findings)
            
            # Generate LLM-enhanced advice if available
            llm_advice = {}
            if self.use_llm and self.llm:
                llm_advice = self._generate_llm_advice(findings)
            
            # Combine and structure final advice
            final_advice = self._combine_advice(rule_based_advice, llm_advice)
            
            return {
                'status': 'success',
                'findings_summary': findings,
                'rule_based_advice': rule_based_advice,
                'llm_enhanced_advice': llm_advice,
                'final_recommendations': final_advice,
                'llm_used': self.use_llm and bool(llm_advice)
            }
            
        except Exception as e:
            logger.error(f"Advisor agent failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'final_recommendations': {
                    'data_quality': ['Error in analysis - please check data'],
                    'modeling': ['Analysis failed - manual review needed'],
                    'next_steps': ['Investigate and fix analysis errors']
                }
            }
    
    def _collect_findings(self, context, baseline_results: Dict, explain_results: Dict, 
                         relations_results: Dict) -> Dict[str, Any]:
        """Collect key findings from all analysis results"""
        findings = {
            'dataset': {
                'shape': context.df.shape,
                'target': context.target_column,
                'task_type': context.task_type,
                'missing_data_pct': (context.df.isnull().sum().sum() / (context.df.shape[0] * context.df.shape[1])) * 100
            },
            'data_quality': {
                'warnings': context.warnings,
                'high_missing_columns': [],
                'constant_columns': [],
                'high_cardinality_columns': []
            },
            'relationships': {},
            'modeling': {},
            'feature_importance': {}
        }
        
        # Data quality analysis
        for col in context.df.columns:
            missing_pct = (context.df[col].isnull().sum() / len(context.df)) * 100
            if missing_pct > 30:
                findings['data_quality']['high_missing_columns'].append({
                    'column': col, 'missing_pct': missing_pct
                })
            
            if context.df[col].nunique() <= 1:
                findings['data_quality']['constant_columns'].append(col)
            
            elif context.df[col].dtype == 'object' and context.df[col].nunique() > 50:
                findings['data_quality']['high_cardinality_columns'].append({
                    'column': col, 'unique_count': context.df[col].nunique()
                })
        
        # Relationships analysis
        if relations_results.get('status') == 'success':
            rel_summary = relations_results.get('relationship_summary', {})
            findings['relationships'] = {
                'high_correlations': len(rel_summary.get('key_findings', [])),
                'multicollinearity_issues': len(rel_summary.get('recommendations', [])),
                'key_findings': rel_summary.get('key_findings', [])[:3]
            }
        
        # Modeling results
        if baseline_results.get('status') == 'success':
            best_model = baseline_results.get('best_model')
            models = baseline_results.get('models', {})
            
            if best_model and best_model in models:
                best_score = models[best_model].get('primary_score', 0)
                findings['modeling'] = {
                    'best_model': best_model,
                    'best_score': best_score,
                    'task_type': baseline_results.get('task_type'),
                    'model_count': len([m for m in models.values() if 'error' not in m]),
                    'failed_models': len([m for m in models.values() if 'error' in m])
                }
                
                # Performance assessment
                if context.task_type == 'classification':
                    if best_score > 0.9:
                        findings['modeling']['performance_level'] = 'excellent'
                    elif best_score > 0.8:
                        findings['modeling']['performance_level'] = 'good'
                    elif best_score > 0.6:
                        findings['modeling']['performance_level'] = 'moderate'
                    else:
                        findings['modeling']['performance_level'] = 'poor'
                else:  # regression
                    # For regression, primary_score is negative RMSE
                    if best_score > -0.1:  # Very low error
                        findings['modeling']['performance_level'] = 'excellent'
                    elif best_score > -0.5:
                        findings['modeling']['performance_level'] = 'good'
                    elif best_score > -1.0:
                        findings['modeling']['performance_level'] = 'moderate'
                    else:
                        findings['modeling']['performance_level'] = 'poor'
        
        # Feature importance
        if explain_results.get('status') == 'success':
            perm_importance = explain_results.get('feature_importance', {}).get('permutation', {})
            if perm_importance and 'importances' in perm_importance:
                top_features = perm_importance['importances'][:5]
                findings['feature_importance'] = {
                    'top_features': [f['feature'] for f in top_features],
                    'importance_scores': [f['importance_mean'] for f in top_features],
                    'methods_used': explain_results.get('methods_used', [])
                }
        
        return findings
    
    def _generate_rule_based_advice(self, findings: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate advice using rule-based heuristics"""
        advice = {
            'data_quality': [],
            'feature_engineering': [],
            'modeling': [],
            'evaluation': [],
            'next_steps': []
        }
        
        # Data Quality Advice
        if findings['dataset']['missing_data_pct'] > 15:
            advice['data_quality'].append(
                f"⚠️ High overall missing data ({findings['dataset']['missing_data_pct']:.1f}%). Consider imputation strategies or data collection improvements."
            )
        
        high_missing = findings['data_quality']['high_missing_columns']
        if high_missing:
            advice['data_quality'].append(
                f"🔍 {len(high_missing)} columns have >30% missing data. Consider dropping: {', '.join([col['column'] for col in high_missing[:3]])}"
            )
        
        constant_cols = findings['data_quality']['constant_columns']
        if constant_cols:
            advice['data_quality'].append(
                f"🧹 Remove {len(constant_cols)} constant columns: {', '.join(constant_cols[:3])}"
            )
        
        high_card_cols = findings['data_quality']['high_cardinality_columns']
        if high_card_cols:
            advice['data_quality'].append(
                f"📊 {len(high_card_cols)} high-cardinality categorical columns may need special encoding (target encoding, hashing, or grouping)"
            )
        
        # Feature Engineering Advice
        if findings.get('relationships', {}).get('high_correlations', 0) > 0:
            advice['feature_engineering'].append(
                "🔗 Strong correlations detected. Consider feature selection or dimensionality reduction (PCA, feature clustering)"
            )
        
        if len(findings['feature_importance'].get('top_features', [])) > 0:
            top_features = findings['feature_importance']['top_features'][:3]
            advice['feature_engineering'].append(
                f"⭐ Focus on top features: {', '.join(top_features)}. Consider creating interaction terms or polynomial features."
            )
        
        # Modeling Advice
        task_type = findings['dataset']['task_type']
        best_model = findings.get('modeling', {}).get('best_model')
        performance = findings.get('modeling', {}).get('performance_level', 'unknown')
        
        if task_type == 'classification':
            advice['modeling'].append(
                "🎯 Classification task detected. Consider stratified cross-validation and class-balanced metrics (F1, balanced accuracy)"
            )
            
            if performance == 'poor':
                advice['modeling'].append(
                    "📈 Low performance detected. Try: 1) More data, 2) Feature engineering, 3) Ensemble methods, 4) Hyperparameter tuning"
                )
            elif performance == 'excellent':
                advice['modeling'].append(
                    "🎉 Excellent performance! Check for data leakage and validate on fresh data."
                )
        
        else:  # regression
            advice['modeling'].append(
                "📊 Regression task detected. Consider cross-validation with appropriate metrics (MAE, RMSE, R²)"
            )
            
            if performance == 'poor':
                advice['modeling'].append(
                    "📉 High error detected. Try: 1) Feature scaling, 2) Polynomial features, 3) Regularization (Ridge/Lasso), 4) Tree-based models"
                )
        
        if best_model:
            if 'Tree' in best_model or 'Forest' in best_model:
                advice['modeling'].append(
                    f"🌳 Best model is tree-based ({best_model}). Trees handle missing data and non-linear relationships well. Consider boosting methods (XGBoost, LightGBM)."
                )
            elif 'Linear' in best_model or 'Logistic' in best_model:
                advice['modeling'].append(
                    f"📏 Best model is linear ({best_model}). Ensure features are scaled and consider regularization. Check for non-linear relationships."
                )
        
        # Evaluation Advice
        if task_type == 'classification':
            advice['evaluation'].append(
                "✅ Use confusion matrix, precision/recall curves, and ROC curves for comprehensive evaluation"
            )
            
            # Check for class imbalance
            target_col = findings['dataset']['target']
            if target_col:
                value_counts = findings.get('class_distribution', {})
                if value_counts:
                    max_ratio = max(value_counts.values()) / sum(value_counts.values())
                    if max_ratio > 0.8:
                        advice['evaluation'].append(
                            "⚖️ Class imbalance detected. Use precision-recall AUC and consider SMOTE/undersampling"
                        )
        else:
            advice['evaluation'].append(
                "📈 For regression, examine residual plots and use cross-validation with MAE/RMSE metrics"
            )
        
        # Next Steps
        advice['next_steps'].append(
            "1️⃣ Address data quality issues identified above"
        )
        
        if findings.get('modeling', {}).get('model_count', 0) > 0:
            advice['next_steps'].append(
                "2️⃣ Implement hyperparameter tuning for the best performing model"
            )
            advice['next_steps'].append(
                "3️⃣ Try ensemble methods combining multiple models"
            )
        
        advice['next_steps'].append(
            "4️⃣ Collect more data if possible to improve model performance"
        )
        advice['next_steps'].append(
            "5️⃣ Deploy the model with proper monitoring and validation pipelines"
        )
        
        return advice
    
    def _generate_llm_advice(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM-enhanced advice"""
        if not self.llm:
            return {}
        
        try:
            # Create a concise summary for the LLM
            summary = self._create_findings_summary(findings)
            
            prompt = f"""
You are an expert ML consultant. Based on the following dataset analysis, provide specific, actionable advice.

DATASET ANALYSIS:
{summary}

Please provide:
1. Key insights about this dataset and ML problem
2. Specific recommendations for improving model performance  
3. Potential pitfalls to watch out for
4. Next steps for a data scientist working on this problem

Be concise, practical, and focus on the most important 3-5 points for each section.
"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return {
                'llm_insights': response.content,
                'model_used': self.llm_model,
                'prompt_length': len(prompt)
            }
            
        except Exception as e:
            logger.error(f"LLM advice generation failed: {str(e)}")
            return {}
    
    def _create_findings_summary(self, findings: Dict[str, Any]) -> str:
        """Create a concise summary for LLM input"""
        summary_parts = []
        
        # Dataset basics
        dataset = findings['dataset']
        summary_parts.append(
            f"Dataset: {dataset['shape'][0]} rows, {dataset['shape'][1]} columns"
        )
        summary_parts.append(
            f"Task: {dataset['task_type']} with target '{dataset['target']}'"
        )
        summary_parts.append(
            f"Missing data: {dataset['missing_data_pct']:.1f}%"
        )
        
        # Data quality issues
        dq = findings['data_quality']
        if dq['high_missing_columns']:
            summary_parts.append(
                f"High missing data in {len(dq['high_missing_columns'])} columns"
            )
        if dq['constant_columns']:
            summary_parts.append(
                f"Found {len(dq['constant_columns'])} constant columns"
            )
        
        # Model performance
        modeling = findings.get('modeling', {})
        if 'best_model' in modeling:
            summary_parts.append(
                f"Best model: {modeling['best_model']} with {modeling['performance_level']} performance"
            )
        
        # Top features
        features = findings.get('feature_importance', {})
        if features.get('top_features'):
            top_3 = ', '.join(features['top_features'][:3])
            summary_parts.append(f"Top features: {top_3}")
        
        return '\n'.join(summary_parts)
    
    def _combine_advice(self, rule_based: Dict[str, List[str]], 
                       llm_advice: Dict[str, Any]) -> Dict[str, List[str]]:
        """Combine rule-based and LLM advice into final recommendations"""
        final_advice = rule_based.copy()
        
        # Add LLM insights as a separate section if available
        if llm_advice and 'llm_insights' in llm_advice:
            final_advice['ai_insights'] = [llm_advice['llm_insights']]
        
        # Ensure we have content in each section
        for section in ['data_quality', 'modeling', 'next_steps']:
            if not final_advice.get(section):
                final_advice[section] = ['No specific recommendations for this area']
        
        return final_advice