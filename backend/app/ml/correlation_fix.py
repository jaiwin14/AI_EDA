"""
Fixed correlation analysis method for AI agent
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import json

async def analyze_correlations_fixed(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
    """Generate correlation analysis with proper data structure"""
    try:
        # Get correlation pairs as simple list
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    correlations.append({
                        "var1": correlation_matrix.columns[i],
                        "var2": correlation_matrix.columns[j], 
                        "correlation": round(float(corr_val), 3)
                    })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        # Take top 10
        top_correlations = correlations[:10]
        
        # Create simple context for Gemini
        correlation_text = "\n".join([
            f"{c['var1']} <-> {c['var2']}: {c['correlation']}" 
            for c in top_correlations
        ])
        
        context = {
            "correlation_data": correlation_text
        }
        
        # Use the analyze method from the class
        from app.ml.ai_agent import AnalysisType
        return await self.analyze(AnalysisType.CORRELATION_INSIGHTS.value, context)
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis_type": "correlation_insights"
        }

# Fixed prompt template
CORRELATION_INSIGHTS_PROMPT = """
You are analyzing correlations in a dataset. Provide insights about relationships between variables.

Correlation Data:
{correlation_data}

Generate 5-7 key insights about:
- Strong positive/negative correlations
- Unexpected relationships
- Business implications
- Recommendations for further analysis

Respond in JSON format:
{
    "insights": [
        "Correlation insight 1",
        "Correlation insight 2", 
        "Correlation insight 3",
        "Correlation insight 4",
        "Correlation insight 5"
    ]
}
"""
