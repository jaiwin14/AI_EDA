"""Analytics endpoints for Exploratory Data Analysis."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from database import Database
from utils.ai_utils import generate_insight
from utils.serialization import to_json_serializable

router = APIRouter()
db = Database()

def detect_outliers(data: pd.Series, method: str = "both") -> dict:
    """Detect outliers using IQR and/or Z-score methods."""
    results = {"outliers": [], "method": method}
    
    if method in ["iqr", "both"]:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        results["iqr"] = {
            "outliers": iqr_outliers.index.tolist(),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    if method in ["zscore", "both"]:
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        results["zscore"] = {
            "outliers": z_outliers.index.tolist(),
            "z_scores": to_json_serializable(z_scores)
        }
    
    return results

@router.get("/{file_id}/univariate")
async def univariate_analysis(file_id: str):
    """Perform univariate analysis on each column."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        results = {}
        for column in df.columns:
            column_type = str(df[column].dtype)
            non_null_count = df[column].count()
            missing_count = df[column].isnull().sum()
            
            analysis = {
                "type": column_type,
                "missing_count": int(missing_count),
                "non_null_count": int(non_null_count)
            }
            
            if pd.api.types.is_numeric_dtype(df[column]):
                analysis.update({
                    "mean": float(df[column].mean()),
                    "median": float(df[column].median()),
                    "std": float(df[column].std()),
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "histogram": px.histogram(df, x=column).to_json()
                })
            else:
                value_counts = df[column].value_counts()
                analysis.update({
                    "unique_count": len(value_counts),
                    "top_values": value_counts.head().to_dict(),
                    "pie_chart": px.pie(values=value_counts.values, 
                                     names=value_counts.index).to_json()
                })
            
            results[column] = analysis
        
        return JSONResponse(content=to_json_serializable(results))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{file_id}/bivariate/{column1}/{column2}")
async def bivariate_analysis(file_id: str, column1: str, column2: str):
    """Perform bivariate analysis between two columns."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        if column1 not in df.columns or column2 not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        is_numeric1 = pd.api.types.is_numeric_dtype(df[column1])
        is_numeric2 = pd.api.types.is_numeric_dtype(df[column2])
        
        results = {
            "type": f"{'numeric' if is_numeric1 else 'categorical'} vs {'numeric' if is_numeric2 else 'categorical'}"
        }
        
        if is_numeric1 and is_numeric2:
            # Numeric vs Numeric
            correlation = df[column1].corr(df[column2])
            results.update({
                "correlation": float(correlation),
                "scatter_plot": px.scatter(df, x=column1, y=column2).to_json()
            })
        elif not is_numeric1 and not is_numeric2:
            # Categorical vs Categorical
            contingency = pd.crosstab(df[column1], df[column2])
            chi2, pvalue = stats.chi2_contingency(contingency)[:2]
            results.update({
                "chi2": float(chi2),
                "pvalue": float(pvalue),
                "heatmap": px.imshow(contingency).to_json()
            })
        else:
            # Mixed: Categorical vs Numeric
            cat_col = column1 if not is_numeric1 else column2
            num_col = column2 if not is_numeric1 else column1
            
            box_plot = px.box(df, x=cat_col, y=num_col)
            violin_plot = px.violin(df, x=cat_col, y=num_col)
            
            results.update({
                "box_plot": box_plot.to_json(),
                "violin_plot": violin_plot.to_json(),
                "group_stats": df.groupby(cat_col)[num_col].describe().to_dict()
            })
        
        return JSONResponse(content=to_json_serializable(results))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{file_id}/correlation")
async def correlation_analysis(file_id: str):
    """Perform correlation analysis on numeric columns."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Generate heatmap
        heatmap = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu_r"
        )
        
        # Find top correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                correlations.append({"column1": col1, "column2": col2, "correlation": corr})
        
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        results = {
            "correlation_matrix": to_json_serializable(correlation_matrix),
            "heatmap": heatmap.to_json(),
            "top_correlations": correlations[:5]
        }
        
        # Generate AI insights
        insight_prompt = f"""
        Analyze the correlation patterns in this dataset:
        Top 5 correlations:
        {results['top_correlations']}
        
        Please provide:
        1. Key relationships identified
        2. Potential implications for analysis
        3. Recommendations for feature selection
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results)
        except:
            results["ai_insights"] = "Unable to generate AI insights at this time."
        
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{file_id}/outliers")
async def outlier_analysis(file_id: str, method: str = "both", columns: List[str] = None):
    """Detect outliers in numeric columns."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}
        
        for column in numeric_cols:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            results[column] = detect_outliers(df[column], method)
            
            # Add visualization
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[column], name=column, boxpoints="outliers"))
            results[column]["box_plot"] = fig.to_json()
        
        # Generate AI insights
        outlier_counts = {
            col: len(info["iqr"]["outliers"]) if method in ["iqr", "both"] else len(info["zscore"]["outliers"])
            for col, info in results.items()
        }
        
        insight_prompt = f"""
        Analyze the outlier patterns in this dataset:
        Outlier counts by column: {outlier_counts}
        
        Please provide:
        1. Assessment of outlier severity
        2. Recommendations for handling outliers
        3. Potential impact on analysis
        """
        
        try:
            results["ai_insights"] = generate_insight(insight_prompt, context=results)
        except:
            results["ai_insights"] = "Unable to generate AI insights at this time."
        
        return JSONResponse(content=to_json_serializable(results))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{file_id}/outliers/treat")
async def treat_outliers(
    file_id: str,
    method: str = "iqr",  # iqr or zscore
    treatment: str = "cap",  # cap or remove
    columns: List[str] = None
):
    """Treat outliers in the dataset."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        df_treated = df.copy()
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        treatment_stats = {"treated_columns": {}}
        
        for column in numeric_cols:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            outliers = detect_outliers(df[column], method)
            outlier_indices = outliers[method]["outliers"]
            
            if not outlier_indices:
                continue
            
            if treatment == "cap":
                if method == "iqr":
                    lower_bound = outliers["iqr"]["lower_bound"]
                    upper_bound = outliers["iqr"]["upper_bound"]
                    df_treated[column] = df_treated[column].clip(lower_bound, upper_bound)
                else:  # zscore
                    z_scores = stats.zscore(df[column])
                    df_treated.loc[np.abs(z_scores) > 3, column] = df_treated[column].median()
            else:  # remove
                df_treated = df_treated[~df_treated.index.isin(outlier_indices)]
            
            treatment_stats["treated_columns"][column] = {
                "outlier_count": len(outlier_indices),
                "treatment_method": treatment,
                "detection_method": method
            }
        
        # Save treated dataframe
        new_file_id = db.save_uploaded_file(df_treated, f"treated_{file_id}.csv")
        
        treatment_stats.update({
            "original_file_id": file_id,
            "treated_file_id": new_file_id,
            "original_rows": len(df),
            "treated_rows": len(df_treated),
            "treatment_date": datetime.now().isoformat()
        })
        
        db.add_cleaning_step(file_id, "outliers", treatment_stats)
        
        return JSONResponse(content=treatment_stats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{file_id}/encode_categorical")
async def analyze_categorical_encoding(file_id: str):
    """Analyze categorical columns and recommend encoding methods."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        analysis = {}
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            value_counts = df[col].value_counts()
            
            analysis[col] = {
                "unique_count": unique_values,
                "sample_values": df[col].unique()[:5].tolist(),
                "value_distribution": value_counts.head().to_dict(),
                "missing_count": df[col].isnull().sum()
            }
        
        # Generate AI insights for encoding recommendations
        if categorical_cols.empty:
            recommendations = "No categorical columns found in the dataset."
        else:
            insight_prompt = f"""
            Analyze the categorical columns and recommend encoding methods:
            Categorical columns: {list(categorical_cols)}
            Analysis: {analysis}
            
            Please provide:
            1. Recommended encoding method for each column (one-hot vs label)
            2. Reasoning for each recommendation
            3. Potential impact on model performance
            """
            
            try:
                recommendations = generate_insight(insight_prompt, context=analysis)
            except:
                recommendations = "Unable to generate encoding recommendations at this time."
        
        analysis["recommendations"] = recommendations
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{file_id}/encode_categorical")
async def encode_categorical(
    file_id: str,
    columns: Dict[str, str]  # column_name: encoding_method (onehot or label)
):
    """Encode categorical columns using specified methods."""
    try:
        df = db.get_dataframe(file_id)
        if df is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        df_encoded = df.copy()
        encoding_stats = {"encoded_columns": {}}
        
        for column, method in columns.items():
            if column not in df.columns:
                continue
            
            if method == "onehot":
                # Create dummy variables
                dummies = pd.get_dummies(df[column], prefix=column)
                
                # Drop original column and add dummy columns
                df_encoded = df_encoded.drop(column, axis=1)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                encoding_stats["encoded_columns"][column] = {
                    "method": "onehot",
                    "created_columns": dummies.columns.tolist()
                }
            else:  # label
                # Create label encoder
                unique_values = df[column].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                
                # Apply encoding
                df_encoded[column] = df[column].map(encoding_map)
                
                encoding_stats["encoded_columns"][column] = {
                    "method": "label",
                    "encoding_map": encoding_map
                }
        
        # Save encoded dataframe
        new_file_id = db.save_uploaded_file(df_encoded, f"encoded_{file_id}.csv")
        
        encoding_stats.update({
            "original_file_id": file_id,
            "encoded_file_id": new_file_id,
            "original_columns": list(df.columns),
            "encoded_columns": list(df_encoded.columns)
        })
        
        db.add_cleaning_step(file_id, "categorical_encoding", encoding_stats)
        
        return JSONResponse(content=encoding_stats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
