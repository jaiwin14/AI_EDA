"""AI utilities for generating insights and explanations."""
import os
import json
from enum import Enum
from typing import List, Dict, Any, Optional
import google.generativeai as genai

class AIProvider(Enum):
    """Enum for supported AI providers"""
    GEMINI = "gemini"

def generate_insight(prompt: str, context: Dict[str, Any] = None, provider: AIProvider = AIProvider.GEMINI) -> str:
    """
    Generate insights or explanations using AI
    
    Parameters:
    -----------
    prompt : str
        The prompt to send to the AI model
    context : dict, optional
        Additional context for the prompt
    
    Returns:
    --------
    str
        Generated response
    """
    system_prompt = """You are an expert data analyst specializing in exploratory data analysis. 
    Provide clear, concise, and technically accurate insights based on the data and context provided.
    Focus on actionable insights and explain your reasoning."""
    
    # Prepare complete prompt
    complete_prompt = system_prompt + "\n\n" + prompt
    if context:
        complete_prompt += "\n\nContext:\n" + json.dumps(context, indent=2)
    
    try:
        return _generate_gemini_insight(complete_prompt)
    except Exception as e:
        print(f"Error generating insight with Gemini: {str(e)}")
        return "Unable to generate insight at this time. Please check your Gemini API key configuration."

def _generate_gemini_insight(prompt: str) -> str:
    """Generate insight using Google's Gemini API."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Gemini API key not found in environment variables")
    
    try:
        # Configure the model
        model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-pro"),
            generation_config={
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
                "max_output_tokens": int(os.getenv("MAX_TOKENS", "1000")),
            }
        )
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")


