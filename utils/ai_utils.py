"""AI utilities for generating insights and explanations."""
import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_gemini() -> Optional[str]:
    """Configure Gemini API and validate the setup."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return "Gemini API key not found in environment variables"
    
    try:
        genai.configure(api_key=api_key)
        # Get available models
        model_list = genai.list_models()
        # Get all available models
        model_names = {model.name for model in model_list}
        
        # List of supported models in order of preference
        supported_models = [
            "models/gemini-1.5-flash",
            "models/gemini-2.5-flash-preview",
        ]
        
        # Find the first available supported model
        selected_model = next((model for model in supported_models if model in model_names), None)
        
        if selected_model:
            print(f"Using Gemini model: {selected_model}")
            os.environ['GEMINI_MODEL'] = selected_model
            return None
        else:
            available_models = ", ".join(model_names)
            return f"No suitable Gemini model available. Available models: {available_models}"
        return None
    except Exception as e:
        return f"Error configuring Gemini API: {str(e)}"

def generate_insight(prompt: str, context: Dict[str, Any] = None) -> str:
    """
    Generate insights or explanations using Gemini AI
    
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
    
    # Verify Gemini configuration
    config_error = configure_gemini()
    if config_error:
        print(f"Gemini configuration error: {config_error}")
        return "Unable to generate insights: Gemini API not properly configured"

    try:
        # Initialize Gemini model with safety settings
        model = genai.GenerativeModel(
            model_name=os.getenv('GEMINI_MODEL', 'models/gemini-1.5-flash'),
            generation_config={
                'temperature': float(os.getenv('TEMPERATURE', '0.7')),
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': int(os.getenv('MAX_TOKENS', '1024')),
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        
        # Generate response
        response = model.generate_content(complete_prompt)
        
        if response and hasattr(response, 'text'):
            return response.text.strip()
        else:
            return "Unable to generate insights: No response from Gemini"
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating insight with Gemini: {error_msg}")
        
        if "API key not found" in error_msg:
            return "Unable to generate insights: Missing Gemini API key"
        elif "model not found" in error_msg:
            return "Unable to generate insights: Gemini Pro model not available"
        else:
            return f"Unable to generate insights: {error_msg}"
        
