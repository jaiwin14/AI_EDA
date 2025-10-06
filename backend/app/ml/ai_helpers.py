import json
import logging
from typing import Any, Dict

from app.ml.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class AISuggester:
    """Wrapper around Gemini to provide structured JSON suggestions."""

    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini
        self.gemini = GeminiClient() if use_gemini else None

    def get_ai_suggestion(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_gemini or not self.gemini:
            return {}
        try:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
            system_prompt = (
                "You are an expert data scientist specializing in data preprocessing.\n"
                "Always respond with a valid JSON object containing your recommendations."
            )
            example_json = '{"recommendation":"","action":"","method":"","parameters":{}}'
            user_prompt = f"""
            Given the following data characteristics:
            {context_str}

            {prompt}

            Please provide your recommendation as a JSON object:
            {example_json}
            """
            response = self.gemini.generate_text(
                prompt=user_prompt,
                system_instruction=system_prompt,
                response_format="json"
            )
            if not response:
                return {}
            try:
                result = json.loads(response)
                if not isinstance(result, dict):
                    return {}
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse Gemini response as JSON")
                return {}
        except Exception as e:
            logger.error(f"AI suggestion error: {e}")
            return {}


