from openai import OpenAI
import os
import json
from typing import Any, Dict
from src.models.base import ModelProvider

class OpenAIModel(ModelProvider):
    """OpenAI model provider that uses GPT-4 for evaluation."""

    def __init__(self, model: str, temp: float, response_format: Any = None):
        """Initialize OpenAI API with unified environment variables."""
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")

        if not api_key:
            raise ValueError("API_KEY is not set in the .env file.")
        if not base_url:
            raise ValueError("BASE_URL is not set in the .env file.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temp = float(temp)
        self.response_format = response_format or False

    def generate(self, prompt: Any):
        """Generate a response using the OpenAI-compatible API."""
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) and 'role' in item and item['role'] in ['user', 'assistant'] for item in prompt):
            pass
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'.")

        if self.response_format:
            # Try structured output first, fall back to manual JSON parsing
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=prompt,
                    temperature=self.temp,
                    response_format=self.response_format
                )
                return response.choices[0].message.parsed
            except Exception:
                # Fallback: request JSON via prompt and parse manually
                prompt[-1]["content"] += "\n\nRespond in JSON format with keys: " + ", ".join(
                    f'"{k}"' for k in self.response_format.model_fields.keys()
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    temperature=self.temp,
                    max_tokens=4096,
                )
                text = response.choices[0].message.content.strip()
                # Extract JSON from possible markdown code blocks
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                return self.response_format.model_validate_json(text)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temp,
                max_tokens=4096,
            )
            return response.choices[0].message.content