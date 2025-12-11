"""LLM generation helpers."""
from __future__ import annotations

from typing import Optional
import importlib.util

from hayagriva.exceptions import GenerationError, MissingDependencyError
from hayagriva.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIGenerator:
    """Minimal OpenAI chat completion wrapper."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        if importlib.util.find_spec("openai") is None:
            raise MissingDependencyError(
                "openai>=1.0.0 is required for generation. Install with `pip install openai`."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise GenerationError(f"Failed to generate response: {exc}") from exc
        content = response.choices[0].message.content if response.choices else ""
        logger.info("Received completion with %d characters", len(content))
        return content
