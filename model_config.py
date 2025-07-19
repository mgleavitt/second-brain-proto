"""Model configuration and cost management module.

This module provides ModelConfig and ModelInfo classes for managing LLM model
selection, cost tracking, and configuration across different agent types.
"""

from typing import Dict
from dataclasses import dataclass
import logging


@dataclass
class ModelInfo:
    """Information about a model including cost per million tokens."""
    name: str
    display_name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window: int
    provider: str  # 'anthropic' or 'google'


class ModelConfig:
    """Manages model selection and cost tracking for different agent types."""

    # Available models with costs (as of July 2025)
    AVAILABLE_MODELS = {
        # Anthropic models
        "claude-3-opus": ModelInfo(
            name="claude-3-opus-20240229",
            display_name="Claude 3 Opus",
            input_cost_per_1m=15.0,
            output_cost_per_1m=75.0,
            context_window=200000,
            provider="anthropic"
        ),
        "claude-3-sonnet": ModelInfo(
            name="claude-3-5-sonnet-20240620",
            display_name="Claude 3.5 Sonnet",
            input_cost_per_1m=3.0,
            output_cost_per_1m=15.0,
            context_window=200000,
            provider="anthropic"
        ),
        "claude-3-haiku": ModelInfo(
            name="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            input_cost_per_1m=0.25,
            output_cost_per_1m=1.25,
            context_window=200000,
            provider="anthropic"
        ),
        # Google models
        "gemini-flash": ModelInfo(
            name="gemini-1.5-flash-latest",
            display_name="Gemini 1.5 Flash",
            input_cost_per_1m=0.075,
            output_cost_per_1m=0.30,
            context_window=1000000,
            provider="google"
        ),
        "gemini-pro": ModelInfo(
            name="gemini-1.5-pro-latest",
            display_name="Gemini 1.5 Pro",
            input_cost_per_1m=3.5,
            output_cost_per_1m=10.5,
            context_window=2000000,
            provider="google"
        )
    }

    # Default models for each agent type
    DEFAULT_MODELS = {
        "document": "gemini-flash",
        "module": "gemini-flash",
        "synthesis": "claude-3-sonnet",
        "routing": "claude-3-haiku",
        "summarizer": "gemini-flash",
        "evaluator": "gemini-flash"
    }

    def __init__(self):
        self.agent_models: Dict[str, str] = self.DEFAULT_MODELS.copy()
        self.logger = logging.getLogger(__name__)

    def set_model(self, agent_type: str, model_key: str) -> None:
        """Set the model for a specific agent type."""
        if agent_type not in self.DEFAULT_MODELS:
            valid_types = list(self.DEFAULT_MODELS.keys())
            raise ValueError(f"Invalid agent type: {agent_type}. Valid types: {valid_types}")

        if model_key not in self.AVAILABLE_MODELS:
            valid_models = list(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Invalid model: {model_key}. Valid models: {valid_models}")

        self.agent_models[agent_type] = model_key
        self.logger.info("Set %s agent to use %s", agent_type, model_key)

    def get_model_name(self, agent_type: str) -> str:
        """Get the actual model name for an agent type."""
        model_key = self.agent_models.get(agent_type, self.DEFAULT_MODELS.get(agent_type))
        return self.AVAILABLE_MODELS[model_key].name

    def get_model_info(self, agent_type: str) -> ModelInfo:
        """Get complete model information for an agent type."""
        model_key = self.agent_models.get(agent_type, self.DEFAULT_MODELS.get(agent_type))
        return self.AVAILABLE_MODELS[model_key]

    def estimate_cost(self, agent_type: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a specific agent's token usage."""
        model_info = self.get_model_info(agent_type)
        input_cost = (input_tokens / 1_000_000) * model_info.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * model_info.output_cost_per_1m
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Dict[str, any]]:
        """Get summary of current model configuration."""
        summary = {}
        for agent_type, model_key in self.agent_models.items():
            model_info = self.AVAILABLE_MODELS[model_key]
            summary[agent_type] = {
                "model": model_key,
                "display_name": model_info.display_name,
                "provider": model_info.provider,
                "input_cost_per_1m": model_info.input_cost_per_1m,
                "output_cost_per_1m": model_info.output_cost_per_1m
            }
        return summary
