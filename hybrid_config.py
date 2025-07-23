"""Configuration for hybrid chat architecture.

This module provides configuration options for the hybrid routing system,
including thresholds, model selections, and routing strategies.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RoutingThresholds:
    """Thresholds for routing decisions."""
    confidence_threshold: float = 0.7
    lightweight_max_cost: float = 0.001
    context_relevance_threshold: float = 0.6
    max_context_tokens: int = 6000
    min_context_messages: int = 1


@dataclass
class ModelConfig:
    """Model configuration for different components."""
    conversation_controller: str = "claude-3-haiku-20240307"
    complexity_analyzer: str = "claude-3-haiku-20240307"
    lightweight_chat: str = "claude-3-haiku-20240307"
    agent_pipeline: str = "claude-3-5-sonnet-20240620"


@dataclass
class CostConfig:
    """Cost estimation configuration."""
    # Lightweight model costs (Claude 3 Haiku)
    lightweight_input_cost_per_1k: float = 0.00025
    lightweight_output_cost_per_1k: float = 0.00125

    # Agent pipeline costs (Claude 3.5 Sonnet)
    agent_input_cost_per_1k: float = 0.003
    agent_output_cost_per_1k: float = 0.015

    # Context overhead estimation
    context_tokens_per_message: int = 10


@dataclass
class PerformanceConfig:
    """Performance and timing configuration."""
    max_lightweight_response_time: float = 0.5  # seconds
    max_routing_decision_time: float = 0.1  # seconds
    max_context_filtering_time: float = 0.2  # seconds
    target_agent_invocation_rate: float = 0.3  # 30%


class HybridConfig:
    """Main configuration class for hybrid chat architecture."""

    def __init__(self):
        self.routing = RoutingThresholds()
        self.models = ModelConfig()
        self.costs = CostConfig()
        self.performance = PerformanceConfig()

        # Category weights for routing decisions
        self.category_weights = {
            # Lightweight categories (positive weights)
            "clarification": 1.0,
            "elaboration": 1.0,
            "confirmation": 1.0,
            "simple_followup": 1.0,
            "conversation_meta": 1.0,

            # Agent-required categories (negative weights)
            "new_complex_topic": -1.0,
            "document_specific": -1.0,
            "synthesis_needed": -1.0,
            "factual_lookup": -1.0,
            "deep_analysis": -1.0,

            # Hybrid categories (neutral weights)
            "partial_context": 0.0,
            "topic_shift": 0.0,
            "complex_followup": 0.0,
        }

        # Context filtering strategies
        self.context_strategies = {
            "relevance": {
                "enabled": True,
                "weight": 0.6,
                "fallback_to_recency": True
            },
            "recency": {
                "enabled": True,
                "weight": 0.3,
                "window_size": 3
            },
            "topic": {
                "enabled": True,
                "weight": 0.1,
                "min_overlap": 1
            }
        }

        # Logging configuration
        self.logging = {
            "level": "INFO",
            "log_routing_decisions": True,
            "log_cost_estimates": True,
            "log_context_filtering": True
        }

    def get_routing_decision_config(self) -> Dict[str, Any]:
        """Get configuration for routing decisions."""
        return {
            "confidence_threshold": self.routing.confidence_threshold,
            "category_weights": self.category_weights,
            "target_agent_invocation_rate": self.performance.target_agent_invocation_rate
        }

    def get_context_filtering_config(self) -> Dict[str, Any]:
        """Get configuration for context filtering."""
        return {
            "max_tokens": self.routing.max_context_tokens,
            "relevance_threshold": self.routing.context_relevance_threshold,
            "strategies": self.context_strategies,
            "min_messages": self.routing.min_context_messages
        }

    def get_cost_estimation_config(self) -> Dict[str, Any]:
        """Get configuration for cost estimation."""
        return {
            "lightweight_input_cost": self.costs.lightweight_input_cost_per_1k,
            "lightweight_output_cost": self.costs.lightweight_output_cost_per_1k,
            "agent_input_cost": self.costs.agent_input_cost_per_1k,
            "agent_output_cost": self.costs.agent_output_cost_per_1k,
            "context_overhead": self.costs.context_tokens_per_message
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "conversation_controller": self.models.conversation_controller,
            "complexity_analyzer": self.models.complexity_analyzer,
            "lightweight_chat": self.models.lightweight_chat,
            "agent_pipeline": self.models.agent_pipeline
        }

    def update_routing_thresholds(self, **kwargs) -> None:
        """Update routing thresholds."""
        for key, value in kwargs.items():
            if hasattr(self.routing, key):
                setattr(self.routing, key, value)

    def update_category_weights(self, category: str, weight: float) -> None:
        """Update weight for a specific category."""
        if category in self.category_weights:
            self.category_weights[category] = weight

    def get_optimized_config(self, target_cost_savings: float = 0.7) -> 'HybridConfig':
        """Get an optimized configuration for target cost savings."""
        config = HybridConfig()

        # Adjust thresholds based on target savings
        if target_cost_savings > 0.8:
            # Aggressive cost savings
            config.routing.confidence_threshold = 0.6
            config.performance.target_agent_invocation_rate = 0.2
        elif target_cost_savings > 0.6:
            # Moderate cost savings
            config.routing.confidence_threshold = 0.7
            config.performance.target_agent_invocation_rate = 0.3
        else:
            # Conservative cost savings
            config.routing.confidence_threshold = 0.8
            config.performance.target_agent_invocation_rate = 0.4

        return config


# Default configuration instance
default_config = HybridConfig()


def get_config() -> HybridConfig:
    """Get the default configuration."""
    return default_config


def create_config_for_use_case(use_case: str) -> HybridConfig:
    """Create a configuration optimized for a specific use case."""
    config = HybridConfig()

    if use_case == "cost_optimized":
        # Maximum cost savings
        config.routing.confidence_threshold = 0.6
        config.performance.target_agent_invocation_rate = 0.2
        config.routing.context_relevance_threshold = 0.5

    elif use_case == "quality_optimized":
        # Maximum quality, higher costs
        config.routing.confidence_threshold = 0.8
        config.performance.target_agent_invocation_rate = 0.5
        config.routing.context_relevance_threshold = 0.7

    elif use_case == "balanced":
        # Balanced approach
        config.routing.confidence_threshold = 0.7
        config.performance.target_agent_invocation_rate = 0.3
        config.routing.context_relevance_threshold = 0.6

    elif use_case == "research":
        # Research mode - more detailed analysis
        config.routing.confidence_threshold = 0.75
        config.performance.target_agent_invocation_rate = 0.4
        config.routing.max_context_tokens = 8000

    elif use_case == "casual":
        # Casual conversation - lightweight preferred
        config.routing.confidence_threshold = 0.65
        config.performance.target_agent_invocation_rate = 0.25
        config.routing.max_context_tokens = 4000

    return config


# Configuration presets
PRESETS = {
    "cost_optimized": "Maximum cost savings, may sacrifice some quality",
    "quality_optimized": "Maximum quality, higher costs",
    "balanced": "Balanced approach between cost and quality",
    "research": "Optimized for research and detailed analysis",
    "casual": "Optimized for casual conversation"
}


def list_presets() -> Dict[str, str]:
    """List available configuration presets."""
    return PRESETS.copy()


def apply_preset(preset_name: str) -> HybridConfig:
    """Apply a configuration preset."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    return create_config_for_use_case(preset_name)
