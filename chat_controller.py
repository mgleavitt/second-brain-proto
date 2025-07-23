"""Lightweight chat controller: routes between cheap continuity model
and full agent pipeline."""

from typing import Callable, Tuple, Dict, Any, List
import logging

# Query complexity categories
LIGHTWEIGHT_CATEGORIES = {
    "clarification",      # "What did you mean by X?"
    "elaboration",        # "Tell me more about Y"
    "confirmation",       # "So you're saying that..."
    "simple_followup",    # "And what about Z?"
    "conversation_meta",  # "Can you summarize what we discussed?"
}

AGENT_REQUIRED_CATEGORIES = {
    "new_complex_topic",  # "Explain the relationship between A and B"
    "document_specific",  # "What does document X say about Y?"
    "synthesis_needed",   # "Compare perspectives across modules"
    "factual_lookup",     # "What is the exact formula for..."
    "deep_analysis",      # "Analyze the implications of..."
}

HYBRID_CATEGORIES = {
    "partial_context",    # Some context needed, but not all
    "topic_shift",        # Changing subjects but building on previous
    "complex_followup",   # Followup requiring document access
}

# Routing thresholds
CONFIDENCE_THRESHOLD = 0.7
LIGHTWEIGHT_MAX_COST = 0.001  # Max cost before considering agents
CONTEXT_RELEVANCE_THRESHOLD = 0.6  # Similarity score for context filtering


class ChatController:
    """Intelligent chat controller with hybrid routing architecture."""

    def __init__(
        self,
        conversation_manager,
        prototype,
        light_model_call: Callable[[str], str],
        classify_query: Callable[[str, any], Tuple[str, float]],
        build_context: Callable[[any, str, int], list],
        max_ctx_tokens: int = 6000,
    ):
        self.cm = conversation_manager
        self.prototype = prototype
        self.light_call = light_model_call
        self.classify = classify_query
        self.build_ctx = build_context
        self.max_ctx = max_ctx_tokens

        # Routing state tracking
        self.last_route = "unknown"
        self.last_category = "unknown"
        self.last_confidence = 0.0
        self.routing_history = []

        # Cost tracking
        self.lightweight_costs = 0.0
        self.agent_costs = 0.0
        self.total_queries = 0

        self.logger = logging.getLogger(__name__)

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process user message with intelligent routing."""
        self.cm.add_message("user", user_input)
        self.total_queries += 1

        # Analyze query complexity
        category, confidence = self.classify(user_input, self.cm)
        self.last_category, self.last_confidence = category, confidence

        # Make routing decision
        route_decision = self._should_invoke_agents(user_input, category, confidence)

        # Execute appropriate response path
        if route_decision["use_lightweight"]:
            self.last_route = "lightweight"
            response = self._generate_lightweight_response(user_input)
            cost = self._estimate_lightweight_cost(user_input, response)
            self.lightweight_costs += cost
        else:
            self.last_route = "agent"
            filtered_context = self._prepare_filtered_context(user_input)
            # Use the intelligent routing method instead of querying all agents
            result = self.prototype.query_with_routing(user_input)
            response = result.get("response", "No response generated")
            cost = self._estimate_agent_cost(user_input, response, len(filtered_context))
            self.agent_costs += cost

        # Log routing decision
        self._log_routing_decision(user_input, category, confidence, route_decision, cost)

        return {
            "response": response,
            "route": self.last_route,
            "category": category,
            "confidence": confidence,
            "cost": cost,
            "context_size": len(filtered_context) if self.last_route == "agent" else 0,
            "reasoning": route_decision.get("reasoning", "")
        }

    def _should_invoke_agents(self, query: str, category: str, confidence: float) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """Determine if agents should be invoked based on query analysis."""
        reasoning = []

        # Clear lightweight cases
        if category in LIGHTWEIGHT_CATEGORIES and confidence >= CONFIDENCE_THRESHOLD:
            reasoning.append(f"High confidence ({confidence:.2f}) lightweight category: {category}")
            return {
                "use_lightweight": True,
                "reasoning": "; ".join(reasoning)
            }

        # Clear agent cases
        if category in AGENT_REQUIRED_CATEGORIES:
            reasoning.append(f"Agent-required category: {category}")
            return {
                "use_lightweight": False,
                "reasoning": "; ".join(reasoning)
            }

        # Hybrid cases - need more analysis
        if category in HYBRID_CATEGORIES:
            reasoning.append(f"Hybrid category: {category}")
            # For hybrid cases, err on side of agent invocation when uncertain
            if confidence < CONFIDENCE_THRESHOLD:
                reasoning.append(f"Low confidence ({confidence:.2f}), using agents for safety")
                return {
                    "use_lightweight": False,
                    "reasoning": "; ".join(reasoning)
                }

        # Default decision based on confidence
        if confidence >= CONFIDENCE_THRESHOLD:
            reasoning.append(f"High confidence ({confidence:.2f}) for lightweight processing")
            return {
                "use_lightweight": True,
                "reasoning": "; ".join(reasoning)
            }
        else:
            reasoning.append(f"Low confidence ({confidence:.2f}), using agents for safety")
            return {
                "use_lightweight": False,
                "reasoning": "; ".join(reasoning)
            }

    def _generate_lightweight_response(self, query: str) -> str:
        """Generate response using lightweight model."""
        prompt = self.cm.build_prompt(query)
        return self.light_call(prompt)

    def _prepare_filtered_context(self, query: str) -> List[Any]:
        """Prepare filtered context for agent queries."""
        return self.build_ctx(self.cm, query, self.max_ctx)

    def _estimate_lightweight_cost(self, query: str, response: str) -> float:
        """Estimate cost for lightweight response."""
        # Rough estimation: $0.00025 per 1K input + $0.00125 per 1K output
        input_tokens = len(query.split()) * 1.3  # Rough token estimation
        output_tokens = len(response.split()) * 1.3
        return (input_tokens * 0.00025 / 1000) + (output_tokens * 0.00125 / 1000)

    def _estimate_agent_cost(self, query: str, response: str, context_size: int) -> float:
        """Estimate cost for agent response."""
        # This is a rough estimation - actual costs depend on the specific agent pipeline
        input_tokens = len(query.split()) * 1.3 + context_size * 10  # Context tokens
        output_tokens = len(response.split()) * 1.3
        return (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)  # Sonnet pricing

    def _log_routing_decision(self, query: str, category: str, confidence: float,
                            decision: Dict[str, Any], cost: float) -> None:
        """Log routing decision for analysis."""
        self.routing_history.append({
            "query": query[:100] + "..." if len(query) > 100 else query,
            "category": category,
            "confidence": confidence,
            "route": decision["use_lightweight"],
            "cost": cost,
            "reasoning": decision.get("reasoning", "")
        })

        self.logger.info(
            "Routing: %s (%.2f) -> %s (%.4f) | %s",
            category, confidence,
            "lightweight" if decision["use_lightweight"] else "agent",
            cost, decision.get("reasoning", "")
        )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        if not self.routing_history:
            return {}

        total = len(self.routing_history)
        lightweight_count = sum(1 for r in self.routing_history if r["route"])
        agent_count = total - lightweight_count

        return {
            "total_queries": total,
            "lightweight_queries": lightweight_count,
            "agent_queries": agent_count,
            "lightweight_percentage": (lightweight_count / total) * 100,
            "total_lightweight_cost": self.lightweight_costs,
            "total_agent_cost": self.agent_costs,
            "total_cost": self.lightweight_costs + self.agent_costs,
            "average_cost_per_query": (self.lightweight_costs + self.agent_costs) / total
        }

    def get_last_route_info(self) -> Dict[str, Any]:
        """Get information about the last routing decision."""
        return {
            "route": self.last_route,
            "category": self.last_category,
            "confidence": self.last_confidence
        }

__all__ = ["ChatController"]
