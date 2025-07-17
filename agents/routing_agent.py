"""Intelligent query routing system for determining how to process user queries.

This module provides a RoutingAgent class that analyzes user queries to determine
the appropriate processing strategy based on complexity, scope, and required resources.
"""

from enum import Enum
from typing import List, Dict, Any


class QueryType(Enum):
    """Enumeration of query types based on complexity and scope."""
    SIMPLE = "simple"           # Single fact lookup
    SINGLE_MODULE = "single"    # Within one module
    CROSS_MODULE = "cross"      # Across modules
    SYNTHESIS = "synthesis"     # Complex reasoning

    @classmethod
    def get_all_types(cls) -> List['QueryType']:
        """Return all available query types."""
        return list(cls)

    @classmethod
    def get_by_complexity_level(cls, level: str) -> 'QueryType':
        """Get query type by complexity level (simple, moderate, complex, advanced)."""
        complexity_mapping = {
            'simple': cls.SIMPLE,
            'moderate': cls.SINGLE_MODULE,
            'complex': cls.CROSS_MODULE,
            'advanced': cls.SYNTHESIS
        }
        return complexity_mapping.get(level, cls.SINGLE_MODULE)

class RoutingAgent:
    """Intelligent query routing based on complexity and scope."""

    def __init__(self):
        self.complexity_keywords = {
            QueryType.SIMPLE: ["what is", "define", "when was", "who is"],
            QueryType.CROSS_MODULE: ["compare", "contrast", "across", "between"],
            QueryType.SYNTHESIS: ["analyze", "evaluate", "design", "create", "how does"]
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine routing strategy."""
        query_lower = query.lower()

        # Determine query type
        query_type = self._classify_query_type(query_lower)

        # Estimate complexity (1-10 scale)
        complexity = self._estimate_complexity(query_lower, query_type)

        # Determine required agents
        required_agents = self._determine_required_agents(query_lower, query_type)

        # Select appropriate model
        recommended_model = self._select_model(query_type, complexity)

        return {
            "query_type": query_type,
            "complexity": complexity,
            "required_agents": required_agents,
            "recommended_model": recommended_model,
            "estimated_cost": self._estimate_cost(query_type, len(required_agents))
        }

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the query type based on keywords and structure."""
        # Check for synthesis indicators
        if any(keyword in query for keyword in self.complexity_keywords[QueryType.SYNTHESIS]):
            return QueryType.SYNTHESIS

        # Check for cross-module indicators
        if any(keyword in query for keyword in self.complexity_keywords[QueryType.CROSS_MODULE]):
            return QueryType.CROSS_MODULE

        # Check for simple query indicators
        if any(keyword in query for keyword in self.complexity_keywords[QueryType.SIMPLE]):
            return QueryType.SIMPLE

        # Default to single module
        return QueryType.SINGLE_MODULE

    def _estimate_complexity(self, query: str, query_type: QueryType) -> int:
        """Estimate query complexity on a 1-10 scale."""
        base_complexity = {
            QueryType.SIMPLE: 2,
            QueryType.SINGLE_MODULE: 4,
            QueryType.CROSS_MODULE: 6,
            QueryType.SYNTHESIS: 8
        }

        complexity = base_complexity[query_type]

        # Adjust based on query length
        if len(query) > 100:
            complexity += 1

        # Adjust based on question count
        question_count = query.count('?')
        if question_count > 1:
            complexity += min(question_count - 1, 2)

        return min(complexity, 10)

    def _determine_required_agents(self, query: str, query_type: QueryType) -> List[str]:
        """Determine which agents are needed for this query."""
        # pylint: disable=unused-argument
        if query_type == QueryType.SIMPLE:
            return ["cache", "single_best_match"]
        if query_type == QueryType.SINGLE_MODULE:
            return ["module_search", "module_agents"]
        if query_type == QueryType.CROSS_MODULE:
            return ["semantic_search", "relevant_modules", "synthesis"]
        # SYNTHESIS
        return ["comprehensive_search", "all_relevant_agents", "advanced_synthesis"]

    def _select_model(self, query_type: QueryType, complexity: int) -> str:
        """Select the appropriate model based on query type and complexity."""
        if query_type == QueryType.SIMPLE or complexity <= 3:
            return "claude-3-haiku-20240307"  # Cheapest, fastest
        if complexity <= 6:
            return "claude-3-sonnet-20240229"  # Balanced
        return "claude-3-opus-20240229"  # Most capable

    def _estimate_cost(self, query_type: QueryType, agent_count: int) -> float:
        """Estimate cost in dollars based on query type."""
        base_costs = {
            QueryType.SIMPLE: 0.01,
            QueryType.SINGLE_MODULE: 0.05,
            QueryType.CROSS_MODULE: 0.20,
            QueryType.SYNTHESIS: 0.50
        }
        return base_costs[query_type] * (1 + (agent_count - 1) * 0.1)
