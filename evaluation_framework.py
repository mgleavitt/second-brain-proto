"""
Evaluation framework for comparing query strategies and measuring performance.

This module provides tools to evaluate the quality and cost-effectiveness
of different query routing strategies in the second brain system.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

from model_config import ModelConfig

@dataclass
class QualityMetrics:
    """Quality evaluation metrics."""
    relevance_score: Optional[float] = None  # 0-1
    completeness_score: Optional[float] = None  # 0-1
    coherence_score: Optional[float] = None  # 0-1

    def overall_score(self) -> Optional[float]:
        """Calculate overall quality score."""
        scores = [s for s in [self.relevance_score, self.completeness_score,
                             self.coherence_score] if s is not None]
        return np.mean(scores) if scores else None


@dataclass
class CostMetrics:
    """Cost and performance metrics."""
    estimated_cost: float
    alternative_cost: Optional[float] = None  # Cost without routing
    cost_savings: Optional[float] = None
    response_time: float = 0.0


@dataclass
class UserFeedback:
    """User feedback and ratings."""
    user_rating: Optional[int] = None  # 1-5
    user_feedback: Optional[str] = None


@dataclass
class QueryMetadata:
    """Basic query metadata."""
    query_id: str
    question: str
    routing_used: bool
    modules_queried: List[str]
    response_length: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating query results."""
    metadata: QueryMetadata

    # Sub-metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    cost: CostMetrics = field(default_factory=lambda: CostMetrics(estimated_cost=0.0))
    feedback: UserFeedback = field(default_factory=UserFeedback)

    def overall_score(self) -> Optional[float]:
        """Calculate overall quality score."""
        return self.quality.overall_score()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        result = asdict(self)
        # Flatten the nested metrics for backward compatibility
        flattened = {
            'query_id': self.metadata.query_id,
            'question': self.metadata.question,
            'routing_used': self.metadata.routing_used,
            'modules_queried': self.metadata.modules_queried,
            'response_length': self.metadata.response_length,
            'timestamp': self.metadata.timestamp,
            'relevance_score': self.quality.relevance_score,
            'completeness_score': self.quality.completeness_score,
            'coherence_score': self.quality.coherence_score,
            'estimated_cost': self.cost.estimated_cost,
            'alternative_cost': self.cost.alternative_cost,
            'cost_savings': self.cost.cost_savings,
            'response_time': self.cost.response_time,
            'user_rating': self.feedback.user_rating,
            'user_feedback': self.feedback.user_feedback,
        }
        result.update(flattened)
        return result

class QueryEvaluator:
    """Evaluate and compare query results with different strategies."""

    def __init__(self, model_config: Optional[ModelConfig] = None,
                 results_dir: str = ".evaluation"):
        self.model_config = model_config or ModelConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Use a fast model for evaluation
        self.model_config.set_model("evaluator", "gemini-flash")

    def evaluate_response(self, question: str, response: str,
                         metadata: Dict[str, Any]) -> EvaluationMetrics:
        """Evaluate a single response using LLM-based scoring."""
        query_metadata = QueryMetadata(
            query_id=self._generate_query_id(),
            question=question,
            routing_used=metadata.get('routing_used', False),
            modules_queried=metadata.get('modules_queried', []),
            response_length=len(response)
        )

        cost_metrics = CostMetrics(
            estimated_cost=metadata.get('cost', 0),
            response_time=metadata.get('response_time', 0)
        )

        metrics = EvaluationMetrics(
            metadata=query_metadata,
            cost=cost_metrics
        )

        # Get LLM evaluation
        eval_scores = self._llm_evaluate(question, response)
        metrics.quality.relevance_score = eval_scores.get('relevance', 0)
        metrics.quality.completeness_score = eval_scores.get('completeness', 0)
        metrics.quality.coherence_score = eval_scores.get('coherence', 0)

        return metrics

    def _llm_evaluate(self, question: str, response: str) -> Dict[str, float]:
        """Use LLM to evaluate response quality."""
        eval_prompt = f"""Evaluate this response on three dimensions, scoring each from 0 to 1:

1. **Relevance** (0-1): How well does the response address the specific question asked?
2. **Completeness** (0-1): Does the response cover all aspects of the question thoroughly?
3. **Coherence** (0-1): Is the response well-organized, clear, and easy to understand?

Question: {question[:500]}

Response: {response[:1500]}

Provide your evaluation as JSON with this format:
{{
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "reasoning": "Brief explanation of scores"
}}"""

        llm = ChatGoogleGenerativeAI(
            model=self.model_config.get_model_name("evaluator"),
            temperature=0.1,
            convert_system_message_to_human=True
        )

        result = llm.invoke(eval_prompt)

        try:
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return {
                    'relevance': float(scores.get('relevance', 0)),
                    'completeness': float(scores.get('completeness', 0)),
                    'coherence': float(scores.get('coherence', 0))
                }
        except (json.JSONDecodeError, ValueError, AttributeError):
            self.logger.error("Failed to parse evaluation scores")

        return {'relevance': 0.5, 'completeness': 0.5, 'coherence': 0.5}

    def compare_strategies(self, question: str,
                          system: Any,  # The main system instance
                          strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare different query strategies."""
        if strategies is None:
            strategies = ['with_routing', 'without_routing']

        results = {}

        for strategy in strategies:
            self.logger.info("Testing strategy: %s", strategy)
            strategy_result = self._run_strategy(question, system, strategy)
            results[strategy] = strategy_result

        # Calculate cost savings
        if 'with_routing' in results and 'without_routing' in results:
            comparison_data = self._calculate_comparison(results)
            results['comparison'] = comparison_data

        # Save results
        self._save_comparison(question, results)

        return results

    def _run_strategy(self, question: str, system: Any, strategy: str) -> Dict[str, Any]:
        """Run a single strategy and return results."""
        # Configure system for strategy
        original_routing = system.use_routing
        if strategy == 'without_routing':
            system.use_routing = False

        # Run query
        start_time = datetime.now()
        response_data = system.query(question)
        elapsed = (datetime.now() - start_time).total_seconds()

        # Evaluate
        metadata = {
            'routing_used': system.use_routing,
            'modules_queried': response_data.get('modules', []),
            'response_time': elapsed,
            'cost': response_data.get('cost', 0)
        }

        metrics = self.evaluate_response(
            question,
            response_data.get('answer', ''),
            metadata
        )

        # Restore original settings
        system.use_routing = original_routing

        return {
            'response': response_data.get('answer', ''),
            'metrics': metrics,
            'cost': metrics.cost.estimated_cost,
            'time': metrics.cost.response_time
        }

    def _calculate_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparison metrics between strategies."""
        cost_with = results['with_routing']['cost']
        cost_without = results['without_routing']['cost']
        savings = cost_without - cost_with
        savings_pct = (savings / cost_without * 100) if cost_without > 0 else 0

        return {
            'cost_savings': savings,
            'cost_savings_percentage': savings_pct,
            'quality_difference': self._compare_quality(
                results['with_routing']['metrics'],
                results['without_routing']['metrics']
            )
        }

    def _compare_quality(self, metrics1: EvaluationMetrics,
                        metrics2: EvaluationMetrics) -> Dict[str, float]:
        """Compare quality scores between two evaluations."""
        relevance_diff = ((metrics1.quality.relevance_score or 0) -
                         (metrics2.quality.relevance_score or 0))
        completeness_diff = ((metrics1.quality.completeness_score or 0) -
                           (metrics2.quality.completeness_score or 0))
        coherence_diff = ((metrics1.quality.coherence_score or 0) -
                         (metrics2.quality.coherence_score or 0))
        overall_diff = (metrics1.overall_score() or 0) - (metrics2.overall_score() or 0)

        return {
            'relevance_diff': relevance_diff,
            'completeness_diff': completeness_diff,
            'coherence_diff': coherence_diff,
            'overall_diff': overall_diff
        }

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        return str(uuid.uuid4())[:8]

    def _save_comparison(self, question: str, results: Dict[str, Any]):
        """Save comparison results to file."""
        filename = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert results to JSON-serializable format
        serializable_results = {}
        for k, v in results.items():
            if k == 'metrics':
                # Skip metrics for now as they're not needed in the saved file
                continue
            if isinstance(v, dict):
                # Handle nested dictionaries
                serializable_results[k] = {}
                for nested_k, nested_v in v.items():
                    if nested_k == 'metrics':
                        serializable_results[k][nested_k] = nested_v.to_dict()
                    else:
                        serializable_results[k][nested_k] = nested_v
            else:
                serializable_results[k] = v

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'question': question,
                'results': serializable_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    def generate_report(self) -> str:
        """Generate evaluation report from saved results."""
        # Load all comparison files
        comparisons = []
        for file in self.results_dir.glob("comparison_*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                comparisons.append(json.load(f))

        if not comparisons:
            return "No evaluation data available."

        # Aggregate statistics
        total_comparisons = len(comparisons)
        cost_savings_list = [c.get('comparison', {}).get('cost_savings', 0)
                            for c in comparisons]
        quality_diff_list = [c.get('comparison', {}).get('quality_difference', {})
                           .get('overall_diff', 0) for c in comparisons]

        avg_cost_savings = np.mean(cost_savings_list)
        avg_quality_diff = np.mean(quality_diff_list)

        report = f"""# Query Evaluation Report

## Summary
- Total evaluations: {total_comparisons}
- Average cost savings with routing: ${avg_cost_savings:.4f}
- Average quality difference: {avg_quality_diff:+.3f}

## Recommendation
"""
        if avg_quality_diff > -0.05 and avg_cost_savings > 0:
            report += "Routing provides cost savings with minimal quality impact. ✓"
        elif avg_quality_diff < -0.1:
            report += "Routing may be impacting quality. Consider adjusting thresholds."
        else:
            report += "Current routing strategy is performing well."

        return report
