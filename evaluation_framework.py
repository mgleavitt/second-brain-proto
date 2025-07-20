import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path
import logging

from model_config import ModelConfig

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating query results."""
    query_id: str
    question: str
    routing_used: bool
    modules_queried: List[str]
    response_length: int
    response_time: float
    estimated_cost: float

    # Quality metrics (to be filled by evaluation)
    relevance_score: Optional[float] = None  # 0-1
    completeness_score: Optional[float] = None  # 0-1
    coherence_score: Optional[float] = None  # 0-1

    # Comparison metrics
    alternative_cost: Optional[float] = None  # Cost without routing
    cost_savings: Optional[float] = None

    # User feedback
    user_rating: Optional[int] = None  # 1-5
    user_feedback: Optional[str] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def overall_score(self) -> Optional[float]:
        """Calculate overall quality score."""
        scores = [s for s in [self.relevance_score, self.completeness_score,
                             self.coherence_score] if s is not None]
        return np.mean(scores) if scores else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
        metrics = EvaluationMetrics(
            query_id=self._generate_query_id(),
            question=question,
            routing_used=metadata.get('routing_used', False),
            modules_queried=metadata.get('modules_queried', []),
            response_length=len(response),
            response_time=metadata.get('response_time', 0),
            estimated_cost=metadata.get('cost', 0)
        )

        # Get LLM evaluation
        eval_scores = self._llm_evaluate(question, response)
        metrics.relevance_score = eval_scores.get('relevance', 0)
        metrics.completeness_score = eval_scores.get('completeness', 0)
        metrics.coherence_score = eval_scores.get('coherence', 0)

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

        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=self.model_config.get_model_name("evaluator"),
            temperature=0.1,
            convert_system_message_to_human=True
        )

        result = llm.invoke(eval_prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return {
                    'relevance': float(scores.get('relevance', 0)),
                    'completeness': float(scores.get('completeness', 0)),
                    'coherence': float(scores.get('coherence', 0))
                }
        except:
            self.logger.error("Failed to parse evaluation scores")

        return {'relevance': 0.5, 'completeness': 0.5, 'coherence': 0.5}

    def compare_strategies(self, question: str,
                          system: Any,  # The main system instance
                          strategies: List[str] = ['with_routing', 'without_routing']) -> Dict[str, Any]:
        """Compare different query strategies."""
        results = {}

        for strategy in strategies:
            self.logger.info(f"Testing strategy: {strategy}")

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

            results[strategy] = {
                'response': response_data.get('answer', ''),
                'metrics': metrics,
                'cost': response_data.get('cost', 0),
                'time': elapsed
            }

            # Restore original settings
            system.use_routing = original_routing

        # Calculate cost savings
        if 'with_routing' in results and 'without_routing' in results:
            cost_with = results['with_routing']['cost']
            cost_without = results['without_routing']['cost']
            savings = cost_without - cost_with
            savings_pct = (savings / cost_without * 100) if cost_without > 0 else 0

            results['comparison'] = {
                'cost_savings': savings,
                'cost_savings_percentage': savings_pct,
                'quality_difference': self._compare_quality(
                    results['with_routing']['metrics'],
                    results['without_routing']['metrics']
                )
            }

        # Save results
        self._save_comparison(question, results)

        return results

    def _compare_quality(self, metrics1: EvaluationMetrics,
                        metrics2: EvaluationMetrics) -> Dict[str, float]:
        """Compare quality scores between two evaluations."""
        return {
            'relevance_diff': (metrics1.relevance_score or 0) - (metrics2.relevance_score or 0),
            'completeness_diff': (metrics1.completeness_score or 0) - (metrics2.completeness_score or 0),
            'coherence_diff': (metrics1.coherence_score or 0) - (metrics2.coherence_score or 0),
            'overall_diff': (metrics1.overall_score() or 0) - (metrics2.overall_score() or 0)
        }

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        import uuid
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
            elif isinstance(v, dict):
                # Handle nested dictionaries
                serializable_results[k] = {}
                for nested_k, nested_v in v.items():
                    if nested_k == 'metrics':
                        serializable_results[k][nested_k] = nested_v.to_dict()
                    else:
                        serializable_results[k][nested_k] = nested_v
            else:
                serializable_results[k] = v

        with open(filename, 'w') as f:
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
            with open(file, 'r') as f:
                comparisons.append(json.load(f))

        if not comparisons:
            return "No evaluation data available."

        # Aggregate statistics
        total_comparisons = len(comparisons)
        avg_cost_savings = np.mean([c.get('comparison', {}).get('cost_savings', 0)
                                   for c in comparisons])
        avg_quality_diff = np.mean([c.get('comparison', {}).get('quality_difference', {}).get('overall_diff', 0)
                                   for c in comparisons])

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