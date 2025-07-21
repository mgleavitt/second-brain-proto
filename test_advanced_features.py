#!/usr/bin/env python3
"""Test advanced optimization and search features."""

import time
from colorama import init, Fore, Style
from summarizer import ModuleSummarizer, ModuleSummary
from model_config import ModelConfig
from semantic_cache import SemanticCache
from embedding_router import EmbeddingRouter
from evaluation_framework import QueryEvaluator

init(autoreset=True)


class DummyDoc:  # pylint: disable=too-few-public-methods
    """Dummy document class for testing purposes."""

    def __init__(self, content):
        self.page_content = content


def test_module_summaries():
    """Test module summarization."""
    print(f"\n{Fore.CYAN}Testing Module Summaries...{Style.RESET_ALL}")

    summarizer = ModuleSummarizer(ModelConfig())

    # Test with dummy documents
    docs = [
        DummyDoc("This module covers machine learning fundamentals including supervised learning, "
                "neural networks, and deep learning."),
        DummyDoc("We explore classification, regression, and clustering algorithms."),
        DummyDoc("Topics include backpropagation, gradient descent, and optimization.")
    ]

    summary = summarizer.get_or_generate_summary("ML_Module", docs, force_regenerate=True)

    print(f"Module: {summary.module_name}")
    print(f"Summary: {summary.summary[:200]}...")
    print(f"Key topics: {summary.key_topics}")
    print("✓ Summarization working")


def test_semantic_cache():
    """Test semantic caching."""
    print(f"\n{Fore.CYAN}Testing Semantic Cache...{Style.RESET_ALL}")

    cache = SemanticCache(similarity_threshold=0.8)

    # Cache a result
    cache.set("What is machine learning?", {
        "answer": "Machine learning is a subset of AI...",
        "cost": 0.001
    })

    # Test exact match
    result = cache.get("What is machine learning?")
    print(f"Exact match: {'✓' if result else '✗'}")

    # Test semantic match
    result = cache.get("Can you explain machine learning?")
    print(f"Semantic match: {'✓' if result else '✗'}")

    # Test non-match
    result = cache.get("What is quantum computing?")
    print(f"Non-match returns None: {'✓' if result is None else '✗'}")

    # Show stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")


def test_embedding_routing():
    """Test embedding-based routing."""
    print(f"\n{Fore.CYAN}Testing Embedding Router...{Style.RESET_ALL}")

    router = EmbeddingRouter()

    # Create dummy summaries
    summaries = {
        "ML_Fundamentals": ModuleSummary(
            module_name="ML_Fundamentals",
            summary="Machine learning basics including algorithms and theory",
            key_topics=["supervised learning", "neural networks", "classification"],
            document_count=10,
            total_tokens=50000,
            content_hash="abc123",
            created_at="2024-01-01",
            model_used="gemini-flash"
        ),
        "Database_Systems": ModuleSummary(
            module_name="Database_Systems",
            summary="Database design, SQL, and transaction management",
            key_topics=["SQL", "ACID", "normalization", "indexes"],
            document_count=8,
            total_tokens=40000,
            content_hash="def456",
            created_at="2024-01-01",
            model_used="gemini-flash"
        )
    }

    # Index modules
    router.index_modules(summaries)

    # Test routing
    query = "How do neural networks learn?"
    results = router.route_query(query)

    print(f"Query: '{query}'")
    print(f"Routed to: {[r[0] for r in results]}")
    print(f"Similarities: {[f'{r[1]:.3f}' for r in results]}")

    # Test explanation
    if results:
        explanation = router.explain_routing(query, results[0][0])
        print(f"Routing explanation: {explanation}")


def test_evaluation_framework():
    """Test query evaluation."""
    print(f"\n{Fore.CYAN}Testing Evaluation Framework...{Style.RESET_ALL}")

    evaluator = QueryEvaluator()

    # Test single evaluation
    metrics = evaluator.evaluate_response(
        question="What is gradient descent?",
        response=("Gradient descent is an optimization algorithm used to "
                 "minimize the cost function..."),
        metadata={
            'routing_used': True,
            'modules_queried': ['ML_Fundamentals'],
            'response_time': 2.5,
            'cost': 0.002
        }
    )

    print("Evaluation scores:")
    print(f"  Relevance: {metrics.quality.relevance_score:.3f}")
    print(f"  Completeness: {metrics.quality.completeness_score:.3f}")
    print(f"  Coherence: {metrics.quality.coherence_score:.3f}")
    print(f"  Overall: {metrics.overall_score():.3f}")


if __name__ == "__main__":
    print(f"{Fore.CYAN}{'='*60}")
    print("Testing Advanced Features")
    print(f"{'='*60}{Style.RESET_ALL}")

    test_module_summaries()
    time.sleep(1)

    test_semantic_cache()
    time.sleep(1)

    test_embedding_routing()
    time.sleep(1)

    test_evaluation_framework()

    print(f"\n{Fore.GREEN}All tests completed!{Style.RESET_ALL}")
