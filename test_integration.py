#!/usr/bin/env python3
"""Integration tests for the complete second brain system."""

import json
import logging
import os
import statistics
import sys
import time

import psutil
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import modules at toplevel to avoid import-outside-toplevel warnings
from evaluation_framework import QueryEvaluator
from interactive_session import InteractiveSession
from semantic_cache import SemanticCache
from summarizer import ModuleSummary
from embedding_router import EmbeddingRouter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init(autoreset=True)


class IntegrationTester:
    """Run comprehensive integration tests."""

    def __init__(self, test_corpus_path: str):
        self.test_corpus_path = test_corpus_path
        self.results = []

    def run_all_tests(self):
        """Run complete test suite."""
        print(f"{Fore.CYAN}{'='*60}")
        print("Second Brain Integration Test Suite")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        # Test 1: End-to-end workflow
        self.test_end_to_end_workflow()

        # Test 2: Performance benchmarks
        self.test_performance_benchmarks()

        # Test 3: Cost comparison
        self.test_cost_analysis()

        # Test 4: Cache effectiveness
        self.test_cache_effectiveness()

        # Test 5: Routing accuracy
        self.test_routing_accuracy()

        # Test 6: Memory usage
        self.test_memory_usage()

        # Generate report
        self.generate_report()

    def test_end_to_end_workflow(self):
        """Test complete workflow from loading to querying."""
        print(f"\n{Fore.YELLOW}Test 1: End-to-End Workflow{Style.RESET_ALL}")

        try:
            # Initialize session
            session = InteractiveSession()

            # Load documents
            print("- Loading test corpus...")
            session.onecmd(f"/load {self.test_corpus_path} --recursive")

            # Generate summaries
            print("- Generating module summaries...")
            session.onecmd("/summarize")

            # Enable embeddings
            print("- Enabling embedding routing...")
            session.onecmd("/use_embeddings on")

            # Test queries
            test_queries = [
                "What are the main machine learning algorithms?",
                "Explain gradient descent optimization",
                "How do neural networks work?"
            ]

            for i, query in enumerate(test_queries):
                print(f"- Testing query {i+1}: '{query[:50]}...'")
                try:
                    session.onecmd(f"/query {query}")
                    time.sleep(1)  # Avoid rate limits
                except Exception as query_error:
                    print(f"  Query {i+1} failed: {query_error}")
                    raise

            # Check costs
            print("- Checking costs...")
            try:
                session.onecmd("/costs")
            except Exception as cost_error:
                print(f"  Cost check failed: {cost_error}")
                raise

            # Save session
            print("- Saving session...")
            try:
                session.onecmd("/save test_session.json")
            except Exception as save_error:
                print(f"  Save failed: {save_error}")
                raise

            self.results.append(("End-to-End Workflow", "PASSED", "All operations completed"))
            print(f"{Fore.GREEN}✓ End-to-end workflow test passed{Style.RESET_ALL}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            import traceback
            error_details = f"{e}\n{traceback.format_exc()}"
            self.results.append(("End-to-End Workflow", "FAILED", error_details))
            print(f"{Fore.RED}✗ End-to-end workflow test failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}Full traceback: {traceback.format_exc()}{Style.RESET_ALL}")

    def test_performance_benchmarks(self):
        """Benchmark query performance."""
        print(f"\n{Fore.YELLOW}Test 2: Performance Benchmarks{Style.RESET_ALL}")

        session = InteractiveSession()
        session.onecmd(f"/load {self.test_corpus_path} --recursive")

        # Test queries of different complexities
        queries = {
            "simple": "What is machine learning?",
            "medium": "Compare supervised and unsupervised learning approaches",
            "complex": ("Analyze the relationship between gradient descent, "
                       "backpropagation, and neural network optimization")
        }

        results = {}

        for complexity, query in queries.items():
            times = []

            # Run each query 3 times
            for _ in range(3):
                start = time.time()
                session.onecmd(f"/query {query} --no-cache")
                elapsed = time.time() - start
                times.append(elapsed)
                time.sleep(1)

            avg_time = statistics.mean(times)
            results[complexity] = avg_time

            print(f"- {complexity}: {avg_time:.2f}s average")

        # Check if performance meets targets
        if (results["simple"] < 5 and results["medium"] < 10 and
                results["complex"] < 20):
            self.results.append(("Performance Benchmarks", "PASSED", f"Times: {results}"))
            print(f"{Fore.GREEN}✓ Performance targets met{Style.RESET_ALL}")
        else:
            self.results.append(("Performance Benchmarks", "WARNING",
                                f"Some queries slow: {results}"))
            print(f"{Fore.YELLOW}⚠ Some queries exceed target times{Style.RESET_ALL}")

    def test_cost_analysis(self):
        """Compare costs with and without optimizations."""
        print(f"\n{Fore.YELLOW}Test 3: Cost Analysis{Style.RESET_ALL}")

        session = InteractiveSession()
        session.onecmd(f"/load {self.test_corpus_path} --recursive")
        session.onecmd("/summarize")

        evaluator = QueryEvaluator()

        test_query = "Explain the key concepts in distributed systems"

        # Compare strategies
        results = evaluator.compare_strategies(test_query, session)

        if "comparison" in results:
            savings = results["comparison"]["cost_savings"]
            savings_pct = results["comparison"]["cost_savings_percentage"]

            print(f"- Cost savings: ${savings:.4f} ({savings_pct:.1f}%)")

            if savings_pct > 30:
                self.results.append(("Cost Analysis", "PASSED",
                                    f"Savings: {savings_pct:.1f}%"))
                print(f"{Fore.GREEN}✓ Significant cost savings achieved{Style.RESET_ALL}")
            else:
                self.results.append(("Cost Analysis", "WARNING",
                                    f"Low savings: {savings_pct:.1f}%"))
                print(f"{Fore.YELLOW}⚠ Lower than expected cost savings{Style.RESET_ALL}")

    def test_cache_effectiveness(self):
        """Test semantic cache hit rates."""
        print(f"\n{Fore.YELLOW}Test 4: Cache Effectiveness{Style.RESET_ALL}")

        cache = SemanticCache(similarity_threshold=0.85)

        # Test similar queries
        query_groups = [
            [
                "What is gradient descent?",
                "Explain gradient descent",
                "How does gradient descent work?",
                "Tell me about gradient descent optimization"
            ],
            [
                "What are neural networks?",
                "Explain neural networks",
                "How do neural networks function?",
                "Describe artificial neural networks"
            ]
        ]

        hits = 0
        total = 0

        for group in query_groups:
            # Cache first query
            cache.set(group[0], {"answer": f"Answer for {group[0]}", "cost": 0.001})

            # Test similar queries
            for query in group[1:]:
                result = cache.get(query)
                total += 1
                if result:
                    hits += 1
                    print(f"  ✓ Cache hit: '{query}'")
                else:
                    print(f"  ✗ Cache miss: '{query}'")

        hit_rate = (hits / total * 100) if total > 0 else 0
        print(f"\n- Cache hit rate: {hit_rate:.1f}%")

        if hit_rate > 60:
            self.results.append(("Cache Effectiveness", "PASSED",
                                f"Hit rate: {hit_rate:.1f}%"))
            print(f"{Fore.GREEN}✓ Good cache hit rate{Style.RESET_ALL}")
        else:
            self.results.append(("Cache Effectiveness", "WARNING",
                                f"Low hit rate: {hit_rate:.1f}%"))
            print(f"{Fore.YELLOW}⚠ Cache hit rate below target{Style.RESET_ALL}")

    def test_routing_accuracy(self):
        """Test if queries route to appropriate modules."""
        print(f"\n{Fore.YELLOW}Test 5: Routing Accuracy{Style.RESET_ALL}")

        router = EmbeddingRouter()

        # Create test summaries
        test_summaries = {
            "ML_Fundamentals": ModuleSummary(
                module_name="ML_Fundamentals",
                summary="Machine learning basics, algorithms, supervised and unsupervised learning",
                key_topics=["classification", "regression", "clustering", "neural networks"],
                document_count=10, total_tokens=50000, content_hash="abc",
                created_at="2024-01-01", model_used="gemini-flash"
            ),
            "Databases": ModuleSummary(
                module_name="Databases",
                summary="Database systems, SQL, transactions, ACID properties",
                key_topics=["SQL", "normalization", "transactions", "indexes"],
                document_count=8, total_tokens=40000, content_hash="def",
                created_at="2024-01-01", model_used="gemini-flash"
            ),
            "Networks": ModuleSummary(
                module_name="Networks",
                summary="Computer networks, protocols, TCP/IP, routing",
                key_topics=["TCP", "IP", "routing", "protocols", "OSI model"],
                document_count=12, total_tokens=60000, content_hash="ghi",
                created_at="2024-01-01", model_used="gemini-flash"
            )
        }

        router.index_modules(test_summaries)

        # Test routing accuracy
        test_cases = [
            ("How do neural networks learn?", "ML_Fundamentals"),
            ("Explain ACID properties in databases", "Databases"),
            ("What is TCP/IP?", "Networks"),
            ("Compare supervised and unsupervised learning", "ML_Fundamentals"),
            ("How to optimize SQL queries?", "Databases")
        ]

        correct = 0
        for query, expected_module in test_cases:
            results = router.route_query(query)
            if results and results[0][0] == expected_module:
                correct += 1
                print(f"  ✓ '{query}' → {expected_module}")
            else:
                actual = results[0][0] if results else "None"
                print(f"  ✗ '{query}' → {actual} (expected {expected_module})")

        accuracy = correct / len(test_cases) * 100
        print(f"\n- Routing accuracy: {accuracy:.1f}%")

        if accuracy >= 80:
            self.results.append(("Routing Accuracy", "PASSED",
                                f"Accuracy: {accuracy:.1f}%"))
            print(f"{Fore.GREEN}✓ Good routing accuracy{Style.RESET_ALL}")
        else:
            self.results.append(("Routing Accuracy", "WARNING",
                                f"Accuracy: {accuracy:.1f}%"))
            print(f"{Fore.YELLOW}⚠ Routing accuracy below target{Style.RESET_ALL}")

    def test_memory_usage(self):
        """Monitor memory usage during operations."""
        print(f"\n{Fore.YELLOW}Test 6: Memory Usage{Style.RESET_ALL}")

        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024
        print(f"- Baseline memory: {baseline_mb:.1f} MB")

        # Load large corpus
        session = InteractiveSession()
        session.onecmd(f"/load {self.test_corpus_path} --recursive")

        # Check memory after loading
        after_load_mb = process.memory_info().rss / 1024 / 1024
        load_increase = after_load_mb - baseline_mb
        print(f"- After loading: {after_load_mb:.1f} MB (+{load_increase:.1f} MB)")

        # Generate summaries
        session.onecmd("/summarize")

        # Check memory after summaries
        after_summary_mb = process.memory_info().rss / 1024 / 1024
        summary_increase = after_summary_mb - after_load_mb
        print(f"- After summaries: {after_summary_mb:.1f} MB (+{summary_increase:.1f} MB)")

        # Total increase
        total_increase = after_summary_mb - baseline_mb

        if total_increase < 500:  # Less than 500MB increase
            self.results.append(("Memory Usage", "PASSED",
                                f"Total increase: {total_increase:.1f} MB"))
            print(f"{Fore.GREEN}✓ Memory usage within limits{Style.RESET_ALL}")
        else:
            self.results.append(("Memory Usage", "WARNING",
                                f"High memory use: {total_increase:.1f} MB"))
            print(f"{Fore.YELLOW}⚠ High memory usage detected{Style.RESET_ALL}")

    def generate_report(self):
        """Generate test report."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print("Test Summary Report")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        # Count results
        passed = sum(1 for r in self.results if r[1] == "PASSED")
        warnings = sum(1 for r in self.results if r[1] == "WARNING")
        failed = sum(1 for r in self.results if r[1] == "FAILED")

        # Display results
        for test_name, status, details in self.results:
            if status == "PASSED":
                color = Fore.GREEN
                symbol = "✓"
            elif status == "WARNING":
                color = Fore.YELLOW
                symbol = "⚠"
            else:
                color = Fore.RED
                symbol = "✗"

            print(f"{color}{symbol} {test_name}: {status}{Style.RESET_ALL}")
            print(f"  {details}")

        # Summary
        print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
        print(f"  Passed: {passed}")
        print(f"  Warnings: {warnings}")
        print(f"  Failed: {failed}")

        # Save report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [{"test": r[0], "status": r[1], "details": r[2]} for r in self.results],
            "summary": {
                "passed": passed,
                "warnings": warnings,
                "failed": failed
            }
        }

        with open("integration_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print("\nReport saved to: integration_test_report.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_integration.py <path_to_test_corpus>")
        sys.exit(1)

    tester = IntegrationTester(sys.argv[1])
    tester.run_all_tests()
