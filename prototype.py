#!/usr/bin/env python3
"""
Second Brain Prototype - Multi-Agent Document Synthesis System

This prototype demonstrates a multi-agent architecture for synthesizing information
across multiple documents using LLMs.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from colorama import init, Fore, Style
from tabulate import tabulate

# Initialize colorama for colored output
init()

# Load environment variables
load_dotenv()

# Configuration constants
DEFAULT_DOCUMENT_MODEL = "gemini-1.5-flash"
DEFAULT_SYNTHESIS_MODEL = "claude-3-opus-20240229"
DOCUMENTS_DIR = "documents"
LOGS_DIR = "logs"
QUERIES_LOG_FILE = "logs/queries.jsonl"

# Cost estimates (per 1K tokens) - these are approximate
COST_ESTIMATES = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},  # $0.075/$0.30 per 1M tokens
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},  # $15/$75 per 1M tokens
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},  # $3/$15 per 1M tokens
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},  # $0.25/$1.25 per 1M tokens
}


class SimpleCache:
    """In-memory cache for query results."""

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache[key] = value

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class QueryLogger:
    """Logs queries and responses to JSON lines file."""

    def __init__(self, log_path: str = QUERIES_LOG_FILE):
        self.log_path = log_path
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Ensure the log directory exists."""
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def log_query(self, query_data: Dict) -> None:
        """Log query data to JSON lines file."""
        query_data["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(query_data) + "\n")


class DocumentAgent:
    """Represents an agent responsible for a single document."""

    def __init__(self, name: str, document_path: str, model: str = DEFAULT_DOCUMENT_MODEL):
        self.name = name
        self.document_path = document_path
        self.model = model
        self.content = self._load_document()
        self.llm = self._initialize_llm()

        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0

    def _load_document(self) -> str:
        """Load document content from file."""
        try:
            with open(self.document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {self.document_path}")
        except Exception as e:
            raise Exception(f"Error loading document {self.document_path}: {e}")

    def _initialize_llm(self):
        """Initialize the LLM based on the model name."""
        if self.model.startswith("gemini"):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        elif self.model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def query(self, question: str) -> Dict[str, Any]:
        """Query the document agent with a question."""
        prompt = self._create_prompt(question)

        try:
            start_time = time.time()
            response = self.llm.invoke([HumanMessage(content=prompt)])
            duration = time.time() - start_time

            # Extract response content
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Estimate token usage and cost
            tokens_used = self._estimate_tokens(prompt, response_text)
            cost = self._calculate_cost(tokens_used)

            # Update tracking
            self.total_cost += cost
            self.total_tokens += tokens_used

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "agent_name": self.name
            }

        except Exception as e:
            print(f"{Fore.RED}Error querying {self.name}: {e}{Style.RESET_ALL}")
            return {
                "response": f"Error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.name
            }

    def _create_prompt(self, question: str) -> str:
        """Create the prompt for the document agent."""
        return f"""You are analyzing a specific document to answer a question.

Document Title: {self.name}
Document Content:
---
{self.content}
---

Question: {question}

Please extract all relevant information from this document that helps answer the question. Include specific quotes or references when applicable. If the document doesn't contain relevant information, state that clearly.

Focus on:
1. Direct answers to the question
2. Related concepts mentioned
3. Specific examples or details
4. Any limitations or caveats mentioned

Provide a clear, concise response based solely on the information in this document."""

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token usage (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage."""
        if self.model not in COST_ESTIMATES:
            return 0.0

        # Convert to per-token cost (costs are per 1M tokens)
        input_cost_per_token = COST_ESTIMATES[self.model]["input"] / 1_000_000
        output_cost_per_token = COST_ESTIMATES[self.model]["output"] / 1_000_000

        # Assume roughly 70% input, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens

        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)


class SynthesisAgent:
    """Synthesizes responses from multiple document agents."""

    def __init__(self, model: str = DEFAULT_SYNTHESIS_MODEL):
        self.model = model
        self.llm = self._initialize_llm()

        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0

    def _initialize_llm(self):
        """Initialize the LLM for synthesis."""
        if self.model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Synthesis agent requires Claude model, got: {self.model}")

    def synthesize(self, question: str, agent_responses: List[Dict]) -> Dict[str, Any]:
        """Synthesize responses from multiple document agents."""
        # Filter out error responses
        valid_responses = [r for r in agent_responses if not r["response"].startswith("Error:")]

        if not valid_responses:
            return {
                "response": "No valid responses from document agents to synthesize.",
                "tokens_used": 0,
                "cost": 0.0,
                "sources": []
            }

        # Format responses for synthesis
        formatted_responses = []
        sources = []
        for resp in valid_responses:
            formatted_responses.append(f"Source: {resp['agent_name']}\n{resp['response']}\n")
            sources.append(resp['agent_name'])

        prompt = self._create_synthesis_prompt(question, formatted_responses)

        try:
            start_time = time.time()
            response = self.llm.invoke([HumanMessage(content=prompt)])
            duration = time.time() - start_time

            response_text = response.content if hasattr(response, 'content') else str(response)

            # Estimate token usage and cost
            tokens_used = self._estimate_tokens(prompt, response_text)
            cost = self._calculate_cost(tokens_used)

            # Update tracking
            self.total_cost += cost
            self.total_tokens += tokens_used

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "sources": sources
            }

        except Exception as e:
            print(f"{Fore.RED}Error in synthesis: {e}{Style.RESET_ALL}")
            return {
                "response": f"Error during synthesis: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "sources": []
            }

    def _create_synthesis_prompt(self, question: str, formatted_responses: List[str]) -> str:
        """Create the synthesis prompt."""
        responses_text = "\n---\n".join(formatted_responses)

        return f"""You are synthesizing information from multiple sources to provide a comprehensive answer.

Original Question: {question}

Source Responses:
{responses_text}

Please create a unified, coherent response that:
1. Synthesizes information from all sources
2. Identifies connections and patterns across sources
3. Notes any contradictions or different perspectives
4. Provides a clear, comprehensive answer

Maintain source attribution by referencing which document each piece of information comes from.

Structure your response to be clear and well-organized, highlighting key insights and relationships between the different sources."""

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token usage."""
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage."""
        if self.model not in COST_ESTIMATES:
            return 0.0

        input_cost_per_token = COST_ESTIMATES[self.model]["input"] / 1_000_000
        output_cost_per_token = COST_ESTIMATES[self.model]["output"] / 1_000_000

        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens

        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)


class SecondBrainPrototype:
    """Main orchestrator for the prototype."""

    def __init__(self):
        self.document_agents = []
        self.synthesis_agent = None  # Initialize lazily when needed
        self.cache = SimpleCache()
        self.logger = QueryLogger()

        # Cost tracking
        self.total_queries = 0
        self.total_cost = 0.0
        self.total_tokens = 0

    def add_document(self, name: str, path: str) -> None:
        """Add a document agent."""
        try:
            agent = DocumentAgent(name, path)
            self.document_agents.append(agent)
            print(f"{Fore.GREEN}✓ Added document agent: {name}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to add document {name}: {e}{Style.RESET_ALL}")

    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Query the second brain system."""
        start_time = time.time()

        # Check cache first
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                print(f"{Fore.YELLOW}Cache hit!{Style.RESET_ALL}")
                return cached_result

        print(f"{Fore.CYAN}Querying {len(self.document_agents)} document agents...{Style.RESET_ALL}")

        # Query all document agents
        agent_responses = []
        for agent in self.document_agents:
            print(f"  Querying {agent.name}...")
            response = agent.query(question)
            agent_responses.append(response)

        # Synthesize responses
        print(f"{Fore.CYAN}Synthesizing responses...{Style.RESET_ALL}")
        if self.synthesis_agent is None:
            self.synthesis_agent = SynthesisAgent()
        synthesis_result = self.synthesis_agent.synthesize(question, agent_responses)

        # Calculate totals
        total_cost = sum(r["cost"] for r in agent_responses) + synthesis_result["cost"]
        total_tokens = sum(r["tokens_used"] for r in agent_responses) + synthesis_result["tokens_used"]
        duration = time.time() - start_time

        # Prepare result
        result = {
            "response": synthesis_result["response"],
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "duration": duration,
            "cache_hit": False,
            "sources": synthesis_result["sources"],
            "agent_responses": agent_responses,
            "synthesis_result": synthesis_result
        }

        # Cache the result
        if use_cache:
            self.cache.set(cache_key, result)

        # Update tracking
        self.total_queries += 1
        self.total_cost += total_cost
        self.total_tokens += total_tokens

        # Log the query
        self.logger.log_query({
            "question": question,
            "result": result,
            "agent_responses": agent_responses
        })

        return result

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics."""
        cache_stats = self.cache.get_stats()

        # Calculate costs by model
        costs_by_model = {}
        for agent in self.document_agents:
            model = agent.model
            if model not in costs_by_model:
                costs_by_model[model] = {"cost": 0.0, "tokens": 0}
            costs_by_model[model]["cost"] += agent.total_cost
            costs_by_model[model]["tokens"] += agent.total_tokens

        # Add synthesis agent costs
        if self.synthesis_agent is not None:
            synthesis_model = self.synthesis_agent.model
            if synthesis_model not in costs_by_model:
                costs_by_model[synthesis_model] = {"cost": 0.0, "tokens": 0}
            costs_by_model[synthesis_model]["cost"] += self.synthesis_agent.total_cost
            costs_by_model[synthesis_model]["tokens"] += self.synthesis_agent.total_tokens

        return {
            "total_queries": self.total_queries,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "cache_stats": cache_stats,
            "costs_by_model": costs_by_model,
            "average_cost_per_query": self.total_cost / self.total_queries if self.total_queries > 0 else 0,
            "estimated_monthly_cost": (self.total_cost / self.total_queries * 100) if self.total_queries > 0 else 0
        }


def display_query_response(question: str, result: Dict[str, Any]):
    """Display query response in a formatted way."""
    print(f"\n{Fore.BLUE}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}=== Query ==={Style.RESET_ALL}")
    print(f"{question}")
    print(f"\n{Fore.BLUE}=== Response ==={Style.RESET_ALL}")
    print(f"{result['response']}")

    if result['sources']:
        print(f"\n{Fore.BLUE}=== Sources ==={Style.RESET_ALL}")
        for source in result['sources']:
            print(f"- {source}")

    print(f"\n{Fore.BLUE}=== Metrics ==={Style.RESET_ALL}")
    print(f"Total Cost: ${result['total_cost']:.4f}")
    print(f"Total Tokens: {result['total_tokens']:,}")
    print(f"Response Time: {result['duration']:.1f}s")
    print(f"Cache Hit: {'Yes' if result['cache_hit'] else 'No'}")
    print(f"{Fore.BLUE}{'='*50}{Style.RESET_ALL}\n")


def display_cost_summary(summary: Dict[str, Any]):
    """Display cost summary in a formatted way."""
    print(f"\n{Fore.BLUE}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}=== Cost Summary ==={Style.RESET_ALL}")
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Cache Hits: {summary['cache_stats']['hits']} ({summary['cache_stats']['hit_rate']:.1%})")
    print(f"Total Cost: ${summary['total_cost']:.4f}")

    print(f"\n{Fore.BLUE}By Model:{Style.RESET_ALL}")
    for model, data in summary['costs_by_model'].items():
        print(f"- {model}: ${data['cost']:.4f} ({data['tokens']:,} tokens)")

    print(f"\nAverage Cost per Query: ${summary['average_cost_per_query']:.4f}")
    print(f"Estimated Monthly Cost (100 queries): ${summary['estimated_monthly_cost']:.2f}")
    print(f"{Fore.BLUE}{'='*50}{Style.RESET_ALL}\n")


def run_test_queries(prototype: SecondBrainPrototype):
    """Run the predefined test queries."""
    test_queries = [
        "What optimization techniques are discussed across the documents?",
        "How do gradient descent and A* search compare in terms of finding optimal solutions?",
        "What role does computation cost play in the algorithms discussed?",
        "Explain the concept of 'attention' if mentioned in any document",
        "What are the trade-offs between different approaches mentioned?"
    ]

    print(f"{Fore.GREEN}Running {len(test_queries)} test queries...{Style.RESET_ALL}\n")

    for i, query in enumerate(test_queries, 1):
        print(f"{Fore.YELLOW}Test Query {i}/{len(test_queries)}{Style.RESET_ALL}")
        result = prototype.query(query)
        display_query_response(query, result)


def interactive_mode(prototype: SecondBrainPrototype):
    """Run interactive mode for user queries."""
    print(f"{Fore.GREEN}Interactive mode - type 'quit' to exit{Style.RESET_ALL}\n")

    while True:
        try:
            question = input(f"{Fore.CYAN}Enter your question: {Style.RESET_ALL}")
            if question.lower() in ['quit', 'exit', 'q']:
                break

            if question.strip():
                result = prototype.query(question)
                display_query_response(question, result)
            else:
                print("Please enter a question.")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exiting interactive mode...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Second Brain Prototype")
    parser.add_argument("command", choices=["query", "test", "costs", "clear-cache", "interactive"],
                       help="Command to run")
    parser.add_argument("--question", "-q", help="Question to ask (for query command)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    # Initialize the prototype
    prototype = SecondBrainPrototype()

    # Add default documents
    default_docs = [
        ("CS229_Agent", "documents/cs229_optimization.txt"),
        ("CS221_Agent", "documents/cs221_search.txt"),
        ("Paper_Agent", "documents/paper_transformers.txt")
    ]

    for name, path in default_docs:
        if os.path.exists(path):
            prototype.add_document(name, path)
        else:
            print(f"{Fore.RED}Warning: Document not found: {path}{Style.RESET_ALL}")

    if not prototype.document_agents:
        print(f"{Fore.RED}Error: No document agents could be loaded. Please check the documents directory.{Style.RESET_ALL}")
        sys.exit(1)

    # Execute command
    if args.command == "query":
        if not args.question:
            print(f"{Fore.RED}Error: Please provide a question with --question{Style.RESET_ALL}")
            sys.exit(1)

        result = prototype.query(args.question, use_cache=not args.no_cache)
        display_query_response(args.question, result)

    elif args.command == "test":
        run_test_queries(prototype)

    elif args.command == "costs":
        summary = prototype.get_cost_summary()
        display_cost_summary(summary)

    elif args.command == "clear-cache":
        prototype.cache.clear()
        print(f"{Fore.GREEN}Cache cleared!{Style.RESET_ALL}")

    elif args.command == "interactive":
        interactive_mode(prototype)


if __name__ == "__main__":
    main()