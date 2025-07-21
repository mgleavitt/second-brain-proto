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

from dotenv import load_dotenv
from colorama import init, Fore, Style

# Import new modules
from loaders.document_loader import DocumentLoader
from loaders.course_loader import CourseModuleLoader
from agents.routing_agent import QueryType
from agents.module_agent import ModuleAgent
from agents.document_agent import DocumentAgent, SynthesisAgent
from prompt_manager import PromptManager
from model_config import ModelConfig
from interactive_session import InteractiveSession

# Initialize colorama for colored output
init()

# Load environment variables
load_dotenv()

# Configuration constants
DOCUMENTS_DIR = "documents"
LOGS_DIR = "logs"
QUERIES_LOG_FILE = "logs/queries.jsonl"


class SimpleCache:  # pylint: disable=too-few-public-methods
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
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "hit_rate": hit_rate
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class QueryLogger:  # pylint: disable=too-few-public-methods
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
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(query_data) + "\n")


class SecondBrainPrototype:  # pylint: disable=too-many-instance-attributes
    """Enhanced orchestrator with routing capabilities."""

    def __init__(self, prompt_manager: Optional[PromptManager] = None):
        self.document_agents = []
        self.module_agents = {}  # New: module-based agents
        self.routing_agent = None  # Will be initialized when modules are added
        self.synthesis_agent = None
        self.prompt_manager = prompt_manager or PromptManager()
        self.cache = SimpleCache()
        self.logger = QueryLogger()

        # Cost tracking
        self.total_queries = 0
        self.total_cost = 0.0
        self.total_tokens = 0

        # Routing configuration
        self.routing_config = {
            'threshold_percentage': 0.5,    # Include modules with >= 50% of top score
            'max_modules_default': 4,       # Default max modules to query
            'sample_chunks': 10,            # Number of chunks to sample for routing
            'weight_decay': True,           # Give more weight to earlier chunks
        }
        self._max_modules_override = None  # For dynamic adjustment based on complexity

    def add_document(self, name: str, path: str) -> None:
        """Add a document agent."""
        try:
            agent = DocumentAgent(name, path, prompt_manager=self.prompt_manager)
            self.document_agents.append(agent)
            success_msg = f"✓ Added document agent: {name}"
            print(f"{Fore.GREEN}{success_msg}{Style.RESET_ALL}")
        except (FileNotFoundError, ValueError) as e:
            error_msg = f"✗ Failed to add document {name}: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")

    def add_module(self, module_name: str, documents: List[Dict]) -> None:
        """Add a module agent for handling multiple related documents."""
        try:
            agent = ModuleAgent(module_name, documents, prompt_manager=self.prompt_manager)
            self.module_agents[module_name] = agent
            doc_count = len(documents)
            msg = (
                f"✓ Added module agent: {module_name} "
                f"({doc_count} docs)"
            )
            print(
                f"{Fore.GREEN}{msg}"
                f"{Style.RESET_ALL}"
            )
        except (ValueError, KeyError) as e:
            print(
                f"{Fore.RED}✗ Failed to add module {module_name}: {e}"
                f"{Style.RESET_ALL}"
            )

    def load_from_paths(self, paths: List[str], recursive: bool = False) -> None:  # pylint: disable=too-many-locals
        """Load documents from specified paths."""
        loader = DocumentLoader()

        for path in paths:
            try:
                # Check if this is a course directory structure
                path_obj = Path(path)
                has_transcripts = (path_obj / "transcripts").exists()
                if path_obj.is_dir() and has_transcripts:
                    # Load as course
                    print(
                        f"{Fore.CYAN}Loading course: {path_obj.name}{Style.RESET_ALL}"
                    )
                    course_loader = CourseModuleLoader(path_obj)
                    modules = course_loader.load_course()

                    for module_name, documents in modules.items():
                        if documents:
                            self.add_module(module_name, documents)
                else:
                    # Load as regular documents
                    docs = loader.load_path(path, recursive)
                    for name, content in docs:
                        # Save to temp file for compatibility
                        temp_path = f"/tmp/{name.replace(' ', '_')}.txt"
                        Path(temp_path).write_text(content, encoding="utf-8")
                        self.add_document(name, temp_path)

            except (FileNotFoundError, PermissionError) as e:
                print(f"{Fore.RED}Error loading {path}: {e}{Style.RESET_ALL}")

    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """Query the second brain system."""
        start_time = time.time()

        # Check cache first
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                print(f"{Fore.YELLOW}Cache hit!{Style.RESET_ALL}")
                return cached_result

        # Combine document agents and module agents
        all_agents = list(self.document_agents)

        # Add module agents to the query list
        for _, module_agent in self.module_agents.items():
            all_agents.append(module_agent)

        agent_count = len(all_agents)
        doc_count = len(self.document_agents)
        module_count = len(self.module_agents)
        print(f"{Fore.CYAN}Querying {agent_count} agents "
              f"({doc_count} documents, {module_count} modules)...{Style.RESET_ALL}")

        # Query all agents
        agent_responses = []
        for agent in all_agents:
            agent_name = (agent.name if hasattr(agent, 'name')
                         else agent.module_name)
            print(f"  Querying {agent_name}...")
            response = agent.query(question)
            agent_responses.append(response)

        # Synthesize responses
        print(f"{Fore.CYAN}Synthesizing responses...{Style.RESET_ALL}")
        if self.synthesis_agent is None:
            self.synthesis_agent = SynthesisAgent(
                model_config=self.prompt_manager.model_config,
                prompt_manager=self.prompt_manager
            )
        synthesis_result = self.synthesis_agent.synthesize(question, agent_responses)

        # Calculate totals
        total_cost = sum(r["cost"] for r in agent_responses) + synthesis_result["cost"]
        total_tokens = (sum(r["tokens_used"] for r in agent_responses) +
                       synthesis_result["tokens_used"])
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

    def _query_single_agent(self, question: str, routing_decision: Dict) -> Dict[str, Any]:
        """Query using single best agent for simple queries."""
        # For now, use the first available agent
        if self.document_agents:
            agent = self.document_agents[0]
            response = agent.query(question)
            return {
                "response": response["response"],
                "total_cost": response["cost"],
                "total_tokens": response["tokens_used"],
                "duration": response["duration"],
                "cache_hit": False,
                "sources": [response["agent_name"]],
                "routing_decision": routing_decision
            }

        return {
            "response": "No agents available for query.",
            "total_cost": 0.0,
            "total_tokens": 0,
            "duration": 0.0,
            "cache_hit": False,
            "sources": [],
            "routing_decision": routing_decision
        }

    def _query_module(self, question: str, routing_decision: Dict) -> Dict[str, Any]:
        """Query relevant module agents."""
        if not self.module_agents:
            # Fall back to regular query if no modules
            return self.query(question)

        # For now, query all module agents
        agent_responses = []
        for module_name, agent in self.module_agents.items():
            print(f"  Querying {module_name}...")
            response = agent.query(question)
            agent_responses.append(response)

        # Synthesize if multiple responses
        if len(agent_responses) > 1:
            if self.synthesis_agent is None:
                self.synthesis_agent = SynthesisAgent(
                    model_config=self.prompt_manager.model_config,
                    prompt_manager=self.prompt_manager
                )
            synthesis_result = self.synthesis_agent.synthesize(
                question, agent_responses
            )
            return {
                "response": synthesis_result["response"],
                "total_cost": (
                    sum(r["cost"] for r in agent_responses)
                    + synthesis_result["cost"]
                ),
                "total_tokens": (
                    sum(r["tokens_used"] for r in agent_responses)
                    + synthesis_result["tokens_used"]
                ),
                "duration": sum(r["duration"] for r in agent_responses),
                "cache_hit": False,
                "sources": synthesis_result["sources"],
                "routing_decision": routing_decision
            }

        response = agent_responses[0]
        return {
            "response": response["response"],
            "total_cost": response["cost"],
            "total_tokens": response["tokens_used"],
            "duration": response["duration"],
            "cache_hit": False,
            "sources": [response["agent_name"]],
            "routing_decision": routing_decision
        }

    def _query_with_synthesis(self, question: str, routing_decision: Dict) -> Dict[str, Any]:
        """Query with full synthesis pipeline."""
        # Use the original query method for complex queries
        result = self.query(question)
        result["routing_decision"] = routing_decision
        return result

    def query_with_routing(self, question: str, use_cache: bool = True) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """Query with intelligent routing based on content relevance."""
        start_time = time.time()

        # Check cache first
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                print(f"{Fore.YELLOW}Cache hit!{Style.RESET_ALL}")
                return cached_result

        # Estimate query complexity
        complexity = self._estimate_query_complexity(question)
        print(f"{Fore.CYAN}Query complexity: {complexity}{Style.RESET_ALL}")

        # Adjust routing based on complexity
        if complexity == "simple":
            self._max_modules_override = 2
        elif complexity == "comparison":
            self._max_modules_override = 4
        elif complexity == "synthesis":
            self._max_modules_override = 6
        else:
            self._max_modules_override = 3

        # Determine relevant modules using content-based routing
        relevant_modules = self._route_query(question)

        print(f"{Fore.CYAN}Routing to {len(relevant_modules)} modules: "
              f"{relevant_modules}{Style.RESET_ALL}")

        # Query only relevant modules
        agent_responses = []
        for module_name in relevant_modules:
            if module_name in self.module_agents:
                print(f"  Querying {module_name}...")
                response = self.module_agents[module_name].query(question)
                agent_responses.append(response)

        # Add any document agents if present
        for agent in self.document_agents:
            print(f"  Querying {agent.name}...")
            response = agent.query(question)
            agent_responses.append(response)

        # Synthesize responses
        print(f"{Fore.CYAN}Synthesizing responses...{Style.RESET_ALL}")
        if self.synthesis_agent is None:
            self.synthesis_agent = SynthesisAgent(
                model_config=self.prompt_manager.model_config,
                prompt_manager=self.prompt_manager
            )
        synthesis_result = self.synthesis_agent.synthesize(question, agent_responses)

        # Calculate totals
        total_cost = sum(r["cost"] for r in agent_responses) + synthesis_result["cost"]
        total_tokens = (sum(r["tokens_used"] for r in agent_responses) +
                       synthesis_result["tokens_used"])
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
            "synthesis_result": synthesis_result,
            "modules_queried": relevant_modules
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
            "agent_responses": agent_responses,
            "modules_queried": relevant_modules
        })

        return result

    def _route_query(self, question: str) -> List[str]:  # pylint: disable=too-many-locals
        """Determine which modules are relevant to the query using content analysis."""
        question_lower = question.lower()
        question_words = set(question_lower.split())

        # Score each module based on content relevance
        module_scores = {}

        for module_name, module_agent in self.module_agents.items():
            score = 0
            matches = 0

            # Check chunk content for keyword matches
            for i, chunk in enumerate(module_agent.chunks[:self.routing_config['sample_chunks']]):
                chunk_words = set(chunk['text'].lower().split())
                common_words = question_words & chunk_words
                if common_words:
                    # Count matches but normalize by position (earlier chunks = higher weight)
                    if self.routing_config['weight_decay']:
                        weight = 1.0 / (i + 1)  # First chunk = 1.0, second = 0.5, etc.
                    else:
                        weight = 1.0
                    matches += len(common_words) * weight

            # Normalize score by number of question words
            if len(question_words) > 0:
                score = matches / len(question_words)

            module_scores[module_name] = score

        # Sort modules by score
        sorted_modules = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)

        # Print all scores for debugging
        for module_name, score in sorted_modules:
            print(f"  {module_name}: relevance score {score:.2f}")

        # Strategy 1: Include modules with score > X% of top score
        relevant_modules = []
        if sorted_modules:
            top_score = sorted_modules[0][1]
            threshold_percentage = self.routing_config['threshold_percentage']

            for module_name, score in sorted_modules:
                if score >= (top_score * threshold_percentage) and score > 0:
                    relevant_modules.append(module_name)

        # Strategy 2: Limit maximum modules to query
        max_modules = getattr(self,  # pylint: disable=no-member
                              '_max_modules_override',
                              self.routing_config['max_modules_default'])
        if len(relevant_modules) > max_modules:
            limit_msg = f"Limiting to top {max_modules} modules for cost control"
            print(f"{Fore.YELLOW}{limit_msg}{Style.RESET_ALL}")
            relevant_modules = relevant_modules[:max_modules]

        # Fallback: If no modules selected, use top 2
        if not relevant_modules and sorted_modules:
            print(f"{Fore.YELLOW}No modules met threshold, using top 2{Style.RESET_ALL}")
            relevant_modules = [m[0] for m in sorted_modules[:2]]

        return relevant_modules

    def _estimate_query_complexity(self, question: str) -> str:
        """Estimate query complexity to adjust routing strategy."""
        question_lower = question.lower()

        # Simple questions (likely single module)
        if any(phrase in question_lower for phrase in ["what is", "define", "explain"]):
            return "simple"

        # Comparison questions (likely need multiple modules)
        comparison_phrases = ["compare", "contrast", "versus", "vs", "difference between"]
        if any(phrase in question_lower for phrase in comparison_phrases):
            return "comparison"

        # Synthesis questions (might need many modules)
        synthesis_phrases = ["analyze", "design", "create", "evaluate", "how does"]
        if any(phrase in question_lower for phrase in synthesis_phrases):
            return "synthesis"

        return "moderate"

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics."""
        cache_stats = self.cache.get_stats()

        # Calculate costs by model
        costs_by_model = {}

        # Add document agent costs
        for agent in self.document_agents:
            model = agent.model
            if model not in costs_by_model:
                costs_by_model[model] = {"cost": 0.0, "tokens": 0}
            costs_by_model[model]["cost"] += agent.total_cost
            costs_by_model[model]["tokens"] += agent.total_tokens

        # Add module agent costs
        for agent in self.module_agents.values():
            model = agent.model
            if model not in costs_by_model:
                costs_by_model[model] = {"cost": 0.0, "tokens": 0}
            costs_by_model[model]["cost"] += agent.total_cost
            costs_by_model[model]["tokens"] += agent.total_tokens

        # Add synthesis agent costs
        if self.synthesis_agent is not None:
            synthesis_model = self.synthesis_agent.model_name
            if synthesis_model not in costs_by_model:
                costs_by_model[synthesis_model] = {"cost": 0.0, "tokens": 0}
            costs_by_model[synthesis_model]["cost"] += self.synthesis_agent.total_cost
            costs_by_model[synthesis_model]["tokens"] += self.synthesis_agent.total_tokens

        avg_cost = self.total_cost / self.total_queries if self.total_queries > 0 else 0
        monthly_estimate = ((self.total_cost / self.total_queries * 100)
                           if self.total_queries > 0 else 0)

        return {
            "total_queries": self.total_queries,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "cache_stats": cache_stats,
            "costs_by_model": costs_by_model,
            "average_cost_per_query": avg_cost,
            "estimated_monthly_cost": monthly_estimate
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
    hit_rate = summary['cache_stats']['hit_rate']
    print(f"Cache Hits: {summary['cache_stats']['hits']} ({hit_rate:.1%})")
    print(f"Total Cost: ${summary['total_cost']:.4f}")

    print(f"\n{Fore.BLUE}By Model:{Style.RESET_ALL}")
    for model, data in summary['costs_by_model'].items():
        print(f"- {model}: ${data['cost']:.4f} ({data['tokens']:,} tokens)")

    print(f"\nAverage Cost per Query: ${summary['average_cost_per_query']:.4f}")
    monthly_cost = summary['estimated_monthly_cost']
    print(
        f"Estimated Monthly Cost (100 queries): ${monthly_cost:.2f}"
    )
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


def interactive_mode(prototype: SecondBrainPrototype):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks
    """Run interactive mode for user queries."""
    print(f"{Fore.GREEN}Interactive mode - type '/quit' to exit{Style.RESET_ALL}\n")

    def print_help():
        print(f"\n{Fore.BLUE}=== Interactive Mode Help ==={Style.RESET_ALL}")
        print("\nNavigation Commands:")
        print("  /quit, /exit, /q              Exit interactive mode")
        print("  /help, /?                     Show this help message")

        print("\nPrompt Management Commands:")
        print("  /prompt-reload                Reload all system prompts from files")
        print("  /prompt-set <key> <file>      Set a custom prompt file for a key")
        print("  /prompt-show <key>            Show the current prompt for a key")
        print("  /prompt-create-defaults       Create default prompt files in the prompts directory") #pylint: disable=line-too-long
        print("  /prompt-help                  Show prompt-specific help")

        print("\nSystem Commands:")
        print("  /costs                        Show cost summary")
        print("  /clear-cache                  Clear the query cache")
        print("  /cache-stats                  Show cache statistics")

        print("\nValid prompt keys: document_single, document_multi, module, synthesis")
        print("\nRegular questions (without /) will be processed using intelligent routing for cost optimization.\n") #pylint: disable=line-too-long

    def print_prompt_help():
        print("\nPrompt management commands:")
        print("  /prompt-reload                Reload all system prompts from files")
        print("  /prompt-set <key> <file>      Set a custom prompt file for a key")
        print("  /prompt-show <key>            Show the current prompt for a key")
        print("  /prompt-create-defaults       Create default prompt files in the prompts directory") #pylint: disable=line-too-long
        print("  /prompt-help                  Show this help message\n")
        print("Valid keys: document_single, document_multi, module, synthesis\n")

    while True:
        try:
            question = input(f"{Fore.CYAN}Enter your question: {Style.RESET_ALL}")

            # Handle all slash commands
            if question.strip().startswith("/"):
                cmd = question.strip().split()[0].lower()
                args = question.strip()[len(cmd):].strip()

                # Navigation commands
                if cmd in ['/quit', '/exit', '/q']:
                    break
                if cmd in ['/help', '/?']:
                    print_help()
                    continue

                # Prompt management commands
                if cmd == "/prompt-reload":
                    reloaded = prototype.prompt_manager.reload_prompts()
                    print(f"Reloaded {len(reloaded)} prompts:")
                    for key, prompt in reloaded.items():
                        print(f"  - {key}: {len(prompt)} characters")
                elif cmd == "/prompt-set":
                    parts = args.split(maxsplit=1)
                    if len(parts) != 2:
                        print("Usage: /prompt-set <prompt_key> <filepath>")
                        print("Valid keys: document_single, document_multi, module, synthesis")
                    else:
                        prompt_key, filepath = parts
                        try:
                            prototype.prompt_manager.set_custom_prompt_path(prompt_key, filepath)
                            print(f"Set custom prompt for '{prompt_key}' from {filepath}")
                        except FileNotFoundError as e:
                            print(f"Error: {e}")
                elif cmd == "/prompt-show":
                    if not args:
                        print("Usage: /prompt-show <prompt_key>")
                        print("Valid keys: document_single, document_multi, module, synthesis")
                    else:
                        prompt_key = args.strip()
                        prompt = prototype.prompt_manager.get_prompt(prompt_key)
                        print(f"\n--- {prompt_key} prompt ---")
                        print(prompt)
                        print("--- end prompt ---\n")
                elif cmd == "/prompt-create-defaults":
                    prototype.prompt_manager.create_default_prompt_files()
                    print(f"Created default prompt files in {prototype.prompt_manager.prompt_dir}")
                elif cmd == "/prompt-help":
                    print_prompt_help()

                # System commands
                elif cmd == "/costs":
                    summary = prototype.get_cost_summary()
                    display_cost_summary(summary)
                elif cmd == "/clear-cache":
                    prototype.cache.clear()
                    print(f"{Fore.GREEN}Cache cleared!{Style.RESET_ALL}")
                elif cmd == "/cache-stats":
                    stats = prototype.cache.get_stats()
                    print("\nCache Statistics:")
                    print(f"  Hits: {stats['hits']}")
                    print(f"  Misses: {stats['misses']}")
                    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
                    print(f"  Cache Size: {stats['size']} entries")
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help or /? for available commands.")
                continue

            # Handle regular questions (no slash)
            if question.strip():
                result = prototype.query_with_routing(question)
                display_query_response(question, result)
            else:
                print("Please enter a question or command (use /help for available commands).")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exiting interactive mode...{Style.RESET_ALL}")
            break
        except (ValueError, IOError) as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def run_scale_test(prototype: SecondBrainPrototype):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Run scale testing with real course materials."""
    test_queries = [
        # Simple queries
        ("What is a B+ tree?", QueryType.SIMPLE),
        ("Define ACID properties", QueryType.SIMPLE),

        # Single module queries
        ("Explain indexing strategies in databases", QueryType.SINGLE_MODULE),
        ("What are the components of a DBMS?", QueryType.SINGLE_MODULE),

        # Cross-module queries
        ("Compare B+ trees with hash indexes for different workloads", QueryType.CROSS_MODULE),
        ("How do transactions relate to recovery mechanisms?", QueryType.CROSS_MODULE),

        # Synthesis queries
        ("Design a database schema for a social media application", QueryType.SYNTHESIS),
        ("Analyze the trade-offs between consistency and performance", QueryType.SYNTHESIS)
    ]

    results = []
    print(f"{Fore.GREEN}Running scale test with {len(test_queries)} queries...{Style.RESET_ALL}\n")

    for query, expected_type in test_queries:
        print(f"{Fore.YELLOW}Testing: {query}{Style.RESET_ALL}")

        # Test with routing
        result = prototype.query_with_routing(query)

        results.append({
            "query": query,
            "expected_type": expected_type,
            "actual_type": result.get('routing_decision', {}).get('query_type'),
            "cost": result['total_cost'],
            "tokens": result['total_tokens'],
            "time": result['duration']
        })

    # Display results summary
    display_scale_test_results(results)


def display_scale_test_results(results: List[Dict]):
    """Display scale test results summary."""
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}=== Scale Test Results ==={Style.RESET_ALL}")

    total_cost = sum(r['cost'] for r in results)
    total_tokens = sum(r['tokens'] for r in results)
    total_time = sum(r['time'] for r in results)

    print(
        f"Total Queries: {len(results)}"
    )
    print(
        f"Total Cost: ${total_cost:.4f}"
    )
    print(
        f"Total Tokens: {total_tokens:,}"
    )
    print(
        f"Total Time: {total_time:.1f}s"
    )
    print(
        f"Average Cost per Query: ${total_cost/len(results):.4f}"
    )

    # Routing accuracy
    correct_routing = sum(1 for r in results if r['expected_type'] == r['actual_type'])
    routing_accuracy = correct_routing / len(results)
    print(
        f"Routing Accuracy: {routing_accuracy:.1%}"
    )

    print(f"\n{Fore.BLUE}By Query Type:{Style.RESET_ALL}")
    type_stats = {}
    for result in results:
        query_type = result['actual_type'].value if result['actual_type'] else 'unknown'
        if query_type not in type_stats:
            type_stats[query_type] = {'count': 0, 'total_cost': 0, 'total_tokens': 0}
        type_stats[query_type]['count'] += 1
        type_stats[query_type]['total_cost'] += result['cost']
        type_stats[query_type]['total_tokens'] += result['tokens']

    for query_type, stats in type_stats.items():
        avg_cost = stats['total_cost'] / stats['count']
        print(f"- {query_type}: {stats['count']} queries, ${avg_cost:.4f} avg cost")

    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")


def _handle_query_command(prototype: SecondBrainPrototype, args):
    """Handle the query command."""
    if not args.question:
        print(f"{Fore.RED}Error: Please provide a question with --question{Style.RESET_ALL}")
        sys.exit(1)

    # Use routing if specified
    if args.use_routing:
        result = prototype.query_with_routing(args.question, use_cache=not args.no_cache)
    else:
        result = prototype.query(args.question, use_cache=not args.no_cache)

    display_query_response(args.question, result)


def _load_default_documents(prototype: SecondBrainPrototype):
    """Load default documents if they exist."""
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


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Enhanced CLI interface."""
    parser = argparse.ArgumentParser(description="Second Brain Prototype - Scale Testing Version")
    parser.add_argument("command", choices=["query", "test", "costs", "clear-cache",
                                          "interactive", "scale-test"],
                       help="Command to run")
    parser.add_argument("--question", "-q", help="Question to ask")
    parser.add_argument("--documents", "-d", nargs="+",
                       help="Documents or directories to load (can specify multiple)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively search directories for documents")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--use-routing", action="store_true",
                       help="Use keyword-based routing to reduce costs")

    # Model selection arguments
    parser.add_argument('--document-model', type=str,
                       help='Model to use for document agents')
    parser.add_argument('--module-model', type=str,
                       help='Model to use for module agents')
    parser.add_argument('--synthesis-model', type=str,
                       help='Model to use for synthesis agent')
    parser.add_argument('--routing-model', type=str,
                       help='Model to use for routing agent')

    # Prompt management arguments
    parser.add_argument('--prompt-dir', type=str, default='prompts',
                        help='Directory containing prompt files')
    parser.add_argument('--document-single-prompt', type=str,
                        help='Path to custom document single prompt file')
    parser.add_argument('--document-multi-prompt', type=str,
                        help='Path to custom document multi prompt file')
    parser.add_argument('--module-prompt', type=str,
                        help='Path to custom module prompt file')
    parser.add_argument('--synthesis-prompt', type=str,
                        help='Path to custom synthesis prompt file')

    args = parser.parse_args()

    # Initialize model configuration
    model_config = ModelConfig()

    # Apply model overrides from CLI
    if args.document_model:
        model_config.set_model('document', args.document_model)
    if args.module_model:
        model_config.set_model('module', args.module_model)
    if args.synthesis_model:
        model_config.set_model('synthesis', args.synthesis_model)
    if args.routing_model:
        model_config.set_model('routing', args.routing_model)

    # Initialize the prototype with custom prompt manager
    prompt_manager = PromptManager(args.prompt_dir)

    # Set custom paths if provided
    if args.document_single_prompt:
        prompt_manager.set_custom_prompt_path('document_single', args.document_single_prompt)
    if args.document_multi_prompt:
        prompt_manager.set_custom_prompt_path('document_multi', args.document_multi_prompt)
    if args.module_prompt:
        prompt_manager.set_custom_prompt_path('module', args.module_prompt)
    if args.synthesis_prompt:
        prompt_manager.set_custom_prompt_path('synthesis', args.synthesis_prompt)

    if args.command == 'interactive':
        # Launch interactive session
        session = InteractiveSession()
        session.model_config = model_config
        session.prompt_manager = prompt_manager
        session.cmdloop()
        return

    prototype = SecondBrainPrototype(prompt_manager=prompt_manager)

    # Load documents
    if args.documents:
        prototype.load_from_paths(args.documents, args.recursive)
    else:
        _load_default_documents(prototype)

    if not prototype.document_agents and not prototype.module_agents:
        error_msg = "Error: No document agents could be loaded. Please check the documents directory." #pylint: disable=line-too-long
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        sys.exit(1)

    # Execute commands
    if args.command == "query":
        _handle_query_command(prototype, args)
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
    elif args.command == "scale-test":
        run_scale_test(prototype)


if __name__ == "__main__":
    main()
