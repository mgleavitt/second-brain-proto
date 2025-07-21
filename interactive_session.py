"""Interactive command-line interface for the second brain system."""
import cmd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from colorama import init, Fore, Style
from tabulate import tabulate
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from model_config import ModelConfig
from prompt_manager import PromptManager
from cache_manager import SimpleCache
from query_logger import QueryLogger
from agents.document_agent import DocumentAgent, SynthesisAgent
from agents.module_agent import ModuleAgent
from agents.routing_agent import RoutingAgent
from loaders.document_loader import DocumentLoader
from summarizer import ModuleSummarizer
from evaluation_framework import QueryEvaluator

# Conditional imports for optional features
try:
    from semantic_cache import SemanticCache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False

try:
    from embedding_router import EmbeddingRouter
    EMBEDDING_ROUTER_AVAILABLE = True
except ImportError:
    EMBEDDING_ROUTER_AVAILABLE = False

init(autoreset=True)  # Initialize colorama


class InteractiveSession(cmd.Cmd):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Interactive command-line interface for the second brain system."""

    intro = f"""
{Fore.CYAN}{'='*60}
Welcome to Second Brain Interactive Mode
Type /help for available commands
{'='*60}{Style.RESET_ALL}
"""
    prompt = f"{Fore.GREEN}second-brain>{Style.RESET_ALL} "

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig()
        self.prompt_manager = PromptManager()
        self.cache = SimpleCache()
        self.query_logger = QueryLogger()

        # Agent storage
        self.document_agents: Dict[str, DocumentAgent] = {}
        self.module_agents: Dict[str, ModuleAgent] = {}
        self.synthesis_agent: Optional[SynthesisAgent] = None
        self.routing_agent: Optional[RoutingAgent] = None

        # Settings
        self.use_routing = True
        self.use_cache = True
        self.loaded_paths: List[str] = []

        # Optional features
        self.summarizer: Optional[ModuleSummarizer] = None
        self.evaluator: Optional[QueryEvaluator] = None
        self.semantic_cache: Optional[Any] = None
        self.embedding_router: Optional[Any] = None
        self.use_embedding_routing = False

        # Configure logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def do_help(self, arg):
        """Show available commands."""
        if arg:
            # Show help for specific command
            cmd.Cmd.do_help(self, arg)
        else:
            print(f"\n{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
            commands = [
                ("/help [command]", "Show help for all commands or a specific command"),
                ("/load <path> [--recursive]", "Load documents from path"),
                ("/query <question>", "Query loaded documents"),
                ("/model <agent_type> <model>", "Set model for agent type"),
                ("/costs", "Show cost summary"),
                ("/clear-cache", "Clear query cache"),
                ("/rebuild", "Reload all documents"),
                ("/save <filename>", "Save session state"),
                ("/load-session <filename>", "Load session state"),
                ("/status", "Show system status"),
                ("/settings", "Show current settings"),
                ("/toggle-routing", "Toggle query routing on/off"),
                ("/toggle-cache", "Toggle caching on/off"),
                ("/prompt-reload", "Reload all system prompts"),
                ("/prompt-set <type> <file>", "Set custom prompt file"),
                ("/prompt-show <type>", "Show current prompt"),
                ("/prompt-create-defaults", "Create default prompt files"),
                ("/summarize [--force]", "Generate module summaries"),
                ("/evaluate <question>", "Evaluate query strategies"),
                ("/semantic-cache [stats|clear-expired]", "Manage semantic cache"),
                ("/use-embeddings [on|off]", "Toggle embedding-based routing"),
                ("/exit", "Exit interactive mode")
            ]

            for cmd_syntax, description in commands:
                print(f"  {Fore.GREEN}{cmd_syntax:<30}{Style.RESET_ALL} {description}")
            print()

    def do_load(self, arg):
        """Load documents from a path. Usage: /load <path> [--recursive]"""
        parts = arg.split()
        if not parts:
            print(f"{Fore.RED}Error: Please provide a path to load{Style.RESET_ALL}")
            return

        path = parts[0]
        recursive = "--recursive" in parts

        if not os.path.exists(path):
            print(f"{Fore.RED}Error: Path does not exist: {path}{Style.RESET_ALL}")
            return

        print(f"{Fore.YELLOW}Loading documents from {path}...{Style.RESET_ALL}")

        try:
            # Load documents based on path type
            if os.path.isfile(path):
                self._load_file(path)
            else:
                self._load_directory(path, recursive)

            self.loaded_paths.append(path)
            print(f"{Fore.GREEN}Successfully loaded documents from {path}{Style.RESET_ALL}")
            self._update_routing_agent()

        except (OSError, ValueError, RuntimeError) as e:
            print(f"{Fore.RED}Error loading documents: {e}{Style.RESET_ALL}")
            self.logger.exception("Error in load command")

    def do_query(self, arg):
        """Query the loaded documents. Usage: /query <question> [--no-cache] [--no-routing]"""
        if not arg:
            print(f"{Fore.RED}Error: Please provide a question{Style.RESET_ALL}")
            return

        # Parse arguments
        no_cache = "--no-cache" in arg
        no_routing = "--no-routing" in arg

        # Extract question (remove flags)
        question = (arg.replace("--no-cache", "")
                      .replace("--no-routing", "")
                      .strip())

        if not self.module_agents and not self.document_agents:
            print(f"{Fore.RED}Error: No documents loaded. Use /load first{Style.RESET_ALL}")
            return

        # Check caches first
        if self._check_caches(question, no_cache):
            return

        print(f"{Fore.YELLOW}Processing query...{Style.RESET_ALL}")

        try:
            result = self._process_query(question, no_routing)
            self._display_query_result(result)
        except (ValueError, AttributeError, RuntimeError) as e:
            print(f"{Fore.RED}Error processing query: {e}{Style.RESET_ALL}")
            self.logger.exception("Error in query command")

    def _check_caches(self, question: str, no_cache: bool) -> bool:
        """Check semantic and regular caches for existing results."""
        # Check semantic cache first
        if (self.use_cache and not no_cache and
            self.semantic_cache is not None):
            cached_result = self.semantic_cache.get(question)
            if cached_result:
                print(f"{Fore.YELLOW}[Using semantic cache]{Style.RESET_ALL}")
                print(cached_result['answer'])
                return True

        # Check regular cache
        if self.use_cache and not no_cache:
            cached_result = self.cache.get(question)
            if cached_result:
                print(f"{Fore.YELLOW}[Using cached result]{Style.RESET_ALL}")
                print(cached_result['answer'])
                return True

        return False

    def _process_query(self, question: str, no_routing: bool) -> Dict[str, Any]:
        """Process the query and return results."""
        start_time = datetime.now()

        # Route to modules
        selected_modules = self._route_query(question, no_routing)

        # Query selected modules
        responses, total_cost = self._query_modules(selected_modules, question)

        # Synthesize responses
        final_answer, synthesis_cost = self._synthesize_responses(question, responses)
        total_cost += synthesis_cost

        # Calculate time
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Cache results
        self._cache_query_result(question, final_answer, total_cost, elapsed_time)

        # Log query
        self.query_logger.log_query(question, final_answer, total_cost, elapsed_time)

        return {
            'answer': final_answer,
            'cost': total_cost,
            'time': elapsed_time
        }

    def _route_query(self, question: str, no_routing: bool) -> List[str]:
        """Route query to appropriate modules."""
        if self.use_routing and not no_routing and self.use_embedding_routing:
            if self.embedding_router is not None:
                routed_modules = self.embedding_router.route_query(question)
                # Filter to only include modules that actually exist
                selected_modules = [m[0] for m in routed_modules
                                  if m[0] in self.module_agents]
                if not selected_modules:
                    # Fall back to all available modules if routing fails
                    selected_modules = list(self.module_agents.keys())
                print(f"{Fore.CYAN}Embedding routing to: {selected_modules}{Style.RESET_ALL}")
            else:
                # Fall back to keyword routing
                selected_modules = self.routing_agent.route_query(question)
                print(f"{Fore.CYAN}Keyword routing to: {selected_modules}{Style.RESET_ALL}")
        elif self.use_routing and not no_routing and self.routing_agent:
            selected_modules = self.routing_agent.route_query(question)
            print(f"{Fore.CYAN}Routing to modules: {selected_modules}{Style.RESET_ALL}")
        else:
            selected_modules = list(self.module_agents.keys())

        return selected_modules

    def _query_modules(self, selected_modules: List[str], question: str) -> tuple:
        """Query selected modules and return responses and total cost."""
        responses = {}
        total_cost = 0.0

        for module_name in selected_modules:
            if module_name in self.module_agents:
                response = self.module_agents[module_name].query(question)
                responses[module_name] = response
                total_cost += response.get('cost', 0)

        return responses, total_cost

    def _synthesize_responses(self, question: str, responses: Dict[str, Any]) -> tuple:
        """Synthesize multiple responses into a single answer."""
        if len(responses) > 1 and self.synthesis_agent:
            synthesis_result = self.synthesis_agent.synthesize(question, responses)
            final_answer = synthesis_result['synthesis']
            synthesis_cost = synthesis_result.get('cost', 0)
        else:
            # Single response
            final_answer = next(iter(responses.values()))['answer']
            synthesis_cost = 0

        return final_answer, synthesis_cost

    def _cache_query_result(self, question: str, answer: str, cost: float, time: float):
        """Cache query results in both semantic and regular caches."""
        result_data = {
            'answer': answer,
            'cost': cost,
            'time': time
        }

        # Cache in semantic cache
        if self.use_cache and self.semantic_cache is not None:
            self.semantic_cache.set(question, result_data)

        # Cache in regular cache
        if self.use_cache:
            self.cache.set(question, result_data)

    def _display_query_result(self, result: Dict[str, Any]):
        """Display query results to the user."""
        print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
        print(result['answer'])
        cost_time_msg = f"Cost: ${result['cost']:.4f} | Time: {result['time']:.2f}s"
        print(f"\n{Fore.CYAN}{cost_time_msg}{Style.RESET_ALL}")

    def query(self, question: str) -> Dict[str, Any]:
        """Query method for evaluation framework compatibility."""
        if not self.module_agents and not self.document_agents:
            return {
                'answer': 'No documents loaded',
                'cost': 0.0,
                'modules': []
            }

        try:
            start_time = datetime.now()

            # Route using embeddings if enabled
            if self.use_routing and getattr(self, 'use_embedding_routing', False):
                if hasattr(self, 'embedding_router'):
                    routed_modules = self.embedding_router.route_query(question)
                    # Filter to only include modules that actually exist
                    selected_modules = [m[0] for m in routed_modules if m[0] in self.module_agents]
                    if not selected_modules:
                        # Fall back to all available modules if routing fails
                        selected_modules = list(self.module_agents.keys())
                else:
                    # Fall back to keyword routing
                    selected_modules = self.routing_agent.route_query(question)
            elif self.use_routing and self.routing_agent:
                selected_modules = self.routing_agent.route_query(question)
            else:
                selected_modules = list(self.module_agents.keys())

            # Query selected modules
            responses = {}
            total_cost = 0.0

            for module_name in selected_modules:
                if module_name in self.module_agents:
                    response = self.module_agents[module_name].query(question)
                    responses[module_name] = response
                    total_cost += response.get('cost', 0)

            # Synthesize if multiple responses
            if len(responses) > 1 and self.synthesis_agent:
                synthesis_result = self.synthesis_agent.synthesize(question, responses)
                final_answer = synthesis_result['synthesis']
                total_cost += synthesis_result.get('cost', 0)
            else:
                # Single response
                final_answer = next(iter(responses.values()))['answer']

            # Calculate time
            elapsed_time = (datetime.now() - start_time).total_seconds()

            return {
                'answer': final_answer,
                'cost': total_cost,
                'modules': selected_modules,
                'time': elapsed_time
            }

        except (ValueError, AttributeError, RuntimeError, OSError) as e:
            return {
                'answer': f'Error processing query: {e}',
                'cost': 0.0,
                'modules': [],
                'time': 0.0
            }

    def do_model(self, arg):
        """Set model for an agent type. Usage: /model <agent_type> <model_name>"""
        parts = arg.split()
        if len(parts) != 2:
            print(f"{Fore.RED}Usage: /model <agent_type> <model_name>{Style.RESET_ALL}")
            print(f"Agent types: {list(self.model_config.DEFAULT_MODELS.keys())}")
            print(f"Models: {list(self.model_config.AVAILABLE_MODELS.keys())}")
            return

        agent_type, model_name = parts

        try:
            self.model_config.set_model(agent_type, model_name)
            print(f"{Fore.GREEN}Set {agent_type} agent to use {model_name}{Style.RESET_ALL}")

            # Reinitialize affected agents
            self._reinitialize_agents()

        except ValueError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    def do_costs(self, _arg):
        """Show cost summary for all queries."""
        summary = self.query_logger.get_summary()

        if not summary['total_queries']:
            print(f"{Fore.YELLOW}No queries logged yet{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Query Cost Summary:{Style.RESET_ALL}")
        print(f"Total queries: {summary['total_queries']}")
        print(f"Total cost: ${summary['total_cost']:.4f}")
        print(f"Average cost: ${summary['average_cost']:.4f}")
        print(f"Total time: {summary['total_time']:.2f}s")

        # Show model configuration costs
        print(f"\n{Fore.CYAN}Current Model Configuration:{Style.RESET_ALL}")
        model_summary = self.model_config.get_summary()

        table_data = []
        for agent_type, info in model_summary.items():
            table_data.append([
                agent_type,
                info['display_name'],
                f"${info['input_cost_per_1m']:.2f}",
                f"${info['output_cost_per_1m']:.2f}"
            ])

        print(tabulate(table_data,
                      headers=["Agent Type", "Model", "Input $/1M", "Output $/1M"],
                      tablefmt="grid"))

    def do_status(self, _arg):
        """Show system status."""
        print(f"\n{Fore.CYAN}System Status:{Style.RESET_ALL}")
        doc_count = sum(len(agent.documents) for agent in self.document_agents.values())
        print(f"Documents loaded: {doc_count}")
        print(f"Modules loaded: {len(self.module_agents)}")
        print(f"Routing: {'Enabled' if self.use_routing else 'Disabled'}")
        print(f"Caching: {'Enabled' if self.use_cache else 'Disabled'}")
        print(f"Cache entries: {len(self.cache.cache)}")

        if self.loaded_paths:
            print(f"\n{Fore.CYAN}Loaded paths:{Style.RESET_ALL}")
            for path in self.loaded_paths:
                print(f"  - {path}")

    def do_save(self, arg):
        """Save session state. Usage: /save <filename>"""
        if not arg:
            print(f"{Fore.RED}Error: Please provide a filename{Style.RESET_ALL}")
            return

        try:
            session_data = {
                'loaded_paths': self.loaded_paths,
                'model_config': self.model_config.agent_models,
                'settings': {
                    'use_routing': self.use_routing,
                    'use_cache': self.use_cache
                },
                'cache': self.cache.cache if self.use_cache else {},
                'timestamp': datetime.now().isoformat()
            }

            with open(arg, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)

            print(f"{Fore.GREEN}Session saved to {arg}{Style.RESET_ALL}")

        except (OSError, ValueError, RuntimeError) as e:
            print(f"{Fore.RED}Error saving session: {e}{Style.RESET_ALL}")

    def do_exit(self, _arg):
        """Exit interactive mode."""
        print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
        return True

    def do_clear_cache(self, _arg):
        """Clear the query cache."""
        self.cache.clear()
        print(f"{Fore.GREEN}Cache cleared{Style.RESET_ALL}")

    def do_rebuild(self, _arg):
        """Reload all documents from loaded paths."""
        if not self.loaded_paths:
            print(f"{Fore.YELLOW}No paths to reload{Style.RESET_ALL}")
            return

        print(f"{Fore.YELLOW}Rebuilding document index...{Style.RESET_ALL}")

        # Clear existing agents
        self.document_agents.clear()
        self.module_agents.clear()

        # Reload all paths
        for path in self.loaded_paths:
            try:
                if os.path.isfile(path):
                    self._load_file(path)
                else:
                    self._load_directory(path, recursive=True)
            except (OSError, ValueError, RuntimeError) as e:
                print(f"{Fore.RED}Error reloading {path}: {e}{Style.RESET_ALL}")

        self._update_routing_agent()
        print(f"{Fore.GREEN}Rebuild complete{Style.RESET_ALL}")

    def do_settings(self, _arg):
        """Show current settings."""
        print(f"\n{Fore.CYAN}Current Settings:{Style.RESET_ALL}")
        settings = [
            ("Routing", "Enabled" if self.use_routing else "Disabled"),
            ("Caching", "Enabled" if self.use_cache else "Disabled"),
            ("Prompt Directory", self.prompt_manager.prompt_dir),
        ]

        for name, value in settings:
            print(f"{name}: {value}")

    def do_toggle_routing(self, _arg):
        """Toggle query routing on/off."""
        self.use_routing = not self.use_routing
        status = "Enabled" if self.use_routing else "Disabled"
        print(f"{Fore.GREEN}Routing {status}{Style.RESET_ALL}")

    def do_toggle_cache(self, _arg):
        """Toggle caching on/off."""
        self.use_cache = not self.use_cache
        status = "Enabled" if self.use_cache else "Disabled"
        print(f"{Fore.GREEN}Caching {status}{Style.RESET_ALL}")

    def do_summarize(self, arg):
        """Generate or regenerate module summaries. Usage: /summarize [--force]"""
        force = "--force" in arg

        if self.summarizer is None:
            self.summarizer = ModuleSummarizer(self.model_config, self.prompt_manager)

        print(f"{Fore.YELLOW}Generating module summaries...{Style.RESET_ALL}")

        for module_name, module_agent in self.module_agents.items():
            summary = self.summarizer.get_or_generate_summary(
                module_name,
                module_agent.documents,
                force_regenerate=force
            )
            print(f"{Fore.GREEN}✓ {module_name}: {len(summary.key_topics)} topics, "
                  f"{summary.document_count} documents{Style.RESET_ALL}")

        # Update routing with summaries
        if self.routing_agent:
            self.routing_agent.set_module_summaries(self.summarizer.summaries)

        print(f"{Fore.GREEN}Summary generation complete{Style.RESET_ALL}")

    def do_evaluate(self, arg):
        """Evaluate query strategies. Usage: /evaluate <question>"""
        if not arg:
            print(f"{Fore.RED}Error: Please provide a question to evaluate{Style.RESET_ALL}")
            return

        if self.evaluator is None:
            self.evaluator = QueryEvaluator(self.model_config)

        print(f"{Fore.YELLOW}Evaluating query strategies...{Style.RESET_ALL}")

        results = self.evaluator.compare_strategies(arg, self)

        # Display results
        comparison = results.get('comparison', {})
        print(f"\n{Fore.CYAN}Evaluation Results:{Style.RESET_ALL}")
        print(f"Cost savings: ${comparison.get('cost_savings', 0):.4f} "
              f"({comparison.get('cost_savings_percentage', 0):.1f}%)")

        quality_diff = comparison.get('quality_difference', {})
        print(f"Quality impact: {quality_diff.get('overall_diff', 0):+.3f}")

        print(f"\n{Fore.CYAN}Detailed Scores:{Style.RESET_ALL}")
        for strategy in ['with_routing', 'without_routing']:
            if strategy in results:
                metrics = results[strategy]['metrics']
                print(f"\n{strategy}:")
                print(f"  Relevance: {metrics.quality.relevance_score:.3f}")
                print(f"  Completeness: {metrics.quality.completeness_score:.3f}")
                print(f"  Coherence: {metrics.quality.coherence_score:.3f}")
                print(f"  Cost: ${results[strategy]['cost']:.4f}")

    def do_semantic_cache(self, arg):
        """Manage semantic cache. Usage: /semantic-cache [stats|clear-expired]"""
        if self.semantic_cache is None and SEMANTIC_CACHE_AVAILABLE:
            self.semantic_cache = SemanticCache()

        if arg == "stats":
            stats = self.semantic_cache.get_stats()
            print(f"\n{Fore.CYAN}Semantic Cache Statistics:{Style.RESET_ALL}")
            for key, value in stats.items():
                print(f"{key}: {value}")

        elif arg == "clear-expired":
            self.semantic_cache.clear_expired()
            print(f"{Fore.GREEN}Cleared expired entries{Style.RESET_ALL}")

        else:
            print("Usage: /semantic-cache [stats|clear-expired]")

    def do_use_embeddings(self, arg):
        """Toggle embedding-based routing. Usage: /use-embeddings [on|off]"""
        if arg.lower() == "on":
            if self.embedding_router is None and EMBEDDING_ROUTER_AVAILABLE:
                self.embedding_router = EmbeddingRouter()
                self._setup_embedding_routing()

            self.use_embedding_routing = True

        elif arg.lower() == "off":
            self.use_embedding_routing = False
            print(f"{Fore.GREEN}Switched back to keyword-based routing{Style.RESET_ALL}")

        else:
            status = "on" if self.use_embedding_routing else "off"
            print(f"Embedding routing is currently: {status}")
            print("Usage: /use-embeddings [on|off]")

    def _setup_embedding_routing(self):
        """Setup embedding routing with existing summaries."""
        if (self.summarizer is not None and
            self.summarizer.summaries):
            # Filter summaries to only include modules that actually exist
            available_modules = set(self.module_agents.keys())
            filtered_summaries = {
                name: summary for name, summary in self.summarizer.summaries.items()
                if name in available_modules
            }

            if filtered_summaries:
                self.embedding_router.index_modules(filtered_summaries)
                print(f"{Fore.GREEN}Embedding-based routing enabled for "
                      f"{len(filtered_summaries)} modules{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}No summaries available for loaded modules. "
                      f"Run /summarize first.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Run /summarize first to generate module summaries"
                  f"{Style.RESET_ALL}")

    def do_load_session(self, arg):
        """Load session state. Usage: /load-session <filename>"""
        if not arg:
            print(f"{Fore.RED}Error: Please provide a filename{Style.RESET_ALL}")
            return

        try:
            with open(arg, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # Restore settings
            self.use_routing = session_data['settings']['use_routing']
            self.use_cache = session_data['settings']['use_cache']

            # Restore model configuration
            for agent_type, model_name in session_data['model_config'].items():
                self.model_config.set_model(agent_type, model_name)

            # Restore cache if enabled
            if self.use_cache and 'cache' in session_data:
                self.cache.cache = session_data['cache']

            # Reload documents
            self.loaded_paths = session_data['loaded_paths']
            self.do_rebuild('')

            print(f"{Fore.GREEN}Session loaded from {arg}{Style.RESET_ALL}")

        except (OSError, ValueError, RuntimeError) as e:
            print(f"{Fore.RED}Error loading session: {e}{Style.RESET_ALL}")

    def _load_file(self, filepath: str):
        """Load a single file as a document."""
        loader = DocumentLoader()
        documents = loader.load_path(filepath)
        if documents:
            # Convert tuples to dicts for DocumentAgent
            doc_dicts = [{'name': name, 'content': content, 'source': filepath}
                        for name, content in documents]
            agent = DocumentAgent(
                documents=doc_dicts,
                model_config=self.model_config,
                prompt_manager=self.prompt_manager
            )
            module_name = Path(filepath).stem
            module_agent = ModuleAgent(
                documents=doc_dicts,
                model_config=self.model_config,
                prompt_manager=self.prompt_manager
            )
            self.document_agents[filepath] = agent
            self.module_agents[module_name] = module_agent
            print(f"  Loaded {len(doc_dicts)} documents from {filepath}")

    def _load_directory(self, dirpath: str, recursive: bool):
        """Load all documents from a directory."""
        loader = DocumentLoader()
        documents = loader.load_path(dirpath, recursive=recursive)
        if documents:
            # Group documents by subdirectory
            doc_groups = {}
            for name, content in documents:
                group_name = Path(dirpath).name
                if group_name not in doc_groups:
                    doc_groups[group_name] = []
                doc_dict = {'name': name, 'content': content, 'source': dirpath}
                doc_groups[group_name].append(doc_dict)
            for group_name, group_docs in doc_groups.items():
                if group_docs:
                    agent = DocumentAgent(
                        documents=group_docs,
                        model_config=self.model_config,
                        prompt_manager=self.prompt_manager
                    )
                    module_agent = ModuleAgent(
                        documents=group_docs,
                        model_config=self.model_config,
                        prompt_manager=self.prompt_manager
                    )
                    agent_path = f"{dirpath}/{group_name}"
                    self.document_agents[agent_path] = agent
                    self.module_agents[group_name] = module_agent
                    print(f"  Loaded {len(group_docs)} documents in module '{group_name}'")

    def _update_routing_agent(self):
        """Update routing agent with current modules."""
        if self.module_agents:
            self.routing_agent = RoutingAgent(
                list(self.module_agents.keys()),
                model_config=self.model_config,
                prompt_manager=self.prompt_manager
            )

            # Create synthesis agent if we have multiple modules
            if len(self.module_agents) > 1:
                self.synthesis_agent = SynthesisAgent(
                    model_config=self.model_config,
                    prompt_manager=self.prompt_manager
                )

    def _reinitialize_agents(self):
        """Reinitialize agents with new model configuration."""
        # Reinitialize all agents with updated model config
        for agent in self.document_agents.values():
            agent.model_config = self.model_config
            agent.model_name = self.model_config.get_model_name("document")
            # Reinitialize LLM based on provider
            model_info = self.model_config.get_model_info("document")
            if model_info.provider == "google":
                agent.llm = ChatGoogleGenerativeAI(model=agent.model_name)
            else:
                agent.llm = ChatAnthropic(model=agent.model_name)

        for agent in self.module_agents.values():
            agent.model_config = self.model_config
            agent.model_name = self.model_config.get_model_name("module")
            # Reinitialize LLM based on provider
            model_info = self.model_config.get_model_info("module")
            if model_info.provider == "google":
                agent.llm = ChatGoogleGenerativeAI(model=agent.model_name)
            else:
                agent.llm = ChatAnthropic(model=agent.model_name)

        if self.routing_agent:
            self.routing_agent.model_config = self.model_config
            self.routing_agent.model_name = self.model_config.get_model_name("routing")
            # Reinitialize LLM based on provider
            model_info = self.model_config.get_model_info("routing")
            if model_info.provider == "google":
                self.routing_agent.llm = ChatGoogleGenerativeAI(model=self.routing_agent.model_name)
            else:
                self.routing_agent.llm = ChatAnthropic(model=self.routing_agent.model_name)

        if self.synthesis_agent:
            self.synthesis_agent.model_config = self.model_config
            self.synthesis_agent.model_name = self.model_config.get_model_name("synthesis")
            # Reinitialize LLM based on provider
            model_info = self.model_config.get_model_info("synthesis")
            if model_info.provider == "google":
                self.synthesis_agent.llm = ChatGoogleGenerativeAI(
                    model=self.synthesis_agent.model_name)
            else:
                self.synthesis_agent.llm = ChatAnthropic(model=self.synthesis_agent.model_name)

    # Override cmd methods for better UX
    def default(self, line):
        """Handle commands that don't start with /"""
        if line.startswith('/'):
            cmd_name = line.split()[0][1:]  # Remove the /
            remainder = line[len(cmd_name)+2:].strip()  # +2 to account for / and space
            if hasattr(self, f'do_{cmd_name}'):
                func = getattr(self, f'do_{cmd_name}')
                if callable(func):
                    func(remainder)
                    return
            print(f"{Fore.RED}Unknown command: {line.split()[0]}{Style.RESET_ALL}")
        else:
            # Treat as a query if no / prefix
            self.do_query(line)

    def emptyline(self):
        """Don't repeat last command on empty line."""
        return
