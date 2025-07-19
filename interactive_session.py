"""Interactive command-line interface for the second brain system."""
import cmd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
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

init(autoreset=True)  # Initialize colorama


class InteractiveSession(cmd.Cmd):  # pylint: disable=too-many-instance-attributes
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

    def do_query(self, arg):  # pylint: disable=too-many-locals
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

        # Check cache
        if self.use_cache and not no_cache:
            cached_result = self.cache.get(question)
            if cached_result:
                print(f"{Fore.YELLOW}[Using cached result]{Style.RESET_ALL}")
                print(cached_result['answer'])
                return

        print(f"{Fore.YELLOW}Processing query...{Style.RESET_ALL}")

        try:
            start_time = datetime.now()

            # Route or query all
            if self.use_routing and not no_routing and self.routing_agent:
                selected_modules = self.routing_agent.route_query(question)
                print(f"{Fore.CYAN}Routing to modules: {selected_modules}{Style.RESET_ALL}")
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

            # Cache result
            if self.use_cache:
                self.cache.set(question, {
                    'answer': final_answer,
                    'cost': total_cost,
                    'time': elapsed_time
                })

            # Log query
            self.query_logger.log_query(question, final_answer, total_cost, elapsed_time)

            # Display result
            print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
            print(final_answer)
            cost_time_msg = f"Cost: ${total_cost:.4f} | Time: {elapsed_time:.2f}s"
            print(f"\n{Fore.CYAN}{cost_time_msg}{Style.RESET_ALL}")

        except (ValueError, AttributeError, RuntimeError) as e:
            print(f"{Fore.RED}Error processing query: {e}{Style.RESET_ALL}")
            self.logger.exception("Error in query command")

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
            remainder = line[len(cmd_name)+1:].strip()
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
