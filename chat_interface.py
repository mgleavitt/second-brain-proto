"""Chat interface for Second Brain system.

This module provides ChatInterface for managing chat mode interactions,
context building, and integration with the existing SecondBrainPrototype.
"""

import logging
from typing import Dict, Optional, Any, Tuple
from colorama import Fore, Style

from chat_controller import ChatController
from light_chat_model import get_lightweight_chat_call
from complexity_analyzer import classify_query
from context_filter import build_context
from conversation_manager import ConversationManager
# Removed circular import - using type hints with string literals


class ChatInterface:
    """Provides dedicated chat interaction interface."""

    def __init__(self, prototype: "SecondBrainPrototype",
                 max_context_tokens: int = 8000,
                 context_strategy: str = "recent",
                 show_cost_per_message: bool = True,
                 show_token_usage: bool = False):
        """Initialize chat interface.

        Args:
            prototype: Reference to SecondBrainPrototype instance
            max_context_tokens: Maximum tokens for context window
            context_strategy: Context selection strategy
            show_cost_per_message: Whether to show cost after each message
            show_token_usage: Whether to show token usage after each message
        """
        self.prototype = prototype
        self.max_context_tokens = max_context_tokens
        self.context_strategy = context_strategy
        self.show_cost_per_message = show_cost_per_message
        self.show_token_usage = show_token_usage

        self.conversation_manager: Optional[ConversationManager] = None
        self.is_chat_active = False
        self.controller = None  # Will be initialized after conversation manager is created

        self.logger = logging.getLogger(__name__)

        # Initialize attributes that were defined outside __init__
        self._last_result_cache = None
        self._last_result_namespace = None

    def start_chat(self, conversation_id: Optional[str] = None) -> None:
        """Start chat mode with optional conversation ID.

        Args:
            conversation_id: ID of existing conversation to load, or None for new
        """
        try:
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                conversation_id=conversation_id,
                persistence_dir=".conversations"
            )

            # Initialize ChatController with the conversation manager
            self.controller = ChatController(
                conversation_manager=self.conversation_manager,
                prototype=self.prototype,
                light_model_call=get_lightweight_chat_call(),
                classify_query=classify_query,
                build_context=build_context,
                max_ctx_tokens=self.max_context_tokens
            )

            self.is_chat_active = True

            # Display welcome message
            self._display_welcome_message()

            # Enter chat loop
            self._chat_loop()

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to start chat: %s", e)
            print(f"{Fore.RED}Error starting chat: {e}{Style.RESET_ALL}")
            self.is_chat_active = False

    def _display_welcome_message(self) -> None:
        """Display welcome message for chat mode."""
        conversation_id = self.conversation_manager.conversation_id
        stats = self.conversation_manager.get_statistics()

        print(f"\n{Fore.CYAN}{'='*60}")
        print("Second Brain Chat Mode")
        print(f"Conversation: {conversation_id}")
        print(f"{'='*60}{Style.RESET_ALL}")

        if stats["total_messages"] > 0:
            print(f"{Fore.YELLOW}Loaded existing conversation with "
                  f"{stats['total_messages']} messages")
            print(f"Total cost: ${stats['total_cost']:.4f} | "
                  f"Duration: {stats['duration_minutes']:.1f} minutes{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Starting new conversation{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}Chat Commands:{Style.RESET_ALL}")
        print("  /help          - Show this help")
        print("  /stats         - Show conversation statistics")
        print("  /export [file] - Export conversation to text file")
        print("  /clear         - Clear current conversation")
        print("  /exit          - Exit chat mode")
        print(f"\n{Fore.CYAN}Type your message or command:{Style.RESET_ALL}\n")

    def _chat_loop(self) -> None:
        """Main chat interaction loop."""
        while self.is_chat_active:
            try:
                # Get user input with better terminal handling
                print(f"{Fore.GREEN}You:{Style.RESET_ALL} ", end='', flush=True)
                user_input = input().strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Process regular message
                self._process_message(user_input)

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n{Fore.YELLOW}End of input{Style.RESET_ALL}")
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error("Error in chat loop: %s", e)
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    def _handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns True if chat should continue, False to exit."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._display_help()
            return True
        if cmd == "/stats":
            self._display_statistics()
            return True
        if cmd == "/routing":
            self._display_routing_statistics()
            return True
        if cmd == "/export":
            self._export_conversation(arg)
            return True
        if cmd == "/clear":
            self._clear_conversation()
            return True
        if cmd == "/exit":
            self._exit_chat()
            return False

        print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}")
        print("Type /help for available commands")
        return True

    def _display_help(self) -> None:
        """Display help information."""
        print(f"\n{Fore.CYAN}Chat Mode Help:{Style.RESET_ALL}")
        print("Type your message normally to chat with the AI.")
        print("Use commands starting with / for special actions:")
        print()
        print("  /help          - Show this help")
        print("  /stats         - Show conversation statistics")
        print("  /routing       - Show routing statistics")
        print("  /export [file] - Export conversation to text file")
        print("  /clear         - Clear current conversation")
        print("  /exit          - Exit chat mode")
        print()
        print("The AI will remember your conversation context and")
        print("can reference previous messages in its responses.")

    def _display_statistics(self) -> None:
        """Display conversation statistics."""
        if not self.conversation_manager:
            print(f"{Fore.RED}No active conversation{Style.RESET_ALL}")
            return

        stats = self.conversation_manager.get_statistics()

        print(f"\n{Fore.CYAN}Conversation Statistics:{Style.RESET_ALL}")
        print(f"  Conversation ID: {self.conversation_manager.conversation_id}")
        print(f"  Total Messages: {stats['total_messages']}")
        print(f"  User Messages: {stats['user_messages']}")
        print(f"  Assistant Messages: {stats['assistant_messages']}")
        print(f"  System Messages: {stats['system_messages']}")
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        print(f"  Total Cost: ${stats['total_cost']:.4f}")
        print(f"  Duration: {stats['duration_minutes']:.1f} minutes")

        if stats['total_messages'] > 0:
            avg_cost = stats['total_cost'] / stats['total_messages']
            print(f"  Average Cost per Message: ${avg_cost:.4f}")

    def _display_routing_statistics(self) -> None:
        """Display routing statistics from the ChatController."""
        if not self.controller:
            print(f"{Fore.RED}No controller available{Style.RESET_ALL}")
            return

        stats = self.controller.get_routing_statistics()
        if not stats:
            print(f"{Fore.YELLOW}No routing statistics available yet{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Routing Statistics:{Style.RESET_ALL}")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Lightweight Queries: {stats['lightweight_queries']} "
              f"({stats['lightweight_percentage']:.1f}%)")
        print(f"  Agent Queries: {stats['agent_queries']}")
        print(f"  Total Lightweight Cost: ${stats['total_lightweight_cost']:.4f}")
        print(f"  Total Agent Cost: ${stats['total_agent_cost']:.4f}")
        print(f"  Total Cost: ${stats['total_cost']:.4f}")
        print(f"  Average Cost per Query: ${stats['average_cost_per_query']:.4f}")

        # Show cost savings if we have both types
        if stats['lightweight_queries'] > 0 and stats['agent_queries'] > 0:
            avg_lightweight_cost = (stats['total_lightweight_cost'] /
                                   stats['lightweight_queries'])
            avg_agent_cost = stats['total_agent_cost'] / stats['agent_queries']
            cost_savings = (avg_agent_cost - avg_lightweight_cost) * stats['lightweight_queries']
            print(f"  Estimated Cost Savings: ${cost_savings:.4f}")

    def _export_conversation(self, file_path: str) -> None:
        """Export conversation to text file."""
        if not self.conversation_manager:
            print(f"{Fore.RED}No active conversation{Style.RESET_ALL}")
            return

        try:
            if not file_path:
                file_path = f"{self.conversation_manager.conversation_id}_export.txt"

            self.conversation_manager.export_to_text(file_path)
            print(f"{Fore.GREEN}Conversation exported to: {file_path}{Style.RESET_ALL}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to export conversation: %s", e)
            print(f"{Fore.RED}Failed to export conversation: {e}{Style.RESET_ALL}")

    def _clear_conversation(self) -> None:
        """Clear the current conversation."""
        if not self.conversation_manager:
            print(f"{Fore.RED}No active conversation{Style.RESET_ALL}")
            return

        response = input(f"{Fore.YELLOW}Are you sure you want to clear this "
                        f"conversation? (y/N): {Style.RESET_ALL}")
        if response.lower() in ['y', 'yes']:
            self.conversation_manager.clear()
            print(f"{Fore.GREEN}Conversation cleared{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}Conversation not cleared{Style.RESET_ALL}")

    def _exit_chat(self) -> None:
        """Exit chat mode."""
        if self.conversation_manager:
            stats = self.conversation_manager.get_statistics()
            print(f"\n{Fore.CYAN}Chat session summary:{Style.RESET_ALL}")
            print(f"  Messages: {stats['total_messages']}")
            print(f"  Cost: ${stats['total_cost']:.4f}")
            print(f"  Duration: {stats['duration_minutes']:.1f} minutes")

        print(f"{Fore.GREEN}Exiting chat mode{Style.RESET_ALL}")
        self.is_chat_active = False

    def _process_message(self, user_input: str) -> None:
        """Process a user message and generate response."""
        if not self.conversation_manager:
            print(f"{Fore.RED}No active conversation{Style.RESET_ALL}")
            return

        if not self.controller:
            print(f"{Fore.RED}Chat controller not initialized{Style.RESET_ALL}")
            return

        try:
            # Route via enhanced ChatController
            result = self.controller.process_message(user_input)

            # Extract response and metadata
            response_text = result["response"]
            meta = {
                "routing": result["route"],
                "category": result["category"],
                "confidence": result["confidence"],
                "cost": result["cost"],
                "context_size": result["context_size"],
                "reasoning": result["reasoning"]
            }

            # Add assistant message to conversation
            self.conversation_manager.add_message("assistant", response_text, metadata=meta)

            # Display response with enhanced metadata
            self._display_response(response_text, meta)

            # Show routing information if enabled
            if self.show_cost_per_message:
                self._display_routing_info(result)

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Error processing message: %s", e)
            print(f"{Fore.RED}Error processing message: {e}{Style.RESET_ALL}")

    def _display_routing_info(self, result: Dict[str, Any]) -> None:
        """Display routing decision information."""
        route = result["route"]
        category = result["category"]
        confidence = result["confidence"]
        cost = result["cost"]
        reasoning = result["reasoning"]

        # Color coding for routes
        route_color = Fore.GREEN if route == "lightweight" else Fore.YELLOW
        category_color = Fore.CYAN

        print(f"\n{Fore.BLUE}Routing Info:{Style.RESET_ALL}")
        print(f"  Route: {route_color}{route}{Style.RESET_ALL}")
        print(f"  Category: {category_color}{category}{Style.RESET_ALL}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Cost: ${cost:.4f}")
        if reasoning:
            print(f"  Reasoning: {reasoning}")
        print()

    def _check_followup_question(self, user_input: str) -> Tuple[bool, Optional[Dict]]:
        """Check if this is a follow-up question that can use cached results."""
        if not hasattr(self, '_last_result_cache'):
            return False, None

        # Follow-up indicators for expansion/clarification
        expansion_indicators = [
            "can you explain", "what do you mean", "tell me more", "expand on",
            "elaborate", "give me more details", "can you clarify", "what about",
            "give an example", "show me", "describe", "define", "what is",
            "how does", "why does", "when would", "where is", "which one"
        ]

        # Follow-up indicators for referencing previous content
        reference_indicators = [
            "the first", "the second", "the third", "this", "that", "it",
            "they", "them", "these", "those", "above", "below", "mentioned",
            "said", "discussed", "talked about", "covered"
        ]

        input_lower = user_input.lower()

        # Check if input contains follow-up indicators
        is_expansion = any(indicator in input_lower for indicator in expansion_indicators)
        is_reference = any(indicator in input_lower for indicator in reference_indicators)

        # Also check if it's a very short question (likely a follow-up)
        is_short = len(user_input.split()) <= 5

        # Check if it's asking about the same topic as the last response
        is_same_topic = self._is_same_topic(user_input)

        if is_expansion or is_reference or is_short or is_same_topic:
            return True, self._last_result_cache

        return False, None

    def _is_same_topic(self, user_input: str) -> bool:
        """Check if the user input is asking about the same topic as the last response."""
        if not hasattr(self, '_last_result_cache') or not self._last_result_cache:
            return False

        # Get the last assistant response
        if not self.conversation_manager:
            return False

        messages = self.conversation_manager.conversation.messages
        if len(messages) < 2:
            return False

        # Get the last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg.role == "assistant":
                last_assistant_msg = msg
                break

        if not last_assistant_msg:
            return False

        # Extract key terms from the last response and current input
        last_response_lower = last_assistant_msg.content.lower()
        current_input_lower = user_input.lower()

        # Simple keyword matching - if they share key terms, likely same topic
        last_words = set(last_response_lower.split())
        current_words = set(current_input_lower.split())

        # Filter out common words
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "may", "might", "this", "that",
            "these", "those", "it", "its", "they", "them", "their"
        }

        last_keywords = last_words - common_words
        current_keywords = current_words - common_words

        # If they share significant keywords, likely same topic
        shared_keywords = last_keywords & current_keywords
        return len(shared_keywords) >= 2  # At least 2 shared keywords

    def _cache_last_result(self, result: Dict, namespace: str) -> None:
        """Cache the last result for potential follow-up questions."""
        self._last_result_cache = result
        self._last_result_namespace = namespace

    def _build_context_prompt(self, current_query: str) -> str:
        """Build context-aware prompt from conversation history."""
        if not self.conversation_manager:
            return current_query

        # Get context window
        context_messages = self.conversation_manager.get_context_window(
            max_tokens=self.max_context_tokens,
            strategy=self.context_strategy
        )

        if not context_messages:
            return current_query

        # Build context prompt
        context_parts = []

        # Add system message if present
        system_messages = [msg for msg in context_messages if msg.role == "system"]
        if system_messages:
            context_parts.append("System context:")
            for msg in system_messages:
                context_parts.append(f"  {msg.content}")
            context_parts.append("")

        # Add conversation history
        conversation_messages = [msg for msg in context_messages if msg.role != "system"]
        if conversation_messages:
            context_parts.append("Previous conversation:")
            for msg in conversation_messages:
                role_display = "You" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_display}: {msg.content}")
            context_parts.append("")

        # Add current query
        context_parts.append(f"Current question: {current_query}")

        # Check if we're approaching token limit
        context_text = "\n".join(context_parts)
        context_tokens = self._count_tokens(context_text)

        if context_tokens > self.max_context_tokens * 0.9:
            self.logger.warning("Context approaching token limit: %d tokens", context_tokens)
            print(f"{Fore.YELLOW}Warning: Conversation context is getting long{Style.RESET_ALL}")

        return context_text

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using conversation manager's tokenizer."""
        if self.conversation_manager and self.conversation_manager.tokenizer:
            return len(self.conversation_manager.tokenizer.encode(text))

        # Fallback estimation
        return len(text) // 4

    def _display_response(self, response: str, metadata: Dict[str, Any]) -> None:
        """Display assistant response with optional metadata."""
        print(f"\n{Fore.BLUE}Assistant:{Style.RESET_ALL} {response}\n")

        # Show metadata if enabled
        if self.show_cost_per_message:
            cost = metadata.get("cost", 0.0)
            print(f"{Fore.YELLOW}Cost: ${cost:.4f}{Style.RESET_ALL}")

        if self.show_token_usage:
            tokens = metadata.get("tokens", 0)
            print(f"{Fore.YELLOW}Tokens: {tokens:,}{Style.RESET_ALL}")

        # Show routing info in debug mode
        if self.logger.isEnabledFor(logging.DEBUG):
            routing = metadata.get("routing_decision", "unknown")
            modules = metadata.get("modules_used", [])
            model = metadata.get("model_used", "unknown")
            print(f"{Fore.CYAN}Debug: {routing} -> {modules} (via {model}){Style.RESET_ALL}")

    def get_conversation_id(self) -> Optional[str]:
        """Get current conversation ID."""
        if self.conversation_manager:
            return self.conversation_manager.conversation_id
        return None

    def is_active(self) -> bool:
        """Check if chat is currently active."""
        return self.is_chat_active
