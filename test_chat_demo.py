#!/usr/bin/env python3
"""
Demo script for testing Second Brain chat mode.

This script demonstrates the basic chat functionality without requiring
full document loading or complex setup.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_manager import ConversationManager
from chat_interface import ChatInterface


class MockPrototype:
    """Mock prototype for testing chat interface."""

    def __init__(self):
        self.total_queries = 0
        self.total_cost = 0.0
        self.total_tokens = 0

    def query(self, question: str, use_cache: bool = True, namespace: Optional[str] = None):
        """Mock query method that returns a simple response."""
        self.total_queries += 1

        # Simple response logic based on the question
        if "hello" in question.lower():
            response = "Hello! I'm your Second Brain assistant. How can I help you today?"
        elif "python" in question.lower():
            response = "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, and automation."
        elif "help" in question.lower():
            response = "I can help you with various topics. Try asking me about programming, data science, or any other subject you're interested in!"
        else:
            response = f"I understand you're asking about: {question}. This is a demo response from the mock prototype."

        # Simulate some cost and token usage
        cost = 0.001 + (len(question) * 0.0001)
        tokens = len(question.split()) + len(response.split())

        self.total_cost += cost
        self.total_tokens += tokens

        return {
            "answer": response,
            "total_cost": cost,
            "total_tokens": tokens,
            "routing_decision": "mock",
            "modules_used": ["mock_module"],
            "model_used": "mock-model"
        }


def demo_conversation_manager():
    """Demo the ConversationManager functionality."""
    print("=" * 60)
    print("ConversationManager Demo")
    print("=" * 60)

    # Create a conversation manager
    manager = ConversationManager(persistence_dir=".demo_conversations")
    print(f"Created conversation: {manager.conversation_id}")

    # Add some messages
    print("\nAdding messages...")
    manager.add_message("user", "Hello, what is Python?")
    manager.add_message("assistant", "Python is a programming language.",
                       metadata={"tokens": 8, "cost": 0.002})
    manager.add_message("user", "Tell me more about its features.")
    manager.add_message("assistant", "Python features include: simple syntax, extensive libraries, cross-platform compatibility, and strong community support.",
                       metadata={"tokens": 20, "cost": 0.005})

    # Show statistics
    stats = manager.get_statistics()
    print(f"\nConversation Statistics:")
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Duration: {stats['duration_minutes']:.1f} minutes")

    # Test context window
    print(f"\nContext window (recent strategy, 100 tokens):")
    context = manager.get_context_window(max_tokens=100, strategy="recent")
    for i, msg in enumerate(context):
        print(f"  {i+1}. [{msg.role}] {msg.content[:50]}...")

    # Export conversation
    print(f"\nExporting conversation...")
    export_content = manager.export_to_text()
    print(f"Export length: {len(export_content)} characters")

    print("\nConversationManager demo completed!")


def demo_chat_interface():
    """Demo the ChatInterface functionality."""
    print("\n" + "=" * 60)
    print("ChatInterface Demo")
    print("=" * 60)

    # Create mock prototype
    prototype = MockPrototype()

    # Create chat interface
    interface = ChatInterface(
        prototype=prototype,
        max_context_tokens=1000,
        context_strategy="recent",
        show_cost_per_message=True,
        show_token_usage=False
    )

    print("Chat interface created. Starting demo conversation...")

    # Simulate a conversation
    demo_messages = [
        "Hello!",
        "What is Python?",
        "Tell me about its features",
        "How does it compare to other languages?",
        "/stats",
        "/exit"
    ]

    # Start chat and simulate input
    interface.start_chat()

    # Note: In a real scenario, the chat loop would handle user input
    # For this demo, we'll just show that the interface was created
    print("Chat interface demo completed!")
    print(f"Mock prototype processed {prototype.total_queries} queries")
    print(f"Total cost: ${prototype.total_cost:.4f}")
    print(f"Total tokens: {prototype.total_tokens}")


def demo_conversation_persistence():
    """Demo conversation persistence and loading."""
    print("\n" + "=" * 60)
    print("Conversation Persistence Demo")
    print("=" * 60)

    # Create a conversation and add messages
    conv_id = "demo_persistence_123"
    manager1 = ConversationManager(
        conversation_id=conv_id,
        persistence_dir=".demo_conversations"
    )

    print(f"Created conversation: {manager1.conversation_id}")

    # Add messages
    manager1.add_message("user", "This is a test message")
    manager1.add_message("assistant", "This is a test response")

    print(f"Added {len(manager1.conversation.messages)} messages")

    # Create a new manager instance and load the same conversation
    manager2 = ConversationManager(
        conversation_id=conv_id,
        persistence_dir=".demo_conversations"
    )

    print(f"Loaded conversation: {manager2.conversation_id}")
    print(f"Found {len(manager2.conversation.messages)} messages")

    # Verify the messages are the same
    if len(manager1.conversation.messages) == len(manager2.conversation.messages):
        print("✓ Persistence test passed!")
    else:
        print("✗ Persistence test failed!")

    print("Conversation persistence demo completed!")


def main():
    """Run all demos."""
    print("Second Brain Chat Mode Demo")
    print("This script demonstrates the chat mode functionality.")
    print()

    try:
        # Run demos
        demo_conversation_manager()
        demo_chat_interface()
        demo_conversation_persistence()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

        # Clean up demo files
        import shutil
        if Path(".demo_conversations").exists():
            shutil.rmtree(".demo_conversations")
            print("Cleaned up demo files.")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()