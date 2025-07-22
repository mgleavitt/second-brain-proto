"""Unit tests for Second Brain chat mode components."""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from conversation_manager import ConversationManager, Message, Conversation
from chat_interface import ChatInterface


class TestMessage(unittest.TestCase):
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        message = Message(
            role="user",
            content="Hello, world!",
            timestamp="2025-01-21T14:30:22",
            metadata={"tokens": 5, "cost": 0.001}
        )

        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello, world!")
        self.assertEqual(message.timestamp, "2025-01-21T14:30:22")
        self.assertEqual(message.metadata, {"tokens": 5, "cost": 0.001})

    def test_message_to_dict(self):
        """Test message serialization."""
        message = Message(
            role="assistant",
            content="Hello there!",
            timestamp="2025-01-21T14:30:22",
            metadata={"tokens": 8, "cost": 0.002}
        )

        data = message.to_dict()
        expected = {
            "role": "assistant",
            "content": "Hello there!",
            "timestamp": "2025-01-21T14:30:22",
            "metadata": {"tokens": 8, "cost": 0.002}
        }
        self.assertEqual(data, expected)

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "timestamp": "2025-01-21T14:30:22",
            "metadata": {"tokens": 12, "cost": 0.003}
        }

        message = Message.from_dict(data)
        self.assertEqual(message.role, "system")
        self.assertEqual(message.content, "You are a helpful assistant.")
        self.assertEqual(message.timestamp, "2025-01-21T14:30:22")
        self.assertEqual(message.metadata, {"tokens": 12, "cost": 0.003})


class TestConversation(unittest.TestCase):
    """Test Conversation dataclass."""

    def test_conversation_creation(self):
        """Test creating a conversation."""
        messages = [
            Message("user", "Hello", "2025-01-21T14:30:22"),
            Message("assistant", "Hi there!", "2025-01-21T14:30:25")
        ]

        conversation = Conversation(
            conversation_id="test_conv",
            created_at="2025-01-21T14:30:22",
            last_updated="2025-01-21T14:30:25",
            messages=messages,
            total_cost=0.005,
            total_tokens=15
        )

        self.assertEqual(conversation.conversation_id, "test_conv")
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.total_cost, 0.005)
        self.assertEqual(conversation.total_tokens, 15)

    def test_conversation_to_dict(self):
        """Test conversation serialization."""
        messages = [
            Message("user", "Hello", "2025-01-21T14:30:22"),
            Message("assistant", "Hi there!", "2025-01-21T14:30:25")
        ]

        conversation = Conversation(
            conversation_id="test_conv",
            created_at="2025-01-21T14:30:22",
            last_updated="2025-01-21T14:30:25",
            messages=messages,
            total_cost=0.005,
            total_tokens=15
        )

        data = conversation.to_dict()
        self.assertEqual(data["conversation_id"], "test_conv")
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["total_cost"], 0.005)
        self.assertEqual(data["total_tokens"], 15)

    def test_conversation_from_dict(self):
        """Test conversation deserialization."""
        data = {
            "conversation_id": "test_conv",
            "created_at": "2025-01-21T14:30:22",
            "last_updated": "2025-01-21T14:30:25",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2025-01-21T14:30:22",
                    "metadata": {}
                },
                {
                    "role": "assistant",
                    "content": "Hi there!",
                    "timestamp": "2025-01-21T14:30:25",
                    "metadata": {}
                }
            ],
            "total_cost": 0.005,
            "total_tokens": 15
        }

        conversation = Conversation.from_dict(data)
        self.assertEqual(conversation.conversation_id, "test_conv")
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.total_cost, 0.005)
        self.assertEqual(conversation.total_tokens, 15)


class TestConversationManager(unittest.TestCase):
    """Test ConversationManager class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_dir = Path(self.temp_dir) / "conversations"

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_new_conversation_creation(self):
        """Test creating a new conversation."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        self.assertIsNotNone(manager.conversation_id)
        self.assertTrue(manager.conversation_id.startswith("chat_"))
        self.assertEqual(len(manager.conversation.messages), 0)
        self.assertEqual(manager.conversation.total_cost, 0.0)
        self.assertEqual(manager.conversation.total_tokens, 0)

    def test_load_existing_conversation(self):
        """Test loading an existing conversation."""
        # Create a conversation file
        conv_id = "test_conv_123"
        conv_file = self.persistence_dir / f"{conv_id}.json"
        self.persistence_dir.mkdir(exist_ok=True)

        conversation_data = {
            "conversation_id": conv_id,
            "created_at": "2025-01-21T14:30:22",
            "last_updated": "2025-01-21T14:30:25",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2025-01-21T14:30:22",
                    "metadata": {}
                }
            ],
            "total_cost": 0.001,
            "total_tokens": 5
        }

        with open(conv_file, 'w') as f:
            json.dump(conversation_data, f)

        # Load the conversation
        manager = ConversationManager(
            conversation_id=conv_id,
            persistence_dir=str(self.persistence_dir)
        )

        self.assertEqual(manager.conversation_id, conv_id)
        self.assertEqual(len(manager.conversation.messages), 1)
        self.assertEqual(manager.conversation.total_cost, 0.001)
        self.assertEqual(manager.conversation.total_tokens, 5)

    def test_add_message(self):
        """Test adding messages to conversation."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Add user message
        user_msg = manager.add_message("user", "Hello, world!")
        self.assertEqual(user_msg.role, "user")
        self.assertEqual(user_msg.content, "Hello, world!")
        self.assertEqual(len(manager.conversation.messages), 1)

        # Add assistant message with metadata
        assistant_msg = manager.add_message(
            "assistant",
            "Hi there!",
            metadata={"tokens": 8, "cost": 0.002}
        )
        self.assertEqual(assistant_msg.role, "assistant")
        self.assertEqual(assistant_msg.content, "Hi there!")
        self.assertEqual(len(manager.conversation.messages), 2)
        self.assertEqual(manager.conversation.total_cost, 0.002)
        self.assertEqual(manager.conversation.total_tokens, 8)

    def test_add_message_validation(self):
        """Test message validation."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Test empty content
        with self.assertRaises(ValueError):
            manager.add_message("user", "")

        # Test invalid role
        with self.assertRaises(ValueError):
            manager.add_message("invalid_role", "Hello")

    def test_context_window_recent_strategy(self):
        """Test context window with recent strategy."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Add several messages
        for i in range(10):
            manager.add_message("user", f"Message {i}")
            manager.add_message("assistant", f"Response {i}")

        # Get context window (should return most recent messages)
        context = manager.get_context_window(max_tokens=100, strategy="recent")

        # Should return some messages (exact number depends on token counting)
        self.assertGreater(len(context), 0)
        self.assertLessEqual(len(context), 20)  # Max 20 messages

    def test_context_window_important_strategy(self):
        """Test context window with important strategy."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Add messages with different characteristics
        manager.add_message("system", "You are a helpful assistant.")
        manager.add_message("user", "Short message")
        manager.add_message("assistant", "Short response")
        manager.add_message("user", "This is a much longer message that should have higher importance score")
        manager.add_message("assistant", "Another short response")

        context = manager.get_context_window(max_tokens=200, strategy="important")

        # System message should be included (highest priority)
        system_messages = [msg for msg in context if msg.role == "system"]
        self.assertGreater(len(system_messages), 0)

    def test_get_statistics(self):
        """Test conversation statistics."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Add some messages
        manager.add_message("user", "Hello", metadata={"cost": 0.001, "tokens": 5})
        manager.add_message("assistant", "Hi!", metadata={"cost": 0.002, "tokens": 8})
        manager.add_message("user", "How are you?", metadata={"cost": 0.001, "tokens": 6})

        stats = manager.get_statistics()

        self.assertEqual(stats["total_messages"], 3)
        self.assertEqual(stats["user_messages"], 2)
        self.assertEqual(stats["assistant_messages"], 1)
        self.assertEqual(stats["total_cost"], 0.004)
        self.assertEqual(stats["total_tokens"], 19)
        self.assertGreater(stats["duration_minutes"], 0)

    def test_export_to_text(self):
        """Test conversation export to text."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Add messages
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")

        # Export to string
        content = manager.export_to_text()

        self.assertIn("Hello", content)
        self.assertIn("Hi there!", content)
        self.assertIn(manager.conversation_id, content)

    def test_clear_conversation(self):
        """Test clearing conversation."""
        manager = ConversationManager(persistence_dir=str(self.persistence_dir))

        # Add messages
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")

        self.assertEqual(len(manager.conversation.messages), 2)

        # Clear conversation
        manager.clear()

        self.assertEqual(len(manager.conversation.messages), 0)
        self.assertEqual(manager.conversation.total_cost, 0.0)
        self.assertEqual(manager.conversation.total_tokens, 0)

    def test_corrupt_file_handling(self):
        """Test handling of corrupt conversation files."""
        # Create a corrupt file
        conv_id = "corrupt_conv"
        conv_file = self.persistence_dir / f"{conv_id}.json"
        self.persistence_dir.mkdir(exist_ok=True)

        with open(conv_file, 'w') as f:
            f.write("invalid json content")

        # Should create new conversation instead of failing
        manager = ConversationManager(
            conversation_id=conv_id,
            persistence_dir=str(self.persistence_dir)
        )

        self.assertEqual(len(manager.conversation.messages), 0)
        self.assertNotEqual(manager.conversation_id, conv_id)


class TestChatInterface(unittest.TestCase):
    """Test ChatInterface class."""

    def setUp(self):
        """Set up test environment."""
        self.mock_prototype = Mock()
        self.mock_prototype.query.return_value = {
            "answer": "Test response",
            "total_cost": 0.001,
            "total_tokens": 10,
            "routing_decision": "synthesis",
            "modules_used": ["module1"],
            "model_used": "claude-3-sonnet"
        }

    def test_chat_interface_initialization(self):
        """Test ChatInterface initialization."""
        interface = ChatInterface(
            prototype=self.mock_prototype,
            max_context_tokens=8000,
            context_strategy="recent",
            show_cost_per_message=True,
            show_token_usage=False
        )

        self.assertEqual(interface.max_context_tokens, 8000)
        self.assertEqual(interface.context_strategy, "recent")
        self.assertTrue(interface.show_cost_per_message)
        self.assertFalse(interface.show_token_usage)
        self.assertFalse(interface.is_chat_active)

    @patch('builtins.input', side_effect=['/exit'])
    @patch('builtins.print')
    def test_start_chat_new_conversation(self, mock_print, mock_input):
        """Test starting a new chat conversation."""
        interface = ChatInterface(prototype=self.mock_prototype)

        interface.start_chat()

        # Should have created conversation manager
        self.assertIsNotNone(interface.conversation_manager)
        self.assertTrue(interface.conversation_manager.conversation_id.startswith("chat_"))

    @patch('builtins.input', side_effect=['/exit'])
    @patch('builtins.print')
    def test_start_chat_load_existing(self, mock_print, mock_input):
        """Test loading an existing conversation."""
        # Create a temporary conversation file
        temp_dir = tempfile.mkdtemp()
        try:
            conv_dir = Path(temp_dir) / "conversations"
            conv_dir.mkdir()

            conv_id = "test_conv_123"
            conv_file = conv_dir / f"{conv_id}.json"

            conversation_data = {
                "conversation_id": conv_id,
                "created_at": "2025-01-21T14:30:22",
                "last_updated": "2025-01-21T14:30:25",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2025-01-21T14:30:22",
                        "metadata": {}
                    }
                ],
                "total_cost": 0.001,
                "total_tokens": 5
            }

            with open(conv_file, 'w') as f:
                json.dump(conversation_data, f)

            interface = ChatInterface(prototype=self.mock_prototype)
            interface.start_chat(conversation_id=conv_id)

            self.assertEqual(interface.conversation_manager.conversation_id, conv_id)
            self.assertEqual(len(interface.conversation_manager.conversation.messages), 1)

        finally:
            shutil.rmtree(temp_dir)

    @patch('builtins.input', side_effect=['Hello', '/exit'])
    @patch('builtins.print')
    def test_process_message(self, mock_print, mock_input):
        """Test processing a user message."""
        interface = ChatInterface(prototype=self.mock_prototype)
        interface.start_chat()

        # The message should be processed in the chat loop
        # We can verify the prototype was called
        self.mock_prototype.query.assert_called()

    def test_build_context_prompt(self):
        """Test building context-aware prompts."""
        interface = ChatInterface(prototype=self.mock_prototype)
        interface.start_chat()

        # Add some conversation history
        interface.conversation_manager.add_message("user", "What is Python?")
        interface.conversation_manager.add_message("assistant", "Python is a programming language.")
        interface.conversation_manager.add_message("user", "Tell me more about it.")

        # Build context prompt
        context_prompt = interface._build_context_prompt("What are its features?")

        # Should include conversation history
        self.assertIn("What is Python?", context_prompt)
        self.assertIn("Python is a programming language.", context_prompt)
        self.assertIn("Tell me more about it.", context_prompt)
        self.assertIn("What are its features?", context_prompt)

    def test_count_tokens(self):
        """Test token counting."""
        interface = ChatInterface(prototype=self.mock_prototype)
        interface.start_chat()

        # Test with simple text
        token_count = interface._count_tokens("Hello, world!")
        self.assertGreater(token_count, 0)

    def test_get_conversation_id(self):
        """Test getting conversation ID."""
        interface = ChatInterface(prototype=self.mock_prototype)

        # Should return None when no conversation is active
        self.assertIsNone(interface.get_conversation_id())

        # Should return ID when conversation is active
        interface.start_chat()
        conv_id = interface.get_conversation_id()
        self.assertIsNotNone(conv_id)
        self.assertTrue(conv_id.startswith("chat_"))


if __name__ == '__main__':
    unittest.main()