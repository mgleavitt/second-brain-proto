"""Conversation management for Second Brain chat mode.

This module provides ConversationManager for managing conversation state,
persistence, and context windows in the chat mode.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import tiktoken


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata")
        )


@dataclass
class Conversation:
    """Represents a complete conversation."""
    conversation_id: str
    created_at: str
    last_updated: str
    messages: List[Message]
    total_cost: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "messages": [msg.to_dict() for msg in self.messages],
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            messages=[Message.from_dict(msg) for msg in data["messages"]],
            total_cost=data.get("total_cost", 0.0),
            total_tokens=data.get("total_tokens", 0)
        )


class ConversationManager:
    """Manages conversation state, persistence, and context windows."""

    def __init__(self, conversation_id: Optional[str] = None,
                 persistence_dir: str = ".conversations"):
        """Initialize conversation manager.

        Args:
            conversation_id: ID for the conversation. If None, generates new ID.
            persistence_dir: Directory to store conversation files.
        """
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        except ImportError:
            self.logger.warning("tiktoken not available, using fallback token counting")
            self.tokenizer = None

        if conversation_id:
            self.conversation_id = conversation_id
            self.conversation = self._load_conversation()
        else:
            self.conversation_id = self._generate_conversation_id()
            self.conversation = self._create_new_conversation()

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"chat_{timestamp}"

    def _create_new_conversation(self) -> Conversation:
        """Create a new conversation."""
        now = datetime.now().isoformat()
        return Conversation(
            conversation_id=self.conversation_id,
            created_at=now,
            last_updated=now,
            messages=[]
        )

    def _get_conversation_file_path(self) -> Path:
        """Get the file path for the current conversation."""
        return self.persistence_dir / f"{self.conversation_id}.json"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

    def add_message(self, role: str, content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the conversation.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional metadata (tokens, cost, model, etc.)

        Returns:
            The created message object
        """
        if not content.strip():
            raise ValueError("Message content cannot be empty")

        if role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {role}. Must be user, assistant, or system")

        # Create message
        timestamp = datetime.now().isoformat()
        message = Message(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata or {}
        )

        # Add to conversation
        self.conversation.messages.append(message)
        self.conversation.last_updated = timestamp

        # Update statistics
        if metadata:
            self.conversation.total_cost += metadata.get("cost", 0.0)
            self.conversation.total_tokens += metadata.get("tokens", 0)

        # Persist immediately
        self._save_conversation()

        self.logger.info("Added %s message to conversation %s", role, self.conversation_id)
        return message

    def get_context_window(self, max_tokens: int = 8000,
                          strategy: str = "recent") -> List[Message]:
        """Get messages within the context window.

        Args:
            max_tokens: Maximum tokens allowed in context
            strategy: Context selection strategy ("recent", "important", "hybrid")

        Returns:
            List of messages within the context window
        """
        if not self.conversation.messages:
            return []

        if strategy == "recent":
            return self._get_recent_context(max_tokens)
        elif strategy == "important":
            return self._get_important_context(max_tokens)
        elif strategy == "hybrid":
            return self._get_hybrid_context(max_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_recent_context(self, max_tokens: int) -> List[Message]:
        """Get most recent messages within token limit."""
        selected_messages = []
        current_tokens = 0

        # Start from most recent messages
        for message in reversed(self.conversation.messages):
            message_tokens = self._count_tokens(message.content)

            if current_tokens + message_tokens <= max_tokens:
                selected_messages.insert(0, message)  # Insert at beginning to maintain order
                current_tokens += message_tokens
            else:
                break

        return selected_messages

    def _get_important_context(self, max_tokens: int) -> List[Message]:
        """Get most important messages within token limit.

        For now, this is a simple implementation that prioritizes
        system messages and longer user messages.
        """
        # Score messages by importance
        scored_messages = []
        for message in self.conversation.messages:
            score = 0
            if message.role == "system":
                score += 1000  # System messages are very important
            elif message.role == "user":
                score += len(message.content)  # Longer messages get higher score
            else:
                score += len(message.content) // 2  # Assistant messages get lower score

            scored_messages.append((score, message))

        # Sort by score (descending)
        scored_messages.sort(key=lambda x: x[0], reverse=True)

        # Select messages within token limit
        selected_messages = []
        current_tokens = 0

        for _, message in scored_messages:
            message_tokens = self._count_tokens(message.content)

            if current_tokens + message_tokens <= max_tokens:
                selected_messages.append(message)
                current_tokens += message_tokens
            else:
                break

        # Sort by timestamp to maintain chronological order
        selected_messages.sort(key=lambda x: x.timestamp)
        return selected_messages

    def _get_hybrid_context(self, max_tokens: int) -> List[Message]:
        """Get hybrid context: recent messages + important older messages."""
        # Get recent context (70% of tokens)
        recent_tokens = int(max_tokens * 0.7)
        recent_messages = self._get_recent_context(recent_tokens)

        # Get important context from remaining messages (30% of tokens)
        remaining_tokens = max_tokens - sum(self._count_tokens(msg.content)
                                                        for msg in recent_messages)

        # Get messages not in recent context
        recent_ids = {id(msg) for msg in recent_messages}
        older_messages = [msg for msg in self.conversation.messages if id(msg) not in recent_ids]

        # Score and select important older messages
        scored_older = []
        for message in older_messages:
            score = 0
            if message.role == "system":
                score += 1000
            elif message.role == "user":
                score += len(message.content)
            else:
                score += len(message.content) // 2
            scored_older.append((score, message))

        scored_older.sort(key=lambda x: x[0], reverse=True)

        important_messages = []
        current_tokens = 0

        for _, message in scored_older:
            message_tokens = self._count_tokens(message.content)
            if current_tokens + message_tokens <= remaining_tokens:
                important_messages.append(message)
                current_tokens += message_tokens
            else:
                break

        # Combine and sort by timestamp
        all_messages = recent_messages + important_messages
        all_messages.sort(key=lambda x: x.timestamp)
        return all_messages

    def _save_conversation(self) -> None:
        """Save conversation to disk."""
        try:
            # Create temporary file for atomic write
            temp_path = self._get_conversation_file_path().with_suffix('.tmp')

            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation.to_dict(), f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.rename(self._get_conversation_file_path())

            self.logger.debug("Saved conversation %s", self.conversation_id)

        except (OSError, TypeError, ValueError) as e:
            self.logger.error("Failed to save conversation %s: %s", self.conversation_id, e)
            raise

    def _load_conversation(self) -> Conversation:
        """Load conversation from disk."""
        file_path = self._get_conversation_file_path()

        if not file_path.exists():
            self.logger.info("Conversation file not found, creating new conversation")
            return self._create_new_conversation()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            conversation = Conversation.from_dict(data)
            self.logger.info("Loaded conversation %s with %d messages",
                           self.conversation_id, len(conversation.messages))
            return conversation

        except (OSError, json.JSONDecodeError, KeyError) as e:
            self.logger.error("Failed to load conversation %s: %s", self.conversation_id, e)
            # Create backup of corrupted file
            backup_path = file_path.with_suffix('.corrupted')
            try:
                file_path.rename(backup_path)
                self.logger.info("Backed up corrupted conversation to %s", backup_path)
            except OSError:
                pass

            # Return new conversation
            return self._create_new_conversation()

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not self.conversation.messages:
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "duration_minutes": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0
            }

        # Calculate duration
        start_time = datetime.fromisoformat(self.conversation.created_at)
        end_time = datetime.fromisoformat(self.conversation.last_updated)
        duration = (end_time - start_time).total_seconds() / 60

        # Count messages by role
        role_counts = {}
        for message in self.conversation.messages:
            role_counts[message.role] = role_counts.get(message.role, 0) + 1

        return {
            "total_messages": len(self.conversation.messages),
            "total_tokens": self.conversation.total_tokens,
            "total_cost": self.conversation.total_cost,
            "duration_minutes": round(duration, 2),
            "user_messages": role_counts.get("user", 0),
            "assistant_messages": role_counts.get("assistant", 0),
            "system_messages": role_counts.get("system", 0)
        }

    def export_to_text(self, file_path: Optional[str] = None) -> str:
        """Export conversation to plain text format."""
        if not file_path:
            file_path = f"{self.conversation_id}_export.txt"

        lines = [
            f"Conversation: {self.conversation_id}",
            f"Created: {self.conversation.created_at}",
            f"Last Updated: {self.conversation.last_updated}",
            f"Total Messages: {len(self.conversation.messages)}",
            f"Total Cost: ${self.conversation.total_cost:.4f}",
            f"Total Tokens: {self.conversation.total_tokens}",
            "",
            "=" * 60,
            ""
        ]

        for message in self.conversation.messages:
            timestamp = datetime.fromisoformat(message.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            lines.extend([
                f"[{timestamp}] {message.role.upper()}:",
                message.content,
                ""
            ])

        content = "\n".join(lines)

        # Write to file if path provided
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info("Exported conversation to %s", file_path)
            except OSError as e:
                self.logger.error("Failed to export conversation: %s", e)
                raise

        return content

    def clear(self) -> None:
        """Clear the current conversation."""
        self.conversation = self._create_new_conversation()
        self._save_conversation()
        self.logger.info("Cleared conversation %s", self.conversation_id)

    def delete(self) -> None:
        """Delete the conversation file."""
        try:
            file_path = self._get_conversation_file_path()
            if file_path.exists():
                file_path.unlink()
                self.logger.info("Deleted conversation file %s", file_path)
        except OSError as e:
            self.logger.error("Failed to delete conversation file: %s", e)
            raise
