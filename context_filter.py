"""Advanced context filtering engine for hybrid chat architecture."""

import logging
from typing import List, Dict, Any
from conversation_manager import Message

class ContextFilter:
    """Intelligent context filtering for agent queries."""

    def __init__(self, max_tokens: int = 6000, relevance_threshold: float = 0.6):
        self.max_tokens = max_tokens
        self.relevance_threshold = relevance_threshold
        self.logger = logging.getLogger(__name__)

    def filter_by_relevance(self, full_context: List[Message], current_query: str) -> List[Message]:
        """Filter context by relevance to current query."""
        if not full_context:
            return []

        # Simple keyword-based relevance scoring
        query_words = set(current_query.lower().split())
        relevant_messages = []

        for message in full_context:
            message_words = set(message.content.lower().split())
            overlap = len(query_words & message_words)
            relevance_score = overlap / len(query_words) if query_words else 0

            if relevance_score >= self.relevance_threshold:
                relevant_messages.append(message)

        # Always include most recent message for continuity
        if (full_context and
            (not relevant_messages or relevant_messages[-1] != full_context[-1])):
            relevant_messages.append(full_context[-1])

        return self._compress_context(relevant_messages)

    def filter_by_recency(self, full_context: List[Message], window_size: int = 3) -> List[Message]:
        """Filter context by recency (keep N most recent exchanges)."""
        if not full_context:
            return []

        # Keep the most recent messages
        if len(full_context) >= window_size:
            recent_messages = full_context[-window_size:]
        else:
            recent_messages = full_context
        return self._compress_context(recent_messages)

    def filter_by_topic(self, full_context: List[Message],
                       topic_keywords: List[str]) -> List[Message]:
        """Filter context by topic similarity."""
        if not full_context or not topic_keywords:
            return self.filter_by_recency(full_context)

        topic_words = set(topic_keywords)
        topic_messages = []

        for message in full_context:
            message_words = set(message.content.lower().split())
            overlap = len(topic_words & message_words)
            if overlap > 0:  # Any topic overlap
                topic_messages.append(message)

        return self._compress_context(topic_messages)

    def _compress_context(self, messages: List[Message]) -> List[Message]:
        """Compress context to fit within token limits."""
        if not messages:
            return []

        # Simple compression: truncate long messages
        compressed = []
        total_tokens = 0

        for message in reversed(messages):  # Start from most recent
            message_tokens = len(message.content.split()) * 1.3  # Rough token estimation

            if total_tokens + message_tokens <= self.max_tokens:
                compressed.insert(0, message)
                total_tokens += message_tokens
            else:
                # Truncate message if it's too long
                max_words = int((self.max_tokens - total_tokens) / 1.3)
                if max_words > 50:  # Only truncate if we can keep substantial content
                    truncated_content = " ".join(message.content.split()[:max_words]) + "..."
                    truncated_message = Message(
                        role=message.role,
                        content=truncated_content,
                        timestamp=message.timestamp,
                        metadata=message.metadata
                    )
                    compressed.insert(0, truncated_message)
                break

        return compressed

    def build_hybrid_context(self, cm, query: str, max_tokens: int) -> List[Message]:  # pylint: disable=unused-argument
        """Build context using hybrid filtering strategy."""
        if not cm.conversation.messages:
            return []

        # Strategy 1: Try relevance-based filtering
        relevant_context = self.filter_by_relevance(cm.conversation.messages, query)

        # Strategy 2: If relevance filtering is too aggressive, fall back to recency
        context_too_small = len(relevant_context) < 2
        conversation_large_enough = len(cm.conversation.messages) > 2
        if context_too_small and conversation_large_enough:
            self.logger.info("Relevance filtering too aggressive, falling back to recency")
            relevant_context = self.filter_by_recency(cm.conversation.messages, 3)

        # Strategy 3: Ensure we have enough context for continuity
        if len(relevant_context) < 1 and cm.conversation.messages:
            relevant_context = [cm.conversation.messages[-1]]  # At least the last message

        # Compress to fit within token limits
        return self._compress_context(relevant_context)

    def get_context_statistics(self, original_context: List[Message],
                             filtered_context: List[Message]) -> Dict[str, Any]:
        """Get statistics about context filtering effectiveness."""
        if not original_context:
            return {}

        original_tokens = sum(len(m.content.split()) * 1.3 for m in original_context)
        filtered_tokens = sum(len(m.content.split()) * 1.3 for m in filtered_context)

        return {
            "original_messages": len(original_context),
            "filtered_messages": len(filtered_context),
            "original_tokens": original_tokens,
            "filtered_tokens": filtered_tokens,
            "reduction_ratio": ((original_tokens - filtered_tokens) / original_tokens
                               if original_tokens > 0 else 0),
            "compression_efficiency": (filtered_tokens / original_tokens
                                     if original_tokens > 0 else 1.0)
        }


# Backward compatibility function
def build_context(cm, query: str, max_tokens: int) -> List[Message]:
    """Simple recency + keyword overlap context builder (legacy function)."""
    filter_engine = ContextFilter(max_tokens=max_tokens)
    return filter_engine.build_hybrid_context(cm, query, max_tokens)

__all__ = ["build_context", "ContextFilter"]
