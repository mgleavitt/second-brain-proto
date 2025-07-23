"""Classify query into routing categories using Claude‑3‑Haiku."""
import json
import logging
import os
from typing import Tuple, Dict, Any

import anthropic

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Enhanced classification categories
LIGHTWEIGHT_CATEGORIES = {
    "clarification",      # "What did you mean by X?"
    "elaboration",        # "Tell me more about Y"
    "confirmation",       # "So you're saying that..."
    "simple_followup",    # "And what about Z?"
    "conversation_meta",  # "Can you summarize what we discussed?"
}

AGENT_REQUIRED_CATEGORIES = {
    "new_complex_topic",  # "Explain the relationship between A and B"
    "document_specific",  # "What does document X say about Y?"
    "synthesis_needed",   # "Compare perspectives across modules"
    "factual_lookup",     # "What is the exact formula for..."
    "deep_analysis",      # "Analyze the implications of..."
}

HYBRID_CATEGORIES = {
    "partial_context",    # Some context needed, but not all
    "topic_shift",        # Changing subjects but building on previous
    "complex_followup",   # Followup requiring document access
}

SYSTEM_PROMPT = """You are a query complexity analyzer for a hybrid chat system.
Analyze the user's query and classify it into one of the following categories:

LIGHTWEIGHT CATEGORIES (handle with cheap model):
- clarification: "What did you mean by X?", "I don't understand Y"
- elaboration: "Tell me more about Y", "Can you expand on that?"
- confirmation: "So you're saying that...", "Is that correct?"
- simple_followup: "And what about Z?", "What else?"
- conversation_meta: "Can you summarize?", "What did we discuss?"

AGENT-REQUIRED CATEGORIES (need full agent pipeline):
- new_complex_topic: "Explain the relationship between A and B"
- document_specific: "What does document X say about Y?"
- synthesis_needed: "Compare perspectives across modules"
- factual_lookup: "What is the exact formula for..."
- deep_analysis: "Analyze the implications of..."

HYBRID CATEGORIES (may need some context):
- partial_context: Some context needed, but not all
- topic_shift: Changing subjects but building on previous
- complex_followup: Followup requiring document access

Return JSON: {"category": "category_name", "confidence": 0.0-1.0}
Confidence should reflect how certain you are about the classification."""

def classify_query(query: str, cm) -> Tuple[str, float]:
    """Classify query complexity and determine routing strategy."""
    try:
        # Get conversation context for better classification
        context = _build_classification_context(cm, query)

        resp = _client.messages.create(
            model="claude-3-haiku-20240307",
            system=SYSTEM_PROMPT,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": context}],
        )

        # Clean the response text to handle extra data
        response_text = resp.content[0].text.strip()

        # Try to extract JSON from the response
        try:
            # Look for JSON object in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                data = json.loads(json_text)
            else:
                # Fallback: try to parse the entire response
                data = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract category from text
            logging.warning("Failed to parse JSON from response: %s", response_text)
            if "category" in response_text.lower():
                # Try to extract category from text
                if any(cat in response_text.lower() for cat in LIGHTWEIGHT_CATEGORIES):
                    category = "clarification"  # Default lightweight category
                elif any(cat in response_text.lower() for cat in AGENT_REQUIRED_CATEGORIES):
                    category = "document_specific"  # Default agent category
                else:
                    category = "unknown"
                confidence = 0.5  # Default confidence
            else:
                category = "unknown"
                confidence = 0.0
            return category, confidence

        category = data.get("category", "unknown")
        confidence = float(data.get("confidence", 0.0))

        # Validate category
        all_categories = (LIGHTWEIGHT_CATEGORIES |
                         AGENT_REQUIRED_CATEGORIES |
                         HYBRID_CATEGORIES)

        if category not in all_categories:
            logging.warning("Unknown category '%s', defaulting to 'unknown'", category)
            category = "unknown"
            confidence = 0.0

        return category, confidence

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error in query classification: %s", e)
        return "unknown", 0.0

def _build_classification_context(cm, query: str) -> str:
    """Build context for classification by analyzing conversation history."""
    if not cm.conversation.messages:
        return f"Current query: {query}\nContext: New conversation"

    # Get recent messages for context
    recent_messages = (cm.conversation.messages[-3:]
                      if len(cm.conversation.messages) >= 3
                      else cm.conversation.messages)

    context_parts = []
    for msg in recent_messages:
        if msg.role == "user":
            context_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            context_parts.append(f"Assistant: {msg.content[:200]}...")

    context = "\n".join(context_parts)

    return f"""Previous conversation:
{context}

Current query: {query}

Classify the current query based on the conversation context."""

def get_category_info(category: str) -> Dict[str, Any]:
    """Get information about a classification category."""
    if category in LIGHTWEIGHT_CATEGORIES:
        return {
            "type": "lightweight",
            "description": "Can be handled by cheap conversation model",
            "examples": _get_category_examples(category)
        }
    if category in AGENT_REQUIRED_CATEGORIES:
        return {
            "type": "agent_required",
            "description": "Requires full agent pipeline with document access",
            "examples": _get_category_examples(category)
        }
    if category in HYBRID_CATEGORIES:
        return {
            "type": "hybrid",
            "description": "May need some context but not full agent pipeline",
            "examples": _get_category_examples(category)
        }

    return {
        "type": "unknown",
        "description": "Unable to classify",
        "examples": []
    }

def _get_category_examples(category: str) -> list:
    """Get example queries for a category."""
    examples = {
        "clarification": [
            "What did you mean by neural networks?",
            "I don't understand the difference between X and Y",
            "Could you clarify that last point?"
        ],
        "elaboration": [
            "Tell me more about that concept",
            "Can you expand on the implications?",
            "What are the details of this approach?"
        ],
        "confirmation": [
            "So you're saying that X equals Y?",
            "Is that correct?",
            "Let me make sure I understand..."
        ],
        "simple_followup": [
            "And what about Z?",
            "What else should I know?",
            "Any other considerations?"
        ],
        "conversation_meta": [
            "Can you summarize what we discussed?",
            "What are the key points so far?",
            "Where were we in the conversation?"
        ],
        "new_complex_topic": [
            "Explain the relationship between A and B",
            "How do these concepts interact?",
            "What's the connection between X and Y?"
        ],
        "document_specific": [
            "What does the database textbook say about normalization?",
            "Find information about X in the course materials",
            "What's in the lecture notes about Y?"
        ],
        "synthesis_needed": [
            "Compare the perspectives across different modules",
            "How do these concepts relate to each other?",
            "Synthesize the information from multiple sources"
        ],
        "factual_lookup": [
            "What is the exact formula for calculating X?",
            "What are the specific steps in the algorithm?",
            "What's the precise definition of Y?"
        ],
        "deep_analysis": [
            "Analyze the implications of this approach",
            "What are the trade-offs involved?",
            "How does this affect the overall system?"
        ]
    }

    return examples.get(category, [])

__all__ = ["classify_query", "get_category_info"]
