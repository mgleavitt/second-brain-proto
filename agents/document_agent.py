"""DocumentAgent: Handles loading, chunking, and querying of individual documents and synthesis
across documents.

This module provides the DocumentAgent and SynthesisAgent classes for document-level and
multi-document LLM-based querying.
"""

from typing import Dict, Any, List, Optional
from prompt_manager import PromptManager
from model_config import ModelConfig
from .base_agent import BaseAgent

DEFAULT_DOCUMENT_MODEL = "gemini-1.5-flash"
DEFAULT_SYNTHESIS_MODEL = "claude-3-opus-20240229"
COST_ESTIMATES = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


class DocumentAgent(BaseAgent):  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Represents an agent responsible for a single document or multiple documents."""
    def __init__(self, documents: List[Dict[str, str]],
                 model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """Initialize the DocumentAgent with documents, model config, and prompt manager."""
        super().__init__(model_config, prompt_manager, "document")
        self.documents = documents

    def query(self, question: str) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """Query the documents for an answer to the given question."""
        # Create context from all documents
        context_parts = []
        for doc in self.documents:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                source = (doc.metadata.get('source', 'Unknown')
                         if hasattr(doc, 'metadata') else 'Unknown')
            else:
                content = doc.get('content', '')
                source = doc.get('source', 'Unknown')

            if question.lower() in content.lower():
                context_parts.append(f"From {source}\n---\n{content}")

        if not context_parts:
            return {
                "answer": "No relevant information found in the loaded documents.",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0
            }

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > 8000:
            context = context[:8000] + "\n\n[Context truncated for length...]"

        # Use PromptManager for document prompt
        system_prompt = self.prompt_manager.get_prompt("document_single")
        prompt = self._create_standard_prompt(system_prompt, context, question)

        return self._invoke_llm_with_tracking(prompt)




class SynthesisAgent(BaseAgent):  # pylint: disable=too-few-public-methods
    """Synthesizes responses from multiple document agents."""
    def __init__(self, model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """Initialize the SynthesisAgent with model config and prompt manager."""
        super().__init__(model_config, prompt_manager, "synthesis")

    def synthesize(self, question: str, agent_responses: List[Dict]) -> Dict[str, Any]:
        """Synthesize responses from multiple agents into a coherent answer."""
        if not agent_responses:
            return {
                "response": "No responses to synthesize.",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "sources": []
            }

        if len(agent_responses) == 1:
            # Single response, return as-is
            response = agent_responses[0]
            return {
                "response": response.get("answer", response.get("response", "No answer provided")),
                "tokens_used": response.get("tokens_used", 0),
                "cost": response.get("cost", 0.0),
                "duration": response.get("duration", 0.0),
                "sources": [response.get("agent_name", "Agent 1")]
            }

        # Format responses for synthesis
        formatted_responses = []
        for i, response in enumerate(agent_responses, 1):
            answer = response.get("answer", response.get("response", "No answer provided"))
            agent_name = response.get("agent_name", f"Agent {i}")
            formatted_responses.append(f"Response from {agent_name}:\n{answer}")

        prompt = self._create_synthesis_prompt(question, formatted_responses)

        result = self._invoke_llm_with_tracking(prompt)
        return {
            "response": result["answer"],
            "tokens_used": result["tokens_used"],
            "cost": result["cost"],
            "duration": result["duration"],
            "sources": [f"Agent {i+1}" for i in range(len(agent_responses))]
        }

    def _create_synthesis_prompt(self, question: str, formatted_responses: List[str]) -> str:
        """Create a prompt for synthesizing multiple agent responses."""
        system_prompt = self.prompt_manager.get_prompt("synthesis")
        responses_text = "\n\n---\n\n".join(formatted_responses)

        synthesis_instruction = (
            "Please synthesize these responses into a comprehensive, coherent answer. "
            "Address any contradictions, highlight key points, and provide a unified "
            "response that best answers the original question."
        )

        return f"""{system_prompt}

Question: {question}

Responses from different sources:
---
{responses_text}
---

{synthesis_instruction}"""
