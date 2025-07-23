"""DocumentAgent: Handles loading, chunking, and querying of individual documents and synthesis
across documents.

This module provides the DocumentAgent and SynthesisAgent classes for document-level and
multi-document LLM-based querying.
"""

from typing import Dict, Any, List, Optional
import hashlib
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

        # Chunk all loaded documents (same as ModuleAgent)
        self.chunks = self._load_and_chunk_documents()

    @property
    def name(self) -> str:
        """Get the name of this document agent based on its documents."""
        if not self.documents:
            return "Empty Document Agent"

        first_doc = self.documents[0]
        if isinstance(first_doc, dict):
            return first_doc.get('name', 'Document Agent')
        else:
            return 'Document Agent'

    def _load_and_chunk_documents(self) -> List[Dict]:
        """Load and chunk all documents using semantic chunking (same as ModuleAgent)."""
        chunks = []
        for doc in self.documents:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                doc_name = (doc.metadata.get('source', 'Unknown')
                           if hasattr(doc, 'metadata') else 'Unknown')
            else:
                content = doc.get('content', '')
                doc_name = doc.get('name', 'Unknown')

            doc_chunks = self._semantic_chunk(content, doc_name)
            chunks.extend(doc_chunks)
        return chunks

    def _semantic_chunk(self, content: str, doc_name: str,
                       chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Create semantic chunks preserving context (same as ModuleAgent)."""
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_size = 0
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'doc_name': doc_name,
                    'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                })
                if overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-2:]
                    current_size = sum(len(p) for p in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            current_chunk.append(para)
            current_size += para_size
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'doc_name': doc_name,
                'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            })
        return chunks

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using keyword matching (same as ModuleAgent)."""
        query_words = set(query.lower().split())
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            score = len(query_words & chunk_words) / len(query_words) if query_words else 0
            scored_chunks.append((score, chunk))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def query(self, question: str) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """Query the documents for an answer to the given question (same as ModuleAgent)."""
        relevant_chunks = self.search_chunks(question, top_k=5)
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the loaded documents.",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.name
            }

        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"From {chunk['doc_name']}:\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > 8000:
            context = context[:8000] + "\n\n[Context truncated for length...]"

        # Use PromptManager for document prompt
        system_prompt = self.prompt_manager.get_prompt("document_single")
        prompt = self._create_standard_prompt(system_prompt, context, question)

        result = self._invoke_llm_with_tracking(prompt)
        result["agent_name"] = self.name
        return result

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
