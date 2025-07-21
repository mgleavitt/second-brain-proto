"""ModuleAgent: Specialized agent for handling course module content and querying documents.

This module defines the ModuleAgent class, which loads, chunks, and queries module documents
using LLMs.
"""

from typing import List, Dict, Any, Optional
import hashlib
from prompt_manager import PromptManager
from model_config import ModelConfig
from .base_agent import BaseAgent


class ModuleAgent(BaseAgent):  # pylint: disable=too-many-instance-attributes
    """Agent specialized for handling course module content."""

    def __init__(self, documents: List[Dict],
                 model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """Initialize the ModuleAgent with documents, model config, and prompt manager."""
        super().__init__(model_config, prompt_manager, "module")
        self.documents = documents

        # Chunk all loaded documents
        self.chunks = self._load_and_chunk_documents()

    def _load_and_chunk_documents(self) -> List[Dict]:
        """Load and chunk all documents using semantic chunking."""
        chunks = []
        for doc in self.documents:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                doc_name = (doc.metadata.get('source', 'Unknown')
                           if hasattr(doc, 'metadata') else 'Unknown')
            else:
                content = doc.get('content', '')
                doc_name = doc.get('source', 'Unknown')

            doc_chunks = self._semantic_chunk(content, doc_name)
            chunks.extend(doc_chunks)
        return chunks

    def _semantic_chunk(self, content: str, doc_name: str,
                       chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Create semantic chunks preserving context."""
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

    def search_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant chunks using keyword matching (simple version)."""
        query_words = set(query.lower().split())
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            score = len(query_words & chunk_words) / len(query_words) if query_words else 0
            scored_chunks.append((score, chunk))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def query(self, question: str) -> Dict[str, Any]:
        """Query the module agent with a question."""
        relevant_chunks = self.search_chunks(question, top_k=5)
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in this module",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0
            }

        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"From {chunk['doc_name']}:\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > 8000:
            context = context[:8000] + "\n\n[Context truncated for length...]"

        system_prompt = self.prompt_manager.get_prompt("module")
        prompt = self._create_standard_prompt(system_prompt, context, question)

        return self._invoke_llm_with_tracking(prompt)
