"""ModuleAgent: Specialized agent for handling course module content and querying documents.

This module defines the ModuleAgent class, which loads, chunks, and queries module documents
using LLMs.
"""

from typing import List, Dict, Any, Optional
import hashlib
import time
from langchain.schema import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt_manager import PromptManager
from model_config import ModelConfig


class ModuleAgent:  # pylint: disable=too-many-instance-attributes
    """Agent specialized for handling course module content."""

    def __init__(self, documents: List[Dict],
                 model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """Initialize the ModuleAgent with documents, model config, and prompt manager."""
        self.documents = documents
        self.model_config = model_config or ModelConfig()
        self.prompt_manager = prompt_manager or PromptManager()

        # Get model name from config
        self.model_name = self.model_config.get_model_name("module")

        # Initialize LLM based on provider
        model_info = self.model_config.get_model_info("module")
        if model_info.provider == "google":
            self.llm = ChatGoogleGenerativeAI(model=self.model_name)
        else:
            self.llm = ChatAnthropic(model=self.model_name)

        self.total_cost = 0.0
        self.total_tokens = 0

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
        prompt = f"""{system_prompt}

Relevant excerpts from the module:
---
{context}
---

Question: {question}

Please provide a comprehensive answer based on the information in these excerpts. Focus on:
1. Direct answers to the question
2. Related concepts mentioned
3. Specific examples or details
4. Any limitations or caveats mentioned

Provide a clear, concise response based solely on the information provided."""

        try:
            start_time = time.time()
            response = self.llm.invoke([HumanMessage(content=prompt)])
            duration = time.time() - start_time
            response_text = response.content if hasattr(response, 'content') else str(response)
            tokens_used = self._estimate_tokens(prompt, response_text)
            cost = self._calculate_cost(tokens_used)
            self.total_cost += cost
            self.total_tokens += tokens_used
            return {
                "answer": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration
            }
        except (ValueError, AttributeError, RuntimeError) as e:
            print(f"Error querying module: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0
            }

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate the number of tokens used for prompt and response."""
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate the estimated cost for the number of tokens used."""
        return self.model_config.estimate_cost("module", int(tokens * 0.7), int(tokens * 0.3))
