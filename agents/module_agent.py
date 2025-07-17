"""ModuleAgent: Specialized agent for handling course module content and querying documents.

This module defines the ModuleAgent class, which loads, chunks, and queries module documents
using LLMs.
"""

from typing import List, Dict, Any
import hashlib
import time
from langchain.schema import HumanMessage
from agents.document_agent import DocumentAgent, DEFAULT_DOCUMENT_MODEL

class ModuleAgent(DocumentAgent):
    """Agent specialized for handling course module content."""

    def __init__(self, module_name: str, documents: List[Dict],
                 model: str = DEFAULT_DOCUMENT_MODEL):
        # Use DocumentAgent's multi-document mode
        super().__init__(module_name, model=model, documents=documents)
        self.module_name = module_name
        self.model = model
        # Chunk all loaded documents
        self.chunks = self._load_and_chunk_documents()
        self.total_cost = 0.0
        self.total_tokens = 0

    def _load_and_chunk_documents(self) -> List[Dict]:
        """Load and chunk all documents using semantic chunking."""
        chunks = []
        for doc in self.contents:
            content = doc['content']
            doc_name = doc['name']
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
                "response": f"No relevant information found in {self.module_name}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.module_name
            }
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"From {chunk['doc_name']}:\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)
        if len(context) > 8000:
            context = context[:8000] + "\n\n[Context truncated for length...]"
        prompt = f"""You are analyzing documents from {self.module_name} to answer a question.

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
            response = self.llm.invoke([self._create_message(prompt)])
            duration = time.time() - start_time
            response_text = response.content if hasattr(response, 'content') else str(response)
            tokens_used = self._estimate_tokens(prompt, response_text)
            cost = self._calculate_cost(tokens_used)
            self.total_cost += cost
            self.total_tokens += tokens_used
            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "agent_name": self.module_name
            }
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error querying {self.module_name}: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.module_name
            }

    def _create_message(self, content: str):
        """Create a message for the LLM."""
        return HumanMessage(content=content)
