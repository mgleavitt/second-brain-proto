from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import os
import sys

# Add the parent directory to the path to import from prototype.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.document_agent import DocumentAgent, DEFAULT_DOCUMENT_MODEL

class ModuleAgent(DocumentAgent):
    """Agent specialized for handling course module content."""

    def __init__(self, module_name: str, documents: List[Dict],
                 model: str = DEFAULT_DOCUMENT_MODEL):
        self.module_name = module_name
        self.documents = documents
        self.model = model
        self.llm = self._initialize_llm()

        # Load and chunk documents
        self.chunks = []
        self._load_and_chunk_documents()

        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0

    def _load_and_chunk_documents(self):
        """Load documents and create semantic chunks."""
        for doc in self.documents:
            content = Path(doc['path']).read_text(encoding='utf-8')

            # Smart chunking based on document structure
            chunks = self._semantic_chunk(content, doc['name'])
            self.chunks.extend(chunks)

    def _semantic_chunk(self, content: str, doc_name: str,
                       chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Create semantic chunks preserving context."""
        chunks = []

        # Split by natural boundaries (paragraphs, sections)
        paragraphs = content.split('\n\n')

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'doc_name': doc_name,
                    'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                })

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-2:]  # Keep last 2 paragraphs
                    current_size = sum(len(p) for p in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Add final chunk
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
        # For scale testing, use simple keyword matching
        # In production, replace with embeddings or BM25
        query_words = set(query.lower().split())

        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            score = len(query_words & chunk_words) / len(query_words)
            scored_chunks.append((score, chunk))

        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def query(self, question: str) -> Dict[str, Any]:
        """Query the module agent with a question."""
        # First, find relevant chunks
        relevant_chunks = self.search_chunks(question, top_k=3)

        if not relevant_chunks:
            return {
                "response": f"No relevant information found in {self.module_name} for this question.",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.module_name,
                "chunks_used": []
            }

        # Create context from relevant chunks
        context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])

        # Create prompt with context
        prompt = f"""You are analyzing content from the {self.module_name} module to answer a question.

Module: {self.module_name}
Relevant Content:
---
{context}
---

Question: {question}

Please extract all relevant information from this module content that helps answer the question. Include specific references when applicable. If the content doesn't contain relevant information, state that clearly.

Focus on:
1. Direct answers to the question
2. Related concepts mentioned
3. Specific examples or details
4. Any limitations or caveats mentioned

Provide a clear, concise response based solely on the information in this module."""

        try:
            import time
            start_time = time.time()
            response = self.llm.invoke([self._create_message(prompt)])
            duration = time.time() - start_time

            # Extract response content
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Estimate token usage and cost
            tokens_used = self._estimate_tokens(prompt, response_text)
            cost = self._calculate_cost(tokens_used)

            # Update tracking
            self.total_cost += cost
            self.total_tokens += tokens_used

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "agent_name": self.module_name,
                "chunks_used": [chunk['chunk_id'] for chunk in relevant_chunks]
            }

        except Exception as e:
            print(f"Error querying {self.module_name}: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.module_name,
                "chunks_used": []
            }

    def _create_message(self, content: str):
        """Create a message for the LLM."""
        from langchain.schema import HumanMessage
        return HumanMessage(content=content)