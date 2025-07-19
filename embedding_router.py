import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle

from model_config import ModelConfig
from summarizer import ModuleSummary

class EmbeddingRouter:
    """Route queries using semantic embeddings instead of keyword matching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 4,
                 min_similarity: float = 0.3):
        self.model_name = model_name
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.logger = logging.getLogger(__name__)

        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(model_name)
        except ImportError:
            self.logger.error("sentence-transformers not installed. Please install with: pip install sentence-transformers")
            raise

        # Storage for module embeddings
        self.module_embeddings: Dict[str, np.ndarray] = {}
        self.module_summaries: Dict[str, ModuleSummary] = {}

    def index_modules(self, summaries: Dict[str, ModuleSummary]):
        """Create embeddings for all module summaries."""
        self.module_summaries = summaries

        for module_name, summary in summaries.items():
            # Combine summary and key topics for richer representation
            text = f"{summary.summary}\n\nKey topics: {', '.join(summary.key_topics)}"

            # Generate embedding
            embedding = self.embed_model.encode(text)
            self.module_embeddings[module_name] = embedding

        self.logger.info(f"Indexed {len(self.module_embeddings)} modules")

    def route_query(self, query: str) -> List[Tuple[str, float]]:
        """Route query to most relevant modules using embeddings."""
        if not self.module_embeddings:
            self.logger.warning("No modules indexed for routing")
            return []

        # Embed query
        query_embedding = self.embed_model.encode(query)

        # Calculate similarities
        similarities = {}
        for module_name, module_embedding in self.module_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, module_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(module_embedding)
            )
            similarities[module_name] = similarity

        # Sort by similarity
        sorted_modules = sorted(similarities.items(),
                              key=lambda x: x[1], reverse=True)

        # Filter by minimum similarity and take top K
        selected = [(name, score) for name, score in sorted_modules
                   if score >= self.min_similarity][:self.top_k]

        if selected:
            self.logger.info(f"Routed to modules: {[m[0] for m in selected]}")
            self.logger.info(f"Similarities: {[f'{m[1]:.3f}' for m in selected]}")

        return selected

    def explain_routing(self, query: str, module_name: str) -> Dict[str, Any]:
        """Explain why a query was routed to a specific module."""
        if module_name not in self.module_embeddings:
            return {"error": "Module not found"}

        query_embedding = self.embed_model.encode(query)
        module_embedding = self.module_embeddings[module_name]

        # Calculate similarity
        similarity = np.dot(query_embedding, module_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(module_embedding)
        )

        # Find most relevant topics
        summary = self.module_summaries[module_name]
        topic_scores = {}
        for topic in summary.key_topics:
            topic_embedding = self.embed_model.encode(topic)
            topic_similarity = np.dot(query_embedding, topic_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(topic_embedding)
            )
            topic_scores[topic] = topic_similarity

        # Sort topics by relevance
        relevant_topics = sorted(topic_scores.items(),
                               key=lambda x: x[1], reverse=True)[:3]

        return {
            "module": module_name,
            "overall_similarity": float(similarity),
            "relevant_topics": relevant_topics,
            "summary_preview": summary.summary[:200] + "..."
        }

    def save_index(self, filepath: str):
        """Save routing index to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'module_embeddings': self.module_embeddings,
                'module_summaries': self.module_summaries,
                'model_name': self.model_name
            }, f)

    def load_index(self, filepath: str):
        """Load routing index from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.module_embeddings = data['module_embeddings']
            self.module_summaries = data['module_summaries']

            # Verify model compatibility
            if data['model_name'] != self.model_name:
                self.logger.warning(f"Index created with different model: {data['model_name']}")