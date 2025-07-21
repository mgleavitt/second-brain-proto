"""Semantic caching system using sentence embeddings for similarity-based query retrieval."""

import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

# Try to import faiss, but don't fail if it's not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False



# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class SemanticCache:  # pylint: disable=too-many-instance-attributes
    """Cache that uses semantic similarity to find related queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.85,
                 cache_dir: str = ".semantic_cache",
                 ttl_hours: int = 168):  # 1 week default
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.logger = logging.getLogger(__name__)

        # Initialize embedding model
        self.logger.info("Loading embedding model: %s", model_name)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Please install with: pip install sentence-transformers"
            )

        self.embed_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Please install with: pip install faiss-cpu")

        self.index: Any = faiss.IndexFlatL2(self.embedding_dim)

        # Storage for cached data
        self.cache_data: Dict[str, Dict[str, Any]] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}

        # Load existing cache
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "semantic_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache_data = data['cache_data']
                    self.query_embeddings = data['query_embeddings']

                    # Rebuild FAISS index
                    if self.query_embeddings:
                        embeddings: np.ndarray = np.array(list(self.query_embeddings.values()))
                        self.index.add(embeddings)  # pylint: disable=no-value-for-parameter  # TODO: drop when FAISS ships stubs

                self.logger.info("Loaded %d cached queries", len(self.cache_data))
            except (pickle.PickleError, KeyError, ValueError) as e:
                self.logger.error("Error loading cache: %s", e)

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "semantic_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'cache_data': self.cache_data,
                    'query_embeddings': self.query_embeddings
                }, f)
        except (pickle.PickleError, OSError) as e:
            self.logger.error("Error saving cache: %s", e)

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for exact query matching."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _is_expired(self, timestamp: str) -> bool:
        """Check if cache entry is expired."""
        entry_time = datetime.fromisoformat(timestamp)
        return datetime.now() - entry_time > self.ttl

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for semantically similar query."""
        query_hash = self._get_query_hash(query)

        # First check exact match
        if query_hash in self.cache_data:
            entry = self.cache_data[query_hash]
            if not self._is_expired(entry['timestamp']):
                self.logger.info("Exact cache hit")
                return entry['data']

        # Check semantic similarity if we have cached queries
        if len(self.query_embeddings) == 0:
            return None

        # Embed query
        query_embedding: np.ndarray = self.embed_model.encode([query])[0]

        # Search for similar queries
        k = min(5, len(self.query_embeddings))  # Search top 5
        distances: np.ndarray
        indices: np.ndarray
        distances, indices = self.index.search(  # pylint: disable=no-value-for-parameter  # TODO: drop when FAISS ships stubs
            query_embedding.reshape(1, -1), k
        )

        # Check if any are similar enough
        query_hashes = list(self.query_embeddings.keys())
        for dist, idx in zip(distances[0], indices[0]):
            # Convert L2 distance to cosine similarity
            similarity = 1 - (dist / 2)  # Approximate conversion

            if similarity >= self.similarity_threshold:
                similar_hash = query_hashes[idx]
                entry = self.cache_data[similar_hash]

                if not self._is_expired(entry['timestamp']):
                    self.logger.info("Semantic cache hit (similarity: %.3f)", similarity)
                    self.logger.info("Original query: %s", entry['original_query'])
                    return entry['data']

        return None

    def set(self, query: str, data: Dict[str, Any]):
        """Cache query result with semantic embedding."""
        query_hash = self._get_query_hash(query)

        # Embed query
        query_embedding: np.ndarray = self.embed_model.encode([query])[0]

        # Store in cache
        self.cache_data[query_hash] = {
            'original_query': query,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

        # Update embeddings and index
        if query_hash in self.query_embeddings:
            # Remove old embedding from index (requires rebuilding)
            self._rebuild_index()

        self.query_embeddings[query_hash] = query_embedding
        self.index.add(query_embedding.reshape(1, -1))  # pylint: disable=no-value-for-parameter  # TODO: drop when FAISS ships stubs

        # Save to disk
        self._save_cache()

        self.logger.info("Cached query: %s...", query[:50])

    def _rebuild_index(self):
        """Rebuild FAISS index from scratch."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.query_embeddings:
            embeddings: np.ndarray = np.array(list(self.query_embeddings.values()))
            self.index.add(embeddings)  # pylint: disable=no-value-for-parameter  # TODO: drop when FAISS ships stubs

    def clear_expired(self):
        """Remove expired entries."""
        expired_hashes = []
        for query_hash, entry in self.cache_data.items():
            if self._is_expired(entry['timestamp']):
                expired_hashes.append(query_hash)

        for query_hash in expired_hashes:
            del self.cache_data[query_hash]
            del self.query_embeddings[query_hash]

        if expired_hashes:
            self._rebuild_index()
            self._save_cache()
            self.logger.info("Cleared %d expired entries", len(expired_hashes))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache_data)
        expired_count = sum(1 for entry in self.cache_data.values()
                          if self._is_expired(entry['timestamp']))

        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_count,
            'expired_entries': expired_count,
            'index_size': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold
        }
