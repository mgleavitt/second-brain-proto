"""Module for generating and caching summaries for educational modules.

This module provides functionality to create comprehensive summaries of educational
content, cache them for efficiency, and use them for improved routing decisions.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from model_config import ModelConfig
from prompt_manager import PromptManager

@dataclass
class ModuleSummary:
    """Summary data for a module."""
    module_name: str
    summary: str
    key_topics: List[str]
    document_count: int
    total_tokens: int
    content_hash: str
    created_at: str
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the summary to a dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleSummary':
        """Create a ModuleSummary instance from a dictionary."""
        return cls(**data)

class ModuleSummarizer:  # pylint: disable=too-many-instance-attributes
    """Generate and cache summaries for modules to improve routing efficiency."""

    SUMMARY_CACHE_FILE = "module_summaries.json"

    def __init__(self, model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None,
                 cache_dir: str = ".cache"):
        """Initialize the ModuleSummarizer.

        Args:
            model_config: Configuration for the language model
            prompt_manager: Manager for prompts
            cache_dir: Directory to store cached summaries
        """
        self.model_config = model_config or ModelConfig()
        self.prompt_manager = prompt_manager or PromptManager()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / self.SUMMARY_CACHE_FILE
        self.summaries: Dict[str, ModuleSummary] = self._load_cache()
        self.logger = logging.getLogger(__name__)

        # Use a fast model for summarization
        self.model_config.set_model("summarizer", "gemini-flash")

    def _load_cache(self) -> Dict[str, ModuleSummary]:
        """Load cached summaries from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {k: ModuleSummary.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error("Error loading summary cache: %s", e)
        return {}

    def _save_cache(self):
        """Save summaries to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.summaries.items()}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            self.logger.error("Error saving summary cache: %s", e)

    def _compute_content_hash(self, documents: List[Any]) -> str:
        """Compute hash of document contents to detect changes."""
        content_parts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                content_parts.append(doc.page_content)
            elif isinstance(doc, dict) and 'content' in doc:
                content_parts.append(doc['content'])
            else:
                content_parts.append(str(doc))
        content = "".join(sorted(content_parts))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_or_generate_summary(self, module_name: str, documents: List[Any],
                               force_regenerate: bool = False) -> ModuleSummary:
        """Get cached summary or generate new one if needed."""
        content_hash = self._compute_content_hash(documents)

        # Check if valid cached summary exists
        if not force_regenerate and module_name in self.summaries:
            cached = self.summaries[module_name]
            if cached.content_hash == content_hash:
                self.logger.info("Using cached summary for %s", module_name)
                return cached

        # Generate new summary
        self.logger.info("Generating summary for %s (%d documents)",
                        module_name, len(documents))
        summary = self._generate_summary(module_name, documents, content_hash)

        # Cache it
        self.summaries[module_name] = summary
        self._save_cache()

        return summary

    def _generate_summary(self, module_name: str, documents: List[Any],
                         content_hash: str) -> ModuleSummary:
        """Generate a comprehensive summary of module contents."""
        # Combine document contents with smart truncation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50000,  # Large chunks for summarization
            chunk_overlap=1000
        )

        # Extract content from documents
        content_parts = self._extract_document_content(documents)
        all_content = "\n\n".join(content_parts)
        chunks = text_splitter.split_text(all_content)

        # Take first few chunks that fit in context
        content_sample = "\n\n".join(chunks[:3])  # Adjust based on model context

        # Generate summary using LLM
        result = self._call_llm_for_summary(module_name, content_sample)

        # Calculate total tokens (approximate)
        total_tokens = self._calculate_total_tokens(documents)

        return ModuleSummary(
            module_name=module_name,
            summary=result.get("summary", ""),
            key_topics=result.get("key_topics", []),
            document_count=len(documents),
            total_tokens=int(total_tokens),
            content_hash=content_hash,
            created_at=datetime.now().isoformat(),
            model_used=self.model_config.get_model_name("summarizer")
        )

    def _extract_document_content(self, documents: List[Any]) -> List[str]:
        """Extract content from various document types."""
        content_parts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                content_parts.append(doc.page_content)
            elif isinstance(doc, dict) and 'content' in doc:
                content_parts.append(doc['content'])
            else:
                content_parts.append(str(doc))
        return content_parts

    def _call_llm_for_summary(self, module_name: str, content_sample: str) -> Dict[str, Any]:
        """Call the LLM to generate a summary."""
        system_prompt = """You are a module summarizer. Create a concise summary that captures:
1. The main purpose and topics covered in this module
2. Key concepts, theories, or techniques discussed
3. Important relationships between topics
4. 5-10 key terms or topics that best represent the content

Format your response as JSON with the following structure:
{
    "summary": "A 2-3 paragraph summary of the module",
    "key_topics": ["topic1", "topic2", ...]
}"""

        llm = ChatGoogleGenerativeAI(
            model=self.model_config.get_model_name("summarizer"),
            temperature=0.3,
            convert_system_message_to_human=True
        )

        prompt = f"""{system_prompt}

Module: {module_name}

Content Sample:
{content_sample[:30000]}  # Limit to reasonable size

Generate a JSON summary following the specified format."""

        response = llm.invoke(prompt)

        # Parse response
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {
                "summary": response.content[:500],
                "key_topics": []
            }
        except (json.JSONDecodeError, AttributeError):
            return {
                "summary": f"Module {module_name} summary generation failed",
                "key_topics": []
            }

    def _calculate_total_tokens(self, documents: List[Any]) -> float:
        """Calculate approximate total tokens in documents."""
        total_tokens = 0.0
        for doc in documents:
            if hasattr(doc, 'page_content'):
                total_tokens += len(doc.page_content.split()) * 1.3
            elif isinstance(doc, dict) and 'content' in doc:
                total_tokens += len(doc['content'].split()) * 1.3
            else:
                total_tokens += len(str(doc).split()) * 1.3
        return total_tokens

    def update_routing_weights(self, summaries: Dict[str, ModuleSummary]) -> Dict[str, float]:
        """Generate routing weights based on module summaries."""
        # Simple implementation - weight by document count and diversity
        weights = {}
        for name, summary in summaries.items():
            # Factor in document count and topic diversity
            doc_weight = min(summary.document_count / 10, 1.0)  # Normalize
            topic_weight = min(len(summary.key_topics) / 5, 1.0)  # Normalize
            weights[name] = (doc_weight + topic_weight) / 2

        return weights
