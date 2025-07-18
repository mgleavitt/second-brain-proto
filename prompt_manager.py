"""
Prompt Manager Module

This module provides a PromptManager class for loading and caching system prompts
from markdown files. It handles prompt file management, caching, and fallback to
default prompts when files are not available.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class PromptManager:
    """Manages loading and caching of system prompts from markdown files."""

    # Default prompt directory
    DEFAULT_PROMPT_DIR = "prompts"

    # Prompt file mapping
    PROMPT_FILES = {
        "document_single": "document_agent_single.md",
        "document_multi": "document_agent_multi.md",
        "module": "module_agent.md",
        "synthesis": "synthesis_agent.md"
    }

    # Hardcoded fallback prompts
    DEFAULT_PROMPTS = {
        "document_single": (
            "You are a document analysis agent. Your task is to answer questions "
            "based on the provided document content. Be thorough and accurate."
        ),
        "document_multi": (
            "You are a multi-document analysis agent. Your task is to answer "
            "questions by synthesizing information from multiple documents. "
            "Identify connections and patterns across documents."
        ),
        "module": (
            "You are a module agent managing multiple related documents. "
            "Coordinate document agents to provide comprehensive answers."
        ),
        "synthesis": (
            "You are a synthesis agent. Combine and synthesize responses from "
            "multiple agents into a coherent, comprehensive answer."
        )
    }

    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompt_dir = Path(prompt_dir or self.DEFAULT_PROMPT_DIR)
        self.prompts_cache: Dict[str, str] = {}
        self.custom_paths: Dict[str, Path] = {}
        self.logger = logging.getLogger(__name__)

        # Create prompt directory if it doesn't exist
        self.prompt_dir.mkdir(exist_ok=True)

        # Load all prompts on initialization
        self.load_all_prompts()

    def load_all_prompts(self) -> None:
        """Load all prompts from files or use defaults."""
        for prompt_key, filename in self.PROMPT_FILES.items():
            self.load_prompt(prompt_key, filename)

    def load_prompt(self, prompt_key: str, filename: Optional[str] = None) -> str:
        """Load a single prompt from file or return default."""
        if filename:
            filepath = self.prompt_dir / filename
        else:
            # Check if custom path was set
            filepath = self.custom_paths.get(prompt_key)
            if not filepath:
                filepath = self.prompt_dir / self.PROMPT_FILES.get(prompt_key, "")

        if filepath and filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    self.prompts_cache[prompt_key] = content
                    self.logger.info("Loaded prompt '%s' from %s", prompt_key, filepath)
                    return content
            except (IOError, OSError) as e:
                self.logger.error("Error loading prompt from %s: %s", filepath, e)

        # Use default prompt
        self.logger.warning(
            "Prompt file not found: %s. Using default prompt for '%s'",
            filepath, prompt_key
        )
        default = self.DEFAULT_PROMPTS.get(prompt_key, "You are a helpful AI assistant.")
        self.prompts_cache[prompt_key] = default
        return default

    def get_prompt(self, prompt_key: str) -> str:
        """Get a prompt from cache or load it."""
        if prompt_key not in self.prompts_cache:
            self.load_prompt(prompt_key)
        return self.prompts_cache[prompt_key]

    def set_custom_prompt_path(self, prompt_key: str, filepath: str) -> None:
        """Set a custom file path for a specific prompt."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        self.custom_paths[prompt_key] = path
        # Reload the prompt with new path
        self.load_prompt(prompt_key)

    def reload_prompts(self) -> Dict[str, str]:
        """Reload all prompts from files."""
        self.prompts_cache.clear()
        self.load_all_prompts()
        return self.prompts_cache

    def create_default_prompt_files(self) -> None:
        """Create default prompt files if they don't exist."""
        for prompt_key, filename in self.PROMPT_FILES.items():
            filepath = self.prompt_dir / filename
            if not filepath.exists():
                default_content = self._format_default_prompt(
                    prompt_key,
                    self.DEFAULT_PROMPTS[prompt_key]
                )
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(default_content)
                self.logger.info("Created default prompt file: %s", filepath)

    def _format_default_prompt(self, prompt_key: str, content: str) -> str:
        """Format default prompt with markdown structure."""
        return f"""# {prompt_key.replace('_', ' ').title()} System Prompt

## Overview
This prompt defines the behavior for the {prompt_key} agent.

## Instructions
{content}

## Guidelines
- Be accurate and thorough
- Cite sources when applicable
- Maintain objectivity

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""