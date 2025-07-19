"""DocumentAgent: Handles loading, chunking, and querying of individual documents and synthesis
across documents.

This module provides the DocumentAgent and SynthesisAgent classes for document-level and
multi-document LLM-based querying.
"""

import time
from typing import Dict, Any, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from colorama import Fore, Style
import google.generativeai as genai  # pylint: disable=unused-import
from prompt_manager import PromptManager
from model_config import ModelConfig

DEFAULT_DOCUMENT_MODEL = "gemini-1.5-flash"
DEFAULT_SYNTHESIS_MODEL = "claude-3-opus-20240229"
COST_ESTIMATES = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


class DocumentAgent:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Represents an agent responsible for a single document or multiple documents."""
    def __init__(self, documents: List[Dict[str, str]],
                 model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """Initialize the DocumentAgent with documents, model config, and prompt manager."""
        self.documents = documents
        self.model_config = model_config or ModelConfig()
        self.prompt_manager = prompt_manager or PromptManager()

        # Get model name from config
        self.model_name = self.model_config.get_model_name("document")

        # Initialize LLM based on provider
        model_info = self.model_config.get_model_info("document")
        if model_info.provider == "google":
            self.llm = ChatGoogleGenerativeAI(model=self.model_name)
        else:
            self.llm = ChatAnthropic(model=self.model_name)

        self.total_cost = 0.0
        self.total_tokens = 0

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
        prompt = f"""{system_prompt}

Relevant excerpts from the documents:
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
        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"{Fore.RED}Error querying documents: {e}{Style.RESET_ALL}")
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
        return self.model_config.estimate_cost("document", int(tokens * 0.7), int(tokens * 0.3))


class SynthesisAgent:  # pylint: disable=too-few-public-methods
    """Synthesizes responses from multiple document agents."""
    def __init__(self, model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """Initialize the SynthesisAgent with model config and prompt manager."""
        self.model_config = model_config or ModelConfig()
        self.prompt_manager = prompt_manager or PromptManager()

        # Get model name from config
        self.model_name = self.model_config.get_model_name("synthesis")

        # Initialize LLM based on provider
        model_info = self.model_config.get_model_info("synthesis")
        if model_info.provider == "google":
            self.llm = ChatGoogleGenerativeAI(model=self.model_name)
        else:
            self.llm = ChatAnthropic(model=self.model_name)

        self.total_cost = 0.0
        self.total_tokens = 0

    def synthesize(self, question: str, agent_responses: List[Dict]) -> Dict[str, Any]:
        """Synthesize responses from multiple agents into a coherent answer."""
        if not agent_responses:
            return {
                "synthesis": "No responses to synthesize.",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0
            }

        if len(agent_responses) == 1:
            # Single response, return as-is
            response = agent_responses[0]
            return {
                "synthesis": response.get("answer", response.get("response", "No answer provided")),
                "tokens_used": response.get("tokens_used", 0),
                "cost": response.get("cost", 0.0),
                "duration": response.get("duration", 0.0)
            }

        # Format responses for synthesis
        formatted_responses = []
        for i, response in enumerate(agent_responses, 1):
            answer = response.get("answer", response.get("response", "No answer provided"))
            agent_name = response.get("agent_name", f"Agent {i}")
            formatted_responses.append(f"Response from {agent_name}:\n{answer}")

        prompt = self._create_synthesis_prompt(question, formatted_responses)

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
                "synthesis": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration
            }
        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"{Fore.RED}Error synthesizing responses: {e}{Style.RESET_ALL}")
            return {
                "synthesis": f"Error during synthesis: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0
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

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate the number of tokens used for prompt and response."""
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate the estimated cost for the number of tokens used."""
        return self.model_config.estimate_cost("synthesis", int(tokens * 0.7), int(tokens * 0.3))
