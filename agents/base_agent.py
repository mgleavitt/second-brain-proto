"""Base agent class with shared functionality for document and module agents."""

import time
from typing import Dict, Any, Optional
from colorama import Fore, Style
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from model_config import ModelConfig
from prompt_manager import PromptManager


class BaseAgent:  # pylint: disable=too-few-public-methods
    """Base class for agents with shared LLM interaction functionality."""

    def __init__(self, model_config: Optional[ModelConfig] = None,
                 prompt_manager: Optional[PromptManager] = None,
                 agent_type: str = "base"):
        """Initialize the base agent with model config and prompt manager."""
        self.model_config = model_config or ModelConfig()
        self.prompt_manager = prompt_manager or PromptManager()
        self.agent_type = agent_type

        # Get model name from config
        self.model_name = self.model_config.get_model_name(agent_type)

        # Initialize LLM based on provider
        model_info = self.model_config.get_model_info(agent_type)
        if model_info.provider == "google":
            self.llm = ChatGoogleGenerativeAI(model=self.model_name)
        else:
            self.llm = ChatAnthropic(model=self.model_name)

        self.total_cost = 0.0
        self.total_tokens = 0

    def _invoke_llm_with_tracking(self, prompt: str, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Invoke LLM and track metrics (cost, tokens, duration)."""
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
                "duration": duration,
                "agent_name": agent_name or self.agent_type
            }
        except (ValueError, RuntimeError, ConnectionError, AttributeError) as e:
            error_msg = f"Error querying {self.agent_type}: {e}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return {
                "answer": f"Error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": agent_name or self.agent_type
            }

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate the number of tokens used for prompt and response."""
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate the estimated cost for the number of tokens used."""
        return self.model_config.estimate_cost(self.agent_type,
                                               int(tokens * 0.7),
                                               int(tokens * 0.3))  # pylint: disable=no-value-for-parameter

    def _create_standard_prompt(self, system_prompt: str, context: str, question: str) -> str:
        """Create a standard prompt format used by both document and module agents."""
        return f"""{system_prompt}

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