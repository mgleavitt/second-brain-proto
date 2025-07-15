import os
import time
import hashlib
from typing import Dict, Any, List
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from colorama import Fore, Style

DEFAULT_DOCUMENT_MODEL = "gemini-1.5-flash"
DEFAULT_SYNTHESIS_MODEL = "claude-3-opus-20240229"
COST_ESTIMATES = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}

class DocumentAgent:
    """Represents an agent responsible for a single document."""
    def __init__(self, name: str, document_path: str, model: str = DEFAULT_DOCUMENT_MODEL):
        self.name = name
        self.document_path = document_path
        self.model = model
        self.content = self._load_document()
        self.llm = self._initialize_llm()
        self.total_cost = 0.0
        self.total_tokens = 0

    def _load_document(self) -> str:
        try:
            with open(self.document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {self.document_path}")
        except Exception as e:
            raise Exception(f"Error loading document {self.document_path}: {e}")

    def _initialize_llm(self):
        if self.model.startswith("gemini"):
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        elif self.model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def query(self, question: str) -> Dict[str, Any]:
        prompt = self._create_prompt(question)
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
                "response": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "agent_name": self.name
            }
        except Exception as e:
            print(f"{Fore.RED}Error querying {self.name}: {e}{Style.RESET_ALL}")
            return {
                "response": f"Error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "agent_name": self.name
            }

    def _create_prompt(self, question: str) -> str:
        return f"""You are analyzing a specific document to answer a question.\n\nDocument Title: {self.name}\nDocument Content:\n---\n{self.content}\n---\n\nQuestion: {question}\n\nPlease extract all relevant information from this document that helps answer the question. Include specific quotes or references when applicable. If the document doesn't contain relevant information, state that clearly.\n\nFocus on:\n1. Direct answers to the question\n2. Related concepts mentioned\n3. Specific examples or details\n4. Any limitations or caveats mentioned\n\nProvide a clear, concise response based solely on the information in this document."""

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        if self.model not in COST_ESTIMATES:
            return 0.0
        input_cost_per_token = COST_ESTIMATES[self.model]["input"] / 1_000_000
        output_cost_per_token = COST_ESTIMATES[self.model]["output"] / 1_000_000
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

class SynthesisAgent:
    """Synthesizes responses from multiple document agents."""
    def __init__(self, model: str = DEFAULT_SYNTHESIS_MODEL):
        self.model = model
        self.llm = self._initialize_llm()
        self.total_cost = 0.0
        self.total_tokens = 0

    def _initialize_llm(self):
        if self.model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Synthesis agent requires Claude model, got: {self.model}")

    def synthesize(self, question: str, agent_responses: List[Dict]) -> Dict[str, Any]:
        valid_responses = [r for r in agent_responses if not r["response"].startswith("Error:")]
        if not valid_responses:
            return {
                "response": "No valid responses from document agents to synthesize.",
                "tokens_used": 0,
                "cost": 0.0,
                "sources": []
            }
        formatted_responses = []
        sources = []
        for resp in valid_responses:
            formatted_responses.append(f"Source: {resp['agent_name']}\n{resp['response']}\n")
            sources.append(resp['agent_name'])
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
                "response": response_text,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "sources": sources
            }
        except Exception as e:
            print(f"{Fore.RED}Error in synthesis: {e}{Style.RESET_ALL}")
            return {
                "response": f"Error during synthesis: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "sources": []
            }
    def _create_synthesis_prompt(self, question: str, formatted_responses: List[str]) -> str:
        responses_text = "\n---\n".join(formatted_responses)
        return f"""You are synthesizing information from multiple sources to provide a comprehensive answer.\n\nOriginal Question: {question}\n\nSource Responses:\n{responses_text}\n\nPlease create a unified, coherent response that:\n1. Synthesizes information from all sources\n2. Identifies connections and patterns across sources\n3. Notes any contradictions or different perspectives\n4. Provides a clear, comprehensive answer\n\nMaintain source attribution by referencing which document each piece of information comes from.\n\nStructure your response to be clear and well-organized, highlighting key insights and relationships between the different sources."""
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens
    def _calculate_cost(self, tokens: int) -> float:
        if self.model not in COST_ESTIMATES:
            return 0.0
        input_cost_per_token = COST_ESTIMATES[self.model]["input"] / 1_000_000
        output_cost_per_token = COST_ESTIMATES[self.model]["output"] / 1_000_000
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)