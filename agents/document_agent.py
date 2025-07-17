"""DocumentAgent: Handles loading, chunking, and querying of individual documents and synthesis
across documents.

This module provides the DocumentAgent and SynthesisAgent classes for document-level and
multi-document LLM-based querying.
"""

import os
import time
from typing import Dict, Any, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from colorama import Fore, Style
import google.generativeai as genai  # pylint: disable=unused-import

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
    def __init__(self, name: str, document_path: Optional[str] = None,
                 model: str = DEFAULT_DOCUMENT_MODEL,
                 documents: Optional[List[Dict[str, str]]] = None):
        """Initialize the DocumentAgent with document name, path, model, and optional multi-docs."""
        self.name = name
        self.document_path = document_path
        self.model = model
        self.documents = documents
        self.llm = self._initialize_llm()
        self.total_cost = 0.0
        self.total_tokens = 0
        self.multi_document_mode = False
        if document_path:
            self.content = self._load_document()
        elif documents:
            self.multi_document_mode = True
            self.contents = self._load_documents()
        else:
            raise ValueError("Either document_path or documents must be provided.")

    def _load_document(self) -> str:
        """Load the document content from the file system."""
        try:
            with open(self.document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Document not found: {self.document_path}") from exc
        except (OSError, UnicodeDecodeError) as e:  # pylint: disable=broad-exception-raised
            raise Exception(f"Error loading document {self.document_path}: {e}") from e  # pylint: disable=broad-exception-raised

    def _load_documents(self) -> List[Dict[str, str]]:
        """Load all documents in multi-document mode."""
        loaded = []
        for doc in self.documents:
            try:
                with open(doc['path'], 'r', encoding='utf-8') as f:
                    loaded.append({
                        'name': doc['name'],
                        'path': doc['path'],
                        'content': f.read()
                    })
            except (FileNotFoundError, OSError, UnicodeDecodeError) as e:
                print(f"{Fore.RED}Error loading {doc['path']}: {e}{Style.RESET_ALL}")
        return loaded

    def _initialize_llm(self):
        """Initialize the language model based on the model name."""
        if self.model.startswith("gemini"):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        if self.model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        raise ValueError(f"Unsupported model: {self.model}")

    def query(self, question: str) -> Dict[str, Any]:
        """Query the document(s) for an answer to the given question."""
        if not self.multi_document_mode:
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
            except (ValueError, RuntimeError, ConnectionError) as e:  # pylint: disable=broad-exception-caught
                print(f"{Fore.RED}Error querying {self.name}: {e}{Style.RESET_ALL}")
                return {
                    "response": f"Error: {str(e)}",
                    "tokens_used": 0,
                    "cost": 0.0,
                    "duration": 0.0,
                    "agent_name": self.name
                }
        else:
            # Multi-document mode: search all docs, combine relevant content
            relevant_contents = []
            for doc in self.contents:
                if question.lower() in doc['content'].lower():
                    relevant_contents.append(f"From {doc['name']}\n---\n{doc['content']}")
            if not relevant_contents:
                return {
                    "response": f"No relevant information found in {self.name}",
                    "tokens_used": 0,
                    "cost": 0.0,
                    "duration": 0.0,
                    "agent_name": self.name
                }
            context = "\n\n---\n\n".join(relevant_contents)
            if len(context) > 8000:
                context = context[:8000] + "\n\n[Context truncated for length...]"

            prompt = f"""You are analyzing multiple documents from {self.name} to answer a question.

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
                    "response": response_text,
                    "tokens_used": tokens_used,
                    "cost": cost,
                    "duration": duration,
                    "agent_name": self.name
                }
            except (ValueError, RuntimeError, ConnectionError) as e:
                print(f"{Fore.RED}Error querying {self.name}: {e}{Style.RESET_ALL}")
                return {
                    "response": f"Error: {str(e)}",
                    "tokens_used": 0,
                    "cost": 0.0,
                    "duration": 0.0,
                    "agent_name": self.name
                }

    def _create_prompt(self, question: str) -> str:
        """Create a prompt for the LLM based on the question and document content."""
        return f"""You are analyzing a specific document to answer a question.

Document Title: {self.name}
Document Content:
---
{self.content}
---

Question: {question}

Please extract all relevant information from this document that helps answer the question. Include specific quotes or references when applicable. If the document doesn't contain relevant information, state that clearly.

Focus on:
1. Direct answers to the question
2. Related concepts mentioned
3. Specific examples or details
4. Any limitations or caveats mentioned

Provide a clear, concise response based solely on the information in this document."""

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate the number of tokens used for prompt and response."""
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate the estimated cost for the number of tokens used."""
        if self.model not in COST_ESTIMATES:
            return 0.0
        input_cost_per_token = COST_ESTIMATES[self.model]["input"] / 1_000_000
        output_cost_per_token = COST_ESTIMATES[self.model]["output"] / 1_000_000
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

class SynthesisAgent:  # pylint: disable=too-few-public-methods
    """Synthesizes responses from multiple document agents."""
    def __init__(self, model: str = DEFAULT_SYNTHESIS_MODEL):
        """Initialize the SynthesisAgent with a model."""
        self.model = model
        self.llm = self._initialize_llm()
        self.total_cost = 0.0
        self.total_tokens = 0

    def _initialize_llm(self):
        """Initialize the language model for synthesis."""
        if self.model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return ChatAnthropic(
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
        raise ValueError(f"Synthesis agent requires Claude model, got: {self.model}")

    def synthesize(self, question: str, agent_responses: List[Dict]) -> Dict[str, Any]:
        """Synthesize a response from multiple agent responses."""
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
        except (ValueError, RuntimeError, ConnectionError) as e:  # pylint: disable=broad-exception-caught
            print(f"{Fore.RED}Error in synthesis: {e}{Style.RESET_ALL}")
            return {
                "response": f"Error during synthesis: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "duration": 0.0,
                "sources": []
            }

    def _create_synthesis_prompt(self, question: str, formatted_responses: List[str]) -> str:
        """Create a synthesis prompt from the question and formatted responses."""
        responses_text = "\n---\n".join(formatted_responses)
        return (
            f"You are synthesizing information from multiple sources to provide a comprehensive answer.\n\n"  # pylint: disable=line-too-long
            f"Original Question: {question}\n\n"
            f"Source Responses:\n{responses_text}\n\n"
            "Please create a unified, coherent response that:\n"
            "1. Synthesizes information from all sources\n"
            "2. Identifies connections and patterns across sources\n"
            "3. Notes any contradictions or different perspectives\n"
            "4. Provides a clear, comprehensive answer\n\n"
            "Maintain source attribution by referencing which document each piece of information comes from.\n\n"
            "Structure your response to be clear and well-organized, highlighting key insights and relationships "  # pylint: disable=line-too-long
            "between the different sources."  # pylint: disable=line-too-long
        )

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate the number of tokens used for prompt and response."""
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate the estimated cost for the number of tokens used."""
        if self.model not in COST_ESTIMATES:
            return 0.0
        input_cost_per_token = COST_ESTIMATES[self.model]["input"] / 1_000_000
        output_cost_per_token = COST_ESTIMATES[self.model]["output"] / 1_000_000
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
