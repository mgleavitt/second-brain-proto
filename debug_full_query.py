#!/usr/bin/env python3
"""Debug script to test the full prototype query process."""

import sys
from pathlib import Path
from loaders.course_loader import CourseModuleLoader
from agents.module_agent import ModuleAgent
from agents.document_agent import SynthesisAgent
from prompt_manager import PromptManager
from model_config import ModelConfig

def test_full_query():
    """Test the full query process like the prototype does."""
    print("Testing full prototype query process...")

    # Load the course
    course_root = Path("./classes/cpsc_8400")
    loader = CourseModuleLoader(course_root)
    modules = loader.load_course()

    # Create module agents
    prompt_manager = PromptManager()
    model_config = ModelConfig()
    module_agents = {}

    for module_name, documents in modules.items():
        agent = ModuleAgent(documents, model_config, prompt_manager, module_name)
        module_agents[module_name] = agent
        print(f"Created {module_name} with {len(documents)} documents")

    # Test query
    query = "compare and contrast greedy algorithms with dynamic programming"
    print(f"\nQuerying all modules for: '{query}'")

    # Query all agents (like the prototype does)
    agent_responses = []
    for module_name, agent in module_agents.items():
        print(f"  Querying {module_name}...")
        response = agent.query(query)
        agent_responses.append(response)
        print(f"    Response: {response['answer'][:100]}...")

    # Create synthesis agent
    synthesis_agent = SynthesisAgent(model_config, prompt_manager)

    # Synthesize responses
    print(f"\nSynthesizing {len(agent_responses)} responses...")
    synthesis_result = synthesis_agent.synthesize(query, agent_responses)

    print(f"\nFinal result:")
    print(f"{synthesis_result['response'][:1000]}...")

if __name__ == "__main__":
    test_full_query()