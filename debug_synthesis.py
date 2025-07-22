#!/usr/bin/env python3
"""Debug script to test synthesis agent with prototype responses."""

import sys
from pathlib import Path
from loaders.course_loader import CourseModuleLoader
from agents.module_agent import ModuleAgent
from agents.document_agent import SynthesisAgent
from prompt_manager import PromptManager
from model_config import ModelConfig

def test_synthesis_with_prototype_responses():
    """Test synthesis with the exact responses the prototype would get."""
    print("Testing synthesis with prototype-style responses...")

    # Load the course
    course_root = Path("./classes/cpsc_8400")
    loader = CourseModuleLoader(course_root)
    modules = loader.load_course()

    # Create module agents exactly like the prototype
    prompt_manager = PromptManager()
    model_config = ModelConfig()
    module_agents = {}

    for module_name, documents in modules.items():
        agent = ModuleAgent(documents, model_config, prompt_manager, module_name)
        module_agents[module_name] = agent

    # Test query
    query = "compare and contrast greedy algorithms and dynamic programming"
    print(f"Query: '{query}'")

    # Query all agents and collect responses
    agent_responses = []
    for module_name, agent in module_agents.items():
        print(f"\nQuerying {module_name}...")
        response = agent.query(query)
        agent_responses.append(response)
        print(f"Response: {response['answer'][:200]}...")

    # Create synthesis agent
    synthesis_agent = SynthesisAgent(model_config, prompt_manager)

    # Synthesize responses
    print(f"\nSynthesizing {len(agent_responses)} responses...")
    synthesis_result = synthesis_agent.synthesize(query, agent_responses)

    print(f"\nFinal synthesis result:")
    print(f"{synthesis_result['response']}")

if __name__ == "__main__":
    test_synthesis_with_prototype_responses()