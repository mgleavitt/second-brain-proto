#!/usr/bin/env python3
"""Debug script to test search functionality."""

import sys
from pathlib import Path
from loaders.course_loader import CourseModuleLoader
from agents.module_agent import ModuleAgent
from prompt_manager import PromptManager
from model_config import ModelConfig

def test_module_search():
    """Test the search functionality of a specific module."""
    print("Testing Module 7 search functionality...")

    # Load the course
    course_root = Path("./classes/cpsc_8400")
    loader = CourseModuleLoader(course_root)
    modules = loader.load_course()

    # Get Module 7
    module_7_docs = modules.get("module_7", [])
    print(f"Module 7 has {len(module_7_docs)} documents")

    # Print document names
    for doc in module_7_docs:
        print(f"  - {doc['name']} ({len(doc.get('content', ''))} chars)")

    # Create a ModuleAgent for Module 7
    prompt_manager = PromptManager()
    model_config = ModelConfig()
    agent = ModuleAgent(module_7_docs, model_config, prompt_manager, "module_7")

    print(f"\nModule agent created with {len(agent.chunks)} chunks")

    # Test search
    query = "compare and contrast greedy algorithms with dynamic programming"
    print(f"\nSearching for: '{query}'")

    relevant_chunks = agent.search_chunks(query, top_k=5)
    print(f"Found {len(relevant_chunks)} relevant chunks:")

    for i, chunk in enumerate(relevant_chunks):
        print(f"\nChunk {i+1} (from {chunk['doc_name']}):")
        print(f"Text: {chunk['text'][:200]}...")

    # Test individual terms
    print(f"\nTesting individual search terms:")
    for term in ["greedy", "dynamic programming", "algorithm"]:
        chunks = agent.search_chunks(term, top_k=3)
        print(f"  '{term}': {len(chunks)} chunks found")
        if chunks:
            print(f"    First chunk from: {chunks[0]['doc_name']}")

    # Test the actual query method
    print(f"\nTesting ModuleAgent.query() method:")
    result = agent.query(query)
    print(f"Result: {result['answer'][:500]}...")

if __name__ == "__main__":
    test_module_search()