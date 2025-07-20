#!/usr/bin/env python3
"""Simple test to verify basic functionality."""

from dotenv import load_dotenv
from interactive_session import InteractiveSession

# Load environment variables
load_dotenv()

print("Testing basic functionality...")

# Create session
session = InteractiveSession()

# Test loading a small directory
print("Loading documents...")
session.onecmd("/load classes/cpsc_8400/advent of code --recursive")

# Test summarize
print("Testing summarize...")
session.onecmd("/summarize")

# Test use-embeddings
print("Testing use-embeddings...")
session.onecmd("/use-embeddings on")

print("Basic test completed!")