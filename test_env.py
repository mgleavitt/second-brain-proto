#!/usr/bin/env python3
"""Test script to verify environment variable loading."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("Environment Variables Test")
print("=" * 40)

# Check if GOOGLE_API_KEY is loaded
google_key = os.getenv('GOOGLE_API_KEY')
if google_key:
    print(f"GOOGLE_API_KEY found: {google_key[:10]}...{google_key[-10:] if len(google_key) > 20 else 'too short'}")
    print(f"Length: {len(google_key)}")

    # Check if it looks like a valid Google API key
    if google_key.startswith('AIza'):
        print("✓ Key format looks valid (starts with 'AIza')")
    else:
        print("✗ Key format looks invalid (should start with 'AIza')")
else:
    print("✗ GOOGLE_API_KEY not found")

# Check if ANTHROPIC_API_KEY is loaded
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_key:
    print(f"ANTHROPIC_API_KEY found: {anthropic_key[:10]}...{anthropic_key[-10:] if len(anthropic_key) > 20 else 'too short'}")
    print(f"Length: {len(anthropic_key)}")

    # Check if it looks like a valid Anthropic API key
    if anthropic_key.startswith('sk-ant-'):
        print("✓ Key format looks valid (starts with 'sk-ant-')")
    else:
        print("✗ Key format looks invalid (should start with 'sk-ant-')")
else:
    print("✗ ANTHROPIC_API_KEY not found")

print("\nAll environment variables:")
for key, value in os.environ.items():
    if 'API_KEY' in key:
        if value:
            print(f"{key}: {value[:10]}...{value[-10:] if len(value) > 20 else 'too short'}")
        else:
            print(f"{key}: (empty)")