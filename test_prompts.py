#!/usr/bin/env python3
"""
Test script for PromptManager functionality.

This script tests the file-based system prompt management system to ensure
all features work correctly including loading, caching, custom paths, and reloading.
"""

import logging
import tempfile
import os
from pathlib import Path
from prompt_manager import PromptManager

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_basic_functionality():
    """Test basic PromptManager functionality."""
    print("=== Testing Basic Functionality ===")

    # Test with default directory
    pm = PromptManager()

    # Test loading all prompts
    for key in pm.PROMPT_FILES:
        prompt = pm.get_prompt(key)
        print(f"✓ {key}: {len(prompt)} characters loaded")
        assert len(prompt) > 0, f"Prompt for {key} should not be empty"

    print("✓ All prompts loaded successfully\n")

def test_custom_prompt_directory():
    """Test PromptManager with custom prompt directory."""
    print("=== Testing Custom Prompt Directory ===")

    # Create temporary directory with custom prompts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

                # Create a custom prompt file with the expected name
        custom_prompt = """# Custom Document Single Prompt

## Overview
This is a custom prompt for testing.

## Instructions
You are a custom document analysis agent for testing purposes.

## Guidelines
- This is a test prompt
- Be thorough in testing
- Report any issues found

---
*Custom test prompt*
"""

        custom_file = temp_path / "document_agent_single.md"
        with open(custom_file, 'w', encoding='utf-8') as f:
            f.write(custom_prompt)

        # Test PromptManager with custom directory
        pm = PromptManager(str(temp_path))

        # Test that custom prompt is loaded
        prompt = pm.get_prompt("document_single")
        assert "custom document analysis agent" in prompt.lower(), "Custom prompt should be loaded"
        print("✓ Custom prompt directory works correctly\n")

def test_custom_prompt_paths():
    """Test setting custom prompt paths."""
    print("=== Testing Custom Prompt Paths ===")

    # Create temporary file with custom prompt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        custom_prompt = """# Custom Synthesis Prompt

## Overview
This is a custom synthesis prompt for testing.

## Instructions
You are a custom synthesis agent for testing purposes.

## Guidelines
- This is a test synthesis prompt
- Combine responses carefully
- Report synthesis quality

---
*Custom test synthesis prompt*
"""
        f.write(custom_prompt)
        temp_file_path = f.name

    try:
        # Test PromptManager with custom path
        pm = PromptManager()
        pm.set_custom_prompt_path('synthesis', temp_file_path)

        # Test that custom prompt is loaded
        prompt = pm.get_prompt('synthesis')
        assert "custom synthesis agent" in prompt.lower(), \
            "Custom synthesis prompt should be loaded" #pylint: disable=line-too-long
        print("✓ Custom prompt paths work correctly")

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

    print()

def test_reload_functionality():
    """Test prompt reloading functionality."""
    print("=== Testing Reload Functionality ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create initial prompt file
        initial_prompt = """# Initial Prompt

## Instructions
This is the initial version of the prompt.

---
*Initial version*
"""

        prompt_file = temp_path / "document_agent_single.md"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(initial_prompt)

        # Initialize PromptManager
        pm = PromptManager(str(temp_path))

        # Load initial prompt
        initial_content = pm.get_prompt("document_single")
        assert "initial version" in initial_content.lower(), "Initial prompt should be loaded"

        # Update the prompt file
        updated_prompt = """# Updated Prompt

## Instructions
This is the updated version of the prompt.

---
*Updated version*
"""

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(updated_prompt)

        # Reload prompts
        reloaded = pm.reload_prompts()

        # Check that updated prompt is loaded
        updated_content = pm.get_prompt("document_single")
        assert "updated version" in updated_content.lower(), "Updated prompt should be loaded"
        assert "initial version" not in updated_content.lower(), "Old prompt should not be present"

        print(f"✓ Reloaded {len(reloaded)} prompts successfully")
        print("✓ Prompt content updated correctly\n")

def test_fallback_to_defaults():
    """Test fallback to default prompts when files don't exist."""
    print("=== Testing Fallback to Defaults ===")

    # Create temporary directory (empty)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize PromptManager with empty directory
        pm = PromptManager(str(temp_path))

        # Test that defaults are used
        for key in pm.PROMPT_FILES:
            prompt = pm.get_prompt(key)
            assert len(prompt) > 0, f"Default prompt for {key} should not be empty"
            print(f"✓ {key}: Using default prompt ({len(prompt)} characters)")

        print("✓ All prompts fallback to defaults correctly\n")

def test_create_default_files():
    """Test creating default prompt files."""
    print("=== Testing Create Default Files ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize PromptManager
        pm = PromptManager(str(temp_path))

        # Create default files
        pm.create_default_prompt_files()

        # Check that files were created
        for key, filename in pm.PROMPT_FILES.items():
            file_path = temp_path / filename
            assert file_path.exists(), f"Default file {filename} should be created"

            # Check file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0, f"Default file {filename} should not be empty"
                assert key.replace('_', ' ').title() in content, \
                    f"File {filename} should contain prompt key"

            print(f"✓ Created default file: {filename}")

        print("✓ All default files created successfully\n")

def run_all_tests():
    """Run all tests."""
    print("🧪 Running PromptManager Tests\n")

    try:
        test_basic_functionality()
        test_custom_prompt_directory()
        test_custom_prompt_paths()
        test_reload_functionality()
        test_fallback_to_defaults()
        test_create_default_files()

        print("🎉 All tests passed successfully!")
        print("\nThe PromptManager is working correctly with all features:")
        print("✓ Basic prompt loading and caching")
        print("✓ Custom prompt directories")
        print("✓ Custom prompt file paths")
        print("✓ Prompt reloading")
        print("✓ Fallback to defaults")
        print("✓ Default file creation")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    run_all_tests()
