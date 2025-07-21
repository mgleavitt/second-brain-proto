#!/usr/bin/env python3
"""Debug script to check command availability."""

from dotenv import load_dotenv
from interactive_session import InteractiveSession

# Load environment variables
load_dotenv()

print("Debugging command availability...")

# Create session
session = InteractiveSession()

# Check if the method exists
print(f"do_use_embeddings exists: {hasattr(session, 'do_use_embeddings')}")

# List all do_ methods
do_methods = [attr for attr in dir(session) if attr.startswith('do_')]
print(f"Available do_ methods: {do_methods}")

def test_command_parsing():
    """Test the command parsing manually."""
    test_line = "/use-embeddings on"
    print(f"Testing line: '{test_line}'")

    if test_line.startswith('/'):
        cmd_name = test_line.split()[0][1:]  # Remove the /
        remainder = test_line[len(cmd_name)+2:].strip()  # +2 to account for / and space
        print(f"cmd_name: '{cmd_name}'")
        print(f"remainder: '{remainder}'")
        print(f"cmd_name type: {type(cmd_name)}")
        print(f"cmd_name repr: {repr(cmd_name)}")
        print(f"cmd_name bytes: {cmd_name.encode()}")
        print(f"hasattr(session, 'do_use_embeddings'): {hasattr(session, 'do_use_embeddings')}")
        print(f"hasattr(session, f'do_{cmd_name}'): {hasattr(session, f'do_{cmd_name}')}")
        print(f"f'do_{cmd_name}' = '{f'do_{cmd_name}'}'")
        print(f"f'do_{cmd_name}' bytes: {f'do_{cmd_name}'.encode()}")
        print(f"'do_use_embeddings' bytes: {'do_use_embeddings'.encode()}")

        # Try direct comparison
        method_name = f'do_{cmd_name}'
        print(f"method_name == 'do_use_embeddings': {method_name == 'do_use_embeddings'}")
        print(f"method_name in do_methods: {method_name in do_methods}")

        # Check character codes
        print(f"method_name ord: {[ord(c) for c in method_name]}")
        print(f"'do_use_embeddings' ord: {[ord(c) for c in 'do_use_embeddings']}")

        # Try with a different approach - use the actual method name from the list
        actual_method = 'do_use_embeddings'
        print(f"actual_method in do_methods: {actual_method in do_methods}")
        print(f"hasattr(session, actual_method): {hasattr(session, actual_method)}")

        if hasattr(session, actual_method):
            func = getattr(session, actual_method)
            print(f"func callable: {callable(func)}")
            if callable(func):
                print("Calling function...")
                func(remainder)
        else:
            print(f"Method {actual_method} not found")

# Test the command parsing
test_command_parsing()

print("Debug completed!")
