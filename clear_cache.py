#!/usr/bin/env python3
"""Script to clear cache and test query."""

import os
import shutil
from pathlib import Path

def clear_cache():
    """Clear all cache files."""
    # Clear the cache directory
    cache_dir = Path("logs")
    if cache_dir.exists():
        for file in cache_dir.glob("*"):
            if file.is_file():
                file.unlink()
        print("Cleared cache files")

    # Clear any other cache files
    cache_files = [
        "cache_manager.py",
        "semantic_cache.py"
    ]

    for cache_file in cache_files:
        if os.path.exists(cache_file):
            print(f"Cache file {cache_file} exists")

if __name__ == "__main__":
    clear_cache()
    print("Cache cleared. Now test the prototype again.")