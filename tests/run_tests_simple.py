#!/usr/bin/env python3
"""
Simple test runner script
"""
import sys
import os
import subprocess
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run tests with pytest"""
    print("Running Complex RAG Tests...")
    print(f"Project root: {project_root}")
    print()

    # Check if pytest is available
    try:
        import pytest
        print("pytest is available")
    except ImportError:
        print("ERROR: pytest is not installed")
        print("Please install it with: pip install pytest pytest-asyncio pytest-cov")
        return 1

    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())