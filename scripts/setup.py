#!/usr/bin/env python3
"""
Project Setup Script

This script sets up the development environment for the Complex RAG project.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_uv():
    """Install uv package manager."""
    print("Installing uv package manager...")
    run_command("curl -LsSf https://astral.sh/uv/install.sh | sh")
    print("Please restart your terminal or run: source ~/.bashrc")


def setup_project():
    """Setup the project with uv."""
    project_root = Path(__file__).parent.parent

    # Change to project root
    import os
    os.chdir(project_root)

    print(f"Setting up project in {project_root}")

    # Install dependencies with uv
    print("Installing dependencies...")
    run_command("uv sync --dev")

    # Install pre-commit hooks
    print("Installing pre-commit hooks...")
    run_command("uv run pre-commit install")

    # Download NLTK data
    print("Downloading NLTK data...")
    run_command("uv run python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"")

    print("✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment: source .venv/bin/activate")
    print("2. Copy .env.example to .env and configure your settings")
    print("3. Run tests: uv run pytest")
    print("4. Start development server: uv run python -m complex_rag.api.main")


def main():
    """Main setup function."""
    print("🚀 Complex RAG Project Setup")
    print("=" * 40)

    if not check_uv_installed():
        print("❌ uv is not installed")
        response = input("Do you want to install uv? (y/N): ")
        if response.lower() == 'y':
            install_uv()
            print("Please restart your terminal and run this script again.")
            return
        else:
            print("Please install uv manually: https://docs.astral.sh/uv/")
            return

    print("✅ uv is installed")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        return

    setup_project()


if __name__ == "__main__":
    main()