"""
Code Repositories Source Handler

Provides specialized processing for code repositories and source files
including syntax highlighting, dependency extraction, and code structure analysis.
"""

from .code_repositories_handler import CodeRepositoriesHandler, CodeFeatures

__all__ = [
    "CodeRepositoriesHandler",
    "CodeFeatures"
]