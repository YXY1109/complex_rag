"""
Structured Data Source Handler

Provides specialized processing for structured data including JSON, XML,
CSV, YAML, TOML and other structured formats.
"""

from .structured_data_handler import StructuredDataHandler, StructuredDataFeatures

__all__ = [
    "StructuredDataHandler",
    "StructuredDataFeatures"
]