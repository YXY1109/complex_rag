"""
Document Parser Source Handlers

This module contains specialized handlers for different document sources,
each optimized for specific content types and processing requirements.
"""

from .web_documents import WebDocumentsHandler
from .office_documents import OfficeDocumentsHandler
from .scanned_documents import ScannedDocumentsHandler
from .structured_data import StructuredDataHandler
from .code_repositories import CodeRepositoriesHandler

__all__ = [
    "WebDocumentsHandler",
    "OfficeDocumentsHandler",
    "ScannedDocumentsHandler",
    "StructuredDataHandler",
    "CodeRepositoriesHandler"
]