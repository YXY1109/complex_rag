"""
Office Documents Source Handler

Provides specialized processing for office documents including PDF, DOCX,
XLSX, PPTX and other office formats.
"""

from .office_documents_handler import OfficeDocumentsHandler, OfficeDocumentFeatures

__all__ = [
    "OfficeDocumentsHandler",
    "OfficeDocumentFeatures"
]