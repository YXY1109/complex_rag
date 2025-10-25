"""
Scanned Documents Source Handler

Provides specialized processing for scanned documents and images
requiring OCR and computer vision.
"""

from .scanned_documents_handler import ScannedDocumentsHandler, ScannedDocumentFeatures

__all__ = [
    "ScannedDocumentsHandler",
    "ScannedDocumentFeatures"
]