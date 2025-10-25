"""
Document Parser Vision Module

This module provides computer vision capabilities for document processing,
including OCR, layout recognition, and image analysis, inspired by RAGFlow's
vision modules.
"""

from .ocr import OCREngine, OCRResult, OCRConfig
from .recognizer import VisionRecognizer, RecognitionTask, RecognitionResult, RecognitionConfig
from .layout_recognizer import (
    LayoutRecognizer,
    LayoutRegion,
    LayoutElementType,
    LayoutConfig
)
from .table_structure_recognizer import (
    TableStructureRecognizer,
    TableStructure,
    TableCell,
    TableRow,
    TableStructureType,
    TableRecognitionConfig
)

__all__ = [
    # OCR
    "OCREngine",
    "OCRResult",
    "OCRConfig",

    # Vision Recognizer
    "VisionRecognizer",
    "RecognitionTask",
    "RecognitionResult",
    "RecognitionConfig",

    # Layout Recognition
    "LayoutRecognizer",
    "LayoutRegion",
    "LayoutElementType",
    "LayoutConfig",

    # Table Structure Recognition
    "TableStructureRecognizer",
    "TableStructure",
    "TableCell",
    "TableRow",
    "TableStructureType",
    "TableRecognitionConfig"
]