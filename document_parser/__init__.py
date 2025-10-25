"""
Document Parser Module

This module provides comprehensive document parsing capabilities,
inspired by RAGFlow's deepdoc architecture, with support for multiple
document sources, vision analysis, and advanced processing pipelines.
"""

# Interfaces and base classes
from .interfaces.source_interface import (
    ParseRequest,
    ParseResponse,
    Content,
    ProcessingStrategy,
    SourceType
)

# Services
from .services.file_source_detector import FileSourceDetector
from .services.processing_strategy_selector import ProcessingStrategySelector
from .services.processing_pipeline import DocumentProcessingPipeline
from .services.batch_processor import BatchProcessor
from .services.multimodal_fusion import MultimodalFusion
from .services.structure_preservation import StructurePreservation

# Source handlers
from .source_handlers.web_documents.web_documents_handler import WebDocumentsHandler
from .source_handlers.office_documents.office_documents_handler import OfficeDocumentsHandler
from .source_handlers.scanned_documents.scanned_documents_handler import ScannedDocumentsHandler
from .source_handlers.structured_data.structured_data_handler import StructuredDataHandler
from .source_handlers.code_repositories.code_repositories_handler import CodeRepositoriesHandler

# Vision modules
from .vision import (
    OCREngine,
    OCRResult,
    OCRConfig,
    VisionRecognizer,
    RecognitionTask,
    RecognitionResult,
    RecognitionConfig,
    LayoutRecognizer,
    LayoutRegion,
    LayoutElementType,
    LayoutConfig,
    TableStructureRecognizer,
    TableStructure,
    TableCell,
    TableRow,
    TableStructureType,
    TableRecognitionConfig
)

# Quality monitoring
from .services.quality_monitor import (
    QualityMonitor,
    QualityMetric,
    QualityConfig
)

# Plugin system
from .plugins import (
    BasePlugin,
    SourceHandlerPlugin,
    ContentProcessorPlugin,
    VisionAnalyzerPlugin,
    QualityCheckerPlugin,
    OutputFormatterPlugin,
    PreprocessorPlugin,
    PostprocessorPlugin,
    TransformerPlugin,
    ValidatorPlugin,
    EnricherPlugin,
    PluginMetadata,
    PluginContext,
    PluginResult,
    PluginType,
    PluginStatus,
    PluginFactory,
    PluginRegistry,
    PluginHotLoader,
    PluginValidator,
    register_plugin
)

__all__ = [
    # Interfaces
    "ParseRequest",
    "ParseResponse",
    "Content",
    "ProcessingStrategy",
    "SourceType",

    # Services
    "FileSourceDetector",
    "ProcessingStrategySelector",
    "DocumentProcessingPipeline",
    "BatchProcessor",
    "MultimodalFusion",
    "StructurePreservation",

    # Source Handlers
    "WebDocumentsHandler",
    "OfficeDocumentsHandler",
    "ScannedDocumentsHandler",
    "StructuredDataHandler",
    "CodeRepositoriesHandler",

    # Vision Modules
    "OCREngine",
    "OCRResult",
    "OCRConfig",
    "VisionRecognizer",
    "RecognitionTask",
    "RecognitionResult",
    "RecognitionConfig",
    "LayoutRecognizer",
    "LayoutRegion",
    "LayoutElementType",
    "LayoutConfig",
    "TableStructureRecognizer",
    "TableStructure",
    "TableCell",
    "TableRow",
    "TableStructureType",
    "TableRecognitionConfig",

    # Quality Monitoring
    "QualityMonitor",
    "QualityMetric",
    "QualityConfig",

    # Plugin System
    "BasePlugin",
    "SourceHandlerPlugin",
    "ContentProcessorPlugin",
    "VisionAnalyzerPlugin",
    "QualityCheckerPlugin",
    "OutputFormatterPlugin",
    "PreprocessorPlugin",
    "PostprocessorPlugin",
    "TransformerPlugin",
    "ValidatorPlugin",
    "EnricherPlugin",
    "PluginMetadata",
    "PluginContext",
    "PluginResult",
    "PluginType",
    "PluginStatus",
    "PluginFactory",
    "PluginRegistry",
    "PluginHotLoader",
    "PluginValidator",
    "register_plugin"
]

# Version information
__version__ = "1.0.0"
__author__ = "Document Parser Team"
__description__ = "Comprehensive document parsing with RAGFlow-inspired architecture"