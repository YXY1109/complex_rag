"""
Document Parser Plugin System

This module provides a comprehensive plugin architecture for extending document
processing capabilities, inspired by RAGFlow's extensible system.
"""

# Core interfaces and base classes
from .plugin_interface import (
    # Base plugin classes
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

    # Data structures
    PluginMetadata,
    PluginContext,
    PluginResult,
    PluginType,
    PluginStatus,

    # Factory and decorators
    PluginFactory,
    register_plugin
)

# Registry and management
from .plugin_registry import (
    PluginRegistry,
    PluginRegistration,
    RegistrationStatus,
    RegistryConfig,
    get_registry,
    set_registry
)

# Hot loading system
from .hot_loader import (
    PluginHotLoader,
    HotLoadingConfig,
    HotLoadingStatus,
    HotLoadingEvent,
    PluginFileWatcher
)

# Validation and testing
from .plugin_validator import (
    PluginValidator,
    ValidationConfig,
    ValidationReport,
    TestResult,
    ValidationLevel,
    TestStatus,
    PluginTestData
)

__all__ = [
    # Core interfaces
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

    # Data structures
    "PluginMetadata",
    "PluginContext",
    "PluginResult",
    "PluginType",
    "PluginStatus",

    # Factory and decorators
    "PluginFactory",
    "register_plugin",

    # Registry
    "PluginRegistry",
    "PluginRegistration",
    "RegistrationStatus",
    "RegistryConfig",
    "get_registry",
    "set_registry",

    # Hot loading
    "PluginHotLoader",
    "HotLoadingConfig",
    "HotLoadingStatus",
    "HotLoadingEvent",
    "PluginFileWatcher",

    # Validation
    "PluginValidator",
    "ValidationConfig",
    "ValidationReport",
    "TestResult",
    "ValidationLevel",
    "TestStatus",
    "PluginTestData"
]

# Version information
__version__ = "1.0.0"
__author__ = "Document Parser Team"
__description__ = "Extensible plugin system for document processing"