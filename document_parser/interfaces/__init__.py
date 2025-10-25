"""
Document Parser Interfaces

This module contains abstract interfaces for document parsing, format conversion, and source handling.
All concrete implementations must inherit from these base classes.
"""

from .parser_interface import (
    # Parser interfaces and models
    ParserInterface,
    ParserConfig,
    ParserCapabilities,
    ParseRequest,
    ParseResponse,
    DocumentChunk,
    DocumentMetadata,
    DocumentFormat,
    ProcessingStatus,

    # Parser exceptions
    ParserException,
    UnsupportedFormatException,
    ValidationException as ParserValidationException,
    ProcessingException as ParserProcessingException,
    TimeoutException as ParserTimeoutException,
)

from .converter_interface import (
    # Converter interfaces and models
    ConverterInterface,
    ConverterConfig,
    ConverterCapabilities,
    ConversionRequest,
    ConversionResponse,
    ConversionCapability,
    ConversionDirection,

    # Converter exceptions
    ConverterException,
    UnsupportedConversionException,
    ValidationException as ConverterValidationException,
    ProcessingException as ConverterProcessingException,
    TimeoutException as ConverterTimeoutException,
)

from .source_interface import (
    # Source handler interfaces and models
    SourceHandlerInterface,
    SourceConfig,
    SourceHandlerCapabilities,
    SourceRequest,
    SourceResponse,
    SourceDetectionResult,
    FileSource,
    ProcessingStrategy,

    # Source handler exceptions
    SourceHandlerException,
    UnsupportedSourceException,
    ValidationException as SourceValidationException,
    ProcessingException as SourceProcessingException,
    TimeoutException as SourceTimeoutException,
    NetworkException,
)

__all__ = [
    # Parser interfaces
    "ParserInterface",
    "ParserConfig",
    "ParserCapabilities",
    "ParseRequest",
    "ParseResponse",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentFormat",
    "ProcessingStatus",
    "ParserException",
    "UnsupportedFormatException",
    "ParserValidationException",
    "ParserProcessingException",
    "ParserTimeoutException",

    # Converter interfaces
    "ConverterInterface",
    "ConverterConfig",
    "ConverterCapabilities",
    "ConversionRequest",
    "ConversionResponse",
    "ConversionCapability",
    "ConversionDirection",
    "ConverterException",
    "UnsupportedConversionException",
    "ConverterValidationException",
    "ConverterProcessingException",
    "ConverterTimeoutException",

    # Source handler interfaces
    "SourceHandlerInterface",
    "SourceConfig",
    "SourceHandlerCapabilities",
    "SourceRequest",
    "SourceResponse",
    "SourceDetectionResult",
    "FileSource",
    "ProcessingStrategy",
    "SourceHandlerException",
    "UnsupportedSourceException",
    "SourceValidationException",
    "SourceProcessingException",
    "SourceTimeoutException",
    "NetworkException",
]