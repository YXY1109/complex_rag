"""
文档解析接口

此模块包含文档解析服务的抽象接口。
"""

from .parser_interface import (
    # 解析器接口和模型
    DocumentParserInterface,
    ParserConfig,
    ParserCapabilities,
    DocumentType,
    ProcessingStrategy,
    ParseResult,
    DocumentMetadata,
    TextChunk,
    ImageInfo,
    TableInfo,

    # 解析器异常
    ParseException,
    UnsupportedFormatError,
    CorruptedFileError,
    ProcessingError,
    ValidationError,
    TimeoutError,
)

from .format_converter_interface import (
    # 格式转换器接口
    FormatConverterInterface,
    ConversionConfig,
    ConversionResult,
    ConversionError,
)

from .source_processor_interface import (
    # 来源处理器接口
    SourceProcessorInterface,
    SourceConfig,
    SourceResult,
    SourceError,
)

__all__ = [
    # 文档解析接口
    "DocumentParserInterface",
    "ParserConfig",
    "ParserCapabilities",
    "DocumentType",
    "ProcessingStrategy",
    "ParseResult",
    "DocumentMetadata",
    "TextChunk",
    "ImageInfo",
    "TableInfo",

    # 解析器异常
    "ParseException",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "ProcessingError",
    "ValidationError",
    "TimeoutError",

    # 格式转换器接口
    "FormatConverterInterface",
    "ConversionConfig",
    "ConversionResult",
    "ConversionError",

    # 来源处理器接口
    "SourceProcessorInterface",
    "SourceConfig",
    "SourceResult",
    "SourceError",
]