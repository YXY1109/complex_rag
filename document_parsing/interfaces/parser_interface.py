"""
文档解析器接口抽象类

此模块定义了文档解析服务的抽象接口。
所有文档解析实现都必须继承自这个基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pydantic import BaseModel, Field
from enum import Enum
import time
from dataclasses import dataclass


class DocumentType(str, Enum):
    """文档类型。"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    PPT = "ppt"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    CSV = "csv"
    IMAGE = "image"
    SCANNED = "scanned"
    MULTIMODAL = "multimodal"


class ProcessingStrategy(str, Enum):
    """处理策略。"""
    EXTRACT_TEXT = "extract_text"
    PRESERVE_LAYOUT = "preserve_layout"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"
    TABLE_EXTRACTION = "table_extraction"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_EXTRACTION = "code_extraction"
    STRUCTURED_DATA = "structured_data"
    FULL_CONTENT = "full_content"


@dataclass
class TextChunk:
    """文本块表示。"""
    content: str
    page_number: Optional[int] = None
    chunk_id: str = ""
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    font_info: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ImageInfo:
    """图像信息。"""
    image_id: str
    page_number: int
    bbox: List[float]  # [x1, y1, x2, y2]
    width: int
    height: int
    format: str
    caption: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TableInfo:
    """表格信息。"""
    table_id: str
    page_number: int
    bbox: List[float]  # [x1, y1, x2, y2]
    rows: int
    columns: int
    headers: Optional[List[str]] = None
    data: Optional[List[List[str]]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DocumentMetadata:
    """文档元数据。"""
    file_name: str
    file_size: int
    file_type: DocumentType
    mime_type: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    encryption: bool = False
    has_images: bool = False
    has_tables: bool = False
    has_forms: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParseResult:
    """解析结果。"""
    success: bool
    metadata: DocumentMetadata
    text_chunks: List[TextChunk] = None
    images: List[ImageInfo] = None
    tables: List[TableInfo] = None
    full_text: str = ""
    structured_data: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    statistics: Optional[Dict[str, Any]] = None


class ParserConfig(BaseModel):
    """解析器配置模型。"""
    parser_name: str
    document_types: List[DocumentType] = Field(description="支持的文档类型")
    processing_strategies: List[ProcessingStrategy] = Field(description="支持的处理策略")

    # OCR设置
    enable_ocr: bool = Field(default=True, description="启用OCR识别")
    ocr_languages: List[str] = Field(default=["zh", "en"], description="OCR识别语言")
    ocr_confidence_threshold: float = Field(default=0.7, description="OCR置信度阈值")

    # 布局分析设置
    enable_layout_analysis: bool = Field(default=True, description="启用布局分析")
    preserve_formatting: bool = Field(default=False, description="保留格式信息")

    # 多模态处理设置
    enable_multimodal: bool = Field(default=False, description="启用多模态处理")
    extract_images: bool = Field(default=True, description="提取图像")
    extract_tables: bool = Field(default=True, description="提取表格")

    # 文本处理设置
    chunk_size: int = Field(default=1000, description="文本块大小")
    chunk_overlap: int = Field(default=200, description="文本块重叠")
    min_chunk_size: int = Field(default=100, description="最小文本块大小")

    # 性能设置
    max_pages: Optional[int] = Field(default=None, description="最大处理页数")
    timeout_seconds: int = Field(default=300, description="超时时间（秒）")
    parallel_processing: bool = Field(default=True, description="并行处理")
    max_workers: int = Field(default=4, description="最大工作线程数")

    # 质量控制设置
    quality_threshold: float = Field(default=0.8, description="质量阈值")
    enable_validation: bool = Field(default=True, description="启用验证")

    # 自定义选项
    custom_options: Optional[Dict[str, Any]] = Field(default=None, description="自定义选项")


class ParserCapabilities(BaseModel):
    """解析器能力模型。"""
    parser_name: str
    supported_formats: List[str]
    supported_strategies: List[ProcessingStrategy]
    max_file_size_mb: Optional[int] = None
    max_pages: Optional[int] = None
    supports_ocr: bool = False
    supports_layout_analysis: bool = False
    supports_multimodal: bool = False
    supports_parallel_processing: bool = False
    supports_streaming: bool = False
    supports_incremental: bool = False
    supports_encryption: bool = False
    supports_compression: bool = False


class DocumentParserInterface(ABC):
    """
    文档解析器的抽象接口。

    此类定义了所有文档解析实现必须遵循的契约。它为不同的文档解析系统
    提供了统一的接口，同时允许特定于提供者的配置和功能。
    """

    def __init__(self, config: ParserConfig):
        """
        使用配置初始化解析器客户端。

        Args:
            config: 解析器配置
        """
        self.config = config
        self.parser_name = config.parser_name
        self.supported_types = config.document_types
        self.strategies = config.processing_strategies
        self.enable_ocr = config.enable_ocr
        self.ocr_languages = config.ocr_languages
        self.enable_layout_analysis = config.enable_layout_analysis
        self.enable_multimodal = config.enable_multimodal
        self.timeout_seconds = config.timeout_seconds
        self._capabilities: Optional[ParserCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> ParserCapabilities:
        """
        获取此解析器提供者的能力。

        Returns:
            ParserCapabilities: 支持功能的信息
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化解析器。

        Returns:
            bool: 如果初始化成功则为True

        Raises:
            ParseException: 如果初始化失败
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        清理解析器资源。
        """
        pass

    @abstractmethod
    async def parse_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy = ProcessingStrategy.EXTRACT_TEXT,
        **kwargs
    ) -> ParseResult:
        """
        解析文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果

        Raises:
            ParseException: 如果解析失败
        """
        pass

    @abstractmethod
    async def parse_bytes(
        self,
        data: bytes,
        file_name: str,
        strategy: ProcessingStrategy = ProcessingStrategy.EXTRACT_TEXT,
        **kwargs
    ) -> ParseResult:
        """
        解析字节数据。

        Args:
            data: 文件字节数据
            file_name: 文件名
            strategy: 处理策略
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果

        Raises:
            ParseException: 如果解析失败
        """
        pass

    @abstractmethod
    async def parse_stream(
        self,
        stream: BinaryIO,
        file_name: str,
        strategy: ProcessingStrategy = ProcessingStrategy.EXTRACT_TEXT,
        **kwargs
    ) -> ParseResult:
        """
        解析流数据。

        Args:
            stream: 文件流
            file_name: 文件名
            strategy: 处理策略
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果

        Raises:
            ParseException: 如果解析失败
        """
        pass

    async def extract_text_only(self, file_path: str) -> str:
        """
        仅提取文本内容。

        Args:
            file_path: 文件路径

        Returns:
            str: 提取的文本

        Raises:
            ParseException: 如果提取失败
        """
        result = await self.parse_file(file_path, ProcessingStrategy.EXTRACT_TEXT)
        if result.success:
            return result.full_text
        else:
            raise ParseException(f"文本提取失败: {result.error_message}")

    async def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        提取文档元数据。

        Args:
            file_path: 文件路径

        Returns:
            DocumentMetadata: 文档元数据

        Raises:
            ParseException: 如果提取失败
        """
        result = await self.parse_file(file_path, ProcessingStrategy.EXTRACT_TEXT)
        if result.success:
            return result.metadata
        else:
            raise ParseException(f"元数据提取失败: {result.error_message}")

    async def extract_images(self, file_path: str) -> List[ImageInfo]:
        """
        提取图像信息。

        Args:
            file_path: 文件路径

        Returns:
            List[ImageInfo]: 图像信息列表

        Raises:
            ParseException: 如果提取失败
        """
        result = await self.parse_file(file_path, ProcessingStrategy.IMAGE_ANALYSIS)
        if result.success and result.images:
            return result.images
        else:
            return []

    async def extract_tables(self, file_path: str) -> List[TableInfo]:
        """
        提取表格信息。

        Args:
            file_path: 文件路径

        Returns:
            List[TableInfo]: 表格信息列表

        Raises:
            ParseException: 如果提取失败
        """
        result = await self.parse_file(file_path, ProcessingStrategy.TABLE_EXTRACTION)
        if result.success and result.tables:
            return result.tables
        else:
            return []

    def supports_format(self, file_format: str) -> bool:
        """
        检查是否支持特定格式。

        Args:
            file_format: 文件格式

        Returns:
            bool: 如果支持格式则为True
        """
        return file_format.lower() in self.capabilities.supported_formats

    def supports_strategy(self, strategy: ProcessingStrategy) -> bool:
        """
        检查是否支持特定策略。

        Args:
            strategy: 处理策略

        Returns:
            bool: 如果支持策略则为True
        """
        return strategy in self.capabilities.supported_strategies

    def get_supported_types(self) -> List[DocumentType]:
        """
        获取支持的文档类型。

        Returns:
            List[DocumentType]: 支持的文档类型列表
        """
        return self.supported_types

    def get_supported_strategies(self) -> List[ProcessingStrategy]:
        """
        获取支持的处理策略。

        Returns:
            List[ProcessingStrategy]: 支持的处理策略列表
        """
        return self.strategies

    def get_provider_info(self) -> Dict[str, Any]:
        """
        获取解析器提供者的信息。

        Returns:
            Dict[str, Any]: 提供者信息
        """
        return {
            "parser_name": self.parser_name,
            "supported_formats": self.capabilities.supported_formats,
            "supported_strategies": [s.value for s in self.capabilities.supported_strategies],
            "max_file_size_mb": self.capabilities.max_file_size_mb,
            "max_pages": self.capabilities.max_pages,
            "features": {
                "supports_ocr": self.capabilities.supports_ocr,
                "supports_layout_analysis": self.capabilities.supports_layout_analysis,
                "supports_multimodal": self.capabilities.supports_multimodal,
                "supports_parallel_processing": self.capabilities.supports_parallel_processing,
                "supports_streaming": self.capabilities.supports_streaming,
                "supports_incremental": self.capabilities.supports_incremental,
                "supports_encryption": self.capabilities.supports_encryption,
                "supports_compression": self.capabilities.supports_compression
            },
            "config": {
                "enable_ocr": self.enable_ocr,
                "ocr_languages": self.ocr_languages,
                "enable_layout_analysis": self.enable_layout_analysis,
                "enable_multimodal": self.enable_multimodal,
                "timeout_seconds": self.timeout_seconds
            }
        }


class ParseException(Exception):
    """解析器引发的异常。"""

    def __init__(
        self,
        message: str,
        parser: str = None,
        file_path: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.parser = parser
        self.file_path = file_path
        self.error_code = error_code


class UnsupportedFormatError(ParseException):
    """不支持的格式引发的异常。"""
    pass


class CorruptedFileError(ParseException):
    """损坏文件引发的异常。"""
    pass


class ProcessingError(ParseException):
    """处理错误引发的异常。"""
    pass


class ValidationError(ParseException):
    """验证错误引发的异常。"""
    pass


class TimeoutError(ParseException):
    """超时错误引发的异常。"""
    pass