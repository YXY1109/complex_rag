"""
Document Parser Interface Abstract Class

This module defines the abstract interface for document parsers.
All document parser implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, BinaryIO
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
import mimetypes
from pathlib import Path


class DocumentFormat(str, Enum):
    """Supported document formats."""
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
    RTF = "rtf"
    ODT = "odt"
    ODS = "ods"
    ODP = "odp"
    EPUB = "epub"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentChunk(BaseModel):
    """Document chunk model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Optional[Dict[str, Any]] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: Optional[str] = None  # text, table, image, code, etc.
    coordinates: Optional[Dict[str, float]] = None  # x, y, width, height
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=lambda: time.time())


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    file_name: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_format: DocumentFormat
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    language: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    processing_time_ms: Optional[float] = None
    parser_version: Optional[str] = None
    processing_notes: Optional[List[str]] = None


class ParseRequest(BaseModel):
    """Document parse request model."""
    file_path: Optional[str] = None
    file_content: Optional[bytes] = None
    file_stream: Optional[BinaryIO] = None
    file_format: Optional[DocumentFormat] = None
    options: Optional[Dict[str, Any]] = None
    chunk_size: Optional[int] = Field(default=1024, ge=100, le=8192)
    chunk_overlap: Optional[int] = Field(default=200, ge=0, le=1000)
    extract_images: bool = False
    extract_tables: bool = True
    extract_metadata: bool = True
    preserve_layout: bool = False
    ocr_enabled: Optional[bool] = None  # None = auto-detect
    language_hint: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None


class ParseResponse(BaseModel):
    """Document parse response model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ProcessingStatus
    document_id: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None
    chunks: List[DocumentChunk] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    parser_info: Optional[Dict[str, Any]] = None


class ParserCapabilities(BaseModel):
    """Parser capabilities model."""
    supported_formats: List[DocumentFormat]
    max_file_size_mb: Optional[int] = None
    supports_images: bool
    supports_tables: bool
    supports_ocr: bool
    supports_layout_analysis: bool
    supports_multi_language: bool
    supports_streaming: bool
    supports_async: bool
    supports_chunking: bool
    supports_metadata_extraction: bool
    supports_preserving_format: bool
    requires_internet: bool
    gpu_accelerated: bool
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ParserConfig(BaseModel):
    """Parser configuration model."""
    provider: str
    model_path: Optional[str] = None
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    timeout: int = Field(default=300, ge=1)
    max_retries: int = Field(default=3, ge=0)
    enable_caching: bool = True
    cache_ttl: int = Field(default=3600, ge=0)  # seconds
    parallel_processing: bool = True
    max_workers: int = Field(default=4, ge=1)
    memory_limit_mb: Optional[int] = Field(default=None, ge=512)
    debug_mode: bool = False
    custom_options: Optional[Dict[str, Any]] = None


class ParserInterface(ABC):
    """
    Abstract interface for document parsers.

    This class defines the contract that all document parser implementations must follow.
    It provides a unified interface for different document parsers while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: ParserConfig):
        """Initialize the parser with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.confidence_threshold = config.confidence_threshold
        self.timeout = config.timeout
        self.enable_caching = config.enable_caching
        self.cache_ttl = config.cache_ttl
        self.parallel_processing = config.parallel_processing
        self.max_workers = config.max_workers
        self._capabilities: Optional[ParserCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> ParserCapabilities:
        """
        Get the capabilities of this parser.

        Returns:
            ParserCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def parse(
        self,
        request: ParseRequest,
        **kwargs
    ) -> ParseResponse:
        """
        Parse a document.

        Args:
            request: Parse request
            **kwargs: Additional provider-specific parameters

        Returns:
            ParseResponse: Parse result

        Raises:
            ParserException: If parsing fails
        """
        pass

    @abstractmethod
    async def parse_stream(
        self,
        request: ParseRequest,
        **kwargs
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        Parse a document and yield chunks as they are processed.

        Args:
            request: Parse request
            **kwargs: Additional provider-specific parameters

        Yields:
            DocumentChunk: Parsed chunks

        Raises:
            ParserException: If parsing fails
        """
        pass

    async def parse_file(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        extract_images: bool = False,
        extract_tables: bool = True,
        **kwargs
    ) -> ParseResponse:
        """
        Parse a document from file path.

        Args:
            file_path: Path to file
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            **kwargs: Additional parameters

        Returns:
            ParseResponse: Parse result
        """
        request = ParseRequest(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_images=extract_images,
            extract_tables=extract_tables,
            **kwargs
        )

        return await self.parse(request)

    async def parse_content(
        self,
        content: bytes,
        file_format: DocumentFormat,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        extract_images: bool = False,
        extract_tables: bool = True,
        **kwargs
    ) -> ParseResponse:
        """
        Parse a document from bytes content.

        Args:
            content: File content as bytes
            file_format: Document format
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            **kwargs: Additional parameters

        Returns:
            ParseResponse: Parse result
        """
        request = ParseRequest(
            file_content=content,
            file_format=file_format,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_images=extract_images,
            extract_tables=extract_tables,
            **kwargs
        )

        return await self.parse(request)

    def detect_format(
        self,
        file_path: Optional[str] = None,
        content: Optional[bytes] = None,
        mime_type: Optional[str] = None
    ) -> DocumentFormat:
        """
        Detect document format.

        Args:
            file_path: File path
            content: File content
            mime_type: MIME type

        Returns:
            DocumentFormat: Detected format
        """
        if file_path:
            extension = Path(file_path).suffix.lower()
            format_map = {
                '.pdf': DocumentFormat.PDF,
                '.docx': DocumentFormat.DOCX,
                '.doc': DocumentFormat.DOC,
                '.xlsx': DocumentFormat.XLSX,
                '.xls': DocumentFormat.XLS,
                '.pptx': DocumentFormat.PPTX,
                '.ppt': DocumentFormat.PPT,
                '.txt': DocumentFormat.TXT,
                '.md': DocumentFormat.MD,
                '.html': DocumentFormat.HTML,
                '.htm': DocumentFormat.HTML,
                '.xml': DocumentFormat.XML,
                '.json': DocumentFormat.JSON,
                '.csv': DocumentFormat.CSV,
                '.rtf': DocumentFormat.RTF,
                '.odt': DocumentFormat.ODT,
                '.ods': DocumentFormat.ODS,
                '.odp': DocumentFormat.ODP,
                '.epub': DocumentFormat.EPUB,
                '.jpg': DocumentFormat.IMAGE,
                '.jpeg': DocumentFormat.IMAGE,
                '.png': DocumentFormat.IMAGE,
                '.gif': DocumentFormat.IMAGE,
                '.bmp': DocumentFormat.IMAGE,
                '.tiff': DocumentFormat.IMAGE,
                '.webp': DocumentFormat.IMAGE,
            }
            return format_map.get(extension, DocumentFormat.UNKNOWN)

        if mime_type:
            mime_map = {
                'application/pdf': DocumentFormat.PDF,
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentFormat.DOCX,
                'application/msword': DocumentFormat.DOC,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentFormat.XLSX,
                'application/vnd.ms-excel': DocumentFormat.XLS,
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': DocumentFormat.PPTX,
                'application/vnd.ms-powerpoint': DocumentFormat.PPT,
                'text/plain': DocumentFormat.TXT,
                'text/markdown': DocumentFormat.MD,
                'text/html': DocumentFormat.HTML,
                'application/xml': DocumentFormat.XML,
                'text/xml': DocumentFormat.XML,
                'application/json': DocumentFormat.JSON,
                'text/csv': DocumentFormat.CSV,
                'application/rtf': DocumentFormat.RTF,
                'application/vnd.oasis.opendocument.text': DocumentFormat.ODT,
                'application/vnd.oasis.opendocument.spreadsheet': DocumentFormat.ODS,
                'application/vnd.oasis.opendocument.presentation': DocumentFormat.ODP,
                'application/epub+zip': DocumentFormat.EPUB,
                'image/jpeg': DocumentFormat.IMAGE,
                'image/png': DocumentFormat.IMAGE,
                'image/gif': DocumentFormat.IMAGE,
                'image/bmp': DocumentFormat.IMAGE,
                'image/tiff': DocumentFormat.IMAGE,
                'image/webp': DocumentFormat.IMAGE,
            }
            return mime_map.get(mime_type, DocumentFormat.UNKNOWN)

        return DocumentFormat.UNKNOWN

    def supports_format(self, file_format: DocumentFormat) -> bool:
        """
        Check if the parser supports a specific format.

        Args:
            file_format: Document format to check

        Returns:
            bool: True if format is supported
        """
        return file_format in self.capabilities.supported_formats

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the parser supports a specific feature.

        Args:
            feature: Feature name (images, tables, ocr, layout, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "images": "supports_images",
            "tables": "supports_tables",
            "ocr": "supports_ocr",
            "layout_analysis": "supports_layout_analysis",
            "multi_language": "supports_multi_language",
            "streaming": "supports_streaming",
            "async": "supports_async",
            "chunking": "supports_chunking",
            "metadata_extraction": "supports_metadata_extraction",
            "preserving_format": "supports_preserving_format",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def validate_request(
        self,
        request: ParseRequest
    ) -> bool:
        """
        Validate a parse request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check if we have data to parse
        if not request.file_path and not request.file_content and not request.file_stream:
            raise ValueError("No file path, content, or stream provided")

        # Check file format support
        if request.file_format and not self.supports_format(request.file_format):
            raise ValueError(f"Format {request.file_format} is not supported")

        # Check chunk size
        if request.chunk_size and (request.chunk_size < 100 or request.chunk_size > 8192):
            raise ValueError("Chunk size must be between 100 and 8192")

        # Check chunk overlap
        if request.chunk_overlap and (request.chunk_overlap < 0 or request.chunk_overlap > 1000):
            raise ValueError("Chunk overlap must be between 0 and 1000")

        return True

    def get_parser_info(self) -> Dict[str, Any]:
        """
        Get information about the parser.

        Returns:
            Dict[str, Any]: Parser information
        """
        return {
            "provider": self.provider_name,
            "capabilities": self.capabilities.dict(),
            "config": {
                "confidence_threshold": self.confidence_threshold,
                "timeout": self.timeout,
                "enable_caching": self.enable_caching,
                "cache_ttl": self.cache_ttl,
                "parallel_processing": self.parallel_processing,
                "max_workers": self.max_workers,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the parser.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test request
            test_content = b"Hello, World! This is a test document."
            test_request = ParseRequest(
                file_content=test_content,
                file_format=DocumentFormat.TXT,
                chunk_size=100
            )

            start_time = time.time()
            response = await self.parse(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_chunks_count": len(response.chunks),
                "test_status": response.status,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "error": str(e)
            }

    async def extract_text(
        self,
        request: ParseRequest,
        **kwargs
    ) -> str:
        """
        Extract plain text from a document.

        Args:
            request: Parse request
            **kwargs: Additional parameters

        Returns:
            str: Extracted text
        """
        response = await self.parse(request, **kwargs)

        # Combine all chunks into a single text
        text_parts = []
        for chunk in response.chunks:
            if chunk.content.strip():
                text_parts.append(chunk.content)

        return "\n\n".join(text_parts)

    async def extract_metadata(
        self,
        request: ParseRequest,
        **kwargs
    ) -> Optional[DocumentMetadata]:
        """
        Extract metadata from a document.

        Args:
            request: Parse request
            **kwargs: Additional parameters

        Returns:
            Optional[DocumentMetadata]: Document metadata
        """
        request.extract_metadata = True
        response = await self.parse(request, **kwargs)
        return response.metadata


class ParserException(Exception):
    """Exception raised by document parsers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        file_path: str = None,
        file_format: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.file_path = file_path
        self.file_format = file_format
        self.error_code = error_code


class UnsupportedFormatException(ParserException):
    """Exception raised when format is not supported."""
    pass


class ValidationException(ParserException):
    """Exception raised when validation fails."""
    pass


class ProcessingException(ParserException):
    """Exception raised when processing fails."""
    pass


class TimeoutException(ParserException):
    """Exception raised when processing times out."""
    pass