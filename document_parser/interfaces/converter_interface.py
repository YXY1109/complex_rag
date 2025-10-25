"""
Document Converter Interface Abstract Class

This module defines the abstract interface for document format converters.
All document converter implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
from pathlib import Path

from .parser_interface import DocumentFormat


class ConversionDirection(str, Enum):
    """Conversion direction types."""
    TO_MARKDOWN = "to_markdown"
    TO_HTML = "to_html"
    TO_TEXT = "to_text"
    TO_JSON = "to_json"
    FROM_MARKDOWN = "from_markdown"
    FROM_HTML = "from_html"
    FROM_TEXT = "from_text"
    FROM_JSON = "from_json"


class ConversionRequest(BaseModel):
    """Document conversion request model."""
    input_format: DocumentFormat
    output_format: DocumentFormat
    input_path: Optional[str] = None
    input_content: Optional[bytes] = None
    input_stream: Optional[BinaryIO] = None
    output_path: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    preserve_images: bool = True
    preserve_tables: bool = True
    preserve_formatting: bool = False
    extract_metadata: bool = True
    language_hint: Optional[str] = None
    custom_css: Optional[str] = None
    template: Optional[str] = None


class ConversionResponse(BaseModel):
    """Document conversion response model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success: bool
    input_format: DocumentFormat
    output_format: DocumentFormat
    output_content: Optional[bytes] = None
    output_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    converter_info: Optional[Dict[str, Any]] = None


class ConversionCapability(BaseModel):
    """Conversion capability model."""
    input_format: DocumentFormat
    output_format: DocumentFormat
    supported: bool
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preserves_images: bool
    preserves_tables: bool
    preserves_formatting: bool
    supports_metadata: bool
    requires_internet: bool = False
    gpu_accelerated: bool = False
    max_file_size_mb: Optional[int] = None


class ConverterCapabilities(BaseModel):
    """Converter capabilities model."""
    provider: str
    supported_conversions: List[ConversionCapability]
    max_file_size_mb: Optional[int] = None
    supports_batch_conversion: bool
    supports_streaming: bool
    supports_async: bool
    supports_metadata_extraction: bool
    supports_image_extraction: bool
    supports_table_extraction: bool
    supports_template_customization: bool
    requires_internet: bool
    gpu_accelerated: bool
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConverterConfig(BaseModel):
    """Converter configuration model."""
    provider: str
    model_path: Optional[str] = None
    timeout: int = Field(default=300, ge=1)
    max_retries: int = Field(default=3, ge=0)
    enable_caching: bool = True
    cache_ttl: int = Field(default=3600, ge=0)  # seconds
    parallel_processing: bool = True
    max_workers: int = Field(default=4, ge=1)
    memory_limit_mb: Optional[int] = Field(default=None, ge=512)
    debug_mode: bool = False
    custom_options: Optional[Dict[str, Any]] = None


class ConverterInterface(ABC):
    """
    Abstract interface for document format converters.

    This class defines the contract that all document converter implementations must follow.
    It provides a unified interface for different document converters while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: ConverterConfig):
        """Initialize the converter with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.timeout = config.timeout
        self.enable_caching = config.enable_caching
        self.cache_ttl = config.cache_ttl
        self.parallel_processing = config.parallel_processing
        self.max_workers = config.max_workers
        self._capabilities: Optional[ConverterCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> ConverterCapabilities:
        """
        Get the capabilities of this converter.

        Returns:
            ConverterCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def convert(
        self,
        request: ConversionRequest,
        **kwargs
    ) -> ConversionResponse:
        """
        Convert a document from one format to another.

        Args:
            request: Conversion request
            **kwargs: Additional provider-specific parameters

        Returns:
            ConversionResponse: Conversion result

        Raises:
            ConverterException: If conversion fails
        """
        pass

    async def convert_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: Optional[DocumentFormat] = None,
        **kwargs
    ) -> ConversionResponse:
        """
        Convert a document from file path.

        Args:
            input_path: Path to input file
            output_path: Path to output file (optional)
            output_format: Output format (optional, auto-detected if not provided)
            **kwargs: Additional parameters

        Returns:
            ConversionResponse: Conversion result
        """
        input_format = self._detect_format_from_path(input_path)

        if output_format is None:
            output_format = self._suggest_output_format(input_format)

        request = ConversionRequest(
            input_format=input_format,
            output_format=output_format,
            input_path=input_path,
            output_path=output_path,
            **kwargs
        )

        return await self.convert(request)

    async def convert_content(
        self,
        content: bytes,
        input_format: DocumentFormat,
        output_format: DocumentFormat,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ConversionResponse:
        """
        Convert a document from bytes content.

        Args:
            content: File content as bytes
            input_format: Input format
            output_format: Output format
            output_path: Path to output file (optional)
            **kwargs: Additional parameters

        Returns:
            ConversionResponse: Conversion result
        """
        request = ConversionRequest(
            input_format=input_format,
            output_format=output_format,
            input_content=content,
            output_path=output_path,
            **kwargs
        )

        return await self.convert(request)

    async def batch_convert(
        self,
        requests: List[ConversionRequest],
        **kwargs
    ) -> List[ConversionResponse]:
        """
        Convert multiple documents.

        Args:
            requests: List of conversion requests
            **kwargs: Additional parameters

        Returns:
            List[ConversionResponse]: Conversion results
        """
        if not self.capabilities.supports_batch_conversion:
            # Fall back to sequential processing
            results = []
            for request in requests:
                result = await self.convert(request, **kwargs)
                results.append(result)
            return results

        # Parallel processing
        import asyncio
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_request(request: ConversionRequest) -> ConversionResponse:
            async with semaphore:
                return await self.convert(request, **kwargs)

        tasks = [process_request(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _detect_format_from_path(self, file_path: str) -> DocumentFormat:
        """
        Detect format from file path.

        Args:
            file_path: File path

        Returns:
            DocumentFormat: Detected format
        """
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
        }
        return format_map.get(extension, DocumentFormat.UNKNOWN)

    def _suggest_output_format(self, input_format: DocumentFormat) -> DocumentFormat:
        """
        Suggest output format based on input format.

        Args:
            input_format: Input format

        Returns:
            DocumentFormat: Suggested output format
        """
        # Default conversion suggestions
        suggestions = {
            DocumentFormat.PDF: DocumentFormat.MD,
            DocumentFormat.DOCX: DocumentFormat.MD,
            DocumentFormat.DOC: DocumentFormat.MD,
            DocumentFormat.HTML: DocumentFormat.MD,
            DocumentFormat.TXT: DocumentFormat.MD,
            DocumentFormat.RTF: DocumentFormat.MD,
            DocumentFormat.ODT: DocumentFormat.MD,
            DocumentFormat.MD: DocumentFormat.HTML,
            DocumentFormat.JSON: DocumentFormat.MD,
            DocumentFormat.XML: DocumentFormat.MD,
        }
        return suggestions.get(input_format, DocumentFormat.MD)

    def supports_conversion(
        self,
        input_format: DocumentFormat,
        output_format: DocumentFormat
    ) -> bool:
        """
        Check if the converter supports a specific conversion.

        Args:
            input_format: Input format
            output_format: Output format

        Returns:
            bool: True if conversion is supported
        """
        for capability in self.capabilities.supported_conversions:
            if capability.input_format == input_format and capability.output_format == output_format:
                return capability.supported
        return False

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the converter supports a specific feature.

        Args:
            feature: Feature name (batch, streaming, async, metadata, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "batch_conversion": "supports_batch_conversion",
            "streaming": "supports_streaming",
            "async": "supports_async",
            "metadata_extraction": "supports_metadata_extraction",
            "image_extraction": "supports_image_extraction",
            "table_extraction": "supports_table_extraction",
            "template_customization": "supports_template_customization",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def validate_request(
        self,
        request: ConversionRequest
    ) -> bool:
        """
        Validate a conversion request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check if we have data to convert
        if not request.input_path and not request.input_content and not request.input_stream:
            raise ValueError("No input path, content, or stream provided")

        # Check conversion support
        if not self.supports_conversion(request.input_format, request.output_format):
            raise ValueError(f"Conversion from {request.input_format} to {request.output_format} is not supported")

        # Check file size
        if request.input_path:
            file_size = Path(request.input_path).stat().st_size if Path(request.input_path).exists() else 0
            max_size = self.capabilities.max_file_size_mb
            if max_size and file_size > max_size * 1024 * 1024:
                raise ValueError(f"File size {file_size} bytes exceeds maximum {max_size} MB")

        return True

    def get_converter_info(self) -> Dict[str, Any]:
        """
        Get information about the converter.

        Returns:
            Dict[str, Any]: Converter information
        """
        return {
            "provider": self.provider_name,
            "capabilities": self.capabilities.dict(),
            "config": {
                "timeout": self.timeout,
                "enable_caching": self.enable_caching,
                "cache_ttl": self.cache_ttl,
                "parallel_processing": self.parallel_processing,
                "max_workers": self.max_workers,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the converter.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test request
            test_content = b"# Test Document\n\nThis is a test markdown document."
            test_request = ConversionRequest(
                input_format=DocumentFormat.MD,
                output_format=DocumentFormat.HTML,
                input_content=test_content
            )

            start_time = time.time()
            response = await self.convert(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_conversion_success": response.success,
                "test_output_size": len(response.output_content) if response.output_content else 0,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "error": str(e)
            }

    async def get_supported_conversions(self) -> List[Dict[str, Any]]:
        """
        Get all supported conversions.

        Returns:
            List[Dict[str, Any]]: List of supported conversions
        """
        conversions = []
        for capability in self.capabilities.supported_conversions:
            if capability.supported:
                conversions.append({
                    "input_format": capability.input_format,
                    "output_format": capability.output_format,
                    "quality_score": capability.quality_score,
                    "speed_score": capability.speed_score,
                    "preserves_images": capability.preserves_images,
                    "preserves_tables": capability.preserves_tables,
                    "preserves_formatting": capability.preserves_formatting,
                })
        return conversions


class ConverterException(Exception):
    """Exception raised by document converters."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        input_format: str = None,
        output_format: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.input_format = input_format
        self.output_format = output_format
        self.error_code = error_code


class UnsupportedConversionException(ConverterException):
    """Exception raised when conversion is not supported."""
    pass


class ValidationException(ConverterException):
    """Exception raised when validation fails."""
    pass


class ProcessingException(ConverterException):
    """Exception raised when processing fails."""
    pass


class TimeoutException(ConverterException):
    """Exception raised when conversion times out."""
    pass