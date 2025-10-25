"""
File Source Handler Interface Abstract Class

This module defines the abstract interface for file source handlers.
All source handler implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from .parser_interface import DocumentFormat, ParseRequest, ParseResponse


class FileSource(str, Enum):
    """File source types."""
    WEB_DOCUMENTS = "web_documents"
    OFFICE_DOCUMENTS = "office_documents"
    SCANNED_DOCUMENTS = "scanned_documents"
    STRUCTURED_DATA = "structured_data"
    CODE_REPOSITORIES = "code_repositories"
    CUSTOM_SOURCES = "custom_sources"
    UNKNOWN = "unknown"


class ProcessingStrategy(str, Enum):
    """Processing strategy types."""
    FAST = "fast"  # Quick processing, lower accuracy
    BALANCED = "balanced"  # Balance between speed and accuracy
    ACCURATE = "accurate"  # High accuracy, slower processing
    AUTO = "auto"  # Auto-select based on file characteristics


class SourceDetectionResult(BaseModel):
    """File source detection result."""
    source: FileSource
    confidence: float = Field(ge=0.0, le=1.0)
    detected_features: List[str] = Field(default_factory=list)
    processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    metadata: Optional[Dict[str, Any]] = None
    detection_time_ms: Optional[float] = None


class SourceConfig(BaseModel):
    """Source handler configuration."""
    provider: str
    enabled: bool = True
    priority: int = Field(default=100, ge=1)  # Higher priority = checked first
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_file_size_mb: Optional[int] = None
    timeout: int = Field(default=300, ge=1)
    max_retries: int = Field(default=3, ge=0)
    custom_options: Optional[Dict[str, Any]] = None


class SourceRequest(BaseModel):
    """Source processing request model."""
    file_path: Optional[str] = None
    file_content: Optional[bytes] = None
    file_stream: Optional[Any] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    force_source: Optional[FileSource] = None
    processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    custom_options: Optional[Dict[str, Any]] = None


class SourceResponse(BaseModel):
    """Source processing response model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: FileSource
    success: bool
    detection_result: Optional[SourceDetectionResult] = None
    processed_content: Optional[bytes] = None
    extracted_metadata: Optional[Dict[str, Any]] = None
    parse_request: Optional[ParseRequest] = None
    parse_response: Optional[ParseResponse] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    source_info: Optional[Dict[str, Any]] = None


class SourceHandlerCapabilities(BaseModel):
    """Source handler capabilities model."""
    source_type: FileSource
    supported_formats: List[DocumentFormat]
    supported_urls: List[str]  # URL schemes, domains, etc.
    max_file_size_mb: Optional[int] = None
    supports_remote: bool
    supports_streaming: bool
    supports_metadata_extraction: bool
    supports_content_transformation: bool
    supports_ocr: bool
    supports_async: bool
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    requires_internet: bool = False
    gpu_accelerated: bool = False


class SourceHandlerInterface(ABC):
    """
    Abstract interface for file source handlers.

    This class defines the contract that all source handler implementations must follow.
    It provides a unified interface for different file source handlers while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: SourceConfig):
        """Initialize the source handler with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.enabled = config.enabled
        self.priority = config.priority
        self.confidence_threshold = config.confidence_threshold
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._capabilities: Optional[SourceHandlerCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> SourceHandlerCapabilities:
        """
        Get the capabilities of this source handler.

        Returns:
            SourceHandlerCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def can_handle(
        self,
        request: SourceRequest,
        **kwargs
    ) -> float:
        """
        Check if this handler can handle the given request.

        Args:
            request: Source request
            **kwargs: Additional parameters

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    async def process_source(
        self,
        request: SourceRequest,
        **kwargs
    ) -> SourceResponse:
        """
        Process a file source.

        Args:
            request: Source processing request
            **kwargs: Additional provider-specific parameters

        Returns:
            SourceResponse: Processing result

        Raises:
            SourceHandlerException: If processing fails
        """
        pass

    @abstractmethod
    async def extract_metadata(
        self,
        request: SourceRequest,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from the source.

        Args:
            request: Source request
            **kwargs: Additional parameters

        Returns:
            Optional[Dict[str, Any]]: Extracted metadata
        """
        pass

    @abstractmethod
    async def transform_content(
        self,
        request: SourceRequest,
        **kwargs
    ) -> Optional[bytes]:
        """
        Transform content for processing.

        Args:
            request: Source request
            **kwargs: Additional parameters

        Returns:
            Optional[bytes]: Transformed content
        """
        pass

    async def detect_source(
        self,
        request: SourceRequest,
        **kwargs
    ) -> SourceDetectionResult:
        """
        Detect the source type of a file.

        Args:
            request: Source request
            **kwargs: Additional parameters

        Returns:
            SourceDetectionResult: Detection result
        """
        start_time = time.time()

        # Check confidence threshold
        confidence = await self.can_handle(request, **kwargs)
        if confidence < self.confidence_threshold:
            return SourceDetectionResult(
                source=FileSource.UNKNOWN,
                confidence=0.0,
                detection_time_ms=(time.time() - start_time) * 1000
            )

        # Determine processing strategy
        strategy = self._determine_processing_strategy(request)

        # Extract features for detection
        features = await self._extract_detection_features(request, **kwargs)

        return SourceDetectionResult(
            source=self.capabilities.source_type,
            confidence=confidence,
            detected_features=features,
            processing_strategy=strategy,
            detection_time_ms=(time.time() - start_time) * 1000
        )

    async def process_file(
        self,
        file_path: str,
        force_source: Optional[FileSource] = None,
        processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO,
        **kwargs
    ) -> SourceResponse:
        """
        Process a file from path.

        Args:
            file_path: Path to file
            force_source: Force specific source type
            processing_strategy: Processing strategy
            **kwargs: Additional parameters

        Returns:
            SourceResponse: Processing result
        """
        request = SourceRequest(
            file_path=file_path,
            force_source=force_source,
            processing_strategy=processing_strategy,
            **kwargs
        )

        return await self.process_source(request)

    async def process_url(
        self,
        url: str,
        force_source: Optional[FileSource] = None,
        processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO,
        **kwargs
    ) -> SourceResponse:
        """
        Process a file from URL.

        Args:
            url: URL to process
            force_source: Force specific source type
            processing_strategy: Processing strategy
            **kwargs: Additional parameters

        Returns:
            SourceResponse: Processing result
        """
        request = SourceRequest(
            url=url,
            force_source=force_source,
            processing_strategy=processing_strategy,
            **kwargs
        )

        return await self.process_source(request)

    async def process_content(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        force_source: Optional[FileSource] = None,
        processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO,
        **kwargs
    ) -> SourceResponse:
        """
        Process file content.

        Args:
            content: File content as bytes
            metadata: File metadata
            force_source: Force specific source type
            processing_strategy: Processing strategy
            **kwargs: Additional parameters

        Returns:
            SourceResponse: Processing result
        """
        request = SourceRequest(
            file_content=content,
            metadata=metadata,
            force_source=force_source,
            processing_strategy=processing_strategy,
            **kwargs
        )

        return await self.process_source(request)

    def _determine_processing_strategy(
        self,
        request: SourceRequest
    ) -> ProcessingStrategy:
        """
        Determine the best processing strategy.

        Args:
            request: Source request

        Returns:
            ProcessingStrategy: Recommended strategy
        """
        if request.processing_strategy != ProcessingStrategy.AUTO:
            return request.processing_strategy

        # Auto-determine based on file characteristics
        if request.file_path:
            file_size = Path(request.file_path).stat().st_size if Path(request.file_path).exists() else 0
        elif request.file_content:
            file_size = len(request.file_content)
        else:
            file_size = 0

        max_size = self.capabilities.max_file_size_mb
        if max_size and file_size > max_size * 1024 * 1024 * 0.8:
            return ProcessingStrategy.FAST
        elif file_size < 1024 * 1024:  # Less than 1MB
            return ProcessingStrategy.ACCURATE
        else:
            return ProcessingStrategy.BALANCED

    async def _extract_detection_features(
        self,
        request: SourceRequest,
        **kwargs
    ) -> List[str]:
        """
        Extract features for source detection.

        Args:
            request: Source request
            **kwargs: Additional parameters

        Returns:
            List[str]: List of detected features
        """
        features = []

        # Add file-based features
        if request.file_path:
            path = Path(request.file_path)
            features.append(f"extension:{path.suffix}")
            features.append(f"filename:{path.name}")
            features.append(f"directory:{path.parent.name}")

        # Add URL-based features
        if request.url:
            parsed = urlparse(request.url)
            features.append(f"scheme:{parsed.scheme}")
            features.append(f"domain:{parsed.netloc}")
            features.append(f"path:{parsed.path}")

        # Add content-based features
        if request.file_content:
            content = request.file_content[:1024]  # First 1KB
            features.append(f"content_size:{len(request.file_content)}")

            # Try to detect text encoding
            try:
                content.decode('utf-8')
                features.append("encoding:utf-8")
            except UnicodeDecodeError:
                try:
                    content.decode('latin-1')
                    features.append("encoding:latin-1")
                except UnicodeDecodeError:
                    features.append("encoding:binary")

        # Add metadata features
        if request.metadata:
            for key, value in request.metadata.items():
                features.append(f"metadata:{key}:{value}")

        return features

    def supports_format(self, file_format: DocumentFormat) -> bool:
        """
        Check if the handler supports a specific format.

        Args:
            file_format: Document format to check

        Returns:
            bool: True if format is supported
        """
        return file_format in self.capabilities.supported_formats

    def supports_url(self, url: str) -> bool:
        """
        Check if the handler supports a specific URL.

        Args:
            url: URL to check

        Returns:
            bool: True if URL is supported
        """
        parsed = urlparse(url)
        return parsed.scheme in self.capabilities.supported_urls

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the handler supports a specific feature.

        Args:
            feature: Feature name (remote, streaming, async, metadata, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "remote": "supports_remote",
            "streaming": "supports_streaming",
            "async": "supports_async",
            "metadata_extraction": "supports_metadata_extraction",
            "content_transformation": "supports_content_transformation",
            "ocr": "supports_ocr",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def validate_request(
        self,
        request: SourceRequest
    ) -> bool:
        """
        Validate a source request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check if we have data to process
        if not request.file_path and not request.file_content and not request.file_stream and not request.url:
            raise ValueError("No file path, content, stream, or URL provided")

        # Check file size
        if request.file_path:
            file_size = Path(request.file_path).stat().st_size if Path(request.file_path).exists() else 0
            max_size = self.capabilities.max_file_size_mb
            if max_size and file_size > max_size * 1024 * 1024:
                raise ValueError(f"File size {file_size} bytes exceeds maximum {max_size} MB")

        if request.file_content:
            file_size = len(request.file_content)
            max_size = self.capabilities.max_file_size_mb
            if max_size and file_size > max_size * 1024 * 1024:
                raise ValueError(f"Content size {file_size} bytes exceeds maximum {max_size} MB")

        return True

    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about the source handler.

        Returns:
            Dict[str, Any]: Handler information
        """
        return {
            "provider": self.provider_name,
            "source_type": self.capabilities.source_type,
            "capabilities": self.capabilities.dict(),
            "config": {
                "enabled": self.enabled,
                "priority": self.priority,
                "confidence_threshold": self.confidence_threshold,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the source handler.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test request
            test_content = b"# Test Document\n\nThis is a test document."
            test_request = SourceRequest(
                file_content=test_content,
                metadata={"filename": "test.md"}
            )

            start_time = time.time()
            confidence = await self.can_handle(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "source_type": self.capabilities.source_type,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_confidence": confidence,
                "enabled": self.enabled,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "source_type": self.capabilities.source_type,
                "error": str(e)
            }


class SourceHandlerException(Exception):
    """Exception raised by source handlers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        source_type: str = None,
        file_path: str = None,
        url: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.source_type = source_type
        self.file_path = file_path
        self.url = url
        self.error_code = error_code


class UnsupportedSourceException(SourceHandlerException):
    """Exception raised when source is not supported."""
    pass


class ValidationException(SourceHandlerException):
    """Exception raised when validation fails."""
    pass


class ProcessingException(SourceHandlerException):
    """Exception raised when processing fails."""
    pass


class TimeoutException(SourceHandlerException):
    """Exception raised when processing times out."""
    pass


class NetworkException(SourceHandlerException):
    """Exception raised when network operations fail."""
    pass