"""
Base Models for AI Services

This module contains common data models used across different AI service interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Generic
from pydantic import BaseModel, Field, validator
from enum import Enum
import time
import uuid


class ProviderType(str, Enum):
    """AI provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    QWEN = "qwen"
    BCE = "bce"
    LOCAL = "local"
    MOCK = "mock"


class ModelType(str, Enum):
    """Model types."""
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VISION = "vision"
    AUDIO = "audio"


class APIError(BaseModel):
    """API error model."""
    message: str
    type: str
    code: Optional[str] = None
    param: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class BaseRequest(BaseModel):
    """Base request model for all AI services."""
    provider: Optional[str] = None
    model: str
    user: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    timeout: Optional[int] = Field(default=None, ge=1)
    max_retries: Optional[int] = Field(default=None, ge=0)
    metadata: Optional[Dict[str, Any]] = None

    @validator('temperature')
    def validate_temperature(cls, v, values):
        """Validate temperature parameter."""
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v

    @validator('top_p')
    def validate_top_p(cls, v, values):
        """Validate top_p parameter."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Top_p must be between 0.0 and 1.0')
        return v


class BaseResponse(BaseModel):
    """Base response model for all AI services."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: str
    created: int = Field(default_factory=lambda: int(time.time()))
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    error: Optional[APIError] = None
    metadata: Optional[Dict[str, Any]] = None


class TokenUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @validator('total_tokens')
    def validate_total_tokens(cls, v, values):
        """Validate that total_tokens equals prompt_tokens + completion_tokens."""
        prompt = values.get('prompt_tokens', 0)
        completion = values.get('completion_tokens', 0)
        if v != prompt + completion:
            raise ValueError('total_tokens must equal prompt_tokens + completion_tokens')
        return v


class ProviderCapabilities(BaseModel):
    """Base provider capabilities."""
    supported_models: List[str]
    max_input_length: int
    max_output_length: int
    max_batch_size: int
    supports_streaming: bool
    supports_batch: bool
    supports_functions: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_system_messages: bool = True
    supports_seed: bool = False
    supports_response_format: bool = False
    pricing_per_1k_tokens: Optional[float] = None
    currency: Optional[str] = None
    average_latency_ms: Optional[float] = None
    provider_type: ProviderType


class ProviderConfig(BaseModel):
    """Base provider configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_after: int = Field(default=1, ge=0)
    enable_fallback: bool = False
    fallback_providers: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow additional fields for provider-specific config


T = TypeVar('T', bound=BaseRequest)
U = TypeVar('U', bound=BaseResponse)


class BaseInterface(ABC, Generic[T, U]):
    """
    Abstract base interface for all AI service providers.

    This class provides common functionality that all AI service interfaces
    should inherit from, including configuration management, validation,
    and error handling.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.model = config.model
        self._capabilities: Optional[ProviderCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Get the capabilities of this provider."""
        pass

    @abstractmethod
    async def _process_request(self, request: T, **kwargs) -> U:
        """Process the request and return response."""
        pass

    @abstractmethod
    def _validate_request(self, request: T) -> bool:
        """Validate the request."""
        pass

    async def process_request(self, request: T, **kwargs) -> U:
        """
        Process a request with validation and error handling.

        Args:
            request: Request to process
            **kwargs: Additional parameters

        Returns:
            Response object

        Raises:
            ProviderException: If processing fails
        """
        try:
            # Validate request
            self._validate_request(request)

            # Process request
            response = await self._process_request(request, **kwargs)

            # Add provider info to response
            response.provider = self.provider_name
            response.model = self.model

            return response

        except Exception as e:
            # Convert to provider exception
            raise ProviderException(
                message=str(e),
                provider=self.provider_name,
                model=self.model,
                error_code=getattr(e, 'error_code', None)
            ) from e

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "streaming": "supports_streaming",
            "batch": "supports_batch",
            "functions": "supports_functions",
            "tools": "supports_tools",
            "vision": "supports_vision",
            "audio": "supports_audio",
            "system_messages": "supports_system_messages",
            "seed": "supports_seed",
            "response_format": "supports_response_format",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model: Model name (defaults to configured model)

        Returns:
            Dict[str, Any]: Model information
        """
        model_name = model or self.model

        return {
            "provider": self.provider_name,
            "model": model_name,
            "capabilities": self.capabilities.dict(),
            "config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "enable_fallback": self.config.enable_fallback,
                "fallback_providers": self.config.fallback_providers,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test request
            test_request = self._create_test_request()

            # Process the test request
            start_time = time.time()
            response = await self.process_request(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "response_time_ms": int((end_time - start_time) * 1000),
                "test_response_id": response.id if response else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e)
            }

    @abstractmethod
    def _create_test_request(self) -> T:
        """Create a test request for health checking."""
        pass


class ProviderException(Exception):
    """Base exception for AI providers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        model: str = None,
        error_code: str = None,
        request_id: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.error_code = error_code
        self.request_id = request_id


class RateLimitException(ProviderException):
    """Exception raised when rate limits are exceeded."""
    pass


class TokenLimitException(ProviderException):
    """Exception raised when token limits are exceeded."""
    pass


class ModelUnavailableException(ProviderException):
    """Exception raised when a model is unavailable."""
    pass


class AuthenticationException(ProviderException):
    """Exception raised when authentication fails."""
    pass


class NetworkException(ProviderException):
    """Exception raised when network errors occur."""
    pass


class TimeoutException(ProviderException):
    """Exception raised when requests timeout."""
    pass


class ValidationException(ProviderException):
    """Exception raised when request validation fails."""
    pass


class ServerException(ProviderException):
    """Exception raised when server errors occur."""
    pass


def create_error_response(
    provider: str,
    model: str,
    error_message: str,
    error_type: str = "server_error",
    error_code: Optional[str] = None,
    object_type: str = "error"
) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        provider: Provider name
        model: Model name
        error_message: Error message
        error_type: Type of error
        error_code: Error code
        object_type: Object type for response

    Returns:
        Dict[str, Any]: Error response
    """
    return {
        "id": str(uuid.uuid4()),
        "object": object_type,
        "created": int(time.time()),
        "provider": provider,
        "model": model,
        "error": {
            "message": error_message,
            "type": error_type,
            "code": error_code,
            "provider": provider,
            "model": model,
            "timestamp": int(time.time())
        }
    }


def handle_provider_error(error: Exception, provider: str, model: str) -> ProviderException:
    """
    Convert different error types to ProviderException.

    Args:
        error: Original error
        provider: Provider name
        model: Model name

    Returns:
        ProviderException: Converted exception
    """
    error_message = str(error)
    error_code = getattr(error, 'error_code', None)

    if isinstance(error, ProviderException):
        return error

    # Check for specific error patterns
    error_message_lower = error_message.lower()

    if "rate limit" in error_message_lower or "too many requests" in error_message_lower:
        return RateLimitException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "token" in error_message_lower and ("limit" in error_message_lower or "exceed" in error_message_lower):
        return TokenLimitException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "model" in error_message_lower and ("not found" in error_message_lower or "unavailable" in error_message_lower):
        return ModelUnavailableException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "auth" in error_message_lower or "unauthorized" in error_message_lower:
        return AuthenticationException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "timeout" in error_message_lower or "timed out" in error_message_lower:
        return TimeoutException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "network" in error_message_lower or "connection" in error_message_lower:
        return NetworkException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "validation" in error_message_lower or "invalid" in error_message_lower:
        return ValidationException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    if "server" in error_message_lower or "internal" in error_message_lower:
        return ServerException(
            message=error_message,
            provider=provider,
            model=model,
            error_code=error_code
        )

    # Default to generic provider exception
    return ProviderException(
        message=error_message,
        provider=provider,
        model=model,
        error_code=error_code
    )