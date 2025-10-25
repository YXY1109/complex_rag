"""
RAG Service Interfaces

This module contains abstract interfaces for all AI service providers.
All concrete implementations must inherit from these base classes.
"""

from .base_models import (
    # Base classes and exceptions
    BaseInterface,
    BaseRequest,
    BaseResponse,
    ProviderCapabilities,
    ProviderConfig,
    ProviderType,
    ModelType,
    APIError,
    TokenUsage,

    # Exception classes
    ProviderException,
    RateLimitException,
    TokenLimitException,
    ModelUnavailableException,
    AuthenticationException,
    NetworkException,
    TimeoutException,
    ValidationException,
    ServerException,

    # Utility functions
    create_error_response,
    handle_provider_error,
)

from .llm_interface import (
    # LLM interfaces and models
    LLMInterface,
    LLMConfig,
    LLMProviderCapabilities,

    # Chat completion models
    ChatRole,
    ChatMessage,
    FunctionCall,
    ToolCall,
    ToolDefinition,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamChunk,

    # LLM exceptions
    LLMException,
)

from .embedding_interface import (
    # Embedding interfaces and models
    EmbeddingInterface,
    EmbeddingConfig,
    EmbeddingProviderCapabilities,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
    EmbeddingInputType,

    # Embedding exceptions
    EmbeddingException,
)

from .rerank_interface import (
    # Rerank interfaces and models
    RerankInterface,
    RerankConfig,
    RerankProviderCapabilities,
    RerankRequest,
    RerankResponse,
    RerankDocument,
    RerankUsage,

    # Rerank exceptions
    RerankException,
)

__all__ = [
    # Base models and exceptions
    "BaseInterface",
    "BaseRequest",
    "BaseResponse",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderType",
    "ModelType",
    "APIError",
    "TokenUsage",
    "ProviderException",
    "RateLimitException",
    "TokenLimitException",
    "ModelUnavailableException",
    "AuthenticationException",
    "NetworkException",
    "TimeoutException",
    "ValidationException",
    "ServerException",
    "create_error_response",
    "handle_provider_error",

    # LLM interfaces
    "LLMInterface",
    "LLMConfig",
    "LLMProviderCapabilities",
    "ChatRole",
    "ChatMessage",
    "FunctionCall",
    "ToolCall",
    "ToolDefinition",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionStreamChunk",
    "LLMException",

    # Embedding interfaces
    "EmbeddingInterface",
    "EmbeddingConfig",
    "EmbeddingProviderCapabilities",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    "EmbeddingInputType",
    "EmbeddingException",

    # Rerank interfaces
    "RerankInterface",
    "RerankConfig",
    "RerankProviderCapabilities",
    "RerankRequest",
    "RerankResponse",
    "RerankDocument",
    "RerankUsage",
    "RerankException",
]