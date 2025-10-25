"""
LLM Interface Abstract Class

This module defines the abstract interface for Large Language Model (LLM) providers.
All LLM implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum


class ChatRole(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Chat message model."""
    role: ChatRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class FunctionCall(BaseModel):
    """Function call model."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call model."""
    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Tool definition model."""
    type: str = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage]
    model: str
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamChunk(BaseModel):
    """Chat completion stream chunk model."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[UsageInfo] = None


class LLMProviderCapabilities(BaseModel):
    """LLM provider capabilities."""
    supported_models: List[str]
    max_tokens: int
    max_context_length: int
    supports_streaming: bool
    supports_functions: bool
    supports_tools: bool
    supports_vision: bool
    supports_audio: bool
    supports_system_messages: bool
    supports_seed: bool
    supports_response_format: bool
    pricing_per_1k_tokens: Optional[float] = None
    currency: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_after: int = Field(default=1, ge=0)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: Optional[int] = None
    default_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    default_frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    default_presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class LLMInterface(ABC):
    """
    Abstract interface for Large Language Model (LLM) providers.

    This class defines the contract that all LLM implementations must follow.
    It provides a unified interface for different LLM providers (OpenAI, Anthropic,
    Ollama, etc.) while allowing provider-specific configurations and capabilities.
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLM provider with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.model = config.model
        self._capabilities: Optional[LLMProviderCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> LLMProviderCapabilities:
        """
        Get the capabilities of this LLM provider.

        Returns:
            LLMProviderCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion response.

        Args:
            request: Chat completion request
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatCompletionResponse: Generated response

        Raises:
            LLMException: If the request fails
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionStreamChunk, None]:
        """
        Generate a streaming chat completion response.

        Args:
            request: Chat completion request
            **kwargs: Additional provider-specific parameters

        Yields:
            ChatCompletionStreamChunk: Response chunks

        Raises:
            LLMException: If the request fails
        """
        pass

    async def simple_chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionStreamChunk, None]]:
        """
        Simple chat interface for common use cases.

        Args:
            messages: List of messages (ChatMessage objects or dicts)
            model: Model to use (overrides config default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Chat completion response or stream
        """
        # Convert dict messages to ChatMessage objects
        chat_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                chat_messages.append(ChatMessage(**msg))
            else:
                chat_messages.append(msg)

        request = ChatCompletionRequest(
            messages=chat_messages,
            model=model or self.model,
            temperature=temperature or self.config.default_temperature,
            max_tokens=max_tokens or self.config.default_max_tokens,
            stream=stream,
            **kwargs
        )

        if stream:
            return self.chat_completion_stream(request)
        else:
            return await self.chat_completion(request)

    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None
    ) -> int:
        """
        Count tokens in text for the specified model.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (defaults to configured model)

        Returns:
            int: Number of tokens
        """
        # Default implementation - override in provider implementations
        # Approximate token count (4 characters per token on average)
        return len(text) // 4

    async def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate cost for token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model to use for pricing

        Returns:
            Optional[float]: Estimated cost in currency, or None if pricing not available
        """
        capabilities = self.capabilities
        if capabilities.pricing_per_1k_tokens is None:
            return None

        total_tokens = prompt_tokens + completion_tokens
        return (total_tokens / 1000) * capabilities.pricing_per_1k_tokens

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name (streaming, functions, tools, vision, audio, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "streaming": "supports_streaming",
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

    async def validate_request(
        self,
        request: ChatCompletionRequest
    ) -> bool:
        """
        Validate a chat completion request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check model
        if request.model not in self.capabilities.supported_models:
            raise ValueError(f"Model {request.model} is not supported")

        # Check max tokens
        if request.max_tokens:
            total_input_tokens = await self._count_messages_tokens(request.messages)
            if total_input_tokens + request.max_tokens > self.capabilities.max_context_length:
                raise ValueError(
                    f"Request exceeds max context length: "
                    f"{total_input_tokens + request.max_tokens} > {self.capabilities.max_context_length}"
                )

        # Check streaming support
        if request.stream and not self.capabilities.supports_streaming:
            raise ValueError("Streaming is not supported by this provider")

        # Check tools support
        if request.tools and not self.capabilities.supports_tools:
            raise ValueError("Tools are not supported by this provider")

        # Check functions support (legacy)
        if any(msg.function_call for msg in request.messages) and not self.capabilities.supports_functions:
            raise ValueError("Function calls are not supported by this provider")

        return True

    async def _count_messages_tokens(self, messages: List[ChatMessage]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of messages

        Returns:
            int: Total number of tokens
        """
        total_tokens = 0
        for message in messages:
            total_tokens += await self.count_tokens(message.content)

            # Add tokens for role and other fields
            total_tokens += 10  # Approximate overhead per message

        return total_tokens

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
                "temperature": self.config.default_temperature,
                "max_tokens": self.config.default_max_tokens,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the LLM provider.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Simple test request
            test_request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=ChatRole.USER, content="Hello")
                ],
                model=self.model,
                max_tokens=10
            )

            response = await self.chat_completion(test_request)

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "response_time_ms": None,  # Could be measured in implementations
                "test_response": response.choices[0].message.content[:50] if response.choices else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e)
            }


class LLMException(Exception):
    """Exception raised by LLM providers."""

    def __init__(self, message: str, provider: str = None, model: str = None, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.error_code = error_code


class RateLimitException(LLMException):
    """Exception raised when rate limits are exceeded."""
    pass


class TokenLimitException(LLMException):
    """Exception raised when token limits are exceeded."""
    pass


class ModelUnavailableException(LLMException):
    """Exception raised when a model is unavailable."""
    pass