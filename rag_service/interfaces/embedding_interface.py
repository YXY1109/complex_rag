"""
Embedding Interface Abstract Class

This module defines the abstract interface for embedding model providers.
All embedding implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingInputType(str, Enum):
    """Embedding input types."""
    DOCUMENT = "document"
    QUERY = "query"
    CODE = "code"


class EmbeddingUsage(BaseModel):
    """Embedding usage information."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    """Single embedding data."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingRequest(BaseModel):
    """Embedding request model."""
    input: Union[str, List[str]]
    model: str
    encoding_format: str = Field(default="float", regex="^(float|base64)$")
    dimensions: Optional[int] = Field(default=None, ge=1)
    user: Optional[str] = None
    input_type: Optional[EmbeddingInputType] = None


class EmbeddingResponse(BaseModel):
    """Embedding response model."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class EmbeddingProviderCapabilities(BaseModel):
    """Embedding provider capabilities."""
    supported_models: List[str]
    max_input_length: int
    max_batch_size: int
    embedding_dimensions: Dict[str, int]  # model -> dimensions
    supports_batch: bool
    supports_different_input_types: bool
    supports_custom_dimensions: bool
    pricing_per_1k_tokens: Optional[float] = None
    currency: Optional[str] = None
    average_latency_ms: Optional[float] = None


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_after: int = Field(default=1, ge=0)
    batch_size: int = Field(default=100, ge=1, le=1000)
    normalize_embeddings: bool = Field(default=True)
    default_encoding_format: str = Field(default="float", regex="^(float|base64)$")


class EmbeddingInterface(ABC):
    """
    Abstract interface for embedding model providers.

    This class defines the contract that all embedding implementations must follow.
    It provides a unified interface for different embedding providers while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding provider with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.model = config.model
        self.batch_size = config.batch_size
        self.normalize_embeddings = config.normalize_embeddings
        self._capabilities: Optional[EmbeddingProviderCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> EmbeddingProviderCapabilities:
        """
        Get the capabilities of this embedding provider.

        Returns:
            EmbeddingProviderCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def create_embedding(
        self,
        request: EmbeddingRequest,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings for the given input.

        Args:
            request: Embedding request
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingResponse: Generated embeddings

        Raises:
            EmbeddingException: If the request fails
        """
        pass

    async def embed_text(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        input_type: Optional[EmbeddingInputType] = None,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs
    ) -> Union[List[List[float]], List[float]]:
        """
        Simple embedding interface for text.

        Args:
            text: Single text or list of texts to embed
            model: Model to use (overrides config default)
            input_type: Type of input (document, query, code)
            encoding_format: Format of embeddings (float, base64)
            dimensions: Custom dimensions (if supported)
            **kwargs: Additional parameters

        Returns:
            Union[List[List[float]], List[float]]: Embeddings
        """
        request = EmbeddingRequest(
            input=text,
            model=model or self.model,
            encoding_format=encoding_format or self.config.default_encoding_format,
            input_type=input_type,
            dimensions=dimensions,
            **kwargs
        )

        response = await self.create_embedding(request)

        # Extract embeddings from response
        embeddings = [data.embedding for data in response.data]

        # Return single embedding if input was a single string
        if isinstance(text, str):
            return embeddings[0] if embeddings else []

        return embeddings

    async def embed_texts_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: Optional[EmbeddingInputType] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            model: Model to use
            input_type: Type of input
            batch_size: Batch size (overrides config default)
            **kwargs: Additional parameters

        Returns:
            List[List[float]]: List of embeddings
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        max_batch_size = self.capabilities.max_batch_size
        batch_size = min(batch_size, max_batch_size)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                request = EmbeddingRequest(
                    input=batch_texts,
                    model=model or self.model,
                    input_type=input_type,
                    **kwargs
                )

                response = await self.create_embedding(request)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                # Fallback to individual processing if batch fails
                for text in batch_texts:
                    try:
                        embedding = await self.embed_text(
                            text=text,
                            model=model,
                            input_type=input_type,
                            **kwargs
                        )
                        all_embeddings.append(embedding)
                    except Exception as individual_error:
                        raise EmbeddingException(
                            f"Failed to embed text '{text[:50]}...': {individual_error}"
                        ) from individual_error

        return all_embeddings

    async def count_tokens(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Union[int, List[int]]:
        """
        Count tokens in text for the specified model.

        Args:
            text: Text or list of texts to count tokens for
            model: Model to use for tokenization

        Returns:
            Union[int, List[int]]: Number of tokens
        """
        # Default implementation - override in provider implementations
        if isinstance(text, str):
            return len(text) // 4  # Approximate
        else:
            return [len(t) // 4 for t in text]

    async def estimate_cost(
        self,
        input_tokens: int,
        model: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            model: Model to use for pricing

        Returns:
            Optional[float]: Estimated cost in currency, or None if pricing not available
        """
        capabilities = self.capabilities
        if capabilities.pricing_per_1k_tokens is None:
            return None

        return (input_tokens / 1000) * capabilities.pricing_per_1k_tokens

    def get_embedding_dimensions(self, model: Optional[str] = None) -> int:
        """
        Get embedding dimensions for a model.

        Args:
            model: Model name (defaults to configured model)

        Returns:
            int: Embedding dimensions
        """
        model_name = model or self.model
        dimensions_map = self.capabilities.embedding_dimensions

        if model_name not in dimensions_map:
            raise ValueError(f"Unknown model: {model_name}")

        return dimensions_map[model_name]

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name (batch, input_types, custom_dimensions, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "batch": "supports_batch",
            "input_types": "supports_different_input_types",
            "custom_dimensions": "supports_custom_dimensions",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def validate_request(
        self,
        request: EmbeddingRequest
    ) -> bool:
        """
        Validate an embedding request.

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

        # Check custom dimensions
        if request.dimensions and not self.capabilities.supports_custom_dimensions:
            raise ValueError("Custom dimensions are not supported by this provider")

        # Check batch size
        if isinstance(request.input, list):
            if len(request.input) > self.capabilities.max_batch_size:
                raise ValueError(
                    f"Batch size {len(request.input)} exceeds max batch size {self.capabilities.max_batch_size}"
                )

            # Check input length
            total_chars = sum(len(text) for text in request.input)
            if total_chars > self.capabilities.max_input_length:
                raise ValueError(
                    f"Input length {total_chars} exceeds max input length {self.capabilities.max_input_length}"
                )
        else:
            if len(request.input) > self.capabilities.max_input_length:
                raise ValueError(
                    f"Input length {len(request.input)} exceeds max input length {self.capabilities.max_input_length}"
                )

        # Check input type support
        if request.input_type and not self.capabilities.supports_different_input_types:
            raise ValueError("Different input types are not supported by this provider")

        return True

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
            "dimensions": self.get_embedding_dimensions(model_name),
            "capabilities": self.capabilities.dict(),
            "config": {
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize_embeddings,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the embedding provider.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Simple test request
            test_request = EmbeddingRequest(
                input="Hello world",
                model=self.model
            )

            response = await self.create_embedding(test_request)

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "response_time_ms": None,  # Could be measured in implementations
                "test_embedding_dimensions": len(response.data[0].embedding) if response.data else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e)
            }

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, dot_product, euclidean)

        Returns:
            float: Similarity score
        """
        import math

        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimensions")

        if metric == "cosine":
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = math.sqrt(sum(a * a for a in embedding1))
            norm2 = math.sqrt(sum(b * b for b in embedding2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        elif metric == "dot_product":
            return sum(a * b for a, b in zip(embedding1, embedding2))

        elif metric == "euclidean":
            # Convert Euclidean distance to similarity
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)))
            return 1.0 / (1.0 + distance)  # Convert to similarity score

        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")


class EmbeddingException(Exception):
    """Exception raised by embedding providers."""

    def __init__(self, message: str, provider: str = None, model: str = None, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.error_code = error_code


class RateLimitException(EmbeddingException):
    """Exception raised when rate limits are exceeded."""
    pass


class TokenLimitException(EmbeddingException):
    """Exception raised when token limits are exceeded."""
    pass


class ModelUnavailableException(EmbeddingException):
    """Exception raised when a model is unavailable."""
    pass