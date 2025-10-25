"""
Rerank Interface Abstract Class

This module defines the abstract interface for rerank model providers.
All rerank implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class RerankRequest(BaseModel):
    """Rerank request model."""
    model: str
    query: str
    documents: List[str]
    top_k: Optional[int] = Field(default=None, ge=1)
    return_documents: bool = Field(default=True)
    return_text: bool = Field(default=True)
    max_chunks_per_doc: Optional[int] = Field(default=None, ge=1)
    overlap_tokens: Optional[int] = Field(default=None, ge=0)
    user: Optional[str] = None


class RerankDocument(BaseModel):
    """Reranked document model."""
    index: int
    relevance_score: float
    document: Optional[str] = None
    text: Optional[str] = None


class RerankResponse(BaseModel):
    """Rerank response model."""
    object: str = "rerank"
    model: str
    results: List[RerankDocument]
    usage: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    created: Optional[int] = None


class RerankUsage(BaseModel):
    """Rerank usage information."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: Optional[int] = None


class RerankProviderCapabilities(BaseModel):
    """Rerank provider capabilities."""
    supported_models: List[str]
    max_query_length: int
    max_document_length: int
    max_documents_per_request: int
    max_top_k: int
    supports_chunking: bool
    supports_overlap: bool
    supports_custom_top_k: bool
    supports_return_documents: bool
    supports_scoring_only: bool
    pricing_per_1k_tokens: Optional[float] = None
    currency: Optional[str] = None
    average_latency_ms: Optional[float] = None


class RerankConfig(BaseModel):
    """Rerank configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_after: int = Field(default=1, ge=0)
    default_top_k: int = Field(default=10, ge=1)
    default_return_documents: bool = True
    default_max_chunks_per_doc: Optional[int] = Field(default=None, ge=1)
    default_overlap_tokens: Optional[int] = Field(default=None, ge=0)


class RerankInterface(ABC):
    """
    Abstract interface for rerank model providers.

    This class defines the contract that all rerank implementations must follow.
    It provides a unified interface for different rerank providers while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: RerankConfig):
        """Initialize the rerank provider with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.model = config.model
        self.default_top_k = config.default_top_k
        self.default_return_documents = config.default_return_documents
        self.default_max_chunks_per_doc = config.default_max_chunks_per_doc
        self.default_overlap_tokens = config.default_overlap_tokens
        self._capabilities: Optional[RerankProviderCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> RerankProviderCapabilities:
        """
        Get the capabilities of this rerank provider.

        Returns:
            RerankProviderCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def rerank(
        self,
        request: RerankRequest,
        **kwargs
    ) -> RerankResponse:
        """
        Rerank documents based on query relevance.

        Args:
            request: Rerank request
            **kwargs: Additional provider-specific parameters

        Returns:
            RerankResponse: Reranked results

        Raises:
            RerankException: If the request fails
        """
        pass

    async def simple_rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        return_documents: Optional[bool] = None,
        **kwargs
    ) -> RerankResponse:
        """
        Simple rerank interface for common use cases.

        Args:
            query: Query string
            documents: List of documents to rerank
            model: Model to use (overrides config default)
            top_k: Number of top results to return
            return_documents: Whether to return document text
            **kwargs: Additional parameters

        Returns:
            RerankResponse: Reranked results
        """
        request = RerankRequest(
            model=model or self.model,
            query=query,
            documents=documents,
            top_k=top_k or self.default_top_k,
            return_documents=return_documents or self.default_return_documents,
            **kwargs
        )

        return await self.rerank(request)

    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[tuple[str, float]]:
        """
        Rerank documents and return list of (document, score) tuples.

        Args:
            query: Query string
            documents: List of documents to rerank
            model: Model to use
            top_k: Number of top results to return
            **kwargs: Additional parameters

        Returns:
            List[tuple[str, float]]: List of (document, score) tuples
        """
        response = await self.simple_rerank(
            query=query,
            documents=documents,
            model=model,
            top_k=top_k,
            return_documents=True,
            **kwargs
        )

        results = []
        for result in response.results:
            doc_text = result.document or result.text or ""
            results.append((doc_text, result.relevance_score))

        return results

    async def rerank_chunks(
        self,
        query: str,
        documents: List[str],
        max_chunks_per_doc: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """
        Rerank document chunks.

        Args:
            query: Query string
            documents: List of documents to chunk and rerank
            max_chunks_per_doc: Maximum chunks per document
            overlap_tokens: Overlap tokens between chunks
            model: Model to use
            top_k: Number of top results to return
            **kwargs: Additional parameters

        Returns:
            RerankResponse: Reranked chunk results
        """
        if not self.capabilities.supports_chunking:
            raise ValueError("Chunking is not supported by this provider")

        request = RerankRequest(
            model=model or self.model,
            query=query,
            documents=documents,
            top_k=top_k or self.default_top_k,
            max_chunks_per_doc=max_chunks_per_doc or self.default_max_chunks_per_doc,
            overlap_tokens=overlap_tokens or self.default_overlap_tokens,
            return_documents=True,
            **kwargs
        )

        return await self.rerank(request)

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
        query_tokens: int,
        document_tokens: int,
        model: Optional[str] = None
    ) -> Optional[float]:
        """
        Estimate cost for token usage.

        Args:
            query_tokens: Number of query tokens
            document_tokens: Number of document tokens
            model: Model to use for pricing

        Returns:
            Optional[float]: Estimated cost in currency, or None if pricing not available
        """
        capabilities = self.capabilities
        if capabilities.pricing_per_1k_tokens is None:
            return None

        total_tokens = query_tokens + document_tokens
        return (total_tokens / 1000) * capabilities.pricing_per_1k_tokens

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name (chunking, overlap, custom_top_k, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "chunking": "supports_chunking",
            "overlap": "supports_overlap",
            "custom_top_k": "supports_custom_top_k",
            "return_documents": "supports_return_documents",
            "scoring_only": "supports_scoring_only",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def validate_request(
        self,
        request: RerankRequest
    ) -> bool:
        """
        Validate a rerank request.

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

        # Check query length
        if len(request.query) > self.capabilities.max_query_length:
            raise ValueError(
                f"Query length {len(request.query)} exceeds max query length {self.capabilities.max_query_length}"
            )

        # Check document count
        if len(request.documents) > self.capabilities.max_documents_per_request:
            raise ValueError(
                f"Document count {len(request.documents)} exceeds max documents per request {self.capabilities.max_documents_per_request}"
            )

        # Check document lengths
        for i, doc in enumerate(request.documents):
            if len(doc) > self.capabilities.max_document_length:
                raise ValueError(
                    f"Document {i} length {len(doc)} exceeds max document length {self.capabilities.max_document_length}"
                )

        # Check top_k
        if request.top_k:
            if request.top_k > self.capabilities.max_top_k:
                raise ValueError(
                    f"top_k {request.top_k} exceeds max top_k {self.capabilities.max_top_k}"
                )

        # Check chunking support
        if request.max_chunks_per_doc and not self.capabilities.supports_chunking:
            raise ValueError("Chunking is not supported by this provider")

        # Check overlap support
        if request.overlap_tokens and not self.capabilities.supports_overlap:
            raise ValueError("Overlap is not supported by this provider")

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
            "capabilities": self.capabilities.dict(),
            "config": {
                "default_top_k": self.default_top_k,
                "default_return_documents": self.default_return_documents,
                "default_max_chunks_per_doc": self.default_max_chunks_per_doc,
                "default_overlap_tokens": self.default_overlap_tokens,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the rerank provider.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Simple test request
            test_request = RerankRequest(
                model=self.model,
                query="What is artificial intelligence?",
                documents=[
                    "Artificial intelligence is a branch of computer science.",
                    "Machine learning is a subset of AI.",
                    "Deep learning uses neural networks."
                ],
                top_k=3
            )

            response = await self.rerank(test_request)

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "response_time_ms": None,  # Could be measured in implementations
                "test_results_count": len(response.results),
                "test_top_score": response.results[0].relevance_score if response.results else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e)
            }

    async def merge_reranked_results(
        self,
        original_documents: List[str],
        reranked_results: RerankResponse,
        preserve_unranked: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Merge reranked results with original documents.

        Args:
            original_documents: Original list of documents
            reranked_results: Reranked results
            preserve_unranked: Whether to include unranked documents

        Returns:
            List[Dict[str, Any]]: Merged results with original indices and scores
        """
        # Create a mapping from document text to original index
        doc_to_index = {doc: i for i, doc in enumerate(original_documents)}

        merged_results = []

        # Add reranked documents
        for result in reranked_results.results:
            doc_text = result.document or result.text or ""
            original_index = doc_to_index.get(doc_text)

            merged_results.append({
                "original_index": original_index,
                "document": doc_text,
                "relevance_score": result.relevance_score,
                "ranked": True
            })

        # Add unranked documents if requested
        if preserve_unranked:
            ranked_texts = {result.document or result.text or "" for result in reranked_results.results}

            for i, doc in enumerate(original_documents):
                if doc not in ranked_texts:
                    merged_results.append({
                        "original_index": i,
                        "document": doc,
                        "relevance_score": 0.0,
                        "ranked": False
                    })

        return merged_results


class RerankException(Exception):
    """Exception raised by rerank providers."""

    def __init__(self, message: str, provider: str = None, model: str = None, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.error_code = error_code


class RateLimitException(RerankException):
    """Exception raised when rate limits are exceeded."""
    pass


class TokenLimitException(RerankException):
    """Exception raised when token limits are exceeded."""
    pass


class ModelUnavailableException(RerankException):
    """Exception raised when a model is unavailable."""
    pass