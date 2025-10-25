"""
Keyword Retriever Interface Abstract Class

This module defines the abstract interface for keyword-based retrievers.
All keyword retriever implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
import re
from dataclasses import dataclass


class QueryType(str, Enum):
    """Query types for keyword search."""
    TERM = "term"
    PHRASE = "phrase"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    REGEX = "regex"
    BOOLEAN = "boolean"


class MatchType(str, Enum):
    """Match types for keyword search."""
    EXACT = "exact"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    FUZZY = "fuzzy"
    REGEX = "regex"


class BooleanOperator(str, Enum):
    """Boolean operators."""
    AND = "and"
    OR = "or"
    NOT = "not"
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"


class TokenizerType(str, Enum):
    """Tokenizer types."""
    WHITESPACE = "whitespace"
    STANDARD = "standard"
    KEYWORD = "keyword"
    NGRAM = "ngram"
    EDGE_NGRAM = "edge_ngram"
    CUSTOM = "custom"


class RankingMode(str, Enum):
    """Ranking modes."""
    TF_IDF = "tf_idf"
    BM25 = "bm25"
    TF = "tf"
    IDF = "idf"
    CUSTOM = "custom"


@dataclass
class KeywordMatch:
    """Keyword match information."""
    text: str
    field: str
    start_position: int
    end_position: int
    score: float
    match_type: MatchType
    term_frequency: int = 1


@dataclass
class KeywordResult:
    """Keyword search result."""
    id: str
    content: str
    score: float
    matches: List[KeywordMatch] = None
    highlights: Dict[str, List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    term_freq_map: Optional[Dict[str, int]] = None
    field_scores: Optional[Dict[str, float]] = None


class KeywordRetrieverConfig(BaseModel):
    """Keyword retriever configuration."""
    provider: str
    index_name: str
    tokenizer_type: TokenizerType = TokenizerType.STANDARD
    analyzer_type: Optional[str] = None  # "english", "chinese", "standard", etc.
    ranking_mode: RankingMode = RankingMode.BM25
    min_term_length: int = Field(default=2, ge=1)
    max_term_length: int = Field(default=50, ge=1)
    enable_stemming: bool = True
    enable_stop_words: bool = True
    enable_synonyms: bool = False
    enable_phonetic: bool = False
    fuzzy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    highlight_enabled: bool = True
    highlight_tags: Tuple[str, str] = ("<em>", "</em>")
    snippet_size: int = Field(default=150, ge=50)
    max_snippets: int = Field(default=3, ge=1)
    cache_enabled: bool = True
    cache_size: int = Field(default=1000, ge=1)
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    debug_mode: bool = False
    custom_stop_words: Optional[List[str]] = None
    custom_synonyms: Optional[Dict[str, List[str]]] = None
    custom_options: Optional[Dict[str, Any]] = None


class KeywordQuery(BaseModel):
    """Keyword query model."""
    query: str
    query_type: QueryType = QueryType.TERM
    field: Optional[str] = None
    fields: Optional[List[str]] = None
    boost: float = Field(default=1.0, ge=0.0)
    match_type: MatchType = MatchType.CONTAINS
    fuzzy_distance: Optional[int] = Field(default=None, ge=1, le=2)
    analyzer: Optional[str] = None
    minimum_should_match: Optional[str] = None  # "75%", "3<60%", etc.


class BooleanClause(BaseModel):
    """Boolean clause model."""
    query: KeywordQuery
    operator: BooleanOperator
    boost: float = Field(default=1.0, ge=0.0)


class BooleanQuery(BaseModel):
    """Boolean query model."""
    clauses: List[BooleanClause]
    minimum_should_match: Optional[str] = None


class KeywordSearchRequest(BaseModel):
    """Keyword search request."""
    query: Union[str, KeywordQuery, BooleanQuery]
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[Dict[str, Any]] = None
    include_matches: bool = False
    include_highlights: bool = True
    include_term_freq: bool = False
    include_field_scores: bool = False
    fields: Optional[List[str]] = None  # Fields to search in
    boost_fields: Optional[Dict[str, float]] = None  # Field boost factors
    minimum_score: Optional[float] = Field(default=None, ge=0.0)
    search_params: Optional[Dict[str, Any]] = None
    custom_options: Optional[Dict[str, Any]] = None


class KeywordSearchResponse(BaseModel):
    """Keyword search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: Optional[str] = None
    results: List[KeywordResult] = Field(default_factory=list)
    total_results: int = 0
    total_hits: Optional[int] = None  # Total matching documents
    search_time_ms: Optional[float] = None
    query_info: Optional[Dict[str, Any]] = None
    search_params_used: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class BatchKeywordSearchRequest(BaseModel):
    """Batch keyword search request."""
    queries: List[Union[str, KeywordQuery, BooleanQuery]]
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[Dict[str, Any]] = None
    include_matches: bool = False
    include_highlights: bool = True
    fields: Optional[List[str]] = None
    boost_fields: Optional[Dict[str, float]] = None
    parallel_search: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)
    search_params: Optional[Dict[str, Any]] = None


class BatchKeywordSearchResponse(BaseModel):
    """Batch keyword search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    responses: List[KeywordSearchResponse] = Field(default_factory=list)
    total_queries: int = 0
    total_results: int = 0
    total_time_ms: Optional[float] = None
    parallel_used: bool = False
    errors: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class KeywordRetrieverCapabilities(BaseModel):
    """Keyword retriever capabilities."""
    provider: str
    supported_query_types: List[QueryType]
    supported_match_types: List[MatchType]
    supported_tokenizers: List[TokenizerType]
    supported_ranking_modes: List[RankingMode]
    supports_boolean_queries: bool
    supports_fuzzy_search: bool
    supports_wildcard_search: bool
    supports_regex_search: bool
    supports_phrase_search: bool
    supports_synonyms: bool
    supports_phonetic_search: bool
    supports_stemming: bool
    supports_highlighting: bool
    supports_snippets: bool
    supports_batch_search: bool
    supports_streaming: bool
    supports_async: bool
    supports_filtering: bool
    supports_field_boosting: bool
    supports_custom_analyzers: bool
    max_query_length: Optional[int] = None
    max_terms_per_query: Optional[int] = None
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class KeywordIndexStats(BaseModel):
    """Keyword index statistics."""
    index_name: str
    total_documents: int
    total_terms: int
    total_field_data: Optional[int] = None
    index_size_mb: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    last_updated: Optional[float] = None
    build_time_ms: Optional[float] = None
    average_query_time_ms: Optional[float] = None
    queries_per_second: Optional[float] = None
    unique_terms: Optional[int] = None
    analyzer_info: Optional[Dict[str, Any]] = None


class KeywordRetrieverInterface(ABC):
    """
    Abstract interface for keyword-based retrievers.

    This class defines the contract that all keyword retriever implementations must follow.
    It provides a unified interface for keyword search while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: KeywordRetrieverConfig):
        """Initialize the keyword retriever with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.index_name = config.index_name
        self.tokenizer_type = config.tokenizer_type
        self.ranking_mode = config.ranking_mode
        self.enable_stemming = config.enable_stemming
        self.enable_stop_words = config.enable_stop_words
        self.enable_synonyms = config.enable_synonyms
        self.enable_phonetic = config.enable_phonetic
        self.fuzzy_threshold = config.fuzzy_threshold
        self.highlight_enabled = config.highlight_enabled
        self.highlight_tags = config.highlight_tags
        self.snippet_size = config.snippet_size
        self.max_snippets = config.max_snippets
        self.cache_enabled = config.cache_enabled
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._capabilities: Optional[KeywordRetrieverCapabilities] = None
        self._index_stats: Optional[KeywordIndexStats] = None
        self._stop_words: Optional[Set[str]] = None
        self._synonyms: Optional[Dict[str, List[str]]] = None

    @property
    @abstractmethod
    def capabilities(self) -> KeywordRetrieverCapabilities:
        """
        Get the capabilities of this keyword retriever.

        Returns:
            KeywordRetrieverCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def initialize_index(self) -> bool:
        """
        Initialize the keyword index.

        Returns:
            bool: True if initialization successful

        Raises:
            KeywordRetrieverException: If initialization fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        request: KeywordSearchRequest,
        **kwargs
    ) -> KeywordSearchResponse:
        """
        Search for documents matching keywords.

        Args:
            request: Keyword search request
            **kwargs: Additional provider-specific parameters

        Returns:
            KeywordSearchResponse: Search results

        Raises:
            KeywordRetrieverException: If search fails
        """
        pass

    async def search_keywords(
        self,
        query: str,
        top_k: int = 10,
        fields: Optional[List[str]] = None,
        minimum_score: Optional[float] = None,
        **kwargs
    ) -> KeywordSearchResponse:
        """
        Search for documents matching keywords (simplified interface).

        Args:
            query: Search query string
            top_k: Number of results to return
            fields: Fields to search in
            minimum_score: Minimum score threshold
            **kwargs: Additional parameters

        Returns:
            KeywordSearchResponse: Search results
        """
        request = KeywordSearchRequest(
            query=query,
            top_k=top_k,
            fields=fields,
            minimum_score=minimum_score,
            **kwargs
        )

        return await self.search(request)

    async def boolean_search(
        self,
        boolean_query: BooleanQuery,
        top_k: int = 10,
        **kwargs
    ) -> KeywordSearchResponse:
        """
        Search using boolean query logic.

        Args:
            boolean_query: Boolean query structure
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            KeywordSearchResponse: Search results
        """
        request = KeywordSearchRequest(
            query=boolean_query,
            top_k=top_k,
            **kwargs
        )

        return await self.search(request)

    async def phrase_search(
        self,
        phrase: str,
        field: Optional[str] = None,
        top_k: int = 10,
        **kwargs
    ) -> KeywordSearchResponse:
        """
        Search for exact phrase matches.

        Args:
            phrase: Phrase to search for
            field: Field to search in
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            KeywordSearchResponse: Search results
        """
        phrase_query = KeywordQuery(
            query=phrase,
            query_type=QueryType.PHRASE,
            field=field,
            match_type=MatchType.EXACT
        )

        request = KeywordSearchRequest(
            query=phrase_query,
            top_k=top_k,
            **kwargs
        )

        return await self.search(request)

    async def fuzzy_search(
        self,
        query: str,
        fuzzy_distance: int = 1,
        top_k: int = 10,
        **kwargs
    ) -> KeywordSearchResponse:
        """
        Search with fuzzy matching.

        Args:
            query: Search query
            fuzzy_distance: Maximum edit distance
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            KeywordSearchResponse: Search results
        """
        fuzzy_query = KeywordQuery(
            query=query,
            query_type=QueryType.FUZZY,
            match_type=MatchType.FUZZY,
            fuzzy_distance=fuzzy_distance
        )

        request = KeywordSearchRequest(
            query=fuzzy_query,
            top_k=top_k,
            **kwargs
        )

        return await self.search(request)

    async def batch_search(
        self,
        request: BatchKeywordSearchRequest,
        **kwargs
    ) -> BatchKeywordSearchResponse:
        """
        Search for multiple queries.

        Args:
            request: Batch search request
            **kwargs: Additional provider-specific parameters

        Returns:
            BatchKeywordSearchResponse: Batch search results

        Raises:
            KeywordRetrieverException: If batch search fails
        """
        if not self.capabilities.supports_batch_search:
            # Fall back to sequential processing
            responses = []
            total_time = 0

            for i, query in enumerate(request.queries):
                single_request = KeywordSearchRequest(
                    query=query,
                    top_k=request.top_k,
                    filter=request.filter,
                    include_matches=request.include_matches,
                    include_highlights=request.include_highlights,
                    fields=request.fields,
                    boost_fields=request.boost_fields,
                    search_params=request.search_params,
                )

                start_time = time.time()
                response = await self.search(single_request, **kwargs)
                end_time = time.time()

                response.query_id = f"batch_query_{i}"
                responses.append(response)
                total_time += (end_time - start_time) * 1000

            return BatchKeywordSearchResponse(
                responses=responses,
                total_queries=len(request.queries),
                total_results=sum(len(r.results) for r in responses),
                total_time_ms=total_time,
                parallel_used=False,
            )

        # Parallel processing
        import asyncio
        semaphore = asyncio.Semaphore(request.max_workers)

        async def process_single_query(query: Union[str, KeywordQuery, BooleanQuery], index: int) -> KeywordSearchResponse:
            async with semaphore:
                single_request = KeywordSearchRequest(
                    query=query,
                    top_k=request.top_k,
                    filter=request.filter,
                    include_matches=request.include_matches,
                    include_highlights=request.include_highlights,
                    fields=request.fields,
                    boost_fields=request.boost_fields,
                    search_params=request.search_params,
                )

                response = await self.search(single_request, **kwargs)
                response.query_id = f"batch_query_{index}"
                return response

        start_time = time.time()

        if request.parallel_search:
            tasks = [
                process_single_query(query, i)
                for i, query in enumerate(request.queries)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            responses = []
            for i, query in enumerate(request.queries):
                response = await process_single_query(query, i)
                responses.append(response)

        end_time = time.time()

        # Handle exceptions
        valid_responses = []
        errors = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                errors.append(f"Query {i} failed: {str(response)}")
            else:
                valid_responses.append(response)

        return BatchKeywordSearchResponse(
            responses=valid_responses,
            total_queries=len(request.queries),
            total_results=sum(len(r.results) for r in valid_responses),
            total_time_ms=(end_time - start_time) * 1000,
            parallel_used=request.parallel_search,
            errors=errors,
        )

    def _tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize a search query.

        Args:
            query: Query string to tokenize

        Returns:
            List[str]: List of tokens
        """
        if self.tokenizer_type == TokenizerType.WHITESPACE:
            return query.split()
        elif self.tokenizer_type == TokenizerType.KEYWORD:
            return [query]
        elif self.tokenizer_type == TokenizerType.STANDARD:
            # Basic standard tokenization
            tokens = re.findall(r'\b\w+\b', query.lower())
            return tokens
        else:
            # Default to whitespace
            return query.split()

    def _filter_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Filter out stop words from tokens.

        Args:
            tokens: List of tokens

        Returns:
            List[str]: Filtered tokens
        """
        if not self.enable_stop_words:
            return tokens

        # Initialize stop words if not done
        if self._stop_words is None:
            # Basic English stop words
            self._stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
            }
            # Add custom stop words
            if self.config.custom_stop_words:
                self._stop_words.update(self.config.custom_stop_words)

        return [token for token in tokens if token not in self._stop_words]

    def _expand_synonyms(self, tokens: List[str]) -> List[str]:
        """
        Expand tokens with synonyms.

        Args:
            tokens: List of tokens

        Returns:
            List[str]: Expanded tokens with synonyms
        """
        if not self.enable_synonyms:
            return tokens

        # Initialize synonyms if not done
        if self._synonyms is None:
            self._synonyms = self.config.custom_synonyms or {}

        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            if token in self._synonyms:
                expanded_tokens.extend(self._synonyms[token])

        return expanded_tokens

    def _create_highlights(
        self,
        content: str,
        matches: List[KeywordMatch],
        max_snippets: Optional[int] = None
    ) -> List[str]:
        """
        Create highlighted snippets from content.

        Args:
            content: Original content
            matches: List of keyword matches
            max_snippets: Maximum number of snippets

        Returns:
            List[str]: Highlighted snippets
        """
        if not self.highlight_enabled or not matches:
            return []

        max_snippets = max_snippets or self.max_snippets
        open_tag, close_tag = self.highlight_tags
        snippets = []

        # Sort matches by position
        sorted_matches = sorted(matches, key=lambda x: x.start_position)

        for match in sorted_matches[:max_snippets]:
            start = max(0, match.start_position - self.snippet_size // 2)
            end = min(len(content), match.end_position + self.snippet_size // 2)

            snippet = content[start:end]
            # Highlight the matched term
            relative_start = match.start_position - start
            relative_end = match.end_position - start

            highlighted = (
                snippet[:relative_start] +
                open_tag +
                snippet[relative_start:relative_end] +
                close_tag +
                snippet[relative_end:]
            )
            snippets.append(highlighted)

        return snippets

    async def get_index_stats(self) -> KeywordIndexStats:
        """
        Get index statistics.

        Returns:
            KeywordIndexStats: Index statistics

        Raises:
            KeywordRetrieverException: If stats retrieval fails
        """
        if self._index_stats is None:
            # Default implementation - override in subclasses
            return KeywordIndexStats(
                index_name=self.index_name,
                total_documents=0,
                total_terms=0,
            )
        return self._index_stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the keyword retriever.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test query
            test_request = KeywordSearchRequest(
                query="test query",
                top_k=5
            )

            start_time = time.time()
            response = await self.search(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "index_name": self.index_name,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_results_count": len(response.results),
                "tokenizer_type": self.tokenizer_type,
                "ranking_mode": self.ranking_mode,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "index_name": self.index_name,
                "error": str(e)
            }

    def supports_query_type(self, query_type: QueryType) -> bool:
        """
        Check if the retriever supports a specific query type.

        Args:
            query_type: Query type to check

        Returns:
            bool: True if query type is supported
        """
        return query_type in self.capabilities.supported_query_types

    def supports_match_type(self, match_type: MatchType) -> bool:
        """
        Check if the retriever supports a specific match type.

        Args:
            match_type: Match type to check

        Returns:
            bool: True if match type is supported
        """
        return match_type in self.capabilities.supported_match_types

    def supports_tokenizer(self, tokenizer_type: TokenizerType) -> bool:
        """
        Check if the retriever supports a specific tokenizer.

        Args:
            tokenizer_type: Tokenizer type to check

        Returns:
            bool: True if tokenizer is supported
        """
        return tokenizer_type in self.capabilities.supported_tokenizers

    async def validate_request(
        self,
        request: KeywordSearchRequest
    ) -> bool:
        """
        Validate a keyword search request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check query length
        if isinstance(request.query, str) and len(request.query.strip()) == 0:
            raise ValueError("Query cannot be empty")

        # Check top_k
        if request.top_k < 1 or request.top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")

        # Check minimum_score
        if request.minimum_score is not None and request.minimum_score < 0.0:
            raise ValueError("minimum_score must be non-negative")

        # Check field boosts
        if request.boost_fields:
            for field, boost in request.boost_fields.items():
                if boost < 0.0:
                    raise ValueError(f"Field boost for '{field}' must be non-negative")

        return True

    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the keyword retriever.

        Returns:
            Dict[str, Any]: Retriever information
        """
        return {
            "provider": self.provider_name,
            "index_name": self.index_name,
            "tokenizer_type": self.tokenizer_type,
            "ranking_mode": self.ranking_mode,
            "capabilities": self.capabilities.dict(),
            "config": {
                "enable_stemming": self.enable_stemming,
                "enable_stop_words": self.enable_stop_words,
                "enable_synonyms": self.enable_synonyms,
                "enable_phonetic": self.enable_phonetic,
                "fuzzy_threshold": self.fuzzy_threshold,
                "highlight_enabled": self.highlight_enabled,
                "cache_enabled": self.cache_enabled,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }


class KeywordRetrieverException(Exception):
    """Exception raised by keyword retrievers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        index_name: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.index_name = index_name
        self.error_code = error_code


class IndexNotFoundException(KeywordRetrieverException):
    """Exception raised when index is not found."""
    pass


class QueryParseException(KeywordRetrieverException):
    """Exception raised when query parsing fails."""
    pass


class SearchTimeoutException(KeywordRetrieverException):
    """Exception raised when search times out."""
    pass


class InsufficientResultsException(KeywordRetrieverException):
    """Exception raised when search returns insufficient results."""
    pass