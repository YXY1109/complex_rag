"""
Core RAG Retriever Module

提供多策略文档检索功能，支持向量检索、关键词检索、图检索等多种策略。
"""

# 接口定义
from .interfaces.retriever_interface import (
    RetrieverInterface,
    MultiStrategyRetrieverInterface,
    RetrievalQuery,
    RetrievalResult,
    MultiStrategyResult,
    DocumentChunk,
    RetrievalStrategy,
    RetrievalMode,
    RetrieverConfig,
    RetrievalException,
    RetrieverInitializationError,
    RetrievalTimeoutError,
    InvalidQueryError,
    DocumentNotFoundError,
    StrategyNotSupportedError,
)

# 检索策略实现
from .strategies.vector_retriever import VectorRetriever
from .strategies.keyword_retriever import BM25Retriever
from .strategies.graph_retriever import GraphRetriever

# 多策略检索器
from .multi_strategy_retriever import (
    MultiStrategyRetriever,
    StrategyConfig,
    FusionConfig,
)

# 工厂类
from .factory import RetrieverFactory

# 版本信息
__version__ = "0.1.0"
__author__ = "RAG Team"

__all__ = [
    # 接口
    "RetrieverInterface",
    "MultiStrategyRetrieverInterface",
    "RetrievalQuery",
    "RetrievalResult",
    "MultiStrategyResult",
    "DocumentChunk",
    "RetrievalStrategy",
    "RetrievalMode",
    "RetrieverConfig",
    "RetrieverException",
    "RetrieverInitializationError",
    "RetrievalTimeoutError",
    "InvalidQueryError",
    "DocumentNotFoundError",
    "StrategyNotSupportedError",

    # 检索策略
    "VectorRetriever",
    "BM25Retriever",
    "GraphRetriever",

    # 多策略检索器
    "MultiStrategyRetriever",
    "StrategyConfig",
    "FusionConfig",

    # 工厂类
    "RetrieverFactory",
]