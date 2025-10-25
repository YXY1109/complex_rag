"""
RAG服务层

基于RAGFlow rag/app架构实现的检索增强生成服务，
提供知识检索、文档重排、生成增强等核心功能。
"""

from .retrieval_engine import RetrievalEngine
from .document_ranker import DocumentRanker
from .context_builder import ContextBuilder
from .generation_service import GenerationService
from .knowledge_manager import KnowledgeManager
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .search_service import SearchService
from .reranker_service import RerankerService
from .chat_service import ChatService
from .interfaces.rag_interface import (
    RAGConfig,
    RAGQuery,
    RAGResult,
    RetrievalResult,
    GenerationResult
)

__all__ = [
    "RetrievalEngine",
    "DocumentRanker",
    "ContextBuilder",
    "GenerationService",
    "KnowledgeManager",
    "EmbeddingService",
    "VectorStore",
    "SearchService",
    "RerankerService",
    "ChatService",
    "RAGConfig",
    "RAGQuery",
    "RAGResult",
    "RetrievalResult",
    "GenerationResult"
]
