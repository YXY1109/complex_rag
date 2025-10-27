"""
检索器接口定义

定义多种检索策略的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class RetrievalStrategy(str, Enum):
    """检索策略"""
    VECTOR = "vector"           # 向量检索
    KEYWORD = "keyword"         # 关键词检索
    HYBRID = "hybrid"          # 混合检索
    GRAPH = "graph"            # 图检索
    SEMANTIC = "semantic"      # 语义检索
    BM25 = "bm25"              # BM25检索
    DENSE = "dense"            # 稠密检索
    SPARSE = "sparse"          # 稀疏检索
    UNIFIED_VECTOR = "unified_vector"  # 统一向量存储检索


class RetrievalMode(str, Enum):
    """检索模式"""
    SINGLE = "single"          # 单一策略
    MULTI = "multi"            # 多策略并行
    SEQUENTIAL = "sequential"  # 顺序检索
    ADAPTIVE = "adaptive"      # 自适应检索


class DocumentChunk(BaseModel):
    """文档片段"""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    source: Optional[str] = None
    chunk_index: int = 0
    start_pos: int = 0
    end_pos: int = 0
    embedding: Optional[List[float]] = None
    created_at: str


class RetrievalQuery(BaseModel):
    """检索查询"""
    text: str
    query_embedding: Optional[List[float]] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    mode: RetrievalMode = RetrievalMode.SINGLE
    top_k: int = Field(default=10)
    min_score: float = Field(default=0.0)
    max_results: int = Field(default=50)
    rerank: bool = Field(default=False)
    expand_query: bool = Field(default=False)
    user_id: Optional[str] = None
    context: Optional[str] = None


class RetrievalResult(BaseModel):
    """检索结果"""
    chunks: List[DocumentChunk]
    query: str
    strategy: RetrievalStrategy
    total_found: int
    search_time_ms: float
    scores: List[float] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class MultiStrategyResult(BaseModel):
    """多策略检索结果"""
    query: str
    results: Dict[RetrievalStrategy, RetrievalResult]
    combined_chunks: List[DocumentChunk]
    strategy_scores: Dict[RetrievalStrategy, float] = Field(default_factory=dict)
    combined_score: float = 0.0
    total_processing_time_ms: float
    best_strategy: Optional[RetrievalStrategy] = None
    created_at: str


class RetrieverConfig(BaseModel):
    """检索器配置"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    mode: RetrievalMode = RetrievalMode.SINGLE
    top_k: int = 10
    min_score: float = 0.0
    max_results: int = 50
    enable_rerank: bool = False
    enable_query_expansion: bool = False
    enable_caching: bool = True
    cache_ttl: int = 3600
    parallel_strategies: List[RetrievalStrategy] = Field(default_factory=list)
    strategy_weights: Dict[RetrievalStrategy, float] = Field(default_factory=dict)


class RetrieverInterface(ABC):
    """
    检索器接口抽象类

    定义不同检索策略的标准接口。
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化检索器

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        添加文档到检索器

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery
    ) -> RetrievalResult:
        """
        执行检索

        Args:
            query: 检索查询

        Returns:
            RetrievalResult: 检索结果
        """
        pass

    @abstractmethod
    async def batch_retrieve(
        self,
        queries: List[RetrievalQuery]
    ) -> List[RetrievalResult]:
        """
        批量检索

        Args:
            queries: 查询列表

        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        pass

    @abstractmethod
    async def delete_documents(
        self,
        document_ids: List[str]
    ) -> bool:
        """
        删除文档

        Args:
            document_ids: 文档ID列表

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def update_document(
        self,
        document_id: str,
        document: Dict[str, Any]
    ) -> bool:
        """
        更新文档

        Args:
            document_id: 文档ID
            document: 更新的文档内容

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def get_document(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取文档

        Args:
            document_id: 文档ID

        Returns:
            Optional[Dict[str, Any]]: 文档内容
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        document_id: str,
        top_k: int = 10
    ) -> List[DocumentChunk]:
        """
        搜索相似文档

        Args:
            document_id: 文档ID
            top_k: 返回数量

        Returns:
            List[DocumentChunk]: 相似文档列表
        """
        pass

    @abstractmethod
    async def expand_query(
        self,
        query: str,
        max_terms: int = 5
    ) -> List[str]:
        """
        查询扩展

        Args:
            query: 原始查询
            max_terms: 最大扩展词数

        Returns:
            List[str]: 扩展后的查询词列表
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass


class MultiStrategyRetrieverInterface(ABC):
    """
    多策略检索器接口

    支持多种检索策略的组合和优化。
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化多策略检索器"""
        pass

    @abstractmethod
    async def add_strategy(
        self,
        strategy: RetrievalStrategy,
        retriever: RetrieverInterface,
        weight: float = 1.0
    ) -> bool:
        """
        添加检索策略

        Args:
            strategy: 检索策略
            retriever: 检索器实例
            weight: 策略权重

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def retrieve_multi_strategy(
        self,
        query: RetrievalQuery
    ) -> MultiStrategyResult:
        """
        多策略检索

        Args:
            query: 检索查询

        Returns:
            MultiStrategyResult: 多策略检索结果
        """
        pass

    @abstractmethod
    async def adaptive_retrieve(
        self,
        query: RetrievalQuery
    ) -> MultiStrategyResult:
        """
        自适应检索

        Args:
            query: 检索查询

        Returns:
            MultiStrategyResult: 自适应检索结果
        """
        pass

    @abstractmethod
    async def optimize_strategy_weights(
        self,
        training_queries: List[RetrievalQuery],
        ground_truth: List[List[str]]
    ) -> Dict[RetrievalStrategy, float]:
        """
        优化策略权重

        Args:
            training_queries: 训练查询
            ground_truth: 真实相关文档

        Returns:
            Dict[RetrievalStrategy, float]: 优化后的权重
        """
        pass


class RetrieverException(Exception):
    """检索器异常基类"""
    pass


class RetrieverInitializationError(RetrieverException):
    """检索器初始化异常"""
    pass


class RetrievalTimeoutError(RetrieverException):
    """检索超时异常"""
    pass


class InvalidQueryError(RetrieverException):
    """无效查询异常"""
    pass


class DocumentNotFoundError(RetrieverException):
    """文档未找到异常"""
    pass


class StrategyNotSupportedError(RetrieverException):
    """不支持的策略异常"""
    pass