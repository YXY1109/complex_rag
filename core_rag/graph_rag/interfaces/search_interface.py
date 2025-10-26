"""
图搜索接口

定义图搜索算法的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from .graph_interface import NodeModel, EdgeModel


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    search_type: str = Field(default="hybrid")  # vector, keyword, hybrid, community
    node_types: List[str] = Field(default_factory=list)
    limit: int = Field(default=10)
    threshold: float = Field(default=0.7)
    user_id: Optional[str] = None
    context: Optional[str] = None


class SearchResult(BaseModel):
    """搜索结果"""
    node: NodeModel
    score: float
    explanation: Optional[str] = None
    matched_fields: List[str] = Field(default_factory=list)
    path: Optional[List[str]] = Field(default=None)


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    query: str
    search_type: str
    created_at: str


class CommunitySearchRequest(BaseModel):
    """社区搜索请求"""
    query: str
    community_types: List[str] = Field(default_factory=list)
    limit: int = Field(default=5)
    min_relevance: float = Field(default=0.5)
    user_id: Optional[str] = None


class CommunitySearchResult(BaseModel):
    """社区搜索结果"""
    community_id: str
    community_name: str
    relevance_score: float
    entity_count: int
    summary: Optional[str] = None
    key_entities: List[str] = Field(default_factory=list)


class CommunitySearchResponse(BaseModel):
    """社区搜索响应"""
    results: List[CommunitySearchResult]
    total_found: int
    search_time_ms: float
    query: str
    created_at: str


class PathSearchRequest(BaseModel):
    """路径搜索请求"""
    source_entity: str
    target_entity: str
    max_path_length: int = Field(default=5)
    path_types: List[str] = Field(default_factory=list)
    weight_threshold: float = Field(default=0.1)
    user_id: Optional[str] = None


class PathSearchResult(BaseModel):
    """路径搜索结果"""
    path: List[str]  # entity IDs
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    total_weight: float
    path_length: int
    relevance_score: float


class PathSearchResponse(BaseModel):
    """路径搜索响应"""
    results: List[PathSearchResult]
    total_found: int
    search_time_ms: float
    source_entity: str
    target_entity: str
    created_at: str


class SearchInterface(ABC):
    """
    图搜索接口抽象类

    定义基于图的搜索算法接口，包括节点搜索、社区搜索、路径搜索等。
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化搜索引擎

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def search_nodes(
        self,
        request: SearchRequest
    ) -> SearchResponse:
        """
        搜索节点

        Args:
            request: 搜索请求

        Returns:
            SearchResponse: 搜索结果
        """
        pass

    @abstractmethod
    async def search_communities(
        self,
        request: CommunitySearchRequest
    ) -> CommunitySearchResponse:
        """
        搜索社区

        Args:
            request: 社区搜索请求

        Returns:
            CommunitySearchResponse: 社区搜索结果
        """
        pass

    @abstractmethod
    async def find_paths(
        self,
        request: PathSearchRequest
    ) -> PathSearchResponse:
        """
        查找路径

        Args:
            request: 路径搜索请求

        Returns:
            PathSearchResponse: 路径搜索结果
        """
        pass

    @abstractmethod
    async def semantic_search(
        self,
        query: str,
        node_types: List[str] = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        语义搜索

        Args:
            query: 查询文本
            node_types: 节点类型过滤
            limit: 限制数量
            threshold: 相似度阈值

        Returns:
            List[SearchResult]: 搜索结果
        """
        pass

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        fields: List[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        关键词搜索

        Args:
            query: 关键词
            fields: 搜索字段
            limit: 限制数量

        Returns:
            List[Result]: 搜索结果
        """
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        node_types: List[str] = None,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        混合搜索（语义+关键词）

        Args:
            query: 查询文本
            semantic_weight: 语义搜索权重
            keyword_weight: 关键词搜索权重
            node_types: 节点类型过滤
            limit: 限制数量
            threshold: 综合阈值

        Returns:
            List[SearchResult]: 搜索结果
        """
        pass

    @abstractmethod
    async def explain_search(
        self,
        query: str,
        result: SearchResult
    ) -> str:
        """
        解释搜索结果

        Args:
            query: 原始查询
            result: 搜索结果

        Returns:
            str: 解释文本
        """
        pass

    @abstractmethod
    async def get_suggestions(
        self,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """
        获取搜索建议

        Args:
            partial_query: 部分查询
            limit: 建议数量

        Returns:
            List[str]: 搜索建议
        """
        pass

    @abstractmethod
    async def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索统计信息

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


class SearchException(Exception):
    """搜索异常基类"""
    pass


class QueryTooLongError(SearchException):
    """查询过长异常"""
    pass


class InvalidSearchTypeError(SearchException):
    """无效搜索类型异常"""
    pass


class SearchTimeoutError(SearchException):
    """搜索超时异常"""
    pass


class NoResultsError(SearchException):
    """无结果异常"""
    pass