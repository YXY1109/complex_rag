"""
图数据库接口

定义图数据库操作的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class GraphNodeType(str, Enum):
    """图节点类型"""
    ENTITY = "entity"
    DOCUMENT = "document"
    COMMUNITY = "community"
    RELATIONSHIP = "relationship"


class NodeModel(BaseModel):
    """图节点模型"""
    id: str
    type: GraphNodeType
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class EdgeModel(BaseModel):
    """图边模型"""
    id: str
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0)
    created_at: str
    updated_at: str


class GraphQuery(BaseModel):
    """图查询模型"""
    query_type: str = Field(default="search")  # search, path, neighbor
    node_types: List[GraphNodeType] = Field(default_factory=list)
    properties_filter: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=100)
    offset: int = Field(default=0)


class GraphPath(BaseModel):
    """图路径模型"""
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    weight: float = Field(default=1.0)
    length: int


class GraphInterface(ABC):
    """
    图数据库抽象接口

    定义图数据库操作的标准接口，支持节点、边的增删改查以及图查询功能。
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化图数据库连接

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def add_node(self, node: NodeModel) -> str:
        """
        添加节点

        Args:
            node: 节点模型

        Returns:
            str: 节点ID
        """
        pass

    @abstractmethod
    async def add_edge(self, edge: EdgeModel) -> str:
        """
        添加边

        Args:
            edge: 边模型

        Returns:
            str: 边ID
        """
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[NodeModel]:
        """
        获取节点

        Args:
            node_id: 节点ID

        Returns:
            Optional[NodeModel]: 节点模型
        """
        pass

    @abstractmethod
    async def get_edge(self, edge_id: str) -> Optional[EdgeModel]:
        """
        获取边

        Args:
            edge_id: 边ID

        Returns:
            Optional[EdgeModel]: 边模型
        """
        pass

    @abstractmethod
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        更新节点属性

        Args:
            node_id: 节点ID
            properties: 属性字典

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def update_edge(self, edge_id: str, properties: Dict[str, Any]) -> bool:
        """
        更新边属性

        Args:
            edge_id: 边ID
            properties: 属性字典

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """
        删除节点

        Args:
            node_id: 节点ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def delete_edge(self, edge_id: str) -> bool:
        """
        删除边

        Args:
            edge_id: 边ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def search_nodes(self, query: GraphQuery) -> List[NodeModel]:
        """
        搜索节点

        Args:
            query: 查询条件

        Returns:
            List[NodeModel]: 节点列表
        """
        pass

    @abstractmethod
    async def search_edges(self, query: GraphQuery) -> List[EdgeModel]:
        """
        搜索边

        Args:
            query: 查询条件

        Returns:
            List[EdgeModel]: 边列表
        """
        pass

    @abstractmethod
    async def get_neighbors(self, node_id: str, direction: str = "both", relationship_types: List[str] = None) -> List[NodeModel]:
        """
        获取节点的邻居

        Args:
            node_id: 节点ID
            direction: 方向（in, out, both）
            relationship_types: 关系类型过滤

        Returns:
            List[NodeModel]: 邻居节点列表
        """
        pass

    @abstractmethod
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[GraphPath]:
        """
        查找路径

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            max_depth: 最大深度

        Returns:
            List[GraphPath]: 路径列表
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息

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


class GraphException(Exception):
    """图数据库异常基类"""
    pass


class NodeNotFoundError(GraphException):
    """节点未找到异常"""
    pass


class EdgeNotFoundError(GraphException):
    """边未找到异常"""
    pass


class GraphConnectionError(GraphException):
    """图数据库连接异常"""
    pass


class QueryExecutionError(GraphException):
    """查询执行异常"""
    pass