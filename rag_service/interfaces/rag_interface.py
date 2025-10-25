"""
RAG服务接口定义

定义检索增强生成服务的核心接口、数据模型和配置参数，
基于RAGFlow架构设计，支持多模态检索和智能生成。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class RetrievalMode(Enum):
    """检索模式。"""

    VECTOR = "vector"           # 向量检索
    HYBRID = "hybrid"           # 混合检索（向量+关键词）
    SEMANTIC = "semantic"       # 语义检索
    FULLTEXT = "fulltext"       # 全文检索
    GRAPH = "graph"            # 图检索
    MULTIMODAL = "multimodal"  # 多模态检索


class RerankMode(Enum):
    """重排模式。"""

    NONE = "none"              # 不重排
    CROSS_ENCODER = "cross_encoder"  # 交叉编码器重排
    BM25 = "bm25"              # BM25重排
    LEARNT = "learnt"          # 学习型重排
    MULTISTAGE = "multistage"  # 多阶段重排


class GenerationMode(Enum):
    """生成模式。"""

    DIRECT = "direct"          # 直接生成
    RAG = "rag"                # RAG生成
    FEWSHOT = "fewshot"        # 少样本生成
    CHAIN_OF_THOUGHT = "cot"   # 思维链生成
    REACT = "react"           # ReAct生成


class ContextStrategy(Enum):
    """上下文策略。"""

    STUFFING = "stuffing"      # 填充策略
    MAP_REDUCE = "map_reduce"  # Map-Reduce策略
    REFINE = "refine"          # 精炼策略
    COMPRESSION = "compression" # 压缩策略


@dataclass
class RAGConfig:
    """RAG配置。"""

    # 检索配置
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 10
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 重排配置
    rerank_mode: RerankMode = RerankMode.CROSS_ENCODER
    rerank_top_k: int = 5
    rerank_threshold: float = 0.5

    # 生成配置
    generation_mode: GenerationMode = GenerationMode.RAG
    context_strategy: ContextStrategy = ContextStrategy.STUFFING
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # 向量化配置
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    normalize_embeddings: bool = True

    # 知识库配置
    knowledge_bases: List[str] = field(default_factory=list)
    enabled_sources: List[str] = field(default_factory=list)
    filter_metadata: Dict[str, Any] = field(default_factory=dict)

    # 高级配置
    enable_query_rewriting: bool = True
    enable_hyde: bool = False  # Hypothetical Document Embeddings
    enable_query_decomposition: bool = True
    enable_self_reflection: bool = False

    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 秒
    cache_size: int = 10000

    # 性能配置
    parallel_retrieval: bool = True
    max_parallel_queries: int = 5
    timeout_seconds: int = 30


@dataclass
class RAGQuery:
    """RAG查询。"""

    query_id: str
    query: str
    context: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None

    # 查询参数
    retrieval_mode: Optional[RetrievalMode] = None
    top_k: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    include_metadata: bool = True
    include_scores: bool = True

    # 生成参数
    generation_mode: Optional[GenerationMode] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None

    # 追踪信息
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())


@dataclass
class DocumentChunk:
    """文档块。"""

    chunk_id: str
    content: str
    document_id: str
    chunk_index: int
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0
    source: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


@dataclass
class RetrievalResult:
    """检索结果。"""

    query_id: str
    chunks: List[DocumentChunk]
    total_found: int
    search_time: float
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        """是否有检索结果。"""
        return len(self.chunks) > 0

    @property
    def top_chunk(self) -> Optional[DocumentChunk]:
        """获取最相关的文档块。"""
        return self.chunks[0] if self.chunks else None


@dataclass
class GenerationContext:
    """生成上下文。"""

    context_chunks: List[DocumentChunk]
    formatted_context: str
    context_length: int
    truncation_info: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """生成结果。"""

    query_id: str
    answer: str
    context: GenerationContext
    generation_time: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    model_info: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """RAG结果。"""

    query_id: str
    query: str
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    total_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        """获取答案。"""
        return self.generation_result.answer

    @property
    def sources(self) -> List[DocumentChunk]:
        """获取来源文档。"""
        return self.retrieval_result.chunks


@dataclass
class KnowledgeBase:
    """知识库。"""

    kb_id: str
    name: str
    description: str
    tenant_id: str
    document_count: int = 0
    chunk_count: int = 0
    embedding_model: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.kb_id:
            self.kb_id = str(uuid.uuid4())


@dataclass
class ChatSession:
    """聊天会话。"""

    session_id: str
    user_id: str
    tenant_id: str
    title: Optional[str] = None
    knowledge_bases: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


# 抽象接口定义
class RAGInterface(ABC):
    """RAG服务接口。"""

    @abstractmethod
    async def initialize(self, config: RAGConfig) -> bool:
        """
        初始化RAG服务。

        Args:
            config: RAG配置

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """清理RAG服务资源。"""
        pass

    @abstractmethod
    async def query(self, query: RAGQuery) -> RAGResult:
        """
        执行RAG查询。

        Args:
            query: RAG查询

        Returns:
            RAGResult: 查询结果
        """
        pass


class RetrievalInterface(ABC):
    """检索接口。"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        knowledge_bases: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        检索相关文档。

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            filters: 过滤条件
            knowledge_bases: 知识库列表

        Returns:
            RetrievalResult: 检索结果
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        knowledge_base_id: str
    ) -> List[str]:
        """
        添加文档到知识库。

        Args:
            documents: 文档列表
            knowledge_base_id: 知识库ID

        Returns:
            List[str]: 文档ID列表
        """
        pass

    @abstractmethod
    async def delete_documents(
        self,
        document_ids: List[str],
        knowledge_base_id: str
    ) -> bool:
        """
        从知识库删除文档。

        Args:
            document_ids: 文档ID列表
            knowledge_base_id: 知识库ID

        Returns:
            bool: 删除是否成功
        """
        pass


class RerankInterface(ABC):
    """重排接口。"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        重排文档。

        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回结果数量

        Returns:
            List[DocumentChunk]: 重排后的文档列表
        """
        pass


class GenerationInterface(ABC):
    """生成接口。"""

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        生成回答。

        Args:
            query: 查询字符串
            context: 上下文
            conversation_history: 对话历史
            **kwargs: 额外参数

        Returns:
            GenerationResult: 生成结果
        """
        pass


class EmbeddingInterface(ABC):
    """嵌入接口。"""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入。

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入。

        Args:
            text: 文本字符串

        Returns:
            List[float]: 嵌入向量
        """
        pass


class VectorStoreInterface(ABC):
    """向量存储接口。"""

    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        添加向量。

        Args:
            vectors: 向量列表
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索相似向量。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: (文档ID, 相似度分数, 元数据) 列表
        """
        pass

    @abstractmethod
    async def delete_vectors(self, document_ids: List[str]) -> bool:
        """
        删除向量。

        Args:
            document_ids: 文档ID列表

        Returns:
            bool: 删除是否成功
        """
        pass


class KnowledgeManagerInterface(ABC):
    """知识库管理接口。"""

    @abstractmethod
    async def create_knowledge_base(
        self,
        name: str,
        description: str,
        tenant_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> KnowledgeBase:
        """
        创建知识库。

        Args:
            name: 知识库名称
            description: 描述
            tenant_id: 租户ID
            config: 配置参数

        Returns:
            KnowledgeBase: 知识库信息
        """
        pass

    @abstractmethod
    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """
        获取知识库信息。

        Args:
            kb_id: 知识库ID

        Returns:
            Optional[KnowledgeBase]: 知识库信息
        """
        pass

    @abstractmethod
    async def list_knowledge_bases(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[KnowledgeBase]:
        """
        列出知识库。

        Args:
            tenant_id: 租户ID
            limit: 限制数量
            offset: 偏移量

        Returns:
            List[KnowledgeBase]: 知识库列表
        """
        pass

    @abstractmethod
    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """
        删除知识库。

        Args:
            kb_id: 知识库ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def update_knowledge_base(
        self,
        kb_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        更新知识库。

        Args:
            kb_id: 知识库ID
            updates: 更新内容

        Returns:
            bool: 更新是否成功
        """
        pass


class ChatInterface(ABC):
    """聊天接口。"""

    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        tenant_id: str,
        title: Optional[str] = None,
        knowledge_bases: Optional[List[str]] = None
    ) -> ChatSession:
        """
        创建聊天会话。

        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            title: 会话标题
            knowledge_bases: 知识库列表

        Returns:
            ChatSession: 聊天会话
        """
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        获取聊天会话。

        Args:
            session_id: 会话ID

        Returns:
            Optional[ChatSession]: 聊天会话
        """
        pass

    @abstractmethod
    async def chat(
        self,
        session_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        发送聊天消息。

        Args:
            session_id: 会话ID
            message: 消息内容
            config: 配置参数

        Returns:
            RAGResult: 聊天回复
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        删除聊天会话。

        Args:
            session_id: 会话ID

        Returns:
            bool: 删除是否成功
        """
        pass


# 错误定义
class RAGException(Exception):
    """RAG服务异常基类。"""
    pass


class RetrievalException(RAGException):
    """检索异常。"""
    pass


class GenerationException(RAGException):
    """生成异常。"""
    pass


class KnowledgeBaseException(RAGException):
    """知识库异常。"""
    pass


class EmbeddingException(RAGException):
    """嵌入异常。"""
    pass


class VectorStoreException(RAGException):
    """向量存储异常。"""
    pass


class ChatException(RAGException):
    """聊天异常。"""
    pass


# 工具函数
def create_rag_query(
    query: str,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> RAGQuery:
    """创建RAG查询。"""
    return RAGQuery(
        query_id=str(uuid.uuid4()),
        query=query,
        user_id=user_id,
        tenant_id=tenant_id,
        **kwargs
    )


def create_rag_config(
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    top_k: int = 10,
    generation_mode: GenerationMode = GenerationMode.RAG,
    **kwargs
) -> RAGConfig:
    """创建RAG配置。"""
    return RAGConfig(
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        generation_mode=generation_mode,
        **kwargs
    )


def validate_rag_config(config: RAGConfig) -> List[str]:
    """验证RAG配置。"""
    errors = []

    if config.top_k <= 0:
        errors.append("top_k 必须大于 0")

    if config.similarity_threshold < 0 or config.similarity_threshold > 1:
        errors.append("similarity_threshold 必须在 0-1 之间")

    if config.chunk_size <= 0:
        errors.append("chunk_size 必须大于 0")

    if config.max_tokens <= 0:
        errors.append("max_tokens 必须大于 0")

    if config.temperature < 0 or config.temperature > 2:
        errors.append("temperature 必须在 0-2 之间")

    return errors


def format_context(
    chunks: List[DocumentChunk],
    max_length: int = 4000,
    include_metadata: bool = False
) -> str:
    """格式化上下文。"""
    context_parts = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.content

        if include_metadata and chunk.metadata:
            metadata_text = f" [来源: {chunk.metadata.get('source', '未知')}]"
            chunk_text += metadata_text

        context_parts.append(f"[文档 {i+1}]\n{chunk_text}")

    # 合并并截断
    full_context = "\n\n".join(context_parts)

    if len(full_context) > max_length:
        # 简单截断，实际应用中可以使用更智能的方法
        truncated = full_context[:max_length - 100] + "\n...[内容已截断]"
        return truncated

    return full_context