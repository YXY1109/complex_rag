"""
RAG流水线接口定义

定义RAG处理流水线的标准接口和数据模型。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class PipelineStage(str, Enum):
    """流水线阶段"""
    QUERY_UNDERSTANDING = "query_understanding"
    RETRIEVAL = "retrieval"
    CONTEXT_BUILDING = "context_building"
    ANSWER_GENERATION = "answer_generation"
    POST_PROCESSING = "post_processing"


class GenerationStrategy(str, Enum):
    """生成策略"""
    CONCISE = "concise"           # 简洁回答
    DETAILED = "detailed"         # 详细回答
    STEP_BY_STEP = "step_by_step" # 分步回答
    STRUCTURED = "structured"     # 结构化回答
    CONVERSATIONAL = "conversational" # 对话式回答


class QueryType(str, Enum):
    """查询类型"""
    FACTUAL = "factual"           # 事实查询
    PROCEDURAL = "procedural"     # 程序查询
    EXPLANATORY = "explanatory"   # 解释查询
    COMPARATIVE = "comparative"   # 比较查询
    CREATIVE = "creative"         # 创造性查询
    CONVERSATIONAL = "conversational" # 对话查询


class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    query_type: Optional[QueryType] = None
    generation_strategy: GenerationStrategy = GenerationStrategy.DETAILED
    max_context_length: int = Field(default=4000)
    max_answer_length: int = Field(default=1000)
    enable_citations: bool = Field(default=True)
    enable_sources: bool = Field(default=True)
    filters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class QueryUnderstanding(BaseModel):
    """查询理解结果"""
    original_query: str
    processed_query: str
    query_type: QueryType
    query_intent: str
    key_entities: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    query_rewrite_suggestions: List[str] = Field(default_factory=list)
    expansion_terms: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """检索结果"""
    chunks: List[Dict[str, Any]]
    total_found: int
    retrieval_time_ms: float
    strategy_used: str
    scores: List[float] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextDocument(BaseModel):
    """上下文文档"""
    id: str
    content: str
    score: float
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    citation_id: Optional[str] = None


class Context(BaseModel):
    """构建的上下文"""
    documents: List[ContextDocument]
    formatted_context: str
    total_length: int
    relevance_score: float = 0.0
    construction_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    """生成的答案"""
    content: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    generation_time_ms: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    """流水线响应"""
    query: str
    answer: Answer
    query_understanding: QueryUnderstanding
    retrieval_results: RetrievalResult
    context: Context
    total_processing_time_ms: float
    stage_times: Dict[str, float] = Field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PipelineConfig(BaseModel):
    """流水线配置"""
    # 查询理解配置
    enable_query_rewrite: bool = True
    enable_query_expansion: bool = True
    enable_intent_detection: bool = True
    max_rewrite_suggestions: int = 3

    # 检索配置
    retrieval_strategies: List[str] = Field(default_factory=lambda: ["vector", "bm25"])
    max_retrieval_results: int = 20
    min_relevance_score: float = 0.3
    enable_reranking: bool = True

    # 上下文构建配置
    max_context_tokens: int = 4000
    context_window_overlap: int = 50
    enable_context_compression: bool = True
    enable_context_ranking: bool = True

    # 答案生成配置
    generation_model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1000
    enable_citation_generation: bool = True
    enable_source_attribution: bool = True

    # 性能配置
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_parallel_processing: bool = True
    timeout_seconds: int = 30


class RAGPipelineInterface(ABC):
    """
    RAG流水线接口抽象类

    定义RAG处理流水线的标准接口。
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化RAG流水线

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def process(
        self,
        request: QueryRequest
    ) -> PipelineResponse:
        """
        处理RAG查询

        Args:
            request: 查询请求

        Returns:
            PipelineResponse: 处理结果
        """
        pass

    @abstractmethod
    async def process_stream(
        self,
        request: QueryRequest
    ) -> AsyncGenerator[str, None]:
        """
        流式处理RAG查询

        Args:
            request: 查询请求

        Yields:
            str: 流式生成的答案片段
        """
        pass

    @abstractmethod
    async def understand_query(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> QueryUnderstanding:
        """
        查询理解

        Args:
            query: 原始查询
            context: 上下文信息
            history: 对话历史

        Returns:
            QueryUnderstanding: 查询理解结果
        """
        pass

    @abstractmethod
    async def retrieve_documents(
        self,
        processed_query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 20
    ) -> RetrievalResult:
        """
        文档检索

        Args:
            processed_query: 处理后的查询
            filters: 过滤条件
            max_results: 最大结果数

        Returns:
            RetrievalResult: 检索结果
        """
        pass

    @abstractmethod
    async def build_context(
        self,
        retrieval_results: RetrievalResult,
        max_length: int = 4000
    ) -> Context:
        """
        构建上下文

        Args:
            retrieval_results: 检索结果
            max_length: 最大长度

        Returns:
            Context: 构建的上下文
        """
        pass

    @abstractmethod
    async def generate_answer(
        self,
        query: str,
        context: Context,
        generation_strategy: GenerationStrategy = GenerationStrategy.DETAILED,
        enable_citations: bool = True
    ) -> Answer:
        """
        生成答案

        Args:
            query: 查询
            context: 上下文
            generation_strategy: 生成策略
            enable_citations: 是否启用引用

        Returns:
            Answer: 生成的答案
        """
        pass

    @abstractmethod
    async def batch_process(
        self,
        requests: List[QueryRequest]
    ) -> List[PipelineResponse]:
        """
        批量处理查询

        Args:
            requests: 查询请求列表

        Returns:
            List[PipelineResponse]: 处理结果列表
        """
        pass

    @abstractmethod
    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        获取流水线统计信息

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


class PipelineException(Exception):
    """流水线异常基类"""
    pass


class QueryUnderstandingError(PipelineException):
    """查询理解异常"""
    pass


class RetrievalError(PipelineException):
    """检索异常"""
    pass


class ContextBuildingError(PipelineException):
    """上下文构建异常"""
    pass


class AnswerGenerationError(PipelineException):
    """答案生成异常"""
    pass


class PipelineTimeoutError(PipelineException):
    """流水线超时异常"""
    pass


class ConfigurationError(PipelineException):
    """配置异常"""
    pass