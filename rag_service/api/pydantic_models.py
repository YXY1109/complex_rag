"""
Pydantic数据模型

定义API请求和响应的数据模型。
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class RetrievalModeEnum(str, Enum):
    """检索模式枚举。"""
    VECTOR = "vector"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    FULLTEXT = "fulltext"


class GenerationModeEnum(str, Enum):
    """生成模式枚举。"""
    DIRECT = "direct"
    STUFFING = "stuffing"
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
    COMPRESSION = "compression"
    CHAIN_OF_THOUGHT = "chain_of_thought"


class RAGQueryRequest(BaseModel):
    """RAG查询请求模型。"""
    query: str = Field(..., description="查询内容", min_length=1, max_length=2000)
    query_id: Optional[str] = Field(None, description="查询ID")
    knowledge_bases: Optional[List[str]] = Field(default_factory=list, description="知识库ID列表")
    retrieval_mode: RetrievalModeEnum = Field(RetrievalModeEnum.HYBRID, description="检索模式")
    generation_mode: GenerationModeEnum = Field(GenerationModeEnum.DIRECT, description="生成模式")
    top_k: int = Field(5, ge=1, le=20, description="检索文档数量")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="相似度阈值")
    max_tokens: int = Field(1000, ge=100, le=4000, description="最大生成token数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="生成温度")
    user_id: Optional[str] = Field(None, description="用户ID")
    tenant_id: Optional[str] = Field(None, description="租户ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="查询元数据")

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('查询内容不能为空')
        return v.strip()


class RAGBatchQueryRequest(BaseModel):
    """批量RAG查询请求模型。"""
    queries: List[RAGQueryRequest] = Field(..., min_items=1, max_items=10, description="查询列表")
    max_concurrent: int = Field(5, ge=1, le=10, description="最大并发数")


class SimpleQARequest(BaseModel):
    """简单问答请求模型。"""
    question: str = Field(..., description="问题", min_length=1, max_length=1000)
    knowledge_bases: Optional[List[str]] = Field(default_factory=list, description="知识库ID列表")


class DocumentSummaryRequest(BaseModel):
    """文档摘要请求模型。"""
    document_content: str = Field(..., description="文档内容", min_length=100)
    max_length: int = Field(500, ge=100, le=2000, description="最大摘要长度")


class DocumentCompareRequest(BaseModel):
    """文档比较请求模型。"""
    documents: List[str] = Field(..., min_items=2, max_items=5, description="文档内容列表")
    criteria: Optional[str] = Field(None, description="比较标准")


class ChatSessionCreateRequest(BaseModel):
    """创建聊天会话请求模型。"""
    title: Optional[str] = Field(None, description="会话标题", max_length=100)
    knowledge_bases: Optional[List[str]] = Field(default_factory=list, description="关联知识库")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="会话配置")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="会话元数据")


class ChatMessageRequest(BaseModel):
    """聊天消息请求模型。"""
    message: str = Field(..., description="消息内容", min_length=1, max_length=2000)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="消息配置")


class KnowledgeBaseCreateRequest(BaseModel):
    """创建知识库请求模型。"""
    name: str = Field(..., description="知识库名称", min_length=1, max_length=100)
    description: str = Field(..., description="知识库描述", min_length=1, max_length=500)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="知识库配置")


class KnowledgeBaseUpdateRequest(BaseModel):
    """更新知识库请求模型。"""
    name: Optional[str] = Field(None, description="新名称", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="新描述", min_length=1, max_length=500)
    config: Optional[Dict[str, Any]] = Field(None, description="新配置")


class DocumentUploadRequest(BaseModel):
    """文档上传请求模型。"""
    title: str = Field(..., description="文档标题", min_length=1, max_length=200)
    content: str = Field(..., description="文档内容", min_length=10)
    file_path: Optional[str] = Field(None, description="文件路径")
    file_type: Optional[str] = Field(None, description="文件类型")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="文档元数据")


class DocumentSearchRequest(BaseModel):
    """文档搜索请求模型。"""
    query: str = Field(..., description="搜索查询", min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50, description="返回结果数量")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="过滤条件")


class DocumentProcessRequest(BaseModel):
    """文档处理请求模型。"""
    force_reprocess: bool = Field(False, description="是否强制重新处理")


class MaintenanceRequest(BaseModel):
    """系统维护请求模型。"""
    action: str = Field(..., description="维护操作类型")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="操作参数")


class ConfigUpdateRequest(BaseModel):
    """配置更新请求模型。"""
    updates: Dict[str, Any] = Field(..., description="配置更新项")


# 响应模型

class BaseResponse(BaseModel):
    """基础响应模型。"""
    success: bool = Field(..., description="操作是否成功")
    message: Optional[str] = Field(None, description="响应消息")
    request_id: Optional[str] = Field(None, description="请求ID")


class DataResponse(BaseResponse):
    """数据响应模型。"""
    data: Dict[str, Any] = Field(..., description="响应数据")


class ErrorResponse(BaseResponse):
    """错误响应模型。"""
    error: bool = Field(True, description="是否为错误")
    error_code: str = Field(..., description="错误代码")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


class ChunkInfo(BaseModel):
    """文档块信息模型。"""
    chunk_id: str = Field(..., description="块ID")
    content: str = Field(..., description="块内容")
    title: Optional[str] = Field(None, description="标题")
    score: float = Field(..., description="相关度分数")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="块元数据")


class RetrievalResult(BaseModel):
    """检索结果模型。"""
    chunks: List[ChunkInfo] = Field(default_factory=list, description="检索到的文档块")
    search_time: float = Field(..., description="检索耗时(秒)")
    total_found: int = Field(..., description="总找到数量")


class GenerationResult(BaseModel):
    """生成结果模型。"""
    model: str = Field(..., description="使用的模型")
    generation_time: float = Field(..., description="生成耗时(秒)")
    token_count: int = Field(..., description="生成token数量")


class RAGQueryResponse(BaseModel):
    """RAG查询响应模型。"""
    query_id: str = Field(..., description="查询ID")
    answer: str = Field(..., description="回答内容")
    retrieval_result: RetrievalResult = Field(..., description="检索结果")
    generation_result: GenerationResult = Field(..., description="生成结果")
    total_time: float = Field(..., description="总耗时(秒)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="查询元数据")


class BatchQueryResult(BaseModel):
    """批量查询结果模型。"""
    query_id: str = Field(..., description="查询ID")
    success: bool = Field(..., description="是否成功")
    answer: Optional[str] = Field(None, description="回答内容")
    error: Optional[str] = Field(None, description="错误信息")
    total_time: float = Field(..., description="耗时(秒)")


class ChatSessionInfo(BaseModel):
    """聊天会话信息模型。"""
    session_id: str = Field(..., description="会话ID")
    title: str = Field(..., description="会话标题")
    user_id: str = Field(..., description="用户ID")
    tenant_id: str = Field(..., description="租户ID")
    knowledge_bases: List[str] = Field(default_factory=list, description="关联知识库")
    config: Dict[str, Any] = Field(default_factory=dict, description="会话配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="会话元数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class ChatMessageInfo(BaseModel):
    """聊天消息信息模型。"""
    message_id: str = Field(..., description="消息ID")
    session_id: str = Field(..., description="会话ID")
    content: str = Field(..., description="消息内容")
    role: str = Field(..., description="角色")
    timestamp: datetime = Field(..., description="时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="消息元数据")


class KnowledgeBaseInfo(BaseModel):
    """知识库信息模型。"""
    kb_id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: str = Field(..., description="知识库描述")
    tenant_id: str = Field(..., description="租户ID")
    created_by: str = Field(..., description="创建者")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    status: str = Field(..., description="状态")
    document_count: int = Field(0, description="文档数量")
    config: Dict[str, Any] = Field(default_factory=dict, description="知识库配置")


class DocumentInfo(BaseModel):
    """文档信息模型。"""
    document_id: str = Field(..., description="文档ID")
    kb_id: str = Field(..., description="知识库ID")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    status: str = Field(..., description="处理状态")
    file_type: Optional[str] = Field(None, description="文件类型")
    file_size: int = Field(0, description="文件大小")


class HealthStatus(BaseModel):
    """健康状态模型。"""
    status: str = Field(..., description="健康状态")
    timestamp: datetime = Field(..., description="检查时间")
    components: Dict[str, Any] = Field(default_factory=dict, description="组件状态")
    issues: List[str] = Field(default_factory=list, description="问题列表")


class ServiceStatus(BaseModel):
    """服务状态模型。"""
    initialized: bool = Field(..., description="是否已初始化")
    initialization_time: Optional[datetime] = Field(None, description="初始化时间")
    components: Dict[str, Any] = Field(default_factory=dict, description="组件状态")


class SystemMetrics(BaseModel):
    """系统指标模型。"""
    cpu_usage: List[Dict[str, Any]] = Field(default_factory=list, description="CPU使用率")
    memory_usage: List[Dict[str, Any]] = Field(default_factory=list, description="内存使用率")
    request_count: List[Dict[str, Any]] = Field(default_factory=list, description="请求数量")
    response_time: List[Dict[str, Any]] = Field(default_factory=list, description="响应时间")
    error_rate: List[Dict[str, Any]] = Field(default_factory=list, description="错误率")


class MaintenanceResult(BaseModel):
    """维护操作结果模型。"""
    action: str = Field(..., description="操作类型")
    status: str = Field(..., description="操作状态")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: datetime = Field(..., description="完成时间")
    affected_items: int = Field(0, description="影响项目数")
    details: str = Field(..., description="操作详情")


class VersionInfo(BaseModel):
    """版本信息模型。"""
    version: str = Field(..., description="版本号")
    build_time: str = Field(..., description="构建时间")
    git_commit: str = Field(..., description="Git提交号")
    python_version: str = Field(..., description="Python版本")
    fastapi_version: str = Field(..., description="FastAPI版本")
    service: str = Field(..., description="服务名称")


class ServiceInfo(BaseModel):
    """服务信息模型。"""
    service_name: str = Field(..., description="服务名称")
    description: str = Field(..., description="服务描述")
    version: str = Field(..., description="版本号")
    documentation: str = Field(..., description="文档地址")
    redoc: str = Field(..., description="ReDoc地址")
    openapi: str = Field(..., description="OpenAPI地址")
    health: str = Field(..., description="健康检查地址")
    admin: Dict[str, str] = Field(default_factory=dict, description="管理接口地址")
    features: List[str] = Field(default_factory=list, description="功能特性")