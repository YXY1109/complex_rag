"""
来源处理器接口抽象类

此模块定义了文档来源处理服务的抽象接口。
所有来源处理器实现都必须继承自这个基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import time


class DocumentSource(str, Enum):
    """文档来源类型。"""
    WEB_DOCUMENTS = "web_documents"  # HTML、Markdown、API文档
    OFFICE_DOCUMENTS = "office_documents"  # PDF、DOCX、Excel、PPT
    SCANNED_DOCUMENTS = "scanned_documents"  # OCR、图片、多模态
    STRUCTURED_DATA = "structured_data"  # JSON、CSV、XML、YAML
    CODE_REPOSITORIES = "code_repositories"  # GitHub、代码文件、技术文档
    LOCAL_FILES = "local_files"  # 本地文件系统
    REMOTE_STORAGE = "remote_storage"  # 远程存储（S3、OSS等）
    DATABASE = "database"  # 数据库文档
    EMAIL = "email"  # 邮件附件
    CHAT = "chat"  # 聊天记录
    CUSTOM = "custom"  # 自定义来源


class ProcessingMode(str, Enum):
    """处理模式。"""
    INCREMENTAL = "incremental"  # 增量处理
    FULL = "full"  # 全量处理
    SCHEDULED = "scheduled"  # 定时处理
    REAL_TIME = "real_time"  # 实时处理
    BATCH = "batch"  # 批量处理


class SourceConfig(BaseModel):
    """来源配置模型。"""
    source_name: str
    source_type: DocumentSource
    processing_mode: ProcessingMode = ProcessingMode.FULL

    # 连接设置
    endpoint: Optional[str] = Field(default=None, description="来源端点URL")
    credentials: Optional[Dict[str, str]] = Field(default=None, description="认证信息")
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP头部")

    # 过滤设置
    include_patterns: List[str] = Field(default_factory=list, description="包含模式")
    exclude_patterns: List[str] = Field(default_factory=list, description="排除模式")
    max_depth: int = Field(default=3, description="最大深度（用于网页爬取）")
    max_files: Optional[int] = Field(default=None, description="最大文件数")

    # 处理设置
    enable_crawling: bool = Field(default=False, description="启用爬取")
    respect_robots_txt: bool = Field(default=True, description="遵守robots.txt")
    user_agent: str = Field(default="ComplexRAG Bot", description="用户代理")

    # 性能设置
    timeout_seconds: int = Field(default=30, description="超时时间（秒）")
    max_concurrent_requests: int = Field(default=5, description="最大并发请求数")
    retry_attempts: int = Field(default=3, description="重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟（秒）")

    # 缓存设置
    enable_cache: bool = Field(default=True, description="启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存TTL（秒）")

    # 自定义选项
    custom_options: Optional[Dict[str, Any]] = Field(default=None, description="自定义选项")


class DocumentReference:
    """文档引用。"""

    def __init__(
        self,
        source_id: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        content_type: Optional[str] = None,
        size: Optional[int] = None,
        last_modified: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.source_id = source_id
        self.url = url
        self.title = title
        self.content_type = content_type
        self.size = size
        self.last_modified = last_modified
        self.metadata = metadata or {}


class SourceResult:
    """来源处理结果。"""

    def __init__(
        self,
        success: bool,
        documents: Optional[List[DocumentReference]] = None,
        processing_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        statistics: Optional[Dict[str, Any]] = None,
        next_cursor: Optional[str] = None
    ):
        self.success = success
        self.documents = documents or []
        self.processing_time_ms = processing_time_ms
        self.error_message = error_message
        self.statistics = statistics or {}
        self.next_cursor = next_cursor


class SourceProcessorInterface(ABC):
    """
    来源处理器的抽象接口。

    此类定义了所有来源处理器实现必须遵循的契约。
    """

    def __init__(self, config: SourceConfig):
        """
        使用配置初始化来源处理器。

        Args:
            config: 来源配置
        """
        self.config = config
        self.source_name = config.source_name
        self.source_type = config.source_type
        self.processing_mode = config.processing_mode

    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到来源。

        Returns:
            bool: 如果连接成功则为True

        Raises:
            SourceError: 如果连接失败
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        断开与来源的连接。
        """
        pass

    @abstractmethod
    async def discover_documents(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = None
    ) -> SourceResult:
        """
        发现文档。

        Args:
            cursor: 分页游标（可选）
            limit: 限制数量（可选）

        Returns:
            SourceResult: 发现结果

        Raises:
            SourceError: 如果发现失败
        """
        pass

    @abstractmethod
    async def fetch_document(
        self,
        document_ref: DocumentReference
    ) -> bytes:
        """
        获取文档内容。

        Args:
            document_ref: 文档引用

        Returns:
            bytes: 文档内容

        Raises:
            SourceError: 如果获取失败
        """
        pass

    @abstractmethod
    async def validate_document(
        self,
        document_ref: DocumentReference
    ) -> bool:
        """
        验证文档。

        Args:
            document_ref: 文档引用

        Returns:
            bool: 如果文档有效则为True

        Raises:
            SourceError: 如果验证失败
        """
        pass

    async def fetch_documents_batch(
        self,
        document_refs: List[DocumentReference]
    ) -> Dict[str, Union[bytes, Exception]]:
        """
        批量获取文档。

        Args:
            document_refs: 文档引用列表

        Returns:
            Dict[str, Union[bytes, Exception]]: 文档内容映射
        """
        results = {}
        for ref in document_refs:
            try:
                content = await self.fetch_document(ref)
                results[ref.source_id] = content
            except Exception as e:
                results[ref.source_id] = e
        return results

    async def monitor_changes(self) -> AsyncIterator[DocumentReference]:
        """
        监控变化（如果支持）。

        Yields:
            DocumentReference: 变化的文档引用
        """
        # 默认实现不支持监控
        return
        yield

    def supports_monitoring(self) -> bool:
        """
        检查是否支持变化监控。

        Returns:
            bool: 如果支持监控则为True
        """
        return False

    def supports_pagination(self) -> bool:
        """
        检查是否支持分页。

        Returns:
            bool: 如果支持分页则为True
        """
        return True

    def get_source_info(self) -> Dict[str, Any]:
        """
        获取来源信息。

        Returns:
            Dict[str, Any]: 来源信息
        """
        return {
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "processing_mode": self.processing_mode.value,
            "endpoint": self.config.endpoint,
            "supports_monitoring": self.supports_monitoring(),
            "supports_pagination": self.supports_pagination(),
            "config": {
                "enable_crawling": self.config.enable_crawling,
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "retry_attempts": self.config.retry_attempts,
                "enable_cache": self.config.enable_cache
            }
        }


class SourceError(Exception):
    """来源错误引发的异常。"""

    def __init__(
        self,
        message: str,
        source: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.source = source
        self.error_code = error_code


# 导出AsyncIterator
try:
    from typing import AsyncIterator
except ImportError:
    # Python < 3.5.2的兼容性
    import typing
    typing.AsyncIterator = typing.AsyncGenerator