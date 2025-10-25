"""
知识管理服务

基于RAGFlow架构的知识库管理服务，
提供知识库的创建、管理、文档处理、权限控制等功能。
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.rag_interface import (
    KnowledgeBase, KnowledgeManagerInterface, KnowledgeBaseException,
    DocumentChunk, RAGConfig
)
from ...infrastructure.database.implementations.relational.mysql_client import MySQLClient
from ...services.vector_store import VectorStore
from ...services.embedding_service import EmbeddingService


class KnowledgeBaseStatus(Enum):
    """知识库状态。"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PROCESSING = "processing"
    ERROR = "error"
    ARCHIVED = "archived"


class DocumentStatus(Enum):
    """文档状态。"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class PermissionLevel(Enum):
    """权限级别。"""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


@dataclass
class Document:
    """文档信息。"""

    document_id: str
    kb_id: str
    title: str
    content: str
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())


@dataclass
class KnowledgeBasePermission:
    """知识库权限。"""

    permission_id: str
    kb_id: str
    user_id: str
    permission_level: PermissionLevel
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.permission_id:
            self.permission_id = str(uuid.uuid4())


class KnowledgeManager(KnowledgeManagerInterface):
    """知识管理服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化知识管理服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 数据库客户端
        self.db_client: Optional[MySQLClient] = None

        # 依赖服务
        self.vector_store: Optional[VectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None

        # 缓存
        self.kb_cache: Dict[str, KnowledgeBase] = {}
        self.permission_cache: Dict[str, Dict[str, PermissionLevel]] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)

        # 配置
        self.max_kb_per_tenant = config.get("max_kb_per_tenant", 100)
        self.max_documents_per_kb = config.get("max_documents_per_kb", 10000)
        self.max_file_size = config.get("max_file_size", 100 * 1024 * 1024)  # 100MB

    async def initialize(
        self,
        db_client: MySQLClient,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ) -> bool:
        """
        初始化知识管理服务。

        Args:
            db_client: 数据库客户端
            vector_store: 向量存储服务
            embedding_service: 嵌入服务

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.db_client = db_client
            self.vector_store = vector_store
            self.embedding_service = embedding_service

            # 创建数据库表
            await self._create_tables()

            self.logger.info("知识管理服务初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"知识管理服务初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理知识管理服务资源。"""
        try:
            self.kb_cache.clear()
            self.permission_cache.clear()
            self.logger.info("知识管理服务资源清理完成")

        except Exception as e:
            self.logger.error(f"知识管理服务清理失败: {e}")

    async def _create_tables(self) -> None:
        """创建数据库表。"""
        if not self.db_client:
            raise KnowledgeBaseException("数据库客户端未初始化")

        # 知识库表
        kb_table_sql = """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            kb_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            tenant_id VARCHAR(255) NOT NULL,
            document_count INT DEFAULT 0,
            chunk_count INT DEFAULT 0,
            embedding_model VARCHAR(255),
            status VARCHAR(50) DEFAULT 'active',
            config JSON,
            created_by VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            metadata JSON,
            INDEX idx_tenant_id (tenant_id),
            INDEX idx_status (status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        # 文档表
        doc_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            document_id VARCHAR(255) PRIMARY KEY,
            kb_id VARCHAR(255) NOT NULL,
            title VARCHAR(500) NOT NULL,
            content LONGTEXT,
            file_path VARCHAR(1000),
            file_type VARCHAR(100),
            file_size BIGINT,
            chunk_count INT DEFAULT 0,
            status VARCHAR(50) DEFAULT 'pending',
            created_by VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            metadata JSON,
            error_message TEXT,
            INDEX idx_kb_id (kb_id),
            INDEX idx_status (status),
            INDEX idx_created_at (created_at),
            INDEX idx_created_by (created_by),
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        # 权限表
        permission_table_sql = """
        CREATE TABLE IF NOT EXISTS kb_permissions (
            permission_id VARCHAR(255) PRIMARY KEY,
            kb_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            permission_level VARCHAR(50) NOT NULL,
            granted_by VARCHAR(255) NOT NULL,
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NULL,
            INDEX idx_kb_id (kb_id),
            INDEX idx_user_id (user_id),
            INDEX idx_permission_level (permission_level),
            UNIQUE KEY unique_kb_user (kb_id, user_id),
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        # 执行创建表
        await self.db_client.execute(kb_table_sql)
        await self.db_client.execute(doc_table_sql)
        await self.db_client.execute(permission_table_sql)

        self.logger.info("数据库表创建完成")

    async def create_knowledge_base(
        self,
        name: str,
        description: str,
        tenant_id: str,
        config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> KnowledgeBase:
        """
        创建知识库。

        Args:
            name: 知识库名称
            description: 描述
            tenant_id: 租户ID
            config: 配置参数
            created_by: 创建者

        Returns:
            KnowledgeBase: 知识库信息
        """
        try:
            # 检查租户知识库数量限制
            await self._check_tenant_kb_limit(tenant_id)

            # 创建知识库对象
            kb = KnowledgeBase(
                name=name,
                description=description,
                tenant_id=tenant_id,
                config=config or {}
            )

            if created_by:
                kb.metadata["created_by"] = created_by

            # 插入数据库
            kb_data = {
                "kb_id": kb.kb_id,
                "name": kb.name,
                "description": kb.description,
                "tenant_id": kb.tenant_id,
                "document_count": kb.document_count,
                "chunk_count": kb.chunk_count,
                "embedding_model": kb.embedding_model,
                "status": KnowledgeBaseStatus.ACTIVE.value,
                "config": json.dumps(kb.config),
                "created_by": created_by,
                "metadata": json.dumps(kb.metadata)
            }

            await self.db_client.insert("knowledge_bases", kb_data)

            # 创建向量集合
            collection_config = {
                "collection_name": f"kb_{kb.kb_id}",
                "dimension": 1536,  # 默认维度，从配置中获取
                "description": f"Collection for knowledge base {kb.name}"
            }

            if self.vector_store:
                await self.vector_store.create_collection(collection_config)

            # 创建者默认获得管理员权限
            if created_by:
                await self._grant_permission(kb.kb_id, created_by, PermissionLevel.ADMIN, created_by)

            # 缓存知识库
            self.kb_cache[kb.kb_id] = kb

            self.logger.info(f"知识库 {kb.name} ({kb.kb_id}) 创建成功")
            return kb

        except Exception as e:
            self.logger.error(f"创建知识库失败: {e}")
            raise KnowledgeBaseException(f"创建知识库失败: {str(e)}")

    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """
        获取知识库信息。

        Args:
            kb_id: 知识库ID

        Returns:
            Optional[KnowledgeBase]: 知识库信息
        """
        # 检查缓存
        if kb_id in self.kb_cache:
            return self.kb_cache[kb_id]

        try:
            # 从数据库查询
            sql = "SELECT * FROM knowledge_bases WHERE kb_id = %s"
            result = await self.db_client.fetch_one(sql, (kb_id,))

            if not result:
                return None

            kb = KnowledgeBase(
                kb_id=result["kb_id"],
                name=result["name"],
                description=result["description"],
                tenant_id=result["tenant_id"],
                document_count=result["document_count"],
                chunk_count=result["chunk_count"],
                embedding_model=result["embedding_model"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
                metadata=json.loads(result["metadata"]) if result["metadata"] else {},
                config=json.loads(result["config"]) if result["config"] else {}
            )

            # 缓存结果
            self.kb_cache[kb_id] = kb
            return kb

        except Exception as e:
            self.logger.error(f"获取知识库失败: {e}")
            return None

    async def list_knowledge_bases(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0,
        status: Optional[KnowledgeBaseStatus] = None,
        user_id: Optional[str] = None
    ) -> List[KnowledgeBase]:
        """
        列出知识库。

        Args:
            tenant_id: 租户ID
            limit: 限制数量
            offset: 偏移量
            status: 状态过滤
            user_id: 用户ID（权限过滤）

        Returns:
            List[KnowledgeBase]: 知识库列表
        """
        try:
            # 构建查询条件
            conditions = ["tenant_id = %s"]
            params = [tenant_id]

            if status:
                conditions.append("status = %s")
                params.append(status.value)

            # 如果指定用户ID，需要检查权限
            if user_id:
                # 这里简化处理，实际应该查询权限表
                conditions.append("(created_by = %s OR kb_id IN (SELECT kb_id FROM kb_permissions WHERE user_id = %s))")
                params.extend([user_id, user_id])

            where_clause = " AND ".join(conditions)
            sql = f"""
            SELECT * FROM knowledge_bases
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])

            results = await self.db_client.fetch_all(sql, tuple(params))

            knowledge_bases = []
            for result in results:
                kb = KnowledgeBase(
                    kb_id=result["kb_id"],
                    name=result["name"],
                    description=result["description"],
                    tenant_id=result["tenant_id"],
                    document_count=result["document_count"],
                    chunk_count=result["chunk_count"],
                    embedding_model=result["embedding_model"],
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                    metadata=json.loads(result["metadata"]) if result["metadata"] else {},
                    config=json.loads(result["config"]) if result["config"] else {}
                )
                knowledge_bases.append(kb)

            return knowledge_bases

        except Exception as e:
            self.logger.error(f"列出知识库失败: {e}")
            return []

    async def update_knowledge_base(
        self,
        kb_id: str,
        updates: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """
        更新知识库。

        Args:
            kb_id: 知识库ID
            updates: 更新内容
            user_id: 操作用户

        Returns:
            bool: 更新是否成功
        """
        try:
            # 检查权限
            if user_id and not await self._check_permission(kb_id, user_id, PermissionLevel.WRITE):
                raise KnowledgeBaseException("权限不足")

            # 构建更新数据
            update_data = {}
            allowed_fields = ["name", "description", "config", "metadata"]

            for field, value in updates.items():
                if field in allowed_fields:
                    if field in ["config", "metadata"]:
                        update_data[field] = json.dumps(value) if value else "{}"
                    else:
                        update_data[field] = value

            if not update_data:
                return True

            update_data["updated_at"] = datetime.now()

            # 执行更新
            await self.db_client.update("knowledge_bases", update_data, "kb_id = %s", (kb_id,))

            # 清除缓存
            if kb_id in self.kb_cache:
                del self.kb_cache[kb_id]

            self.logger.info(f"知识库 {kb_id} 更新成功")
            return True

        except Exception as e:
            self.logger.error(f"更新知识库失败: {e}")
            return False

    async def delete_knowledge_base(self, kb_id: str, user_id: Optional[str] = None) -> bool:
        """
        删除知识库。

        Args:
            kb_id: 知识库ID
            user_id: 操作用户

        Returns:
            bool: 删除是否成功
        """
        try:
            # 检查权限
            if user_id and not await self._check_permission(kb_id, user_id, PermissionLevel.ADMIN):
                raise KnowledgeBaseException("权限不足")

            # 删除向量数据
            if self.vector_store:
                collection_name = f"kb_{kb_id}"
                try:
                    # 这里需要实现向量集合的删除
                    self.logger.info(f"删除向量集合 {collection_name}")
                except Exception as e:
                    self.logger.warning(f"删除向量集合失败: {e}")

            # 删除数据库记录（级联删除文档和权限）
            await self.db_client.delete("knowledge_bases", "kb_id = %s", (kb_id,))

            # 清除缓存
            if kb_id in self.kb_cache:
                del self.kb_cache[kb_id]

            if kb_id in self.permission_cache:
                del self.permission_cache[kb_id]

            self.logger.info(f"知识库 {kb_id} 删除成功")
            return True

        except Exception as e:
            self.logger.error(f"删除知识库失败: {e}")
            return False

    async def add_document(
        self,
        kb_id: str,
        title: str,
        content: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> str:
        """
        添加文档到知识库。

        Args:
            kb_id: 知识库ID
            title: 文档标题
            content: 文档内容
            file_path: 文件路径
            file_type: 文件类型
            file_size: 文件大小
            metadata: 元数据
            created_by: 创建者

        Returns:
            str: 文档ID
        """
        try:
            # 检查权限
            if created_by and not await self._check_permission(kb_id, created_by, PermissionLevel.WRITE):
                raise KnowledgeBaseException("权限不足")

            # 检查知识库是否存在
            kb = await self.get_knowledge_base(kb_id)
            if not kb:
                raise KnowledgeBaseException("知识库不存在")

            # 检查文档数量限制
            await self._check_kb_document_limit(kb_id)

            # 创建文档对象
            doc = Document(
                kb_id=kb_id,
                title=title,
                content=content,
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                created_by=created_by,
                metadata=metadata or {}
            )

            # 插入数据库
            doc_data = {
                "document_id": doc.document_id,
                "kb_id": doc.kb_id,
                "title": doc.title,
                "content": doc.content,
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "status": DocumentStatus.PROCESSING.value,
                "created_by": doc.created_by,
                "metadata": json.dumps(doc.metadata)
            }

            await self.db_client.insert("documents", doc_data)

            # 异步处理文档
            asyncio.create_task(self._process_document(doc.document_id))

            self.logger.info(f"文档 {doc.title} ({doc.document_id}) 添加成功")
            return doc.document_id

        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise KnowledgeBaseException(f"添加文档失败: {str(e)}")

    async def _process_document(self, document_id: str) -> None:
        """
        处理文档（分块、向量化）。

        Args:
            document_id: 文档ID
        """
        try:
            # 获取文档信息
            doc = await self._get_document(document_id)
            if not doc:
                return

            # 更新状态为处理中
            await self._update_document_status(document_id, DocumentStatus.PROCESSING)

            # 文档分块
            chunks = await self._chunk_document(doc)
            if not chunks:
                await self._update_document_status(document_id, DocumentStatus.FAILED, "文档分块失败")
                return

            # 生成嵌入向量
            if self.embedding_service:
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = await self.embedding_service.embed(chunk_texts)

                # 添加到向量存储
                if self.vector_store:
                    collection_name = f"kb_{doc.kb_id}"
                    documents = []
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "content": chunk.content,
                            "metadata": {
                                "document_id": doc.document_id,
                                "chunk_index": chunk.chunk_index,
                                "title": doc.title,
                                "file_type": doc.file_type,
                                **chunk.metadata
                            },
                            "document_id": doc.document_id,
                            "chunk_index": chunk.chunk_index,
                            "collection_name": collection_name
                        })

                    await self.vector_store.add_vectors(embeddings, documents)

            # 更新文档状态和统计
            await self._update_document_status(document_id, DocumentStatus.COMPLETED)
            await self._update_kb_statistics(doc.kb_id, len(chunks))

            self.logger.info(f"文档 {document_id} 处理完成，生成 {len(chunks)} 个块")

        except Exception as e:
            self.logger.error(f"处理文档 {document_id} 失败: {e}")
            await self._update_document_status(document_id, DocumentStatus.FAILED, str(e))

    async def _chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """
        文档分块。

        Args:
            doc: 文档对象

        Returns:
            List[DocumentChunk]: 文档块列表
        """
        try:
            chunks = []
            content = doc.content

            # 简单的分块策略：按段落分割
            paragraphs = content.split('\n\n')
            current_chunk = ""
            chunk_index = 0
            max_chunk_size = 1000  # 最大块大小

            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            content=current_chunk.strip(),
                            document_id=doc.document_id,
                            chunk_index=chunk_index,
                            title=doc.title,
                            metadata={
                                "file_type": doc.file_type,
                                "chunk_size": len(current_chunk)
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                    current_chunk = paragraph + "\n\n"

            # 处理最后一个块
            if current_chunk.strip():
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=current_chunk.strip(),
                    document_id=doc.document_id,
                    chunk_index=chunk_index,
                    title=doc.title,
                    metadata={
                        "file_type": doc.file_type,
                        "chunk_size": len(current_chunk)
                    }
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.error(f"文档分块失败: {e}")
            return []

    async def _get_document(self, document_id: str) -> Optional[Document]:
        """获取文档信息。"""
        try:
            sql = "SELECT * FROM documents WHERE document_id = %s"
            result = await self.db_client.fetch_one(sql, (document_id,))

            if not result:
                return None

            return Document(
                document_id=result["document_id"],
                kb_id=result["kb_id"],
                title=result["title"],
                content=result["content"],
                file_path=result["file_path"],
                file_type=result["file_type"],
                file_size=result["file_size"],
                chunk_count=result["chunk_count"],
                status=DocumentStatus(result["status"]),
                created_by=result["created_by"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
                metadata=json.loads(result["metadata"]) if result["metadata"] else {},
                error_message=result["error_message"]
            )

        except Exception as e:
            self.logger.error(f"获取文档信息失败: {e}")
            return None

    async def _update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> None:
        """更新文档状态。"""
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.now()
            }

            if error_message:
                update_data["error_message"] = error_message

            await self.db_client.update(
                "documents",
                update_data,
                "document_id = %s",
                (document_id,)
            )

        except Exception as e:
            self.logger.error(f"更新文档状态失败: {e}")

    async def _update_kb_statistics(self, kb_id: str, chunk_count: int) -> None:
        """更新知识库统计信息。"""
        try:
            # 获取当前统计
            sql = """
            SELECT
                COUNT(*) as doc_count,
                SUM(chunk_count) as total_chunks
            FROM documents
            WHERE kb_id = %s AND status = 'completed'
            """
            result = await self.db_client.fetch_one(sql, (kb_id,))

            if result:
                update_data = {
                    "document_count": result["doc_count"],
                    "chunk_count": result["total_chunks"] or 0,
                    "updated_at": datetime.now()
                }

                await self.db_client.update(
                    "knowledge_bases",
                    update_data,
                    "kb_id = %s",
                    (kb_id,)
                )

                # 更新缓存
                if kb_id in self.kb_cache:
                    self.kb_cache[kb_id].document_count = result["doc_count"]
                    self.kb_cache[kb_id].chunk_count = result["total_chunks"] or 0

        except Exception as e:
            self.logger.error(f"更新知识库统计失败: {e}")

    async def _check_tenant_kb_limit(self, tenant_id: str) -> None:
        """检查租户知识库数量限制。"""
        try:
            sql = "SELECT COUNT(*) as count FROM knowledge_bases WHERE tenant_id = %s"
            result = await self.db_client.fetch_one(sql, (tenant_id,))

            if result and result["count"] >= self.max_kb_per_tenant:
                raise KnowledgeBaseException(f"租户知识库数量已达到限制 ({self.max_kb_per_tenant})")

        except Exception as e:
            if isinstance(e, KnowledgeBaseException):
                raise
            self.logger.error(f"检查租户知识库限制失败: {e}")

    async def _check_kb_document_limit(self, kb_id: str) -> None:
        """检查知识库文档数量限制。"""
        try:
            sql = "SELECT COUNT(*) as count FROM documents WHERE kb_id = %s"
            result = await self.db_client.fetch_one(sql, (kb_id,))

            if result and result["count"] >= self.max_documents_per_kb:
                raise KnowledgeBaseException(f"知识库文档数量已达到限制 ({self.max_documents_per_kb})")

        except Exception as e:
            if isinstance(e, KnowledgeBaseException):
                raise
            self.logger.error(f"检查知识库文档限制失败: {e}")

    async def _grant_permission(
        self,
        kb_id: str,
        user_id: str,
        permission_level: PermissionLevel,
        granted_by: str,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """授予权限。"""
        try:
            permission = KnowledgeBasePermission(
                kb_id=kb_id,
                user_id=user_id,
                permission_level=permission_level,
                granted_by=granted_by,
                expires_at=expires_at
            )

            permission_data = {
                "permission_id": permission.permission_id,
                "kb_id": permission.kb_id,
                "user_id": permission.user_id,
                "permission_level": permission.permission_level.value,
                "granted_by": permission.granted_by,
                "expires_at": permission.expires_at
            }

            await self.db_client.insert("kb_permissions", permission_data)

            # 清除权限缓存
            if kb_id in self.permission_cache:
                self.permission_cache[kb_id][user_id] = permission_level

            return True

        except Exception as e:
            self.logger.error(f"授予权限失败: {e}")
            return False

    async def _check_permission(
        self,
        kb_id: str,
        user_id: str,
        required_level: PermissionLevel
    ) -> bool:
        """检查权限。"""
        try:
            # 检查缓存
            if kb_id in self.permission_cache and user_id in self.permission_cache[kb_id]:
                user_permission = self.permission_cache[kb_id][user_id]
                return self._has_permission_level(user_permission, required_level)

            # 查询数据库
            sql = """
            SELECT permission_level FROM kb_permissions
            WHERE kb_id = %s AND user_id = %s
            AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY
                CASE permission_level
                    WHEN 'owner' THEN 1
                    WHEN 'admin' THEN 2
                    WHEN 'write' THEN 3
                    WHEN 'read' THEN 4
                END
            LIMIT 1
            """
            result = await self.db_client.fetch_one(sql, (kb_id, user_id))

            if result:
                user_permission = PermissionLevel(result["permission_level"])

                # 缓存权限
                if kb_id not in self.permission_cache:
                    self.permission_cache[kb_id] = {}
                self.permission_cache[kb_id][user_id] = user_permission

                return self._has_permission_level(user_permission, required_level)

            return False

        except Exception as e:
            self.logger.error(f"检查权限失败: {e}")
            return False

    def _has_permission_level(
        self,
        user_permission: PermissionLevel,
        required_level: PermissionLevel
    ) -> bool:
        """检查权限级别是否满足要求。"""
        level_hierarchy = {
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2,
            PermissionLevel.ADMIN: 3,
            PermissionLevel.OWNER: 4
        }

        return level_hierarchy.get(user_permission, 0) >= level_hierarchy.get(required_level, 0)

    async def get_knowledge_base_statistics(self, kb_id: str) -> Dict[str, Any]:
        """获取知识库统计信息。"""
        try:
            kb = await self.get_knowledge_base(kb_id)
            if not kb:
                return {}

            # 获取文档统计
            sql = """
            SELECT
                status,
                COUNT(*) as count,
                SUM(chunk_count) as total_chunks,
                SUM(file_size) as total_size
            FROM documents
            WHERE kb_id = %s
            GROUP BY status
            """
            doc_stats = await self.db_client.fetch_all(sql, (kb_id,))

            # 获取权限统计
            permission_sql = """
            SELECT permission_level, COUNT(*) as count
            FROM kb_permissions
            WHERE kb_id = %s AND (expires_at IS NULL OR expires_at > NOW())
            GROUP BY permission_level
            """
            permission_stats = await self.db_client.fetch_all(permission_sql, (kb_id,))

            return {
                "kb_info": {
                    "kb_id": kb.kb_id,
                    "name": kb.name,
                    "description": kb.description,
                    "created_at": kb.created_at,
                    "updated_at": kb.updated_at
                },
                "document_stats": {
                    "total_documents": kb.document_count,
                    "total_chunks": kb.chunk_count,
                    "by_status": {stat["status"]: stat["count"] for stat in doc_stats}
                },
                "permission_stats": {
                    stat["permission_level"]: stat["count"] for stat in permission_stats
                },
                "storage_stats": {
                    "total_size": sum(stat["total_size"] or 0 for stat in doc_stats)
                }
            }

        except Exception as e:
            self.logger.error(f"获取知识库统计失败: {e}")
            return {}

    async def search_documents(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        在知识库中搜索文档。

        Args:
            kb_id: 知识库ID
            query: 查询字符串
            top_k: 返回结果数量
            filters: 过滤条件
            user_id: 用户ID

        Returns:
            List[DocumentChunk]: 文档块列表
        """
        try:
            # 检查权限
            if user_id and not await self._check_permission(kb_id, user_id, PermissionLevel.READ):
                raise KnowledgeBaseException("权限不足")

            # 使用向量存储搜索
            if self.vector_store and self.embedding_service:
                # 生成查询向量
                query_vector = await self.embedding_service.embed_single(query)

                # 搜索向量
                collection_name = f"kb_{kb_id}"
                search_results = await self.vector_store.search(
                    query_vector=query_vector,
                    top_k=top_k,
                    filters=filters,
                    collection_name=collection_name
                )

                # 转换为DocumentChunk
                chunks = []
                for doc_id, score, metadata in search_results:
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=metadata.get("content", ""),
                        document_id=metadata.get("document_id", ""),
                        chunk_index=metadata.get("chunk_index", 0),
                        title=metadata.get("title", ""),
                        score=score,
                        metadata=metadata
                    )
                    chunks.append(chunk)

                return chunks

            return []

        except Exception as e:
            self.logger.error(f"搜索文档失败: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "database": False,
            "vector_store": False,
            "embedding_service": False,
            "cache_stats": {
                "kb_cache_size": len(self.kb_cache),
                "permission_cache_size": len(self.permission_cache)
            },
            "errors": []
        }

        # 检查数据库连接
        try:
            if self.db_client:
                await self.db_client.execute("SELECT 1")
                health_status["database"] = True
        except Exception as e:
            health_status["errors"].append(f"Database: {str(e)}")

        # 检查向量存储
        try:
            if self.vector_store:
                vs_health = await self.vector_store.health_check()
                health_status["vector_store"] = vs_health.get("milvus", False) or vs_health.get("elasticsearch", False)
        except Exception as e:
            health_status["errors"].append(f"Vector store: {str(e)}")

        # 检查嵌入服务
        try:
            if self.embedding_service:
                embed_health = await self.embedding_service.health_check()
                health_status["embedding_service"] = embed_health.get("status") == "healthy"
        except Exception as e:
            health_status["errors"].append(f"Embedding service: {str(e)}")

        # 总体状态
        if health_status["errors"]:
            health_status["status"] = "degraded"

        return health_status