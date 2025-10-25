"""
聊天服务

基于RAGFlow架构的智能聊天服务，
支持多轮对话、上下文管理、会话持久化等功能。
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.rag_interface import (
    ChatInterface, ChatSession, RAGQuery, RAGResult, ChatException,
    RAGConfig, GenerationMode
)
from .retrieval_engine import RetrievalEngine
from .context_builder import ContextBuilder
from .generation_service import GenerationService


class SessionStatus(Enum):
    """会话状态。"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MessageRole(Enum):
    """消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """聊天消息。"""

    message_id: str
    session_id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[Dict[str, int]] = None
    context_chunks: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    feedback_score: Optional[float] = None

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "token_usage": self.token_usage,
            "context_chunks": self.context_chunks,
            "generation_time": self.generation_time,
            "feedback_score": self.feedback_score
        }


@dataclass
class ChatConfig:
    """聊天配置。"""

    # 会话配置
    max_messages_per_session: int = 100
    max_session_duration_hours: int = 24 * 7  # 7天
    auto_archive_inactive_sessions: bool = True
    inactive_threshold_hours: int = 24

    # 对话配置
    max_conversation_history: int = 10
    context_window_size: int = 4000
    enable_context_continuity: bool = True

    # RAG配置
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    enable_conversation_aware_search: bool = True

    # 生成配置
    temperature: float = 0.7
    max_tokens: int = 2048
    generation_mode: GenerationMode = GenerationMode.RAG

    # 增强功能
    enable_message_feedback: bool = True
    enable_session_summary: bool = True
    enable_proactive_suggestions: bool = False


@dataclass
class SessionSummary:
    """会话摘要。"""

    session_id: str
    title: str
    summary: str
    key_topics: List[str]
    message_count: int
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ChatService(ChatInterface):
    """聊天服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化聊天服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 依赖服务
        self.retrieval_engine: Optional[RetrievalEngine] = None
        self.context_builder: Optional[ContextBuilder] = None
        self.generation_service: Optional[GenerationService] = None

        # 聊天配置
        self.default_chat_config = ChatConfig(**config.get("chat", {}))

        # 会话管理
        self.sessions: Dict[str, ChatSession] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}

        # 摘要缓存
        self.session_summaries: Dict[str, SessionSummary] = {}

        # 统计信息
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_messages": 0,
            "average_messages_per_session": 0.0,
            "total_feedback_score": 0.0,
            "feedback_count": 0
        }

        # 定时任务
        self.cleanup_task: Optional[asyncio.Task] = None

    async def initialize(
        self,
        retrieval_engine: RetrievalEngine,
        context_builder: ContextBuilder,
        generation_service: GenerationService
    ) -> bool:
        """
        初始化聊天服务。

        Args:
            retrieval_engine: 检索引擎
            context_builder: 上下文构建器
            generation_service: 生成服务

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.retrieval_engine = retrieval_engine
            self.context_builder = context_builder
            self.generation_service = generation_service

            # 启动清理任务
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.logger.info("聊天服务初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"聊天服务初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理聊天服务资源。"""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            self.sessions.clear()
            self.messages.clear()
            self.session_summaries.clear()

            self.logger.info("聊天服务资源清理完成")

        except Exception as e:
            self.logger.error(f"聊天服务清理失败: {e}")

    async def create_session(
        self,
        user_id: str,
        tenant_id: str,
        title: Optional[str] = None,
        knowledge_bases: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """
        创建聊天会话。

        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            title: 会话标题
            knowledge_bases: 知识库列表
            config: 会话配置
            metadata: 元数据

        Returns:
            ChatSession: 聊天会话
        """
        try:
            session_id = str(uuid.uuid4())

            # 创建会话对象
            session = ChatSession(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                title=title or f"新对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                knowledge_bases=knowledge_bases or [],
                config=config or {},
                metadata=metadata or {}
            )

            # 存储会话
            self.sessions[session_id] = session
            self.messages[session_id] = []

            # 更新统计
            self.stats["total_sessions"] += 1
            self.stats["active_sessions"] += 1

            self.logger.info(f"创建聊天会话 {session_id}，用户: {user_id}")
            return session

        except Exception as e:
            self.logger.error(f"创建聊天会话失败: {e}")
            raise ChatException(f"创建聊天会话失败: {str(e)}")

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        获取聊天会话。

        Args:
            session_id: 会话ID

        Returns:
            Optional[ChatSession]: 聊天会话
        """
        return self.sessions.get(session_id)

    async def list_sessions(
        self,
        user_id: str,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[SessionStatus] = None
    ) -> List[ChatSession]:
        """
        列出用户的聊天会话。

        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 限制数量
            offset: 偏移量
            status: 状态过滤

        Returns:
            List[ChatSession]: 聊天会话列表
        """
        user_sessions = []

        for session in self.sessions.values():
            if session.user_id == user_id and session.tenant_id == tenant_id:
                if status is None or self._get_session_status(session) == status:
                    user_sessions.append(session)

        # 按更新时间排序
        user_sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return user_sessions[offset:offset + limit]

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
        try:
            # 检查会话是否存在
            session = self.sessions.get(session_id)
            if not session:
                raise ChatException(f"会话 {session_id} 不存在")

            # 创建用户消息
            user_message = ChatMessage(
                session_id=session_id,
                role=MessageRole.USER,
                content=message
            )
            self.messages[session_id].append(user_message)

            # 合并配置
            chat_config = ChatConfig(**{**self.default_chat_config.__dict__, **(config or {})})

            # 构建对话历史
            conversation_history = self._build_conversation_history(session_id, chat_config)

            # 执行RAG查询
            rag_query = RAGQuery(
                query_id=str(uuid.uuid4()),
                query=message,
                conversation_history=conversation_history,
                user_id=session.user_id,
                tenant_id=session.tenant_id,
                session_id=session_id,
                top_k=chat_config.retrieval_top_k,
                similarity_threshold=chat_config.similarity_threshold
            )

            # 检索相关文档
            retrieval_result = await self.retrieval_engine.retrieve(
                query=message,
                top_k=chat_config.retrieval_top_k,
                knowledge_bases=session.knowledge_bases
            )

            # 构建上下文
            context_result = await self.context_builder.build_context(
                chunks=retrieval_result.chunks,
                query=message,
                rag_query=rag_query
            )

            # 生成回答
            generation_result = await self.generation_service.generate(
                query=message,
                context=context_result.formatted_context,
                conversation_history=conversation_history,
                rag_query=rag_query
            )

            # 创建助手消息
            assistant_message = ChatMessage(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=generation_result.answer,
                token_usage=generation_result.token_usage,
                context_chunks=[chunk.chunk_id for chunk in context_result.context_chunks],
                generation_time=generation_result.generation_time
            )
            self.messages[session_id].append(assistant_message)

            # 更新会话时间
            session.updated_at = datetime.now()
            session.messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in self.messages[session_id]
            ]

            # 更新统计
            self.stats["total_messages"] += 2  # 用户消息 + 助手消息

            # 创建RAG结果
            rag_result = RAGResult(
                query_id=rag_query.query_id,
                query=message,
                retrieval_result=retrieval_result,
                generation_result=generation_result,
                total_time=retrieval_result.search_time + generation_result.generation_time,
                success=True,
                metadata={
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "message_count": len(self.messages[session_id])
                }
            )

            # 自动生成会话标题（如果是第一条消息）
            if len(self.messages[session_id]) == 2:  # 用户 + 助手
                await self._generate_session_title(session_id, message, generation_result.answer)

            self.logger.info(f"聊天完成，会话: {session_id}，消息: {len(self.messages[session_id])}")

            return rag_result

        except Exception as e:
            self.logger.error(f"聊天失败: {e}")
            raise ChatException(f"聊天失败: {str(e)}")

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天。

        Args:
            session_id: 会话ID
            message: 消息内容
            config: 配置参数

        Yields:
            str: 流式回复内容
        """
        try:
            # 检查会话是否存在
            session = self.sessions.get(session_id)
            if not session:
                raise ChatException(f"会话 {session_id} 不存在")

            # 创建用户消息
            user_message = ChatMessage(
                session_id=session_id,
                role=MessageRole.USER,
                content=message
            )
            self.messages[session_id].append(user_message)

            # 合并配置
            chat_config = ChatConfig(**{**self.default_chat_config.__dict__, **(config or {})})

            # 构建对话历史
            conversation_history = self._build_conversation_history(session_id, chat_config)

            # 检索相关文档
            retrieval_result = await self.retrieval_engine.retrieve(
                query=message,
                top_k=chat_config.retrieval_top_k,
                knowledge_bases=session.knowledge_bases
            )

            # 构建上下文
            context_result = await self.context_builder.build_context(
                chunks=retrieval_result.chunks,
                query=message
            )

            # 流式生成
            full_response = ""
            start_time = datetime.now()

            async for chunk in self.generation_service.generate_stream(
                query=message,
                context=context_result.formatted_context,
                conversation_history=conversation_history
            ):
                full_response += chunk.delta
                yield chunk.delta

            # 创建助手消息
            assistant_message = ChatMessage(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                generation_time=(datetime.now() - start_time).total_seconds()
            )
            self.messages[session_id].append(assistant_message)

            # 更新会话
            session.updated_at = datetime.now()
            session.messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in self.messages[session_id]
            ]

            # 自动生成会话标题
            if len(self.messages[session_id]) == 2:
                await self._generate_session_title(session_id, message, full_response)

        except Exception as e:
            self.logger.error(f"流式聊天失败: {e}")
            yield f"抱歉，聊天过程中出现错误: {str(e)}"

    async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        删除聊天会话。

        Args:
            session_id: 会话ID
            user_id: 用户ID（权限验证）

        Returns:
            bool: 删除是否成功
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                return False

            # 权限验证
            if user_id and session.user_id != user_id:
                raise ChatException("权限不足")

            # 删除会话相关数据
            del self.sessions[session_id]
            if session_id in self.messages:
                del self.messages[session_id]
            if session_id in self.session_summaries:
                del self.session_summaries[session_id]

            # 更新统计
            if self.stats["active_sessions"] > 0:
                self.stats["active_sessions"] -= 1

            self.logger.info(f"删除聊天会话 {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"删除聊天会话失败: {e}")
            return False

    async def archive_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        归档聊天会话。

        Args:
            session_id: 会话ID
            user_id: 用户ID（权限验证）

        Returns:
            bool: 归档是否成功
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                return False

            # 权限验证
            if user_id and session.user_id != user_id:
                raise ChatException("权限不足")

            # 生成会话摘要
            if self.default_chat_config.enable_session_summary:
                await self._generate_session_summary(session_id)

            # 更新会话状态
            session.metadata["archived"] = True
            session.metadata["archived_at"] = datetime.now().isoformat()

            # 更新统计
            if self.stats["active_sessions"] > 0:
                self.stats["active_sessions"] -= 1

            self.logger.info(f"归档聊天会话 {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"归档聊天会话失败: {e}")
            return False

    async def add_message_feedback(
        self,
        session_id: str,
        message_id: str,
        score: float,
        feedback: Optional[str] = None
    ) -> bool:
        """
        添加消息反馈。

        Args:
            session_id: 会话ID
            message_id: 消息ID
            score: 反馈分数（0-1）
            feedback: 反馈内容

        Returns:
            bool: 添加是否成功
        """
        try:
            if session_id not in self.messages:
                return False

            # 查找消息
            for message in self.messages[session_id]:
                if message.message_id == message_id:
                    message.feedback_score = score
                    if feedback:
                        message.metadata["feedback"] = feedback

                    # 更新统计
                    self.stats["feedback_count"] += 1
                    total_score = self.stats["total_feedback_score"] * (self.stats["feedback_count"] - 1) + score
                    self.stats["total_feedback_score"] = total_score / self.stats["feedback_count"]

                    self.logger.info(f"添加消息反馈 {message_id}，分数: {score}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"添加消息反馈失败: {e}")
            return False

    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        role: Optional[MessageRole] = None
    ) -> List[ChatMessage]:
        """
        获取会话消息。

        Args:
            session_id: 会话ID
            limit: 限制数量
            offset: 偏移量
            role: 角色过滤

        Returns:
            List[ChatMessage]: 消息列表
        """
        if session_id not in self.messages:
            return []

        messages = self.messages[session_id]

        # 角色过滤
        if role:
            messages = [msg for msg in messages if msg.role == role]

        # 按时间排序
        messages.sort(key=lambda m: m.timestamp)

        return messages[offset:offset + limit]

    async def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """
        获取会话摘要。

        Args:
            session_id: 会话ID

        Returns:
            Optional[SessionSummary]: 会话摘要
        """
        # 如果已有摘要，直接返回
        if session_id in self.session_summaries:
            return self.session_summaries[session_id]

        # 实时生成摘要
        return await self._generate_session_summary(session_id)

    def _build_conversation_history(
        self,
        session_id: str,
        config: ChatConfig
    ) -> List[Dict[str, str]]:
        """构建对话历史。"""
        if session_id not in self.messages:
            return []

        messages = self.messages[session_id]

        # 限制历史记录数量
        history_limit = min(config.max_conversation_history * 2, len(messages))
        recent_messages = messages[-history_limit:]

        # 转换为标准格式
        conversation_history = []
        for msg in recent_messages:
            conversation_history.append({
                "role": msg.role.value,
                "content": msg.content
            })

        return conversation_history

    async def _generate_session_title(self, session_id: str, user_message: str, assistant_response: str) -> None:
        """生成会话标题。"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return

            # 简化实现：使用用户消息的前50个字符作为标题
            title = user_message[:50] + ("..." if len(user_message) > 50 else "")

            # 如果太短，结合助手回复
            if len(title) < 20:
                combined = user_message + " " + assistant_response
                title = combined[:50] + ("..." if len(combined) > 50 else "")

            session.title = title
            self.logger.info(f"生成会话标题: {title}")

        except Exception as e:
            self.logger.error(f"生成会话标题失败: {e}")

    async def _generate_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """生成会话摘要。"""
        try:
            session = self.sessions.get(session_id)
            if not session or session_id not in self.messages:
                return None

            messages = self.messages[session_id]
            if not messages:
                return None

            # 提取关键信息
            user_messages = [msg.content for msg in messages if msg.role == MessageRole.USER]
            key_topics = self._extract_key_topics(user_messages)

            # 生成摘要（简化实现）
            summary_text = f"包含 {len(messages)} 条消息的对话，主要涉及: {', '.join(key_topics[:3])}"

            summary = SessionSummary(
                session_id=session_id,
                title=session.title,
                summary=summary_text,
                key_topics=key_topics,
                message_count=len(messages)
            )

            # 缓存摘要
            self.session_summaries[session_id] = summary

            return summary

        except Exception as e:
            self.logger.error(f"生成会话摘要失败: {e}")
            return None

    def _extract_key_topics(self, messages: List[str]) -> List[str]:
        """提取关键主题。"""
        # 简化实现：基于词频提取关键词
        import re
        from collections import Counter

        # 合并所有消息
        all_text = " ".join(messages)

        # 提取中文词汇
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', all_text)
        # 提取英文词汇
        english_words = re.findall(r'[a-zA-Z]+', all_text)

        all_words = chinese_words + english_words

        # 过滤停用词（简化）
        stop_words = {'的', '了', '是', '在', '我', '你', '他', '她', '我们', '你们', '他们', '这', '那', '这个', '那个', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [word for word in all_words if len(word) > 1 and word.lower() not in stop_words]

        # 统计词频
        word_count = Counter(filtered_words)

        # 返回前5个高频词
        return [word for word, count in word_count.most_common(5)]

    def _get_session_status(self, session: ChatSession) -> SessionStatus:
        """获取会话状态。"""
        # 检查是否已归档
        if session.metadata.get("archived"):
            return SessionStatus.ARCHIVED

        # 检查是否活跃
        inactive_threshold = timedelta(hours=self.default_chat_config.inactive_threshold_hours)
        if datetime.now() - session.updated_at > inactive_threshold:
            return SessionStatus.INACTIVE

        return SessionStatus.ACTIVE

    async def _cleanup_loop(self) -> None:
        """清理循环。"""
        while True:
            try:
                await self._cleanup_inactive_sessions()
                await self._cleanup_old_messages()
                await asyncio.sleep(3600)  # 每小时清理一次
            except Exception as e:
                self.logger.error(f"清理循环异常: {e}")
                await asyncio.sleep(300)  # 5分钟后重试

    async def _cleanup_inactive_sessions(self) -> None:
        """清理非活跃会话。"""
        if not self.default_chat_config.auto_archive_inactive_sessions:
            return

        inactive_threshold = timedelta(hours=self.default_chat_config.inactive_threshold_hours)
        current_time = datetime.now()

        sessions_to_archive = []
        for session_id, session in self.sessions.items():
            if current_time - session.updated_at > inactive_threshold:
                if self._get_session_status(session) == SessionStatus.INACTIVE:
                    sessions_to_archive.append(session_id)

        for session_id in sessions_to_archive:
            await self.archive_session(session_id)

        if sessions_to_archive:
            self.logger.info(f"归档了 {len(sessions_to_archive)} 个非活跃会话")

    async def _cleanup_old_messages(self) -> None:
        """清理旧消息。"""
        # 这里可以实现消息的清理逻辑，比如删除超过保留期的消息
        pass

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        # 计算平均消息数
        avg_messages = 0.0
        if self.sessions:
            total_messages = sum(len(self.messages.get(session_id, [])) for session_id in self.sessions)
            avg_messages = total_messages / len(self.sessions)

        avg_feedback_score = 0.0
        if self.stats["feedback_count"] > 0:
            avg_feedback_score = self.stats["total_feedback_score"] / self.stats["feedback_count"]

        return {
            **self.stats,
            "average_messages_per_session": avg_messages,
            "average_feedback_score": avg_feedback_score,
            "session_summaries_count": len(self.session_summaries),
            "active_session_count": self.stats["active_sessions"]
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "total_sessions": len(self.sessions),
            "total_messages": sum(len(messages) for messages in self.messages.values()),
            "active_sessions": self.stats["active_sessions"],
            "dependencies": {},
            "errors": []
        }

        # 检查依赖服务
        try:
            if self.retrieval_engine:
                retrieval_health = await self.retrieval_engine.health_check()
                health_status["dependencies"]["retrieval_engine"] = retrieval_health.get("status") == "healthy"
        except Exception as e:
            health_status["errors"].append(f"Retrieval engine: {str(e)}")

        try:
            if self.context_builder:
                # 上下文构建器健康检查（简化）
                health_status["dependencies"]["context_builder"] = True
        except Exception as e:
            health_status["errors"].append(f"Context builder: {str(e)}")

        try:
            if self.generation_service:
                generation_health = await self.generation_service.health_check()
                health_status["dependencies"]["generation_service"] = generation_health.get("status") == "healthy"
        except Exception as e:
            health_status["errors"].append(f"Generation service: {str(e)}")

        # 总体状态
        if health_status["errors"]:
            health_status["status"] = "degraded"

        return health_status