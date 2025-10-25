"""
Memory服务

提供对话记忆管理服务实现（Mem0集成）。
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from ..infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("rag_service.memory_service")


class MemoryService:
    """
    Memory服务类

    封装Mem0接口，提供统一的对话记忆管理服务。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Memory服务

        Args:
            config: Memory配置
        """
        self.config = config
        self.mem0_client = None
        self.enabled = config.get("enabled", True)
        self.provider = config.get("provider", "mem0")
        self.max_tokens = config.get("max_tokens", 2000)
        self.ttl = config.get("ttl", 3600)  # 1小时
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.max_memories_per_user = config.get("max_memories_per_user", 1000)

        # 内存存储（简单实现，生产环境应使用数据库）
        self._memory_store: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_index: Dict[str, List[str]] = {}  # 用于快速搜索的索引

    async def initialize(self) -> None:
        """初始化Memory服务"""
        try:
            if not self.enabled:
                structured_logger.info("Memory服务已禁用")
                return

            # TODO: 实现Mem0客户端初始化逻辑
            # 这里使用内存存储作为简单实现
            self._cleanup_expired_memories()

            structured_logger.info(
                "Memory服务初始化完成",
                extra={
                    "provider": self.provider,
                    "enabled": self.enabled,
                    "max_tokens": self.max_tokens,
                    "ttl": self.ttl,
                }
            )

        except Exception as e:
            structured_logger.error(
                f"Memory服务初始化失败: {e}",
                extra={
                    "provider": self.provider,
                    "error": str(e),
                }
            )
            raise Exception(f"Failed to initialize Memory service: {e}")

    async def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相关记忆

        Args:
            query: 查询字符串
            user_id: 用户ID
            limit: 返回结果数量
            threshold: 相似度阈值

        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        if not self.enabled:
            return []

        try:
            user_id = user_id or "default"
            threshold = threshold or self.similarity_threshold

            # 获取用户的记忆
            user_memories = self._memory_store.get(user_id, [])

            # 简单的文本匹配搜索（生产环境应使用向量搜索）
            matching_memories = []
            query_lower = query.lower()

            for memory in user_memories:
                # 检查记忆是否过期
                if self._is_memory_expired(memory):
                    continue

                # 计算简单的文本相似度
                content_lower = memory["content"].lower()
                similarity = self._calculate_text_similarity(query_lower, content_lower)

                if similarity >= threshold:
                    matching_memories.append({
                        **memory,
                        "similarity_score": similarity,
                    })

            # 按相似度排序并限制数量
            matching_memories.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matching_memories[:limit]

        except Exception as e:
            structured_logger.error(
                f"记忆搜索失败: {e}",
                extra={
                    "user_id": user_id,
                    "query": query[:100],  # 只记录前100个字符
                    "error": str(e),
                }
            )
            return []

    async def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加新记忆

        Args:
            content: 记忆内容
            user_id: 用户ID
            metadata: 元数据

        Returns:
            Dict[str, Any]: 添加结果
        """
        if not self.enabled:
            return {"success": False, "message": "Memory service is disabled"}

        try:
            user_id = user_id or "default"

            # 检查内容长度
            if len(content) > self.max_tokens * 4:  # 粗略估算token数
                content = content[:self.max_tokens * 4]

            # 创建记忆对象
            memory = {
                "id": str(uuid.uuid4()),
                "content": content,
                "user_id": user_id,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=self.ttl)).isoformat(),
            }

            # 添加到存储
            if user_id not in self._memory_store:
                self._memory_store[user_id] = []

            # 检查用户记忆数量限制
            if len(self._memory_store[user_id]) >= self.max_memories_per_user:
                # 删除最旧的记忆
                oldest_memory = min(self._memory_store[user_id], key=lambda x: x["created_at"])
                self._memory_store[user_id].remove(oldest_memory)
                structured_logger.info(
                    f"用户 {user_id} 记忆数量超限，删除最旧记忆",
                    extra={"memory_id": oldest_memory["id"]}
                )

            self._memory_store[user_id].append(memory)

            # 更新搜索索引
            self._update_search_index(user_id, memory)

            structured_logger.info(
                "添加新记忆",
                extra={
                    "memory_id": memory["id"],
                    "user_id": user_id,
                    "content_length": len(content),
                }
            )

            return {
                "success": True,
                "memory_id": memory["id"],
                "created_at": memory["created_at"],
            }

        except Exception as e:
            structured_logger.error(
                f"添加记忆失败: {e}",
                extra={
                    "user_id": user_id,
                    "content_length": len(content) if content else 0,
                    "error": str(e),
                }
            )
            return {
                "success": False,
                "message": f"Failed to add memory: {str(e)}"
            }

    async def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID
            user_id: 用户ID

        Returns:
            bool: 删除是否成功
        """
        if not self.enabled:
            return False

        try:
            user_id = user_id or "default"

            if user_id not in self._memory_store:
                return False

            # 查找并删除记忆
            for i, memory in enumerate(self._memory_store[user_id]):
                if memory["id"] == memory_id:
                    deleted_memory = self._memory_store[user_id].pop(i)
                    self._remove_from_search_index(user_id, deleted_memory)

                    structured_logger.info(
                        "删除记忆",
                        extra={
                            "memory_id": memory_id,
                            "user_id": user_id,
                        }
                    )
                    return True

            return False

        except Exception as e:
            structured_logger.error(
                f"删除记忆失败: {e}",
                extra={
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "error": str(e),
                }
            )
            return False

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            metadata: 新元数据
            user_id: 用户ID

        Returns:
            Dict[str, Any]: 更新结果
        """
        if not self.enabled:
            return {"success": False, "message": "Memory service is disabled"}

        try:
            user_id = user_id or "default"

            if user_id not in self._memory_store:
                return {"success": False, "message": "User not found"}

            # 查找记忆
            for i, memory in enumerate(self._memory_store[user_id]):
                if memory["id"] == memory_id:
                    # 更新记忆
                    if content is not None:
                        if len(content) > self.max_tokens * 4:
                            content = content[:self.max_tokens * 4]
                        memory["content"] = content

                    if metadata is not None:
                        memory["metadata"].update(metadata)

                    memory["updated_at"] = datetime.utcnow().isoformat()

                    # 更新搜索索引
                    self._update_search_index(user_id, memory)

                    structured_logger.info(
                        "更新记忆",
                        extra={
                            "memory_id": memory_id,
                            "user_id": user_id,
                            "content_updated": content is not None,
                        }
                    )

                    return {
                        "success": True,
                        "memory_id": memory_id,
                        "updated_at": memory["updated_at"],
                    }

            return {"success": False, "message": "Memory not found"}

        except Exception as e:
            structured_logger.error(
                f"更新记忆失败: {e}",
                extra={
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "error": str(e),
                }
            )
            return {
                "success": False,
                "message": f"Failed to update memory: {str(e)}"
            }

    async def get_memory_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取记忆历史

        Args:
            user_id: 用户ID
            limit: 返回结果数量
            offset: 偏移量

        Returns:
            List[Dict[str, Any]]: 记忆历史
        """
        if not self.enabled:
            return []

        try:
            user_id = user_id or "default"

            user_memories = self._memory_store.get(user_id, [])

            # 过滤过期记忆
            valid_memories = [
                memory for memory in user_memories
                if not self._is_memory_expired(memory)
            ]

            # 按创建时间倒序排序
            valid_memories.sort(key=lambda x: x["created_at"], reverse=True)

            # 应用分页
            start = offset
            end = start + limit
            return valid_memories[start:end]

        except Exception as e:
            structured_logger.error(
                f"获取记忆历史失败: {e}",
                extra={
                    "user_id": user_id,
                    "limit": limit,
                    "offset": offset,
                    "error": str(e),
                }
            )
            return []

    async def clear_all_memories(self, user_id: Optional[str] = None) -> bool:
        """
        清除所有记忆

        Args:
            user_id: 用户ID

        Returns:
            bool: 清除是否成功
        """
        if not self.enabled:
            return False

        try:
            user_id = user_id or "default"

            if user_id in self._memory_store:
                memory_count = len(self._memory_store[user_id])
                del self._memory_store[user_id]

                if user_id in self._memory_index:
                    del self._memory_index[user_id]

                structured_logger.info(
                    "清除所有记忆",
                    extra={
                        "user_id": user_id,
                        "cleared_count": memory_count,
                    }
                )
                return True

            return False

        except Exception as e:
            structured_logger.error(
                f"清除记忆失败: {e}",
                extra={
                    "user_id": user_id,
                    "error": str(e),
                }
            )
            return False

    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取记忆统计信息

        Args:
            user_id: 用户ID

        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            user_id = user_id or "default"

            user_memories = self._memory_store.get(user_id, [])
            valid_memories = [
                memory for memory in user_memories
                if not self._is_memory_expired(memory)
            ]

            total_chars = sum(len(memory["content"]) for memory in valid_memories)
            avg_chars = total_chars / len(valid_memories) if valid_memories else 0

            return {
                "enabled": self.enabled,
                "user_id": user_id,
                "total_memories": len(user_memories),
                "valid_memories": len(valid_memories),
                "expired_memories": len(user_memories) - len(valid_memories),
                "total_characters": total_chars,
                "average_characters_per_memory": round(avg_chars, 2),
                "max_tokens": self.max_tokens,
                "ttl_seconds": self.ttl,
            }

        except Exception as e:
            structured_logger.error(
                f"获取记忆统计失败: {e}",
                extra={
                    "user_id": user_id,
                    "error": str(e),
                }
            )
            return {"enabled": self.enabled, "error": str(e)}

    def _is_memory_expired(self, memory: Dict[str, Any]) -> bool:
        """检查记忆是否过期"""
        try:
            expires_at = datetime.fromisoformat(memory["expires_at"])
            return datetime.utcnow() > expires_at
        except (KeyError, ValueError):
            return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算简单的文本相似度"""
        # 使用Jaccard相似度
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _update_search_index(self, user_id: str, memory: Dict[str, Any]) -> None:
        """更新搜索索引"""
        if user_id not in self._memory_index:
            self._memory_index[user_id] = []

        # 简单的关键词索引
        words = memory["content"].lower().split()
        for word in words:
            if word not in self._memory_index[user_id]:
                self._memory_index[user_id].append(word)

    def _remove_from_search_index(self, user_id: str, memory: Dict[str, Any]) -> None:
        """从搜索索引中移除记忆"""
        # 简单实现：重建索引
        if user_id in self._memory_store:
            self._memory_index[user_id] = []
            for mem in self._memory_store[user_id]:
                self._update_search_index(user_id, mem)

    def _cleanup_expired_memories(self) -> None:
        """清理过期记忆"""
        try:
            expired_count = 0
            for user_id, memories in list(self._memory_store.items()):
                valid_memories = [
                    memory for memory in memories
                    if not self._is_memory_expired(memory)
                ]

                expired_count += len(memories) - len(valid_memories)
                self._memory_store[user_id] = valid_memories

            if expired_count > 0:
                structured_logger.info(
                    "清理过期记忆",
                    extra={"expired_count": expired_count}
                )

        except Exception as e:
            structured_logger.error(f"清理过期记忆失败: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            if not self.enabled:
                return {
                    "status": "disabled",
                    "provider": self.provider,
                    "message": "Memory service is disabled",
                }

            # 统计信息
            total_users = len(self._memory_store)
            total_memories = sum(len(memories) for memories in self._memory_store.values())

            # 测试基本功能
            test_memory = await self.add_memory(
                content="Health check test memory",
                user_id="health_check",
                metadata={"test": True}
            )

            search_result = await self.search_memories(
                query="health check",
                user_id="health_check",
                limit=1
            )

            await self.delete_memory(test_memory["memory_id"], "health_check")

            success = (
                test_memory.get("success", False) and
                len(search_result) > 0 and
                search_result[0].get("similarity_score", 0) > 0
            )

            return {
                "status": "healthy" if success else "unhealthy",
                "provider": self.provider,
                "total_users": total_users,
                "total_memories": total_memories,
                "max_tokens": self.max_tokens,
                "ttl_seconds": self.ttl,
                "test_result": success,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 清理过期记忆
            self._cleanup_expired_memories()

            # 清理内存存储
            self._memory_store.clear()
            self._memory_index.clear()

            structured_logger.info("Memory服务资源清理完成")

        except Exception as e:
            structured_logger.error(f"Memory服务资源清理失败: {e}")