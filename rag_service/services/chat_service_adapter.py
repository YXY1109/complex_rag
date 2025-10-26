"""
对话服务适配器
为API层提供简化的对话服务接口
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
import time
import uuid
import asyncio

from infrastructure.monitoring.loguru_logger import logger


class ChatService:
    """对话服务类 - API适配器"""

    def __init__(self):
        """初始化对话服务"""
        logger.info("初始化对话服务适配器")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        knowledge_base_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        retrieval_config: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, int]]:
        """
        生成对话回复

        Args:
            messages: 对话消息列表
            model: 使用的模型
            temperature: 生成温度
            max_tokens: 最大token数
            knowledge_base_id: 知识库ID
            conversation_id: 对话ID
            retrieval_config: 检索配置

        Returns:
            tuple: (回复内容, 使用量统计)
        """
        logger.info(f"生成对话回复，模型: {model}, 对话ID: {conversation_id}")

        # 模拟处理时间
        await asyncio.sleep(0.5)

        # 获取最后一条用户消息
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        # 生成模拟回复
        response = f"这是对您问题的回复：{user_message[:50]}... (模拟回复)"

        # 模拟使用量统计
        usage = {
            "prompt_tokens": len(user_message) // 4,
            "completion_tokens": len(response) // 4,
            "total_tokens": (len(user_message) + len(response)) // 4
        }

        return response, usage

    async def generate_stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        knowledge_base_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        retrieval_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        生成流式对话回复

        Args:
            messages: 对话消息列表
            model: 使用的模型
            temperature: 生成温度
            max_tokens: 最大token数
            knowledge_base_id: 知识库ID
            conversation_id: 对话ID
            retrieval_config: 检索配置

        Yields:
            str: 流式响应块
        """
        logger.info(f"生成流式对话回复，模型: {model}, 对话ID: {conversation_id}")

        # 获取最后一条用户消息
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        # 生成模拟回复
        full_response = f"这是对您问题的流式回复：{user_message[:50]}... (模拟流式回复)"

        # 分块发送
        words = full_response.split()
        chunk_data = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": None
            }]
        }

        for i, word in enumerate(words):
            chunk_data["choices"][0]["delta"]["content"] = " " + word if i > 0 else word
            import json
            yield json.dumps(chunk_data, ensure_ascii=False)
            await asyncio.sleep(0.1)  # 模拟流式延迟

        # 发送结束标记
        chunk_data["choices"][0]["delta"] = {}
        chunk_data["choices"][0]["finish_reason"] = "stop"
        import json
        yield json.dumps(chunk_data, ensure_ascii=False)

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            conversation_id: 对话ID

        Returns:
            List[Dict[str, Any]]: 对话历史
        """
        logger.info(f"获取对话历史: {conversation_id}")

        # 模拟返回历史记录
        return [
            {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": "你好，这是一个测试消息",
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "你好！我是AI助手，很高兴为您服务。",
                "timestamp": "2024-01-01T10:00:01Z"
            }
        ]

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除对话

        Args:
            conversation_id: 对话ID

        Returns:
            bool: 删除是否成功
        """
        logger.info(f"删除对话: {conversation_id}")

        # 模拟删除操作
        await asyncio.sleep(0.1)
        return True