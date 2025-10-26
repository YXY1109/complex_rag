"""
Milvus客户端适配器
为健康检查提供简化的向量数据库接口
"""
import asyncio
from infrastructure.monitoring.loguru_logger import logger


class MilvusClient:
    """Milvus客户端类 - 适配器"""

    def __init__(self):
        """初始化Milvus客户端"""
        logger.info("初始化Milvus客户端适配器")

    async def health_check(self) -> bool:
        """
        检查Milvus健康状态

        Returns:
            bool: 是否健康
        """
        logger.info("检查Milvus健康状态")

        # 模拟健康检查
        await asyncio.sleep(0.1)
        return True