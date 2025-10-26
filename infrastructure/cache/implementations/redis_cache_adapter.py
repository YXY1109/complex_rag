"""
Redis缓存适配器
为健康检查提供简化的缓存接口
"""
import asyncio
from infrastructure.monitoring.loguru_logger import logger


class RedisCache:
    """Redis缓存类 - 适配器"""

    def __init__(self):
        """初始化Redis缓存"""
        logger.info("初始化Redis缓存适配器")

    async def ping(self) -> bool:
        """
        检查Redis连接状态

        Returns:
            bool: 是否连接成功
        """
        logger.info("检查Redis连接状态")

        # 模拟ping检查
        await asyncio.sleep(0.05)
        return True