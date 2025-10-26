"""
MySQL客户端适配器
为健康检查提供简化的数据库接口
"""
import asyncio
from infrastructure.monitoring.loguru_logger import logger


class MySQLClient:
    """MySQL客户端类 - 适配器"""

    def __init__(self):
        """初始化MySQL客户端"""
        logger.info("初始化MySQL客户端适配器")

    async def execute_query(self, query: str) -> Any:
        """
        执行SQL查询

        Args:
            query: SQL查询语句

        Returns:
            Any: 查询结果
        """
        logger.info(f"执行MySQL查询: {query}")

        # 模拟查询执行
        await asyncio.sleep(0.1)

        if "SELECT 1" in query:
            return [{"1": 1}]

        return []