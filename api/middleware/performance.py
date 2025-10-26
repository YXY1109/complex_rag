"""
性能监控中间件
收集API性能指标，包括响应时间、请求频率、错误率等
"""
import time
from typing import Callable, Dict, Any
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from infrastructure.monitoring.loguru_logger import logger


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    性能监控中间件
    收集和分析API性能指标
    """

    def __init__(self, app, max_history: int = 1000):
        """
        初始化性能监控中间件

        Args:
            app: ASGI应用实例
            max_history: 最大历史记录数量
        """
        super().__init__(app)
        self.max_history = max_history
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理HTTP请求并收集性能指标

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        start_time = time.time()
        path = request.url.path
        method = request.method
        key = f"{method} {path}"

        try:
            # 执行下一个中间件或路由处理器
            response = await call_next(request)

            # 计算处理时间
            process_time = time.time() - start_time

            # 更新性能指标
            self.request_times[key].append(process_time)
            self.request_counts[key] += 1

            # 记录性能日志
            if process_time > 2.0:  # 超过2秒的请求记录警告
                logger.warning(
                    "慢请求警告",
                    extra={
                        "method": method,
                        "path": path,
                        "process_time": round(process_time, 4),
                        "threshold": 2.0
                    }
                )

            # 在响应头中添加性能信息
            response.headers["X-Response-Time"] = f"{process_time:.4f}s"
            response.headers["X-Request-Count"] = str(self.request_counts[key])

            return response

        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time

            # 更新错误计数
            self.error_counts[key] += 1

            # 记录错误性能日志
            logger.error(
                "请求错误性能统计",
                extra={
                    "method": method,
                    "path": path,
                    "error": str(e),
                    "process_time": round(process_time, 4),
                    "error_count": self.error_counts[key]
                }
            )

            # 重新抛出异常
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            Dict[str, Any]: 性能统计数据
        """
        stats = {}

        for key, times in self.request_times.items():
            if times:
                stats[key] = {
                    "request_count": self.request_counts[key],
                    "error_count": self.error_counts[key],
                    "avg_response_time": sum(times) / len(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "p95_response_time": self._percentile(times, 0.95),
                    "p99_response_time": self._percentile(times, 0.99),
                    "error_rate": self.error_counts[key] / self.request_counts[key] if self.request_counts[key] > 0 else 0.0
                }

        return stats

    def _percentile(self, data: deque, percentile: float) -> float:
        """
        计算数据的百分位数

        Args:
            data: 数据集合
            percentile: 百分位数 (0.0-1.0)

        Returns:
            float: 百分位数值
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]