"""
API监控和指标收集中间件
实时收集API性能指标、错误统计和业务数据
"""
import time
import json
import asyncio
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from infrastructure.monitoring.loguru_logger import logger


class MetricType(str, Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """指标值"""
    name: str
    type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    request_count: int = 0
    request_duration_total: float = 0.0
    request_duration_avg: float = 0.0
    request_duration_max: float = 0.0
    request_duration_min: float = float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    active_requests: int = 0
    requests_per_second: float = 0.0


@dataclass
class ResourceMetrics:
    """资源指标"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes_sent: int = 0
    network_io_bytes_recv: int = 0
    open_files: int = 0
    threads: int = 0


@dataclass
class BusinessMetrics:
    """业务指标"""
    total_conversations: int = 0
    total_messages: int = 0
    total_documents: int = 0
    total_searches: int = 0
    active_users: int = 0
    knowledge_bases: int = 0
    cache_hit_rate: float = 0.0


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        """初始化指标收集器"""
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)

        # 性能指标
        self.performance_metrics = PerformanceMetrics()
        self.resource_metrics = ResourceMetrics()
        self.business_metrics = BusinessMetrics()

        # 请求统计
        self.request_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.error_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 时间窗口统计
        self.time_windows = {
            "1m": deque(maxlen=60),      # 1分钟窗口（每秒一个数据点）
            "5m": deque(maxlen=300),     # 5分钟窗口
            "15m": deque(maxlen=900),    # 15分钟窗口
            "1h": deque(maxlen=3600),    # 1小时窗口
            "24h": deque(maxlen=86400)   # 24小时窗口
        }

        # 启动系统资源监控
        self.start_time = time.time()
        self.last_system_check = 0
        self.system_check_interval = 5  # 秒

    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """增加计数器"""
        key = self._make_key(name, labels)
        self.counters[key] += value

        # 记录指标
        metric = MetricValue(
            name=name,
            type=MetricType.COUNTER,
            value=float(self.counters[key]),
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表盘指标"""
        key = self._make_key(name, labels)
        self.gauges[key] = value

        # 记录指标
        metric = MetricValue(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图指标"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

        # 记录指标
        metric = MetricValue(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """记录计时器指标"""
        key = self._make_key(name, labels)
        self.timers[key].append(duration)

        # 记录指标
        metric = MetricValue(
            name=name,
            type=MetricType.TIMER,
            value=duration,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)

    def update_request_metrics(self, request: Request, response: Response, duration: float):
        """更新请求指标"""
        path = request.url.path
        method = request.method
        status_code = response.status_code

        # 更新请求统计
        key = f"{method}:{path}"
        if key not in self.request_stats:
            self.request_stats[key] = {
                "count": 0,
                "total_duration": 0.0,
                "max_duration": 0.0,
                "min_duration": float('inf'),
                "errors": 0
            }

        stats = self.request_stats[key]
        stats["count"] += 1
        stats["total_duration"] += duration
        stats["max_duration"] = max(stats["max_duration"], duration)
        stats["min_duration"] = min(stats["min_duration"], duration)

        if status_code >= 400:
            stats["errors"] += 1

        # 记录指标
        self.increment_counter("http_requests_total", 1, {
            "method": method,
            "path": path,
            "status": str(status_code)
        })

        self.record_timer("http_request_duration_seconds", duration, {
            "method": method,
            "path": path,
            "status": str(status_code)
        })

        # 更新性能指标
        self.performance_metrics.request_count += 1
        self.performance_metrics.request_duration_total += duration
        self.performance_metrics.request_duration_avg = (
            self.performance_metrics.request_duration_total / self.performance_metrics.request_count
        )
        self.performance_metrics.request_duration_max = max(
            self.performance_metrics.request_duration_max, duration
        )
        self.performance_metrics.request_duration_min = min(
            self.performance_metrics.request_duration_min, duration
        )

        if status_code >= 400:
            self.performance_metrics.error_count += 1

        self.performance_metrics.error_rate = (
            self.performance_metrics.error_count / max(1, self.performance_metrics.request_count)
        )

        # 记录时间窗口数据
        self._record_time_window_data(duration, status_code)

    def update_system_metrics(self):
        """更新系统指标"""
        current_time = time.time()

        # 限制系统检查频率
        if current_time - self.last_system_check < self.system_check_interval:
            return

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.resource_metrics.cpu_percent = cpu_percent
            self.set_gauge("system_cpu_percent", cpu_percent)

            # 内存使用率
            memory = psutil.virtual_memory()
            self.resource_metrics.memory_percent = memory.percent
            self.set_gauge("system_memory_percent", memory.percent)
            self.set_gauge("system_memory_used_bytes", memory.used)
            self.set_gauge("system_memory_available_bytes", memory.available)

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            self.resource_metrics.disk_usage_percent = disk.percent
            self.set_gauge("system_disk_percent", disk.percent)
            self.set_gauge("system_disk_used_bytes", disk.used)
            self.set_gauge("system_disk_free_bytes", disk.free)

            # 网络IO
            net_io = psutil.net_io_counters()
            self.resource_metrics.network_io_bytes_sent = net_io.bytes_sent
            self.resource_metrics.network_io_bytes_recv = net_io.bytes_recv
            self.set_gauge("system_network_bytes_sent_total", net_io.bytes_sent)
            self.set_gauge("system_network_bytes_recv_total", net_io.bytes_recv)

            # 进程信息
            process = psutil.Process()
            self.resource_metrics.open_files = process.num_fds()
            self.resource_metrics.threads = process.num_threads()
            self.set_gauge("process_open_files", process.num_fds())
            self.set_gauge("process_threads", process.num_threads())

            self.last_system_check = current_time

        except Exception as e:
            logger.error(f"系统指标更新失败: {str(e)}")

    def update_business_metrics(self, **kwargs):
        """更新业务指标"""
        for key, value in kwargs.items():
            if hasattr(self.business_metrics, key):
                setattr(self.business_metrics, key, value)
                self.set_gauge(f"business_{key}", float(value))

    def _record_time_window_data(self, duration: float, status_code: int):
        """记录时间窗口数据"""
        current_time = time.time()

        # 记录到所有时间窗口
        for window_name in self.time_windows:
            window = self.time_windows[window_name]
            window.append({
                "timestamp": current_time,
                "duration": duration,
                "status_code": status_code,
                "is_error": status_code >= 400
            })

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成指标键"""
        if not labels:
            return name

        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}[{','.join(label_parts)}]"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        # 计算时间窗口统计
        window_stats = {}
        for window_name, window in self.time_windows.items():
            if not window:
                continue

            durations = [d["duration"] for d in window]
            errors = [d["is_error"] for d in window]
            request_rate = len(window) / max(1, self._get_window_duration(window_name))

            window_stats[window_name] = {
                "request_count": len(window),
                "avg_duration": sum(durations) / max(1, len(durations)),
                "max_duration": max(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "error_count": sum(errors),
                "error_rate": sum(errors) / max(1, len(window)),
                "requests_per_second": request_rate
            }

        return {
            "performance": asdict(self.performance_metrics),
            "resource": asdict(self.resource_metrics),
            "business": asdict(self.business_metrics),
            "time_windows": window_stats,
            "metrics_count": {
                name: len(values) for name, values in self.metrics.items()
            },
            "collection_info": {
                "start_time": self.start_time,
                "uptime_seconds": time.time() - self.start_time,
                "last_system_check": self.last_system_check
            }
        }

    def _get_window_duration(self, window_name: str) -> float:
        """获取时间窗口持续时间（秒）"""
        durations = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "24h": 86400
        }
        return durations.get(window_name, 60)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """监控中间件"""

    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控中间件

        Args:
            app: ASGI应用
            config: 监控配置
        """
        super().__init__(app)
        self.config = config or {}
        self.metrics_collector = MetricsCollector()

        # 监控配置
        self.enable_request_logging = self.config.get("enable_request_logging", True)
        self.enable_performance_tracking = self.config.get("enable_performance_tracking", True)
        self.enable_system_monitoring = self.config.get("enable_system_monitoring", True)
        self.metrics_retention_hours = self.config.get("metrics_retention_hours", 24)

        # 启动后台任务
        self._start_background_tasks()

    async def dispatch(self, request: Request, call_next):
        """
        处理请求并收集监控指标

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        start_time = time.time()
        request_id = self._generate_request_id(request)

        # 添加请求ID到响应头
        response = await self._call_next_with_monitoring(request_id, request, call_next, start_time)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 更新指标
        if self.enable_performance_tracking:
            self.metrics_collector.update_request_metrics(request, response, processing_time)

        # 添加监控头到响应
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"

        return response

    async def _call_next_with_monitoring(
        self,
        request_id: str,
        request: Request,
        call_next: Callable,
        start_time: float
    ) -> Response:
        """调用下一个中间件并记录指标"""
        try:
            # 记录活跃请求数
            self.metrics_collector.performance_metrics.active_requests += 1

            # 执行请求
            response = await call_next(request)

            return response

        except Exception as e:
            # 记录错误指标
            self.metrics_collector.increment_counter("http_requests_failed", 1, {
                "path": request.url.path,
                "method": request.method,
                "error_type": type(e).__name__
            })

            logger.error(
                "请求处理异常",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "error": str(e),
                    "duration": time.time() - start_time
                },
                exc_info=True
            )

            raise

        finally:
            # 减少活跃请求数
            self.metrics_collector.performance_metrics.active_requests = max(
                0, self.metrics_collector.performance_metrics.active_requests - 1
            )

    def _generate_request_id(self, request: Request) -> str:
        """生成请求ID"""
        import uuid
        import hashlib

        # 使用时间戳、随机数和请求信息生成唯一ID
        timestamp = str(int(time.time() * 1000))
        random_part = str(uuid.uuid4())[:8]
        request_info = f"{request.method}:{request.url.path}"
        request_hash = hashlib.md5(request_info.encode()).hexdigest()[:8]

        return f"req_{timestamp}_{random_part}_{request_hash}"

    def _start_background_tasks(self):
        """启动后台任务"""
        if self.enable_system_monitoring:
            # 启动系统监控任务
            asyncio.create_task(self._system_monitoring_loop())

        # 启动指标清理任务
        asyncio.create_task(self._metrics_cleanup_loop())

    async def _system_monitoring_loop(self):
        """系统监控循环"""
        while True:
            try:
                self.metrics_collector.update_system_metrics()
                await asyncio.sleep(self.config.get("system_monitoring_interval", 5))
            except Exception as e:
                logger.error(f"系统监控循环异常: {str(e)}")
                await asyncio.sleep(30)  # 出错时等待30秒

    async def _metrics_cleanup_loop(self):
        """指标清理循环"""
        while True:
            try:
                # 保留指定时间内的指标
                cutoff_time = time.time() - (self.metrics_retention_hours * 3600)

                # 清理过期指标
                for metric_name in list(self.metrics_collector.metrics.keys()):
                    self.metrics_collector.metrics[metric_name] = [
                        metric for metric in self.metrics_collector.metrics[metric_name]
                        if metric.timestamp > cutoff_time
                    ]

                # 清理过期时间窗口数据
                for window_name in list(self.metrics_collector.time_windows.keys()):
                    window = self.metrics_collector.time_windows[window_name]
                    cutoff = cutoff_time - self._get_window_duration(window_name)
                    self.metrics_collector.time_windows[window_name] = deque(
                        (item for item in window if item["timestamp"] > cutoff),
                        maxlen=len(window)
                    )

                await asyncio.sleep(3600)  # 每小时清理一次

            except Exception as e:
                logger.error(f"指标清理循环异常: {str(e)}")
                await asyncio.sleep(3600)

    def _get_window_duration(self, window_name: str) -> float:
        """获取时间窗口持续时间"""
        durations = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "24h": 86400
        }
        return durations.get(window_name, 60)

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return self.metrics_collector.get_metrics_summary()

    def update_business_metrics(self, **kwargs):
        """更新业务指标"""
        self.metrics_collector.update_business_metrics(**kwargs)


class MonitoringManager:
    """监控管理器"""

    def __init__(self):
        self.middleware: Optional[MonitoringMiddleware] = None

    def configure(self, config_dict: Dict[str, Any]) -> MonitoringMiddleware:
        """配置监控中间件"""
        # 创建中间件的占位符，实际使用时需要传入app实例
        class PlaceholderMiddleware:
            def __init__(self):
                self.config = config_dict

        self.middleware = PlaceholderMiddleware()
        return self.middleware  # type: ignore

    def get_middleware(self, app):
        """获取监控中间件实例"""
        if not self.middleware:
            raise ValueError("请先调用configure()方法配置监控中间件")

        return MonitoringMiddleware(app, self.middleware.config)


# 全局监控管理器实例
monitoring_manager = MonitoringManager()