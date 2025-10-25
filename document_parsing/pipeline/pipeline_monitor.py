"""
流水线监控器

监控流水线的执行状态、性能指标、错误情况等，
提供实时监控和报警功能。
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import uuid

from .pipeline_config import PipelineContext, StageExecutionResult
from .pipeline_orchestrator import OrchestratorState


class MetricType(Enum):
    """指标类型。"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """报警级别。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PipelineMetric:
    """流水线指标。"""

    metric_id: str
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = str(uuid.uuid4())


@dataclass
class PipelineAlert:
    """流水线报警。"""

    alert_id: str
    level: AlertLevel
    title: str
    message: str
    pipeline_id: str
    execution_id: Optional[str] = None
    stage_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())


@dataclass
class ExecutionSnapshot:
    """执行快照。"""

    snapshot_id: str
    pipeline_id: str
    execution_id: str
    state: str
    progress: float
    current_stage: Optional[str]
    completed_stages: List[str]
    failed_stages: List[str]
    execution_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.snapshot_id:
            self.snapshot_id = str(uuid.uuid4())


class PipelineMonitor:
    """流水线监控器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控器。

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 监控状态
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # 数据存储
        self.metrics: List[PipelineMetric] = []
        self.alerts: List[PipelineAlert] = []
        self.snapshots: List[ExecutionSnapshot] = []

        # 阈值配置
        self.thresholds = {
            "execution_time": 300.0,      # 执行时间阈值（秒）
            "error_rate": 0.1,           # 错误率阈值
            "memory_usage": 0.8,         # 内存使用率阈值
            "cpu_usage": 0.8,            # CPU使用率阈值
            "stage_timeout": 60.0        # 阶段超时阈值
        }

        # 回调函数
        self.alert_handlers: List[Callable] = []
        self.metric_handlers: List[Callable] = []

        # 统计信息
        self.statistics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_metrics": 0,
            "total_alerts": 0
        }

    async def start_monitoring(self) -> None:
        """启动监控。"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("流水线监控器已启动")

    async def stop_monitoring(self) -> None:
        """停止监控。"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("流水线监控器已停止")

    async def _monitoring_loop(self) -> None:
        """监控循环。"""
        while self.is_monitoring:
            try:
                # 清理过期数据
                await self._cleanup_expired_data()

                # 检查系统指标
                await self._check_system_metrics()

                # 生成监控报告
                await self._generate_monitoring_report()

                await asyncio.sleep(self.config.get("monitor_interval", 30))

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(10)

    async def record_execution_start(self, context: PipelineContext) -> None:
        """记录执行开始。"""
        snapshot = ExecutionSnapshot(
            snapshot_id=str(uuid.uuid4()),
            pipeline_id=context.pipeline_id,
            execution_id=context.execution_id,
            state="running",
            progress=0.0,
            current_stage=None,
            completed_stages=[],
            failed_stages=[],
            execution_time=0.0
        )
        self.snapshots.append(snapshot)

        # 记录指标
        await self.record_metric(
            name="pipeline_execution_started",
            metric_type=MetricType.COUNTER,
            value=1,
            labels={"pipeline_id": context.pipeline_id}
        )

        self.statistics["total_executions"] += 1

    async def record_stage_completion(
        self,
        context: PipelineContext,
        stage_result: StageExecutionResult
    ) -> None:
        """记录阶段完成。"""
        # 更新快照
        snapshot = self._get_latest_snapshot(context.execution_id)
        if snapshot:
            if stage_result.success:
                snapshot.completed_stages.append(stage_result.stage_id)
            else:
                snapshot.failed_stages.append(stage_result.stage_id)

            snapshot.current_stage = stage_result.stage_id
            snapshot.execution_time = sum(context.execution_times.values())
            snapshot.metrics["stage_duration"] = stage_result.execution_time

        # 记录指标
        await self.record_metric(
            name="stage_execution_time",
            metric_type=MetricType.HISTOGRAM,
            value=stage_result.execution_time,
            labels={
                "pipeline_id": context.pipeline_id,
                "stage_id": stage_result.stage_id,
                "stage_type": stage_result.stage_type.value,
                "status": stage_result.status.value
            }
        )

        if stage_result.failed:
            await self.record_metric(
                name="stage_execution_failed",
                metric_type=MetricType.COUNTER,
                value=1,
                labels={
                    "pipeline_id": context.pipeline_id,
                    "stage_id": stage_result.stage_id,
                    "error_type": "execution_error"
                }
            )

            # 检查是否需要报警
            await self._check_stage_failure(context, stage_result)

    async def record_execution_completion(
        self,
        context: PipelineContext,
        results: List[StageExecutionResult]
    ) -> None:
        """记录执行完成。"""
        success_count = len([r for r in results if r.success])
        total_count = len(results)
        execution_time = sum(context.execution_times.values())

        # 更新快照
        snapshot = self._get_latest_snapshot(context.execution_id)
        if snapshot:
            snapshot.state = "completed" if success_count == total_count else "failed"
            snapshot.progress = success_count / total_count if total_count > 0 else 0
            snapshot.execution_time = execution_time

        # 记录指标
        await self.record_metric(
            name="pipeline_execution_time",
            metric_type=MetricType.HISTOGRAM,
            value=execution_time,
            labels={
                "pipeline_id": context.pipeline_id,
                "status": "success" if success_count == total_count else "failed"
            }
        )

        if success_count == total_count:
            self.statistics["successful_executions"] += 1
        else:
            self.statistics["failed_executions"] += 1

        # 更新平均执行时间
        if self.statistics["total_executions"] > 0:
            total_time = self.statistics["average_execution_time"] * (self.statistics["total_executions"] - 1) + execution_time
            self.statistics["average_execution_time"] = total_time / self.statistics["total_executions"]

        # 检查执行时间阈值
        if execution_time > self.thresholds["execution_time"]:
            await self.create_alert(
                level=AlertLevel.WARNING,
                title="执行时间过长",
                message=f"流水线 {context.pipeline_id} 执行时间 {execution_time:.2f}s 超过阈值 {self.thresholds['execution_time']}s",
                pipeline_id=context.pipeline_id,
                execution_id=context.execution_id
            )

    async def record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = ""
    ) -> None:
        """记录指标。"""
        metric = PipelineMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
            description=description
        )
        self.metrics.append(metric)
        self.statistics["total_metrics"] += 1

        # 调用指标处理器
        for handler in self.metric_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(metric)
                else:
                    handler(metric)
            except Exception as e:
                self.logger.error(f"指标处理器执行失败: {e}")

    async def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        pipeline_id: str,
        execution_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> str:
        """创建报警。"""
        alert = PipelineAlert(
            level=level,
            title=title,
            message=message,
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            stage_id=stage_id,
            labels=labels or {}
        )
        self.alerts.append(alert)
        self.statistics["total_alerts"] += 1

        self.logger.warning(f"[{level.value.upper()}] {title}: {message}")

        # 调用报警处理器
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"报警处理器执行失败: {e}")

        return alert.alert_id

    async def _check_stage_failure(
        self,
        context: PipelineContext,
        stage_result: StageExecutionResult
    ) -> None:
        """检查阶段失败报警。"""
        error_rate_threshold = self.config.get("error_rate_threshold", 0.1)

        # 计算最近的错误率
        recent_metrics = [
            m for m in self.metrics
            if m.name == "stage_execution_failed" and
               m.timestamp > datetime.now() - timedelta(minutes=10)
        ]

        if recent_metrics:
            recent_failures = sum(m.value for m in recent_metrics)
            recent_total = len([
                m for m in self.metrics
                if m.name == "stage_execution_started" and
                   m.timestamp > datetime.now() - timedelta(minutes=10)
            ])

            if recent_total > 0:
                error_rate = recent_failures / recent_total
                if error_rate > error_rate_threshold:
                    await self.create_alert(
                        level=AlertLevel.WARNING,
                        title="阶段错误率过高",
                        message=f"最近10分钟阶段错误率 {error_rate:.2%} 超过阈值 {error_rate_threshold:.2%}",
                        pipeline_id=context.pipeline_id,
                        execution_id=context.execution_id,
                        stage_id=stage_result.stage_id
                    )

    async def _check_system_metrics(self) -> None:
        """检查系统指标。"""
        try:
            import psutil

            # 检查内存使用率
            memory = psutil.virtual_memory()
            if memory.percent / 100 > self.thresholds["memory_usage"]:
                await self.create_alert(
                    level=AlertLevel.WARNING,
                    title="内存使用率过高",
                    message=f"系统内存使用率 {memory.percent:.1f}% 超过阈值 {self.thresholds['memory_usage']*100:.1f}%",
                    pipeline_id="system"
                )

            # 记录系统指标
            await self.record_metric(
                name="system_memory_usage",
                metric_type=MetricType.GAUGE,
                value=memory.percent,
                labels={"type": "system"}
            )

            # 检查CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent / 100 > self.thresholds["cpu_usage"]:
                await self.create_alert(
                    level=AlertLevel.WARNING,
                    title="CPU使用率过高",
                    message=f"系统CPU使用率 {cpu_percent:.1f}% 超过阈值 {self.thresholds['cpu_usage']*100:.1f}%",
                    pipeline_id="system"
                )

            await self.record_metric(
                name="system_cpu_usage",
                metric_type=MetricType.GAUGE,
                value=cpu_percent,
                labels={"type": "system"}
            )

        except ImportError:
            self.logger.warning("psutil 未安装，系统监控功能受限")
        except Exception as e:
            self.logger.error(f"检查系统指标失败: {e}")

    async def _cleanup_expired_data(self) -> None:
        """清理过期数据。"""
        retention_days = self.config.get("retention_days", 7)
        cutoff_time = datetime.now() - timedelta(days=retention_days)

        # 清理过期指标
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

        # 清理过期报警
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time or not a.resolved]

        # 清理过期快照
        self.snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

    async def _generate_monitoring_report(self) -> None:
        """生成监控报告。"""
        if not self.config.get("enable_reports", True):
            return

        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.statistics.copy(),
                "active_alerts": len([a for a in self.alerts if not a.resolved]),
                "recent_metrics": len([
                    m for m in self.metrics
                    if m.timestamp > datetime.now() - timedelta(hours=1)
                ]),
                "thresholds": self.thresholds.copy()
            }

            self.logger.info(f"监控报告: {json.dumps(report, ensure_ascii=False)}")

        except Exception as e:
            self.logger.error(f"生成监控报告失败: {e}")

    def _get_latest_snapshot(self, execution_id: str) -> Optional[ExecutionSnapshot]:
        """获取最新的执行快照。"""
        matching_snapshots = [s for s in self.snapshots if s.execution_id == execution_id]
        return max(matching_snapshots, key=lambda s: s.timestamp) if matching_snapshots else None

    def get_metrics(
        self,
        name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PipelineMetric]:
        """获取指标。"""
        filtered_metrics = self.metrics

        if name:
            filtered_metrics = [m for m in filtered_metrics if m.name == name]

        if labels:
            filtered_metrics = [
                m for m in filtered_metrics
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]

        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]

        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]

        # 按时间倒序排列
        filtered_metrics.sort(key=lambda m: m.timestamp, reverse=True)

        return filtered_metrics[:limit]

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        pipeline_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[PipelineAlert]:
        """获取报警。"""
        filtered_alerts = self.alerts

        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]

        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]

        if pipeline_id:
            filtered_alerts = [a for a in filtered_alerts if a.pipeline_id == pipeline_id]

        # 按时间倒序排列
        filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return filtered_alerts[:limit]

    def get_snapshots(
        self,
        pipeline_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[ExecutionSnapshot]:
        """获取执行快照。"""
        filtered_snapshots = self.snapshots

        if pipeline_id:
            filtered_snapshots = [s for s in filtered_snapshots if s.pipeline_id == pipeline_id]

        if execution_id:
            filtered_snapshots = [s for s in filtered_snapshots if s.execution_id == execution_id]

        # 按时间倒序排列
        filtered_snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        return filtered_snapshots[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return self.statistics.copy()

    async def resolve_alert(self, alert_id: str) -> bool:
        """解决报警。"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"报警已解决: {alert.title}")
                return True
        return False

    def add_alert_handler(self, handler: Callable) -> None:
        """添加报警处理器。"""
        self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable) -> None:
        """移除报警处理器。"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)

    def add_metric_handler(self, handler: Callable) -> None:
        """添加指标处理器。"""
        self.metric_handlers.append(handler)

    def remove_metric_handler(self, handler: Callable) -> None:
        """移除指标处理器。"""
        if handler in self.metric_handlers:
            self.metric_handlers.remove(handler)

    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """更新阈值配置。"""
        self.thresholds.update(thresholds)
        self.logger.info(f"阈值配置已更新: {thresholds}")

    async def export_metrics(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """导出指标数据。"""
        metrics = self.get_metrics(start_time=start_time, end_time=end_time)

        if format == "json":
            data = [
                {
                    "metric_id": m.metric_id,
                    "name": m.name,
                    "type": m.metric_type.value,
                    "value": m.value,
                    "labels": m.labels,
                    "timestamp": m.timestamp.isoformat(),
                    "description": m.description
                }
                for m in metrics
            ]
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    async def export_alerts(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """导出报警数据。"""
        alerts = self.get_alerts()

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]

        if format == "json":
            data = [
                {
                    "alert_id": a.alert_id,
                    "level": a.level.value,
                    "title": a.title,
                    "message": a.message,
                    "pipeline_id": a.pipeline_id,
                    "execution_id": a.execution_id,
                    "stage_id": a.stage_id,
                    "labels": a.labels,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None
                }
                for a in alerts
            ]
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")