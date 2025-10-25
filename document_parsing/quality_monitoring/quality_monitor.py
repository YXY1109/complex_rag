"""
处理质量监控机制

此模块提供文档处理质量监控功能，
包括实时质量评估、性能监控、错误检测和质量报告。
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timezone
import threading
import uuid

from ..interfaces.parser_interface import ParseResult, DocumentMetadata, TextChunk, ImageInfo, TableInfo


class QualityMetricType(Enum):
    """质量指标类型。"""
    ACCURACY = "accuracy"           # 准确性
    COMPLETENESS = "completeness"   # 完整性
    CONSISTENCY = "consistency"     # 一致性
    READABILITY = "readability"     # 可读性
    STRUCTURE = "structure"         # 结构化程度
    PERFORMANCE = "performance"     # 性能
    RELIABILITY = "reliability"     # 可靠性


class QualityLevel(Enum):
    """质量等级。"""
    EXCELLENT = "excellent"    # 优秀 (0.9-1.0)
    GOOD = "good"             # 良好 (0.7-0.9)
    ACCEPTABLE = "acceptable"  # 可接受 (0.5-0.7)
    POOR = "poor"             # 差 (0.3-0.5)
    UNACCEPTABLE = "unacceptable"  # 不可接受 (0.0-0.3)


class AlertSeverity(Enum):
    """告警严重程度。"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """质量指标。"""
    name: str
    metric_type: QualityMetricType
    value: float
    threshold: float
    unit: str = ""
    description: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityScore:
    """质量评分。"""
    overall_score: float
    level: QualityLevel
    metrics: Dict[QualityMetricType, QualityMetric] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessingSession:
    """处理会话。"""
    session_id: str
    file_path: str
    strategy: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    quality_score: Optional[QualityScore] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAlert:
    """质量告警。"""
    alert_id: str
    session_id: str
    severity: AlertSeverity
    metric_type: QualityMetricType
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None


class QualityEvaluator:
    """质量评估器。"""

    def __init__(self):
        """初始化质量评估器。"""
        self.evaluators = {
            QualityMetricType.ACCURACY: self._evaluate_accuracy,
            QualityMetricType.COMPLETENESS: self._evaluate_completeness,
            QualityMetricType.CONSISTENCY: self._evaluate_consistency,
            QualityMetricType.READABILITY: self._evaluate_readability,
            QualityMetricType.STRUCTURE: self._evaluate_structure,
            QualityMetricType.PERFORMANCE: self._evaluate_performance,
            QualityMetricType.RELIABILITY: self._evaluate_reliability
        }

    def evaluate_quality(self, parse_result: ParseResult, processing_time: float) -> QualityScore:
        """
        评估解析结果质量。

        Args:
            parse_result: 解析结果
            processing_time: 处理时间

        Returns:
            QualityScore: 质量评分
        """
        metrics = {}

        # 评估各项指标
        for metric_type, evaluator in self.evaluators.items():
            try:
                metric = evaluator(parse_result, processing_time)
                metrics[metric_type] = metric
            except Exception as e:
                # 评估失败时使用默认值
                metrics[metric_type] = QualityMetric(
                    name=metric_type.value,
                    metric_type=metric_type,
                    value=0.5,
                    threshold=0.7,
                    description=f"评估失败: {str(e)}"
                )

        # 计算综合评分
        overall_score = self._calculate_overall_score(metrics)
        level = self._determine_quality_level(overall_score)

        return QualityScore(
            overall_score=overall_score,
            level=level,
            metrics=metrics
        )

    def _evaluate_accuracy(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估准确性。"""
        score = 0.7  # 基础分数

        # 检查文本提取质量
        if parse_result.full_text:
            text_length = len(parse_result.full_text)
            if text_length > 100:  # 有足够的文本
                score += 0.1

            # 检查是否包含有意义的内容（非空白字符）
            meaningful_chars = len([c for c in parse_result.full_text if not c.isspace()])
            if meaningful_chars / text_length > 0.8:
                score += 0.1

        # 检查文本块质量
        if parse_result.text_chunks:
            avg_chunk_length = statistics.mean([len(chunk.content) for chunk in parse_result.text_chunks])
            if avg_chunk_length > 50:  # 块长度合理
                score += 0.1

            # 检查置信度
            avg_confidence = statistics.mean([chunk.confidence for chunk in parse_result.text_chunks])
            score += (avg_confidence - 0.5) * 0.2  # 根据置信度调整

        return QualityMetric(
            name="准确性",
            metric_type=QualityMetricType.ACCURACY,
            value=min(max(score, 0.0), 1.0),
            threshold=0.7,
            unit="分数",
            description="文本提取的准确程度"
        )

    def _evaluate_completeness(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估完整性。"""
        score = 0.6  # 基础分数

        metadata = parse_result.metadata

        # 检查文本完整性
        if parse_result.full_text and len(parse_result.full_text) > 0:
            score += 0.2

        # 检查结构完整性
        if metadata.has_images and parse_result.images:
            score += 0.1
        if metadata.has_tables and parse_result.tables:
            score += 0.1

        # 检查元数据完整性
        metadata_fields = [metadata.title, metadata.author, metadata.created_date]
        complete_fields = sum(1 for field in metadata_fields if field)
        score += (complete_fields / len(metadata_fields)) * 0.1

        return QualityMetric(
            name="完整性",
            metric_type=QualityMetricType.COMPLETENESS,
            value=min(max(score, 0.0), 1.0),
            threshold=0.7,
            unit="分数",
            description="文档信息提取的完整程度"
        )

    def _evaluate_consistency(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估一致性。"""
        score = 0.8  # 基础分数

        # 检查文本块一致性
        if parse_result.text_chunks and len(parse_result.text_chunks) > 1:
            # 检查块长度分布是否合理
            chunk_lengths = [len(chunk.content) for chunk in parse_result.text_chunks]
            if len(chunk_lengths) > 1:
                length_std = statistics.stdev(chunk_lengths)
                length_mean = statistics.mean(chunk_lengths)
                cv = length_std / length_mean if length_mean > 0 else 0

                # 变异系数越小，一致性越好
                if cv < 0.5:
                    score += 0.1
                elif cv > 1.0:
                    score -= 0.1

        # 检查格式一致性
        if parse_result.text_chunks:
            # 检查字体信息一致性（如果有的话）
            font_infos = [chunk.font_info for chunk in parse_result.text_chunks if chunk.font_info]
            if font_infos:
                # 简化的一致性检查
                score += 0.1

        return QualityMetric(
            name="一致性",
            metric_type=QualityMetricType.CONSISTENCY,
            value=min(max(score, 0.0), 1.0),
            threshold=0.6,
            unit="分数",
            description="文档结构和格式的一致程度"
        )

    def _evaluate_readability(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估可读性。"""
        score = 0.7  # 基础分数

        if not parse_result.full_text:
            return QualityMetric(
                name="可读性",
                metric_type=QualityMetricType.READABILITY,
                value=0.0,
                threshold=0.6,
                unit="分数",
                description="无文本内容"
            )

        text = parse_result.full_text

        # 检查文本格式
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:  # 有段落结构
            score += 0.1

        # 检查句子结构
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 1:  # 有句子结构
            score += 0.1

        # 检查字符编码质量（无乱码）
        try:
            text.encode('utf-8').decode('utf-8')
            score += 0.1
        except UnicodeError:
            score -= 0.3

        return QualityMetric(
            name="可读性",
            metric_type=QualityMetricType.READABILITY,
            value=min(max(score, 0.0), 1.0),
            threshold=0.6,
            unit="分数",
            description="文本内容的可读程度"
        )

    def _evaluate_structure(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估结构化程度。"""
        score = 0.5  # 基础分数

        # 检查文本块结构
        if parse_result.text_chunks:
            chunk_count = len(parse_result.text_chunks)
            if chunk_count > 1:
                score += 0.2  # 有分块结构

            # 检查是否有位置信息
            chunks_with_bbox = sum(1 for chunk in parse_result.text_chunks if chunk.bbox)
            if chunks_with_bbox > 0:
                score += 0.1 * (chunks_with_bbox / chunk_count)

        # 检查表格结构
        if parse_result.tables:
            table_count = len(parse_result.tables)
            score += 0.1 * min(table_count / 5, 1.0)  # 最多5个表格

        # 检查图像结构
        if parse_result.images:
            image_count = len(parse_result.images)
            score += 0.1 * min(image_count / 10, 1.0)  # 最多10个图像

        return QualityMetric(
            name="结构化程度",
            metric_type=QualityMetricType.STRUCTURE,
            value=min(max(score, 0.0), 1.0),
            threshold=0.6,
            unit="分数",
            description="文档结构化信息的提取程度"
        )

    def _evaluate_performance(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估性能。"""
        # 计算处理速度（字符/秒）
        text_length = len(parse_result.full_text) if parse_result.full_text else 0
        processing_speed = text_length / processing_time if processing_time > 0 else 0

        # 性能评分（基于处理速度）
        if processing_speed > 1000:  # 优秀
            score = 1.0
        elif processing_speed > 500:  # 良好
            score = 0.8
        elif processing_speed > 200:  # 可接受
            score = 0.6
        elif processing_speed > 50:   # 差
            score = 0.4
        else:  # 不可接受
            score = 0.2

        return QualityMetric(
            name="性能",
            metric_type=QualityMetricType.PERFORMANCE,
            value=score,
            threshold=0.6,
            unit="分数",
            description=f"处理速度: {processing_speed:.0f} 字符/秒"
        )

    def _evaluate_reliability(self, parse_result: ParseResult, processing_time: float) -> QualityMetric:
        """评估可靠性。"""
        score = 0.8  # 基础分数

        # 检查是否有错误
        if parse_result.error_message:
            score -= 0.4

        # 检查是否有警告
        if parse_result.warnings:
            warning_count = len(parse_result.warnings)
            score -= min(warning_count * 0.1, 0.3)

        # 检查处理时间是否合理
        if processing_time > 300:  # 超过5分钟
            score -= 0.1

        return QualityMetric(
            name="可靠性",
            metric_type=QualityMetricType.RELIABILITY,
            value=min(max(score, 0.0), 1.0),
            threshold=0.7,
            unit="分数",
            description="处理过程的稳定性和可靠性"
        )

    def _calculate_overall_score(self, metrics: Dict[QualityMetricType, QualityMetric]) -> float:
        """计算综合评分。"""
        if not metrics:
            return 0.0

        # 加权计算综合评分
        weights = {
            QualityMetricType.ACCURACY: 0.25,
            QualityMetricType.COMPLETENESS: 0.20,
            QualityMetricType.CONSISTENCY: 0.15,
            QualityMetricType.READABILITY: 0.10,
            QualityMetricType.STRUCTURE: 0.15,
            QualityMetricType.PERFORMANCE: 0.10,
            QualityMetricType.RELIABILITY: 0.05
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric_type, metric in metrics.items():
            weight = weights.get(metric_type, 0.1)
            weighted_sum += metric.value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """确定质量等级。"""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


class QualityMonitor:
    """
    质量监控器。

    提供实时质量监控、告警和报告功能。
    """

    def __init__(self):
        """初始化质量监控器。"""
        self.evaluator = QualityEvaluator()
        self.sessions: Dict[str, ProcessingSession] = {}
        self.alerts: List[QualityAlert] = []
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        self.thresholds = self._initialize_thresholds()
        self._lock = threading.Lock()

    def _initialize_thresholds(self) -> Dict[QualityMetricType, Dict[str, float]]:
        """初始化告警阈值。"""
        return {
            QualityMetricType.ACCURACY: {
                "warning": 0.6,
                "error": 0.4,
                "critical": 0.2
            },
            QualityMetricType.COMPLETENESS: {
                "warning": 0.6,
                "error": 0.4,
                "critical": 0.2
            },
            QualityMetricType.CONSISTENCY: {
                "warning": 0.5,
                "error": 0.3,
                "critical": 0.1
            },
            QualityMetricType.READABILITY: {
                "warning": 0.5,
                "error": 0.3,
                "critical": 0.1
            },
            QualityMetricType.STRUCTURE: {
                "warning": 0.5,
                "error": 0.3,
                "critical": 0.1
            },
            QualityMetricType.PERFORMANCE: {
                "warning": 0.4,
                "error": 0.2,
                "critical": 0.1
            },
            QualityMetricType.RELIABILITY: {
                "warning": 0.6,
                "error": 0.4,
                "critical": 0.2
            }
        }

    def start_session(self, file_path: str, strategy: str) -> str:
        """
        开始处理会话。

        Args:
            file_path: 文件路径
            strategy: 处理策略

        Returns:
            str: 会话ID
        """
        session_id = str(uuid.uuid4())
        session = ProcessingSession(
            session_id=session_id,
            file_path=file_path,
            strategy=strategy,
            start_time=time.time()
        )

        with self._lock:
            self.sessions[session_id] = session

        return session_id

    def end_session(
        self,
        session_id: str,
        parse_result: ParseResult,
        processing_time: float
    ) -> QualityScore:
        """
        结束处理会话并评估质量。

        Args:
            session_id: 会话ID
            parse_result: 解析结果
            processing_time: 处理时间

        Returns:
            QualityScore: 质量评分
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"会话不存在: {session_id}")

            session = self.sessions[session_id]
            session.end_time = time.time()
            session.success = parse_result.success
            session.error_message = parse_result.error_message

            # 记录性能指标
            session.performance_metrics = {
                "processing_time": processing_time,
                "text_length": len(parse_result.full_text) if parse_result.full_text else 0,
                "chunk_count": len(parse_result.text_chunks) if parse_result.text_chunks else 0,
                "image_count": len(parse_result.images) if parse_result.images else 0,
                "table_count": len(parse_result.tables) if parse_result.tables else 0
            }

        # 评估质量
        quality_score = self.evaluator.evaluate_quality(parse_result, processing_time)

        with self._lock:
            session.quality_score = quality_score

        # 检查告警条件
        self._check_alerts(session_id, quality_score)

        return quality_score

    def _check_alerts(self, session_id: str, quality_score: QualityScore) -> None:
        """检查告警条件。"""
        for metric_type, metric in quality_score.metrics.items():
            threshold_config = self.thresholds.get(metric_type, {})

            # 检查是否需要告警
            severity = None
            if metric.value <= threshold_config.get("critical", 0):
                severity = AlertSeverity.CRITICAL
            elif metric.value <= threshold_config.get("error", 0):
                severity = AlertSeverity.ERROR
            elif metric.value <= threshold_config.get("warning", 0):
                severity = AlertSeverity.WARNING

            if severity:
                alert = QualityAlert(
                    alert_id=str(uuid.uuid4()),
                    session_id=session_id,
                    severity=severity,
                    metric_type=metric_type,
                    message=f"{metric.name}低于阈值: {metric.value:.2f} < {metric.threshold:.2f}",
                    value=metric.value,
                    threshold=metric.threshold
                )

                with self._lock:
                    self.alerts.append(alert)

                # 触发告警回调
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"告警回调失败: {e}")

    def add_alert_callback(self, callback: Callable[[QualityAlert], None]) -> None:
        """添加告警回调函数。"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[QualityAlert], None]) -> None:
        """移除告警回调函数。"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def get_session(self, session_id: str) -> Optional[ProcessingSession]:
        """获取处理会话。"""
        with self._lock:
            return self.sessions.get(session_id)

    def get_recent_sessions(self, limit: int = 100) -> List[ProcessingSession]:
        """获取最近的处理会话。"""
        with self._lock:
            sessions = list(self.sessions.values())
            sessions.sort(key=lambda s: s.start_time, reverse=True)
            return sessions[:limit]

    def get_active_alerts(self) -> List[QualityAlert]:
        """获取活跃的告警。"""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警。"""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = time.time()
                    return True
        return False

    def get_quality_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取质量统计信息。

        Args:
            hours: 统计时间范围（小时）

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            # 获取指定时间范围内的会话
            cutoff_time = time.time() - (hours * 3600)
            recent_sessions = [
                session for session in self.sessions.values()
                if session.start_time >= cutoff_time and session.quality_score
            ]

            if not recent_sessions:
                return {
                    "total_sessions": 0,
                    "time_range_hours": hours,
                    "average_score": 0.0,
                    "quality_distribution": {},
                    "metric_averages": {}
                }

            # 计算统计信息
            scores = [session.quality_score.overall_score for session in recent_sessions]
            average_score = statistics.mean(scores)

            # 质量等级分布
            quality_distribution = {}
            for level in QualityLevel:
                count = sum(1 for s in recent_sessions if s.quality_score.level == level)
                quality_distribution[level.value] = count

            # 各项指标平均值
            metric_averages = {}
            for metric_type in QualityMetricType:
                values = [
                    session.quality_score.metrics[metric_type].value
                    for session in recent_sessions
                    if metric_type in session.quality_score.metrics
                ]
                if values:
                    metric_averages[metric_type.value] = {
                        "average": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0
                    }

            return {
                "total_sessions": len(recent_sessions),
                "time_range_hours": hours,
                "average_score": average_score,
                "quality_distribution": quality_distribution,
                "metric_averages": metric_averages,
                "success_rate": sum(1 for s in recent_sessions if s.success) / len(recent_sessions)
            }

    def generate_quality_report(self, session_id: str) -> Dict[str, Any]:
        """
        生成质量报告。

        Args:
            session_id: 会话ID

        Returns:
            Dict[str, Any]: 质量报告
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return {"error": "会话不存在"}

        if not session.quality_score:
            return {"error": "质量评分不可用"}

        report = {
            "session_info": {
                "session_id": session.session_id,
                "file_path": session.file_path,
                "strategy": session.strategy,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": session.end_time - session.start_time if session.end_time else None,
                "success": session.success
            },
            "quality_score": {
                "overall_score": session.quality_score.overall_score,
                "level": session.quality_score.level.value,
                "timestamp": session.quality_score.timestamp
            },
            "metrics": {},
            "performance_metrics": session.performance_metrics,
            "recommendations": []
        }

        # 添加指标详情
        for metric_type, metric in session.quality_score.metrics.items():
            report["metrics"][metric_type.value] = {
                "name": metric.name,
                "value": metric.value,
                "threshold": metric.threshold,
                "unit": metric.unit,
                "description": metric.description
            }

        # 生成改进建议
        recommendations = self._generate_recommendations(session.quality_score)
        report["recommendations"] = recommendations

        return report

    def _generate_recommendations(self, quality_score: QualityScore) -> List[str]:
        """生成改进建议。"""
        recommendations = []

        for metric_type, metric in quality_score.metrics.items():
            if metric.value < metric.threshold:
                if metric_type == QualityMetricType.ACCURACY:
                    recommendations.append("考虑使用更精确的OCR引擎或调整OCR参数")
                    recommendations.append("检查文档质量，确保图像清晰度")
                elif metric_type == QualityMetricType.COMPLETENESS:
                    recommendations.append("检查是否启用了所有必要的处理选项")
                    recommendations.append("确认文档是否包含可提取的内容")
                elif metric_type == QualityMetricType.CONSISTENCY:
                    recommendations.append("调整文本分块参数以提高一致性")
                elif metric_type == QualityMetricType.READABILITY:
                    recommendations.append("检查文本编码设置，避免乱码问题")
                elif metric_type == QualityMetricType.STRUCTURE:
                    recommendations.append("启用布局分析以更好地提取结构信息")
                elif metric_type == QualityMetricType.PERFORMANCE:
                    recommendations.append("考虑优化处理参数或使用更高效的策略")
                elif metric_type == QualityMetricType.RELIABILITY:
                    recommendations.append("检查文档格式兼容性和处理稳定性")

        # 基于整体评分的建议
        if quality_score.overall_score < 0.5:
            recommendations.append("建议尝试不同的处理策略以改善整体质量")
        elif quality_score.overall_score > 0.8:
            recommendations.append("当前质量良好，可以考虑使用更高精度的设置")

        return recommendations


# 全局质量监控器实例
_global_quality_monitor: Optional[QualityMonitor] = None


def get_quality_monitor() -> QualityMonitor:
    """获取全局质量监控器实例。"""
    global _global_quality_monitor
    if _global_quality_monitor is None:
        _global_quality_monitor = QualityMonitor()
    return _global_quality_monitor


def set_quality_monitor(monitor: QualityMonitor) -> None:
    """设置全局质量监控器实例。"""
    global _global_quality_monitor
    _global_quality_monitor = monitor


# 导出
__all__ = [
    'QualityMonitor',
    'QualityEvaluator',
    'QualityMetric',
    'QualityScore',
    'ProcessingSession',
    'QualityAlert',
    'QualityMetricType',
    'QualityLevel',
    'AlertSeverity',
    'get_quality_monitor',
    'set_quality_monitor'
]