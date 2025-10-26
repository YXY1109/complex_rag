"""
处理质量监控机制模块

此模块提供文档处理质量监控功能，
包括实时质量评估、性能监控、错误检测和质量报告。
"""

from .quality_monitor import (
    # 监控器类
    QualityMonitor,
    QualityEvaluator,

    # 数据模型
    QualityMetric,
    QualityScore,
    ProcessingSession,
    QualityAlert,

    # 枚举类型
    QualityMetricType,
    QualityLevel,
    AlertSeverity,

    # 全局函数
    get_quality_monitor,
    set_quality_monitor,
)

__all__ = [
    # 监控器类
    "QualityMonitor",
    "QualityEvaluator",

    # 数据模型
    "QualityMetric",
    "QualityScore",
    "ProcessingSession",
    "QualityAlert",

    # 枚举类型
    "QualityMetricType",
    "QualityLevel",
    "AlertSeverity",

    # 全局函数
    "get_quality_monitor",
    "set_quality_monitor",
]