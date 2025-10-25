"""
文件来源检测模块

此模块提供自动检测文档来源和类型的功能，
基于URL模式、文件类型、内容分析等多维度分析。
"""

from .source_detector import (
    # 检测器类
    SourceDetector,
    URLPatternDetector,
    FileTypeDetector,
    ContentAnalyzer,

    # 结果模型
    SourceDetectionResult,
    ConfidenceLevel,
)

__all__ = [
    # 检测器类
    "SourceDetector",
    "URLPatternDetector",
    "FileTypeDetector",
    "ContentAnalyzer",

    # 结果模型
    "SourceDetectionResult",
    "ConfidenceLevel",
]