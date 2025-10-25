"""
Document Parser Services

This module provides core services for document processing including
source detection, strategy selection, configuration management, and quality monitoring.
"""

from .file_source_detector import FileSourceDetector, SourceFeatures
from .processing_strategy_selector import (
    ProcessingStrategySelector,
    StrategyRecommendation,
    StrategyParams,
    ProcessingRequirement
)
from .strategy_configurator import (
    StrategyConfigurator,
    SourceStrategyConfig,
    GlobalStrategyConfig,
    ConfigFormat
)
from .quality_monitor import (
    QualityMonitor,
    QualityMetric,
    QualityLevel,
    QualityMeasurement,
    ProcessingSession,
    QualityReport
)

__all__ = [
    # Source Detection
    "FileSourceDetector",
    "SourceFeatures",

    # Strategy Selection
    "ProcessingStrategySelector",
    "StrategyRecommendation",
    "StrategyParams",
    "ProcessingRequirement",

    # Configuration
    "StrategyConfigurator",
    "SourceStrategyConfig",
    "GlobalStrategyConfig",
    "ConfigFormat",

    # Quality Monitoring
    "QualityMonitor",
    "QualityMetric",
    "QualityLevel",
    "QualityMeasurement",
    "ProcessingSession",
    "QualityReport"
]