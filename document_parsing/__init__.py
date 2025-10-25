"""
文档解析层模块

此模块提供全面的文档解析功能，包括：
- 文档来源自动检测
- 智能处理策略选择
- 策略参数配置管理
- 处理质量监控

基于RAGFlow deepdoc + rag/app架构设计。
"""

# 接口层
from .interfaces import (
    # 解析器接口
    DocumentParserInterface,
    ParserConfig,
    ParserCapabilities,
    DocumentType,
    ProcessingStrategy,
    ParseResult,
    DocumentMetadata,
    TextChunk,
    ImageInfo,
    TableInfo,

    # 解析器异常
    ParseException,
    UnsupportedFormatError,
    CorruptedFileError,
    ProcessingError,
    ValidationError,
    TimeoutError,

    # 格式转换器接口
    FormatConverterInterface,
    ConversionConfig,
    ConversionResult,
    ConversionError,

    # 来源处理器接口
    SourceProcessorInterface,
    SourceConfig,
    SourceResult,
    SourceError,
)

# 来源检测模块
from .source_detection import (
    SourceDetector,
    URLPatternDetector,
    FileTypeDetector,
    ContentAnalyzer,
    SourceDetectionResult,
    ConfidenceLevel,
)

# 策略选择模块
from .strategy_selection import (
    StrategySelector,
    StrategyScorer,
    ProcessingComplexity,
    QualityRequirement,
    StrategyWeights,
    ProcessingProfile,
    StrategyRecommendation,
)

# 策略配置模块
from .strategy_config import (
    StrategyConfigManager,
    ProcessingStrategyConfig,
    OCRConfig,
    LayoutConfig,
    TableConfig,
    ImageConfig,
    TextConfig,
    OCRProvider,
    LayoutAnalyzer,
    TableExtractor,
    get_config_manager,
    set_config_manager,
)

# 质量监控模块
from .quality_monitoring import (
    QualityMonitor,
    QualityEvaluator,
    QualityMetric,
    QualityScore,
    ProcessingSession,
    QualityAlert,
    QualityMetricType,
    QualityLevel,
    AlertSeverity,
    get_quality_monitor,
    set_quality_monitor,
)

__all__ = [
    # 接口层
    "DocumentParserInterface",
    "ParserConfig",
    "ParserCapabilities",
    "DocumentType",
    "ProcessingStrategy",
    "ParseResult",
    "DocumentMetadata",
    "TextChunk",
    "ImageInfo",
    "TableInfo",
    "ParseException",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "ProcessingError",
    "ValidationError",
    "TimeoutError",
    "FormatConverterInterface",
    "ConversionConfig",
    "ConversionResult",
    "ConversionError",
    "SourceProcessorInterface",
    "SourceConfig",
    "SourceResult",
    "SourceError",

    # 来源检测
    "SourceDetector",
    "URLPatternDetector",
    "FileTypeDetector",
    "ContentAnalyzer",
    "SourceDetectionResult",
    "ConfidenceLevel",

    # 策略选择
    "StrategySelector",
    "StrategyScorer",
    "ProcessingComplexity",
    "QualityRequirement",
    "StrategyWeights",
    "ProcessingProfile",
    "StrategyRecommendation",

    # 策略配置
    "StrategyConfigManager",
    "ProcessingStrategyConfig",
    "OCRConfig",
    "LayoutConfig",
    "TableConfig",
    "ImageConfig",
    "TextConfig",
    "OCRProvider",
    "LayoutAnalyzer",
    "TableExtractor",
    "get_config_manager",
    "set_config_manager",

    # 质量监控
    "QualityMonitor",
    "QualityEvaluator",
    "QualityMetric",
    "QualityScore",
    "ProcessingSession",
    "QualityAlert",
    "QualityMetricType",
    "QualityLevel",
    "AlertSeverity",
    "get_quality_monitor",
    "set_quality_monitor",
]