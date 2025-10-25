"""
处理策略参数配置模块

此模块提供不同文档来源的处理策略参数配置，
包括OCR设置、布局分析、多模态处理等各种参数。
"""

from .strategy_config import (
    # 配置管理器
    StrategyConfigManager,

    # 配置模型
    ProcessingStrategyConfig,
    OCRConfig,
    LayoutConfig,
    TableConfig,
    ImageConfig,
    TextConfig,

    # 枚举类型
    OCRProvider,
    LayoutAnalyzer,
    TableExtractor,

    # 全局函数
    get_config_manager,
    set_config_manager,
)

__all__ = [
    # 配置管理器
    "StrategyConfigManager",

    # 配置模型
    "ProcessingStrategyConfig",
    "OCRConfig",
    "LayoutConfig",
    "TableConfig",
    "ImageConfig",
    "TextConfig",

    # 枚举类型
    "OCRProvider",
    "LayoutAnalyzer",
    "TableExtractor",

    # 全局函数
    "get_config_manager",
    "set_config_manager",
]