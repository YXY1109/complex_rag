"""
来源专用处理器模块

此模块提供针对不同文档来源的专用处理器实现，
参考RAGFlow rag/app架构设计。
"""

# 基础处理器
from .base_processor import BaseProcessor

# 网页文档处理器
from .web_processor import WebDocumentProcessor

# 办公文档处理器
from .office_processor import OfficeDocumentProcessor

# 扫描文档处理器
from .scanned_processor import ScannedDocumentProcessor

# 结构化数据处理器
from .structured_processor import StructuredDataProcessor

# 代码仓库处理器
from .code_processor import CodeRepositoryProcessor

__all__ = [
    # 基础处理器
    "BaseProcessor",

    # 专用处理器
    "WebDocumentProcessor",
    "OfficeDocumentProcessor",
    "ScannedDocumentProcessor",
    "StructuredDataProcessor",
    "CodeRepositoryProcessor",
]