"""
通义千问模型提供商

提供通义千问模型的LLM服务实现。
"""

from .llm_provider import QwenLLMProvider

__all__ = [
    "QwenLLMProvider",
]