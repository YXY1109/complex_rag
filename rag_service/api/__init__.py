"""
RAG服务API层

提供RESTful API接口，支持所有RAG功能的HTTP访问。
"""

from .app import create_app
from .middleware import setup_middleware
from .dependencies import get_rag_service, get_current_user

__all__ = [
    "create_app",
    "setup_middleware",
    "get_rag_service",
    "get_current_user"
]