"""
API路由模块

包含所有API路由的定义和组织。
"""

from . import rag_routes
from . import chat_routes
from . import knowledge_routes
from . import document_routes
from . import admin_routes
from . import health_routes

__all__ = [
    "rag_routes",
    "chat_routes",
    "knowledge_routes",
    "document_routes",
    "admin_routes",
    "health_routes"
]