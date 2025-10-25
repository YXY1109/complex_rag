"""
Database Models

This module contains all SQLAlchemy database models for the RAG system.
Based on RAGFlow architecture with multi-tenant support.
"""

# Base models
from .base import (
    Base,
    BaseModel,
    TenantBaseModel,
    TimestampMixin,
    MetadataMixin,
    AuditMixin,
    Status,
    Category,
    generate_uuid,
    current_utc_time,
    ModelType,
    TenantModelType,
)

# User and tenant models
from .user import (
    Tenant,
    User,
    UserTenant,
)

# Knowledge management models
from .knowledge import (
    KnowledgeBase,
    Document,
    DocumentChunk,
)

# Chat and conversation models
from .chat import (
    ChatSession,
    ChatMessage,
    ChatFeedback,
)

# Knowledge graph models
from .graph import (
    GraphEntity,
    GraphRelation,
    EntityMention,
)

# Export all models
__all__ = [
    # Base models and utilities
    'Base',
    'BaseModel',
    'TenantBaseModel',
    'TimestampMixin',
    'MetadataMixin',
    'AuditMixin',
    'Status',
    'Category',
    'generate_uuid',
    'current_utc_time',
    'ModelType',
    'TenantModelType',

    # User and tenant models
    'Tenant',
    'User',
    'UserTenant',

    # Knowledge management models
    'KnowledgeBase',
    'Document',
    'DocumentChunk',

    # Chat and conversation models
    'ChatSession',
    'ChatMessage',
    'ChatFeedback',

    # Knowledge graph models
    'GraphEntity',
    'GraphRelation',
    'EntityMention',
]
