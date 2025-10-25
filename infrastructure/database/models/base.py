"""
Base SQLAlchemy Models

This module contains the base model classes for all SQLAlchemy models.
Based on RAGFlow architecture with multi-tenant support.
"""

from datetime import datetime
from typing import Any, Dict, Optional, TypeVar, Type
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declared_attr
from sqlalchemy.sql import func
import uuid


Base = declarative_base()


class BaseModel(Base):
    """
    Base model class for all SQLAlchemy models.

    Provides common fields and functionality for all models:
    - id: Primary key with UUID
    - created_at: Creation timestamp
    - updated_at: Last update timestamp
    - metadata: JSON metadata storage
    """

    __abstract__ = True

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    metadata = Column(JSON, default=dict, nullable=False)

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        name = cls.__name__
        # Convert CamelCase to snake_case
        import re
        table_name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        # Add 's' for plural if not already plural
        if not table_name.endswith('s'):
            table_name += 's'
        return table_name

    def to_dict(self, exclude: Optional[set] = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.

        Args:
            exclude: Set of field names to exclude

        Returns:
            Dict[str, Any]: Model data as dictionary
        """
        exclude = exclude or set()
        result = {}

        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[column.name] = value

        return result

    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[set] = None) -> None:
        """
        Update model instance from dictionary.

        Args:
            data: Dictionary with field values
            exclude: Set of field names to exclude from update
        """
        exclude = exclude or {'id', 'created_at', 'updated_at'}

        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def create(cls: Type['BaseModel'], **kwargs) -> 'BaseModel':
        """
        Create new model instance.

        Args:
            **kwargs: Field values

        Returns:
            BaseModel: New instance
        """
        return cls(**kwargs)

    def __repr__(self) -> str:
        """String representation of model."""
        return f"<{self.__class__.__name__}(id='{self.id}')>"


class TenantBaseModel(BaseModel):
    """
    Base model for tenant-aware entities.

    Extends BaseModel with tenant support:
    - tenant_id: Foreign key to tenant
    - Additional tenant-scoped functionality
    """

    __abstract__ = True

    tenant_id = Column(String(36), nullable=False, index=True)

    @declared_attr
    def __table_args__(cls):
        """Generate table arguments for tenant models."""
        # Add index on tenant_id for all tenant models
        return {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}

    @classmethod
    def create_for_tenant(cls: Type['TenantBaseModel'], tenant_id: str, **kwargs) -> 'TenantBaseModel':
        """
        Create new model instance for specific tenant.

        Args:
            tenant_id: Tenant ID
            **kwargs: Field values

        Returns:
            TenantBaseModel: New instance
        """
        return cls(tenant_id=tenant_id, **kwargs)

    def __repr__(self) -> str:
        """String representation of tenant model."""
        return f"<{self.__class__.__name__}(id='{self.id}', tenant_id='{self.tenant_id}')>"


class TimestampMixin:
    """
    Mixin class for timestamp fields.

    Provides additional timestamp functionality:
    - deleted_at: Soft delete timestamp
    - expires_at: Expiration timestamp
    """

    deleted_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None

    @property
    def is_expired(self) -> bool:
        """Check if record is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def soft_delete(self) -> None:
        """Mark record as soft deleted."""
        self.deleted_at = datetime.utcnow()

    def restore(self) -> None:
        """Restore soft deleted record."""
        self.deleted_at = None


class MetadataMixin:
    """
    Mixin class for enhanced metadata functionality.

    Provides additional metadata fields and methods:
    - tags: List of tags
    - category: Category classification
    - status: Status field
    """

    tags = Column(JSON, default=list, nullable=False)
    category = Column(String(100), nullable=True)
    status = Column(String(50), default='active', nullable=False)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the record."""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the record."""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if record has a specific tag."""
        return self.tags is not None and tag in self.tags

    def get_tags(self) -> list:
        """Get all tags."""
        return self.tags or []


class AuditMixin:
    """
    Mixin class for audit functionality.

    Provides audit fields:
    - created_by: User who created the record
    - updated_by: User who last updated the record
    - version: Version number for optimistic locking
    """

    created_by = Column(String(36), nullable=True)
    updated_by = Column(String(36), nullable=True)
    version = Column(Integer, default=1, nullable=False)

    def increment_version(self) -> None:
        """Increment version number."""
        self.version += 1

    def set_creator(self, user_id: str) -> None:
        """Set creator user ID."""
        self.created_by = user_id

    def set_updater(self, user_id: str) -> None:
        """Set updater user ID."""
        self.updated_by = user_id


# Type variables for better type hints
ModelType = TypeVar('ModelType', bound=BaseModel)
TenantModelType = TypeVar('TenantModelType', bound=TenantBaseModel)


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def current_utc_time() -> datetime:
    """Get current UTC time."""
    return datetime.utcnow()


# Common status constants
class Status:
    """Common status constants."""
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    DELETED = 'deleted'
    ARCHIVED = 'archived'


# Common category constants
class Category:
    """Common category constants."""
    DEFAULT = 'default'
    SYSTEM = 'system'
    USER = 'user'
    DOCUMENT = 'document'
    KNOWLEDGE = 'knowledge'
    CHAT = 'chat'
    GRAPH = 'graph'