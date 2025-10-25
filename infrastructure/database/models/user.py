"""
User and Tenant Models

This module contains user, tenant, and user-tenant relationship models.
Based on RAGFlow multi-tenant architecture.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid_property import hybrid_property

from .base import BaseModel, TenantBaseModel, Status, generate_uuid, current_utc_time


class Tenant(BaseModel, TimestampMixin):
    """
    Tenant model for multi-tenant architecture.

    Represents a tenant/organization in the system:
    - Basic tenant information
    - Configuration and settings
    - Subscription and limits
    - Status and metadata
    """

    __tablename__ = 'tenants'

    # Basic tenant information
    name = Column(String(100), nullable=False, index=True)
    slug = Column(String(50), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    # Contact information
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    website = Column(String(255), nullable=True)
    address = Column(Text, nullable=True)

    # Configuration
    timezone = Column(String(50), default='UTC', nullable=False)
    language = Column(String(10), default='en', nullable=False)
    locale = Column(String(10), default='en_US', nullable=False)

    # Subscription and limits
    plan = Column(String(50), default='free', nullable=False)  # free, pro, enterprise
    max_users = Column(Integer, default=10, nullable=False)
    max_documents = Column(Integer, default=1000, nullable=False)
    max_storage_mb = Column(Integer, default=1024, nullable=False)  # 1GB

    # Status and dates
    status = Column(String(20), default=Status.ACTIVE, nullable=False, index=True)
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    subscription_ends_at = Column(DateTime(timezone=True), nullable=True)

    # Settings and preferences
    settings = Column(JSON, default=dict, nullable=False)
    preferences = Column(JSON, default=dict, nullable=False)
    features = Column(JSON, default=list, nullable=False)  # Enabled features

    # Relationships
    users = relationship('User', back_populates='tenant', lazy='dynamic')
    user_tenants = relationship('UserTenant', back_populates='tenant', lazy='dynamic')
    knowledge_bases = relationship('KnowledgeBase', back_populates='tenant', lazy='dynamic')

    @validates('slug')
    def validate_slug(self, key, slug):
        """Validate slug format."""
        if not slug or not slug.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Slug must contain only alphanumeric characters, hyphens, and underscores')
        return slug.lower()

    @hybrid_property
    def is_trial(self) -> bool:
        """Check if tenant is in trial period."""
        if self.trial_ends_at is None:
            return False
        return current_utc_time() < self.trial_ends_at

    @hybrid_property
    def is_subscription_active(self) -> bool:
        """Check if subscription is active."""
        if self.subscription_ends_at is None:
            return True  # No expiration means active
        return current_utc_time() < self.subscription_ends_at

    @hybrid_property
    def days_until_trial_ends(self) -> Optional[int]:
        """Days until trial ends."""
        if self.trial_ends_at is None:
            return None
        delta = self.trial_ends_at - current_utc_time()
        return max(0, delta.days)

    @hybrid_property
    def days_until_subscription_ends(self) -> Optional[int]:
        """Days until subscription ends."""
        if self.subscription_ends_at is None:
            return None
        delta = self.subscription_ends_at - current_utc_time()
        return max(0, delta.days)

    def has_feature(self, feature: str) -> bool:
        """Check if tenant has a specific feature enabled."""
        return self.features is not None and feature in self.features

    def enable_feature(self, feature: str) -> None:
        """Enable a feature for the tenant."""
        if self.features is None:
            self.features = []
        if feature not in self.features:
            self.features.append(feature)

    def disable_feature(self, feature: str) -> None:
        """Disable a feature for the tenant."""
        if self.features and feature in self.features:
            self.features.remove(feature)

    def get_user_count(self) -> int:
        """Get current user count."""
        return self.users.count()

    def can_add_user(self) -> bool:
        """Check if tenant can add more users."""
        return self.get_user_count() < self.max_users

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        if not self.settings:
            return default
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        if not self.preferences:
            return default
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """Set a preference value."""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value


class User(BaseModel, TimestampMixin, MetadataMixin):
    """
    User model representing system users.

    Represents a user in the system:
    - Authentication and profile information
    - Roles and permissions
    - Settings and preferences
    - Activity tracking
    """

    __tablename__ = 'users'

    # Authentication fields
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)

    # Profile information
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    display_name = Column(String(100), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)

    # Status and verification
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)

    # Authentication settings
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(String(45), nullable=True)  # IPv6 compatible
    login_count = Column(Integer, default=0, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)

    # Password management
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    password_reset_token = Column(String(255), nullable=True)
    password_reset_expires_at = Column(DateTime(timezone=True), nullable=True)
    email_verification_token = Column(String(255), nullable=True)

    # Settings and preferences
    settings = Column(JSON, default=dict, nullable=False)
    preferences = Column(JSON, default=dict, nullable=False)

    # Relationships
    user_tenants = relationship('UserTenant', back_populates='user', lazy='dynamic')

    # Indexes
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_username_active', 'username', 'is_active'),
        Index('idx_user_last_login', 'last_login_at'),
    )

    @validates('email')
    def validate_email(self, key, email):
        """Basic email validation."""
        if '@' not in email:
            raise ValueError('Invalid email format')
        return email.lower()

    @hybrid_property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username

    @hybrid_property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return current_utc_time() < self.locked_until

    @hybrid_property
    def needs_password_reset(self) -> bool:
        """Check if user needs to reset password."""
        if self.password_reset_expires_at is None:
            return False
        return current_utc_time() < self.password_reset_expires_at

    def lock_account(self, hours: int = 24) -> None:
        """Lock user account for specified hours."""
        self.locked_until = current_utc_time() + timedelta(hours=hours)

    def unlock_account(self) -> None:
        """Unlock user account."""
        self.locked_until = None
        self.failed_login_attempts = 0

    def record_login(self, ip_address: Optional[str] = None) -> None:
        """Record successful login."""
        self.last_login_at = current_utc_time()
        self.last_login_ip = ip_address
        self.login_count += 1
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_failed_login(self) -> None:
        """Record failed login attempt."""
        self.failed_login_attempts += 1
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            self.lock_account(1)  # Lock for 1 hour

    def generate_password_reset_token(self) -> str:
        """Generate password reset token."""
        self.password_reset_token = generate_uuid()
        self.password_reset_expires_at = current_utc_time() + timedelta(hours=24)
        return self.password_reset_token

    def clear_password_reset_token(self) -> None:
        """Clear password reset token."""
        self.password_reset_token = None
        self.password_reset_expires_at = None

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        if not self.settings:
            return default
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        if not self.preferences:
            return default
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """Set a preference value."""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value


class UserTenant(BaseModel, AuditMixin):
    """
    User-Tenant relationship model.

    Represents the relationship between users and tenants:
    - User roles and permissions within tenant
    - Membership status and dates
    - Tenant-specific settings
    """

    __tablename__ = 'user_tenants'

    # Foreign keys
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    tenant_id = Column(String(36), ForeignKey('tenants.id'), nullable=False)

    # Role and permissions
    role = Column(String(50), default='member', nullable=False)  # owner, admin, member, viewer
    permissions = Column(JSON, default=list, nullable=False)  # Additional permissions

    # Membership status
    is_active = Column(Boolean, default=True, nullable=False)
    invited_by = Column(String(36), nullable=True)
    invited_at = Column(DateTime(timezone=True), nullable=True)
    joined_at = Column(DateTime(timezone=True), nullable=True)
    left_at = Column(DateTime(timezone=True), nullable=True)

    # Settings and preferences
    settings = Column(JSON, default=dict, nullable=False)
    preferences = Column(JSON, default=dict, nullable=False)

    # Relationships
    user = relationship('User', back_populates='user_tenants')
    tenant = relationship('Tenant', back_populates='user_tenants')

    # Indexes
    __table_args__ = (
        Index('idx_user_tenant_user', 'user_id'),
        Index('idx_user_tenant_tenant', 'tenant_id'),
        Index('idx_user_tenant_active', 'user_id', 'tenant_id', 'is_active'),
        Index('idx_user_tenant_role', 'tenant_id', 'role'),
    )

    @validates('role')
    def validate_role(self, key, role):
        """Validate role value."""
        valid_roles = ['owner', 'admin', 'member', 'viewer']
        if role not in valid_roles:
            raise ValueError(f'Role must be one of: {valid_roles}')
        return role

    @hybrid_property
    def is_member(self) -> bool:
        """Check if user is an active member."""
        return self.is_active and self.left_at is None

    @hybrid_property
    def membership_days(self) -> Optional[int]:
        """Days since user joined tenant."""
        if not self.joined_at:
            return None
        end_date = self.left_at or current_utc_time()
        delta = end_date - self.joined_at
        return delta.days

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        if not self.permissions:
            return False
        return permission in self.permissions

    def add_permission(self, permission: str) -> None:
        """Add a permission to the user."""
        if not self.permissions:
            self.permissions = []
        if permission not in self.permissions:
            self.permissions.append(permission)

    def remove_permission(self, permission: str) -> None:
        """Remove a permission from the user."""
        if self.permissions and permission in self.permissions:
            self.permissions.remove(permission)

    def join_tenant(self, user_id: str) -> None:
        """Mark user as joined tenant."""
        self.joined_at = current_utc_time()
        self.left_at = None
        self.is_active = True

    def leave_tenant(self) -> None:
        """Mark user as left tenant."""
        self.left_at = current_utc_time()
        self.is_active = False

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a tenant-specific setting value."""
        if not self.settings:
            return default
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a tenant-specific setting value."""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value