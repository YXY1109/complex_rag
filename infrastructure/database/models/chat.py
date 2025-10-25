"""
Chat and Graph Models

This module contains chat session, message, and graph models.
Based on RAGFlow conversation and knowledge graph architecture.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey, Float, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid_property import hybrid_property

from .base import TenantBaseModel, Status, generate_uuid, current_utc_time


class ChatSession(TenantBaseModel, TimestampMixin, MetadataMixin):
    """
    Chat Session model for managing conversations.

    Represents a chat session within a tenant:
    - Basic session information
    - Configuration and settings
    - Message history
    - Context and metadata
    """

    __tablename__ = 'chat_sessions'

    # Foreign keys
    knowledge_base_id = Column(String(36), ForeignKey('knowledge_bases.id'), nullable=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)

    # Basic information
    title = Column(String(200), nullable=True, index=True)
    description = Column(Text, nullable=True)
    session_type = Column(String(50), default='chat', nullable=False)  # chat, qa, analysis

    # Configuration
    model_name = Column(String(100), default='gpt-3.5-turbo', nullable=False)
    temperature = Column(Float, default=0.7, nullable=False)
    max_tokens = Column(Integer, default=2000, nullable=False)
    system_prompt = Column(Text, nullable=True)

    # RAG settings
    use_rag = Column(Boolean, default=True, nullable=False)
    similarity_threshold = Column(Float, default=0.7, nullable=False)
    max_context_chunks = Column(Integer, default=5, nullable=False)
    enable_follow_up_questions = Column(Boolean, default=True, nullable=False)

    # Search settings
    search_strategy = Column(String(50), default='hybrid', nullable=False)
    rerank_enabled = Column(Boolean, default=True, nullable=False)
    include_sources = Column(Boolean, default=True, nullable=False)

    # Session statistics
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens_used = Column(Integer, default=0, nullable=False)
    total_cost = Column(Float, default=0.0, nullable=False)
    last_message_at = Column(DateTime(timezone=True), nullable=True)

    # Status and dates
    status = Column(String(20), default=Status.ACTIVE, nullable=False, index=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    knowledge_base = relationship('KnowledgeBase', back_populates='chat_sessions')
    messages = relationship('ChatMessage', back_populates='session', lazy='dynamic',
                           cascade='all, delete-orphan')
    feedbacks = relationship('ChatFeedback', back_populates='session', lazy='dynamic')

    # Indexes
    __table_args__ = (
        Index('idx_session_user', 'user_id'),
        Index('idx_session_kb', 'knowledge_base_id'),
        Index('idx_session_status', 'status'),
        Index('idx_session_last_message', 'last_message_at'),
        Index('idx_session_expires', 'expires_at'),
    )

    @validates('temperature')
    def validate_temperature(self, key, temp):
        """Validate temperature value."""
        if not 0.0 <= temp <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return temp

    @validates('max_tokens')
    def validate_max_tokens(self, key, tokens):
        """Validate max tokens."""
        if tokens < 1 or tokens > 32000:
            raise ValueError('Max tokens must be between 1 and 32000')
        return tokens

    @validates('similarity_threshold')
    def validate_similarity_threshold(self, key, threshold):
        """Validate similarity threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return threshold

    @hybrid_property
    def is_active(self) -> bool:
        """Check if session is active."""
        return (self.status == Status.ACTIVE and
                not self.is_expired and
                not self.is_deleted)

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at is None:
            return False
        return current_utc_time() > self.expires_at

    @hybrid_property
    def is_archived(self) -> bool:
        """Check if session is archived."""
        return self.archived_at is not None

    @hybrid_property
    def average_tokens_per_message(self) -> Optional[float]:
        """Calculate average tokens per message."""
        if self.message_count == 0:
            return None
        return self.total_tokens_used / self.message_count

    @hybrid_property
    def session_duration(self) -> Optional[timedelta]:
        """Calculate session duration."""
        if not self.last_message_at:
            return None
        return self.last_message_at - self.created_at

    def add_message(self, message_count: int = 1, tokens_used: int = 0, cost: float = 0.0) -> None:
        """Update session statistics with new message."""
        self.message_count += message_count
        self.total_tokens_used += tokens_used
        self.total_cost += cost
        self.last_message_at = current_utc_time()

    def archive(self) -> None:
        """Archive the session."""
        self.archived_at = current_utc_time()
        self.status = Status.ARCHIVED

    def restore(self) -> None:
        """Restore archived session."""
        self.archived_at = None
        self.status = Status.ACTIVE

    def extend_session(self, days: int = 30) -> None:
        """Extend session expiration."""
        if self.expires_at is None:
            self.expires_at = current_utc_time() + timedelta(days=days)
        else:
            self.expires_at = self.expires_at + timedelta(days=days)

    def get_context_summary(self) -> Dict[str, Any]:
        """Get context summary for the session."""
        return {
            'title': self.title,
            'message_count': self.message_count,
            'total_tokens': self.total_tokens_used,
            'total_cost': self.total_cost,
            'last_activity': self.last_message_at.isoformat() if self.last_message_at else None,
            'duration_days': self.session_duration.days if self.session_duration else None,
            'knowledge_base_id': self.knowledge_base_id,
            'model_name': self.model_name,
            'use_rag': self.use_rag
        }


class ChatMessage(TenantBaseModel, TimestampMixin):
    """
    Chat Message model for storing conversation messages.

    Represents a message within a chat session:
    - Message content and metadata
    - Role and type information
    - Context and references
    - Processing information
    """

    __tablename__ = 'chat_messages'

    # Foreign keys
    session_id = Column(String(36), ForeignKey('chat_sessions.id'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)

    # Message content
    content = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    message_type = Column(String(30), default='text', nullable=False)  # text, image, file

    # Processing information
    model_name = Column(String(100), nullable=True)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    tokens_used = Column(Integer, default=0, nullable=False)
    cost = Column(Float, default=0.0, nullable=False)
    processing_time_ms = Column(Integer, nullable=True)

    # RAG context
    context_chunks = Column(JSON, default=list, nullable=False)  # Referenced document chunks
    context_sources = Column(JSON, default=list, nullable=False)  # Source information
    search_query = Column(Text, nullable=True)  # Original search query
    search_results = Column(JSON, default=list, nullable=False)  # Search results

    # Message metadata
    parent_message_id = Column(String(36), nullable=True)  # For message threading
    is_edited = Column(Boolean, default=False, nullable=False)
    edited_at = Column(DateTime(timezone=True), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Feedback and ratings
    user_rating = Column(Integer, nullable=True)  # 1-5 rating
    user_feedback = Column(Text, nullable=True)
    corrected_answer = Column(Text, nullable=True)

    # Relationships
    session = relationship('ChatSession', back_populates='messages')
    feedbacks = relationship('ChatFeedback', back_populates='message', lazy='dynamic')

    # Indexes
    __table_args__ = (
        Index('idx_message_session', 'session_id'),
        Index('idx_message_user', 'user_id'),
        Index('idx_message_role', 'session_id', 'role'),
        Index('idx_message_created', 'created_at'),
        Index('idx_message_parent', 'parent_message_id'),
    )

    @validates('role')
    def validate_role(self, key, role):
        """Validate message role."""
        valid_roles = ['user', 'assistant', 'system']
        if role not in valid_roles:
            raise ValueError(f'Role must be one of: {valid_roles}')
        return role

    @validates('message_type')
    def validate_message_type(self, key, msg_type):
        """Validate message type."""
        valid_types = ['text', 'image', 'file', 'audio', 'video']
        if msg_type not in valid_types:
            raise ValueError(f'Message type must be one of: {valid_types}')
        return msg_type

    @validates('user_rating')
    def validate_user_rating(self, key, rating):
        """Validate user rating."""
        if rating is not None and (rating < 1 or rating > 5):
            raise ValueError('User rating must be between 1 and 5')
        return rating

    @hybrid_property
    def is_from_user(self) -> bool:
        """Check if message is from user."""
        return self.role == 'user'

    @hybrid_property
    def is_from_assistant(self) -> bool:
        """Check if message is from assistant."""
        return self.role == 'assistant'

    @hybrid_property
    def has_context(self) -> bool:
        """Check if message has RAG context."""
        return len(self.context_chunks) > 0

    @hybrid_property
    def processing_time_seconds(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.processing_time_ms is None:
            return None
        return self.processing_time_ms / 1000.0

    def edit_content(self, new_content: str) -> None:
        """Edit message content."""
        self.content = new_content
        self.is_edited = True
        self.edited_at = current_utc_time()

    def soft_delete(self) -> None:
        """Soft delete message."""
        self.is_deleted = True
        self.deleted_at = current_utc_time()

    def restore(self) -> None:
        """Restore soft deleted message."""
        self.is_deleted = False
        self.deleted_at = None

    def add_context_chunk(self, chunk_id: str, chunk_info: Dict[str, Any]) -> None:
        """Add a context chunk to the message."""
        if not self.context_chunks:
            self.context_chunks = []
        chunk_entry = {'chunk_id': chunk_id, **chunk_info}
        if chunk_entry not in self.context_chunks:
            self.context_chunks.append(chunk_entry)

    def set_rating(self, rating: int, feedback: Optional[str] = None) -> None:
        """Set user rating and feedback."""
        self.user_rating = rating
        if feedback:
            self.user_feedback = feedback


class ChatFeedback(TenantBaseModel, TimestampMixin):
    """
    Chat Feedback model for storing user feedback on messages.

    Represents feedback on chat messages:
    - Rating and sentiment
    - Detailed feedback comments
    - Correction information
    - Analytics data
    """

    __tablename__ = 'chat_feedbacks'

    # Foreign keys
    session_id = Column(String(36), ForeignKey('chat_sessions.id'), nullable=False)
    message_id = Column(String(36), ForeignKey('chat_messages.id'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)

    # Feedback content
    rating = Column(Integer, nullable=False)  # 1-5 rating
    sentiment = Column(String(20), nullable=True)  # positive, negative, neutral
    feedback_type = Column(String(30), default='general', nullable=False)  # general, accuracy, helpfulness, clarity

    # Detailed feedback
    comment = Column(Text, nullable=True)
    aspects = Column(JSON, default=dict, nullable=False)  # Aspect-based feedback
    suggestions = Column(JSON, default=list, nullable=False)  # Improvement suggestions

    # Correction information
    is_correction = Column(Boolean, default=False, nullable=False)
    corrected_content = Column(Text, nullable=True)
    correction_reason = Column(Text, nullable=True)

    # Analytics
    response_time_rating = Column(Integer, nullable=True)  # 1-5 rating for response time
    accuracy_rating = Column(Integer, nullable=True)  # 1-5 rating for accuracy
    helpfulness_rating = Column(Integer, nullable=True)  # 1-5 rating for helpfulness

    # Metadata
    source = Column(String(50), nullable=True)  # Where feedback was collected
    metadata = Column(JSON, default=dict, nullable=False)

    # Relationships
    session = relationship('ChatSession', back_populates='feedbacks')
    message = relationship('ChatMessage', back_populates='feedbacks')

    # Indexes
    __table_args__ = (
        Index('idx_feedback_session', 'session_id'),
        Index('idx_feedback_message', 'message_id'),
        Index('idx_feedback_user', 'user_id'),
        Index('idx_feedback_rating', 'rating'),
        Index('idx_feedback_type', 'feedback_type'),
        Index('idx_feedback_created', 'created_at'),
    )

    @validates('rating')
    def validate_rating(self, key, rating):
        """Validate rating value."""
        if not 1 <= rating <= 5:
            raise ValueError('Rating must be between 1 and 5')
        return rating

    @validates('sentiment')
    def validate_sentiment(self, key, sentiment):
        """Validate sentiment value."""
        valid_sentiments = ['positive', 'negative', 'neutral']
        if sentiment not in valid_sentiments:
            raise ValueError(f'Sentiment must be one of: {valid_sentiments}')
        return sentiment

    @validates('feedback_type')
    def validate_feedback_type(self, key, feedback_type):
        """Validate feedback type."""
        valid_types = ['general', 'accuracy', 'helpfulness', 'clarity', 'completeness', 'relevance']
        if feedback_type not in valid_types:
            raise ValueError(f'Feedback type must be one of: {valid_types}')
        return feedback_type

    @hybrid_property
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        return self.rating >= 4

    @hybrid_property
    def is_negative(self) -> bool:
        """Check if feedback is negative."""
        return self.rating <= 2

    def set_aspect_rating(self, aspect: str, rating: int) -> None:
        """Set rating for a specific aspect."""
        if not self.aspects:
            self.aspects = {}
        self.aspects[aspect] = rating

    def get_aspect_rating(self, aspect: str, default: int = 3) -> int:
        """Get rating for a specific aspect."""
        if not self.aspects:
            return default
        return self.aspects.get(aspect, default)

    def add_suggestion(self, suggestion: str) -> None:
        """Add an improvement suggestion."""
        if not self.suggestions:
            self.suggestions = []
        if suggestion not in self.suggestions:
            self.suggestions.append(suggestion)

    def mark_as_correction(self, corrected_content: str, reason: Optional[str] = None) -> None:
        """Mark feedback as a correction."""
        self.is_correction = True
        self.corrected_content = corrected_content
        self.correction_reason = reason

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary."""
        return {
            'rating': self.rating,
            'sentiment': self.sentiment,
            'feedback_type': self.feedback_type,
            'is_positive': self.is_positive,
            'is_negative': self.is_negative,
            'is_correction': self.is_correction,
            'aspects': self.aspects or {},
            'suggestions_count': len(self.suggestions) if self.suggestions else 0,
            'has_comment': bool(self.comment),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }