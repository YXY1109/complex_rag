"""
Knowledge Base and Document Models

This module contains knowledge base and document models.
Based on RAGFlow knowledge management architecture.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey, Float, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid_property import hybrid_property

from .base import TenantBaseModel, Status, generate_uuid, current_utc_time


class KnowledgeBase(TenantBaseModel, TimestampMixin, MetadataMixin):
    """
    Knowledge Base model for organizing documents and knowledge.

    Represents a knowledge base within a tenant:
    - Basic knowledge base information
    - Configuration and settings
    - Document management
    - Access control and permissions
    """

    __tablename__ = 'knowledge_bases'

    # Basic information
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    slug = Column(String(100), nullable=False, index=True)

    # Configuration
    embedding_model = Column(String(100), default='text-embedding-ada-002', nullable=False)
    chunk_size = Column(Integer, default=512, nullable=False)
    chunk_overlap = Column(Integer, default=50, nullable=False)
    rerank_model = Column(String(100), nullable=True)

    # Processing settings
    auto_process = Column(Boolean, default=True, nullable=False)
    extract_images = Column(Boolean, default=False, nullable=False)
    extract_tables = Column(Boolean, default=True, nullable=False)
    ocr_enabled = Column(Boolean, default=False, nullable=False)
    language_detection = Column(Boolean, default=True, nullable=False)

    # Search settings
    search_mode = Column(String(50), default='hybrid', nullable=False)  # vector, keyword, hybrid
    similarity_threshold = Column(Float, default=0.7, nullable=False)
    max_results = Column(Integer, default=10, nullable=False)
    enable_rerank = Column(Boolean, default=True, nullable=False)

    # Vector database settings
    vector_db_config = Column(JSON, default=dict, nullable=False)
    index_name = Column(String(100), nullable=True)

    # Statistics
    document_count = Column(Integer, default=0, nullable=False)
    total_chunks = Column(Integer, default=0, nullable=False)
    total_size_mb = Column(Float, default=0.0, nullable=False)
    last_processed_at = Column(DateTime(timezone=True), nullable=True)

    # Status and dates
    status = Column(String(20), default=Status.ACTIVE, nullable=False, index=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    documents = relationship('Document', back_populates='knowledge_base', lazy='dynamic')
    chunks = relationship('DocumentChunk', back_populates='knowledge_base', lazy='dynamic')
    chat_sessions = relationship('ChatSession', back_populates='knowledge_base', lazy='dynamic')

    # Indexes
    __table_args__ = (
        Index('idx_kb_tenant_name', 'tenant_id', 'name'),
        Index('idx_kb_tenant_slug', 'tenant_id', 'slug'),
        Index('idx_kb_status', 'status'),
        Index('idx_kb_last_processed', 'last_processed_at'),
    )

    @validates('slug')
    def validate_slug(self, key, slug):
        """Validate slug format."""
        if not slug or not slug.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Slug must contain only alphanumeric characters, hyphens, and underscores')
        return slug.lower()

    @validates('search_mode')
    def validate_search_mode(self, key, mode):
        """Validate search mode."""
        valid_modes = ['vector', 'keyword', 'hybrid']
        if mode not in valid_modes:
            raise ValueError(f'Search mode must be one of: {valid_modes}')
        return mode

    @hybrid_property
    def is_published(self) -> bool:
        """Check if knowledge base is published."""
        return self.published_at is not None and self.published_at <= current_utc_time()

    @hybrid_property
    def is_archived(self) -> bool:
        """Check if knowledge base is archived."""
        return self.archived_at is not None

    @hybrid_property
    def is_processing(self) -> bool:
        """Check if knowledge base is currently processing."""
        return self.status == Status.PROCESSING

    @hybrid_property
    def average_document_size(self) -> Optional[float]:
        """Calculate average document size in MB."""
        if self.document_count == 0:
            return None
        return self.total_size_mb / self.document_count

    def publish(self) -> None:
        """Publish knowledge base."""
        self.published_at = current_utc_time()
        self.status = Status.ACTIVE

    def archive(self) -> None:
        """Archive knowledge base."""
        self.archived_at = current_utc_time()
        self.status = Status.ARCHIVED

    def restore(self) -> None:
        """Restore archived knowledge base."""
        self.archived_at = None
        self.status = Status.ACTIVE

    def update_statistics(self) -> None:
        """Update knowledge base statistics."""
        doc_count = self.documents.filter_by(is_deleted=False).count()
        chunk_count = self.chunks.filter_by(is_deleted=False).count()

        total_size = 0.0
        for doc in self.documents.filter_by(is_deleted=False):
            if doc.size_mb:
                total_size += doc.size_mb

        self.document_count = doc_count
        self.total_chunks = chunk_count
        self.total_size_mb = total_size
        self.last_processed_at = current_utc_time()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if not self.vector_db_config:
            return default
        return self.vector_db_config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        if not self.vector_db_config:
            self.vector_db_config = {}
        self.vector_db_config[key] = value


class Document(TenantBaseModel, TimestampMixin, MetadataMixin):
    """
    Document model for managing uploaded documents.

    Represents a document within a knowledge base:
    - Basic document information
    - Processing status and settings
    - File information and metadata
    - Relationship with knowledge base
    """

    __tablename__ = 'documents'

    # Foreign keys
    knowledge_base_id = Column(String(36), ForeignKey('knowledge_bases.id'), nullable=False)

    # Basic information
    title = Column(String(500), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    size_mb = Column(Float, nullable=False)
    mime_type = Column(String(100), nullable=True)
    file_hash = Column(String(64), nullable=True, index=True)  # SHA-256 hash

    # Content information
    content_type = Column(String(50), nullable=True)  # text, pdf, docx, etc.
    language = Column(String(10), nullable=True)
    encoding = Column(String(20), default='utf-8', nullable=False)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)

    # Processing status
    status = Column(String(20), default=Status.PENDING, nullable=False, index=True)
    processing_progress = Column(Float, default=0.0, nullable=False)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_error = Column(Text, nullable=True)

    # Chunking information
    chunk_count = Column(Integer, default=0, nullable=False)
    chunk_strategy = Column(String(50), default='fixed_size', nullable=False)
    last_chunked_at = Column(DateTime(timezone=True), nullable=True)

    # Vector information
    embedding_model = Column(String(100), nullable=True)
    vector_count = Column(Integer, default=0, nullable=False)
    last_vectorized_at = Column(DateTime(timezone=True), nullable=True)

    # OCR and extraction
    ocr_required = Column(Boolean, default=False, nullable=False)
    ocr_completed = Column(Boolean, default=False, nullable=False)
    images_extracted = Column(Boolean, default=False, nullable=False)
    tables_extracted = Column(Boolean, default=False, nullable=False)

    # Access control
    is_public = Column(Boolean, default=False, nullable=False)
    access_level = Column(String(20), default='private', nullable=False)  # private, team, public

    # Relationships
    knowledge_base = relationship('KnowledgeBase', back_populates='documents')
    chunks = relationship('DocumentChunk', back_populates='document', lazy='dynamic',
                          cascade='all, delete-orphan')

    # Indexes
    __table_args__ = (
        Index('idx_doc_kb_tenant', 'knowledge_base_id', 'tenant_id'),
        Index('idx_doc_status', 'status'),
        Index('idx_doc_file_hash', 'file_hash'),
        Index('idx_doc_content_type', 'content_type'),
        Index('idx_doc_processing', 'processing_started_at', 'status'),
    )

    @validates('file_size_bytes')
    def validate_file_size(self, key, size):
        """Validate file size."""
        if size < 0:
            raise ValueError('File size cannot be negative')
        return size

    @validates('size_mb')
    def validate_size_mb(self, key, size):
        """Validate size in MB."""
        if size < 0:
            raise ValueError('Size in MB cannot be negative')
        return size

    @hybrid_property
    def is_processed(self) -> bool:
        """Check if document processing is complete."""
        return self.status == Status.COMPLETED

    @hybrid_property
    def is_processing(self) -> bool:
        """Check if document is currently processing."""
        return self.status == Status.PROCESSING

    @hybrid_property
    def has_processing_error(self) -> bool:
        """Check if document has processing error."""
        return self.status == Status.FAILED and self.processing_error is not None

    @hybrid_property
    def processing_duration(self) -> Optional[timedelta]:
        """Get processing duration."""
        if not self.processing_started_at or not self.processing_completed_at:
            return None
        return self.processing_completed_at - self.processing_started_at

    @hybrid_property
    def is_vectorized(self) -> bool:
        """Check if document has been vectorized."""
        return self.vector_count > 0 and self.last_vectorized_at is not None

    def start_processing(self) -> None:
        """Mark document as processing started."""
        self.status = Status.PROCESSING
        self.processing_started_at = current_utc_time()
        self.processing_progress = 0.0
        self.processing_error = None

    def update_progress(self, progress: float) -> None:
        """Update processing progress."""
        self.processing_progress = max(0.0, min(100.0, progress))

    def complete_processing(self) -> None:
        """Mark document processing as complete."""
        self.status = Status.COMPLETED
        self.processing_completed_at = current_utc_time()
        self.processing_progress = 100.0

    def fail_processing(self, error_message: str) -> None:
        """Mark document processing as failed."""
        self.status = Status.FAILED
        self.processing_completed_at = current_utc_time()
        self.processing_error = error_message

    def reset_processing(self) -> None:
        """Reset document processing status."""
        self.status = Status.PENDING
        self.processing_started_at = None
        self.processing_completed_at = None
        self.processing_progress = 0.0
        self.processing_error = None

    def update_chunk_info(self, chunk_count: int, strategy: str = 'fixed_size') -> None:
        """Update chunking information."""
        self.chunk_count = chunk_count
        self.chunk_strategy = strategy
        self.last_chunked_at = current_utc_time()

    def update_vector_info(self, vector_count: int, model: str) -> None:
        """Update vector information."""
        self.vector_count = vector_count
        self.embedding_model = model
        self.last_vectorized_at = current_utc_time()


class DocumentChunk(TenantBaseModel, TimestampMixin):
    """
    Document Chunk model for managing document chunks.

    Represents a chunk of a document:
    - Chunk content and metadata
    - Vector embeddings
    - Position and relationship information
    - Search and retrieval settings
    """

    __tablename__ = 'document_chunks'

    # Foreign keys
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    knowledge_base_id = Column(String(36), ForeignKey('knowledge_bases.id'), nullable=False)

    # Content
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default='text', nullable=False)
    language = Column(String(10), nullable=True)

    # Position information
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    start_line = Column(Integer, nullable=True)
    end_line = Column(Integer, nullable=True)

    # Chunk metadata
    token_count = Column(Integer, nullable=True)
    sentence_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)

    # Vector information
    embedding_model = Column(String(100), nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    vector_id = Column(String(100), nullable=True)  # ID in vector database
    embedding_updated_at = Column(DateTime(timezone=True), nullable=True)

    # Search and retrieval
    search_keywords = Column(JSON, default=list, nullable=False)
    entities = Column(JSON, default=list, nullable=False)  # Extracted entities
    summary = Column(Text, nullable=True)  # Chunk summary

    # Quality metrics
    quality_score = Column(Float, nullable=True)  # Content quality score
    relevance_score = Column(Float, nullable=True)  # Relevance to document
    importance_score = Column(Float, nullable=True)  # Importance score

    # Relationships
    document = relationship('Document', back_populates='chunks')
    knowledge_base = relationship('KnowledgeBase', back_populates='chunks')

    # Indexes
    __table_args__ = (
        Index('idx_chunk_document', 'document_id'),
        Index('idx_chunk_kb', 'knowledge_base_id'),
        Index('idx_chunk_index', 'document_id', 'chunk_index'),
        Index('idx_chunk_page', 'document_id', 'page_number'),
        Index('idx_chunk_vector', 'vector_id'),
    )

    @validates('chunk_index')
    def validate_chunk_index(self, key, index):
        """Validate chunk index."""
        if index < 0:
            raise ValueError('Chunk index cannot be negative')
        return index

    @hybrid_property
    def has_embedding(self) -> bool:
        """Check if chunk has embedding."""
        return self.vector_id is not None and self.embedding_updated_at is not None

    @hybrid_property
    def char_length(self) -> Optional[int]:
        """Get character length of chunk."""
        if self.start_char is not None and self.end_char is not None:
            return self.end_char - self.start_char
        return len(self.content) if self.content else None

    @hybrid_property
    def is_on_page(self) -> bool:
        """Check if chunk has page information."""
        return self.page_number is not None

    def add_search_keyword(self, keyword: str) -> None:
        """Add a search keyword."""
        if not self.search_keywords:
            self.search_keywords = []
        if keyword not in self.search_keywords:
            self.search_keywords.append(keyword)

    def remove_search_keyword(self, keyword: str) -> None:
        """Remove a search keyword."""
        if self.search_keywords and keyword in self.search_keywords:
            self.search_keywords.remove(keyword)

    def add_entity(self, entity_type: str, entity_value: str) -> None:
        """Add an extracted entity."""
        if not self.entities:
            self.entities = []
        entity = {'type': entity_type, 'value': entity_value}
        if entity not in self.entities:
            self.entities.append(entity)

    def update_embedding(self, vector_id: str, model: str, dimension: int) -> None:
        """Update embedding information."""
        self.vector_id = vector_id
        self.embedding_model = model
        self.embedding_dimension = dimension
        self.embedding_updated_at = current_utc_time()

    def get_position_summary(self) -> str:
        """Get a summary of chunk position."""
        parts = []
        if self.chunk_index is not None:
            parts.append(f"Chunk {self.chunk_index}")
        if self.page_number is not None:
            parts.append(f"Page {self.page_number}")
        if self.start_line is not None and self.end_line is not None:
            parts.append(f"Lines {self.start_line}-{self.end_line}")
        return ", ".join(parts) if parts else "Unknown position"